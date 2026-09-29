"""Generic, recoverable distributed training for finite Stage A/B experiments.

This module deliberately knows nothing about a particular chemistry dataset.
Dataset adapters supply a deterministic batch provider, while this runner owns
the public epoch budget, internal optimizer-update mapping, DDP rank lanes,
atomic recovery state, and candidate checkpoint publication.  It records paths,
counts, and explicit configuration only; it never computes content hashes.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from retro_guided_generation.guidance.io import ensure_output_directory, write_json
from retro_guided_generation.training.epochs import (
    EpochBudget,
    EpochTrainingPlan,
    EpochValue,
    epoch_value_json,
    epoch_value_slug,
    parse_epoch_value,
)


FORMAL_TRAINING_SCHEMA = "retro_guided_generation.formal_training.v2"
LEGACY_FORMAL_TRAINING_SCHEMA = "retro_guided_generation.formal_training.v1"
FORMAL_CHECKPOINT_REQUEST_FILENAME = "checkpoint_request"
FORMAL_STOP_AFTER_EPOCHS_ENV = "FORMAL_STOP_AFTER_EPOCHS"
FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV = "FORMAL_STOP_AFTER_COMPLETED_STEPS"
FORMAL_HEARTBEAT_HISTORY_ENV = "FORMAL_HEARTBEAT_HISTORY"
FORMAL_PROFILE_STEPS_ENV = "FORMAL_PROFILE_STEPS"
FORMAL_CPU_PREFETCH_DEPTH_ENV = "FORMAL_CPU_PREFETCH_DEPTH"
FORMAL_CPU_PREFETCH_ASSEMBLERS_ENV = "FORMAL_CPU_PREFETCH_ASSEMBLERS"
FORMAL_ALLOW_BATCH_RESUME_AMENDMENT_ENV = "FORMAL_ALLOW_BATCH_RESUME_AMENDMENT"
FORMAL_ALLOW_EXPOSURE_RESUME_AMENDMENT_ENV = (
    "FORMAL_ALLOW_EXPOSURE_RESUME_AMENDMENT"
)
FORMAL_ALLOW_EXPOSURE_SCHEDULE_RESUME_AMENDMENT_ENV = (
    "FORMAL_ALLOW_EXPOSURE_SCHEDULE_RESUME_AMENDMENT"
)


@runtime_checkable
class FormalBatchProvider(Protocol):
    """Minimal data adapter contract required by :func:`run_formal_training`."""

    def next_batch(
        self,
        *,
        epoch: int,
        draw_index: int,
        draw_stride: int,
        device: torch.device | str | None,
        evaluation: bool,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class FormalTrainingConfig:
    """Dataset-neutral contract for one finite, restartable training stage.

    New runs declare exact epoch budgets.  The optional ``*_steps`` fields are
    retained only so existing v1 run configurations and recovery artifacts can
    still be resumed without changing their historical sampling behavior.
    """

    stage: str
    output_dir: Path
    expected_world_size: int
    per_rank_batch_groups: int
    global_physical_batch_groups: int
    max_epochs: EpochValue | None = None
    candidate_epochs: tuple[EpochValue, ...] = ()
    max_steps: int | None = None
    candidate_steps: tuple[int, ...] = ()
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.0
    amsgrad: bool = False
    gradient_clip_norm: float | None = 1.0
    seed: int = 42
    first_epoch: int = 0
    recovery_interval_epochs: EpochValue | None = None
    heartbeat_interval_epochs: EpochValue | None = None
    # Legacy v1 controls.  New configs must not set these fields.
    epoch: int | None = None
    recovery_interval_steps: int | None = None
    heartbeat_interval_steps: int | None = None
    precision: str = "none"
    resume: bool = True
    deterministic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise ValueError("stage must be a non-empty string")
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a pathlib.Path")
        for name, value in (
            ("expected_world_size", self.expected_world_size),
            ("per_rank_batch_groups", self.per_rank_batch_groups),
            ("global_physical_batch_groups", self.global_physical_batch_groups),
            ("gradient_accumulation_steps", self.gradient_accumulation_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.global_physical_batch_groups != (
            self.expected_world_size * self.per_rank_batch_groups
        ):
            raise ValueError(
                "global_physical_batch_groups must equal "
                "expected_world_size * per_rank_batch_groups"
            )
        epoch_budget_requested = self.max_epochs is not None or bool(self.candidate_epochs)
        step_budget_requested = self.max_steps is not None or bool(self.candidate_steps)
        if epoch_budget_requested == step_budget_requested:
            raise ValueError(
                "configure exactly one training budget: max_epochs/candidate_epochs "
                "for new runs, or legacy max_steps/candidate_steps"
            )
        if epoch_budget_requested:
            if self.max_epochs is None:
                raise ValueError("max_epochs is required with candidate_epochs")
            if self.epoch is not None:
                raise ValueError("new epoch configs use first_epoch, not legacy epoch")
            if self.recovery_interval_steps is not None or self.heartbeat_interval_steps is not None:
                raise ValueError(
                    "new epoch configs use recovery_interval_epochs and "
                    "heartbeat_interval_epochs"
                )
            budget = EpochBudget(
                max_epochs=self.max_epochs,
                candidate_epochs=self.candidate_epochs,
                first_epoch=self.first_epoch,
            )
            recovery = parse_epoch_value(
                Fraction(1, 40)
                if self.recovery_interval_epochs is None
                else self.recovery_interval_epochs,
                "recovery_interval_epochs",
            )
            heartbeat = parse_epoch_value(
                Fraction(1, 400)
                if self.heartbeat_interval_epochs is None
                else self.heartbeat_interval_epochs,
                "heartbeat_interval_epochs",
            )
            object.__setattr__(self, "max_epochs", budget.max_epochs)
            object.__setattr__(self, "candidate_epochs", budget.candidate_epochs)
            object.__setattr__(self, "recovery_interval_epochs", recovery)
            object.__setattr__(self, "heartbeat_interval_epochs", heartbeat)
        else:
            if self.max_steps is None:
                raise ValueError("legacy max_steps is required with candidate_steps")
            for name, value in (
                ("max_steps", self.max_steps),
                (
                    "recovery_interval_steps",
                    1_000 if self.recovery_interval_steps is None else self.recovery_interval_steps,
                ),
                (
                    "heartbeat_interval_steps",
                    100 if self.heartbeat_interval_steps is None else self.heartbeat_interval_steps,
                ),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            if not isinstance(self.candidate_steps, tuple) or not self.candidate_steps:
                raise ValueError("candidate_steps must be a non-empty tuple")
            if any(
                isinstance(step, bool) or not isinstance(step, int)
                for step in self.candidate_steps
            ):
                raise ValueError("candidate_steps must contain integers")
            if tuple(sorted(set(self.candidate_steps))) != self.candidate_steps:
                raise ValueError("candidate_steps must be sorted and unique")
            if self.candidate_steps[-1] != self.max_steps:
                raise ValueError("candidate_steps must include the terminal max_steps")
            if self.candidate_steps[0] < 1 or self.candidate_steps[-1] > self.max_steps:
                raise ValueError("candidate_steps must be inside [1, max_steps]")
            if self.recovery_interval_epochs is not None or self.heartbeat_interval_epochs is not None:
                raise ValueError("legacy step configs cannot set epoch-based intervals")
            sampler_epoch = self.first_epoch if self.epoch is None else self.epoch
            if (
                isinstance(sampler_epoch, bool)
                or not isinstance(sampler_epoch, int)
                or sampler_epoch < 0
            ):
                raise ValueError("legacy epoch must be a non-negative integer")
            object.__setattr__(self, "epoch", sampler_epoch)
            object.__setattr__(
                self,
                "recovery_interval_steps",
                1_000 if self.recovery_interval_steps is None else self.recovery_interval_steps,
            )
            object.__setattr__(
                self,
                "heartbeat_interval_steps",
                100 if self.heartbeat_interval_steps is None else self.heartbeat_interval_steps,
            )
        for name, value, positive in (
            ("learning_rate", self.learning_rate, True),
            ("weight_decay", self.weight_decay, False),
        ):
            number = float(value)
            if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if self.gradient_clip_norm is not None:
            clip = float(self.gradient_clip_norm)
            if not math.isfinite(clip) or clip <= 0.0:
                raise ValueError("gradient_clip_norm must be positive when supplied")
        if not isinstance(self.amsgrad, bool):
            raise TypeError("amsgrad must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.first_epoch, bool)
            or not isinstance(self.first_epoch, int)
            or self.first_epoch < 0
        ):
            raise ValueError("first_epoch must be a non-negative integer")
        if not isinstance(self.resume, bool):
            raise TypeError("resume must be a bool")
        if not isinstance(self.deterministic, bool):
            raise TypeError("deterministic must be a bool")
        if self.precision not in {"none", "bf16-mixed"}:
            raise ValueError("precision must be 'none' or 'bf16-mixed'")

    @property
    def effective_global_batch_groups(self) -> int:
        return self.global_physical_batch_groups * self.gradient_accumulation_steps

    @property
    def budget_mode(self) -> str:
        return "epochs" if self.max_epochs is not None else "legacy_steps"

    @property
    def epoch_budget(self) -> EpochBudget | None:
        if self.max_epochs is None:
            return None
        return EpochBudget(
            max_epochs=self.max_epochs,
            candidate_epochs=self.candidate_epochs,
            first_epoch=self.first_epoch,
        )

    @property
    def initial_sampler_epoch(self) -> int:
        if self.budget_mode == "epochs":
            return self.first_epoch
        if self.epoch is None:  # pragma: no cover - legacy config normalizes it
            raise AssertionError("legacy sampler epoch is unavailable")
        return self.epoch

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "stage": self.stage,
            "output_dir": str(self.output_dir),
            "expected_world_size": self.expected_world_size,
            "per_rank_batch_groups": self.per_rank_batch_groups,
            "global_physical_batch_groups": self.global_physical_batch_groups,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "amsgrad": self.amsgrad,
            "gradient_clip_norm": self.gradient_clip_norm,
            "seed": self.seed,
            "precision": self.precision,
            "resume": self.resume,
            "deterministic": self.deterministic,
            "effective_global_batch_groups": self.effective_global_batch_groups,
        }
        if self.budget_mode == "epochs":
            budget = self.epoch_budget
            if budget is None:  # pragma: no cover - property follows budget mode
                raise AssertionError("epoch budget is unavailable")
            value.update(budget.as_dict())
            value.update(
                {
                    "recovery_interval_epochs": epoch_value_json(
                        self.recovery_interval_epochs
                    ),
                    "heartbeat_interval_epochs": epoch_value_json(
                        self.heartbeat_interval_epochs
                    ),
                }
            )
        else:
            value.update(
                {
                    "max_steps": self.max_steps,
                    "candidate_steps": list(self.candidate_steps),
                    "epoch": self.epoch,
                    "recovery_interval_steps": self.recovery_interval_steps,
                    "heartbeat_interval_steps": self.heartbeat_interval_steps,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class ResolvedFormalTrainingPlan:
    """Internal optimizer-update plan resolved from a public training budget."""

    budget_mode: str
    max_optimizer_updates: int
    candidate_optimizer_updates: tuple[int, ...]
    recovery_interval_updates: int
    heartbeat_interval_updates: int
    epoch_plan: EpochTrainingPlan | None = None

    @property
    def schema_version(self) -> str:
        return (
            FORMAL_TRAINING_SCHEMA
            if self.budget_mode == "epochs"
            else LEGACY_FORMAL_TRAINING_SCHEMA
        )

    def completed_epochs(self, completed_optimizer_updates: int) -> Fraction | None:
        if self.epoch_plan is None:
            return None
        return self.epoch_plan.completed_epochs(completed_optimizer_updates)

    def sampler_epoch(
        self,
        config: FormalTrainingConfig,
        completed_optimizer_updates: int,
    ) -> int:
        if self.epoch_plan is not None:
            return self.epoch_plan.sampler_epoch(completed_optimizer_updates)
        if config.epoch is None:  # pragma: no cover - legacy config normalizes it
            raise AssertionError("legacy sampler epoch is unavailable")
        return config.epoch

    def is_epoch_boundary(self, completed_optimizer_updates: int) -> bool:
        return (
            self.epoch_plan is not None
            and self.epoch_plan.is_epoch_boundary(completed_optimizer_updates)
        )

    def candidate_epoch(self, completed_optimizer_updates: int) -> Fraction | None:
        if self.epoch_plan is None:
            return None
        return self.epoch_plan.candidate_epoch(completed_optimizer_updates)

    def as_dict(self) -> dict[str, object]:
        if self.epoch_plan is not None:
            return self.epoch_plan.as_dict()
        return {
            "budget_mode": "legacy_steps",
            "max_optimizer_updates": self.max_optimizer_updates,
            "candidate_optimizer_updates": list(self.candidate_optimizer_updates),
            "recovery_interval_updates": self.recovery_interval_updates,
            "heartbeat_interval_updates": self.heartbeat_interval_updates,
        }


def resolve_formal_training_plan(
    provider: FormalBatchProvider,
    config: FormalTrainingConfig,
) -> ResolvedFormalTrainingPlan:
    """Resolve an epoch budget after the dataset adapter is available."""

    if config.budget_mode == "legacy_steps":
        if (
            config.max_steps is None
            or config.recovery_interval_steps is None
            or config.heartbeat_interval_steps is None
        ):  # pragma: no cover - config validation normalizes these fields
            raise AssertionError("legacy step budget is incomplete")
        return ResolvedFormalTrainingPlan(
            budget_mode="legacy_steps",
            max_optimizer_updates=config.max_steps,
            candidate_optimizer_updates=config.candidate_steps,
            recovery_interval_updates=config.recovery_interval_steps,
            heartbeat_interval_updates=config.heartbeat_interval_steps,
        )
    epoch_size_groups = getattr(provider, "epoch_size_groups", None)
    if (
        isinstance(epoch_size_groups, bool)
        or not isinstance(epoch_size_groups, int)
        or epoch_size_groups < 1
    ):
        raise TypeError(
            "epoch-based formal training requires provider.epoch_size_groups "
            "as a positive integer"
        )
    budget = config.epoch_budget
    if budget is None:  # pragma: no cover - budget mode guarantees this
        raise AssertionError("epoch budget is unavailable")
    epoch_plan = EpochTrainingPlan.resolve(
        budget=budget,
        epoch_size_groups=epoch_size_groups,
        effective_global_batch_groups=config.effective_global_batch_groups,
        recovery_interval_epochs=config.recovery_interval_epochs,
        heartbeat_interval_epochs=config.heartbeat_interval_epochs,
    )
    return ResolvedFormalTrainingPlan(
        budget_mode="epochs",
        max_optimizer_updates=epoch_plan.max_optimizer_updates,
        candidate_optimizer_updates=epoch_plan.candidate_optimizer_updates,
        recovery_interval_updates=epoch_plan.recovery_interval_updates,
        heartbeat_interval_updates=epoch_plan.heartbeat_interval_updates,
        epoch_plan=epoch_plan,
    )


def formal_candidate_checkpoint_path(
    config: FormalTrainingConfig,
    candidate: Fraction | int,
) -> Path:
    """Return the canonical candidate path for a public epoch or legacy step."""

    if config.budget_mode == "epochs":
        epoch = parse_epoch_value(candidate, "candidate_epoch")
        if epoch not in config.candidate_epochs:
            raise ValueError("candidate epoch is not declared by this run")
        return config.output_dir / "checkpoints" / f"epoch_{epoch_value_slug(epoch)}.ckpt"
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        raise ValueError("legacy candidate step must be an integer")
    if candidate not in config.candidate_steps:
        raise ValueError("candidate step is not declared by this run")
    return config.output_dir / "checkpoints" / f"step_{candidate:06d}.ckpt"


def configure_deterministic_execution(enabled: bool) -> None:
    """Apply the registered deterministic CUDA/PyTorch execution policy.

    The Pista adapter invokes this before it selects a CUDA device and builds a
    model.  Generic callers invoke it again at formal-run entry, which is
    harmless and keeps the policy enforced at both public entry points.
    """

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    if not enabled:
        return
    # cuBLAS reads this setting at CUDA-context initialization.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@dataclass(frozen=True, slots=True)
class DistributedTrainingContext:
    """The explicit process/rank topology used for one formal run."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


@dataclass(frozen=True, slots=True)
class FormalTrainingStep:
    """One completed optimizer update, reduced over all active ranks."""

    step: int
    next_draw_index: int
    group_count: int
    candidate_count: int
    loss: float
    metrics: Mapping[str, float]
    completed_epochs: Fraction | None = None
    sampler_epoch: int | None = None

    def as_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "next_draw_index": self.next_draw_index,
            "group_count": self.group_count,
            "candidate_count": self.candidate_count,
            "loss": self.loss,
            "metrics": dict(self.metrics),
        }
        if self.completed_epochs is None:
            value["step"] = self.step
        else:
            value.update(
                {
                    "completed_epochs": epoch_value_json(self.completed_epochs),
                    "optimizer_update": self.step,
                    "sampler_epoch": self.sampler_epoch,
                }
            )
        return value


@dataclass(frozen=True, slots=True)
class FormalTrainingResult:
    """Bounded final summary; detailed histories remain on disk."""

    completed_steps: int
    mean_loss: float
    mean_metrics: Mapping[str, float]
    mean_metric_update_counts: Mapping[str, int]
    last_step: FormalTrainingStep
    candidate_checkpoints: tuple[Path, ...]
    recovered: bool
    context: DistributedTrainingContext
    plan: ResolvedFormalTrainingPlan

    @property
    def completed_optimizer_updates(self) -> int:
        return self.completed_steps

    @property
    def completed_epochs(self) -> Fraction | None:
        return self.plan.completed_epochs(self.completed_steps)


def initialize_distributed_training(device: torch.device | str) -> DistributedTrainingContext:
    """Initialize an environment-driven DDP context, or return a single rank.

    ``torchrun`` supplies ``RANK``, ``WORLD_SIZE`` and ``LOCAL_RANK``.  The
    function intentionally does not create a process group unless more than
    one rank was requested, keeping local smoke tests framework-free.
    """

    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    raw_rank = os.environ.get("RANK", "0")
    raw_local_rank = os.environ.get("LOCAL_RANK", raw_rank)
    try:
        world_size = int(raw_world_size)
        rank = int(raw_rank)
        local_rank = int(raw_local_rank)
    except ValueError as error:
        raise RuntimeError("RANK, WORLD_SIZE, and LOCAL_RANK must be integers") from error
    if world_size < 1 or not 0 <= rank < world_size or local_rank < 0:
        raise RuntimeError("invalid distributed rank environment")
    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA training was requested but CUDA is unavailable")
        # ``run.formal`` intentionally stores the portable device name
        # ``cuda``.  For a single-rank torchrun process there is no DDP rank
        # remapping, but PyTorch still requires an explicit CUDA index when
        # setting the device.  Fall back to LOCAL_RANK (normally zero) when
        # the portable request did not carry one.
        target_index = local_rank if world_size > 1 else (
            requested.index if requested.index is not None else local_rank
        )
        target = torch.device("cuda", target_index)
        torch.cuda.set_device(target)
        backend = "nccl"
    else:
        target = requested
        backend = "gloo"
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable for multi-rank training")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        if dist.get_world_size() != world_size or dist.get_rank() != rank:
            raise RuntimeError("existing process group differs from torchrun rank environment")
    elif dist.is_initialized() and (dist.get_world_size() != 1 or dist.get_rank() != 0):
        raise RuntimeError("single-rank configuration conflicts with an active process group")
    return DistributedTrainingContext(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=target,
    )


def _barrier(context: DistributedTrainingContext) -> None:
    if context.distributed:
        dist.barrier()


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _model_rank_recovery_state(model: nn.Module) -> dict[str, object] | None:
    """Collect optional rank-local model runtime state for exact recovery."""

    unwrapped = _unwrap(model)
    save_hook = getattr(unwrapped, "formal_rank_recovery_state", None)
    load_hook = getattr(unwrapped, "load_formal_rank_recovery_state", None)
    if (save_hook is None) != (load_hook is None):
        raise TypeError(
            "formal rank recovery requires both formal_rank_recovery_state and "
            "load_formal_rank_recovery_state"
        )
    if save_hook is None:
        return None
    if not callable(save_hook) or not callable(load_hook):
        raise TypeError("formal rank recovery hooks must be callable")
    value = save_hook()
    if not isinstance(value, Mapping):
        raise TypeError("formal_rank_recovery_state must return a mapping")
    return dict(value)


def _load_model_rank_recovery_state(
    model: nn.Module,
    value: object,
) -> None:
    """Restore optional rank-local model runtime state after model weights."""

    unwrapped = _unwrap(model)
    save_hook = getattr(unwrapped, "formal_rank_recovery_state", None)
    load_hook = getattr(unwrapped, "load_formal_rank_recovery_state", None)
    if (save_hook is None) != (load_hook is None):
        raise TypeError(
            "formal rank recovery requires both formal_rank_recovery_state and "
            "load_formal_rank_recovery_state"
        )
    if load_hook is None:
        if value is not None:
            raise RuntimeError(
                "rank recovery contains model runtime state but the active model "
                "does not support it"
            )
        return
    if not callable(save_hook) or not callable(load_hook):
        raise TypeError("formal rank recovery hooks must be callable")
    if not isinstance(value, Mapping):
        raise RuntimeError(
            "rank recovery is missing required model_rank_recovery_state"
        )
    load_hook(dict(value))


def _atomic_torch_save(path: Path, payload: Mapping[str, object], *, replace: bool) -> None:
    if not replace and (path.exists() or path.is_symlink()):
        raise FileExistsError(f"refusing to overwrite formal-training artifact: {path}")
    ensure_output_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
        # Isolated compute workers may write as an unprivileged identity. Keep the
        # artifact owner unchanged, but make completed checkpoints/recovery
        # states readable by the submitting user for monitoring and restart.
        # Keep checkpoint files writable by the shared run group while
        # leaving directory traversal controlled by the run-root policy.
        os.chmod(path, 0o664)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_torch_mapping(path: Path, *, name: str) -> Mapping[str, object]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - old torch compatibility
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must contain a mapping")
    return value


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} must be a mapping")
    return value


def _require_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{name} must be an integer >= {minimum}")
    return value


def _scalar_metrics(values: object) -> dict[str, Tensor]:
    if not isinstance(values, Mapping):
        return {}
    result: dict[str, Tensor] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not isinstance(value, Tensor) or value.numel() != 1:
            continue
        # Keep diagnostics on the training device until the single packed
        # reduction below.  Calling .cpu() here turns every named metric into
        # a separate synchronization point on CUDA.
        result[name] = value.detach().float().reshape(())
    return result


def _draw_group_count(draw: object) -> int:
    conditions = getattr(draw, "conditions", None)
    if not isinstance(conditions, tuple):
        raise RuntimeError("formal batch draw must expose conditions as a tuple")
    return len(conditions)


def _draw_candidate_count(draw: object) -> int:
    batch = getattr(draw, "batch", None)
    graphs = getattr(batch, "graphs", None)
    candidates = getattr(graphs, "p_X", None)
    if not isinstance(candidates, Tensor) or candidates.ndim < 1:
        raise RuntimeError("formal batch draw must expose batch.graphs.p_X")
    return int(candidates.shape[0])


def _draw_padded_graph_cells(draw: object) -> int:
    """Return the dense graph-cell proxy used by rank-balanced batching."""

    logical = getattr(draw, "logical_padded_graph_cells", None)
    if logical is not None:
        if isinstance(logical, bool) or not isinstance(logical, int) or logical < 1:
            raise RuntimeError("draw.logical_padded_graph_cells must be a positive integer")
        return logical
    batch = getattr(draw, "batch", None)
    graphs = getattr(batch, "graphs", None)
    candidates = getattr(graphs, "p_X", None)
    if not isinstance(candidates, Tensor) or candidates.ndim < 2:
        raise RuntimeError("formal batch draw must expose rank-two-or-higher batch.graphs.p_X")
    return int(candidates.shape[0]) * int(candidates.shape[1]) * int(candidates.shape[1])


def _draw_next_index(draw: object) -> int:
    return _require_integer(getattr(draw, "next_draw_index", None), "draw.next_draw_index")


def _reduced_scalar(value: float, context: DistributedTrainingContext) -> float:
    return _reduced_scalars((value,), context)[0]


def _reduced_scalars(
    values: tuple[Tensor | float, ...], context: DistributedTrainingContext
) -> tuple[float, ...]:
    """Reduce several scalar values with one device vector and one collective."""

    if not values:
        return ()
    packed: list[Tensor] = []
    for value in values:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise RuntimeError("cannot reduce a non-scalar tensor")
            scalar = value.detach().to(device=context.device, dtype=torch.float64).reshape(())
        else:
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("cannot reduce a non-finite scalar")
            scalar = torch.tensor(number, dtype=torch.float64, device=context.device)
        packed.append(scalar)
    reduced = torch.stack(packed)
    if context.distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced /= context.world_size
    return tuple(float(value) for value in reduced.detach().cpu().tolist())


def _reduced_scalar_bounds(
    values: tuple[Tensor | float, ...], context: DistributedTrainingContext
) -> tuple[tuple[float, float], ...]:
    """Return per-value ``(min, max)`` across ranks without changing training.

    This is intentionally used only for durable heartbeat/candidate telemetry.
    Average metrics remain on the inexpensive every-step collective path.
    """

    if not values:
        return ()
    packed: list[Tensor] = []
    for value in values:
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise RuntimeError("cannot bound a non-scalar tensor")
            scalar = value.detach().to(device=context.device, dtype=torch.float64).reshape(())
        else:
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError("cannot bound a non-finite scalar")
            scalar = torch.tensor(number, dtype=torch.float64, device=context.device)
        packed.append(scalar)
    lower = torch.stack(packed)
    upper = lower.clone()
    if context.distributed:
        dist.all_reduce(lower, op=dist.ReduceOp.MIN)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX)
    return tuple(zip(
        (float(value) for value in lower.detach().cpu().tolist()),
        (float(value) for value in upper.detach().cpu().tolist()),
        strict=True,
    ))


def _reduced_metrics(
    values: Mapping[str, Tensor | float], context: DistributedTrainingContext
) -> dict[str, float]:
    packed_names: list[str] = []
    packed_values: list[Tensor | float] = []
    for name in sorted(values):
        value = values[name]
        if isinstance(value, Tensor) and value.numel() != 1:
            continue
        if not isinstance(value, Tensor) and not isinstance(value, (int, float)):
            continue
        packed_names.append(name)
        packed_values.append(value)
    reduced = _reduced_scalars(tuple(packed_values), context)
    return dict(zip(packed_names, reduced, strict=True))


def _move_draw_to_device(draw: object, device: torch.device) -> object:
    """Move a CPU-prefetched draw through its adapter-owned batch mover."""

    mover = getattr(draw, "to_device", None)
    if callable(mover):
        return mover(device)
    return draw


def _step_from_mapping(value: object) -> FormalTrainingStep:
    row = _require_mapping(value, "recovery.last_step")
    metrics = _require_mapping(row.get("metrics"), "recovery.last_step.metrics")
    metric_values: dict[str, float] = {}
    for name, number in metrics.items():
        if not isinstance(name, str) or isinstance(number, bool) or not isinstance(number, (int, float)):
            raise RuntimeError("recovery.last_step.metrics must be scalar values")
        metric_values[name] = float(number)
    loss = row.get("loss")
    if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
        raise RuntimeError("recovery.last_step.loss must be finite")
    epoch_value = row.get("completed_epochs")
    completed_epochs = (
        None
        if epoch_value is None
        else parse_epoch_value(
            epoch_value, "recovery.last_update.completed_epochs", allow_zero=True
        )
    )
    update_value = row.get("optimizer_update", row.get("step"))
    sampler_epoch = row.get("sampler_epoch")
    return FormalTrainingStep(
        step=_require_integer(
            update_value, "recovery.last_update.optimizer_update", minimum=1
        ),
        next_draw_index=_require_integer(
            row.get("next_draw_index"), "recovery.last_step.next_draw_index"
        ),
        group_count=_require_integer(
            row.get("group_count"), "recovery.last_step.group_count", minimum=1
        ),
        candidate_count=_require_integer(
            row.get("candidate_count"), "recovery.last_step.candidate_count", minimum=1
        ),
        loss=float(loss),
        metrics=metric_values,
        completed_epochs=completed_epochs,
        sampler_epoch=(
            None
            if sampler_epoch is None
            else _require_integer(
                sampler_epoch, "recovery.last_update.sampler_epoch"
            )
        ),
    )


def _seed_rank_rng(config: FormalTrainingConfig, context: DistributedTrainingContext) -> torch.Generator:
    rank_seed = config.seed + context.rank * 1_000_003
    torch.manual_seed(rank_seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed(rank_seed)
    generator = torch.Generator(device=context.device)
    generator.manual_seed(rank_seed + 97)
    return generator


def _recovery_paths(config: FormalTrainingConfig, context: DistributedTrainingContext) -> tuple[Path, Path]:
    root = config.output_dir / "recovery"
    return root / "global.pt", root / f"rank-{context.rank:02d}.pt"


def _checkpoint_request_path(config: FormalTrainingConfig) -> Path:
    """Return the shared, operator-created request consumed at a safe step."""

    return config.output_dir / FORMAL_CHECKPOINT_REQUEST_FILENAME


def _checkpoint_requested(
    config: FormalTrainingConfig,
    context: DistributedTrainingContext,
) -> bool:
    """Collectively observe whether an operator requested an early recovery.

    The request is deliberately outside ``FormalTrainingConfig`` so it does
    not alter checkpoint identity or invalidate an otherwise exact resume. A
    max-reduced flag makes a delayed GPFS metadata observation on one rank
    sufficient for every DDP rank to enter the same recovery barriers.
    """

    request_path = _checkpoint_request_path(config)
    requested = request_path.is_file() and not request_path.is_symlink()
    if context.distributed:
        flag = torch.tensor(
            int(requested),
            dtype=torch.int64,
            device=context.device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        requested = bool(flag.item())
    return requested


def _clear_checkpoint_request(
    config: FormalTrainingConfig,
    context: DistributedTrainingContext,
) -> None:
    """Remove one consumed request only after every rank saved its state."""

    if context.rank == 0:
        try:
            _checkpoint_request_path(config).unlink()
        except FileNotFoundError:
            pass
    _barrier(context)


def _recovery_config_matches(value: object, config: FormalTrainingConfig) -> bool:
    current = config.as_dict()
    if value == current:
        return True
    # A deliberately explicit continuation amendment permits the optimized
    # Pista branch to resume model/optimizer/RNG state while changing only
    # physical batch topology (including rank count and gradient accumulation)
    # and the output directory.  The effective global batch remains immutable;
    # leaving it in the compared mapping below makes that invariant executable
    # rather than relying on this comment. Ordinary runs do not receive this
    # escape hatch.
    if not isinstance(value, Mapping):
        return False
    batch_resume = os.environ.get(FORMAL_ALLOW_BATCH_RESUME_AMENDMENT_ENV) == "1"
    exposure_resume = (
        os.environ.get(FORMAL_ALLOW_EXPOSURE_RESUME_AMENDMENT_ENV) == "1"
    )
    exposure_schedule_resume = (
        os.environ.get(FORMAL_ALLOW_EXPOSURE_SCHEDULE_RESUME_AMENDMENT_ENV)
        == "1"
    )
    if not batch_resume and not exposure_resume and not exposure_schedule_resume:
        return False
    mutable = {
        "output_dir",
        "expected_world_size",
        "per_rank_batch_groups",
        "global_physical_batch_groups",
        "gradient_accumulation_steps",
    }
    # The exposure-resume gate is deliberately separate from the historical
    # physical-batch consolidation gate.  It admits a changed effective group
    # batch only for an explicitly confirmed, isolated continuation.  Step and
    # candidate schedules, optimizer settings, precision, seed, and every
    # other formal field must still match at this admission point; a later
    # exposure-mapped formal schedule needs its own explicit gate.
    if exposure_resume or exposure_schedule_resume:
        mutable.add("effective_global_batch_groups")
    if exposure_schedule_resume:
        mutable.update(
            {
                "max_steps",
                "candidate_steps",
                "recovery_interval_steps",
                "heartbeat_interval_steps",
                "max_epochs",
                "candidate_epochs",
                "recovery_interval_epochs",
                "heartbeat_interval_epochs",
            }
        )
    previous = dict(value)
    for key in mutable:
        previous.pop(key, None)
        current.pop(key, None)
    return previous == current


def _mean_metrics(
    metric_sums: Mapping[str, float],
    metric_update_counts: Mapping[str, int],
) -> dict[str, float]:
    if set(metric_sums) != set(metric_update_counts):
        raise RuntimeError("formal metric sums and update counts differ")
    return {
        name: float(metric_sums[name]) / int(metric_update_counts[name])
        for name in sorted(metric_sums)
        if int(metric_update_counts[name]) > 0
    }


def _restore_if_available(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    config: FormalTrainingConfig,
    context: DistributedTrainingContext,
    plan: ResolvedFormalTrainingPlan,
    model_recovery_contract: Mapping[str, object] | None,
) -> tuple[
    int,
    int,
    float,
    dict[str, float],
    dict[str, int],
    FormalTrainingStep | None,
    bool,
]:
    global_path, rank_path = _recovery_paths(config, context)
    has_global = global_path.is_file()
    has_rank = rank_path.is_file()
    if not has_global and not has_rank:
        return 0, context.rank, 0.0, {}, {}, None, False
    if not config.resume:
        raise RuntimeError("formal recovery state exists but config.resume is false")
    if has_global != has_rank:
        raise RuntimeError("formal recovery state is incomplete; global and rank states must coexist")
    global_state = _load_torch_mapping(global_path, name="formal global recovery")
    rank_state = _load_torch_mapping(rank_path, name="formal rank recovery")
    for name, state in (("global", global_state), ("rank", rank_state)):
        if state.get("schema_version") != plan.schema_version:
            raise RuntimeError(f"unsupported {name} recovery schema")
        if not _recovery_config_matches(state.get("run_config"), config):
            raise RuntimeError(f"{name} recovery configuration differs from this run contract")
        if _require_integer(state.get("world_size"), f"{name} recovery.world_size", minimum=1) != context.world_size:
            raise RuntimeError(f"{name} recovery world size differs from the active allocation")
        if plan.epoch_plan is not None and state.get("epoch_plan") != plan.as_dict():
            raise RuntimeError(f"{name} recovery epoch plan differs from this run")
        if model_recovery_contract is not None and state.get(
            "model_recovery_contract"
        ) != dict(model_recovery_contract):
            raise RuntimeError(
                f"{name} recovery model-side schedule contract differs from this run"
            )
    if _require_integer(rank_state.get("rank"), "rank recovery.rank") != context.rank:
        raise RuntimeError("rank recovery does not belong to this DDP rank")
    completed_key = (
        "completed_optimizer_updates"
        if plan.epoch_plan is not None
        else "completed_steps"
    )
    completed_steps = _require_integer(
        global_state.get(completed_key), f"global recovery.{completed_key}"
    )
    if completed_steps > plan.max_optimizer_updates:
        raise RuntimeError("formal recovery exceeds the configured training budget")
    if (
        _require_integer(
            rank_state.get(completed_key), f"rank recovery.{completed_key}"
        )
        != completed_steps
    ):
        raise RuntimeError("global and rank recovery states disagree on completed progress")
    model_state = _require_mapping(global_state.get("state_dict"), "global recovery.state_dict")
    optimizer_state = _require_mapping(
        global_state.get("optimizer_state"), "global recovery.optimizer_state"
    )
    _unwrap(model).load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(dict(optimizer_state))
    _load_model_rank_recovery_state(
        model,
        rank_state.get("model_rank_recovery_state"),
    )
    generator_state = rank_state.get("generator_state")
    if not isinstance(generator_state, Tensor):
        raise RuntimeError("rank recovery.generator_state must be a tensor")
    generator.set_state(generator_state)
    cpu_rng_state = rank_state.get("cpu_rng_state")
    if not isinstance(cpu_rng_state, Tensor):
        raise RuntimeError("rank recovery.cpu_rng_state must be a tensor")
    torch.set_rng_state(cpu_rng_state)
    if context.device.type == "cuda":
        cuda_rng_state = rank_state.get("cuda_rng_state")
        if not isinstance(cuda_rng_state, Tensor):
            raise RuntimeError("rank recovery.cuda_rng_state must be a tensor")
        torch.cuda.set_rng_state(cuda_rng_state, device=context.device)
    draw_index = _require_integer(rank_state.get("next_draw_index"), "rank recovery.next_draw_index")
    loss_sum = global_state.get("loss_sum")
    if isinstance(loss_sum, bool) or not isinstance(loss_sum, (int, float)) or not math.isfinite(float(loss_sum)):
        raise RuntimeError("global recovery.loss_sum must be finite")
    raw_metric_sums = global_state.get("metric_sums", {})
    raw_metric_counts = global_state.get("metric_update_counts", {})
    if not isinstance(raw_metric_sums, Mapping) or not isinstance(
        raw_metric_counts, Mapping
    ):
        raise RuntimeError("global recovery metric accumulators must be objects")
    if set(raw_metric_sums) != set(raw_metric_counts):
        raise RuntimeError("global recovery metric accumulators differ")
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}
    for raw_name in raw_metric_sums:
        if not isinstance(raw_name, str) or not raw_name:
            raise RuntimeError("global recovery metric name must be non-empty")
        raw_sum = raw_metric_sums[raw_name]
        if (
            isinstance(raw_sum, bool)
            or not isinstance(raw_sum, (int, float))
            or not math.isfinite(float(raw_sum))
        ):
            raise RuntimeError(
                f"global recovery metric sum {raw_name!r} must be finite"
            )
        count = _require_integer(
            raw_metric_counts[raw_name],
            f"global recovery.metric_update_counts[{raw_name!r}]",
        )
        if count > completed_steps:
            raise RuntimeError("global recovery metric count exceeds completed progress")
        metric_sums[raw_name] = float(raw_sum)
        metric_counts[raw_name] = count
    last_key = "last_update" if plan.epoch_plan is not None else "last_step"
    last_step = None if completed_steps == 0 else _step_from_mapping(global_state.get(last_key))
    if last_step is not None and last_step.step != completed_steps:
        raise RuntimeError("global recovery last update disagrees with completed progress")
    return (
        completed_steps,
        draw_index,
        float(loss_sum),
        metric_sums,
        metric_counts,
        last_step,
        True,
    )


def _save_recovery(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    config: FormalTrainingConfig,
    context: DistributedTrainingContext,
    plan: ResolvedFormalTrainingPlan,
    completed_steps: int,
    draw_index: int,
    loss_sum: float,
    metric_sums: Mapping[str, float],
    metric_update_counts: Mapping[str, int],
    last_step: FormalTrainingStep,
    model_recovery_contract: Mapping[str, object] | None,
) -> None:
    global_path, rank_path = _recovery_paths(config, context)
    completed_key = (
        "completed_optimizer_updates"
        if plan.epoch_plan is not None
        else "completed_steps"
    )
    rank_payload: dict[str, object] = {
        "schema_version": plan.schema_version,
        "artifact_kind": "rank_recovery",
        "run_config": config.as_dict(),
        "rank": context.rank,
        "world_size": context.world_size,
        completed_key: completed_steps,
        "next_draw_index": draw_index,
        "generator_state": generator.get_state(),
        "cpu_rng_state": torch.get_rng_state(),
    }
    if plan.epoch_plan is not None:
        rank_payload["epoch_plan"] = plan.as_dict()
    if model_recovery_contract is not None:
        rank_payload["model_recovery_contract"] = dict(model_recovery_contract)
    model_rank_recovery_state = _model_rank_recovery_state(model)
    if model_rank_recovery_state is not None:
        rank_payload["model_rank_recovery_state"] = model_rank_recovery_state
    if context.device.type == "cuda":
        rank_payload["cuda_rng_state"] = torch.cuda.get_rng_state(context.device)
    _atomic_torch_save(rank_path, rank_payload, replace=True)
    _barrier(context)
    if context.rank == 0:
        global_payload: dict[str, object] = {
            "schema_version": plan.schema_version,
            "artifact_kind": "global_recovery",
            "run_config": config.as_dict(),
            "world_size": context.world_size,
            completed_key: completed_steps,
            "loss_sum": loss_sum,
            "metric_sums": dict(metric_sums),
            "metric_update_counts": dict(metric_update_counts),
            "state_dict": _unwrap(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }
        global_payload[
            "last_update" if plan.epoch_plan is not None else "last_step"
        ] = last_step.as_dict()
        if plan.epoch_plan is not None:
            global_payload["epoch_plan"] = plan.as_dict()
        if model_recovery_contract is not None:
            global_payload["model_recovery_contract"] = dict(
                model_recovery_contract
            )
        _atomic_torch_save(global_path, global_payload, replace=True)
    _barrier(context)


def _candidate_path(
    config: FormalTrainingConfig,
    plan: ResolvedFormalTrainingPlan,
    completed_steps: int,
) -> Path:
    candidate_epoch = plan.candidate_epoch(completed_steps)
    if candidate_epoch is not None:
        return formal_candidate_checkpoint_path(config, candidate_epoch)
    return formal_candidate_checkpoint_path(config, completed_steps)


def _candidate_paths(
    config: FormalTrainingConfig,
    plan: ResolvedFormalTrainingPlan,
) -> tuple[Path, ...]:
    if plan.epoch_plan is not None:
        return tuple(
            formal_candidate_checkpoint_path(config, epoch)
            for epoch in plan.epoch_plan.budget.candidate_epochs
        )
    return tuple(
        formal_candidate_checkpoint_path(config, step)
        for step in plan.candidate_optimizer_updates
    )


def _save_candidate(
    *,
    model: nn.Module,
    config: FormalTrainingConfig,
    context: DistributedTrainingContext,
    plan: ResolvedFormalTrainingPlan,
    completed_steps: int,
    loss_sum: float,
    metric_sums: Mapping[str, float],
    metric_update_counts: Mapping[str, int],
    last_step: FormalTrainingStep,
    run_metadata: Mapping[str, object],
) -> Path:
    path = _candidate_path(config, plan, completed_steps)
    _barrier(context)
    if context.rank == 0:
        payload: dict[str, object] = {
            "schema_version": plan.schema_version,
            "artifact_kind": "candidate_checkpoint",
            "run_config": config.as_dict(),
            "run_metadata": dict(run_metadata),
            "mean_loss": loss_sum / completed_steps,
            "mean_metrics": _mean_metrics(metric_sums, metric_update_counts),
            "mean_metric_update_counts": dict(metric_update_counts),
            "state_dict": _unwrap(model).state_dict(),
        }
        if plan.epoch_plan is None:
            payload["completed_steps"] = completed_steps
            payload["last_step"] = last_step.as_dict()
        else:
            candidate_epoch = plan.candidate_epoch(completed_steps)
            if candidate_epoch is None:  # pragma: no cover - caller gates candidates
                raise AssertionError("candidate update has no epoch coordinate")
            payload.update(
                {
                    "candidate_epoch": epoch_value_json(candidate_epoch),
                    "completed_epochs": epoch_value_json(
                        plan.epoch_plan.completed_epochs(completed_steps)
                    ),
                    "completed_optimizer_updates": completed_steps,
                    "epoch_plan": plan.as_dict(),
                    "last_update": last_step.as_dict(),
                }
            )
        _atomic_torch_save(path, payload, replace=False)
    _barrier(context)
    return path


def _write_heartbeat(
    *,
    config: FormalTrainingConfig,
    context: DistributedTrainingContext,
    plan: ResolvedFormalTrainingPlan,
    completed_steps: int,
    loss_sum: float,
    metric_sums: Mapping[str, float],
    metric_update_counts: Mapping[str, int],
    last_step: FormalTrainingStep,
    recovered: bool,
    state: str,
    profile: Mapping[str, object] | None = None,
) -> None:
    if context.rank != 0:
        return
    payload: dict[str, object] = {
        "schema_version": plan.schema_version,
        "state": state,
        "stage": config.stage,
        "mean_loss": loss_sum / completed_steps,
        "mean_metrics": _mean_metrics(metric_sums, metric_update_counts),
        "mean_metric_update_counts": dict(metric_update_counts),
        "expected_world_size": config.expected_world_size,
        "active_world_size": context.world_size,
        "global_physical_batch_groups": config.global_physical_batch_groups,
        "effective_global_batch_groups": config.effective_global_batch_groups,
        "recovered": recovered,
    }
    if plan.epoch_plan is None:
        payload.update(
            {
                "completed_steps": completed_steps,
                "max_steps": config.max_steps,
                "last_step": last_step.as_dict(),
                "candidate_steps": list(config.candidate_steps),
            }
        )
        history_coordinate = f"step_{completed_steps:06d}"
    else:
        completed_epochs = plan.epoch_plan.completed_epochs(completed_steps)
        payload.update(
            {
                "completed_epochs": epoch_value_json(completed_epochs),
                "max_epochs": epoch_value_json(plan.epoch_plan.budget.max_epochs),
                "completed_optimizer_updates": completed_steps,
                "last_update": last_step.as_dict(),
                "candidate_epochs": [
                    epoch_value_json(value)
                    for value in plan.epoch_plan.budget.candidate_epochs
                ],
                "epoch_plan": plan.as_dict(),
            }
        )
        history_coordinate = f"epoch_{epoch_value_slug(completed_epochs)}"
    if profile is not None:
        payload["profile"] = dict(profile)
    names = {
        "running": "heartbeat.json",
        "pilot_complete": "pilot_complete.json",
        "complete": "training_complete.json",
    }
    try:
        name = names[state]
    except KeyError as error:
        raise ValueError(f"unsupported formal training state: {state}") from error
    reports_root = config.output_dir / "reports"
    write_json(reports_root / name, payload)
    if os.environ.get(FORMAL_HEARTBEAT_HISTORY_ENV) == "1":
        history_path = (
            reports_root
            / "heartbeat_history"
            / f"{history_coordinate}_{state}.json"
        )
        if history_path.exists() or history_path.is_symlink():
            raise FileExistsError(
                f"refusing to overwrite formal heartbeat history: {history_path}"
            )
        write_json(history_path, payload)


def _profile_series(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("formal profile series cannot be empty")
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "mean": sum(values) / len(values),
        "median": median,
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _formal_profile_payload(
    samples: Mapping[str, list[float]],
    *,
    world_size: int,
    first_completed_step: int,
    last_completed_step: int,
    boundary_metrics: Mapping[str, float],
) -> dict[str, object]:
    step_wall = samples["step_wall_seconds"]
    elapsed = sum(step_wall)
    if elapsed <= 0.0:
        raise RuntimeError("formal profile elapsed time must be positive")
    per_rank_groups = sum(samples["groups"])
    per_rank_candidates = sum(samples["candidates"])
    per_rank_cells = sum(samples["logical_padded_graph_cells"])
    cuda_total_memory_bytes = None
    if torch.cuda.is_available():
        cuda_total_memory_bytes = int(
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        )
    return {
        "schema_version": "retro_guided_generation.formal_step_profile.v1",
        "sample_count": len(step_wall),
        "first_completed_step": first_completed_step,
        "last_completed_step": last_completed_step,
        "step_wall_seconds": _profile_series(step_wall),
        "prefetch_wait_seconds": _profile_series(samples["prefetch_wait_seconds"]),
        "host_to_device_seconds": _profile_series(samples["host_to_device_seconds"]),
        "optimizer_window_wall_seconds": elapsed,
        "global_groups_per_second": per_rank_groups * world_size / elapsed,
        "global_candidates_per_second": per_rank_candidates * world_size / elapsed,
        "global_logical_graph_cells_per_second": per_rank_cells * world_size / elapsed,
        "optimizer_window_GPU_hours": elapsed * world_size / 3600.0,
        "boundary_rank_step_wall_seconds_min": boundary_metrics.get(
            "runtime_rank_step_wall_seconds_min"
        ),
        "boundary_rank_step_wall_seconds_max": boundary_metrics.get(
            "runtime_rank_step_wall_seconds_max"
        ),
        "boundary_cuda_peak_allocated_bytes_max": boundary_metrics.get(
            "runtime_cuda_peak_allocated_bytes_max"
        ),
        "boundary_cuda_peak_reserved_bytes_max": boundary_metrics.get(
            "runtime_cuda_peak_reserved_bytes_max"
        ),
        "cuda_total_memory_bytes": cuda_total_memory_bytes,
    }


def _pilot_stop_after_optimizer_updates(
    config: FormalTrainingConfig,
    plan: ResolvedFormalTrainingPlan,
) -> int | None:
    """Return an explicitly requested pilot boundary as optimizer updates.

    The boundary is deliberately an environment-only execution control.  It
    is excluded from ``FormalTrainingConfig`` and therefore cannot alter the
    recovery identity, optimizer, batch, or training equations.  Reaching it
    always publishes a complete recovery before the bounded process exits.
    """

    raw_epochs = os.environ.get(FORMAL_STOP_AFTER_EPOCHS_ENV)
    raw_steps = os.environ.get(FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV)
    if raw_epochs is not None and raw_steps is not None:
        raise RuntimeError("configure only one formal pilot boundary")
    if plan.epoch_plan is not None:
        if raw_steps is not None:
            raise RuntimeError(
                f"epoch-based runs use {FORMAL_STOP_AFTER_EPOCHS_ENV}, not "
                f"{FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV}"
            )
        if raw_epochs is None:
            return None
        requested = parse_epoch_value(raw_epochs, FORMAL_STOP_AFTER_EPOCHS_ENV)
        value = math.ceil(requested * plan.epoch_plan.optimizer_updates_per_epoch)
        if value < 1 or value >= plan.max_optimizer_updates:
            raise RuntimeError(
                f"{FORMAL_STOP_AFTER_EPOCHS_ENV} must resolve inside (0, max_epochs)"
            )
        return value
    if raw_epochs is not None:
        raise RuntimeError(
            f"legacy step runs use {FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV}"
        )
    if raw_steps is None:
        return None
    try:
        value = int(raw_steps)
    except ValueError as error:
        raise RuntimeError(
            f"{FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV} must be an integer"
        ) from error
    if value < 1 or value >= plan.max_optimizer_updates:
        raise RuntimeError(
            f"{FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV} must be inside [1, max_steps)"
        )
    return value


def _positive_environment_integer(name: str, *, default: int) -> int:
    """Read one opt-in positive integer without changing recovery identity."""

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return value


def _wrap_for_ddp(model: nn.Module, context: DistributedTrainingContext) -> nn.Module:
    if not context.distributed:
        return model
    # The four-rank topology can place different branch lanes on different
    # ranks.  In that case a branch-specific projector may be legitimately
    # unused on one rank for an iteration, which otherwise makes DDP report
    # an unfinished reduction on the next forward.  Keep the normal two-rank
    # path fast and enable the extra graph traversal only for the explicitly
    # authorized fast4 retry through the launcher environment.
    find_unused_parameters = os.environ.get("FORMAL_DDP_FIND_UNUSED_PARAMETERS", "0") == "1"
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.device.index],
            output_device=context.device.index,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            find_unused_parameters=find_unused_parameters,
        )
    return DistributedDataParallel(
        model,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )


def run_formal_training(
    *,
    model: nn.Module,
    provider: FormalBatchProvider,
    config: FormalTrainingConfig,
    device: torch.device | str,
    run_metadata: Mapping[str, object] | None = None,
    on_step: Callable[[FormalTrainingStep], None] | None = None,
) -> FormalTrainingResult:
    """Run exactly the contracted epoch budget with DDP-safe recovery.

    Each rank consumes its own deterministic draw lane, offset by ``rank`` and
    advanced by ``world_size``.  The provider must return a complete groupwise
    batch of ``per_rank_batch_groups`` every microbatch; a short batch is a
    hard error rather than an implicit change to the experiment's batch budget.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(provider, FormalBatchProvider):
        raise TypeError("provider must implement the formal batch-provider contract")
    if not isinstance(config, FormalTrainingConfig):
        raise TypeError("config must be FormalTrainingConfig")
    plan = resolve_formal_training_plan(provider, config)
    configure_deterministic_execution(config.deterministic)
    metadata = {} if run_metadata is None else dict(run_metadata)
    if plan.epoch_plan is not None:
        plan_hook = getattr(model, "configure_formal_training_plan", None)
        if plan_hook is not None and not callable(plan_hook):
            raise TypeError(
                "configure_formal_training_plan must be callable when provided"
            )
        epoch_hook = getattr(model, "configure_formal_epoch_plan", None)
        if epoch_hook is not None and not callable(epoch_hook):
            raise TypeError("configure_formal_epoch_plan must be callable when provided")
        if callable(plan_hook):
            resolved_schedules = plan_hook(
                optimizer_updates_per_epoch=(
                    plan.epoch_plan.optimizer_updates_per_epoch
                ),
                max_optimizer_updates=plan.max_optimizer_updates,
                candidate_optimizer_updates=plan.candidate_optimizer_updates,
            )
        elif callable(epoch_hook):
            resolved_schedules = epoch_hook(
                optimizer_updates_per_epoch=(
                    plan.epoch_plan.optimizer_updates_per_epoch
                )
            )
        else:
            resolved_schedules = None
        if resolved_schedules is not None:
            if not isinstance(resolved_schedules, Mapping):
                raise TypeError(
                    "formal plan configuration hook must return resolved schedule metadata"
                )
            if "resolved_epoch_schedules" in metadata:
                raise ValueError(
                    "run_metadata.resolved_epoch_schedules is owned by the training runner"
                )
            metadata["resolved_epoch_schedules"] = dict(resolved_schedules)
    recovery_contract_hook = getattr(model, "formal_recovery_contract", None)
    if recovery_contract_hook is not None and not callable(recovery_contract_hook):
        raise TypeError("formal_recovery_contract must be callable when provided")
    model_recovery_contract = None
    if callable(recovery_contract_hook):
        resolved_recovery_contract = recovery_contract_hook()
        if not isinstance(resolved_recovery_contract, Mapping):
            raise TypeError("formal_recovery_contract must return a mapping")
        model_recovery_contract = dict(resolved_recovery_contract)
        if "model_recovery_contract" in metadata:
            raise ValueError(
                "run_metadata.model_recovery_contract is owned by the training runner"
            )
        metadata["model_recovery_contract"] = dict(model_recovery_contract)
    context = initialize_distributed_training(device)
    if config.precision == "bf16-mixed" and context.device.type != "cuda":
        raise RuntimeError("bf16-mixed formal training requires a CUDA device")
    if context.world_size != config.expected_world_size:
        raise RuntimeError(
            f"formal run requires world_size={config.expected_world_size}, "
            f"but allocation provides {context.world_size}"
        )
    ensure_output_directory(config.output_dir)
    model.to(context.device)
    wrapped_model = _wrap_for_ddp(model, context)
    wrapped_model.train()
    parameters = [parameter for parameter in _unwrap(wrapped_model).parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("formal training model exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        amsgrad=config.amsgrad,
    )
    generator = _seed_rank_rng(config, context)
    (
        completed_steps,
        draw_index,
        loss_sum,
        metric_sums,
        metric_update_counts,
        last_step,
        recovered,
    ) = _restore_if_available(
        model=wrapped_model,
        optimizer=optimizer,
        generator=generator,
        config=config,
        context=context,
        plan=plan,
        model_recovery_contract=model_recovery_contract,
    )
    pilot_stop_after = _pilot_stop_after_optimizer_updates(config, plan)
    profile_steps = os.environ.get(FORMAL_PROFILE_STEPS_ENV) == "1"
    if profile_steps and pilot_stop_after is None:
        raise RuntimeError(
            f"{FORMAL_PROFILE_STEPS_ENV}=1 requires "
            + (
                FORMAL_STOP_AFTER_EPOCHS_ENV
                if plan.epoch_plan is not None
                else FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV
            )
        )
    if pilot_stop_after is not None and completed_steps > pilot_stop_after:
        raise RuntimeError(
            "formal recovery completed_steps exceeds the requested pilot boundary"
        )
    if pilot_stop_after is not None and completed_steps == pilot_stop_after:
        if last_step is None:  # pragma: no cover - recovery schema rejects this
            raise RuntimeError("pilot recovery state has no last_step")
        _write_heartbeat(
            config=config,
            context=context,
            plan=plan,
            completed_steps=completed_steps,
            loss_sum=loss_sum,
            metric_sums=metric_sums,
            metric_update_counts=metric_update_counts,
            last_step=last_step,
            recovered=recovered,
            state="pilot_complete",
        )
        return FormalTrainingResult(
            completed_steps=completed_steps,
            mean_loss=loss_sum / completed_steps,
            mean_metrics=_mean_metrics(metric_sums, metric_update_counts),
            mean_metric_update_counts=dict(metric_update_counts),
            last_step=last_step,
            candidate_checkpoints=_candidate_paths(config, plan),
            recovered=recovered,
            context=context,
            plan=plan,
        )
    if completed_steps == plan.max_optimizer_updates:
        if last_step is None:  # pragma: no cover - recovery schema rejects this
            raise RuntimeError("terminal recovery state has no last_step")
        _write_heartbeat(
            config=config,
            context=context,
            plan=plan,
            completed_steps=completed_steps,
            loss_sum=loss_sum,
            metric_sums=metric_sums,
            metric_update_counts=metric_update_counts,
            last_step=last_step,
            recovered=recovered,
            state="complete",
        )
        return FormalTrainingResult(
            completed_steps=completed_steps,
            mean_loss=loss_sum / completed_steps,
            mean_metrics=_mean_metrics(metric_sums, metric_update_counts),
            mean_metric_update_counts=dict(metric_update_counts),
            last_step=last_step,
            candidate_checkpoints=_candidate_paths(config, plan),
            recovered=recovered,
            context=context,
            plan=plan,
        )

    candidate_steps = frozenset(plan.candidate_optimizer_updates)
    rank_balanced_batch = getattr(provider, "next_rank_balanced_batch", None)
    rank_balanced_enabled = bool(getattr(provider, "supports_cost_balanced_rank_batching", False))
    if rank_balanced_enabled and not callable(rank_balanced_batch):
        raise TypeError(
            "provider advertises cost-balanced rank batching but does not expose "
            "next_rank_balanced_batch"
        )
    # CPU prefetch is safe across gradient-accumulated micro-batches: the
    # provider receives an explicit draw index, and the next draw is queued
    # only after the current draw's successor is known.  Keeping this path on
    # for accumulation > 1 overlaps the CPU materialization of the next
    # micro-batch with the current GPU forward/backward instead of making the
    # effective batch a serial CPU/GPU pipeline.
    prefetch_enabled = bool(getattr(provider, "supports_cpu_prefetch", False))
    prefetch_depth = _positive_environment_integer(
        FORMAL_CPU_PREFETCH_DEPTH_ENV,
        default=1,
    )
    prefetch_assemblers = _positive_environment_integer(
        FORMAL_CPU_PREFETCH_ASSEMBLERS_ENV,
        default=1,
    )
    prefetch_window = getattr(provider, "next_batch_window", None)
    prefetch_window_enabled = prefetch_depth > 1
    if prefetch_window_enabled:
        if not prefetch_enabled:
            raise RuntimeError(
                f"{FORMAL_CPU_PREFETCH_DEPTH_ENV}>1 requires CPU prefetch support"
            )
        if rank_balanced_enabled:
            raise RuntimeError(
                "multi-batch CPU prefetch does not support cost-balanced rank batching"
            )
        if not bool(getattr(provider, "supports_cpu_prefetch_window", False)) or not callable(
            prefetch_window
        ):
            raise RuntimeError(
                f"{FORMAL_CPU_PREFETCH_DEPTH_ENV}>1 requires a provider "
                "with next_batch_window support"
            )
    prefetch_executor: ThreadPoolExecutor | None = None
    prefetch_future: Future[object] | None = None
    prefetched_draws: deque[object] = deque()
    if prefetch_enabled:
        # Each rank owns one bounded CPU producer.  It prepares the next full
        # batch while the current one is in forward/backward, without changing
        # the draw ordinal or the checkpoint state.
        prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pista-prefetch")

    pilot_completed = False
    profile_first_completed_step = completed_steps + 1
    profile_samples: dict[str, list[float]] = {
        "step_wall_seconds": [],
        "prefetch_wait_seconds": [],
        "host_to_device_seconds": [],
        "groups": [],
        "candidates": [],
        "logical_padded_graph_cells": [],
    }
    runtime_event_timing = os.environ.get("FORMAL_RUNTIME_EVENT_TIMING") == "1"

    def submit_prefetch(start_index: int, sampler_epoch: int) -> Future[object]:
        if prefetch_executor is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("CPU prefetch executor is not initialized")
        if prefetch_window_enabled:
            return prefetch_executor.submit(
                prefetch_window,
                epoch=sampler_epoch,
                draw_index=start_index,
                draw_stride=context.world_size,
                count=prefetch_depth,
                assembler_workers=prefetch_assemblers,
                device="cpu",
                evaluation=False,
            )
        if rank_balanced_enabled:
            return prefetch_executor.submit(
                rank_balanced_batch,
                epoch=sampler_epoch,
                draw_index=start_index,
                rank=context.rank,
                world_size=context.world_size,
                per_rank_batch_groups=config.per_rank_batch_groups,
                device="cpu",
                evaluation=False,
            )
        return prefetch_executor.submit(
            provider.next_batch,
            epoch=sampler_epoch,
            draw_index=start_index,
            draw_stride=context.world_size,
            device="cpu",
            evaluation=False,
        )

    try:
        while completed_steps < plan.max_optimizer_updates:
            sampler_epoch = plan.sampler_epoch(config, completed_steps)
            step_wall_started = time.perf_counter()
            prefetch_wait_seconds = 0.0
            host_to_device_seconds = 0.0
            rollout_refresh_seconds = 0.0
            rollout_refresh_count = 0
            optimizer.zero_grad(set_to_none=True)
            micro_losses: list[Tensor] = []
            metric_totals: dict[str, Tensor] = {}
            group_count = 0
            candidate_count = 0
            max_local_candidate_count = 0
            max_local_padded_graph_cells = 0
            local_padded_graph_cells = 0
            for micro_step in range(config.gradient_accumulation_steps):
                if prefetch_enabled:
                    if prefetch_window_enabled:
                        if not prefetched_draws:
                            if prefetch_future is None:
                                prefetch_future = submit_prefetch(draw_index, sampler_epoch)
                            prefetch_wait_started = time.perf_counter()
                            window = prefetch_future.result()
                            prefetch_wait_seconds += time.perf_counter() - prefetch_wait_started
                            if not isinstance(window, (tuple, list)) or len(window) != prefetch_depth:
                                raise RuntimeError(
                                    "provider emitted an incomplete CPU prefetch window"
                                )
                            window_draw_index = draw_index
                            for window_draw in window:
                                window_next_draw_index = _draw_next_index(window_draw)
                                if window_next_draw_index <= window_draw_index:
                                    raise RuntimeError(
                                        "provider did not advance a prefetched window draw"
                                    )
                                prefetched_draws.append(window_draw)
                                window_draw_index = window_next_draw_index
                            prefetch_future = submit_prefetch(
                                window_draw_index, sampler_epoch
                            )
                        draw = prefetched_draws.popleft()
                        next_draw_index = _draw_next_index(draw)
                    else:
                        if prefetch_future is None:
                            prefetch_future = submit_prefetch(draw_index, sampler_epoch)
                        prefetch_wait_started = time.perf_counter()
                        draw = prefetch_future.result()
                        prefetch_wait_seconds += time.perf_counter() - prefetch_wait_started
                        next_draw_index = _draw_next_index(draw)
                        prefetch_future = submit_prefetch(
                            next_draw_index, sampler_epoch
                        )
                    host_to_device_started = time.perf_counter()
                    draw = _move_draw_to_device(draw, context.device)
                    host_to_device_seconds += time.perf_counter() - host_to_device_started
                else:
                    if rank_balanced_enabled:
                        draw = rank_balanced_batch(
                            epoch=sampler_epoch,
                            draw_index=draw_index,
                            rank=context.rank,
                            world_size=context.world_size,
                            per_rank_batch_groups=config.per_rank_batch_groups,
                            device=context.device,
                            evaluation=False,
                        )
                    else:
                        draw = provider.next_batch(
                            epoch=sampler_epoch,
                            draw_index=draw_index,
                            draw_stride=context.world_size,
                            device=context.device,
                            evaluation=False,
                        )
                    next_draw_index = _draw_next_index(draw)
                local_groups = _draw_group_count(draw)
                if local_groups != config.per_rank_batch_groups:
                    raise RuntimeError(
                        "provider emitted an incomplete batch: expected "
                        f"{config.per_rank_batch_groups} groups per rank, got {local_groups}"
                    )
                if next_draw_index <= draw_index:
                    raise RuntimeError("provider did not advance draw.next_draw_index")
                sync_context = (
                    wrapped_model.no_sync()
                    if isinstance(wrapped_model, DistributedDataParallel)
                    and micro_step + 1 < config.gradient_accumulation_steps
                    else nullcontext()
                )
                autocast_context = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if config.precision == "bf16-mixed"
                    else nullcontext()
                )
                stage_hook = getattr(_unwrap(wrapped_model), "set_formal_training_step", None)
                if stage_hook is not None and not callable(stage_hook):
                    raise TypeError("set_formal_training_step must be callable when provided")
                if callable(stage_hook):
                    stage_hook(completed_steps)
                rollout_hook = getattr(_unwrap(wrapped_model), "maybe_refresh_rollout_replay", None)
                if rollout_hook is not None and not callable(rollout_hook):
                    raise TypeError("maybe_refresh_rollout_replay must be callable when provided")
                if callable(rollout_hook):
                    # The hook is deliberately before the autograd forward:
                    # it samples bounded current-student trajectories under
                    # inference mode, then the normal loss differentiates on
                    # the detached replay states.  It never affects the
                    # provider's canonical data draw or corruption generator.
                    rollout_started = time.perf_counter() if runtime_event_timing else 0.0
                    rollout_refreshed = rollout_hook(
                        getattr(draw, "batch"), completed_steps=completed_steps
                    )
                    if not isinstance(rollout_refreshed, bool):
                        raise TypeError("maybe_refresh_rollout_replay must return a bool")
                    if rollout_refreshed and runtime_event_timing:
                        # Sampling results feed the immediately following
                        # differentiable replay, so synchronizing here exposes
                        # its true periodic cost without changing dependency
                        # order.  Non-refresh steps never pay this sync.
                        if context.device.type == "cuda":
                            torch.cuda.synchronize(context.device)
                        rollout_refresh_seconds += time.perf_counter() - rollout_started
                        rollout_refresh_count += 1
                with sync_context, autocast_context:
                    output = wrapped_model(getattr(draw, "batch"), generator=generator)
                    loss = getattr(output, "loss", None)
                    if not isinstance(loss, Tensor) or loss.numel() != 1:
                        raise RuntimeError("formal stage produced a non-scalar loss")
                    if context.device.type == "cuda":
                        torch._assert_async(torch.isfinite(loss), "formal stage produced a non-finite loss")
                    elif not bool(torch.isfinite(loss)):
                        raise RuntimeError("formal stage produced a non-finite scalar loss")
                    (loss / config.gradient_accumulation_steps).backward()
                micro_losses.append(loss.detach().float().reshape(()))
                output_metrics = getattr(output, "metrics", None)
                if output_metrics is None:
                    output_metrics = getattr(getattr(output, "objective", None), "metrics", {})
                for name, value in _scalar_metrics(output_metrics).items():
                    previous = metric_totals.get(name)
                    metric_totals[name] = value if previous is None else previous + value
                group_count += local_groups
                local_candidates = _draw_candidate_count(draw)
                candidate_count += local_candidates
                max_local_candidate_count = max(max_local_candidate_count, local_candidates)
                max_local_padded_graph_cells = max(
                    max_local_padded_graph_cells, _draw_padded_graph_cells(draw)
                )
                local_padded_graph_cells += _draw_padded_graph_cells(draw)
                draw_index = next_draw_index
            if config.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip_norm))
            optimizer.step()
            completed_steps += 1
            local_loss = torch.stack(micro_losses).mean()
            reduced_metrics = _reduced_metrics(
                {
                    name: value / config.gradient_accumulation_steps
                    for name, value in metric_totals.items()
                },
                context,
            )
            local_step_wall_seconds = time.perf_counter() - step_wall_started
            if runtime_event_timing:
                (
                    reduced_loss,
                    reduced_groups_value,
                    reduced_candidates_value,
                    reduced_prefetch_wait_seconds,
                    reduced_host_to_device_seconds,
                    reduced_rollout_refresh_seconds,
                    reduced_rollout_refresh_count,
                    reduced_logical_padded_graph_cells,
                    reduced_step_wall_seconds,
                ) = _reduced_scalars(
                    (
                        local_loss,
                        float(group_count),
                        float(candidate_count),
                        prefetch_wait_seconds,
                        host_to_device_seconds,
                        rollout_refresh_seconds,
                        float(rollout_refresh_count),
                        float(local_padded_graph_cells),
                        local_step_wall_seconds,
                    ),
                    context,
                )
                reduced_metrics.update(
                    {
                        "runtime_prefetch_wait_seconds": reduced_prefetch_wait_seconds,
                        "runtime_host_to_device_seconds": reduced_host_to_device_seconds,
                        "runtime_rollout_refresh_seconds": reduced_rollout_refresh_seconds,
                        "runtime_rollout_refresh_count": reduced_rollout_refresh_count,
                        "runtime_logical_padded_graph_cells": (
                            reduced_logical_padded_graph_cells
                        ),
                        "runtime_step_wall_seconds": reduced_step_wall_seconds,
                    }
                )
            else:
                (
                    reduced_loss,
                    reduced_groups_value,
                    reduced_candidates_value,
                    reduced_prefetch_wait_seconds,
                    reduced_host_to_device_seconds,
                    reduced_logical_padded_graph_cells,
                    reduced_step_wall_seconds,
                ) = _reduced_scalars(
                    (
                        local_loss,
                        float(group_count),
                        float(candidate_count),
                        prefetch_wait_seconds,
                        host_to_device_seconds,
                        float(local_padded_graph_cells),
                        local_step_wall_seconds,
                    ),
                    context,
                )
                reduced_metrics.update(
                    {
                        "runtime_prefetch_wait_seconds": reduced_prefetch_wait_seconds,
                        "runtime_host_to_device_seconds": reduced_host_to_device_seconds,
                        "runtime_logical_padded_graph_cells": (
                            reduced_logical_padded_graph_cells
                        ),
                        "runtime_step_wall_seconds": reduced_step_wall_seconds,
                    }
                )
            loss_share_hook = getattr(
                _unwrap(wrapped_model),
                "update_formal_loss_share_controller",
                None,
            )
            if loss_share_hook is not None and not callable(loss_share_hook):
                raise TypeError(
                    "update_formal_loss_share_controller must be callable when provided"
                )
            if callable(loss_share_hook):
                # All inputs have already been averaged across gradient
                # accumulation and all DDP ranks.  The hook therefore mutates
                # identical next-update controller state on every rank without
                # putting a collective in the autograd forward.
                controller_metrics = loss_share_hook(reduced_metrics)
                if not isinstance(controller_metrics, Mapping):
                    raise TypeError(
                        "update_formal_loss_share_controller must return a mapping"
                    )
                reduced_metrics.update(
                    {
                        name: float(value)
                        for name, value in controller_metrics.items()
                    }
                )
            if (
                runtime_event_timing
                and reduced_rollout_refresh_count > 0.0
                and context.rank == 0
            ):
                print(
                    "formal_runtime_event kind=rollout_refresh "
                    f"source_completed_steps={completed_steps - 1} "
                    f"seconds={reduced_rollout_refresh_seconds:.9f}",
                    flush=True,
                )
            reduced_groups = int(round(reduced_groups_value))
            reduced_candidates = int(round(reduced_candidates_value))
            if plan.is_epoch_boundary(completed_steps):
                # Prefetch may already contain the next old-epoch draw.  Drop
                # that bounded speculative work and restart every rank lane at
                # its canonical ordinal for the newly seeded sampler epoch.
                if prefetch_future is not None:
                    prefetch_future.cancel()
                    prefetch_future = None
                prefetched_draws.clear()
                draw_index = context.rank
            is_candidate = completed_steps in candidate_steps
            emits_rank_telemetry = (
                is_candidate
                or completed_steps % plan.heartbeat_interval_updates == 0
                or (
                    pilot_stop_after is not None
                    and completed_steps == pilot_stop_after
                )
            )
            if emits_rank_telemetry:
                cuda_allocated_bytes = 0.0
                cuda_peak_allocated_bytes = 0.0
                cuda_reserved_bytes = 0.0
                cuda_peak_reserved_bytes = 0.0
                if context.device.type == "cuda":
                    cuda_allocated_bytes = float(torch.cuda.memory_allocated(context.device))
                    cuda_peak_allocated_bytes = float(torch.cuda.max_memory_allocated(context.device))
                    cuda_reserved_bytes = float(torch.cuda.memory_reserved(context.device))
                    cuda_peak_reserved_bytes = float(torch.cuda.max_memory_reserved(context.device))
                for name, (lower, upper) in zip(
                    (
                        "runtime_rank_candidate_count",
                        "runtime_rank_padded_graph_cells",
                        "runtime_rank_step_wall_seconds",
                        "runtime_cuda_allocated_bytes",
                        "runtime_cuda_peak_allocated_bytes",
                        "runtime_cuda_reserved_bytes",
                        "runtime_cuda_peak_reserved_bytes",
                    ),
                    _reduced_scalar_bounds(
                        (
                            float(max_local_candidate_count),
                            float(max_local_padded_graph_cells),
                            local_step_wall_seconds,
                            cuda_allocated_bytes,
                            cuda_peak_allocated_bytes,
                            cuda_reserved_bytes,
                            cuda_peak_reserved_bytes,
                        ),
                        context,
                    ),
                    strict=True,
                ):
                    reduced_metrics[f"{name}_min"] = lower
                    reduced_metrics[f"{name}_max"] = upper
            if profile_steps:
                profile_samples["step_wall_seconds"].append(reduced_step_wall_seconds)
                profile_samples["prefetch_wait_seconds"].append(
                    reduced_prefetch_wait_seconds
                )
                profile_samples["host_to_device_seconds"].append(
                    reduced_host_to_device_seconds
                )
                profile_samples["groups"].append(float(reduced_groups))
                profile_samples["candidates"].append(float(reduced_candidates))
                profile_samples["logical_padded_graph_cells"].append(
                    reduced_logical_padded_graph_cells
                )
            last_step = FormalTrainingStep(
                step=completed_steps,
                next_draw_index=draw_index,
                group_count=reduced_groups,
                candidate_count=reduced_candidates,
                loss=reduced_loss,
                metrics=reduced_metrics,
                completed_epochs=plan.completed_epochs(completed_steps),
                sampler_epoch=(
                    sampler_epoch if plan.epoch_plan is not None else None
                ),
            )
            loss_sum += reduced_loss
            for name, raw_value in reduced_metrics.items():
                value = float(raw_value)
                if not math.isfinite(value):
                    raise RuntimeError(
                        f"formal reduced metric {name!r} is not finite"
                    )
                metric_sums[name] = metric_sums.get(name, 0.0) + value
                metric_update_counts[name] = metric_update_counts.get(name, 0) + 1
            if on_step is not None:
                on_step(last_step)
            if is_candidate:
                _save_candidate(
                    model=wrapped_model,
                    config=config,
                    context=context,
                    plan=plan,
                    completed_steps=completed_steps,
                    loss_sum=loss_sum,
                    metric_sums=metric_sums,
                    metric_update_counts=metric_update_counts,
                    last_step=last_step,
                    run_metadata=metadata,
                )
            checkpoint_requested = _checkpoint_requested(config, context)
            reached_pilot_boundary = (
                pilot_stop_after is not None and completed_steps == pilot_stop_after
            )
            if (
                is_candidate
                or completed_steps % plan.recovery_interval_updates == 0
                or checkpoint_requested
                or reached_pilot_boundary
            ):
                recovery_started = time.perf_counter() if runtime_event_timing else 0.0
                _save_recovery(
                    model=wrapped_model,
                    optimizer=optimizer,
                    generator=generator,
                    config=config,
                    context=context,
                    plan=plan,
                    completed_steps=completed_steps,
                    draw_index=draw_index,
                    loss_sum=loss_sum,
                    metric_sums=metric_sums,
                    metric_update_counts=metric_update_counts,
                    last_step=last_step,
                    model_recovery_contract=model_recovery_contract,
                )
                if runtime_event_timing and context.rank == 0:
                    recovery_write_seconds = time.perf_counter() - recovery_started
                    print(
                        "formal_runtime_event kind=recovery_write "
                        f"completed_steps={completed_steps} "
                        f"seconds={recovery_write_seconds:.9f}",
                        flush=True,
                    )
            if checkpoint_requested:
                _clear_checkpoint_request(config, context)
            if (
                is_candidate
                or completed_steps % plan.heartbeat_interval_updates == 0
                or reached_pilot_boundary
            ):
                _write_heartbeat(
                    config=config,
                    context=context,
                    plan=plan,
                    completed_steps=completed_steps,
                    loss_sum=loss_sum,
                    metric_sums=metric_sums,
                    metric_update_counts=metric_update_counts,
                    last_step=last_step,
                    recovered=recovered,
                    state="pilot_complete" if reached_pilot_boundary else "running",
                    profile=(
                        _formal_profile_payload(
                            profile_samples,
                            world_size=context.world_size,
                            first_completed_step=profile_first_completed_step,
                            last_completed_step=completed_steps,
                            boundary_metrics=last_step.metrics,
                        )
                        if reached_pilot_boundary and profile_steps
                        else None
                    ),
                )
            if reached_pilot_boundary:
                pilot_completed = True
                break
    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True, cancel_futures=True)
    if last_step is None:  # pragma: no cover - every budget validates positive
        raise AssertionError("formal training loop did not execute a step")
    if pilot_completed:
        return FormalTrainingResult(
            completed_steps=completed_steps,
            mean_loss=loss_sum / completed_steps,
            mean_metrics=_mean_metrics(metric_sums, metric_update_counts),
            mean_metric_update_counts=dict(metric_update_counts),
            last_step=last_step,
            candidate_checkpoints=_candidate_paths(config, plan),
            recovered=recovered,
            context=context,
            plan=plan,
        )
    _write_heartbeat(
        config=config,
        context=context,
        plan=plan,
        completed_steps=completed_steps,
        loss_sum=loss_sum,
        metric_sums=metric_sums,
        metric_update_counts=metric_update_counts,
        last_step=last_step,
        recovered=recovered,
        state="complete",
    )
    return FormalTrainingResult(
        completed_steps=completed_steps,
        mean_loss=loss_sum / completed_steps,
        mean_metrics=_mean_metrics(metric_sums, metric_update_counts),
        mean_metric_update_counts=dict(metric_update_counts),
        last_step=last_step,
        candidate_checkpoints=_candidate_paths(config, plan),
        recovered=recovered,
        context=context,
        plan=plan,
    )


__all__ = [
    "FORMAL_ALLOW_BATCH_RESUME_AMENDMENT_ENV",
    "FORMAL_ALLOW_EXPOSURE_RESUME_AMENDMENT_ENV",
    "FORMAL_ALLOW_EXPOSURE_SCHEDULE_RESUME_AMENDMENT_ENV",
    "FORMAL_CHECKPOINT_REQUEST_FILENAME",
    "FORMAL_STOP_AFTER_EPOCHS_ENV",
    "FORMAL_STOP_AFTER_COMPLETED_STEPS_ENV",
    "FORMAL_TRAINING_SCHEMA",
    "LEGACY_FORMAL_TRAINING_SCHEMA",
    "DistributedTrainingContext",
    "FormalBatchProvider",
    "FormalTrainingConfig",
    "FormalTrainingResult",
    "FormalTrainingStep",
    "ResolvedFormalTrainingPlan",
    "formal_candidate_checkpoint_path",
    "initialize_distributed_training",
    "resolve_formal_training_plan",
    "run_formal_training",
]
