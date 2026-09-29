"""Bounded, framework-free optimization for Stage A and Stage B assemblies."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction

import torch
from torch import Tensor, nn

from retro_guided_generation.data.runtime.pista import PistaStageABatchProvider
from retro_guided_generation.training.epochs import (
    EpochBudget,
    EpochTrainingPlan,
    EpochValue,
    parse_epoch_value,
)


@dataclass(frozen=True, slots=True)
class GuidedTrainingConfig:
    """Explicit finite training-loop controls.

    This is intentionally a bounded loop: resource allocation, process
    lifetime, checkpoint retention, and formal experiment tracking remain the
    caller's responsibility.
    """

    learning_rate: float
    max_epochs: EpochValue | None = None
    max_steps: int | None = None
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = 1.0
    seed: int = 42
    first_epoch: int = 0
    epoch: int | None = None

    def __post_init__(self) -> None:
        epoch_budget = self.max_epochs is not None
        step_budget = self.max_steps is not None
        if epoch_budget == step_budget:
            raise ValueError(
                "configure max_epochs for new runs or legacy max_steps, but not both"
            )
        if epoch_budget:
            if self.epoch is not None:
                raise ValueError("new epoch configs use first_epoch, not legacy epoch")
            object.__setattr__(
                self, "max_epochs", parse_epoch_value(self.max_epochs, "max_epochs")
            )
        elif (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be a positive integer")
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
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            isinstance(self.first_epoch, bool)
            or not isinstance(self.first_epoch, int)
            or self.first_epoch < 0
        ):
            raise ValueError("first_epoch must be a non-negative integer")
        if step_budget:
            sampler_epoch = self.first_epoch if self.epoch is None else self.epoch
            if (
                isinstance(sampler_epoch, bool)
                or not isinstance(sampler_epoch, int)
                or sampler_epoch < 0
            ):
                raise ValueError("legacy epoch must be a non-negative integer")
            object.__setattr__(self, "epoch", sampler_epoch)

    @property
    def budget_mode(self) -> str:
        return "epochs" if self.max_epochs is not None else "legacy_steps"


@dataclass(frozen=True, slots=True)
class GuidedTrainingStep:
    """Compact scalar report for one completed optimizer update."""

    step: int
    draw_index: int
    next_draw_index: int
    group_count: int
    candidate_count: int
    loss: float
    metrics: Mapping[str, float]
    completed_epochs: Fraction | None = None
    sampler_epoch: int | None = None


@dataclass(frozen=True, slots=True)
class GuidedTrainingResult:
    """Result summary without retaining an unbounded per-step history."""

    completed_steps: int
    next_draw_index: int
    mean_loss: float
    last_step: GuidedTrainingStep
    completed_epochs: Fraction | None = None


def _scalar_metrics(values: Mapping[str, Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(value, Tensor) or value.numel() != 1:
            continue
        result[name] = float(value.detach().float().cpu())
    return result


def run_guided_training(
    *,
    model: nn.Module,
    provider: PistaStageABatchProvider,
    config: GuidedTrainingConfig,
    device: torch.device | str,
    start_draw_index: int = 0,
    on_step: Callable[[GuidedTrainingStep], None] | None = None,
) -> GuidedTrainingResult:
    """Optimize a Stage A/B model for exactly the configured epoch budget."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch module")
    if not isinstance(provider, PistaStageABatchProvider):
        raise TypeError("provider must be PistaStageABatchProvider")
    if not isinstance(config, GuidedTrainingConfig):
        raise TypeError("config must be GuidedTrainingConfig")
    if isinstance(start_draw_index, bool) or not isinstance(start_draw_index, int) or start_draw_index < 0:
        raise ValueError("start_draw_index must be a non-negative integer")
    target_device = torch.device(device)
    model.to(target_device)
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("guided training model exposes no trainable parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    generator = torch.Generator(device=target_device).manual_seed(config.seed)
    draw_index = start_draw_index
    epoch_plan: EpochTrainingPlan | None = None
    if config.budget_mode == "epochs":
        if config.max_epochs is None:  # pragma: no cover - config validates this
            raise AssertionError("epoch budget is unavailable")
        epoch_plan = EpochTrainingPlan.resolve(
            budget=EpochBudget(
                max_epochs=config.max_epochs,
                candidate_epochs=(config.max_epochs,),
                first_epoch=config.first_epoch,
            ),
            epoch_size_groups=provider.epoch_size_groups,
            effective_global_batch_groups=provider.max_groups_per_batch,
            recovery_interval_epochs=1,
            heartbeat_interval_epochs=1,
        )
        max_optimizer_updates = epoch_plan.max_optimizer_updates
        epoch_hook = getattr(model, "configure_formal_epoch_plan", None)
        if epoch_hook is not None and not callable(epoch_hook):
            raise TypeError("configure_formal_epoch_plan must be callable when provided")
        if callable(epoch_hook):
            epoch_hook(
                optimizer_updates_per_epoch=epoch_plan.optimizer_updates_per_epoch
            )
    else:
        if config.max_steps is None or config.epoch is None:  # pragma: no cover
            raise AssertionError("legacy step budget is unavailable")
        max_optimizer_updates = config.max_steps
    total_loss = 0.0
    last_step: GuidedTrainingStep | None = None
    for step in range(max_optimizer_updates):
        sampler_epoch = (
            epoch_plan.sampler_epoch(step) if epoch_plan is not None else config.epoch
        )
        draw = provider.next_batch(
            epoch=sampler_epoch,
            draw_index=draw_index,
            device=target_device,
            evaluation=False,
        )
        optimizer.zero_grad(set_to_none=True)
        stage_hook = getattr(model, "set_formal_training_step", None)
        if stage_hook is not None and not callable(stage_hook):
            raise TypeError("set_formal_training_step must be callable when provided")
        if callable(stage_hook):
            stage_hook(step)
        output = model(draw.batch, generator=generator)
        loss = getattr(output, "loss", None)
        if not isinstance(loss, Tensor) or loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise RuntimeError("guided stage produced a non-finite scalar loss")
        loss.backward()
        if config.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, float(config.gradient_clip_norm))
        optimizer.step()
        metrics = _scalar_metrics(getattr(getattr(output, "objective", None), "metrics", {}))
        scalar_loss = float(loss.detach().float().cpu())
        total_loss += scalar_loss
        completed_updates = step + 1
        completed_epochs = (
            epoch_plan.completed_epochs(completed_updates)
            if epoch_plan is not None
            else None
        )
        next_draw_index = draw.next_draw_index
        if epoch_plan is not None and epoch_plan.is_epoch_boundary(completed_updates):
            next_draw_index = 0
        step_report = GuidedTrainingStep(
            step=step,
            draw_index=draw_index,
            next_draw_index=next_draw_index,
            group_count=len(draw.conditions),
            candidate_count=int(draw.batch.graphs.p_X.shape[0]),
            loss=scalar_loss,
            metrics=metrics,
            completed_epochs=completed_epochs,
            sampler_epoch=sampler_epoch if epoch_plan is not None else None,
        )
        if on_step is not None:
            on_step(step_report)
        last_step = step_report
        draw_index = next_draw_index
    if last_step is None:  # pragma: no cover - config prevents this
        raise AssertionError("bounded training loop did not execute a step")
    return GuidedTrainingResult(
        completed_steps=max_optimizer_updates,
        next_draw_index=draw_index,
        mean_loss=total_loss / max_optimizer_updates,
        last_step=last_step,
        completed_epochs=(
            epoch_plan.completed_epochs(max_optimizer_updates)
            if epoch_plan is not None
            else None
        ),
    )


__all__ = [
    "GuidedTrainingConfig",
    "GuidedTrainingResult",
    "GuidedTrainingStep",
    "run_guided_training",
]
