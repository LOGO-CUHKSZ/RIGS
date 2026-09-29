"""Portable execution profiles for generation-side evaluation adapters.

The scientific evaluation contract owns the frozen manifest, model, seeds,
trajectory count, and scorer.  This module owns only auditable *execution*
choices such as trajectory batching.  An adapter may use its own namespaced
options, but it must not silently change a supplied profile.

No content hashes are calculated or validated by this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .io import EvaluationError, INTEGRITY_POLICY, load_document, now_utc, require_int, require_text


EXECUTION_PROFILE_SCHEMA = "rgg_inference_execution_profile.v1"
EXECUTION_BENCHMARK_SCHEMA = "rgg_inference_execution_benchmark.v1"
EXECUTION_SELECTION_SCHEMA = "rgg_inference_execution_selection.v1"
CUDA_LAYOUT_SCHEMA = "rgg_inference_cuda_layout.v1"
CPU_LAYOUT_SCHEMA = "rgg_inference_cpu_layout.v1"
DEFAULT_TRAJECTORY_BATCH = 2


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationError(f"{name} must be an object")
    return dict(value)


def _finite_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be a positive finite number")
    result = float(value)
    if result <= 0.0 or result == float("inf") or result != result:
        raise EvaluationError(f"{name} must be a positive finite number")
    return result


def _adapter_options(value: Any, name: str) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    source = _mapping(value, name)
    result: dict[str, dict[str, Any]] = {}
    for adapter, raw_options in source.items():
        adapter_name = require_text(adapter, f"{name} adapter")
        result[adapter_name] = _mapping(raw_options, f"{name}.{adapter_name}")
    return result


def _launcher_options(value: Any, name: str) -> dict[str, Any]:
    """Keep portable launcher settings separate from adapter internals.

    These settings describe only orchestration (for example, sharding and arm
    concurrency).  A generator must not reinterpret them as model settings.
    """

    if value is None:
        return {}
    return _mapping(value, name)


@dataclass(frozen=True, slots=True)
class InferenceExecutionProfile:
    """A model-agnostic execution contract with adapter-scoped options."""

    profile_id: str
    trajectory_batch: int
    adapter_options: Mapping[str, Mapping[str, Any]]
    launcher_options: Mapping[str, Any] | None = None
    source_path: str | None = None

    def __post_init__(self) -> None:
        require_text(self.profile_id, "execution profile id")
        require_int(self.trajectory_batch, "execution profile trajectory_batch", minimum=1)
        normalized = _adapter_options(self.adapter_options, "execution profile adapter_options")
        object.__setattr__(self, "adapter_options", normalized)
        object.__setattr__(
            self,
            "launcher_options",
            _launcher_options(self.launcher_options, "execution profile launcher_options"),
        )

    def options_for(self, adapter: str) -> dict[str, Any]:
        return dict(self.adapter_options.get(require_text(adapter, "adapter"), {}))

    def launcher(self) -> dict[str, Any]:
        """Return immutable-profile launcher choices as a mutable copy."""

        return dict(self.launcher_options or {})

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": EXECUTION_PROFILE_SCHEMA,
            "profile_id": self.profile_id,
            "trajectory_batch": self.trajectory_batch,
            "adapter_options": {
                name: dict(options) for name, options in sorted(self.adapter_options.items())
            },
            "launcher_options": self.launcher(),
        }
        if self.source_path is not None:
            result["source_path"] = self.source_path
        return result


def profile_from_mapping(value: Mapping[str, Any], *, source_path: str | None = None) -> InferenceExecutionProfile:
    """Validate one portable execution-profile document or embedded profile."""

    document = _mapping(value, "execution profile")
    schema = require_text(document.get("schema_version"), "execution profile schema_version")
    if schema != EXECUTION_PROFILE_SCHEMA:
        raise EvaluationError(f"unsupported execution profile schema {schema!r}")
    profile_id = require_text(document.get("profile_id"), "execution profile profile_id")
    trajectory_batch = require_int(document.get("trajectory_batch"), "execution profile trajectory_batch", minimum=1)
    return InferenceExecutionProfile(
        profile_id=profile_id,
        trajectory_batch=trajectory_batch,
        adapter_options=_adapter_options(document.get("adapter_options"), "execution profile adapter_options"),
        launcher_options=_launcher_options(document.get("launcher_options"), "execution profile launcher_options"),
        source_path=source_path,
    )


def load_inference_execution_profile(path: str | Path) -> InferenceExecutionProfile:
    """Load a profile once without mutating the source document."""

    source = Path(path).expanduser().resolve(strict=True)
    return profile_from_mapping(load_document(source), source_path=str(source))


def resolve_inference_execution_profile(
    *,
    profile_path: str | Path | None,
    trajectory_batch: int | None,
    default_trajectory_batch: int = DEFAULT_TRAJECTORY_BATCH,
) -> InferenceExecutionProfile:
    """Resolve explicit CLI choices against an optional portable profile.

    A conflicting CLI trajectory batch is rejected instead of quietly producing
    a run whose logged profile does not describe its actual execution.
    """

    default_batch = require_int(
        default_trajectory_batch,
        "default trajectory_batch",
        minimum=1,
    )
    if profile_path is None:
        chosen = default_batch if trajectory_batch is None else require_int(
            trajectory_batch,
            "trajectory_batch",
            minimum=1,
        )
        return InferenceExecutionProfile(
            profile_id="default" if trajectory_batch is None else f"cli_trajectory_batch_{chosen}",
            trajectory_batch=chosen,
            adapter_options={},
            launcher_options={},
        )
    profile = load_inference_execution_profile(profile_path)
    if trajectory_batch is not None:
        explicit = require_int(trajectory_batch, "trajectory_batch", minimum=1)
        if explicit != profile.trajectory_batch:
            raise EvaluationError(
                "trajectory_batch conflicts with execution profile; "
                "edit the profile or omit the CLI override"
            )
    return profile


def _validated_benchmark_record(value: Any, index: int) -> dict[str, Any] | None:
    record = _mapping(value, f"benchmark records[{index}]")
    status = require_text(record.get("status"), f"benchmark records[{index}].status")
    if status not in {"passed", "failed", "skipped"}:
        raise EvaluationError(f"benchmark records[{index}].status is unsupported")
    profile = profile_from_mapping(
        _mapping(record.get("profile"), f"benchmark records[{index}].profile")
    )
    validation = _mapping(record.get("validation"), f"benchmark records[{index}].validation")
    if not validation or any(value is not True for value in validation.values()):
        if status == "passed":
            raise EvaluationError(
                f"benchmark records[{index}] cannot pass without all validation checks true"
            )
        return None
    if status != "passed":
        return None
    wall_seconds = _finite_positive_float(
        record.get("wall_seconds"),
        f"benchmark records[{index}].wall_seconds",
    )
    trajectory_count = require_int(
        record.get("trajectory_count"),
        f"benchmark records[{index}].trajectory_count",
        minimum=1,
    )
    return {
        "profile": profile.to_dict(),
        "wall_seconds": wall_seconds,
        "trajectory_count": trajectory_count,
        "seconds_per_trajectory": wall_seconds / trajectory_count,
        "validation": validation,
    }


def select_inference_execution_profile(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    """Choose the fastest fully validated execution profile.

    The score is wall-clock seconds per generated trajectory, so future
    adapters may benchmark different smoke-cohort sizes without changing the
    selection rule.  Power is diagnostic evidence, never a surrogate objective.
    """

    document = _mapping(benchmark, "execution benchmark")
    schema = require_text(document.get("schema_version"), "execution benchmark schema_version")
    if schema != EXECUTION_BENCHMARK_SCHEMA:
        raise EvaluationError(f"unsupported execution benchmark schema {schema!r}")
    records = document.get("records")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise EvaluationError("execution benchmark records must be a non-empty list")
    eligible = [
        result
        for index, record in enumerate(records)
        if (result := _validated_benchmark_record(record, index)) is not None
    ]
    if not eligible:
        raise EvaluationError("execution benchmark has no fully validated passed record")
    ranked = sorted(
        eligible,
        key=lambda item: (item["seconds_per_trajectory"], item["profile"]["profile_id"]),
    )
    return {
        "schema_version": EXECUTION_SELECTION_SCHEMA,
        "status": "selected",
        "selected_at": now_utc(),
        "selection_metric": "wall_seconds_per_trajectory",
        "selected": ranked[0],
        "eligible_records": ranked,
        "integrity_policy": INTEGRITY_POLICY,
    }


def validate_cuda_layout(expected_visible_devices: int) -> dict[str, Any]:
    """Validate the CUDA devices exposed to one inference process.

    Schedulers may present an allocated GPU subset differently from host-level
    tools such as ``nvidia-smi``.  Generation therefore gates on PyTorch's
    process-visible CUDA layout, which is the layout the adapter will actually
    use.  This function is deliberately read-only: it allocates no tensors and
    performs no content-integrity work.
    """

    expected = require_int(
        expected_visible_devices,
        "expected visible CUDA device count",
        minimum=1,
    )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise EvaluationError("CUDA layout validation requires torch") from error
    if not torch.cuda.is_available():
        raise EvaluationError("torch.cuda.is_available() is false")
    observed = int(torch.cuda.device_count())
    if observed != expected:
        raise EvaluationError(
            "process-visible CUDA device count "
            f"{observed} differs from expected {expected}"
        )
    names = [str(torch.cuda.get_device_name(index)) for index in range(observed)]
    return {
        "schema_version": CUDA_LAYOUT_SCHEMA,
        "status": "passed",
        "expected_visible_device_count": expected,
        "visible_device_count": observed,
        "device_names": names,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "torch_cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "integrity_policy": INTEGRITY_POLICY,
    }


def validate_cpu_layout(minimum_visible_cpus: int) -> dict[str, Any]:
    """Validate CPUs available to this process independently of thread limits.

    Shell helpers such as ``nproc`` may honour OpenMP environment variables and
    therefore report a configured thread cap instead of the CPUs assigned by a
    scheduler.  Linux CPU affinity is the process-level allocation that the
    launcher can actually use, so it remains valid after setting OMP/MKL/BLAS
    thread limits for a GPU inference process.
    """

    minimum = require_int(
        minimum_visible_cpus,
        "minimum visible CPU count",
        minimum=1,
    )
    if not hasattr(os, "sched_getaffinity"):
        raise EvaluationError("CPU layout validation requires os.sched_getaffinity on Linux")
    visible = len(os.sched_getaffinity(0))
    if visible < minimum:
        raise EvaluationError(
            "process-visible CPU count "
            f"{visible} is below required minimum {minimum}"
        )
    return {
        "schema_version": CPU_LAYOUT_SCHEMA,
        "status": "passed",
        "minimum_visible_cpu_count": minimum,
        "visible_cpu_count": visible,
        "integrity_policy": INTEGRITY_POLICY,
    }


__all__ = [
    "DEFAULT_TRAJECTORY_BATCH",
    "CUDA_LAYOUT_SCHEMA",
    "CPU_LAYOUT_SCHEMA",
    "EXECUTION_BENCHMARK_SCHEMA",
    "EXECUTION_PROFILE_SCHEMA",
    "EXECUTION_SELECTION_SCHEMA",
    "InferenceExecutionProfile",
    "load_inference_execution_profile",
    "profile_from_mapping",
    "resolve_inference_execution_profile",
    "select_inference_execution_profile",
    "validate_cpu_layout",
    "validate_cuda_layout",
]
