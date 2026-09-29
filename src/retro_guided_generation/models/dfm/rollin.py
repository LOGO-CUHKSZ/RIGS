"""Dataset-neutral one-step positive recovery for categorical GraphDiT.

The recovery state is built from a monotone predefined-Diffusion bridge.  A
student predicts an endpoint at time ``t`` without gradient, one categorical
transition reaches ``s=t+1/K``, then the caller re-forwards with gradients and
recovers the known positive endpoint.  No dataset identity or cache metadata is
part of this primitive.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor


_TIME_POLICIES = frozenset({"uniform_midpoint32_v1", "low_t_mixture50_midpoint32_v1"})


def _epoch_fraction(value: object, name: str) -> Fraction:
    """Normalize a non-negative public epoch schedule without float drift."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative epoch number")
    try:
        if isinstance(value, Fraction):
            result = value
        elif isinstance(value, int):
            result = Fraction(value, 1)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError
            result = Fraction(str(value))
        elif isinstance(value, str) and value.strip():
            result = Fraction(value.strip())
        else:
            raise ValueError
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"{name} must be a non-negative epoch number or exact fraction"
        ) from error
    if result < 0:
        raise ValueError(f"{name} must be a non-negative epoch number")
    return result


def _epoch_json(value: Fraction) -> int | str:
    return value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True, slots=True)
class PositiveRecoveryConfig:
    """Explicit contract for clean-control or REC1 student-roll-in recovery."""

    mode: str = "disabled"
    weight: float = 0.0
    warmup_steps: int = 0
    ramp_steps: int = 0
    warmup_epochs: Fraction | int | float | str | None = None
    ramp_epochs: Fraction | int | float | str | None = None
    branch: str = "structure"
    sampling_steps: int = 50
    time_policy: str = "uniform_midpoint32_v1"
    time_policy_seed: int = 97_031
    time_min: float = 0.1
    time_max: float = 0.9
    time_bins: int = 32
    low_fraction: float = 0.5
    low_time_min: float = 0.02
    low_time_max: float = 0.25
    high_time_min: float = 0.25
    high_time_max: float = 0.9
    max_groups: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in {"disabled", "clean_control", "student_rollin"}:
            raise ValueError("positive recovery mode must be disabled, clean_control, or student_rollin")
        if not isinstance(self.branch, str) or not self.branch.strip():
            raise ValueError("positive recovery branch must be a non-empty string")
        for name, value in (
            ("warmup_steps", self.warmup_steps),
            ("ramp_steps", self.ramp_steps),
            ("sampling_steps", self.sampling_steps),
            ("time_bins", self.time_bins),
        ):
            minimum = 2 if name == "sampling_steps" else 1 if name == "time_bins" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        epoch_schedule_requested = (
            self.warmup_epochs is not None or self.ramp_epochs is not None
        )
        if epoch_schedule_requested:
            if self.warmup_epochs is None or self.ramp_epochs is None:
                raise ValueError(
                    "positive recovery epoch scheduling requires both "
                    "warmup_epochs and ramp_epochs"
                )
            if self.warmup_steps != 0 or self.ramp_steps != 0:
                raise ValueError(
                    "positive recovery cannot mix epoch and step schedules"
                )
            object.__setattr__(
                self,
                "warmup_epochs",
                _epoch_fraction(self.warmup_epochs, "warmup_epochs"),
            )
            object.__setattr__(
                self,
                "ramp_epochs",
                _epoch_fraction(self.ramp_epochs, "ramp_epochs"),
            )
        if isinstance(self.time_policy_seed, bool) or not isinstance(self.time_policy_seed, int):
            raise ValueError("time_policy_seed must be an integer")
        if self.max_groups is not None and (
            isinstance(self.max_groups, bool) or not isinstance(self.max_groups, int) or self.max_groups < 1
        ):
            raise ValueError("max_groups must be a positive integer or None")
        for name, value, positive in (
            ("weight", self.weight, False),
            ("time_min", self.time_min, True),
            ("time_max", self.time_max, True),
            ("low_fraction", self.low_fraction, True),
            ("low_time_min", self.low_time_min, True),
            ("low_time_max", self.low_time_max, True),
            ("high_time_min", self.high_time_min, True),
            ("high_time_max", self.high_time_max, True),
        ):
            number = float(value)
            if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not 0.02 <= float(self.time_min) < float(self.time_max) <= 0.98:
            raise ValueError("recovery time_min/time_max must satisfy 0.02 <= min < max <= 0.98")
        if not 0.0 < float(self.low_fraction) < 1.0:
            raise ValueError("low_fraction must lie strictly between zero and one")
        policy = str(self.time_policy).strip().lower()
        if policy not in _TIME_POLICIES:
            raise ValueError(f"unsupported positive recovery time_policy {self.time_policy!r}")
        if mode != "disabled" and float(self.weight) <= 0.0:
            raise ValueError("enabled positive recovery requires a positive weight")
        if mode != "disabled" and float(self.time_max) + 1.0 / int(self.sampling_steps) > 0.98:
            raise ValueError("recovery requires time_max + 1 / sampling_steps <= 0.98")
        if policy == "low_t_mixture50_midpoint32_v1":
            fixed = (
                self.time_bins == 32
                and math.isclose(float(self.time_min), 0.1)
                and math.isclose(float(self.time_max), 0.9)
                and math.isclose(float(self.low_fraction), 0.5)
                and math.isclose(float(self.low_time_min), 0.02)
                and math.isclose(float(self.low_time_max), 0.25)
                and math.isclose(float(self.high_time_min), 0.25)
                and math.isclose(float(self.high_time_max), 0.9)
            )
            if not fixed:
                raise ValueError("low_t_mixture50_midpoint32_v1 requires the fixed LT50 schedule")

    @property
    def normalized_mode(self) -> str:
        return str(self.mode).strip().lower()

    @property
    def enabled(self) -> bool:
        return self.normalized_mode != "disabled"

    @property
    def schedule_mode(self) -> str:
        return "epochs" if self.warmup_epochs is not None else "legacy_steps"

    def resolved_update_schedule(
        self,
        *,
        optimizer_updates_per_epoch: int | None = None,
    ) -> tuple[int, int]:
        """Resolve the public schedule to the runner's internal update grid."""

        if self.schedule_mode == "legacy_steps":
            return self.warmup_steps, self.ramp_steps
        if (
            isinstance(optimizer_updates_per_epoch, bool)
            or not isinstance(optimizer_updates_per_epoch, int)
            or optimizer_updates_per_epoch < 1
        ):
            raise RuntimeError(
                "epoch-based positive recovery requires optimizer_updates_per_epoch"
            )
        if not isinstance(self.warmup_epochs, Fraction) or not isinstance(
            self.ramp_epochs, Fraction
        ):  # pragma: no cover - post-init normalizes the epoch schedule
            raise AssertionError("positive recovery epoch schedule is not normalized")
        return (
            math.ceil(self.warmup_epochs * optimizer_updates_per_epoch),
            math.ceil(self.ramp_epochs * optimizer_updates_per_epoch),
        )

    def resolved_schedule_dict(
        self,
        *,
        optimizer_updates_per_epoch: int | None = None,
    ) -> dict[str, object]:
        warmup_updates, ramp_updates = self.resolved_update_schedule(
            optimizer_updates_per_epoch=optimizer_updates_per_epoch
        )
        value: dict[str, object] = {
            "schedule_mode": self.schedule_mode,
            "warmup_optimizer_updates": warmup_updates,
            "ramp_optimizer_updates": ramp_updates,
            "maximum_weight": float(self.weight),
        }
        if self.schedule_mode == "epochs":
            if not isinstance(self.warmup_epochs, Fraction) or not isinstance(
                self.ramp_epochs, Fraction
            ):  # pragma: no cover - post-init normalizes the epoch schedule
                raise AssertionError("positive recovery epoch schedule is not normalized")
            value.update(
                {
                    "warmup_epochs": _epoch_json(self.warmup_epochs),
                    "ramp_epochs": _epoch_json(self.ramp_epochs),
                    "full_weight_epoch": _epoch_json(
                        self.warmup_epochs + self.ramp_epochs
                    ),
                }
            )
        return value

    def effective_weight(
        self,
        completed_steps: int,
        *,
        optimizer_updates_per_epoch: int | None = None,
    ) -> float:
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or completed_steps < 0:
            raise ValueError("completed_steps must be a non-negative integer")
        if not self.enabled:
            return 0.0
        warmup_steps, ramp_steps = self.resolved_update_schedule(
            optimizer_updates_per_epoch=optimizer_updates_per_epoch
        )
        if completed_steps < warmup_steps:
            return 0.0
        scale = 1.0 if ramp_steps == 0 else min(
            1.0, (completed_steps - warmup_steps + 1) / ramp_steps
        )
        return float(self.weight) * scale


@dataclass(frozen=True, slots=True)
class PositiveRecoveryTime:
    """One common t->s bridge selected from the configured deterministic policy."""

    t: float
    s: float
    bucket: int
    component: str


@dataclass(frozen=True, slots=True)
class StudentRollInState:
    """Detached one-step sampled state consumed by the recovery re-forward."""

    t_X: Tensor
    t_E: Tensor


def positive_recovery_time(config: PositiveRecoveryConfig, completed_steps: int) -> PositiveRecoveryTime:
    """Return the common recovery bridge for one optimizer step."""

    if not isinstance(config, PositiveRecoveryConfig):
        raise TypeError("config must be PositiveRecoveryConfig")
    if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or completed_steps < 0:
        raise ValueError("completed_steps must be a non-negative integer")
    seed = int(config.time_policy_seed) + completed_steps
    bucket = (seed * 7_919 + 1_543) % int(config.time_bins)
    policy = str(config.time_policy).strip().lower()
    if policy == "uniform_midpoint32_v1":
        fraction = (bucket + 0.5) / int(config.time_bins)
        t = float(config.time_min) + fraction * (float(config.time_max) - float(config.time_min))
        component = "uniform"
    else:
        component_bins = int(config.time_bins) // 2
        if bucket < component_bins:
            lower, upper, component_bucket, component = (
                float(config.low_time_min),
                float(config.low_time_max),
                bucket,
                "low",
            )
        else:
            lower, upper, component_bucket, component = (
                float(config.high_time_min),
                float(config.high_time_max),
                bucket - component_bins,
                "high",
            )
        t = lower + ((component_bucket + 0.5) / component_bins) * (upper - lower)
    s = t + 1.0 / int(config.sampling_steps)
    if not 0.02 <= t < s <= 0.98:
        raise RuntimeError("positive recovery time policy emitted an invalid non-terminal bridge")
    return PositiveRecoveryTime(t=t, s=s, bucket=bucket, component=component)


def _require_generator(generator: torch.Generator, device: torch.device) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    if torch.device(generator.device).type != device.type:
        raise ValueError("generator must be device-local to the roll-in tensors")


def _sample_categorical(probability: Tensor, *, generator: torch.Generator) -> Tensor:
    if probability.ndim < 2 or probability.shape[-1] < 2:
        raise ValueError("categorical probability tensor must have a class dimension >= 2")
    flat = probability.float().reshape(-1, probability.shape[-1]).clamp_min(0.0)
    normalized = flat / flat.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(flat.dtype).tiny)
    return torch.multinomial(normalized, 1, generator=generator).reshape(probability.shape[:-1])


def _sample_endpoint(logits: Tensor, *, node_mask: Tensor, generator: torch.Generator, edges: bool) -> Tensor:
    probability = torch.softmax(logits.detach().float(), dim=-1)
    if edges:
        if probability.ndim != 4 or probability.shape[1] != probability.shape[2]:
            raise ValueError("edge logits must have shape [B,N,N,K]")
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        probability = probability.clone()
        probability[~valid] = 1.0 / probability.shape[-1]
        diagonal = torch.arange(probability.shape[1], device=probability.device)
        probability[:, diagonal, diagonal] = 1.0 / probability.shape[-1]
        indices = _sample_categorical(probability, generator=generator)
        upper = torch.triu(indices, diagonal=1)
        return upper + upper.transpose(1, 2)
    if probability.ndim != 3 or probability.shape[:2] != node_mask.shape:
        raise ValueError("node logits must have shape [B,N,K]")
    probability = probability.clone()
    probability[~node_mask] = 1.0 / probability.shape[-1]
    return _sample_categorical(probability, generator=generator)


def _sample_categorical_per_sample_generators(
    probability: Tensor,
    *,
    generators: Sequence[torch.Generator],
) -> Tensor:
    """Sample each batch row from its own stream without global RNG state.

    This preserves the reference sampler's per-trajectory generator contract,
    while allowing the deterministic probability construction around it to
    remain batched.  Each row deliberately calls the same categorical helper
    as the reference path, so the stream consumption order within a row is
    unchanged.
    """

    if probability.ndim < 2:
        raise ValueError("categorical probability tensor must have a batch and class dimension")
    if len(generators) != probability.shape[0]:
        raise ValueError("per-sample generators must match categorical batch size")
    samples: list[Tensor] = []
    for index, generator in enumerate(generators):
        _require_generator(generator, probability.device)
        samples.append(_sample_categorical(probability[index : index + 1], generator=generator))
    return torch.cat(samples, dim=0)


_STATELESS_COUNTER_MASK = (1 << 32) - 1
_STATELESS_COUNTER_MIX = 0x45D9F3B


def _require_stateless_seeds(
    seeds: Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    """Validate one non-negative frozen seed per trajectory row."""

    if not isinstance(seeds, Tensor):
        raise TypeError("stateless sampling seeds must be a tensor")
    if seeds.ndim != 1 or seeds.shape[0] != batch_size:
        raise ValueError("stateless sampling seeds must have shape [B]")
    if seeds.device != device:
        raise ValueError("stateless sampling seeds must share the roll-in device")
    if seeds.dtype == torch.bool or seeds.is_floating_point() or seeds.is_complex():
        raise TypeError("stateless sampling seeds must have an integer dtype")
    result = seeds.to(dtype=torch.int64)
    if bool(torch.any(result < 0)):
        raise ValueError("stateless sampling seeds must be non-negative")
    return result


def _stateless_counter_uniform(
    seeds: Tensor,
    *,
    event_count: int,
    step_index: int,
    stream_id: int,
) -> Tensor:
    """Return reproducible row-local uniforms without mutable RNG state.

    The counter is keyed by the frozen trajectory seed, diffusion step,
    categorical sampling slot, and the flattened event position.  This makes
    results independent of grouping, physical batch size, and sharding while
    retaining an explicit random stream for every trajectory.
    """

    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1:
        raise ValueError("stateless categorical event_count must be a positive integer")
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("stateless categorical step_index must be a non-negative integer")
    if isinstance(stream_id, bool) or not isinstance(stream_id, int) or stream_id < 0:
        raise ValueError("stateless categorical stream_id must be a non-negative integer")
    normalized = _require_stateless_seeds(
        seeds,
        batch_size=seeds.shape[0],
        device=seeds.device,
    )
    offsets = torch.arange(event_count, device=normalized.device, dtype=torch.int64)
    # All arithmetic is explicitly reduced modulo 2**32.  The two finalizer
    # multiplications stay within signed int64 before that reduction.
    step_key = ((step_index + 1) * 0x9E3779B1) & _STATELESS_COUNTER_MASK
    stream_key = ((stream_id + 1) * 0x85EBCA77) & _STATELESS_COUNTER_MASK
    state = (normalized[:, None] ^ offsets[None, :] ^ step_key ^ stream_key) & _STATELESS_COUNTER_MASK
    state = state ^ (state >> 16)
    state = (state * _STATELESS_COUNTER_MIX) & _STATELESS_COUNTER_MASK
    state = state ^ (state >> 16)
    state = (state * _STATELESS_COUNTER_MIX) & _STATELESS_COUNTER_MASK
    state = state ^ (state >> 16)
    return (state.to(dtype=torch.float32) + 0.5) * (1.0 / float(1 << 32))


def _sample_categorical_stateless(
    probability: Tensor,
    *,
    seeds: Tensor,
    step_index: int,
    stream_id: int,
) -> Tensor:
    """Vectorized inverse-CDF categorical draws from row-local counters."""

    if probability.ndim < 2 or probability.shape[-1] < 2:
        raise ValueError("categorical probability tensor must have a class dimension >= 2")
    batch_size = probability.shape[0]
    normalized_seeds = _require_stateless_seeds(
        seeds,
        batch_size=batch_size,
        device=probability.device,
    )
    classes = probability.shape[-1]
    flat = probability.float().reshape(batch_size, -1, classes).clamp_min(0.0)
    normalized = flat / flat.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(flat.dtype).tiny)
    uniform = _stateless_counter_uniform(
        normalized_seeds,
        event_count=flat.shape[1],
        step_index=step_index,
        stream_id=stream_id,
    )
    cumulative = normalized.cumsum(dim=-1)
    indices = (uniform.unsqueeze(-1) > cumulative).sum(dim=-1).clamp_max(classes - 1)
    return indices.to(dtype=torch.long).reshape(probability.shape[:-1])


def _sample_endpoint_per_sample_generators(
    logits: Tensor,
    *,
    node_mask: Tensor,
    generators: Sequence[torch.Generator],
    edges: bool,
) -> Tensor:
    """Batched endpoint probability construction plus row-local sampling."""

    probability = torch.softmax(logits.detach().float(), dim=-1)
    if edges:
        if probability.ndim != 4 or probability.shape[1] != probability.shape[2]:
            raise ValueError("edge logits must have shape [B,N,N,K]")
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        probability = probability.clone()
        probability[~valid] = 1.0 / probability.shape[-1]
        diagonal = torch.arange(probability.shape[1], device=probability.device)
        probability[:, diagonal, diagonal] = 1.0 / probability.shape[-1]
        indices = _sample_categorical_per_sample_generators(probability, generators=generators)
        upper = torch.triu(indices, diagonal=1)
        return upper + upper.transpose(1, 2)
    if probability.ndim != 3 or probability.shape[:2] != node_mask.shape:
        raise ValueError("node logits must have shape [B,N,K]")
    probability = probability.clone()
    probability[~node_mask] = 1.0 / probability.shape[-1]
    return _sample_categorical_per_sample_generators(probability, generators=generators)


def _sample_endpoint_stateless(
    logits: Tensor,
    *,
    node_mask: Tensor,
    seeds: Tensor,
    step_index: int,
    stream_id: int,
    edges: bool,
) -> Tensor:
    """Batched endpoint sampling using a stateless per-trajectory counter."""

    probability = torch.softmax(logits.detach().float(), dim=-1)
    if edges:
        if probability.ndim != 4 or probability.shape[1] != probability.shape[2]:
            raise ValueError("edge logits must have shape [B,N,N,K]")
        valid = node_mask[:, :, None] & node_mask[:, None, :]
        probability = probability.clone()
        probability[~valid] = 1.0 / probability.shape[-1]
        diagonal = torch.arange(probability.shape[1], device=probability.device)
        probability[:, diagonal, diagonal] = 1.0 / probability.shape[-1]
        indices = _sample_categorical_stateless(
            probability,
            seeds=seeds,
            step_index=step_index,
            stream_id=stream_id,
        )
        upper = torch.triu(indices, diagonal=1)
        return upper + upper.transpose(1, 2)
    if probability.ndim != 3 or probability.shape[:2] != node_mask.shape:
        raise ValueError("node logits must have shape [B,N,K]")
    probability = probability.clone()
    probability[~node_mask] = 1.0 / probability.shape[-1]
    return _sample_categorical_stateless(
        probability,
        seeds=seeds,
        step_index=step_index,
        stream_id=stream_id,
    )


def _predefined_step_probability(
    *,
    current: Tensor,
    product: Tensor,
    predicted_index: Tensor,
    t: Tensor,
    s: Tensor,
) -> Tensor:
    """R-star one-step transition used by predefined categorical DFM at eta=omega=0."""

    if current.shape != product.shape or current.shape[:-1] != predicted_index.shape:
        raise ValueError("current/product/predicted categorical state shapes disagree")
    if t.ndim != 1 or s.shape != t.shape or t.shape[0] != current.shape[0]:
        raise ValueError("t and s must be [B] and match categorical batch size")
    if not bool(torch.all(s > t)):
        raise ValueError("predefined roll-in requires a strictly increasing t->s bridge")
    predicted = F.one_hot(predicted_index, num_classes=current.shape[-1]).float()
    view_shape = (t.shape[0],) + (1,) * (current.ndim - 2) + (1,)
    time = t.float().reshape(view_shape)
    p_t = (1.0 - time) * product.float() + time * predicted
    delta = predicted - product.float()
    current_index = current.argmax(dim=-1, keepdim=True)
    delta_at_current = delta.gather(-1, current_index)
    p_at_current = p_t.gather(-1, current_index).squeeze(-1)
    support = (p_t > 0.0).sum(dim=-1).to(dtype=torch.float32)
    rate = torch.relu(delta - delta_at_current) / (
        support.unsqueeze(-1) * p_at_current.unsqueeze(-1)
    )
    rate = torch.nan_to_num(rate, nan=0.0, posinf=0.0, neginf=0.0)
    rate = torch.where(p_t > 0.0, rate, torch.zeros_like(rate))
    rate = torch.where(
        p_at_current.unsqueeze(-1) > 0.0,
        rate,
        torch.zeros_like(rate),
    )
    dt = (s - t).float().reshape(view_shape)
    probability = rate * dt
    probability.scatter_(-1, current_index, 0.0)
    stay = (1.0 - probability.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    probability.scatter_(-1, current_index, stay)
    return probability


def sample_predefined_student_transition(
    *,
    logits: Mapping[str, Tensor],
    current_X: Tensor,
    current_E: Tensor,
    p_X: Tensor,
    p_E: Tensor,
    node_mask: Tensor,
    p_mask: Tensor,
    t: Tensor,
    s: Tensor,
    generator: torch.Generator,
) -> StudentRollInState:
    """Sample one detached eta=omega=0 predefined-Diffusion student transition."""

    try:
        node_logits, edge_logits = logits["X"], logits["E"]
    except KeyError as error:
        raise KeyError("student roll-in logits must contain X and E") from error
    if (
        not isinstance(node_logits, Tensor)
        or not isinstance(edge_logits, Tensor)
        or node_logits.shape != current_X.shape
        or edge_logits.shape != current_E.shape
        or p_X.shape != current_X.shape
        or p_E.shape != current_E.shape
    ):
        raise ValueError("student roll-in tensors must share node and edge categorical shapes")
    if node_mask.shape != current_X.shape[:2] or p_mask.shape != node_mask.shape:
        raise ValueError("student roll-in masks differ from node state shape")
    device = current_X.device
    values = (node_logits, edge_logits, current_E, p_X, p_E, node_mask, p_mask, t, s)
    if any(value.device != device for value in values):
        raise ValueError("student roll-in tensors must share one device")
    _require_generator(generator, device)
    predicted_X = _sample_endpoint(node_logits, node_mask=node_mask.bool(), generator=generator, edges=False)
    predicted_E = _sample_endpoint(edge_logits, node_mask=node_mask.bool(), generator=generator, edges=True)
    probability_X = _predefined_step_probability(
        current=current_X,
        product=p_X,
        predicted_index=predicted_X,
        t=t,
        s=s,
    )
    probability_E = _predefined_step_probability(
        current=current_E,
        product=p_E,
        predicted_index=predicted_E,
        t=t,
        s=s,
    )
    next_X = F.one_hot(
        _sample_categorical(probability_X, generator=generator), num_classes=current_X.shape[-1]
    ).to(dtype=current_X.dtype)
    next_E = F.one_hot(
        _sample_categorical(probability_E, generator=generator), num_classes=current_E.shape[-1]
    ).to(dtype=current_E.dtype)
    next_X = torch.where(p_mask.bool().unsqueeze(-1), p_X, next_X)
    valid_edge = node_mask.bool()[:, :, None] & node_mask.bool()[:, None, :]
    next_X = next_X * node_mask.bool().unsqueeze(-1)
    next_E = next_E * valid_edge.unsqueeze(-1)
    diagonal = torch.arange(next_E.shape[1], device=device)
    next_E[:, diagonal, diagonal] = 0
    next_E[:, diagonal, diagonal, 0] = 1
    upper = torch.triu(next_E.argmax(dim=-1), diagonal=1)
    upper = upper + upper.transpose(1, 2)
    next_E = F.one_hot(upper, num_classes=next_E.shape[-1]).to(dtype=current_E.dtype)
    next_E = next_E * valid_edge.unsqueeze(-1)
    next_E[:, diagonal, diagonal] = 0
    next_E[:, diagonal, diagonal, 0] = 1
    return StudentRollInState(t_X=next_X.detach(), t_E=next_E.detach())


def sample_predefined_student_transition_per_sample_generators(
    *,
    logits: Mapping[str, Tensor],
    current_X: Tensor,
    current_E: Tensor,
    p_X: Tensor,
    p_E: Tensor,
    node_mask: Tensor,
    p_mask: Tensor,
    t: Tensor,
    s: Tensor,
    generators: Sequence[torch.Generator],
) -> StudentRollInState:
    """Vectorize deterministic one-step math while retaining one RNG per row.

    ``sample_predefined_student_transition`` is the semantic reference.  This
    sibling exists only for inference profiles that have passed a direct-output
    gate.  It batches softmax, bridge-rate, masking, and symmetry work, but
    does not batch the categorical draws across independent generators.
    """

    try:
        node_logits, edge_logits = logits["X"], logits["E"]
    except KeyError as error:
        raise KeyError("student roll-in logits must contain X and E") from error
    if (
        not isinstance(node_logits, Tensor)
        or not isinstance(edge_logits, Tensor)
        or node_logits.shape != current_X.shape
        or edge_logits.shape != current_E.shape
        or p_X.shape != current_X.shape
        or p_E.shape != current_E.shape
    ):
        raise ValueError("student roll-in tensors must share node and edge categorical shapes")
    if node_mask.shape != current_X.shape[:2] or p_mask.shape != node_mask.shape:
        raise ValueError("student roll-in masks differ from node state shape")
    device = current_X.device
    values = (node_logits, edge_logits, current_E, p_X, p_E, node_mask, p_mask, t, s)
    if any(value.device != device for value in values):
        raise ValueError("student roll-in tensors must share one device")
    if len(generators) != current_X.shape[0]:
        raise ValueError("per-sample generators must have one entry per batch row")
    for generator in generators:
        _require_generator(generator, device)

    predicted_X = _sample_endpoint_per_sample_generators(
        node_logits,
        node_mask=node_mask.bool(),
        generators=generators,
        edges=False,
    )
    predicted_E = _sample_endpoint_per_sample_generators(
        edge_logits,
        node_mask=node_mask.bool(),
        generators=generators,
        edges=True,
    )
    probability_X = _predefined_step_probability(
        current=current_X,
        product=p_X,
        predicted_index=predicted_X,
        t=t,
        s=s,
    )
    probability_E = _predefined_step_probability(
        current=current_E,
        product=p_E,
        predicted_index=predicted_E,
        t=t,
        s=s,
    )
    next_X = F.one_hot(
        _sample_categorical_per_sample_generators(probability_X, generators=generators),
        num_classes=current_X.shape[-1],
    ).to(dtype=current_X.dtype)
    next_E = F.one_hot(
        _sample_categorical_per_sample_generators(probability_E, generators=generators),
        num_classes=current_E.shape[-1],
    ).to(dtype=current_E.dtype)
    next_X = torch.where(p_mask.bool().unsqueeze(-1), p_X, next_X)
    valid_edge = node_mask.bool()[:, :, None] & node_mask.bool()[:, None, :]
    next_X = next_X * node_mask.bool().unsqueeze(-1)
    next_E = next_E * valid_edge.unsqueeze(-1)
    diagonal = torch.arange(next_E.shape[1], device=device)
    next_E[:, diagonal, diagonal] = 0
    next_E[:, diagonal, diagonal, 0] = 1
    upper = torch.triu(next_E.argmax(dim=-1), diagonal=1)
    upper = upper + upper.transpose(1, 2)
    next_E = F.one_hot(upper, num_classes=next_E.shape[-1]).to(dtype=current_E.dtype)
    next_E = next_E * valid_edge.unsqueeze(-1)
    next_E[:, diagonal, diagonal] = 0
    next_E[:, diagonal, diagonal, 0] = 1
    return StudentRollInState(t_X=next_X.detach(), t_E=next_E.detach())


def sample_predefined_student_transition_stateless(
    *,
    logits: Mapping[str, Tensor],
    current_X: Tensor,
    current_E: Tensor,
    p_X: Tensor,
    p_E: Tensor,
    node_mask: Tensor,
    p_mask: Tensor,
    t: Tensor,
    s: Tensor,
    seeds: Tensor,
    step_index: int,
) -> StudentRollInState:
    """Sample one bridge transition with vectorized stateless categorical RNG.

    This is an opt-in inference kernel.  It has the same categorical
    probabilities and fixed bridge as the reference sampler, but replaces
    mutable per-row generator streams with deterministic counter uniforms
    keyed to every frozen trajectory seed.
    """

    try:
        node_logits, edge_logits = logits["X"], logits["E"]
    except KeyError as error:
        raise KeyError("student roll-in logits must contain X and E") from error
    if (
        not isinstance(node_logits, Tensor)
        or not isinstance(edge_logits, Tensor)
        or node_logits.shape != current_X.shape
        or edge_logits.shape != current_E.shape
        or p_X.shape != current_X.shape
        or p_E.shape != current_E.shape
    ):
        raise ValueError("student roll-in tensors must share node and edge categorical shapes")
    if node_mask.shape != current_X.shape[:2] or p_mask.shape != node_mask.shape:
        raise ValueError("student roll-in masks differ from node state shape")
    device = current_X.device
    values = (node_logits, edge_logits, current_E, p_X, p_E, node_mask, p_mask, t, s)
    if any(value.device != device for value in values):
        raise ValueError("student roll-in tensors must share one device")
    normalized_seeds = _require_stateless_seeds(
        seeds,
        batch_size=current_X.shape[0],
        device=device,
    )
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("stateless transition step_index must be a non-negative integer")

    predicted_X = _sample_endpoint_stateless(
        node_logits,
        node_mask=node_mask.bool(),
        seeds=normalized_seeds,
        step_index=step_index,
        stream_id=0,
        edges=False,
    )
    predicted_E = _sample_endpoint_stateless(
        edge_logits,
        node_mask=node_mask.bool(),
        seeds=normalized_seeds,
        step_index=step_index,
        stream_id=1,
        edges=True,
    )
    probability_X = _predefined_step_probability(
        current=current_X,
        product=p_X,
        predicted_index=predicted_X,
        t=t,
        s=s,
    )
    probability_E = _predefined_step_probability(
        current=current_E,
        product=p_E,
        predicted_index=predicted_E,
        t=t,
        s=s,
    )
    next_X = F.one_hot(
        _sample_categorical_stateless(
            probability_X,
            seeds=normalized_seeds,
            step_index=step_index,
            stream_id=2,
        ),
        num_classes=current_X.shape[-1],
    ).to(dtype=current_X.dtype)
    next_E = F.one_hot(
        _sample_categorical_stateless(
            probability_E,
            seeds=normalized_seeds,
            step_index=step_index,
            stream_id=3,
        ),
        num_classes=current_E.shape[-1],
    ).to(dtype=current_E.dtype)
    next_X = torch.where(p_mask.bool().unsqueeze(-1), p_X, next_X)
    valid_edge = node_mask.bool()[:, :, None] & node_mask.bool()[:, None, :]
    next_X = next_X * node_mask.bool().unsqueeze(-1)
    next_E = next_E * valid_edge.unsqueeze(-1)
    diagonal = torch.arange(next_E.shape[1], device=device)
    next_E[:, diagonal, diagonal] = 0
    next_E[:, diagonal, diagonal, 0] = 1
    upper = torch.triu(next_E.argmax(dim=-1), diagonal=1)
    upper = upper + upper.transpose(1, 2)
    next_E = F.one_hot(upper, num_classes=next_E.shape[-1]).to(dtype=current_E.dtype)
    next_E = next_E * valid_edge.unsqueeze(-1)
    next_E[:, diagonal, diagonal] = 0
    next_E[:, diagonal, diagonal, 0] = 1
    return StudentRollInState(t_X=next_X.detach(), t_E=next_E.detach())


__all__ = [
    "PositiveRecoveryConfig",
    "PositiveRecoveryTime",
    "StudentRollInState",
    "positive_recovery_time",
    "sample_predefined_student_transition",
    "sample_predefined_student_transition_per_sample_generators",
    "sample_predefined_student_transition_stateless",
]
