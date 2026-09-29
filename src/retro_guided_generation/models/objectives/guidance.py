"""Pure-tensor, branch-aware objectives for guided GraphDiT training.

This module intentionally accepts an explicit supervision contract rather than
the historical Lightning batch class.  The objective equations and groupwise
normalization remain compatible with the V4 guided Stage A/B design.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..dfm.coupled_corruption import CorruptedGraphBatch
from ..dfm.losses import (
    build_group_position_masks,
    common_dfm_per_graph,
    common_reference_kl_per_graph,
    full_dfm_per_graph,
    full_reference_kl_per_graph,
    positive_full_dfm_per_graph,
    preference_dfm_per_graph,
    squared_hinge,
)


_PRESERVATION_MODES = frozenset(
    {
        "legacy",
        "true_trust",
        "positive_dfm",
        "positive_trust",
        "reference_kl",
        "reference_kl_shift",
    }
)
_CANDIDATE_SUPERVISION_MODES = frozenset(
    {
        "contrastive",
        "positive_sft",
        "positive_rank",
        "positive_rank_hybrid",
        "positive_negative_sampled",
        "positive_negative_reverse",
    }
)
_METRIC_GUIDANCE_MODES = frozenset(
    {
        "none",
        "candidate_teacher_listwise_kl",
        "teacher_listwise_kl",
        "label_pairwise",
    }
)
_CANDIDATE_SCORE_CALIBRATION_MODES = frozenset(
    {
        "none",
        "group_reference_mean",
    }
)


def _assert_condition(condition: Tensor, message: str, error_type: type[Exception] = ValueError) -> None:
    if condition.numel() != 1:
        raise ValueError("validation condition must be scalar")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise error_type(message)


def _finite_hyperparameter(name: str, value: float, *, positive: bool) -> float:
    result = float(value)
    if not math.isfinite(result) or (result <= 0.0 if positive else result < 0.0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class ObjectiveNormalization:
    """Optional fixed full-epoch scaling for case-weighted train batches."""

    epoch_normalization: float

    def __post_init__(self) -> None:
        _finite_hyperparameter("epoch_normalization", self.epoch_normalization, positive=True)


@dataclass(frozen=True, slots=True)
class GuidedObjectiveBatch:
    """Model-independent candidate supervision required by guided objectives.

    Candidate rows must be contiguous by group.  Every probability field is
    normalized separately inside its group.  ``guidance_branch`` is required
    only when a branch-aware hybrid objective is enabled.
    """

    group_ptr: Tensor
    endpoint_probability: Tensor
    teacher_probability: Tensor
    structural_label: Tensor
    listwise_eligible: Tensor
    guidance_branch: tuple[str, ...] | None = None
    group_index: Tensor | None = None
    group_loss_weight: Tensor | None = None
    branch_group_loss_weight: Tensor | None = None
    objective_normalization: ObjectiveNormalization | None = None


@dataclass(frozen=True, slots=True)
class GuidedObjectiveOutput:
    """Loss plus compact, named scalar and group-level diagnostics."""

    loss: Tensor
    per_candidate_score: Tensor
    per_candidate_prediction_probability: Tensor
    eligible_group_mask: Tensor
    structure_group_mask: Tensor
    metric_group_mask: Tensor
    metrics: Mapping[str, Tensor]
    per_group: Mapping[str, Tensor]

    def metric(self, name: str) -> Tensor:
        return self.metrics[name]

    def group_metric(self, name: str) -> Tensor:
        return self.per_group[name]

    @property
    def guided_dfm(self) -> Tensor:
        return self.metrics["common_dfm"]


@dataclass(frozen=True, slots=True)
class GuidanceObjectiveConfig:
    """Explicit configuration for V4-compatible Stage A/B objective terms."""

    edge_weight: float = 10.0
    temperature: float = 0.5
    candidate_score_calibration: str = "group_reference_mean"
    candidate_score_scale_floor: float = 1.0
    common_weight: float = 1.0
    listwise_weight: float = 0.1
    anchor_weight: float = 0.05
    hybrid_branch_objective: bool = False
    # Keep the historical objective-level defaults for legacy Stage-A callers.
    # Stage B owns its stricter structure-only defaults and metric opt-in gates.
    structure_guidance_weight: float = 0.5
    metric_guidance_weight: float = 0.5
    metric_guidance_mode: str = "none"
    metric_pairwise_margin: float = 0.1
    pairwise_huber_delta: float = 0.25
    pairwise_margin_max: float = 4.0
    preservation_mode: str = "legacy"
    positive_dfm_weight: float = 0.0
    trust_region_weight: float | None = None
    common_kl_budget: float = 0.02
    full_reference_kl_guard_weight: float = 0.0
    full_reference_kl_budget: float = 0.02
    candidate_supervision_mode: str = "contrastive"
    positive_supervision_weight: float = 0.1
    negative_supervision_weight: float = 0.1
    negative_reverse_weight: float = 0.1
    negative_reverse_margin: float = 0.0
    negative_reverse_temperature: float = 0.1
    route_ranking_weight: float = 0.05
    route_ranking_margin: float = 0.1

    def __post_init__(self) -> None:
        for name in (
            "edge_weight",
            "common_weight",
            "listwise_weight",
            "anchor_weight",
            "structure_guidance_weight",
            "metric_guidance_weight",
            "metric_pairwise_margin",
            "positive_dfm_weight",
            "common_kl_budget",
            "full_reference_kl_guard_weight",
            "full_reference_kl_budget",
            "positive_supervision_weight",
            "negative_supervision_weight",
            "negative_reverse_weight",
            "negative_reverse_margin",
            "route_ranking_weight",
            "route_ranking_margin",
        ):
            _finite_hyperparameter(name, getattr(self, name), positive=False)
        for name in (
            "temperature",
            "candidate_score_scale_floor",
            "pairwise_huber_delta",
            "pairwise_margin_max",
            "negative_reverse_temperature",
        ):
            _finite_hyperparameter(name, getattr(self, name), positive=True)
        if self.trust_region_weight is not None:
            _finite_hyperparameter("trust_region_weight", self.trust_region_weight, positive=False)
        if not isinstance(self.hybrid_branch_objective, bool):
            raise TypeError("hybrid_branch_objective must be a bool")
        if not math.isclose(
            float(self.structure_guidance_weight) + float(self.metric_guidance_weight),
            1.0,
            abs_tol=1.0e-8,
            rel_tol=0.0,
        ):
            raise ValueError("structure_guidance_weight + metric_guidance_weight must equal 1")
        mode = str(self.preservation_mode).strip().lower()
        if mode not in _PRESERVATION_MODES:
            raise ValueError(f"unsupported preservation_mode: {self.preservation_mode!r}")
        if float(self.full_reference_kl_guard_weight) > 0.0 and mode not in {
            "reference_kl",
            "reference_kl_shift",
        }:
            raise ValueError(
                "full_reference_kl_guard_weight requires "
                "preservation_mode='reference_kl'"
            )
        candidate_mode = str(self.candidate_supervision_mode).strip().lower()
        if candidate_mode not in _CANDIDATE_SUPERVISION_MODES:
            raise ValueError(f"unsupported candidate_supervision_mode: {self.candidate_supervision_mode!r}")
        metric_mode = str(self.metric_guidance_mode).strip().lower()
        if metric_mode not in _METRIC_GUIDANCE_MODES:
            raise ValueError(f"unsupported metric_guidance_mode: {self.metric_guidance_mode!r}")
        calibration_mode = str(self.candidate_score_calibration).strip().lower()
        if calibration_mode not in _CANDIDATE_SCORE_CALIBRATION_MODES:
            raise ValueError(
                "unsupported candidate_score_calibration: "
                f"{self.candidate_score_calibration!r}"
            )


@dataclass(frozen=True, slots=True)
class _GroupLayout:
    ptr: Tensor
    sizes: Tensor
    index: Tensor
    num_groups: int
    num_candidates: int


def _group_layout(group_ptr: Tensor, num_candidates: int, device: torch.device) -> _GroupLayout:
    if not isinstance(group_ptr, Tensor) or group_ptr.ndim != 1 or group_ptr.numel() < 2:
        raise ValueError("group_ptr must be a one-dimensional CSR pointer")
    if group_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_ptr must use an integer dtype")
    ptr = group_ptr.to(device=device, dtype=torch.long)
    sizes = ptr[1:] - ptr[:-1]
    valid = (ptr[0] == 0) & (ptr[-1] == num_candidates) & torch.all(sizes > 0)
    _assert_condition(valid, "group_ptr must contain positive contiguous segments covering candidates")
    groups = int(ptr.numel() - 1)
    index = torch.repeat_interleave(
        torch.arange(groups, device=device, dtype=torch.long), sizes, output_size=num_candidates
    )
    return _GroupLayout(ptr, sizes, index, groups, num_candidates)


def _validate_probability(value: Tensor, name: str, layout: _GroupLayout, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor) or value.shape != (layout.num_candidates,):
        raise ValueError(f"{name} must have shape [{layout.num_candidates}]")
    result = value.to(device=device, dtype=torch.float32)
    _assert_condition(torch.isfinite(result).all() & torch.all(result >= 0.0), f"{name} must be finite and non-negative")
    group_sum = torch.segment_reduce(result, "sum", lengths=layout.sizes)
    _assert_condition(
        torch.isclose(group_sum, torch.ones_like(group_sum), atol=1.0e-6, rtol=0.0).all(),
        f"{name} must be normalized within every group",
    )
    return result


def _validate_group_weights(value: Tensor | None, name: str, groups: int, device: torch.device, fallback: Tensor | None = None) -> Tensor:
    if value is None:
        if fallback is not None:
            return fallback
        return torch.ones(groups, device=device, dtype=torch.float32)
    if not isinstance(value, Tensor) or value.shape != (groups,):
        raise ValueError(f"{name} must have shape [{groups}]")
    result = value.to(device=device, dtype=torch.float32)
    _assert_condition(torch.isfinite(result).all() & torch.all(result > 0.0), f"{name} must be finite and positive")
    return result


def _weighted_mean(values: Tensor, weights: Tensor, zero_source: Tensor, selected: Tensor | None = None) -> Tensor:
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("group values and weights must be matching one-dimensional tensors")
    if selected is not None:
        if selected.shape != values.shape or selected.dtype != torch.bool:
            raise ValueError("selected must be bool[G]")
        values, weights = values[selected], weights[selected]
    if values.numel() == 0:
        return zero_source.sum() * 0.0
    return (values * weights).sum() / weights.sum()


def _training_reduce(values: Tensor, weights: Tensor, normalization: ObjectiveNormalization | None, legacy_mean: Tensor) -> Tensor:
    if normalization is None:
        return legacy_mean
    if not isinstance(normalization, ObjectiveNormalization):
        raise TypeError("objective_normalization must be ObjectiveNormalization or None")
    return (values * weights).sum() * values.new_tensor(normalization.epoch_normalization)


def _endpoint_reduce(candidate_value: Tensor, endpoint_probability: Tensor, layout: _GroupLayout) -> Tensor:
    if candidate_value.shape != endpoint_probability.shape:
        raise ValueError("candidate values and endpoint probabilities must have matching shape")
    return torch.segment_reduce(candidate_value * endpoint_probability, "sum", lengths=layout.sizes)


def _calibrate_reference_relative_score(
    raw_score: Tensor,
    reference_dfm: Tensor,
    layout: _GroupLayout,
    mode: str,
    scale_floor: float,
) -> tuple[Tensor, Tensor]:
    """Make reference-minus-student DFM scores dimensionless per group.

    A single detached scale is shared by every candidate in a group.  This is
    important: it preserves the exact within-group ranking and every pairwise
    ordering from the established reference-relative objective, while making
    its temperature and margins independent of the base generator's raw DFM
    magnitude.  The floor avoids amplifying already well-calibrated losses.
    """

    if raw_score.shape != (layout.num_candidates,) or reference_dfm.shape != raw_score.shape:
        raise ValueError("raw_score and reference_dfm must both have shape [C]")
    calibration_mode = str(mode).strip().lower()
    if calibration_mode not in _CANDIDATE_SCORE_CALIBRATION_MODES:
        raise ValueError(f"unsupported candidate score calibration: {mode!r}")
    floor = _finite_hyperparameter("candidate_score_scale_floor", scale_floor, positive=True)
    score32 = raw_score.float()
    reference32 = reference_dfm.detach().float()
    _assert_condition(
        torch.isfinite(score32).all() & torch.isfinite(reference32).all(),
        "candidate score calibration requires finite DFM values",
    )
    if calibration_mode == "none":
        group_scale = torch.ones(
            layout.num_groups,
            device=score32.device,
            dtype=torch.float32,
        )
    else:
        group_scale = torch.segment_reduce(
            reference32.abs(), "mean", lengths=layout.sizes
        ).clamp_min(floor)
    return score32 / group_scale.index_select(0, layout.index), group_scale


def _listwise_statistics(
    scores: Tensor,
    teacher_probability: Tensor,
    labels: Tensor,
    eligible: Tensor,
    layout: _GroupLayout,
    temperature: float,
) -> dict[str, Tensor]:
    scaled = scores.float() / float(temperature)
    group_max = torch.segment_reduce(scaled, "max", lengths=layout.sizes)
    shifted = scaled - group_max.index_select(0, layout.index)
    group_exp_sum = torch.segment_reduce(shifted.exp(), "sum", lengths=layout.sizes)
    log_probability = shifted - group_exp_sum.log().index_select(0, layout.index)
    prediction_probability = log_probability.exp()
    group_ce = -torch.segment_reduce(teacher_probability * log_probability, "sum", lengths=layout.sizes)
    group_teacher_entropy = -torch.segment_reduce(
        torch.special.xlogy(teacher_probability, teacher_probability), "sum", lengths=layout.sizes
    )
    group_predicted_entropy = -torch.segment_reduce(
        prediction_probability * log_probability, "sum", lengths=layout.sizes
    )
    positive, negative = labels == 1, labels == 0
    positive_count = torch.segment_reduce(positive.float(), "sum", lengths=layout.sizes)
    negative_count = torch.segment_reduce(negative.float(), "sum", lengths=layout.sizes)
    _assert_condition(
        ((~eligible) | ((positive_count > 0.0) & (negative_count > 0.0))).all(),
        "every eligible group must contain positive and negative candidates",
    )
    positive_sum = torch.segment_reduce(torch.where(positive, scores, torch.zeros_like(scores)), "sum", lengths=layout.sizes)
    negative_sum = torch.segment_reduce(torch.where(negative, scores, torch.zeros_like(scores)), "sum", lengths=layout.sizes)
    group_margin = positive_sum / positive_count.clamp_min(1.0) - negative_sum / negative_count.clamp_min(1.0)
    same_group = layout.index[:, None] == layout.index[None, :]
    pair_mask = positive[:, None] & negative[None, :] & same_group
    pair_delta = scores[:, None] - scores[None, :]
    pair_credit = ((pair_delta > 0).float() + 0.5 * (pair_delta == 0).float()) * pair_mask.float()
    group_pair_credit = torch.segment_reduce(pair_credit.sum(dim=1), "sum", lengths=layout.sizes)
    group_pair_count = torch.segment_reduce(pair_mask.sum(dim=1, dtype=torch.float32), "sum", lengths=layout.sizes)
    group_pair_accuracy = group_pair_credit / group_pair_count.clamp_min(1.0)
    group_score_max = torch.segment_reduce(scores, "max", lengths=layout.sizes)
    group_teacher_max = torch.segment_reduce(teacher_probability, "max", lengths=layout.sizes)
    predicted_top = scores == group_score_max.index_select(0, layout.index)
    teacher_top = teacher_probability == group_teacher_max.index_select(0, layout.index)
    overlap = torch.segment_reduce((predicted_top & teacher_top).float(), "sum", lengths=layout.sizes)
    predicted_count = torch.segment_reduce(predicted_top.float(), "sum", lengths=layout.sizes)
    mean = torch.segment_reduce(scores, "mean", lengths=layout.sizes)
    std = torch.segment_reduce((scores - mean.index_select(0, layout.index)).square(), "mean", lengths=layout.sizes).clamp_min(0.0).sqrt()
    active = eligible.float()
    return {
        "prediction_probability": prediction_probability,
        "listwise_ce": group_ce * active,
        "teacher_entropy": group_teacher_entropy * active,
        "teacher_kl": (group_ce - group_teacher_entropy) * active,
        "score_margin": group_margin * active,
        "pair_accuracy": group_pair_accuracy * active,
        "top1_accuracy": overlap / predicted_count.clamp_min(1.0) * active,
        "predicted_entropy": group_predicted_entropy * active,
        "score_mean": mean * active,
        "score_std": std * active,
        "score_max": group_score_max * active,
    }


def _structure_pairwise_statistics(
    scores: Tensor,
    teacher_probability: Tensor,
    labels: Tensor,
    eligible: Tensor,
    structure_group: Tensor,
    layout: _GroupLayout,
    temperature: float,
    huber_delta: float,
    margin_max: float,
) -> dict[str, Tensor]:
    selected_group = structure_group & eligible
    selected_candidate = selected_group.index_select(0, layout.index)
    positive, negative = labels == 1, labels == 0
    same_group = layout.index[:, None] == layout.index[None, :]
    pair_mask = positive[:, None] & negative[None, :] & same_group & selected_candidate[:, None]
    raw_target = float(temperature) * (
        teacher_probability.clamp_min(torch.finfo(torch.float32).tiny).log()[:, None]
        - teacher_probability.clamp_min(torch.finfo(torch.float32).tiny).log()[None, :]
    )
    _assert_condition(((~pair_mask) | (raw_target > 0.0)).all(), "structure teacher probabilities must rank positives above negatives")
    target = raw_target.clamp(max=float(margin_max))
    score_delta = scores[:, None] - scores[None, :]
    violation = (target - score_delta).clamp_min(0.0)
    loss = torch.where(
        violation <= float(huber_delta),
        0.5 * violation.square() / float(huber_delta),
        violation - 0.5 * float(huber_delta),
    )
    pair_float = pair_mask.float()
    per_candidate_count = pair_mask.sum(dim=1, dtype=torch.float32)
    group_count = torch.segment_reduce(per_candidate_count, "sum", lengths=layout.sizes)
    group_loss = torch.segment_reduce((loss * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / group_count.clamp_min(1.0)
    group_target = torch.segment_reduce((target * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / group_count.clamp_min(1.0)
    group_satisfaction = torch.segment_reduce(((score_delta >= target).float() * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / group_count.clamp_min(1.0)
    active = selected_group.float()
    return {
        "pairwise_loss": group_loss * active,
        "pair_count": group_count * active,
        "target_margin": group_target * active,
        "margin_satisfaction": group_satisfaction * active,
    }


def _route_pairwise_statistics(
    scores: Tensor,
    labels: Tensor,
    eligible: Tensor,
    layout: _GroupLayout,
    margin: float,
    huber_delta: float,
) -> dict[str, Tensor]:
    positive, negative = labels == 1, labels == 0
    same_group = layout.index[:, None] == layout.index[None, :]
    selected_candidate = eligible.index_select(0, layout.index)
    pair_mask = positive[:, None] & negative[None, :] & same_group & selected_candidate[:, None]
    delta = scores[:, None] - scores[None, :]
    violation = (float(margin) - delta).clamp_min(0.0)
    loss = torch.where(
        violation <= float(huber_delta),
        0.5 * violation.square() / float(huber_delta),
        violation - 0.5 * float(huber_delta),
    )
    pair_float = pair_mask.float()
    candidate_count = pair_mask.sum(dim=1, dtype=torch.float32)
    group_count = torch.segment_reduce(candidate_count, "sum", lengths=layout.sizes)
    group_loss = torch.segment_reduce((loss * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / group_count.clamp_min(1.0)
    group_margin = torch.segment_reduce((delta * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / group_count.clamp_min(1.0)
    selected = eligible & (group_count > 0.0)
    active = selected.float()
    return {"pairwise_loss": group_loss * active, "gain_margin": group_margin * active, "selected": selected}


def _metric_label_pairwise_statistics(
    scores: Tensor,
    labels: Tensor,
    eligible: Tensor,
    metric_group: Tensor,
    layout: _GroupLayout,
    margin: float,
) -> dict[str, Tensor]:
    """V4-style label-pairwise metric supervision, normalized within group.

    Every selected group contributes the mean over its positive-by-negative
    pairs, so a larger candidate pool cannot dominate a smaller one.  The
    softplus form is intentionally independent of teacher-probability gaps:
    it tests whether binary metric labels supply a cleaner ranking signal than
    teacher-distribution matching.
    """

    selected_group = metric_group & eligible
    selected_candidate = selected_group.index_select(0, layout.index)
    positive, negative = labels == 1, labels == 0
    same_group = layout.index[:, None] == layout.index[None, :]
    pair_mask = positive[:, None] & negative[None, :] & same_group & selected_candidate[:, None]
    score_delta = scores[:, None] - scores[None, :]
    pair_loss = F.softplus(scores.new_tensor(float(margin)) - score_delta)
    pair_float = pair_mask.float()
    per_candidate_count = pair_mask.sum(dim=1, dtype=torch.float32)
    group_count = torch.segment_reduce(per_candidate_count, "sum", lengths=layout.sizes)
    divisor = group_count.clamp_min(1.0)
    group_loss = torch.segment_reduce((pair_loss * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / divisor
    group_margin = torch.segment_reduce((score_delta * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / divisor
    group_accuracy = torch.segment_reduce(((score_delta > 0.0).float() * pair_float).sum(dim=1), "sum", lengths=layout.sizes) / divisor
    selected = selected_group & (group_count > 0.0)
    active = selected.float()
    return {
        "pairwise_loss": group_loss * active,
        "pair_count": group_count * active,
        "score_margin": group_margin * active,
        "pair_accuracy": group_accuracy * active,
        "selected": selected,
    }


class GuidanceObjective(nn.Module):
    """V4-compatible guided ranking and preservation objective.

    It is deliberately independent of a trainer framework.  The surrounding
    Stage A/B module owns optimizer scheduling, while this component owns only
    coupled-loss equations and their explicit groupwise reductions.
    """

    def __init__(self, config: GuidanceObjectiveConfig | None = None, **overrides: object) -> None:
        super().__init__()
        if config is None:
            config = GuidanceObjectiveConfig(**overrides)
        elif not isinstance(config, GuidanceObjectiveConfig):
            raise TypeError("config must be GuidanceObjectiveConfig or None")
        elif overrides:
            raise ValueError("pass either a config or keyword overrides, not both")
        self.config = config
        self.edge_weight = float(config.edge_weight)
        self.temperature = float(config.temperature)
        self.candidate_score_calibration = str(
            config.candidate_score_calibration
        ).strip().lower()
        self.candidate_score_scale_floor = float(config.candidate_score_scale_floor)
        self.common_weight = float(config.common_weight)
        self.listwise_weight = float(config.listwise_weight)
        self.anchor_weight = float(config.anchor_weight)
        self.hybrid_branch_objective = config.hybrid_branch_objective
        self.structure_guidance_weight = float(config.structure_guidance_weight)
        self.metric_guidance_weight = float(config.metric_guidance_weight)
        self.metric_guidance_mode = str(config.metric_guidance_mode).strip().lower()
        self.metric_pairwise_margin = float(config.metric_pairwise_margin)
        self.pairwise_huber_delta = float(config.pairwise_huber_delta)
        self.pairwise_margin_max = float(config.pairwise_margin_max)
        self.preservation_mode = str(config.preservation_mode).strip().lower()
        self.positive_dfm_weight = float(config.positive_dfm_weight)
        self.trust_region_weight = float(
            config.anchor_weight if config.trust_region_weight is None else config.trust_region_weight
        )
        self.common_kl_budget = float(config.common_kl_budget)
        self.full_reference_kl_guard_weight = float(
            config.full_reference_kl_guard_weight
        )
        self.full_reference_kl_budget = float(config.full_reference_kl_budget)
        self.candidate_supervision_mode = str(config.candidate_supervision_mode).strip().lower()
        self.positive_supervision_weight = float(config.positive_supervision_weight)
        self.negative_supervision_weight = float(config.negative_supervision_weight)
        self.negative_reverse_weight = float(config.negative_reverse_weight)
        self.negative_reverse_margin = float(config.negative_reverse_margin)
        self.negative_reverse_temperature = float(config.negative_reverse_temperature)
        self.route_ranking_weight = float(config.route_ranking_weight)
        self.route_ranking_margin = float(config.route_ranking_margin)

    def _validate_batch(
        self,
        batch: GuidedObjectiveBatch,
        corrupted: CorruptedGraphBatch,
        num_candidates: int,
        device: torch.device,
    ) -> tuple[_GroupLayout, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if not isinstance(batch, GuidedObjectiveBatch):
            raise TypeError("batch must be GuidedObjectiveBatch")
        layout = _group_layout(batch.group_ptr, num_candidates, device)
        if batch.group_index is not None:
            if not isinstance(batch.group_index, Tensor) or batch.group_index.shape != (num_candidates,):
                raise ValueError(f"group_index must have shape [{num_candidates}]")
            _assert_condition(
                torch.all(batch.group_index.to(device=device, dtype=torch.long) == layout.index),
                "group_index is inconsistent with group_ptr",
            )
        if corrupted.t_group.shape != (layout.num_groups,) or corrupted.t_graph.shape != (num_candidates,):
            raise ValueError("corruption time tensors differ from group layout")
        _assert_condition(
            torch.all(corrupted.t_graph.to(device).float() == corrupted.t_group.to(device).float().index_select(0, layout.index)),
            "corrupted t_graph is inconsistent with group membership",
        )
        endpoint_probability = _validate_probability(
            batch.endpoint_probability, "endpoint_probability", layout, device
        )
        teacher_probability = _validate_probability(
            batch.teacher_probability, "teacher_probability", layout, device
        )
        labels = batch.structural_label.to(device=device)
        if labels.shape != (num_candidates,):
            raise ValueError(f"structural_label must have shape [{num_candidates}]")
        _assert_condition(torch.all((labels == 0) | (labels == 1)), "structural_label must contain only 0 and 1")
        eligible = batch.listwise_eligible.to(device=device, dtype=torch.bool)
        if eligible.shape != (layout.num_groups,):
            raise ValueError(f"listwise_eligible must have shape [{layout.num_groups}]")
        group_weight = _validate_group_weights(
            batch.group_loss_weight, "group_loss_weight", layout.num_groups, device
        )
        branch_weight = _validate_group_weights(
            batch.branch_group_loss_weight,
            "branch_group_loss_weight",
            layout.num_groups,
            device,
            fallback=group_weight,
        )
        if self.hybrid_branch_objective:
            if not isinstance(batch.guidance_branch, tuple) or len(batch.guidance_branch) != layout.num_groups:
                raise ValueError("hybrid objective requires one guidance branch per group")
            branches = tuple(str(value) for value in batch.guidance_branch)
            invalid = sorted(set(branches) - {"structure", "metric"})
            if invalid:
                raise ValueError(f"unknown guidance branches: {invalid}")
            structure_group = torch.tensor(
                [value == "structure" for value in branches], device=device, dtype=torch.bool
            )
            metric_group = ~structure_group
        else:
            structure_group = torch.zeros(layout.num_groups, device=device, dtype=torch.bool)
            metric_group = ~structure_group
        return (
            layout,
            endpoint_probability,
            teacher_probability,
            labels,
            eligible,
            group_weight,
            branch_weight,
            structure_group,
            metric_group,
        )

    @staticmethod
    def _freeze_mapping(values: dict[str, Tensor]) -> Mapping[str, Tensor]:
        return MappingProxyType(values)

    def forward(
        self,
        student_logits: Mapping[str, Tensor],
        reference_logits: Mapping[str, Tensor],
        corrupted: CorruptedGraphBatch,
        batch: GuidedObjectiveBatch,
    ) -> GuidedObjectiveOutput:
        """Evaluate all configured guided terms on one shared corruption draw."""

        if not isinstance(corrupted, CorruptedGraphBatch):
            raise TypeError("corrupted must be CorruptedGraphBatch")
        try:
            student_x, student_e = student_logits["X"], student_logits["E"]
            reference_x, reference_e = reference_logits["X"], reference_logits["E"]
        except KeyError as error:
            raise KeyError("student and reference logits must contain X and E") from error
        if student_x.device != corrupted.target_X.device:
            raise ValueError("student logits and corruption must share a device")
        if reference_x.device != student_x.device or reference_e.device != student_x.device:
            raise ValueError("student and reference logits must share a device")
        candidates = int(student_x.size(0))
        (
            layout,
            endpoint_probability,
            teacher_probability,
            labels,
            cached_eligible,
            group_weight,
            branch_weight,
            structure_group,
            metric_group,
        ) = self._validate_batch(batch, corrupted, candidates, student_x.device)
        targets = {"X": corrupted.target_X, "E": corrupted.target_E}
        masks = {"X": corrupted.node_mask}
        position_masks = build_group_position_masks(targets, masks, layout.ptr)
        common_student = common_dfm_per_graph(
            student_logits,
            targets,
            masks,
            layout.ptr,
            edge_weight=self.edge_weight,
            position_masks=position_masks,
        )
        student_preference, eligible = preference_dfm_per_graph(
            student_logits,
            targets,
            masks,
            layout.ptr,
            edge_weight=self.edge_weight,
            position_masks=position_masks,
        )
        _assert_condition(
            torch.all(eligible == cached_eligible),
            "runtime preference eligibility differs from batch supervision",
            AssertionError,
        )
        detached_reference = {"X": reference_x.detach(), "E": reference_e.detach()}
        reference_preference, reference_eligible = preference_dfm_per_graph(
            detached_reference,
            targets,
            masks,
            layout.ptr,
            edge_weight=self.edge_weight,
            position_masks=position_masks,
        )
        _assert_condition(
            torch.all(eligible == reference_eligible),
            "student and reference preference eligibility differ",
            AssertionError,
        )
        if self.preservation_mode in {"reference_kl", "reference_kl_shift"}:
            anchor_parts = full_reference_kl_per_graph(
                student_logits, detached_reference, masks, edge_weight=self.edge_weight
            )
        else:
            anchor_parts = common_reference_kl_per_graph(
                student_logits, detached_reference, position_masks, edge_weight=self.edge_weight
            )

        per_group_common_dfm = _endpoint_reduce(common_student.total, endpoint_probability, layout)
        per_group_anchor_kl = _endpoint_reduce(anchor_parts.total, endpoint_probability, layout)
        common_dfm = _weighted_mean(per_group_common_dfm, group_weight, common_student.total)
        anchor_kl = _weighted_mean(per_group_anchor_kl, group_weight, anchor_parts.total)
        common_objective = _training_reduce(
            per_group_common_dfm,
            group_weight,
            batch.objective_normalization,
            common_dfm,
        )
        anchor_objective = _training_reduce(
            per_group_anchor_kl,
            group_weight,
            batch.objective_normalization,
            anchor_kl,
        )
        per_group_full_reference_kl_guard_excess = (
            per_group_anchor_kl - self.full_reference_kl_budget
        ).clamp_min(0.0)
        full_reference_kl_guard = _weighted_mean(
            per_group_full_reference_kl_guard_excess,
            group_weight,
            anchor_parts.total,
        )
        full_reference_kl_guard_objective = _training_reduce(
            per_group_full_reference_kl_guard_excess,
            group_weight,
            batch.objective_normalization,
            full_reference_kl_guard,
        )
        full_reference_kl_guard_component = (
            self.full_reference_kl_guard_weight
            * full_reference_kl_guard_objective
        )

        positive_student, positive_probability, positive_count = positive_full_dfm_per_graph(
            student_logits,
            targets,
            masks,
            layout.ptr,
            labels,
            teacher_probability,
            edge_weight=self.edge_weight,
        )
        positive_reference, reference_positive_probability, reference_positive_count = positive_full_dfm_per_graph(
            detached_reference,
            targets,
            masks,
            layout.ptr,
            labels,
            teacher_probability,
            edge_weight=self.edge_weight,
        )
        _assert_condition(
            torch.all(positive_probability == reference_positive_probability)
            & torch.all(positive_count == reference_positive_count),
            "positive teacher normalization differs between student and reference",
            AssertionError,
        )
        positive_selected = positive_count > 0.0
        positive_full_dfm = _weighted_mean(positive_student.total, group_weight, positive_student.total, positive_selected)
        positive_reference_dfm = _weighted_mean(positive_reference.total, group_weight, positive_student.total, positive_selected)
        positive_dfm_objective = _training_reduce(
            positive_student.total,
            group_weight,
            batch.objective_normalization,
            positive_full_dfm,
        )

        negative_labels = 1 - labels
        negative_student, negative_probability, negative_count = positive_full_dfm_per_graph(
            student_logits,
            targets,
            masks,
            layout.ptr,
            negative_labels,
            teacher_probability,
            edge_weight=self.edge_weight,
        )
        negative_reference, reference_negative_probability, reference_negative_count = positive_full_dfm_per_graph(
            detached_reference,
            targets,
            masks,
            layout.ptr,
            negative_labels,
            teacher_probability,
            edge_weight=self.edge_weight,
        )
        _assert_condition(
            torch.all(negative_probability == reference_negative_probability)
            & torch.all(negative_count == reference_negative_count),
            "negative teacher normalization differs between student and reference",
            AssertionError,
        )
        negative_selected = negative_count > 0.0
        negative_full_dfm = _weighted_mean(negative_student.total, group_weight, negative_student.total, negative_selected)
        negative_reference_dfm = _weighted_mean(negative_reference.total, group_weight, negative_student.total, negative_selected)
        negative_dfm_objective = _training_reduce(
            negative_student.total,
            group_weight,
            batch.objective_normalization,
            negative_full_dfm,
        )
        per_group_negative_reverse_shift = negative_student.total - negative_reference.total
        reverse_argument = (
            self.negative_reverse_margin - per_group_negative_reverse_shift
        ) / self.negative_reverse_temperature
        per_group_negative_reverse_objective = (
            self.negative_reverse_temperature
            * F.softplus(reverse_argument)
            * negative_selected.float()
        )
        per_group_negative_reverse_satisfaction = (
            (per_group_negative_reverse_shift >= self.negative_reverse_margin).float()
            * negative_selected.float()
        )
        negative_reverse_objective = _training_reduce(
            per_group_negative_reverse_objective,
            group_weight,
            batch.objective_normalization,
            _weighted_mean(
                per_group_negative_reverse_objective,
                group_weight,
                negative_student.total,
                negative_selected,
            ),
        )

        full_student = full_dfm_per_graph(student_logits, targets, masks, edge_weight=self.edge_weight)
        full_reference = full_dfm_per_graph(detached_reference, targets, masks, edge_weight=self.edge_weight)
        raw_route_gain_score = full_reference.total - full_student.total
        route_gain_score, route_score_scale = _calibrate_reference_relative_score(
            raw_route_gain_score,
            full_reference.total,
            layout,
            self.candidate_score_calibration,
            self.candidate_score_scale_floor,
        )
        route_pairwise = _route_pairwise_statistics(
            route_gain_score,
            labels,
            eligible,
            layout,
            self.route_ranking_margin,
            self.pairwise_huber_delta,
        )
        route_selected = route_pairwise["selected"]
        route_ranking_loss = _weighted_mean(
            route_pairwise["pairwise_loss"], group_weight, full_student.total, route_selected
        )
        route_ranking_objective = _training_reduce(
            route_pairwise["pairwise_loss"],
            group_weight,
            batch.objective_normalization,
            route_ranking_loss,
        )
        per_group_full_student_dfm = torch.segment_reduce(
            full_student.total.float(), "mean", lengths=layout.sizes
        )
        per_group_full_reference_dfm = torch.segment_reduce(
            full_reference.total.float(), "mean", lengths=layout.sizes
        )
        per_group_route_gain_score_mean = torch.segment_reduce(
            route_gain_score, "mean", lengths=layout.sizes
        )
        route_gain_centered = route_gain_score - per_group_route_gain_score_mean.index_select(
            0, layout.index
        )
        per_group_route_gain_score_std = torch.sqrt(
            torch.segment_reduce(
                route_gain_centered.square(), "mean", lengths=layout.sizes
            ).clamp_min(0.0)
        )
        per_group_route_gain_score_max_abs = torch.segment_reduce(
            route_gain_score.abs(), "max", lengths=layout.sizes
        )

        if self.preservation_mode in {"reference_kl", "reference_kl_shift"}:
            per_group_common_kl_excess = per_group_anchor_kl * 0.0
            trust_region = per_group_anchor_kl.sum() * 0.0
            trust_region_objective = per_group_anchor_kl.sum() * 0.0
        else:
            per_group_common_kl_excess = squared_hinge(per_group_anchor_kl, self.common_kl_budget)
            trust_region = _weighted_mean(per_group_common_kl_excess, group_weight, anchor_parts.total)
            trust_region_objective = _training_reduce(
                per_group_common_kl_excess,
                group_weight,
                batch.objective_normalization,
                trust_region,
            )

        raw_improvement_score = reference_preference.total - student_preference.total
        improvement_score, candidate_score_scale = _calibrate_reference_relative_score(
            raw_improvement_score,
            reference_preference.total,
            layout,
            self.candidate_score_calibration,
            self.candidate_score_scale_floor,
        )
        listwise = _listwise_statistics(
            improvement_score,
            teacher_probability,
            labels,
            eligible,
            layout,
            self.temperature,
        )
        prediction_probability = listwise["prediction_probability"]
        per_group_student_candidate_dfm = torch.segment_reduce(
            student_preference.total.float(), "mean", lengths=layout.sizes
        )
        per_group_reference_candidate_dfm = torch.segment_reduce(
            reference_preference.total.float(), "mean", lengths=layout.sizes
        )
        group_listwise_ce = listwise["listwise_ce"]
        all_eligible = {"weights": group_weight, "zero_source": student_preference.total, "selected": eligible}
        listwise_ce = _weighted_mean(group_listwise_ce, **all_eligible)
        legacy_listwise_objective = _training_reduce(
            group_listwise_ce,
            group_weight,
            batch.objective_normalization,
            _weighted_mean(group_listwise_ce, group_weight, student_preference.total),
        )

        positive = labels == 1
        negative = labels == 0
        teacher_positive_mass = torch.segment_reduce(
            torch.where(positive, teacher_probability, torch.zeros_like(teacher_probability)), "sum", lengths=layout.sizes
        )
        predicted_positive_mass = torch.segment_reduce(
            torch.where(positive, prediction_probability, torch.zeros_like(prediction_probability)), "sum", lengths=layout.sizes
        )
        teacher_negative_mass = torch.segment_reduce(
            torch.where(negative, teacher_probability, torch.zeros_like(teacher_probability)), "sum", lengths=layout.sizes
        )
        predicted_negative_mass = torch.segment_reduce(
            torch.where(negative, prediction_probability, torch.zeros_like(prediction_probability)), "sum", lengths=layout.sizes
        )
        positive_shift = predicted_positive_mass - teacher_positive_mass
        negative_shift = teacher_negative_mass - predicted_negative_mass
        positive_label_count = torch.segment_reduce(positive.float(), "sum", lengths=layout.sizes)
        negative_label_count = torch.segment_reduce(negative.float(), "sum", lengths=layout.sizes)
        shift_selected = eligible & (positive_label_count > 0.0) & (negative_label_count > 0.0)
        per_group_probability_shift = 0.5 * (
            (self.common_kl_budget - positive_shift).clamp_min(0.0).square()
            + (self.common_kl_budget - negative_shift).clamp_min(0.0).square()
        ) * shift_selected.float()
        probability_shift_objective = _training_reduce(
            per_group_probability_shift,
            group_weight,
            batch.objective_normalization,
            _weighted_mean(per_group_probability_shift, group_weight, student_preference.total, shift_selected),
        )

        structure = _structure_pairwise_statistics(
            improvement_score,
            teacher_probability,
            labels,
            eligible,
            structure_group,
            layout,
            self.temperature,
            self.pairwise_huber_delta,
            self.pairwise_margin_max,
        )
        structure_selected = structure_group & eligible
        metric_selected = metric_group & eligible
        structure_pairwise_loss = _weighted_mean(
            structure["pairwise_loss"], branch_weight, student_preference.total, structure_selected
        )
        metric_listwise_kl = _weighted_mean(
            listwise["teacher_kl"] * metric_selected.float(),
            branch_weight,
            student_preference.total,
            metric_selected,
        )
        if self.metric_guidance_mode == "label_pairwise":
            metric_pairwise = _metric_label_pairwise_statistics(
                improvement_score,
                labels,
                eligible,
                metric_group,
                layout,
                self.metric_pairwise_margin,
            )
        else:
            # Preserve the default teacher-listwise execution path and its
            # allocation profile: pairwise tensors are materialized only for
            # an explicit label-pairwise experiment.
            zero_group = metric_selected.float() * student_preference.total.sum() * 0.0
            metric_pairwise = {
                "pairwise_loss": zero_group,
                "pair_count": zero_group,
                "score_margin": zero_group,
                "pair_accuracy": zero_group,
                "selected": torch.zeros_like(metric_selected),
            }
        metric_pairwise_selected = metric_pairwise["selected"]
        metric_pairwise_loss = _weighted_mean(
            metric_pairwise["pairwise_loss"],
            branch_weight,
            student_preference.total,
            metric_pairwise_selected,
        )
        metric_teacher_listwise_objective = _training_reduce(
            listwise["teacher_kl"] * metric_selected.float(),
            branch_weight,
            batch.objective_normalization,
            metric_listwise_kl,
        )
        metric_label_pairwise_objective = _training_reduce(
            metric_pairwise["pairwise_loss"],
            branch_weight,
            batch.objective_normalization,
            metric_pairwise_loss,
        )
        if self.metric_guidance_mode in {
            "candidate_teacher_listwise_kl",
            "teacher_listwise_kl",
        }:
            metric_rank_objective = metric_teacher_listwise_objective
        elif self.metric_guidance_mode == "label_pairwise":
            metric_rank_objective = metric_label_pairwise_objective
        else:
            # Metric teacher KL is an explicit ablation, not an assumed
            # requirement.  The established positive-DFM and route-ranking
            # path remains the neutral default until metric guidance has
            # independent efficacy evidence.
            metric_rank_objective = metric_teacher_listwise_objective * 0.0
        if self.hybrid_branch_objective:
            structure_objective = _training_reduce(
                structure["pairwise_loss"],
                branch_weight,
                batch.objective_normalization,
                structure_pairwise_loss,
            )
            metric_objective = metric_rank_objective
            guidance_objective = (
                self.structure_guidance_weight * structure_objective
                + self.metric_guidance_weight * metric_objective
            )
            listwise_objective = guidance_objective
        else:
            structure_objective = structure_pairwise_loss
            metric_objective = legacy_listwise_objective
            guidance_objective = legacy_listwise_objective
            listwise_objective = legacy_listwise_objective

        legacy_anchor_component = self.anchor_weight * anchor_objective if self.preservation_mode == "legacy" else anchor_objective * 0.0
        positive_component = self.positive_dfm_weight * positive_dfm_objective if self.preservation_mode in {"positive_dfm", "positive_trust"} else positive_dfm_objective * 0.0
        trust_component = self.trust_region_weight * trust_region_objective if self.preservation_mode in {"true_trust", "positive_trust"} else trust_region_objective * 0.0
        contrastive_loss = (
            self.common_weight * common_objective
            + self.listwise_weight * listwise_objective
            + legacy_anchor_component
            + positive_component
            + trust_component
        )
        metric_rank_component = (
            self.listwise_weight
            * self.metric_guidance_weight
            * metric_rank_objective
        )
        if self.candidate_supervision_mode == "contrastive":
            candidate_objective = contrastive_loss
        elif self.candidate_supervision_mode == "positive_sft":
            candidate_objective = self.positive_supervision_weight * positive_dfm_objective
        elif self.candidate_supervision_mode == "positive_rank":
            candidate_objective = (
                self.positive_supervision_weight * positive_dfm_objective
                + self.route_ranking_weight * route_ranking_objective
                + metric_rank_component
            )
        elif self.candidate_supervision_mode == "positive_rank_hybrid":
            candidate_objective = (
                self.positive_supervision_weight * positive_dfm_objective
                + self.route_ranking_weight * route_ranking_objective
                + self.listwise_weight * guidance_objective
            )
        elif self.candidate_supervision_mode == "positive_negative_reverse":
            candidate_objective = (
                self.positive_supervision_weight * positive_dfm_objective
                + self.negative_reverse_weight * negative_reverse_objective
            )
        else:
            candidate_objective = (
                self.positive_supervision_weight * positive_dfm_objective
                + self.negative_supervision_weight * negative_dfm_objective
            )
        if self.full_reference_kl_guard_weight > 0.0:
            candidate_objective = (
                candidate_objective + full_reference_kl_guard_component
            )

        scalar_metrics = {
            "common_dfm": common_dfm,
            "listwise_ce": listwise_ce,
            "guidance_objective": guidance_objective,
            "structure_pairwise_loss": structure_pairwise_loss,
            "structure_objective": structure_objective,
            "metric_listwise_ce": _weighted_mean(group_listwise_ce * metric_selected.float(), branch_weight, student_preference.total, metric_selected),
            "metric_listwise_kl": metric_listwise_kl,
            # This is KL from precomputed metric teacher q to a Student DFM
            # candidate-score distribution p.  It does not evaluate metrics
            # on newly sampled Student routes; retain the old key only as a
            # logging compatibility alias.
            "metric_candidate_teacher_kl": metric_listwise_kl,
            "metric_pairwise_loss": metric_pairwise_loss,
            "metric_pairwise_margin": _weighted_mean(
                metric_pairwise["score_margin"],
                branch_weight,
                student_preference.total,
                metric_pairwise_selected,
            ),
            "metric_pairwise_accuracy": _weighted_mean(
                metric_pairwise["pair_accuracy"],
                branch_weight,
                student_preference.total,
                metric_pairwise_selected,
            ),
            "metric_pairwise_group_count": metric_pairwise_selected.sum(dtype=torch.int64),
            "metric_objective": metric_objective,
            "metric_rank_objective": metric_rank_objective,
            "metric_rank_component": metric_rank_component,
            "anchor_kl": anchor_kl,
            "full_reference_kl_guard": full_reference_kl_guard,
            "full_reference_kl_guard_objective": full_reference_kl_guard_objective,
            "full_reference_kl_guard_component": full_reference_kl_guard_component,
            "common_objective": common_objective,
            "listwise_objective": listwise_objective,
            "anchor_objective": anchor_objective,
            "teacher_entropy": _weighted_mean(listwise["teacher_entropy"], **all_eligible),
            "teacher_kl": _weighted_mean(listwise["teacher_kl"], **all_eligible),
            "score_margin": _weighted_mean(listwise["score_margin"], **all_eligible),
            "pair_accuracy": _weighted_mean(listwise["pair_accuracy"], **all_eligible),
            "top1_accuracy": _weighted_mean(listwise["top1_accuracy"], **all_eligible),
            "predicted_entropy": _weighted_mean(listwise["predicted_entropy"], **all_eligible),
            "score_mean": _weighted_mean(listwise["score_mean"], **all_eligible),
            "score_std": _weighted_mean(listwise["score_std"], **all_eligible),
            "score_max": _weighted_mean(listwise["score_max"], **all_eligible),
            "raw_score_mean": _weighted_mean(
                listwise["score_mean"] * candidate_score_scale,
                **all_eligible,
            ),
            "raw_score_std": _weighted_mean(
                listwise["score_std"] * candidate_score_scale,
                **all_eligible,
            ),
            "raw_score_max": _weighted_mean(
                listwise["score_max"] * candidate_score_scale,
                **all_eligible,
            ),
            "candidate_score_scale": _weighted_mean(
                candidate_score_scale,
                group_weight,
                student_preference.total,
                eligible,
            ),
            "student_candidate_dfm": _weighted_mean(
                per_group_student_candidate_dfm,
                group_weight,
                student_preference.total,
                eligible,
            ),
            "reference_candidate_dfm": _weighted_mean(
                per_group_reference_candidate_dfm,
                group_weight,
                student_preference.total,
                eligible,
            ),
            "positive_full_dfm": positive_full_dfm,
            "positive_full_dfm_reference": positive_reference_dfm,
            "positive_full_dfm_minus_reference": positive_full_dfm - positive_reference_dfm,
            "positive_dfm_objective": positive_dfm_objective,
            "negative_full_dfm": negative_full_dfm,
            "negative_full_dfm_reference": negative_reference_dfm,
            "negative_full_dfm_minus_reference": negative_full_dfm - negative_reference_dfm,
            "negative_dfm_objective": negative_dfm_objective,
            "negative_reverse_shift": _weighted_mean(per_group_negative_reverse_shift, group_weight, negative_student.total, negative_selected),
            "negative_reverse_objective": negative_reverse_objective,
            "negative_reverse_margin_satisfaction": _weighted_mean(per_group_negative_reverse_satisfaction, group_weight, negative_student.total, negative_selected),
            "route_ranking_loss": route_ranking_loss,
            "route_ranking_objective": route_ranking_objective,
            "route_gain_margin": _weighted_mean(route_pairwise["gain_margin"], group_weight, full_student.total, route_selected),
            "raw_route_gain_margin": _weighted_mean(
                route_pairwise["gain_margin"] * route_score_scale,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "route_score_scale": _weighted_mean(
                route_score_scale,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "student_full_dfm": _weighted_mean(
                per_group_full_student_dfm,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "reference_full_dfm": _weighted_mean(
                per_group_full_reference_dfm,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "route_gain_score_mean": _weighted_mean(
                per_group_route_gain_score_mean,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "route_gain_score_std": _weighted_mean(
                per_group_route_gain_score_std,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "route_gain_score_max_abs": _weighted_mean(
                per_group_route_gain_score_max_abs,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "raw_route_gain_score_mean": _weighted_mean(
                per_group_route_gain_score_mean * route_score_scale,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "raw_route_gain_score_std": _weighted_mean(
                per_group_route_gain_score_std * route_score_scale,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "raw_route_gain_score_max_abs": _weighted_mean(
                per_group_route_gain_score_max_abs * route_score_scale,
                group_weight,
                full_student.total,
                route_selected,
            ),
            "trust_region": trust_region,
            "trust_region_objective": trust_region_objective,
            "positive_probability_shift": _weighted_mean(positive_shift * shift_selected.float(), group_weight, student_preference.total, shift_selected),
            "negative_probability_shift": _weighted_mean(negative_shift * shift_selected.float(), group_weight, student_preference.total, shift_selected),
            "probability_shift_objective": probability_shift_objective,
            "eligible_group_count": eligible.sum(dtype=torch.int64),
            "structure_group_count": structure_selected.sum(dtype=torch.int64),
            "metric_group_count": metric_selected.sum(dtype=torch.int64),
            "candidate_supervision_objective": candidate_objective,
        }
        group_metrics = {
            "common_dfm": per_group_common_dfm,
            "anchor_kl": per_group_anchor_kl,
            "full_reference_kl_guard_excess": per_group_full_reference_kl_guard_excess,
            "common_kl_excess": per_group_common_kl_excess,
            "listwise_ce": group_listwise_ce,
            "teacher_entropy": listwise["teacher_entropy"],
            "teacher_kl": listwise["teacher_kl"],
            "score_margin": listwise["score_margin"],
            "pair_accuracy": listwise["pair_accuracy"],
            "top1_accuracy": listwise["top1_accuracy"],
            "predicted_entropy": listwise["predicted_entropy"],
            "score_mean": listwise["score_mean"],
            "score_std": listwise["score_std"],
            "score_max": listwise["score_max"],
            "raw_score_mean": listwise["score_mean"] * candidate_score_scale,
            "raw_score_std": listwise["score_std"] * candidate_score_scale,
            "raw_score_max": listwise["score_max"] * candidate_score_scale,
            "candidate_score_scale": candidate_score_scale,
            "student_candidate_dfm": per_group_student_candidate_dfm,
            "reference_candidate_dfm": per_group_reference_candidate_dfm,
            "structure_pairwise_loss": structure["pairwise_loss"],
            "structure_pair_count": structure["pair_count"],
            "structure_target_margin": structure["target_margin"],
            "structure_margin_satisfaction": structure["margin_satisfaction"],
            "metric_listwise_ce": group_listwise_ce * metric_selected.float(),
            "metric_listwise_kl": listwise["teacher_kl"] * metric_selected.float(),
            "metric_candidate_teacher_kl": listwise["teacher_kl"]
            * metric_selected.float(),
            "metric_pairwise_loss": metric_pairwise["pairwise_loss"],
            "metric_pair_count": metric_pairwise["pair_count"],
            "metric_pairwise_margin": metric_pairwise["score_margin"],
            "metric_pairwise_accuracy": metric_pairwise["pair_accuracy"],
            "positive_full_dfm": positive_student.total,
            "positive_full_dfm_reference": positive_reference.total,
            "positive_candidate_count": positive_count,
            "negative_full_dfm": negative_student.total,
            "negative_full_dfm_reference": negative_reference.total,
            "negative_candidate_count": negative_count,
            "negative_reverse_shift": per_group_negative_reverse_shift,
            "negative_reverse_objective": per_group_negative_reverse_objective,
            "negative_reverse_margin_satisfaction": per_group_negative_reverse_satisfaction,
            "route_ranking_loss": route_pairwise["pairwise_loss"],
            "route_gain_margin": route_pairwise["gain_margin"],
            "raw_route_gain_margin": route_pairwise["gain_margin"] * route_score_scale,
            "route_score_scale": route_score_scale,
            "student_full_dfm": per_group_full_student_dfm,
            "reference_full_dfm": per_group_full_reference_dfm,
            "route_gain_score_mean": per_group_route_gain_score_mean,
            "route_gain_score_std": per_group_route_gain_score_std,
            "route_gain_score_max_abs": per_group_route_gain_score_max_abs,
            "raw_route_gain_score_mean": per_group_route_gain_score_mean
            * route_score_scale,
            "raw_route_gain_score_std": per_group_route_gain_score_std
            * route_score_scale,
            "raw_route_gain_score_max_abs": per_group_route_gain_score_max_abs
            * route_score_scale,
            "positive_probability_shift": positive_shift * shift_selected.float(),
            "negative_probability_shift": negative_shift * shift_selected.float(),
            "probability_shift": per_group_probability_shift,
        }
        return GuidedObjectiveOutput(
            loss=candidate_objective,
            per_candidate_score=improvement_score,
            per_candidate_prediction_probability=prediction_probability,
            eligible_group_mask=eligible,
            structure_group_mask=structure_group,
            metric_group_mask=metric_group,
            metrics=self._freeze_mapping(scalar_metrics),
            per_group=self._freeze_mapping(group_metrics),
        )


__all__ = [
    "GuidanceObjective",
    "GuidanceObjectiveConfig",
    "GuidedObjectiveBatch",
    "GuidedObjectiveOutput",
    "ObjectiveNormalization",
]
