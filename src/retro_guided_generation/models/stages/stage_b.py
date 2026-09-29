"""Unified one-branch Stage B with explicit Stage A projector transfer.

Stage B keeps the base GraphDiT frozen, carries only the text-projector side of
Stage A into a fresh guided student, and leaves every generation cross-attention
block at its normal zero-output initialization.  This prevents an accidental
copy of Stage A's generation path while retaining its text-space alignment.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
import math
import os
from pathlib import Path

import torch
from torch import Tensor, nn

from ...checkpoints import load_legacy_state_dict
from ..dfm import (
    CorruptedGraphBatch,
    CoupledSelectorField,
    PredefinedSamplingConfig,
    PositiveRecoveryConfig,
    build_coupled_predefined_corruption,
    full_dfm_per_graph,
    full_reference_kl_per_graph,
    positive_recovery_time,
    sample_predefined_graph,
    sample_coupled_selector_field,
    sample_predefined_student_transition,
)
from ..graphdit import (
    GraphDiT,
    GraphDiTConfig,
    GraphDiTStateLoadReport,
    GroupedTokenGuidanceAdapter,
    TokenCrossAttentionGraphDiT,
    TokenGuidanceAdapter,
    load_graphdit_state_dict,
)
from ..guidance.architecture import resolve_guidance_architecture
from ..objectives import (
    GuidanceObjective,
    GuidanceObjectiveConfig,
    GuidedObjectiveOutput,
    SemanticAlignmentObjective,
    SemanticAlignmentOutput,
)
from .stage_a import (
    AdapterOnlyStageA,
    GraphConditionBatch,
    GuidedStageABatch,
    _expected_group_index,
    _validate_graphs,
)


def standard_stage_b_objective(
    selected_branch: str = "structure",
    *,
    metric_guidance_mode: str = "none",
) -> GuidanceObjectiveConfig:
    """Return the calibrated unified Stage-B candidate objective.

    The new shared framework keeps the established reference-relative route
    ordering, but applies a detached per-group reference scale before the
    positive-rank margin.  The selected data branch is one-hot.  Metric
    teacher/listwise supervision is a separate explicit option: a metric-only
    run may keep ``metric_guidance_mode='none'`` and still train through the
    positive-DFM, route-ranking, semantic-alignment, reference-KL, and REC1
    terms.
    """

    if selected_branch not in {"structure", "metric"}:
        raise ValueError("selected_branch must be structure or metric")
    normalized_metric_mode = str(metric_guidance_mode).strip().lower()
    if selected_branch == "structure" and normalized_metric_mode != "none":
        raise ValueError("structure Stage B cannot enable a metric guidance mode")

    return GuidanceObjectiveConfig(
        edge_weight=10.0,
        temperature=0.5,
        common_weight=0.0,
        listwise_weight=0.05,
        anchor_weight=0.0,
        hybrid_branch_objective=True,
        structure_guidance_weight=1.0 if selected_branch == "structure" else 0.0,
        metric_guidance_weight=1.0 if selected_branch == "metric" else 0.0,
        metric_guidance_mode=normalized_metric_mode,
        candidate_score_calibration="group_reference_mean",
        candidate_score_scale_floor=1.0,
        preservation_mode="reference_kl",
        positive_dfm_weight=0.0,
        trust_region_weight=1.0,
        common_kl_budget=0.0,
        candidate_supervision_mode="positive_rank",
        positive_supervision_weight=0.1,
        negative_supervision_weight=0.0,
        negative_reverse_weight=0.0,
        route_ranking_weight=0.05,
        route_ranking_margin=0.1,
    )


@dataclass(frozen=True, slots=True)
class SemanticAlignmentConfig:
    """Dataset-neutral frozen-route semantic alignment configuration.

    The route embeddings themselves remain a data-adapter concern.  This
    config only describes how an already-frozen candidate semantic vector is
    compared with grouped text-projector vectors.
    """

    weight: float = 0.05
    selected_branch: str = "structure"
    pair_weight: float = 0.5
    pair_margin: float = 0.1
    endpoint_probability_floor: float = 1.0e-8
    logit_scale: float = 10.0

    def __post_init__(self) -> None:
        if self.selected_branch not in {"structure", "metric", "all"}:
            raise ValueError("semantic selected_branch must be structure, metric, or all")
        for name, value, strictly_positive in (
            ("weight", self.weight, True),
            ("pair_weight", self.pair_weight, False),
            ("pair_margin", self.pair_margin, False),
            ("endpoint_probability_floor", self.endpoint_probability_floor, True),
            ("logit_scale", self.logit_scale, True),
        ):
            number = float(value)
            if not math.isfinite(number) or (number <= 0.0 if strictly_positive else number < 0.0):
                qualifier = "positive" if strictly_positive else "non-negative"
                raise ValueError(f"semantic {name} must be finite and {qualifier}")


@dataclass(frozen=True, slots=True)
class TeacherSemanticLossShareControllerConfig:
    """One-sided controller for metric-teacher/semantic loss dominance.

    The controller observes the globally reduced, optimizer-update-level loss
    decomposition.  It never increases either configured coefficient: healthy
    and warning regions retain the current scale, while a sustained share over
    ``trigger_share`` lowers the common teacher/semantic scale toward
    ``target_share``.  The two terms share one scale because they are
    dataset-specific alignment teachers occupying the same loss-budget slot.
    """

    enabled: bool = False
    healthy_max_share: float = 0.40
    observe_max_share: float = 0.45
    trigger_share: float = 0.50
    target_share: float = 0.45
    ema_decay: float = 0.99
    warmup_updates: int = 1
    minimum_scale: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("teacher/semantic loss-share controller enabled must be a bool")
        values = {
            "healthy_max_share": float(self.healthy_max_share),
            "observe_max_share": float(self.observe_max_share),
            "trigger_share": float(self.trigger_share),
            "target_share": float(self.target_share),
            "ema_decay": float(self.ema_decay),
            "minimum_scale": float(self.minimum_scale),
        }
        if any(not math.isfinite(value) for value in values.values()):
            raise ValueError("teacher/semantic controller values must be finite")
        if not (
            0.0
            < values["healthy_max_share"]
            < values["observe_max_share"]
            < values["trigger_share"]
            < 1.0
        ):
            raise ValueError(
                "teacher/semantic share thresholds must satisfy "
                "0 < healthy < observe < trigger < 1"
            )
        if not 0.0 < values["target_share"] < values["trigger_share"]:
            raise ValueError(
                "teacher/semantic target_share must lie strictly between 0 and trigger_share"
            )
        if not 0.0 <= values["ema_decay"] < 1.0:
            raise ValueError("teacher/semantic ema_decay must lie in [0, 1)")
        if (
            isinstance(self.warmup_updates, bool)
            or not isinstance(self.warmup_updates, int)
            or self.warmup_updates < 1
        ):
            raise ValueError("teacher/semantic warmup_updates must be a positive integer")
        if not 0.0 <= values["minimum_scale"] <= 1.0:
            raise ValueError("teacher/semantic minimum_scale must lie in [0, 1]")

    def resolved_contract(self) -> dict[str, object]:
        """Return the auditable state-machine boundary definitions."""

        return {
            "enabled": self.enabled,
            "healthy_share": f"< {float(self.healthy_max_share):.2f}",
            "observe_share": (
                f"[{float(self.healthy_max_share):.2f}, "
                f"{float(self.observe_max_share):.2f})"
            ),
            "warning_share": (
                f"[{float(self.observe_max_share):.2f}, "
                f"{float(self.trigger_share):.2f}]"
            ),
            "adaptive_share": f"> {float(self.trigger_share):.2f}",
            "target_share": float(self.target_share),
            "ema_decay": float(self.ema_decay),
            "warmup_updates": self.warmup_updates,
            "minimum_scale": float(self.minimum_scale),
            "policy": "one_sided_proportional_downscale_v1",
        }


@dataclass(frozen=True, slots=True)
class RolloutReferenceKLConfig:
    """Bounded on-policy replay contract for Stage-B reference KL.

    The replay states are sampled without autograd from the current student at
    a fixed finite schedule.  On normal optimization forwards, only the
    student is differentiated while the frozen base supplies the reference
    distribution on exactly those detached states.
    """

    enabled: bool = False
    weight: float = 1.0
    refresh_interval_steps: int = 100
    max_groups: int = 8
    trajectories_per_group: int = 1
    sampling_steps: int = 50
    trace_points: int = 4
    seed: int = 73_027

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("rollout reference-KL enabled must be a bool")
        if not math.isfinite(float(self.weight)) or float(self.weight) < 0.0:
            raise ValueError("rollout reference-KL weight must be finite and non-negative")
        for name in (
            "refresh_interval_steps",
            "max_groups",
            "trajectories_per_group",
            "sampling_steps",
            "trace_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"rollout reference-KL {name} must be a positive integer")
        if self.trace_points > self.sampling_steps:
            raise ValueError("rollout reference-KL trace_points cannot exceed sampling_steps")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("rollout reference-KL seed must be an integer")
        if self.enabled and float(self.weight) <= 0.0:
            raise ValueError("enabled rollout reference-KL requires a positive weight")


@dataclass(frozen=True, slots=True)
class GuidanceResidualScheduleConfig:
    """Dataset-neutral optimizer-update controller for guided residuals.

    ``completed_optimizer_updates`` is the number of updates completed before
    the current update.  A run with ``U`` updates therefore evaluates this
    controller at ``0, ..., U - 1``.  The default fixed value of ``1.0`` is the
    exact backward-compatible behavior for configs without a named recipe.
    """

    mode: str = "fixed"
    start: float = 1.0
    end: float = 1.0

    def __post_init__(self) -> None:
        if self.normalized_mode not in {"fixed", "cosine"}:
            raise ValueError("guidance residual schedule mode must be fixed or cosine")
        for name, value in (("start", self.start), ("end", self.end)):
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(
                    f"guidance residual schedule {name} must be finite and lie in [0, 1]"
                )
        if self.normalized_mode == "fixed" and not math.isclose(
            float(self.start), float(self.end), rel_tol=0.0, abs_tol=0.0
        ):
            raise ValueError("fixed guidance residual schedule requires start == end")

    @property
    def normalized_mode(self) -> str:
        return str(self.mode).strip().lower()

    def scale(
        self,
        completed_optimizer_updates: int,
        *,
        total_optimizer_updates: int,
    ) -> float:
        """Return the residual scale for one not-yet-completed update."""

        if (
            isinstance(total_optimizer_updates, bool)
            or not isinstance(total_optimizer_updates, int)
            or total_optimizer_updates < 1
        ):
            raise ValueError("total_optimizer_updates must be a positive integer")
        if (
            isinstance(completed_optimizer_updates, bool)
            or not isinstance(completed_optimizer_updates, int)
            or not 0 <= completed_optimizer_updates < total_optimizer_updates
        ):
            raise ValueError(
                "completed_optimizer_updates must lie in "
                "[0, total_optimizer_updates)"
            )
        if self.normalized_mode == "fixed":
            return float(self.start)
        progress = completed_optimizer_updates / max(1, total_optimizer_updates - 1)
        return float(self.end) + 0.5 * (float(self.start) - float(self.end)) * (
            1.0 + math.cos(math.pi * progress)
        )

    def resolved_contract(self, *, total_optimizer_updates: int) -> dict[str, object]:
        """Return the immutable, recovery-comparable schedule contract."""

        first = self.scale(0, total_optimizer_updates=total_optimizer_updates)
        final = self.scale(
            total_optimizer_updates - 1,
            total_optimizer_updates=total_optimizer_updates,
        )
        midpoint = float(self.end) + 0.5 * (
            float(self.start) - float(self.end)
        )
        return {
            "coordinate": "completed_optimizer_updates_before_current_update",
            "mode": self.normalized_mode,
            "start": float(self.start),
            "end": float(self.end),
            "total_optimizer_updates": total_optimizer_updates,
            "first_update_scale": first,
            "midpoint_coordinate": (total_optimizer_updates - 1) / 2.0,
            "midpoint_scale": midpoint,
            "final_update_scale": final,
        }


def standard_stage_b_semantic_alignment(
    selected_branch: str = "structure",
) -> SemanticAlignmentConfig:
    """Return the unified frozen-route semantic-alignment defaults."""

    return SemanticAlignmentConfig(
        weight=0.05,
        selected_branch=selected_branch,
        pair_weight=0.5,
        pair_margin=0.1,
        endpoint_probability_floor=1.0e-8,
        logit_scale=10.0,
    )


def standard_stage_b_teacher_semantic_loss_share_controller(
) -> TeacherSemanticLossShareControllerConfig:
    """Return the universal 40/45/50-percent teacher-share policy."""

    return TeacherSemanticLossShareControllerConfig(enabled=True)


def standard_stage_b_positive_recovery(
    selected_branch: str = "structure",
    *,
    budget_mode: str = "legacy_steps",
) -> PositiveRecoveryConfig:
    """Return the unified branch-matched REC1 schedule."""

    if selected_branch not in {"structure", "metric"}:
        raise ValueError("selected_branch must be structure or metric")
    if budget_mode not in {"epochs", "legacy_steps"}:
        raise ValueError("budget_mode must be epochs or legacy_steps")
    schedule = (
        {"warmup_epochs": "1/3", "ramp_epochs": "1/3"}
        if budget_mode == "epochs"
        else {"warmup_steps": 13_192, "ramp_steps": 13_192}
    )
    return PositiveRecoveryConfig(
        mode="student_rollin",
        weight=0.03,
        branch=selected_branch,
        sampling_steps=50,
        time_policy="low_t_mixture50_midpoint32_v1",
        time_policy_seed=97_031,
        time_min=0.1,
        time_max=0.9,
        time_bins=32,
        low_fraction=0.5,
        low_time_min=0.02,
        low_time_max=0.25,
        high_time_min=0.25,
        high_time_max=0.9,
        max_groups=8,
        **schedule,
    )


def standard_stage_b_rollout_reference_kl() -> RolloutReferenceKLConfig:
    """Return the unified on-policy reference-KL replay defaults."""

    return RolloutReferenceKLConfig(
        enabled=True,
        weight=1.0,
        refresh_interval_steps=100,
        max_groups=8,
        trajectories_per_group=1,
        sampling_steps=50,
        trace_points=4,
        seed=73_027,
    )


@dataclass(frozen=True, slots=True)
class StageBConfig:
    """Public construction contract for one-branch frozen-backbone Stage B."""

    graphdit: GraphDiTConfig
    guidance_architecture: str
    guidance_text_dim: int = 4096
    semantic_dim: int | None = None
    condition_dim: int | None = None
    objective: GuidanceObjectiveConfig = field(default_factory=standard_stage_b_objective)
    freeze_transferred_projectors: bool = False
    semantic_alignment: SemanticAlignmentConfig | None = field(
        default_factory=standard_stage_b_semantic_alignment
    )
    teacher_semantic_loss_share_controller: TeacherSemanticLossShareControllerConfig = field(
        default_factory=TeacherSemanticLossShareControllerConfig
    )
    positive_recovery: PositiveRecoveryConfig = field(
        default_factory=standard_stage_b_positive_recovery
    )
    rollout_reference_kl: RolloutReferenceKLConfig = field(
        default_factory=standard_stage_b_rollout_reference_kl
    )
    guidance_residual_schedule: GuidanceResidualScheduleConfig = field(
        default_factory=GuidanceResidualScheduleConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.graphdit, GraphDiTConfig):
            raise TypeError("graphdit must be GraphDiTConfig")
        if (
            isinstance(self.guidance_text_dim, bool)
            or not isinstance(self.guidance_text_dim, int)
            or self.guidance_text_dim < 1
        ):
            raise ValueError("guidance_text_dim must be a positive integer")
        if self.semantic_dim is not None and (
            isinstance(self.semantic_dim, bool)
            or not isinstance(self.semantic_dim, int)
            or self.semantic_dim < 1
        ):
            raise ValueError("semantic_dim must be a positive integer or None")
        if self.condition_dim is not None and (
            isinstance(self.condition_dim, bool)
            or not isinstance(self.condition_dim, int)
            or self.condition_dim < 1
        ):
            raise ValueError("condition_dim must be a positive integer or None")
        if not isinstance(self.guidance_architecture, str) or not self.guidance_architecture.strip():
            raise ValueError(
                "Stage B requires an explicitly confirmed registered guidance_architecture ID"
            )
        architecture = resolve_guidance_architecture(self.guidance_architecture)
        if not architecture.projector_groups:
            raise ValueError(
                "standard Stage B requires a projected-token architecture with "
                "transferable projector groups"
            )
        if self.resolved_semantic_dim != self.resolved_condition_dim:
            raise ValueError(
                "Stage-B projector width must equal generation-model conditioning k; "
                "rebuild Stage A and its native route-embedding cache at that k"
            )
        if not isinstance(self.objective, GuidanceObjectiveConfig):
            raise TypeError("objective must be GuidanceObjectiveConfig")
        if self.objective.hybrid_branch_objective is not True:
            raise ValueError("Stage B requires objective.hybrid_branch_objective=True")
        candidate_mode = str(self.objective.candidate_supervision_mode).strip().lower()
        if candidate_mode not in {"contrastive", "positive_rank", "positive_rank_hybrid"}:
            raise ValueError(
                "Stage B requires candidate_supervision_mode='contrastive', "
                "'positive_rank', or 'positive_rank_hybrid'"
            )
        if candidate_mode in {"positive_rank", "positive_rank_hybrid"} and float(self.objective.route_ranking_weight) <= 0.0:
            raise ValueError("Stage B positive-rank supervision requires a positive route_ranking_weight")
        metric_mode = str(self.objective.metric_guidance_mode).strip().lower()
        metric_weight = float(self.objective.metric_guidance_weight)
        if metric_mode != "none" and metric_weight <= 0.0:
            raise ValueError(
                "enabled Stage B metric guidance requires a positive metric_guidance_weight"
            )
        if len(self.active_guidance_branches) != 1:
            raise ValueError(
                "Stage B requires an exact one-hot structure or metric branch contract"
            )
        if not isinstance(self.freeze_transferred_projectors, bool):
            raise TypeError("freeze_transferred_projectors must be a bool")
        if self.semantic_alignment is not None and not isinstance(
            self.semantic_alignment, SemanticAlignmentConfig
        ):
            raise TypeError("semantic_alignment must be SemanticAlignmentConfig or None")
        if not isinstance(
            self.teacher_semantic_loss_share_controller,
            TeacherSemanticLossShareControllerConfig,
        ):
            raise TypeError(
                "teacher_semantic_loss_share_controller must be "
                "TeacherSemanticLossShareControllerConfig"
            )
        if not isinstance(self.positive_recovery, PositiveRecoveryConfig):
            raise TypeError("positive_recovery must be PositiveRecoveryConfig")
        if not isinstance(self.rollout_reference_kl, RolloutReferenceKLConfig):
            raise TypeError("rollout_reference_kl must be RolloutReferenceKLConfig")
        if not isinstance(
            self.guidance_residual_schedule, GuidanceResidualScheduleConfig
        ):
            raise TypeError(
                "guidance_residual_schedule must be GuidanceResidualScheduleConfig"
            )
        if self.positive_recovery.enabled and candidate_mode not in {"positive_rank", "positive_rank_hybrid"}:
            raise ValueError("positive recovery requires positive-rank candidate supervision")
        if self.positive_recovery.enabled and self.positive_recovery.branch not in {"metric", "structure"}:
            raise ValueError("Stage B positive recovery branch must be metric or structure")
        if (
            self.positive_recovery.enabled
            and self.positive_recovery.branch not in self.active_guidance_branches
        ):
            raise ValueError(
                "Stage B positive recovery branch is disabled by the guidance objective"
            )
        if (
            self.semantic_alignment is not None
            and self.semantic_alignment.selected_branch not in self.active_guidance_branches
        ):
            raise ValueError(
                "Stage B semantic alignment branch is disabled by the guidance objective"
            )
        if self.rollout_reference_kl.enabled and str(self.objective.preservation_mode).strip().lower() not in {
            "reference_kl",
            "reference_kl_shift",
        }:
            raise ValueError(
                "rollout reference-KL requires objective.preservation_mode='reference_kl'"
            )

    @property
    def resolved_semantic_dim(self) -> int:
        """Return the native Stage-A semantic-coordinate width."""

        return self.graphdit.x_dim if self.semantic_dim is None else self.semantic_dim

    @property
    def resolved_condition_dim(self) -> int:
        """Resolve generator conditioning width ``k`` from the backbone by default."""

        return self.graphdit.x_dim if self.condition_dim is None else self.condition_dim

    @property
    def active_guidance_branches(self) -> tuple[str, ...]:
        """Return branches that are allowed to contribute to this Stage B run."""

        branches: list[str] = []
        if float(self.objective.structure_guidance_weight) > 0.0:
            branches.append("structure")
        if float(self.objective.metric_guidance_weight) > 0.0:
            branches.append("metric")
        if not branches:  # pragma: no cover - objective weights sum to one
            raise AssertionError("Stage B has no active guidance branch")
        return tuple(branches)


def _projector_state_keys(adapter: nn.Module) -> tuple[str, ...]:
    """Identify projector-side adapter tensors independently of architecture IDs."""

    prefixes = (
        "text_norm.",
        "text_projection.",
        "projector_groups.",
        "null_token",
        "null_tokens.",
    )
    return tuple(
        name
        for name in adapter.state_dict()
        if name == "null_token" or name.startswith(prefixes)
    )


_PROJECTOR_PARAMETER_SUFFIXES = (
    "text_norm.weight",
    "text_norm.bias",
    "text_projection.weight",
    "text_projection.bias",
)

def _projector_parameter_groups(state: Mapping[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    """Collect transferable text projectors independently of adapter topology.

    A shared Stage A projector is one group with an empty prefix; a grouped
    Stage B adapter exposes one group per injection-layer set.  This allows a
    checkpoint handoff to broadcast a trained shared projector into fresh
    grouped projectors without transferring any generation cross-attention.
    """

    groups: dict[str, dict[str, Tensor]] = {}
    for name, value in state.items():
        if not isinstance(value, Tensor):
            continue
        for suffix in _PROJECTOR_PARAMETER_SUFFIXES:
            if name.endswith(suffix):
                prefix = name[: -len(suffix)]
                groups.setdefault(prefix, {})[suffix] = value
                break
    expected = frozenset(_PROJECTOR_PARAMETER_SUFFIXES)
    incomplete = {
        prefix: sorted(expected - set(values))
        for prefix, values in groups.items()
        if set(values) != expected
    }
    if incomplete:
        raise RuntimeError(
            "guidance projector state is incomplete for prefixes "
            f"{incomplete}"
        )
    return groups


def _transferable_projector_state(adapter: nn.Module) -> dict[str, Tensor]:
    state = adapter.state_dict()
    groups = _projector_parameter_groups(state)
    if not groups:
        raise ValueError(
            "the selected guidance architecture has no separable text projector state; "
            "Stage B projector transfer requires projected token guidance"
        )
    return {
        prefix + suffix: value.detach().clone()
        for prefix, values in groups.items()
        for suffix, value in values.items()
    }


def _cross_attention_parameter_names(adapter: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, _parameter in adapter.named_parameters() if name.startswith("cross_attention."))


def _condition_adapter_parameter_names(adapter: nn.Module) -> tuple[str, ...]:
    prefixes = ("condition_adapter.", "condition_adapters.")
    return tuple(
        name
        for name, _parameter in adapter.named_parameters()
        if name.startswith(prefixes)
    )


def extract_stage_a_projector_state(stage: AdapterOnlyStageA) -> dict[str, Tensor]:
    """Copy only Stage A projector tensors for Stage B initialization.

    The result contains adapter-local keys, which makes it suitable for a
    checkpoint-independent handoff and keeps Stage A cross-attention tensors
    out by construction.
    """

    if not isinstance(stage, AdapterOnlyStageA):
        raise TypeError("stage must be AdapterOnlyStageA")
    return _transferable_projector_state(stage.student.guidance_adapter)


def _adapter_projector_state_from_checkpoint(
    state_dict: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Recover adapter-local projector keys from common Stage A checkpoints."""

    # Projector-only Stage A checkpoints deliberately have no GraphDiT or
    # adapter wrapper.  Their formal candidate state is simply
    # ``projector.{text_norm,text_projection}.*``.  Broadcast that one shared
    # embedding-model-to-k projector across the Stage-B topology; no other
    # Stage-A state is transferable.
    prefixes = ("student.guidance_adapter.", "guidance_adapter.", "projector.")
    result: dict[str, Tensor] = {}
    for raw_name, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        for prefix in prefixes:
            if raw_name.startswith(prefix):
                local_name = raw_name[len(prefix) :]
                if _projector_state_keys_from_name((local_name,)):
                    result[local_name] = value.detach().clone()
                break
    return result


def _projector_state_keys_from_name(names: tuple[str, ...]) -> tuple[str, ...]:
    prefixes = (
        "text_norm.",
        "text_projection.",
        "projector_groups.",
        "null_token",
        "null_tokens.",
    )
    return tuple(
        name
        for name in names
        if name == "null_token" or name.startswith(prefixes)
    )


def _initialize_projectors(
    student: TokenCrossAttentionGraphDiT,
    projector_state: Mapping[str, Tensor],
) -> tuple[str, ...]:
    target = student.guidance_adapter.state_dict()
    target_groups = _projector_parameter_groups(target)
    source_groups = _projector_parameter_groups(projector_state)
    if not target_groups or not source_groups:
        raise ValueError(
            "Stage B projector transfer requires source and target projected token adapters"
        )
    target_prefixes = tuple(target_groups)
    source_prefixes = tuple(source_groups)
    if len(source_prefixes) == 1:
        source_for_target = {
            target_prefix: source_groups[source_prefixes[0]]
            for target_prefix in target_prefixes
        }
    elif source_prefixes == target_prefixes:
        source_for_target = {
            target_prefix: source_groups[target_prefix]
            for target_prefix in target_prefixes
        }
    else:
        raise RuntimeError(
            "Stage A projector topology cannot be transferred into Stage B: "
            f"source_groups={source_prefixes}, target_groups={target_prefixes}"
        )
    updates: dict[str, Tensor] = {}
    for target_prefix, source_values in source_for_target.items():
        for suffix in _PROJECTOR_PARAMETER_SUFFIXES:
            target_name = target_prefix + suffix
            source_value = source_values[suffix]
            if tuple(source_value.shape) != tuple(target[target_name].shape):
                raise RuntimeError(
                    "Stage A projector shape differs from Stage B adapter for "
                    f"{target_name!r}"
                )
            updates[target_name] = source_value
    student.guidance_adapter.load_state_dict(updates, strict=False)
    return tuple(updates)


@dataclass(frozen=True, slots=True)
class StageBInitializationReport:
    """Initialization facts required to reproduce the Stage B handoff."""

    reference_load: GraphDiTStateLoadReport
    student_load: GraphDiTStateLoadReport
    transferred_projector_names: tuple[str, ...]
    fresh_condition_adapter_names: tuple[str, ...]
    fresh_cross_attention_names: tuple[str, ...]
    trainable_parameter_names: tuple[str, ...]
    architecture_contract: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_load": self.reference_load.as_dict(),
            "student_load": self.student_load.as_dict(),
            "transferred_projector_names": list(self.transferred_projector_names),
            "fresh_condition_adapter_names": list(self.fresh_condition_adapter_names),
            "fresh_cross_attention_names": list(self.fresh_cross_attention_names),
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "architecture_contract": dict(self.architecture_contract),
        }


@dataclass(frozen=True, slots=True)
class StageBForwardOutput:
    """One Stage B forward pass on shared reference/student corruption."""

    loss: Tensor
    objective: GuidedObjectiveOutput
    corruption: CorruptedGraphBatch
    student_logits: Mapping[str, Tensor]
    reference_logits: Mapping[str, Tensor]
    semantic: SemanticAlignmentOutput | None = None
    semantic_weighted_objective: Tensor | None = None
    semantic_configured_weighted_objective: Tensor | None = None
    teacher_semantic_configured_objective: Tensor | None = None
    teacher_semantic_effective_objective: Tensor | None = None
    teacher_semantic_controller_scale: Tensor | None = None
    positive_recovery: "PositiveRecoveryOutput | None" = None
    rollout_reference_kl: "RolloutReferenceKLOutput | None" = None
    guidance_residual_scale: Tensor | None = None

    @property
    def metrics(self) -> dict[str, Tensor]:
        """Aggregate optional Stage B components for generic formal runners."""

        result = dict(self.objective.metrics)
        result["stage_b_total_loss"] = self.loss
        if self.semantic is not None:
            result.update(self.semantic.metrics())
            if self.semantic_weighted_objective is None:  # pragma: no cover - paired construction
                raise AssertionError("semantic output is present without its weighted objective")
            result["semantic_weighted_objective"] = self.semantic_weighted_objective
            if self.semantic_configured_weighted_objective is not None:
                result["semantic_configured_weighted_objective"] = (
                    self.semantic_configured_weighted_objective
                )
        if self.teacher_semantic_configured_objective is not None:
            if (
                self.teacher_semantic_effective_objective is None
                or self.teacher_semantic_controller_scale is None
            ):  # pragma: no cover - paired construction
                raise AssertionError(
                    "teacher/semantic configured contribution lacks controller telemetry"
                )
            result["teacher_semantic_configured_objective"] = (
                self.teacher_semantic_configured_objective
            )
            result["teacher_semantic_effective_objective"] = (
                self.teacher_semantic_effective_objective
            )
            result["teacher_semantic_controller_scale"] = (
                self.teacher_semantic_controller_scale
            )
        if self.positive_recovery is not None:
            result.update(self.positive_recovery.metrics())
        if self.rollout_reference_kl is not None:
            result.update(self.rollout_reference_kl.metrics())
        if self.guidance_residual_scale is not None:
            result["guidance_residual_scale"] = self.guidance_residual_scale
        return result


@dataclass(frozen=True, slots=True)
class PositiveRecoveryOutput:
    """Diagnostics for one optional clean-control or REC1 recovery update."""

    loss: Tensor
    ce: Tensor
    effective_weight: Tensor
    ready: Tensor
    state_count: Tensor
    common_t: Tensor
    common_s: Tensor
    node_rollin_change_fraction: Tensor
    edge_rollin_change_fraction: Tensor

    def metrics(self) -> dict[str, Tensor]:
        return {
            "positive_recovery_ce": self.ce,
            "positive_recovery_loss": self.loss,
            "positive_recovery_effective_weight": self.effective_weight,
            "positive_recovery_ready": self.ready,
            "positive_recovery_state_count": self.state_count,
            "positive_recovery_common_t": self.common_t,
            "positive_recovery_common_s": self.common_s,
            "positive_recovery_node_rollin_change_fraction": self.node_rollin_change_fraction,
            "positive_recovery_edge_rollin_change_fraction": self.edge_rollin_change_fraction,
        }


@dataclass(frozen=True, slots=True)
class RolloutReferenceKLOutput:
    """One differentiable replay KL evaluation plus bounded replay telemetry."""

    loss: Tensor
    kl: Tensor
    ready: Tensor
    state_count: Tensor
    trajectory_count: Tensor
    group_count: Tensor
    refresh_step: Tensor

    def metrics(self) -> dict[str, Tensor]:
        return {
            "rollout_replay_kl": self.kl,
            "rollout_replay_loss": self.loss,
            "rollout_replay_ready": self.ready,
            "rollout_replay_state_count": self.state_count,
            "rollout_replay_trajectory_count": self.trajectory_count,
            "rollout_replay_group_count": self.group_count,
            "rollout_replay_refresh_step": self.refresh_step,
        }


ROLLOUT_REPLAY_RANK_RECOVERY_SCHEMA = (
    "retro_guided_generation.stage_b.rollout_replay_rank_recovery.v1"
)
ROLLOUT_REPLAY_KEYS = frozenset(
    {
        "t",
        "r_X",
        "r_E",
        "p_X",
        "p_E",
        "node_mask",
        "p_mask",
        "cond",
        "tokens",
        "token_mask",
        "group_index",
    }
)
ROLLOUT_REPLAY_STAT_KEYS = frozenset(
    {"ready", "state_count", "trajectory_count", "group_count"}
)
TEACHER_SEMANTIC_CONTROLLER_STATE_SCHEMA = (
    "retro_guided_generation.stage_b.teacher_semantic_loss_share_controller.v1"
)
TEACHER_SEMANTIC_CONTROLLER_STATUS_CODES = {
    "healthy": 0,
    "observe": 1,
    "warning": 2,
    "adaptive": 3,
}


class AdapterStageB(nn.Module):
    """Frozen base GraphDiT plus Stage A-initialized, trainable guidance adapter."""

    def __init__(
        self,
        *,
        reference: GraphDiT,
        student: TokenCrossAttentionGraphDiT,
        objective: GuidanceObjective,
        freeze_transferred_projectors: bool = False,
        semantic_alignment: SemanticAlignmentConfig | None = None,
        teacher_semantic_loss_share_controller: TeacherSemanticLossShareControllerConfig = (
            TeacherSemanticLossShareControllerConfig()
        ),
        positive_recovery: PositiveRecoveryConfig = PositiveRecoveryConfig(),
        rollout_reference_kl: RolloutReferenceKLConfig = RolloutReferenceKLConfig(),
        guidance_residual_schedule: GuidanceResidualScheduleConfig = (
            GuidanceResidualScheduleConfig()
        ),
    ) -> None:
        super().__init__()
        if not isinstance(reference, GraphDiT) or not isinstance(student, TokenCrossAttentionGraphDiT):
            raise TypeError("reference and student must be GraphDiT model instances")
        if not isinstance(objective, GuidanceObjective):
            raise TypeError("objective must be GuidanceObjective")
        if reference.config != student.config:
            raise ValueError("reference and student must have identical GraphDiT configs")
        if objective.config.hybrid_branch_objective is not True:
            raise ValueError("Stage B requires a hybrid branch-aware objective")
        candidate_mode = str(objective.config.candidate_supervision_mode).strip().lower()
        if candidate_mode not in {"contrastive", "positive_rank", "positive_rank_hybrid"}:
            raise ValueError("Stage B requires contrastive or positive-rank candidate supervision")
        if candidate_mode in {"positive_rank", "positive_rank_hybrid"} and float(objective.config.route_ranking_weight) <= 0.0:
            raise ValueError("Stage B positive-rank supervision requires a positive route_ranking_weight")
        if not isinstance(freeze_transferred_projectors, bool):
            raise TypeError("freeze_transferred_projectors must be a bool")
        if semantic_alignment is not None and not isinstance(semantic_alignment, SemanticAlignmentConfig):
            raise TypeError("semantic_alignment must be SemanticAlignmentConfig or None")
        if not isinstance(
            teacher_semantic_loss_share_controller,
            TeacherSemanticLossShareControllerConfig,
        ):
            raise TypeError(
                "teacher_semantic_loss_share_controller must be "
                "TeacherSemanticLossShareControllerConfig"
            )
        if not isinstance(positive_recovery, PositiveRecoveryConfig):
            raise TypeError("positive_recovery must be PositiveRecoveryConfig")
        if not isinstance(rollout_reference_kl, RolloutReferenceKLConfig):
            raise TypeError("rollout_reference_kl must be RolloutReferenceKLConfig")
        if not isinstance(
            guidance_residual_schedule, GuidanceResidualScheduleConfig
        ):
            raise TypeError(
                "guidance_residual_schedule must be GuidanceResidualScheduleConfig"
            )
        if positive_recovery.enabled and candidate_mode not in {"positive_rank", "positive_rank_hybrid"}:
            raise ValueError("positive recovery requires positive-rank candidate supervision")
        if positive_recovery.enabled and positive_recovery.branch not in {"metric", "structure"}:
            raise ValueError("Stage B positive recovery branch must be metric or structure")
        if rollout_reference_kl.enabled and str(objective.config.preservation_mode).strip().lower() not in {
            "reference_kl",
            "reference_kl_shift",
        }:
            raise ValueError("rollout reference-KL requires objective.preservation_mode='reference_kl'")
        if semantic_alignment is not None and not isinstance(
            student.guidance_adapter, (GroupedTokenGuidanceAdapter, TokenGuidanceAdapter)
        ):
            raise ValueError("semantic alignment requires a projected token guidance adapter")
        self.reference = reference
        self.student = student
        self.objective = objective
        self.freeze_transferred_projectors = freeze_transferred_projectors
        self.semantic_alignment_config = semantic_alignment
        self.teacher_semantic_loss_share_controller_config = (
            teacher_semantic_loss_share_controller
        )
        self.semantic_alignment_objective = (
            None
            if semantic_alignment is None
            else SemanticAlignmentObjective(
                selected_branch=semantic_alignment.selected_branch,
                pair_weight=semantic_alignment.pair_weight,
                pair_margin=semantic_alignment.pair_margin,
                endpoint_probability_floor=semantic_alignment.endpoint_probability_floor,
            )
        )
        self.positive_recovery_config = positive_recovery
        self.rollout_reference_kl_config = rollout_reference_kl
        self.guidance_residual_schedule = guidance_residual_schedule
        # Replay is detached runtime state.  It is excluded from model and
        # candidate state_dicts, but the formal runner persists it per rank so
        # a recovery resume keeps the exact same refresh interval and gradient.
        self._rollout_replay: dict[str, Tensor] | None = None
        self._rollout_replay_stats: dict[str, int] = {}
        self._rollout_replay_step: int | None = None
        self._formal_training_step = 0
        self._formal_optimizer_updates_per_epoch: int | None = None
        self._formal_max_optimizer_updates: int | None = None
        self._guidance_residual_scale_value = float(
            guidance_residual_schedule.start
        )
        # Keep one stable scalar Tensor so torch.compile does not specialize a
        # new Student graph for every cosine value.  It follows model.to() and
        # is reconstructed from the schedule on resume rather than persisted.
        self.register_buffer(
            "_guidance_residual_scale_tensor",
            torch.tensor(self._guidance_residual_scale_value, dtype=torch.float32),
            persistent=False,
        )
        # This update-level state is persisted by the formal rank-recovery
        # hook, not by the inference/candidate state_dict.  Candidate model
        # identity and Stage-C promotion therefore remain reference/student
        # only, while an interrupted optimizer run resumes the exact scale.
        for name, value, dtype in (
            ("_teacher_semantic_scale_tensor", 1.0, torch.float64),
            ("_teacher_semantic_ema_base", 0.0, torch.float64),
            ("_teacher_semantic_ema_effective", 0.0, torch.float64),
            ("_teacher_semantic_ema_other", 0.0, torch.float64),
            ("_teacher_semantic_last_share", 0.0, torch.float64),
            ("_teacher_semantic_updates", 0, torch.int64),
            ("_teacher_semantic_adaptations", 0, torch.int64),
        ):
            self.register_buffer(
                name,
                torch.tensor(value, dtype=dtype),
                persistent=False,
            )
        self._compiled_reference_forward: Callable[..., object] | None = None
        self._compiled_student_forward: Callable[..., object] | None = None
        self._compiled_reference_disabled = False
        self._compiled_student_disabled = False
        self._eager_student_sampling = os.environ.get("STAGE_B_EAGER_SAMPLING") == "1"
        if os.environ.get("STAGE_B_TORCH_COMPILE") == "1":
            compile_fn = getattr(torch, "compile", None)
            if compile_fn is None:  # pragma: no cover - all formal workers use torch 2.x
                raise RuntimeError("Stage-B torch.compile was requested but this torch has no compiler")
            compile_mode = os.environ.get("STAGE_B_TORCH_COMPILE_MODE", "default").strip() or "default"
            compile_dynamic = os.environ.get("STAGE_B_TORCH_COMPILE_DYNAMIC", "1") != "0"
            compile_kwargs = {"mode": compile_mode, "dynamic": compile_dynamic}
            # Compile only bound forward callables.  The containing AdapterStageB
            # module remains unwrapped, so recovery/candidate state_dict keys and
            # optimizer parameter identity stay identical to the non-compiled run.
            self._compiled_reference_forward = compile_fn(
                self.reference.forward,
                **compile_kwargs,
            )
            self._compiled_student_forward = compile_fn(
                self.student.forward,
                **compile_kwargs,
            )
            print(
                "Stage-B compile partition: train_and_replay=compiled sampling="
                + ("eager" if self._eager_student_sampling else "compiled_shared"),
                flush=True,
            )
        self._freeze_stage_b()

    def _reference_forward(self, **kwargs: object) -> object:
        compiled = self._compiled_reference_forward
        if compiled is None or self._compiled_reference_disabled:
            return self.reference(**kwargs)
        try:
            return compiled(**kwargs)
        except Exception as error:  # pragma: no cover - exercised on worker/compiler mismatch
            # The compiler is an optional execution optimization.  If a worker
            # backend rejects this graph, keep the exact eager equations and
            # continue from the same batch rather than losing the recovery run.
            self._compiled_reference_disabled = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Stage-B reference torch.compile disabled after fallback: {error}", flush=True)
            return self.reference(**kwargs)

    def _student_train_forward(self, **kwargs: object) -> object:
        """Run differentiable train/replay work through the compiled callable."""

        kwargs = self._guided_student_kwargs(kwargs)
        compiled = self._compiled_student_forward
        if compiled is None or self._compiled_student_disabled:
            return self.student(**kwargs)
        try:
            return compiled(**kwargs)
        except Exception as error:  # pragma: no cover - exercised on worker/compiler mismatch
            self._compiled_student_disabled = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Stage-B student torch.compile disabled after fallback: {error}", flush=True)
            return self.student(**kwargs)

    def _student_sampling_forward(self, **kwargs: object) -> object:
        """Route detached sampling through the explicitly selected partition.

        The formal runner invokes rollout refresh outside its BF16 autocast
        context, while normal training and differentiable replay run inside
        autocast.  Reusing one compiled callable across those global states
        causes Dynamo recompilation.  The B4 pilot selects the original module
        directly; leaving the switch unset preserves the pre-B4 shared
        compiled callable for an exact execution fallback.  Neither route
        changes model parameters, equations, RNG, or checkpoint identity.
        """

        if self._eager_student_sampling:
            return self.student(**self._guided_student_kwargs(kwargs))
        return self._student_train_forward(**kwargs)

    def _guided_student_kwargs(
        self, kwargs: Mapping[str, object]
    ) -> dict[str, object]:
        """Bind the controller to every guided Student invocation."""

        result = dict(kwargs)
        if "guidance_residual_scale" in result:
            raise ValueError(
                "AdapterStageB owns guidance_residual_scale; callers cannot override it"
            )
        if (
            "guidance_tokens_group" in result
            or bool(result.get("guidance_null", False))
        ):
            result["guidance_residual_scale"] = (
                self._guidance_residual_scale_tensor
            )
        return result

    @property
    def graphdit_config(self) -> GraphDiTConfig:
        return self.reference.config

    def set_formal_training_step(self, completed_steps: int) -> None:
        """Bind REC1 and residual schedules to the same update coordinate."""

        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or completed_steps < 0:
            raise ValueError("completed_steps must be a non-negative integer")
        self._formal_training_step = completed_steps
        total = self._formal_max_optimizer_updates
        if total is not None:
            if completed_steps >= total:
                raise ValueError(
                    "completed_steps must remain below the configured formal budget"
                )
            value = self.guidance_residual_schedule.scale(
                completed_steps,
                total_optimizer_updates=total,
            )
            self._guidance_residual_scale_value = value
            with torch.no_grad():
                self._guidance_residual_scale_tensor.fill_(value)

    @property
    def current_guidance_residual_scale(self) -> float:
        return self._guidance_residual_scale_value

    @property
    def current_teacher_semantic_loss_scale(self) -> float:
        """Return the one-sided scale applied to both auxiliary teachers."""

        return float(self._teacher_semantic_scale_tensor.detach().cpu())

    def _teacher_semantic_share_status(self, share: float) -> tuple[str, int]:
        config = self.teacher_semantic_loss_share_controller_config
        if share < float(config.healthy_max_share):
            name = "healthy"
        elif share < float(config.observe_max_share):
            name = "observe"
        elif share <= float(config.trigger_share):
            name = "warning"
        else:
            name = "adaptive"
        return name, TEACHER_SEMANTIC_CONTROLLER_STATUS_CODES[name]

    @staticmethod
    def _finite_reduced_metric(
        metrics: Mapping[str, object], name: str
    ) -> float:
        try:
            raw = metrics[name]
        except KeyError as error:
            raise RuntimeError(
                f"teacher/semantic controller requires reduced metric {name!r}"
            ) from error
        if isinstance(raw, Tensor):
            if raw.numel() != 1:
                raise RuntimeError(
                    f"teacher/semantic controller metric {name!r} must be scalar"
                )
            value = float(raw.detach().cpu())
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise RuntimeError(
                f"teacher/semantic controller metric {name!r} must be numeric"
            )
        else:
            value = float(raw)
        if not math.isfinite(value):
            raise RuntimeError(
                f"teacher/semantic controller metric {name!r} must be finite"
            )
        return value

    def update_formal_loss_share_controller(
        self, reduced_metrics: Mapping[str, object]
    ) -> dict[str, float]:
        """Observe one globally reduced update and set the next update's scale.

        Formal training calls this exactly once after ``optimizer.step`` and
        after DDP/gradient-accumulation reduction.  Consequently every rank
        applies the same coefficient on the next update without adding a
        collective inside the model forward.
        """

        config = self.teacher_semantic_loss_share_controller_config
        if not config.enabled:
            return {}
        configured = self._finite_reduced_metric(
            reduced_metrics, "teacher_semantic_configured_objective"
        )
        effective = self._finite_reduced_metric(
            reduced_metrics, "teacher_semantic_effective_objective"
        )
        total = self._finite_reduced_metric(reduced_metrics, "stage_b_total_loss")
        # Losses are normally accumulated in FP32, but BF16 autocast can
        # leave a one-ULP closure difference between a separately logged
        # component and the summed total.  Tolerate only that numerical seam.
        tolerance = 1.0e-5 * max(1.0, abs(configured), abs(effective), abs(total))
        if configured < -tolerance or effective < -tolerance or total < -tolerance:
            raise RuntimeError(
                "teacher/semantic controller requires non-negative loss contributions"
            )
        configured = max(0.0, configured)
        effective = max(0.0, effective)
        total = max(0.0, total)
        if effective > configured + tolerance:
            raise RuntimeError(
                "teacher/semantic effective contribution exceeds its configured contribution"
            )
        if effective > total + tolerance:
            raise RuntimeError(
                "teacher/semantic contribution exceeds Stage-B total loss"
            )
        other = max(0.0, total - effective)

        previous_updates = int(self._teacher_semantic_updates.detach().cpu())
        decay = 0.0 if previous_updates == 0 else float(config.ema_decay)

        def ema(previous: Tensor, observed: float) -> float:
            return decay * float(previous.detach().cpu()) + (1.0 - decay) * observed

        ema_base = ema(self._teacher_semantic_ema_base, configured)
        ema_effective = ema(self._teacher_semantic_ema_effective, effective)
        ema_other = ema(self._teacher_semantic_ema_other, other)
        denominator = ema_effective + ema_other
        share = 0.0 if denominator <= 0.0 else ema_effective / denominator
        status_name, status_code = self._teacher_semantic_share_status(share)
        updates = previous_updates + 1
        ready = updates >= config.warmup_updates
        previous_scale = self.current_teacher_semantic_loss_scale
        next_scale = previous_scale
        triggered = ready and status_name == "adaptive" and ema_base > 0.0
        if triggered:
            target_odds = float(config.target_share) / (
                1.0 - float(config.target_share)
            )
            desired_scale = target_odds * ema_other / ema_base
            next_scale = max(
                float(config.minimum_scale),
                min(previous_scale, desired_scale),
            )
        adapted = next_scale < previous_scale - 1.0e-15
        adaptations = int(self._teacher_semantic_adaptations.detach().cpu()) + int(
            adapted
        )
        with torch.no_grad():
            self._teacher_semantic_scale_tensor.fill_(next_scale)
            self._teacher_semantic_ema_base.fill_(ema_base)
            self._teacher_semantic_ema_effective.fill_(ema_effective)
            self._teacher_semantic_ema_other.fill_(ema_other)
            self._teacher_semantic_last_share.fill_(share)
            self._teacher_semantic_updates.fill_(updates)
            self._teacher_semantic_adaptations.fill_(adaptations)
        return {
            "teacher_semantic_share_ema": share,
            "teacher_semantic_share_status_code": float(status_code),
            "teacher_semantic_controller_ready": float(ready),
            "teacher_semantic_controller_triggered": float(triggered),
            "teacher_semantic_controller_scale_next": next_scale,
            "teacher_semantic_controller_adaptation_count": float(adaptations),
        }

    def configure_formal_training_plan(
        self,
        *,
        optimizer_updates_per_epoch: int,
        max_optimizer_updates: int,
        candidate_optimizer_updates: tuple[int, ...],
    ) -> dict[str, object]:
        """Resolve all Stage-B schedules from the generic formal plan."""

        if (
            isinstance(max_optimizer_updates, bool)
            or not isinstance(max_optimizer_updates, int)
            or max_optimizer_updates < 1
        ):
            raise ValueError("max_optimizer_updates must be a positive integer")
        if not isinstance(candidate_optimizer_updates, tuple) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= max_optimizer_updates
            for value in candidate_optimizer_updates
        ):
            raise ValueError(
                "candidate_optimizer_updates must be an in-budget integer tuple"
            )
        self._formal_max_optimizer_updates = max_optimizer_updates
        legacy = self.configure_formal_epoch_plan(
            optimizer_updates_per_epoch=optimizer_updates_per_epoch
        )
        self.set_formal_training_step(self._formal_training_step)
        return {
            **legacy,
            "guidance_residual": self.guidance_residual_schedule.resolved_contract(
                total_optimizer_updates=max_optimizer_updates
            ),
            "candidate_optimizer_updates": list(candidate_optimizer_updates),
        }

    def formal_recovery_contract(self) -> dict[str, object]:
        """Return model-side state that must match before recovery resume."""

        total = self._formal_max_optimizer_updates
        if total is None:
            raise RuntimeError(
                "formal training plan must be configured before recovery contract"
            )
        result: dict[str, object] = {
            "guidance_residual": self.guidance_residual_schedule.resolved_contract(
                total_optimizer_updates=total
            ),
            "rank_recovery_state_schema": ROLLOUT_REPLAY_RANK_RECOVERY_SCHEMA,
        }
        if self.teacher_semantic_loss_share_controller_config.enabled:
            result["teacher_semantic_loss_share_controller"] = (
                self.teacher_semantic_loss_share_controller_config.resolved_contract()
            )
            result["teacher_semantic_controller_state_schema"] = (
                TEACHER_SEMANTIC_CONTROLLER_STATE_SCHEMA
            )
        return result

    def _teacher_semantic_controller_recovery_state(
        self,
    ) -> dict[str, object] | None:
        if not self.teacher_semantic_loss_share_controller_config.enabled:
            return None
        return {
            "schema_version": TEACHER_SEMANTIC_CONTROLLER_STATE_SCHEMA,
            "scale": self.current_teacher_semantic_loss_scale,
            "ema_base": float(self._teacher_semantic_ema_base.detach().cpu()),
            "ema_effective": float(
                self._teacher_semantic_ema_effective.detach().cpu()
            ),
            "ema_other": float(self._teacher_semantic_ema_other.detach().cpu()),
            "last_share": float(self._teacher_semantic_last_share.detach().cpu()),
            "updates": int(self._teacher_semantic_updates.detach().cpu()),
            "adaptations": int(
                self._teacher_semantic_adaptations.detach().cpu()
            ),
        }

    def _load_teacher_semantic_controller_recovery_state(
        self, value: object
    ) -> None:
        enabled = self.teacher_semantic_loss_share_controller_config.enabled
        if not enabled:
            if value is not None:
                raise RuntimeError(
                    "disabled teacher/semantic controller received recovery state"
                )
            return
        if not isinstance(value, Mapping):
            raise RuntimeError(
                "enabled teacher/semantic controller is missing recovery state"
            )
        if value.get("schema_version") != TEACHER_SEMANTIC_CONTROLLER_STATE_SCHEMA:
            raise RuntimeError("unsupported teacher/semantic controller recovery schema")
        expected = {
            "schema_version",
            "scale",
            "ema_base",
            "ema_effective",
            "ema_other",
            "last_share",
            "updates",
            "adaptations",
        }
        if set(value) != expected:
            raise RuntimeError(
                "teacher/semantic controller recovery fields are incomplete"
            )
        numeric: dict[str, float] = {}
        for name in (
            "scale",
            "ema_base",
            "ema_effective",
            "ema_other",
            "last_share",
        ):
            raw = value[name]
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
            ):
                raise RuntimeError(
                    f"teacher/semantic controller recovery {name!r} must be finite"
                )
            numeric[name] = float(raw)
        if not 0.0 <= numeric["scale"] <= 1.0:
            raise RuntimeError("teacher/semantic controller recovery scale is invalid")
        if any(numeric[name] < 0.0 for name in ("ema_base", "ema_effective", "ema_other")):
            raise RuntimeError("teacher/semantic controller recovery EMA is negative")
        if not 0.0 <= numeric["last_share"] <= 1.0:
            raise RuntimeError("teacher/semantic controller recovery share is invalid")
        integers: dict[str, int] = {}
        for name in ("updates", "adaptations"):
            raw = value[name]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise RuntimeError(
                    f"teacher/semantic controller recovery {name!r} is invalid"
                )
            integers[name] = raw
        if integers["adaptations"] > integers["updates"]:
            raise RuntimeError(
                "teacher/semantic controller adaptations exceed observed updates"
            )
        with torch.no_grad():
            self._teacher_semantic_scale_tensor.fill_(numeric["scale"])
            self._teacher_semantic_ema_base.fill_(numeric["ema_base"])
            self._teacher_semantic_ema_effective.fill_(numeric["ema_effective"])
            self._teacher_semantic_ema_other.fill_(numeric["ema_other"])
            self._teacher_semantic_last_share.fill_(numeric["last_share"])
            self._teacher_semantic_updates.fill_(integers["updates"])
            self._teacher_semantic_adaptations.fill_(integers["adaptations"])

    def formal_rank_recovery_state(self) -> dict[str, object]:
        """Return detached rank-local rollout replay for exact formal resume."""

        enabled = bool(self.rollout_reference_kl_config.enabled)
        replay = self._rollout_replay
        step = self._rollout_replay_step
        stats = self._rollout_replay_stats
        if enabled:
            if replay is None or step is None:
                raise RuntimeError(
                    "enabled rollout reference-KL has no replay to persist"
                )
            if set(replay) != ROLLOUT_REPLAY_KEYS:
                raise RuntimeError("rollout replay tensor keys are incomplete")
            if set(stats) != ROLLOUT_REPLAY_STAT_KEYS:
                raise RuntimeError("rollout replay statistic keys are incomplete")
        elif replay is not None or step is not None or stats:
            raise RuntimeError("disabled rollout reference-KL retained replay state")
        return {
            "schema_version": ROLLOUT_REPLAY_RANK_RECOVERY_SCHEMA,
            "enabled": enabled,
            "teacher_semantic_controller_state": (
                self._teacher_semantic_controller_recovery_state()
            ),
            "rollout_replay_step": step,
            "rollout_replay_stats": dict(stats),
            "rollout_replay": (
                None
                if replay is None
                else {
                    name: value.detach().to(device="cpu").clone()
                    for name, value in replay.items()
                }
            ),
        }

    def load_formal_rank_recovery_state(
        self,
        value: Mapping[str, object],
    ) -> None:
        """Restore detached rollout replay onto this rank's active device."""

        if not isinstance(value, Mapping):
            raise TypeError("formal rank recovery state must be a mapping")
        if value.get("schema_version") != ROLLOUT_REPLAY_RANK_RECOVERY_SCHEMA:
            raise RuntimeError("unsupported Stage-B rollout replay recovery schema")
        enabled = value.get("enabled")
        if not isinstance(enabled, bool):
            raise RuntimeError("rollout replay recovery enabled flag must be a bool")
        if enabled != bool(self.rollout_reference_kl_config.enabled):
            raise RuntimeError("rollout replay recovery enablement differs from config")
        self._load_teacher_semantic_controller_recovery_state(
            value.get("teacher_semantic_controller_state")
        )
        raw_replay = value.get("rollout_replay")
        raw_step = value.get("rollout_replay_step")
        raw_stats = value.get("rollout_replay_stats")
        if not enabled:
            if raw_replay is not None or raw_step is not None or raw_stats != {}:
                raise RuntimeError("disabled rollout replay recovery must be empty")
            self._rollout_replay = None
            self._rollout_replay_step = None
            self._rollout_replay_stats = {}
            return
        if isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0:
            raise RuntimeError("rollout replay recovery step must be non-negative")
        if not isinstance(raw_replay, Mapping) or set(raw_replay) != ROLLOUT_REPLAY_KEYS:
            raise RuntimeError("rollout replay recovery tensors are incomplete")
        if not isinstance(raw_stats, Mapping) or set(raw_stats) != ROLLOUT_REPLAY_STAT_KEYS:
            raise RuntimeError("rollout replay recovery statistics are incomplete")
        stats: dict[str, int] = {}
        for name in ROLLOUT_REPLAY_STAT_KEYS:
            raw_stat = raw_stats[name]
            if isinstance(raw_stat, bool) or not isinstance(raw_stat, int) or raw_stat < 0:
                raise RuntimeError(
                    f"rollout replay recovery statistic {name!r} is invalid"
                )
            stats[name] = raw_stat
        if stats["ready"] != 1:
            raise RuntimeError("enabled rollout replay recovery must be ready")
        device = self._guidance_residual_scale_tensor.device
        replay: dict[str, Tensor] = {}
        for name in ROLLOUT_REPLAY_KEYS:
            raw_tensor = raw_replay[name]
            if not isinstance(raw_tensor, Tensor):
                raise RuntimeError(
                    f"rollout replay recovery value {name!r} must be a tensor"
                )
            replay[name] = self._normal_clone(raw_tensor.to(device=device))
        self._rollout_replay = replay
        self._rollout_replay_step = raw_step
        self._rollout_replay_stats = stats

    def configure_formal_epoch_plan(
        self,
        *,
        optimizer_updates_per_epoch: int,
    ) -> dict[str, object]:
        """Resolve public epoch schedules onto the exact optimizer-update grid."""

        if (
            isinstance(optimizer_updates_per_epoch, bool)
            or not isinstance(optimizer_updates_per_epoch, int)
            or optimizer_updates_per_epoch < 1
        ):
            raise ValueError("optimizer_updates_per_epoch must be a positive integer")
        self._formal_optimizer_updates_per_epoch = optimizer_updates_per_epoch
        return {
            "positive_recovery": self.positive_recovery_config.resolved_schedule_dict(
                optimizer_updates_per_epoch=optimizer_updates_per_epoch
            )
        }

    def _positive_recovery_effective_weight(self) -> float:
        return self.positive_recovery_config.effective_weight(
            self._formal_training_step,
            optimizer_updates_per_epoch=self._formal_optimizer_updates_per_epoch,
        )

    def _freeze_stage_b(self) -> None:
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        for parameter in self.student.guidance_adapter.parameters():
            parameter.requires_grad_(True)
        if self.freeze_transferred_projectors:
            projector_names = set(_projector_state_keys(self.student.guidance_adapter))
            for name, parameter in self.student.guidance_adapter.named_parameters():
                if name in projector_names:
                    parameter.requires_grad_(False)
        self.assert_stage_b_trainable()

    def assert_stage_b_trainable(self) -> None:
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("student.guidance_adapter.")
        ]
        if unexpected:
            raise RuntimeError(f"Stage B unexpectedly exposes non-adapter parameters: {unexpected}")
        cross_attention = _cross_attention_parameter_names(self.student.guidance_adapter)
        frozen_cross_attention = [
            name
            for name, parameter in self.student.guidance_adapter.named_parameters()
            if name in cross_attention and not parameter.requires_grad
        ]
        if frozen_cross_attention:
            raise RuntimeError(
                f"Stage B unexpectedly freezes generation cross-attention: {frozen_cross_attention}"
            )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        self.assert_stage_b_trainable()
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> "AdapterStageB":
        super().train(mode)
        self.reference.eval()
        return self

    def _validate_batch(self, batch: GuidedStageABatch) -> tuple[int, int]:
        if not isinstance(batch, GuidedStageABatch):
            raise TypeError("batch must be GuidedStageABatch")
        candidates = _validate_graphs(batch.graphs)
        expected_index, groups = _expected_group_index(
            batch.supervision.group_ptr, candidates, batch.graphs.p_X.device
        )
        branches = batch.supervision.guidance_branch
        if branches is None or len(branches) != groups:
            raise ValueError("Stage B requires one guidance branch label per group")
        active_branches = {
            branch
            for branch, weight in (
                ("structure", self.objective.config.structure_guidance_weight),
                ("metric", self.objective.config.metric_guidance_weight),
            )
            if float(weight) > 0.0
            and not (
                branch == "metric"
                and self.objective.metric_guidance_mode == "none"
            )
        }
        unexpected_branches = sorted(set(branches) - active_branches)
        if unexpected_branches:
            raise ValueError(
                "Stage B batch contains branches disabled by its objective: "
                f"{unexpected_branches}"
            )
        if (
            not isinstance(batch.group_index, Tensor)
            or batch.group_index.dtype != torch.int64
            or batch.group_index.shape != (candidates,)
            or batch.group_index.device != batch.graphs.p_X.device
        ):
            raise ValueError("group_index must be device-local int64[B]")
        if batch.group_index.device.type == "cuda":
            torch._assert_async(
                torch.all(batch.group_index == expected_index),
                "group_index is inconsistent with group_ptr",
            )
        elif not bool(torch.all(batch.group_index == expected_index)):
            raise ValueError("group_index is inconsistent with group_ptr")
        if batch.guidance_tokens_group.ndim != 3 or batch.guidance_tokens_group.shape[0] != groups:
            raise ValueError("guidance_tokens_group must have shape [G,L,D]")
        if (
            batch.guidance_token_mask_group.dtype != torch.bool
            or batch.guidance_token_mask_group.shape != batch.guidance_tokens_group.shape[:2]
        ):
            raise ValueError("guidance_token_mask_group must be bool [G,L]")
        if (
            batch.guidance_tokens_group.device != batch.graphs.p_X.device
            or batch.guidance_token_mask_group.device != batch.graphs.p_X.device
        ):
            raise ValueError("guidance token tensors must share the graph device")
        if batch.graphs.cond.shape[1] != self.graphdit_config.condition_dim:
            raise ValueError("batch cond width differs from GraphDiT config")
        return candidates, groups

    @staticmethod
    def _virtual_lane_group_ptr(batch: GuidedStageABatch) -> tuple[int, ...] | None:
        pointer = batch.virtual_microbatch_group_ptr
        if pointer is None:
            return None
        if (
            not isinstance(pointer, Tensor)
            or pointer.ndim != 1
            or pointer.numel() < 2
            or pointer.dtype not in {torch.int32, torch.int64}
        ):
            raise TypeError(
                "virtual_microbatch_group_ptr must be a non-empty integer CSR pointer"
            )
        values = tuple(pointer.detach().to(device="cpu", dtype=torch.long).tolist())
        groups = int(batch.supervision.group_ptr.numel() - 1)
        if values[0] != 0 or values[-1] != groups or any(
            right <= left for left, right in zip(values, values[1:])
        ):
            raise ValueError(
                "virtual_microbatch_group_ptr must cover all groups with non-empty lanes"
            )
        return values

    @staticmethod
    def _slice_graphs_for_lane(
        graphs: GraphConditionBatch,
        *,
        candidate_start: int,
        candidate_stop: int,
        nodes: int,
    ) -> GraphConditionBatch:
        candidate = slice(candidate_start, candidate_stop)
        return GraphConditionBatch(
            p_X=graphs.p_X[candidate, :nodes],
            p_E=graphs.p_E[candidate, :nodes, :nodes],
            r_X=graphs.r_X[candidate, :nodes],
            r_E=graphs.r_E[candidate, :nodes, :nodes],
            node_mask=graphs.node_mask[candidate, :nodes],
            p_mask=graphs.p_mask[candidate, :nodes],
            cond=graphs.cond[candidate],
        )

    @classmethod
    def _slice_virtual_lane_batch(
        cls,
        batch: GuidedStageABatch,
        *,
        group_start: int,
        group_stop: int,
    ) -> GuidedStageABatch:
        ptr = batch.supervision.group_ptr
        candidate_start = int(ptr[group_start].detach().to(device="cpu"))
        candidate_stop = int(ptr[group_stop].detach().to(device="cpu"))
        lane_mask = batch.graphs.node_mask[candidate_start:candidate_stop]
        nodes = int(lane_mask.sum(dim=1).max().detach().to(device="cpu"))
        if nodes < 1:
            raise ValueError("virtual microbatch lane contains no active graph nodes")
        candidate = slice(candidate_start, candidate_stop)
        group = slice(group_start, group_stop)
        supervision = batch.supervision
        local_supervision = replace(
            supervision,
            group_ptr=supervision.group_ptr[group_start : group_stop + 1] - candidate_start,
            endpoint_probability=supervision.endpoint_probability[candidate],
            teacher_probability=supervision.teacher_probability[candidate],
            structural_label=supervision.structural_label[candidate],
            listwise_eligible=supervision.listwise_eligible[group],
            guidance_branch=(
                None
                if supervision.guidance_branch is None
                else tuple(supervision.guidance_branch[group])
            ),
            group_index=(
                None
                if supervision.group_index is None
                else supervision.group_index[candidate] - group_start
            ),
            group_loss_weight=(
                None
                if supervision.group_loss_weight is None
                else supervision.group_loss_weight[group]
            ),
            branch_group_loss_weight=(
                None
                if supervision.branch_group_loss_weight is None
                else supervision.branch_group_loss_weight[group]
            ),
        )
        return replace(
            batch,
            graphs=cls._slice_graphs_for_lane(
                batch.graphs,
                candidate_start=candidate_start,
                candidate_stop=candidate_stop,
                nodes=nodes,
            ),
            supervision=local_supervision,
            group_index=batch.group_index[candidate] - group_start,
            guidance_tokens_group=batch.guidance_tokens_group[group],
            guidance_token_mask_group=batch.guidance_token_mask_group[group],
            route_semantic_candidate=(
                None
                if batch.route_semantic_candidate is None
                else batch.route_semantic_candidate[candidate]
            ),
            shuffled_guidance_tokens_group=(
                None
                if batch.shuffled_guidance_tokens_group is None
                else batch.shuffled_guidance_tokens_group[group]
            ),
            shuffled_guidance_token_mask_group=(
                None
                if batch.shuffled_guidance_token_mask_group is None
                else batch.shuffled_guidance_token_mask_group[group]
            ),
            virtual_microbatch_group_ptr=None,
        )

    @classmethod
    def _virtual_lane_batches(
        cls, batch: GuidedStageABatch
    ) -> tuple[GuidedStageABatch, ...]:
        pointer = cls._virtual_lane_group_ptr(batch)
        if pointer is None:
            return (batch,)
        return tuple(
            cls._slice_virtual_lane_batch(
                batch,
                group_start=group_start,
                group_stop=group_stop,
            )
            for group_start, group_stop in zip(pointer, pointer[1:])
        )

    @staticmethod
    def _pad_corruption_tensor(value: Tensor, *, nodes: int) -> Tensor:
        missing = nodes - int(value.shape[1])
        if missing < 0:
            raise ValueError("lane corruption exceeds merged graph width")
        if missing == 0:
            return value
        if value.ndim == 2:
            return torch.nn.functional.pad(value, (0, missing))
        if value.ndim == 3:
            return torch.nn.functional.pad(value, (0, 0, 0, missing))
        if value.ndim == 4:
            return torch.nn.functional.pad(value, (0, 0, 0, missing, 0, missing))
        raise ValueError("unsupported lane corruption tensor rank")

    @classmethod
    def _merge_lane_corruptions(
        cls,
        corruptions: tuple[CorruptedGraphBatch, ...],
        *,
        nodes: int,
    ) -> CorruptedGraphBatch:
        if not corruptions:
            raise ValueError("cannot merge an empty corruption list")

        def merged(name: str) -> Tensor:
            return torch.cat(
                [
                    cls._pad_corruption_tensor(getattr(item, name), nodes=nodes)
                    for item in corruptions
                ],
                dim=0,
            )

        return CorruptedGraphBatch(
            t_group=torch.cat([item.t_group for item in corruptions], dim=0),
            t_graph=torch.cat([item.t_graph for item in corruptions], dim=0),
            t_X=merged("t_X"),
            t_E=merged("t_E"),
            target_X=merged("target_X"),
            target_E=merged("target_E"),
            p_X=merged("p_X"),
            p_E=merged("p_E"),
            node_mask=merged("node_mask"),
            p_mask=merged("p_mask"),
        )

    @staticmethod
    def _slice_logits_for_lane(
        logits: Mapping[str, Tensor],
        *,
        candidate_start: int,
        candidate_stop: int,
        nodes: int,
    ) -> dict[str, Tensor]:
        return {
            "X": logits["X"][candidate_start:candidate_stop, :nodes],
            "E": logits["E"][candidate_start:candidate_stop, :nodes, :nodes],
        }

    @staticmethod
    def _mean_tensors(values: tuple[Tensor, ...]) -> Tensor:
        if not values:
            raise ValueError("cannot average an empty tensor tuple")
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result / len(values)

    @classmethod
    def _merge_lane_objectives(
        cls, outputs: tuple[GuidedObjectiveOutput, ...]
    ) -> GuidedObjectiveOutput:
        if len(outputs) == 1:
            return outputs[0]
        metric_names = tuple(outputs[0].metrics)
        group_metric_names = tuple(outputs[0].per_group)
        if any(tuple(output.metrics) != metric_names for output in outputs[1:]):
            raise RuntimeError("virtual lanes emitted different objective metrics")
        if any(tuple(output.per_group) != group_metric_names for output in outputs[1:]):
            raise RuntimeError("virtual lanes emitted different group metrics")
        return GuidedObjectiveOutput(
            loss=cls._mean_tensors(tuple(output.loss for output in outputs)),
            per_candidate_score=torch.cat(
                [output.per_candidate_score for output in outputs], dim=0
            ),
            per_candidate_prediction_probability=torch.cat(
                [output.per_candidate_prediction_probability for output in outputs], dim=0
            ),
            eligible_group_mask=torch.cat(
                [output.eligible_group_mask for output in outputs], dim=0
            ),
            structure_group_mask=torch.cat(
                [output.structure_group_mask for output in outputs], dim=0
            ),
            metric_group_mask=torch.cat(
                [output.metric_group_mask for output in outputs], dim=0
            ),
            metrics={
                name: cls._mean_tensors(tuple(output.metrics[name] for output in outputs))
                for name in metric_names
            },
            per_group={
                name: torch.cat([output.per_group[name] for output in outputs], dim=0)
                for name in group_metric_names
            },
        )

    @staticmethod
    def _index_graphs(graphs: GraphConditionBatch, indices: Tensor) -> GraphConditionBatch:
        """Select candidate rows without changing product masks or vocabulary."""

        if not isinstance(indices, Tensor) or indices.ndim != 1 or indices.dtype not in {
            torch.int32,
            torch.int64,
        }:
            raise TypeError("candidate indices must be a one-dimensional integer Tensor")
        if indices.device != graphs.p_X.device:
            raise ValueError("candidate indices must share the graph device")
        return GraphConditionBatch(
            p_X=graphs.p_X.index_select(0, indices),
            p_E=graphs.p_E.index_select(0, indices),
            r_X=graphs.r_X.index_select(0, indices),
            r_E=graphs.r_E.index_select(0, indices),
            node_mask=graphs.node_mask.index_select(0, indices),
            p_mask=graphs.p_mask.index_select(0, indices),
            cond=graphs.cond.index_select(0, indices),
        )

    @staticmethod
    def _normal_clone(value: Tensor) -> Tensor:
        """Turn an inference-mode rollout tensor into a normal replay input."""

        if not isinstance(value, Tensor):
            raise TypeError("rollout replay values must be tensors")
        with torch.inference_mode(False):
            return value.detach().clone()

    def _select_rollout_groups(self, batch: GuidedStageABatch) -> Tensor:
        """Choose bounded contrastive groups without changing the data sampler."""

        config = self.rollout_reference_kl_config
        branches = batch.supervision.guidance_branch
        if branches is None:
            raise ValueError("rollout reference-KL requires branch-aware supervision")
        ptr = batch.supervision.group_ptr.detach().to(device="cpu", dtype=torch.long)
        labels = batch.supervision.structural_label.detach().to(device="cpu", dtype=torch.long)
        preferred: list[int] = []
        fallback: list[int] = []
        for group_id, _branch in enumerate(branches):
            start, stop = int(ptr[group_id]), int(ptr[group_id + 1])
            group_labels = labels[start:stop]
            has_positive = bool(torch.any(group_labels == 1))
            has_negative = bool(torch.any(group_labels == 0))
            if has_positive and has_negative:
                preferred.append(group_id)
            elif has_positive:
                fallback.append(group_id)
        candidates = preferred + fallback
        if not candidates:
            raise RuntimeError("rollout reference-KL batch has no positive endpoints")
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            int(config.seed) + 1_000_003 * self._formal_training_step + 10_007 * rank
        )
        order = torch.randperm(len(candidates), generator=generator).tolist()
        selected = [candidates[index] for index in order[: min(len(candidates), config.max_groups)]]
        return torch.tensor(selected, dtype=torch.long, device=batch.graphs.p_X.device)

    @torch.no_grad()
    def refresh_rollout_replay(
        self,
        batch: GuidedStageABatch,
        *,
        completed_steps: int,
    ) -> dict[str, int]:
        """Sample current guided trajectories and retain detached trace states.

        This method is deliberately called by the generic formal runner before
        the optimization forward.  It owns no dataset sampler and does not
        alter the canonical true/shuffled wording draw; it only reuses the
        already materialized guided batch for a bounded on-policy trace.
        """

        config = self.rollout_reference_kl_config
        if not config.enabled:
            return {"ready": 0, "state_count": 0, "trajectory_count": 0, "group_count": 0}
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int) or completed_steps < 0:
            raise ValueError("completed_steps must be a non-negative integer")
        # Historical accumulation refreshed replay on the first microbatch and
        # the second call observed the same completed step and reused it.  A
        # merged physical batch therefore refreshes from lane zero only.
        batch = self._virtual_lane_batches(batch)[0]
        _candidates, _groups = self._validate_batch(batch)
        selected_groups = self._select_rollout_groups(batch)
        ptr = batch.supervision.group_ptr.to(device=batch.graphs.p_X.device, dtype=torch.long)
        # Product graphs are shared within every semantic group.  Selecting
        # the first contiguous candidate therefore chooses a deterministic
        # base graph without changing the endpoint probabilities or labels.
        selected_candidates = ptr.index_select(0, selected_groups)
        anchors = self._index_graphs(batch.graphs, selected_candidates)
        base_count = int(selected_groups.numel())
        repeat = torch.arange(base_count, device=anchors.p_X.device, dtype=torch.long).repeat_interleave(
            int(config.trajectories_per_group)
        )
        rollout_graphs = self._index_graphs(anchors, repeat)
        rollout_groups = selected_groups.index_select(0, repeat)
        rollout_tokens = batch.guidance_tokens_group.index_select(0, rollout_groups)
        rollout_mask = batch.guidance_token_mask_group.index_select(0, rollout_groups)
        trajectory_count = int(rollout_groups.numel())
        local_group_index = torch.arange(
            trajectory_count, dtype=torch.long, device=anchors.p_X.device
        )
        trace_steps = tuple(
            sorted(
                {
                    int(
                        round(
                            point * (int(config.sampling_steps) - 1)
                            / max(1, int(config.trace_points) - 1)
                        )
                    )
                    for point in range(int(config.trace_points))
                }
            )
        )
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        generator = torch.Generator(device=anchors.p_X.device)
        generator.manual_seed(
            int(config.seed) + 1_000_003 * completed_steps + 10_007 * rank
        )

        def guided_model(**kwargs: object) -> Mapping[str, Tensor]:
            logits = self._student_sampling_forward(**kwargs)
            if not isinstance(logits, Mapping):
                raise TypeError("guided rollout student did not return logits mapping")
            return logits

        sampled = sample_predefined_graph(
            guided_model,
            p_X=rollout_graphs.p_X,
            p_E=rollout_graphs.p_E,
            node_mask=rollout_graphs.node_mask,
            p_mask=rollout_graphs.p_mask,
            cond=rollout_graphs.cond,
            generator=generator,
            config=PredefinedSamplingConfig(steps=config.sampling_steps),
            model_kwargs={
                "guidance_tokens_group": rollout_tokens,
                "guidance_token_mask_group": rollout_mask,
                "guidance_group_index": local_group_index,
            },
            trace_steps=trace_steps,
        )
        if not sampled.traces:
            raise RuntimeError("rollout reference-KL sampler produced no trace states")
        trace_count = len(sampled.traces)

        def repeat_trace(value: Tensor) -> Tensor:
            return value.repeat((trace_count,) + (1,) * (value.ndim - 1))

        self._rollout_replay = {
            "t": torch.cat([self._normal_clone(trace.t) for trace in sampled.traces], dim=0),
            "r_X": torch.cat([self._normal_clone(trace.r_X) for trace in sampled.traces], dim=0),
            "r_E": torch.cat([self._normal_clone(trace.r_E) for trace in sampled.traces], dim=0),
            "p_X": repeat_trace(self._normal_clone(rollout_graphs.p_X)),
            "p_E": repeat_trace(self._normal_clone(rollout_graphs.p_E)),
            "node_mask": repeat_trace(self._normal_clone(rollout_graphs.node_mask)),
            "p_mask": repeat_trace(self._normal_clone(rollout_graphs.p_mask)),
            "cond": repeat_trace(self._normal_clone(rollout_graphs.cond)),
            "tokens": repeat_trace(self._normal_clone(rollout_tokens)),
            "token_mask": repeat_trace(self._normal_clone(rollout_mask)),
            "group_index": local_group_index.repeat(trace_count),
        }
        self._rollout_replay_step = completed_steps
        self._rollout_replay_stats = {
            "ready": 1,
            "state_count": int(trajectory_count * trace_count),
            "trajectory_count": trajectory_count,
            "group_count": base_count,
        }
        return dict(self._rollout_replay_stats)

    def maybe_refresh_rollout_replay(
        self, batch: GuidedStageABatch, *, completed_steps: int
    ) -> bool:
        """Refresh first-step/resume replay and every registered interval."""

        config = self.rollout_reference_kl_config
        if not config.enabled:
            return False
        needs_refresh = (
            self._rollout_replay is None
            or self._rollout_replay_step is None
            or completed_steps == 0
            or completed_steps - self._rollout_replay_step >= config.refresh_interval_steps
        )
        if needs_refresh:
            self.refresh_rollout_replay(batch, completed_steps=completed_steps)
            return True
        return False

    def _rollout_reference_kl_forward(self, batch: GuidedStageABatch) -> RolloutReferenceKLOutput:
        """Differentiate student KL on the current detached on-policy replay."""

        config = self.rollout_reference_kl_config
        zero = batch.graphs.p_X.sum() * 0.0
        if not config.enabled:
            return RolloutReferenceKLOutput(
                loss=zero,
                kl=zero,
                ready=zero,
                state_count=zero,
                trajectory_count=zero,
                group_count=zero,
                refresh_step=zero,
            )
        if self._rollout_replay is None or self._rollout_replay_step is None:
            raise RuntimeError(
                "rollout reference-KL is enabled but replay is absent; formal training must refresh it"
            )
        replay = self._rollout_replay
        student_logits = self._student_train_forward(
            t=replay["t"],
            cond=replay["cond"],
            guidance_tokens_group=replay["tokens"],
            guidance_token_mask_group=replay["token_mask"],
            guidance_group_index=replay["group_index"],
            t_X=replay["r_X"],
            t_E=replay["r_E"],
            p_X=replay["p_X"],
            p_E=replay["p_E"],
            p_mask=replay["p_mask"],
            node_mask=replay["node_mask"],
        )
        with torch.no_grad():
            reference_logits = self._reference_forward(
                t=replay["t"],
                cond=replay["cond"],
                t_X=replay["r_X"],
                t_E=replay["r_E"],
                p_X=replay["p_X"],
                p_E=replay["p_E"],
                p_mask=replay["p_mask"],
                node_mask=replay["node_mask"],
            )
        if not isinstance(student_logits, Mapping) or not isinstance(reference_logits, Mapping):
            raise TypeError("rollout reference-KL models must return logits mappings")
        kl = full_reference_kl_per_graph(
            student_logits,
            reference_logits,
            {"X": replay["node_mask"]},
            edge_weight=float(self.objective.config.edge_weight),
        ).total.mean()
        return RolloutReferenceKLOutput(
            loss=float(config.weight) * kl,
            kl=kl,
            ready=kl.new_tensor(1.0),
            state_count=kl.new_tensor(float(self._rollout_replay_stats["state_count"])),
            trajectory_count=kl.new_tensor(float(self._rollout_replay_stats["trajectory_count"])),
            group_count=kl.new_tensor(float(self._rollout_replay_stats["group_count"])),
            refresh_step=kl.new_tensor(float(self._rollout_replay_step)),
        )

    def _select_positive_recovery_anchors(
        self, batch: GuidedStageABatch
    ) -> tuple[Tensor, Tensor]:
        """Choose one positive endpoint per selected recovery-branch group.

        This is intentionally independent of a dataset ID: the shared
        ``GuidedObjectiveBatch`` representation supplies group contiguity,
        labels, and branch names.  A step-derived CPU generator makes the
        choice recoverable without tying it to the main corruption stream.
        """

        config = self.positive_recovery_config
        branches = batch.supervision.guidance_branch
        if branches is None:
            raise ValueError("positive recovery requires branch-aware supervision")
        ptr = batch.supervision.group_ptr.detach().to(device="cpu", dtype=torch.long)
        labels = batch.supervision.structural_label.detach().to(device="cpu", dtype=torch.long)
        eligible: list[tuple[int, tuple[int, ...]]] = []
        for group_id, branch in enumerate(branches):
            if branch != config.branch:
                continue
            start, stop = int(ptr[group_id]), int(ptr[group_id + 1])
            positives = tuple(index for index in range(start, stop) if int(labels[index]) == 1)
            if positives:
                eligible.append((group_id, positives))
        if not eligible:
            empty = torch.empty(0, device=batch.graphs.p_X.device, dtype=torch.long)
            return empty, empty
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.time_policy_seed) + self._formal_training_step)
        limit = len(eligible) if config.max_groups is None else min(len(eligible), config.max_groups)
        order = torch.randperm(len(eligible), generator=generator).tolist()
        selected_groups: list[int] = []
        selected_candidates: list[int] = []
        for index in order[:limit]:
            group_id, positives = eligible[index]
            position = int(torch.randint(len(positives), (1,), generator=generator).item())
            selected_groups.append(group_id)
            selected_candidates.append(positives[position])
        device = batch.graphs.p_X.device
        return (
            torch.tensor(selected_groups, device=device, dtype=torch.long),
            torch.tensor(selected_candidates, device=device, dtype=torch.long),
        )

    @staticmethod
    def _empty_positive_recovery(
        zero: Tensor,
        *,
        effective_weight: float,
    ) -> PositiveRecoveryOutput:
        return PositiveRecoveryOutput(
            loss=zero,
            ce=zero,
            effective_weight=zero.new_tensor(effective_weight),
            ready=zero,
            state_count=zero,
            common_t=zero,
            common_s=zero,
            node_rollin_change_fraction=zero,
            edge_rollin_change_fraction=zero,
        )

    def _positive_recovery_forward(
        self,
        batch: GuidedStageABatch,
        *,
        generator: torch.Generator,
    ) -> PositiveRecoveryOutput:
        """Apply detached clean-control or REC1 state recovery at one bridge."""

        config = self.positive_recovery_config
        zero = batch.graphs.p_X.sum() * 0.0
        effective_weight = self._positive_recovery_effective_weight()
        if effective_weight == 0.0:
            return self._empty_positive_recovery(zero, effective_weight=effective_weight)
        selected_groups, selected_candidates = self._select_positive_recovery_anchors(batch)
        if selected_groups.numel() == 0:
            return self._empty_positive_recovery(zero, effective_weight=effective_weight)

        anchors = self._index_graphs(batch.graphs, selected_candidates)
        group_count = int(selected_groups.numel())
        device = anchors.p_X.device
        local_group_index = torch.arange(group_count, device=device, dtype=torch.long)
        schedule = positive_recovery_time(config, self._formal_training_step)
        t_group = torch.full((group_count,), schedule.t, device=device, dtype=torch.float32)
        s_group = torch.full((group_count,), schedule.s, device=device, dtype=torch.float32)
        selector_field = sample_coupled_selector_field(
            num_groups=group_count,
            num_nodes=int(anchors.p_X.shape[1]),
            device=device,
            generator=generator,
        )
        bridge_t = build_coupled_predefined_corruption(
            anchors,
            local_group_index,
            group_count,
            generator,
            t_group=t_group,
            selector_field=selector_field,
        )
        bridge_s = build_coupled_predefined_corruption(
            anchors,
            local_group_index,
            group_count,
            generator,
            t_group=s_group,
            selector_field=selector_field,
        )
        tokens = batch.guidance_tokens_group.index_select(0, selected_groups)
        token_mask = batch.guidance_token_mask_group.index_select(0, selected_groups)
        with torch.no_grad():
            bridge_logits = self._student_sampling_forward(
                t=bridge_t.t_graph,
                cond=anchors.cond,
                guidance_tokens_group=tokens,
                guidance_token_mask_group=token_mask,
                guidance_group_index=local_group_index,
                **bridge_t.model_inputs(),
            )
            if config.normalized_mode == "student_rollin":
                sampled = sample_predefined_student_transition(
                    logits=bridge_logits,
                    current_X=bridge_t.t_X,
                    current_E=bridge_t.t_E,
                    p_X=anchors.p_X,
                    p_E=anchors.p_E,
                    node_mask=anchors.node_mask,
                    p_mask=anchors.p_mask,
                    t=bridge_t.t_graph,
                    s=bridge_s.t_graph,
                    generator=generator,
                )
                state_x, state_e = sampled.t_X, sampled.t_E
            else:
                state_x, state_e = bridge_s.t_X.detach(), bridge_s.t_E.detach()
        recovery_logits = self._student_train_forward(
            t=bridge_s.t_graph,
            cond=anchors.cond,
            guidance_tokens_group=tokens,
            guidance_token_mask_group=token_mask,
            guidance_group_index=local_group_index,
            t_X=state_x,
            t_E=state_e,
            p_X=anchors.p_X,
            p_E=anchors.p_E,
            p_mask=anchors.p_mask,
            node_mask=anchors.node_mask,
        )
        ce = full_dfm_per_graph(
            recovery_logits,
            {"X": anchors.r_X, "E": anchors.r_E},
            anchors.node_mask,
            edge_weight=self.objective.config.edge_weight,
        ).total.mean()
        loss = ce * effective_weight
        node_mask = anchors.node_mask.bool()
        edge_mask = node_mask[:, :, None] & node_mask[:, None, :]
        node_changed = (state_x.argmax(dim=-1) != bridge_s.t_X.argmax(dim=-1)) & node_mask
        edge_changed = (state_e.argmax(dim=-1) != bridge_s.t_E.argmax(dim=-1)) & edge_mask
        return PositiveRecoveryOutput(
            loss=loss,
            ce=ce,
            effective_weight=ce.new_tensor(effective_weight),
            ready=ce.new_tensor(1.0),
            state_count=ce.new_tensor(float(group_count)),
            common_t=ce.new_tensor(schedule.t),
            common_s=ce.new_tensor(schedule.s),
            node_rollin_change_fraction=node_changed.float().sum() / node_mask.sum().clamp_min(1),
            edge_rollin_change_fraction=edge_changed.float().sum() / edge_mask.sum().clamp_min(1),
        )

    def _semantic_alignment_forward(self, batch: GuidedStageABatch) -> SemanticAlignmentOutput:
        """Compare registered projector-group embeddings with frozen routes."""

        config = self.semantic_alignment_config
        objective = self.semantic_alignment_objective
        if config is None or objective is None:
            raise RuntimeError("semantic alignment is disabled")
        route = batch.route_semantic_candidate
        if route is None:
            raise RuntimeError("semantic alignment requires frozen route semantic embeddings")
        if route.requires_grad:
            raise ValueError("route semantic embeddings must remain frozen")
        adapter = self.student.guidance_adapter
        if not isinstance(adapter, (GroupedTokenGuidanceAdapter, TokenGuidanceAdapter)):
            raise RuntimeError("semantic alignment requires projected text projectors")
        branches = batch.supervision.guidance_branch
        if branches is None:
            raise ValueError("semantic alignment requires branch-aware supervision")
        return objective(
            group_embeddings=adapter.group_semantic_embeddings(
                batch.guidance_tokens_group, batch.guidance_token_mask_group
            ),
            route_embedding=route.detach(),
            group_ptr=batch.supervision.group_ptr,
            group_index=batch.group_index,
            teacher_probability=batch.supervision.teacher_probability,
            endpoint_probability=batch.supervision.endpoint_probability,
            structural_label=batch.supervision.structural_label,
            guidance_branch=branches,
            listwise_eligible=batch.supervision.listwise_eligible,
            logit_scale=config.logit_scale,
            virtual_microbatch_group_ptr=batch.virtual_microbatch_group_ptr,
        )

    def forward(
        self,
        batch: GuidedStageABatch,
        *,
        generator: torch.Generator,
        t_group: Tensor | None = None,
        selector_field: CoupledSelectorField | None = None,
    ) -> StageBForwardOutput:
        """Run Stage B's reference and guided student on identical corruption."""

        _candidates, groups = self._validate_batch(batch)
        lane_batches = self._virtual_lane_batches(batch)
        lane_corruptions: list[CorruptedGraphBatch] = []
        group_offset = 0
        for lane_batch in lane_batches:
            lane_candidates, lane_groups = self._validate_batch(lane_batch)
            lane_nodes = int(lane_batch.graphs.p_X.shape[1])
            lane_t_group = (
                None
                if t_group is None
                else t_group[group_offset : group_offset + lane_groups]
            )
            lane_selector = None
            if selector_field is not None:
                lane_selector = CoupledSelectorField(
                    node_uniform_group=selector_field.node_uniform_group[
                        group_offset : group_offset + lane_groups, :lane_nodes
                    ],
                    edge_uniform_group=selector_field.edge_uniform_group[
                        group_offset : group_offset + lane_groups,
                        :lane_nodes,
                        :lane_nodes,
                    ],
                )
            lane_corruption = build_coupled_predefined_corruption(
                lane_batch.graphs,
                lane_batch.group_index,
                lane_groups,
                generator,
                t_group=lane_t_group,
                selector_field=lane_selector,
            )
            if int(lane_corruption.t_graph.numel()) != lane_candidates:
                raise AssertionError("virtual lane corruption candidate count changed")
            lane_corruptions.append(lane_corruption)
            group_offset += lane_groups
        if group_offset != groups:
            raise AssertionError("virtual lanes do not cover the merged group batch")
        corruption = self._merge_lane_corruptions(
            tuple(lane_corruptions), nodes=int(batch.graphs.p_X.shape[1])
        )
        with torch.no_grad():
            reference_logits = self._reference_forward(
                t=corruption.t_graph,
                cond=batch.graphs.cond,
                **corruption.model_inputs(),
            )
        student_logits = self._student_train_forward(
            t=corruption.t_graph,
            cond=batch.graphs.cond,
            guidance_tokens_group=batch.guidance_tokens_group,
            guidance_token_mask_group=batch.guidance_token_mask_group,
            guidance_group_index=batch.group_index,
            **corruption.model_inputs(),
        )
        lane_objectives: list[GuidedObjectiveOutput] = []
        candidate_offset = 0
        for lane_batch, lane_corruption in zip(
            lane_batches, lane_corruptions, strict=True
        ):
            lane_candidates = int(lane_corruption.t_graph.numel())
            candidate_stop = candidate_offset + lane_candidates
            lane_nodes = int(lane_batch.graphs.p_X.shape[1])
            lane_objectives.append(
                self.objective(
                    self._slice_logits_for_lane(
                        student_logits,
                        candidate_start=candidate_offset,
                        candidate_stop=candidate_stop,
                        nodes=lane_nodes,
                    ),
                    self._slice_logits_for_lane(
                        reference_logits,
                        candidate_start=candidate_offset,
                        candidate_stop=candidate_stop,
                        nodes=lane_nodes,
                    ),
                    lane_corruption,
                    lane_batch.supervision,
                )
            )
            candidate_offset = candidate_stop
        if candidate_offset != _candidates:
            raise AssertionError("virtual lanes do not cover the merged candidate batch")
        objective = self._merge_lane_objectives(tuple(lane_objectives))
        controller_enabled = (
            self.teacher_semantic_loss_share_controller_config.enabled
        )
        controller_scale = objective.loss.new_tensor(
            self.current_teacher_semantic_loss_scale
        )
        metric_configured_objective = objective.metrics["metric_rank_component"]
        metric_effective_objective = metric_configured_objective
        teacher_semantic_configured_objective = None
        teacher_semantic_effective_objective = None
        if controller_enabled:
            metric_effective_objective = (
                controller_scale * metric_configured_objective
            )
            objective_metrics = dict(objective.metrics)
            objective_metrics["metric_rank_component_configured"] = (
                metric_configured_objective
            )
            objective_metrics["metric_rank_component"] = metric_effective_objective
            objective_metrics["metric_rank_effective_scale"] = controller_scale
            objective = replace(
                objective,
                loss=(
                    objective.loss
                    - metric_configured_objective
                    + metric_effective_objective
                ),
                metrics=objective_metrics,
            )
        loss = objective.loss
        semantic = None
        semantic_weighted_objective = None
        semantic_configured_weighted_objective = None
        if self.semantic_alignment_config is not None:
            semantic = self._semantic_alignment_forward(batch)
            semantic_configured_weighted_objective = (
                self.semantic_alignment_config.weight * semantic.loss
            )
            semantic_weighted_objective = semantic_configured_weighted_objective
            if controller_enabled:
                semantic_weighted_objective = (
                    controller_scale * semantic_configured_weighted_objective
                )
            loss = loss + semantic_weighted_objective
        if controller_enabled:
            semantic_zero = loss.new_zeros(())
            semantic_configured = (
                semantic_zero
                if semantic_configured_weighted_objective is None
                else semantic_configured_weighted_objective
            )
            semantic_effective = (
                semantic_zero
                if semantic_weighted_objective is None
                else semantic_weighted_objective
            )
            teacher_semantic_configured_objective = (
                metric_configured_objective + semantic_configured
            )
            teacher_semantic_effective_objective = (
                metric_effective_objective + semantic_effective
            )
        positive_recovery = None
        if self.positive_recovery_config.enabled:
            if (
                len(lane_batches) > 1
                and self._positive_recovery_effective_weight() > 0.0
            ):
                raise RuntimeError(
                    "virtual microbatch merge reached active positive recovery; "
                    "REC1 lane-RNG equivalence must pass before promotion beyond its warmup"
                )
            positive_recovery = self._positive_recovery_forward(batch, generator=generator)
            loss = loss + positive_recovery.loss
        rollout_reference_kl = None
        if self.rollout_reference_kl_config.enabled:
            rollout_reference_kl = self._rollout_reference_kl_forward(batch)
            loss = loss + rollout_reference_kl.loss
        return StageBForwardOutput(
            loss=loss,
            objective=objective,
            corruption=corruption,
            student_logits=student_logits,
            reference_logits=reference_logits,
            semantic=semantic,
            semantic_weighted_objective=semantic_weighted_objective,
            semantic_configured_weighted_objective=(
                semantic_configured_weighted_objective
            ),
            teacher_semantic_configured_objective=(
                teacher_semantic_configured_objective
            ),
            teacher_semantic_effective_objective=(
                teacher_semantic_effective_objective
            ),
            teacher_semantic_controller_scale=(
                controller_scale if controller_enabled else None
            ),
            positive_recovery=positive_recovery,
            rollout_reference_kl=rollout_reference_kl,
            guidance_residual_scale=loss.new_tensor(
                self._guidance_residual_scale_value
            ),
        )


def build_adapter_stage_b_from_checkpoint(
    config: StageBConfig,
    *,
    base_checkpoint_path: str | Path,
    stage_a_checkpoint_path: str | Path,
    trust_base_checkpoint: bool = False,
    trust_stage_a_checkpoint: bool = False,
    map_location: str = "cpu",
) -> tuple[AdapterStageB, StageBInitializationReport]:
    """Build Stage B from a base GraphDiT and a saved Stage A adapter state."""

    if not isinstance(config, StageBConfig):
        raise TypeError("config must be StageBConfig")
    if config.resolved_semantic_dim != config.resolved_condition_dim:
        raise ValueError(
            "Stage-B projector width must equal generation-model conditioning k; "
            "rebuild Stage A and its native route-embedding cache at that k"
        )
    base_state = load_legacy_state_dict(
        base_checkpoint_path,
        map_location=map_location,
        trust_checkpoint=trust_base_checkpoint,
    )
    stage_a_state = load_legacy_state_dict(
        stage_a_checkpoint_path,
        map_location=map_location,
        trust_checkpoint=trust_stage_a_checkpoint,
    )
    reference = GraphDiT(config.graphdit)
    student = TokenCrossAttentionGraphDiT(
        config.graphdit,
        text_dim=config.guidance_text_dim,
        semantic_dim=config.resolved_semantic_dim,
        condition_dim=config.condition_dim,
        architecture=config.guidance_architecture,
    )
    reference_report = load_graphdit_state_dict(reference, base_state, strict=True)
    student_report = load_graphdit_state_dict(student, base_state, strict=False)
    non_adapter_missing = tuple(
        name for name in student_report.missing_keys if not name.startswith("guidance_adapter.")
    )
    if student_report.unexpected_keys or non_adapter_missing:
        raise RuntimeError(
            "base checkpoint did not initialize the complete Stage B backbone: "
            f"missing={non_adapter_missing}, unexpected={student_report.unexpected_keys}"
        )
    transferred_projectors = _initialize_projectors(
        student,
        _adapter_projector_state_from_checkpoint(stage_a_state),
    )
    stage = AdapterStageB(
        reference=reference,
        student=student,
        objective=GuidanceObjective(config.objective),
        freeze_transferred_projectors=config.freeze_transferred_projectors,
        semantic_alignment=config.semantic_alignment,
        teacher_semantic_loss_share_controller=(
            config.teacher_semantic_loss_share_controller
        ),
        positive_recovery=config.positive_recovery,
        rollout_reference_kl=config.rollout_reference_kl,
        guidance_residual_schedule=config.guidance_residual_schedule,
    )
    report = StageBInitializationReport(
        reference_load=reference_report,
        student_load=student_report,
        transferred_projector_names=transferred_projectors,
        fresh_condition_adapter_names=_condition_adapter_parameter_names(
            student.guidance_adapter
        ),
        fresh_cross_attention_names=_cross_attention_parameter_names(student.guidance_adapter),
        trainable_parameter_names=stage.trainable_parameter_names(),
        architecture_contract=student.architecture_contract(),
    )
    return stage, report


def build_adapter_stage_b_from_stage_a(
    config: StageBConfig,
    stage_a: AdapterOnlyStageA,
) -> tuple[AdapterStageB, StageBInitializationReport]:
    """Create fresh Stage B generation adapters directly from an in-memory Stage A.

    This is useful for bounded smoke tests and for a trainer that keeps the
    completed Stage A module resident.  The Stage A student itself is never
    reused as the Stage B student, so its cross-attention state cannot leak
    across the stage boundary.
    """

    if not isinstance(config, StageBConfig):
        raise TypeError("config must be StageBConfig")
    if config.resolved_semantic_dim != config.resolved_condition_dim:
        raise ValueError(
            "Stage-B projector width must equal generation-model conditioning k"
        )
    if not isinstance(stage_a, AdapterOnlyStageA):
        raise TypeError("stage_a must be AdapterOnlyStageA")
    if stage_a.graphdit_config != config.graphdit:
        raise ValueError("Stage A and Stage B GraphDiT configs differ")
    base_state = stage_a.reference.state_dict()
    reference = GraphDiT(config.graphdit)
    student = TokenCrossAttentionGraphDiT(
        config.graphdit,
        text_dim=config.guidance_text_dim,
        semantic_dim=config.resolved_semantic_dim,
        condition_dim=config.condition_dim,
        architecture=config.guidance_architecture,
    )
    reference_report = load_graphdit_state_dict(reference, base_state, strict=True)
    student_report = load_graphdit_state_dict(student, base_state, strict=False)
    non_adapter_missing = tuple(
        name for name in student_report.missing_keys if not name.startswith("guidance_adapter.")
    )
    if student_report.unexpected_keys or non_adapter_missing:
        raise RuntimeError(
            "Stage A reference did not initialize the complete Stage B backbone: "
            f"missing={non_adapter_missing}, unexpected={student_report.unexpected_keys}"
        )
    transferred_projectors = _initialize_projectors(
        student, extract_stage_a_projector_state(stage_a)
    )
    stage = AdapterStageB(
        reference=reference,
        student=student,
        objective=GuidanceObjective(config.objective),
        freeze_transferred_projectors=config.freeze_transferred_projectors,
        semantic_alignment=config.semantic_alignment,
        teacher_semantic_loss_share_controller=(
            config.teacher_semantic_loss_share_controller
        ),
        positive_recovery=config.positive_recovery,
        rollout_reference_kl=config.rollout_reference_kl,
        guidance_residual_schedule=config.guidance_residual_schedule,
    )
    report = StageBInitializationReport(
        reference_load=reference_report,
        student_load=student_report,
        transferred_projector_names=transferred_projectors,
        fresh_condition_adapter_names=_condition_adapter_parameter_names(
            student.guidance_adapter
        ),
        fresh_cross_attention_names=_cross_attention_parameter_names(student.guidance_adapter),
        trainable_parameter_names=stage.trainable_parameter_names(),
        architecture_contract=student.architecture_contract(),
    )
    return stage, report


__all__ = [
    "AdapterStageB",
    "StageBConfig",
    "StageBForwardOutput",
    "StageBInitializationReport",
    "SemanticAlignmentConfig",
    "TeacherSemanticLossShareControllerConfig",
    "RolloutReferenceKLConfig",
    "PositiveRecoveryOutput",
    "RolloutReferenceKLOutput",
    "build_adapter_stage_b_from_checkpoint",
    "build_adapter_stage_b_from_stage_a",
    "extract_stage_a_projector_state",
    "standard_stage_b_objective",
    "standard_stage_b_positive_recovery",
    "standard_stage_b_rollout_reference_kl",
    "standard_stage_b_semantic_alignment",
    "standard_stage_b_teacher_semantic_loss_share_controller",
]
