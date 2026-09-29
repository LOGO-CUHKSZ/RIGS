"""Locked, dataset-neutral guidance training recipes.

The Markdown recipe is the normative human-readable contract.  This module is
its machine-enforced registry: task configs select a recipe and an allowed
alpha policy, while dataset paths, model widths, and resource choices remain
outside the registry.  A task config may repeat immutable values for audit
purposes, but it may not change them.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import math

from retro_guided_generation.models.dfm import PositiveRecoveryConfig
from retro_guided_generation.models.objectives import GuidanceObjectiveConfig
from retro_guided_generation.models.stages import (
    GuidanceResidualScheduleConfig,
    ProjectorOnlyStageAConfig,
    RolloutReferenceKLConfig,
    TeacherSemanticLossShareControllerConfig,
)


UNIVERSAL_ALPHA_FIXED_VS_COSINE_RECIPE_ID = (
    "universal_alpha_fixed_vs_cosine_training_recipe_v1"
)
UNIVERSAL_ALPHA_FIXED_VS_COSINE_RECIPE_SOURCE = (
    "training&val-guidance/"
    "universal_alpha_fixed_vs_cosine_training_recipe_v1.md"
)

FIXED_ALPHA025_POLICY = "fixed_alpha025"
COSINE_ALPHA025_TO015_POLICY = "cosine_alpha025_to015"
FIXED_ALPHA1_POLICY = "fixed_alpha1"
STAGE_B_ALPHA_POLICIES = frozenset(
    {FIXED_ALPHA025_POLICY, COSINE_ALPHA025_TO015_POLICY, FIXED_ALPHA1_POLICY}
)


@dataclass(frozen=True, slots=True)
class GuidanceTrainingRecipe:
    """One validated selection from the locked guidance recipe registry."""

    recipe_id: str
    stage: str
    alpha_policy: str | None = None

    def __post_init__(self) -> None:
        if self.recipe_id != UNIVERSAL_ALPHA_FIXED_VS_COSINE_RECIPE_ID:
            raise ValueError(f"unknown guidance training recipe {self.recipe_id!r}")
        if self.stage not in {"stage-a", "stage-b"}:
            raise ValueError("guidance training recipe stage must be stage-a or stage-b")
        if self.stage == "stage-a":
            if self.alpha_policy is not None:
                raise ValueError("Stage A recipe selection cannot define alpha_policy")
        elif self.alpha_policy not in STAGE_B_ALPHA_POLICIES:
            raise ValueError(
                "Stage B recipe alpha_policy must be fixed_alpha025, "
                "cosine_alpha025_to015, or the user-confirmed "
                "experiment-local fixed_alpha1 override"
            )

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        stage: str,
    ) -> "GuidanceTrainingRecipe":
        if not isinstance(value, Mapping):
            raise ValueError("training_recipe must be an object")
        unknown = set(value) - {"id", "alpha_policy"}
        if unknown:
            raise ValueError(
                f"training_recipe contains unsupported fields: {sorted(unknown)}"
            )
        recipe_id = value.get("id")
        if not isinstance(recipe_id, str) or not recipe_id.strip():
            raise ValueError("training_recipe.id must be a non-empty string")
        alpha_policy = value.get("alpha_policy")
        if alpha_policy is not None and not isinstance(alpha_policy, str):
            raise ValueError("training_recipe.alpha_policy must be a string")
        return cls(
            recipe_id=recipe_id.strip(),
            stage=stage,
            alpha_policy=(
                None if alpha_policy is None else alpha_policy.strip().lower()
            ),
        )

    def stage_b_objective(self, selected_branch: str) -> GuidanceObjectiveConfig:
        self._require_stage_b()
        if selected_branch not in {"structure", "metric"}:
            raise ValueError("selected_branch must be structure or metric")
        return GuidanceObjectiveConfig(
            edge_weight=10.0,
            temperature=0.5,
            candidate_score_calibration="none",
            candidate_score_scale_floor=1.0,
            common_weight=0.0,
            listwise_weight=0.05,
            anchor_weight=0.0,
            hybrid_branch_objective=True,
            structure_guidance_weight=(
                1.0 if selected_branch == "structure" else 0.0
            ),
            metric_guidance_weight=(1.0 if selected_branch == "metric" else 0.0),
            metric_guidance_mode=(
                "teacher_listwise_kl" if selected_branch == "metric" else "none"
            ),
            preservation_mode="reference_kl",
            positive_dfm_weight=0.0,
            trust_region_weight=0.0,
            common_kl_budget=0.0,
            full_reference_kl_guard_weight=0.05,
            full_reference_kl_budget=0.02,
            candidate_supervision_mode="positive_rank",
            positive_supervision_weight=0.10,
            negative_supervision_weight=0.0,
            negative_reverse_weight=0.0,
            route_ranking_weight=0.05,
            route_ranking_margin=0.10,
        )

    def stage_b_positive_recovery(
        self, selected_branch: str
    ) -> PositiveRecoveryConfig:
        self._require_stage_b()
        if selected_branch not in {"structure", "metric"}:
            raise ValueError("selected_branch must be structure or metric")
        return PositiveRecoveryConfig(
            mode="disabled",
            weight=0.0,
            branch=selected_branch,
        )

    def stage_b_rollout_reference_kl(self) -> RolloutReferenceKLConfig:
        self._require_stage_b()
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

    def stage_b_teacher_semantic_loss_share_controller(
        self,
    ) -> TeacherSemanticLossShareControllerConfig:
        """Return the universal one-sided 40/45/50-percent controller."""

        self._require_stage_b()
        return TeacherSemanticLossShareControllerConfig(enabled=True)

    def stage_b_guidance_residual_schedule(
        self,
    ) -> GuidanceResidualScheduleConfig:
        self._require_stage_b()
        if self.alpha_policy == FIXED_ALPHA025_POLICY:
            return GuidanceResidualScheduleConfig(
                mode="fixed", start=0.25, end=0.25
            )
        if self.alpha_policy == COSINE_ALPHA025_TO015_POLICY:
            return GuidanceResidualScheduleConfig(
                mode="cosine", start=0.25, end=0.15
            )
        if self.alpha_policy == FIXED_ALPHA1_POLICY:
            return GuidanceResidualScheduleConfig(
                mode="fixed", start=1.0, end=1.0
            )
        raise AssertionError("validated Stage-B alpha policy is absent")

    def validate_projector_only_stage_a(
        self, config: ProjectorOnlyStageAConfig
    ) -> None:
        if self.stage != "stage-a":
            raise RuntimeError("projector Stage-A validation requires a Stage-A recipe")
        expected = {
            "listwise_weight": 1.0,
            "pair_weight": 0.5,
            "pair_margin": 0.1,
            "identity_weight": 0.0,
            "endpoint_probability_floor": 1.0e-8,
            "logit_scale": 10.0,
        }
        mismatched = {
            name: {"expected": target, "observed": float(getattr(config, name))}
            for name, target in expected.items()
            if not math.isclose(
                float(getattr(config, name)), target, rel_tol=0.0, abs_tol=0.0
            )
        }
        if mismatched:
            raise ValueError(
                "projector-only Stage A overrides immutable recipe fields: "
                f"{mismatched}"
            )

    def resolved_contract(self, selected_branch: str | None = None) -> dict[str, object]:
        result: dict[str, object] = {
            "id": self.recipe_id,
            "source_document": UNIVERSAL_ALPHA_FIXED_VS_COSINE_RECIPE_SOURCE,
            "stage": self.stage,
            "alpha_policy": self.alpha_policy,
        }
        if self.stage == "stage-a":
            result["immutable_stage_a"] = {
                "trainable_scope": "shared_text_projector_only",
                "listwise_weight": 1.0,
                "pair_weight": 0.5,
                "pair_margin": 0.1,
                "identity_weight": 0.0,
                "identity_objective_enabled": False,
                "shuffled_identity_pairs_materialized": False,
                "endpoint_probability_floor": 1.0e-8,
                "logit_scale": 10.0,
            }
        else:
            if selected_branch is None:
                raise ValueError("Stage-B resolved recipe requires selected_branch")
            result["immutable_stage_b"] = {
                "objective": asdict(self.stage_b_objective(selected_branch)),
                "semantic_alignment": None,
                "teacher_semantic_loss_share_controller": asdict(
                    self.stage_b_teacher_semantic_loss_share_controller()
                ),
                "positive_recovery": asdict(
                    self.stage_b_positive_recovery(selected_branch)
                ),
                "rollout_reference_kl": asdict(
                    self.stage_b_rollout_reference_kl()
                ),
                "guidance_residual_schedule": asdict(
                    self.stage_b_guidance_residual_schedule()
                ),
            }
        return result

    def _require_stage_b(self) -> None:
        if self.stage != "stage-b":
            raise RuntimeError("requested Stage-B fields from a Stage-A recipe")
