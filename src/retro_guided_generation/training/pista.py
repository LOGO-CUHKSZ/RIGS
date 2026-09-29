"""Runnable Pista Stage A/B specifications and bounded preflight helpers."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from retro_guided_generation.data.runtime.pista import (
    DEFAULT_CPU_BATCH_WORKERS,
    DEFAULT_MAX_OPEN_GRAPH_SHARDS,
    PistaStageABatchProvider,
    PistaVirtualMicrobatchMerge,
)
from retro_guided_generation.guidance.io import ensure_output_directory, write_json
from retro_guided_generation.models.dfm import PositiveRecoveryConfig
from retro_guided_generation.models.graphdit import GraphDiTConfig
from retro_guided_generation.models.guidance.architecture import (
    resolve_guidance_architecture,
)
from retro_guided_generation.models.objectives import GuidanceObjectiveConfig
from retro_guided_generation.models.stages import (
    GuidanceResidualScheduleConfig,
    ProjectorOnlyStageAConfig,
    RolloutReferenceKLConfig,
    StageAConfig,
    StageBConfig,
    SemanticAlignmentConfig,
    TeacherSemanticLossShareControllerConfig,
    build_adapter_only_stage_a_from_checkpoint,
    build_adapter_stage_b_from_checkpoint,
    build_projector_only_stage_a,
    standard_stage_b_objective,
    standard_stage_b_positive_recovery,
    standard_stage_b_rollout_reference_kl,
    standard_stage_b_semantic_alignment,
    standard_stage_b_teacher_semantic_loss_share_controller,
)
from retro_guided_generation.training.formal import (
    FormalTrainingConfig,
    FormalTrainingResult,
    configure_deterministic_execution,
    initialize_distributed_training,
    resolve_formal_training_plan,
    run_formal_training,
)
from retro_guided_generation.training.epochs import (
    epoch_value_json,
    parse_epoch_value,
)
from retro_guided_generation.training.guidance_recipes import (
    GuidanceTrainingRecipe,
)
from retro_guided_generation.training.loop import GuidedTrainingConfig, run_guided_training


PISTA_TRAINING_SCHEMA = "retro_guided_generation.pista_training.v1"


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _path(value: object, name: str, *, must_exist: bool) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty path string")
    path = Path(value).expanduser()
    if must_exist:
        return path.resolve(strict=True)
    return Path(os.path.abspath(path))


def _optional_path(value: object, name: str) -> Path | None:
    if value is None:
        return None
    return _path(value, name, must_exist=True)


def _bool(value: object, name: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _integer(value: object, name: str, *, default: int | None = None) -> int:
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _formal_training_config(
    *,
    stage: str,
    value: object,
    training: GuidedTrainingConfig,
) -> FormalTrainingConfig | None:
    if value is None:
        return None
    formal = _mapping(value, "training config.run.formal")
    budget: dict[str, object]
    if training.budget_mode == "epochs":
        if "candidate_steps" in formal:
            raise ValueError("epoch training uses run.formal.candidate_epochs")
        raw_candidates = formal.get("candidate_epochs")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError(
                "training config.run.formal.candidate_epochs must be a non-empty list"
            )
        budget = {
            "max_epochs": training.max_epochs,
            "candidate_epochs": tuple(
                parse_epoch_value(
                    item,
                    f"training config.run.formal.candidate_epochs[{index}]",
                )
                for index, item in enumerate(raw_candidates)
            ),
            "first_epoch": training.first_epoch,
            "recovery_interval_epochs": (
                None
                if formal.get("recovery_interval_epochs") is None
                else parse_epoch_value(
                    formal.get("recovery_interval_epochs"),
                    "training config.run.formal.recovery_interval_epochs",
                )
            ),
            "heartbeat_interval_epochs": (
                None
                if formal.get("heartbeat_interval_epochs") is None
                else parse_epoch_value(
                    formal.get("heartbeat_interval_epochs"),
                    "training config.run.formal.heartbeat_interval_epochs",
                )
            ),
        }
    else:
        if "candidate_epochs" in formal:
            raise ValueError("legacy step training uses run.formal.candidate_steps")
        raw_candidates = formal.get("candidate_steps")
        if not isinstance(raw_candidates, list):
            raise ValueError("training config.run.formal.candidate_steps must be a list")
        budget = {
            "max_steps": training.max_steps,
            "candidate_steps": tuple(
                _integer(
                    item, f"training config.run.formal.candidate_steps[{index}]"
                )
                for index, item in enumerate(raw_candidates)
            ),
            "epoch": training.epoch,
            "recovery_interval_steps": _integer(
                formal.get("recovery_interval_steps"),
                "training config.run.formal.recovery_interval_steps",
                default=1_000,
            ),
            "heartbeat_interval_steps": _integer(
                formal.get("heartbeat_interval_steps"),
                "training config.run.formal.heartbeat_interval_steps",
                default=100,
            ),
        }
    return FormalTrainingConfig(
        stage=stage,
        output_dir=_path(
            formal.get("output_dir"),
            "training config.run.formal.output_dir",
            must_exist=False,
        ),
        expected_world_size=_integer(
            formal.get("expected_world_size"),
            "training config.run.formal.expected_world_size",
        ),
        per_rank_batch_groups=_integer(
            formal.get("per_rank_batch_groups"),
            "training config.run.formal.per_rank_batch_groups",
        ),
        global_physical_batch_groups=_integer(
            formal.get("global_physical_batch_groups"),
            "training config.run.formal.global_physical_batch_groups",
        ),
        gradient_accumulation_steps=_integer(
            formal.get("gradient_accumulation_steps"),
            "training config.run.formal.gradient_accumulation_steps",
            default=1,
        ),
        learning_rate=training.learning_rate,
        weight_decay=training.weight_decay,
        amsgrad=_bool(
            formal.get("amsgrad"),
            "training config.run.formal.amsgrad",
            default=False,
        ),
        gradient_clip_norm=training.gradient_clip_norm,
        seed=training.seed,
        precision=str(formal.get("precision", "none")),
        resume=_bool(formal.get("resume"), "training config.run.formal.resume", default=True),
        deterministic=_bool(
            formal.get("deterministic"),
            "training config.run.formal.deterministic",
            default=False,
        ),
        **budget,
    )


@dataclass(frozen=True, slots=True)
class PistaTrainingSpec:
    """All explicit inputs needed by the local Pista training runner."""

    stage: str
    corpus_root: Path
    graph_cache_root: Path
    token_cache_root: Path
    guidance_text_table: Path
    token_extension_root: Path | None
    base_checkpoint: Path
    stage_a_checkpoint: Path | None
    output_checkpoint: Path | None
    graphdit: GraphDiTConfig
    guidance_text_dim: int
    guidance_architecture: str | None
    objective: GuidanceObjectiveConfig
    training: GuidedTrainingConfig
    device: str
    branch_weights: Mapping[str, float]
    sampler_mode: str
    max_groups_per_batch: int
    max_candidates_per_batch: int
    max_open_graph_shards: int
    max_padded_graph_cells_per_batch: int | None
    trust_base_checkpoint: bool
    trust_stage_a_checkpoint: bool
    semantic_route_embedding_dim: int
    cpu_batch_workers: int = DEFAULT_CPU_BATCH_WORKERS
    cost_balanced_rank_batching: bool = False
    virtual_microbatch_merge: PistaVirtualMicrobatchMerge | None = None
    formal_training: FormalTrainingConfig | None = None
    semantic_route_cache_root: Path | None = None
    projector_only_metadata_root: Path | None = None
    semantic_alignment: SemanticAlignmentConfig | None = None
    teacher_semantic_loss_share_controller: TeacherSemanticLossShareControllerConfig = field(
        default_factory=TeacherSemanticLossShareControllerConfig
    )
    positive_recovery: PositiveRecoveryConfig = field(default_factory=PositiveRecoveryConfig)
    rollout_reference_kl: RolloutReferenceKLConfig = field(
        default_factory=RolloutReferenceKLConfig
    )
    stage_a_implementation: str = "adapter_only_graphdit_v1"
    projector_only_stage_a: ProjectorOnlyStageAConfig | None = None
    condition_dim: int | None = None
    selected_branch: str | None = None
    training_recipe: GuidanceTrainingRecipe | None = None

    def __post_init__(self) -> None:
        if self.stage not in {"stage-a", "stage-b"}:
            raise ValueError("stage must be 'stage-a' or 'stage-b'")
        if self.training_recipe is not None:
            if not isinstance(self.training_recipe, GuidanceTrainingRecipe):
                raise TypeError(
                    "training_recipe must be GuidanceTrainingRecipe or None"
                )
            if self.training_recipe.stage != self.stage:
                raise ValueError("training_recipe stage must match the Pista stage")
            if (
                self.stage == "stage-a"
                and self.stage_a_implementation
                != "projector_only_embedding_alignment_v1"
            ):
                raise ValueError(
                    "the universal training_recipe requires projector-only Stage A"
                )
        if self.stage == "stage-b" and self.stage_a_checkpoint is None:
            raise ValueError("stage-b requires stage_a_checkpoint")
        if not isinstance(self.positive_recovery, PositiveRecoveryConfig):
            raise TypeError("positive_recovery must be PositiveRecoveryConfig")
        if not isinstance(
            self.teacher_semantic_loss_share_controller,
            TeacherSemanticLossShareControllerConfig,
        ):
            raise TypeError(
                "teacher_semantic_loss_share_controller must be "
                "TeacherSemanticLossShareControllerConfig"
            )
        if not isinstance(self.rollout_reference_kl, RolloutReferenceKLConfig):
            raise TypeError("rollout_reference_kl must be RolloutReferenceKLConfig")
        if not isinstance(self.cost_balanced_rank_batching, bool):
            raise TypeError("cost_balanced_rank_batching must be a bool")
        if self.virtual_microbatch_merge is not None:
            if not isinstance(self.virtual_microbatch_merge, PistaVirtualMicrobatchMerge):
                raise TypeError(
                    "virtual_microbatch_merge must be PistaVirtualMicrobatchMerge or None"
                )
            if not (
                self.stage == "stage-a"
                and self.stage_a_implementation == "projector_only_embedding_alignment_v1"
            ):
                raise ValueError(
                    "virtual microbatch merge is only valid for projector-only Stage A"
                )
            if self.cost_balanced_rank_batching:
                raise ValueError(
                    "virtual microbatch merge is incompatible with cost-balanced rank batching"
                )
        if self.stage == "stage-a" and self.semantic_alignment is not None:
            raise ValueError("semantic alignment is only supported by Stage B")
        if (
            self.stage == "stage-a"
            and self.teacher_semantic_loss_share_controller.enabled
        ):
            raise ValueError(
                "teacher/semantic loss-share control is only supported by Stage B"
            )
        if self.stage == "stage-a" and self.positive_recovery.enabled:
            raise ValueError("positive recovery is only supported by Stage B")
        if self.stage == "stage-a" and self.rollout_reference_kl.enabled:
            raise ValueError("rollout reference-KL is only supported by Stage B")
        if self.stage_a_implementation not in {
            "adapter_only_graphdit_v1",
            "projector_only_embedding_alignment_v1",
        }:
            raise ValueError("unsupported stage_a_implementation")
        requires_standard_topology = (
            self.stage == "stage-b"
            or self.stage_a_implementation == "projector_only_embedding_alignment_v1"
        )
        if requires_standard_topology and (
            not isinstance(self.guidance_architecture, str)
            or not self.guidance_architecture.strip()
        ):
            raise ValueError(
                "standard Stage A/B requires an explicit registered guidance_architecture ID"
            )
        if requires_standard_topology and self.selected_branch not in {"structure", "metric"}:
            raise ValueError(
                "standard Stage A/B requires selected_branch to be structure or metric"
            )
        if requires_standard_topology:
            architecture = resolve_guidance_architecture(self.guidance_architecture)
            if not architecture.projector_groups:
                raise ValueError(
                    "standard Stage A/B requires a projected-token architecture "
                    "with transferable projector groups"
                )
        if self.stage == "stage-b" and self.stage_a_implementation != "adapter_only_graphdit_v1":
            raise ValueError("stage_a_implementation is only selectable for Stage A")
        if self.stage == "stage-a":
            if self.stage_a_implementation == "projector_only_embedding_alignment_v1":
                if not isinstance(self.projector_only_stage_a, ProjectorOnlyStageAConfig):
                    raise TypeError(
                        "projector-only Stage A requires projector_only_stage_a configuration"
                    )
                if self.semantic_route_cache_root is None:
                    raise ValueError("projector-only Stage A requires paths.semantic_route_cache_root")
            elif self.projector_only_stage_a is not None:
                raise ValueError(
                    "projector_only_stage_a may only accompany projector_only_embedding_alignment_v1"
                )
        elif self.projector_only_stage_a is not None:
            raise ValueError("projector_only_stage_a is only valid for Stage A")
        if self.projector_only_metadata_root is not None:
            if not (
                self.stage == "stage-a"
                and self.stage_a_implementation == "projector_only_embedding_alignment_v1"
            ):
                raise ValueError(
                    "projector_only_metadata_root is only valid for projector-only Stage A"
                )
            if self.semantic_route_cache_root is None:
                raise ValueError(
                    "projector_only_metadata_root requires paths.semantic_route_cache_root"
                )
        if not isinstance(self.graphdit, GraphDiTConfig):
            raise TypeError("graphdit must be GraphDiTConfig")
        if isinstance(self.guidance_text_dim, bool) or not isinstance(self.guidance_text_dim, int):
            raise ValueError("guidance_text_dim must be an integer")
        if self.guidance_text_dim != 4096:
            raise ValueError("the supplied Pista Qwen token cache requires guidance_text_dim=4096")
        if (
            isinstance(self.semantic_route_embedding_dim, bool)
            or not isinstance(self.semantic_route_embedding_dim, int)
            or self.semantic_route_embedding_dim < 1
        ):
            raise ValueError("semantic_route_embedding_dim must be a positive integer")
        if self.condition_dim is not None and (
            isinstance(self.condition_dim, bool)
            or not isinstance(self.condition_dim, int)
            or self.condition_dim < 1
        ):
            raise ValueError("condition_dim must be a positive integer or None")
        if (
            requires_standard_topology
            and self.semantic_route_embedding_dim != self.resolved_condition_dim
        ):
            raise ValueError(
                "standard Stage A/B semantic/projector width must equal generation model k"
            )
        if not isinstance(self.objective, GuidanceObjectiveConfig):
            raise TypeError("objective must be GuidanceObjectiveConfig")
        if not isinstance(self.training, GuidedTrainingConfig):
            raise TypeError("training must be GuidedTrainingConfig")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty device string")
        if set(self.branch_weights) - {"metric", "structure"}:
            raise ValueError("branch_weights has unknown branches")
        weights = {branch: float(self.branch_weights.get(branch, 0.0)) for branch in ("metric", "structure")}
        if any(value < 0.0 for value in weights.values()) or sum(weights.values()) <= 0.0:
            raise ValueError("branch_weights must be non-negative with positive total")
        if self.stage == "stage-b":
            metric_mode = str(self.objective.metric_guidance_mode).strip().lower()
            metric_weight = float(self.objective.metric_guidance_weight)
            if metric_mode != "none" and metric_weight <= 0.0:
                raise ValueError(
                    "enabled Stage-B metric guidance requires a positive metric_guidance_weight"
                )
            sampled_branches = {branch for branch, weight in weights.items() if weight > 0.0}
            active_branches = {
                branch
                for branch, weight in (
                    ("structure", self.objective.structure_guidance_weight),
                    ("metric", self.objective.metric_guidance_weight),
                )
                if float(weight) > 0.0
            }
            if sampled_branches != active_branches:
                raise ValueError(
                    "Stage-B data branch weights must exactly match the objective's active branches"
                )
        if self.sampler_mode not in {"hierarchical", "branch_cycle"}:
            raise ValueError("sampler_mode must be 'hierarchical' or 'branch_cycle'")
        for name, value in (
            ("max_groups_per_batch", self.max_groups_per_batch),
            ("max_candidates_per_batch", self.max_candidates_per_batch),
            ("max_open_graph_shards", self.max_open_graph_shards),
            ("cpu_batch_workers", self.cpu_batch_workers),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_padded_graph_cells_per_batch is not None and (
            isinstance(self.max_padded_graph_cells_per_batch, bool)
            or not isinstance(self.max_padded_graph_cells_per_batch, int)
            or self.max_padded_graph_cells_per_batch < 1
        ):
            raise ValueError("max_padded_graph_cells_per_batch must be positive or None")
        if self.formal_training is not None:
            if not isinstance(self.formal_training, FormalTrainingConfig):
                raise TypeError("formal_training must be FormalTrainingConfig or None")
            if self.formal_training.stage != self.stage:
                raise ValueError("formal_training.stage must match the Pista stage")
            if self.formal_training.per_rank_batch_groups != self.max_groups_per_batch:
                raise ValueError(
                    "formal per_rank_batch_groups must equal data.max_groups_per_batch"
                )
        if self.semantic_alignment is not None:
            if not isinstance(self.semantic_alignment, SemanticAlignmentConfig):
                raise TypeError("semantic_alignment must be SemanticAlignmentConfig or None")
            if self.semantic_route_cache_root is None:
                raise ValueError("semantic alignment requires paths.semantic_route_cache_root")
        if self.selected_branch is not None:
            if self.selected_branch not in {"structure", "metric"}:
                raise ValueError("selected_branch must be structure, metric, or None")
            expected = {"structure": 0.0, "metric": 0.0}
            expected[self.selected_branch] = 1.0
            observed = {
                branch: float(self.branch_weights.get(branch, 0.0))
                for branch in ("structure", "metric")
            }
            if observed != expected:
                raise ValueError(
                    "selected_branch requires an exact one-hot structure/metric branch contract"
                )
        if self.training_recipe is not None and self.stage == "stage-b":
            if self.selected_branch not in {"structure", "metric"}:
                raise ValueError(
                    "Stage-B training_recipe requires selected_branch"
                )
            if self.objective != self.training_recipe.stage_b_objective(
                self.selected_branch
            ):
                raise ValueError(
                    "Stage-B objective differs from immutable training_recipe"
                )
            if self.semantic_alignment is not None:
                raise ValueError(
                    "Stage-B semantic alignment differs from immutable training_recipe"
                )
            if self.teacher_semantic_loss_share_controller != (
                self.training_recipe.stage_b_teacher_semantic_loss_share_controller()
            ):
                raise ValueError(
                    "Stage-B teacher/semantic controller differs from immutable training_recipe"
                )
            if self.positive_recovery != (
                self.training_recipe.stage_b_positive_recovery(
                    self.selected_branch
                )
            ):
                raise ValueError(
                    "Stage-B positive recovery differs from immutable training_recipe"
                )
            if self.rollout_reference_kl != (
                self.training_recipe.stage_b_rollout_reference_kl()
            ):
                raise ValueError(
                    "Stage-B rollout Reference-KL differs from immutable training_recipe"
                )
        if self.stage_a_implementation == "projector_only_embedding_alignment_v1":
            projector = self.projector_only_stage_a
            if projector is None:  # pragma: no cover - checked above
                raise AssertionError("missing projector-only Stage-A config")
            if projector.guidance_text_dim != self.guidance_text_dim:
                raise ValueError("Stage-A projector text width differs from Pista token width")
            if projector.semantic_dim != self.semantic_route_embedding_dim:
                raise ValueError("Stage-A projector output width differs from semantic cache width")
            if projector.semantic_dim != self.resolved_condition_dim:
                raise ValueError(
                    "Stage-A projector output width differs from generation model k"
                )
            if self.selected_branch != projector.selected_branch:
                raise ValueError("projector-only Stage A selected_branch must match the hard data branch")
            if self.training_recipe is not None:
                self.training_recipe.validate_projector_only_stage_a(projector)

    @property
    def resolved_condition_dim(self) -> int:
        """Resolve cross-attention width ``k`` from the generation model."""

        return self.graphdit.x_dim if self.condition_dim is None else self.condition_dim


def load_pista_training_spec(path: str | Path) -> PistaTrainingSpec:
    """Load a non-secret YAML specification with no implicit default paths."""

    config_path = Path(path).expanduser().resolve(strict=True)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"invalid Pista training YAML: {error}") from error
    config = _mapping(raw, "training config")
    if config.get("schema_version") != PISTA_TRAINING_SCHEMA:
        raise ValueError(f"training config schema_version must be {PISTA_TRAINING_SCHEMA!r}")
    stage = config.get("stage")
    if stage not in {"stage-a", "stage-b"}:
        raise ValueError("training config stage must be 'stage-a' or 'stage-b'")
    recipe_raw = config.get("training_recipe")
    training_recipe = (
        None
        if recipe_raw is None
        else GuidanceTrainingRecipe.from_mapping(recipe_raw, stage=str(stage))
    )
    paths = _mapping(config.get("paths"), "training config.paths")
    model = _mapping(config.get("model"), "training config.model")
    data = _mapping(config.get("data"), "training config.data")
    run = _mapping(config.get("run"), "training config.run")
    training_raw = _mapping(run.get("training"), "training config.run.training")
    training = GuidedTrainingConfig(**dict(training_raw))
    graphdit_raw = _mapping(model.get("graphdit"), "training config.model.graphdit")
    stage_a_implementation = str(
        model.get("stage_a_implementation", "adapter_only_graphdit_v1")
    )
    selected_branch = (
        None
        if model.get("selected_branch") is None
        else str(model.get("selected_branch"))
    )
    guidance_architecture = model.get("guidance_architecture")
    requires_standard_topology = (
        stage == "stage-b"
        or stage_a_implementation == "projector_only_embedding_alignment_v1"
    )
    if requires_standard_topology and (
        not isinstance(guidance_architecture, str)
        or not guidance_architecture.strip()
    ):
        raise ValueError(
            "standard Stage A/B requires model.guidance_architecture to be an "
            "explicit registered architecture ID"
        )
    if requires_standard_topology and selected_branch not in {"structure", "metric"}:
        raise ValueError(
            "standard Stage A/B requires model.selected_branch to be structure or metric"
        )
    objective_raw = model.get("objective")
    if stage == "stage-b":
        if training_recipe is None:
            objective = standard_stage_b_objective(str(selected_branch))
            if objective_raw is not None:
                objective = replace(
                    objective,
                    **dict(_mapping(objective_raw, "model.objective")),
                )
        else:
            objective = training_recipe.stage_b_objective(str(selected_branch))
            if objective_raw is not None:
                repeated = replace(
                    objective,
                    **dict(_mapping(objective_raw, "model.objective")),
                )
                if repeated != objective:
                    raise ValueError(
                        "model.objective overrides immutable training_recipe fields"
                    )
    elif objective_raw is None:
        objective = GuidanceObjectiveConfig(hybrid_branch_objective=True)
    else:
        objective = GuidanceObjectiveConfig(
            **dict(_mapping(objective_raw, "model.objective"))
        )
    semantic_present = "semantic_alignment" in model
    semantic_raw = model.get("semantic_alignment")
    semantic_alignment = (
        (
            None
            if training_recipe is not None
            else standard_stage_b_semantic_alignment(str(selected_branch))
        )
        if stage == "stage-b"
        else None
    )
    if semantic_present and semantic_raw is None:
        semantic_alignment = None
    elif semantic_raw is not None:
        semantic_overrides = dict(
            _mapping(semantic_raw, "training config.model.semantic_alignment")
        )
        enabled = semantic_overrides.pop("enabled", True)
        if not isinstance(enabled, bool):
            raise TypeError("model.semantic_alignment.enabled must be a bool")
        if not enabled:
            if semantic_overrides:
                raise ValueError(
                    "disabled model.semantic_alignment cannot define loss fields"
                )
            semantic_alignment = None
        else:
            semantic_alignment = (
                SemanticAlignmentConfig(**semantic_overrides)
                if semantic_alignment is None
                else replace(semantic_alignment, **semantic_overrides)
            )
    if training_recipe is not None and stage == "stage-b" and semantic_alignment is not None:
        raise ValueError(
            "model.semantic_alignment overrides immutable training_recipe fields"
        )
    controller_raw = model.get("teacher_semantic_loss_share_controller")
    teacher_semantic_loss_share_controller = (
        (
            training_recipe.stage_b_teacher_semantic_loss_share_controller()
            if training_recipe is not None
            else standard_stage_b_teacher_semantic_loss_share_controller()
        )
        if stage == "stage-b"
        else TeacherSemanticLossShareControllerConfig()
    )
    if controller_raw is not None:
        repeated_controller = replace(
            teacher_semantic_loss_share_controller,
            **dict(
                _mapping(
                    controller_raw,
                    "training config.model.teacher_semantic_loss_share_controller",
                )
            ),
        )
        if (
            training_recipe is not None
            and repeated_controller != teacher_semantic_loss_share_controller
        ):
            raise ValueError(
                "model.teacher_semantic_loss_share_controller overrides immutable "
                "training_recipe fields"
            )
        teacher_semantic_loss_share_controller = repeated_controller
    recovery_raw = model.get("positive_recovery")
    positive_recovery = (
        (
            training_recipe.stage_b_positive_recovery(str(selected_branch))
            if training_recipe is not None
            else standard_stage_b_positive_recovery(
                str(selected_branch), budget_mode=training.budget_mode
            )
        )
        if stage == "stage-b"
        else PositiveRecoveryConfig()
    )
    if recovery_raw is not None:
        recovery_overrides = dict(
            _mapping(
                recovery_raw,
                "training config.model.positive_recovery",
            )
        )
        if training.budget_mode == "epochs" and {
            "warmup_steps",
            "ramp_steps",
        } & recovery_overrides.keys():
            raise ValueError(
                "epoch training uses positive_recovery warmup_epochs/ramp_epochs"
            )
        if training.budget_mode == "legacy_steps" and {
            "warmup_epochs",
            "ramp_epochs",
        } & recovery_overrides.keys():
            raise ValueError(
                "legacy step training uses positive_recovery warmup_steps/ramp_steps"
            )
        repeated_recovery = replace(
            positive_recovery,
            **recovery_overrides,
        )
        if training_recipe is not None and repeated_recovery != positive_recovery:
            raise ValueError(
                "model.positive_recovery overrides immutable training_recipe fields"
            )
        positive_recovery = repeated_recovery
    rollout_raw = model.get("rollout_reference_kl")
    rollout_reference_kl = (
        (
            training_recipe.stage_b_rollout_reference_kl()
            if training_recipe is not None
            else standard_stage_b_rollout_reference_kl()
        )
        if stage == "stage-b"
        else RolloutReferenceKLConfig()
    )
    if rollout_raw is not None:
        repeated_rollout = replace(
            rollout_reference_kl,
            **dict(
                _mapping(
                    rollout_raw,
                    "training config.model.rollout_reference_kl",
                )
            ),
        )
        if training_recipe is not None and repeated_rollout != rollout_reference_kl:
            raise ValueError(
                "model.rollout_reference_kl overrides immutable training_recipe fields"
            )
        rollout_reference_kl = repeated_rollout
    configured_condition_dim = (
        None
        if model.get("condition_dim") is None
        else _integer(
            model.get("condition_dim"),
            "training config.model.condition_dim",
        )
    )
    projector_raw = model.get("projector_only_stage_a")
    projector_only_stage_a = (
        None
        if projector_raw is None
        else ProjectorOnlyStageAConfig(
            **dict(_mapping(projector_raw, "training config.model.projector_only_stage_a"))
        )
    )
    output_value = paths.get("output_checkpoint")
    virtual_merge_raw = data.get("virtual_microbatch_merge")
    virtual_microbatch_merge = (
        None
        if virtual_merge_raw is None
        else PistaVirtualMicrobatchMerge(
            lane_groups=_integer(
                _mapping(
                    virtual_merge_raw,
                    "training config.data.virtual_microbatch_merge",
                ).get("lane_groups"),
                "data.virtual_microbatch_merge.lane_groups",
            ),
            lane_max_candidates=_integer(
                _mapping(
                    virtual_merge_raw,
                    "training config.data.virtual_microbatch_merge",
                ).get("lane_max_candidates"),
                "data.virtual_microbatch_merge.lane_max_candidates",
            ),
            lane_max_padded_graph_cells=(
                None
                if _mapping(
                    virtual_merge_raw,
                    "training config.data.virtual_microbatch_merge",
                ).get("lane_max_padded_graph_cells") is None
                else _integer(
                    _mapping(
                        virtual_merge_raw,
                        "training config.data.virtual_microbatch_merge",
                    ).get("lane_max_padded_graph_cells"),
                    "data.virtual_microbatch_merge.lane_max_padded_graph_cells",
                )
            ),
            merge_lanes=_integer(
                _mapping(
                    virtual_merge_raw,
                    "training config.data.virtual_microbatch_merge",
                ).get("merge_lanes"),
                "data.virtual_microbatch_merge.merge_lanes",
            ),
        )
    )
    return PistaTrainingSpec(
        stage=stage,
        corpus_root=_path(paths.get("corpus_root"), "paths.corpus_root", must_exist=True),
        graph_cache_root=_path(paths.get("graph_cache_root"), "paths.graph_cache_root", must_exist=True),
        token_cache_root=_path(paths.get("token_cache_root"), "paths.token_cache_root", must_exist=True),
        guidance_text_table=_path(
            paths.get("guidance_text_table"), "paths.guidance_text_table", must_exist=True
        ),
        token_extension_root=_optional_path(
            paths.get("token_extension_root"), "paths.token_extension_root"
        ),
        base_checkpoint=_path(paths.get("base_checkpoint"), "paths.base_checkpoint", must_exist=True),
        stage_a_checkpoint=_optional_path(paths.get("stage_a_checkpoint"), "paths.stage_a_checkpoint"),
        output_checkpoint=(
            None
            if output_value is None
            else _path(output_value, "paths.output_checkpoint", must_exist=False)
        ),
        graphdit=GraphDiTConfig(**dict(graphdit_raw)),
        guidance_text_dim=int(model.get("guidance_text_dim", 4096)),
        guidance_architecture=guidance_architecture,
        objective=objective,
        training=training,
        device=str(run.get("device", "cuda")),
        branch_weights={
            "metric": float(
                data.get(
                    "metric_branch_weight",
                    1.0 if selected_branch == "metric" else 0.0 if selected_branch else 0.5,
                )
            ),
            "structure": float(
                data.get(
                    "structure_branch_weight",
                    1.0 if selected_branch == "structure" else 0.0 if selected_branch else 0.5,
                )
            ),
        },
        sampler_mode=str(data.get("sampler_mode", "branch_cycle")),
        max_groups_per_batch=int(data.get("max_groups_per_batch", 4)),
        max_candidates_per_batch=int(data.get("max_candidates_per_batch", 120)),
        max_open_graph_shards=int(
            data.get("max_open_graph_shards", DEFAULT_MAX_OPEN_GRAPH_SHARDS)
        ),
        cpu_batch_workers=_integer(
            data.get("cpu_batch_workers"),
            "data.cpu_batch_workers",
            default=DEFAULT_CPU_BATCH_WORKERS,
        ),
        max_padded_graph_cells_per_batch=(
            None
            if data.get("max_padded_graph_cells_per_batch") is None
            else int(data["max_padded_graph_cells_per_batch"])
        ),
        trust_base_checkpoint=_bool(paths.get("trust_base_checkpoint"), "paths.trust_base_checkpoint"),
        trust_stage_a_checkpoint=_bool(
            paths.get("trust_stage_a_checkpoint"), "paths.trust_stage_a_checkpoint"
        ),
        cost_balanced_rank_batching=_bool(
            data.get("cost_balanced_rank_batching"),
            "data.cost_balanced_rank_batching",
            default=False,
        ),
        virtual_microbatch_merge=virtual_microbatch_merge,
        formal_training=_formal_training_config(
            stage=stage,
            value=run.get("formal"),
            training=training,
        ),
        semantic_route_cache_root=_optional_path(
            paths.get("semantic_route_cache_root"), "paths.semantic_route_cache_root"
        ),
        projector_only_metadata_root=_optional_path(
            paths.get("projector_only_metadata_root"),
            "paths.projector_only_metadata_root",
        ),
        semantic_alignment=semantic_alignment,
        teacher_semantic_loss_share_controller=(
            teacher_semantic_loss_share_controller
        ),
        positive_recovery=positive_recovery,
        rollout_reference_kl=rollout_reference_kl,
        stage_a_implementation=stage_a_implementation,
        projector_only_stage_a=projector_only_stage_a,
        semantic_route_embedding_dim=_integer(
            model.get("semantic_route_embedding_dim"),
            "training config.model.semantic_route_embedding_dim",
            default=(
                projector_only_stage_a.semantic_dim
                if projector_only_stage_a is not None
                else GraphDiTConfig(**dict(graphdit_raw)).x_dim
            ),
        ),
        condition_dim=configured_condition_dim,
        selected_branch=selected_branch,
        training_recipe=training_recipe,
    )


def _provider(spec: PistaTrainingSpec, split: str) -> PistaStageABatchProvider:
    needs_semantic_cache = (
        spec.stage_a_implementation == "projector_only_embedding_alignment_v1"
        or spec.semantic_alignment is not None
    )
    return PistaStageABatchProvider.from_pista_paths(
        corpus_root=spec.corpus_root,
        graph_cache_root=spec.graph_cache_root,
        token_cache_root=spec.token_cache_root,
        guidance_text_table=spec.guidance_text_table,
        token_extension_root=spec.token_extension_root,
        semantic_route_cache_root=(
            spec.semantic_route_cache_root
            if needs_semantic_cache
            else None
        ),
        projector_only_metadata_root=(
            spec.projector_only_metadata_root
            if (
                split == "train"
                and spec.stage == "stage-a"
                and spec.stage_a_implementation == "projector_only_embedding_alignment_v1"
            )
            else None
        ),
        semantic_route_embedding_dim=spec.semantic_route_embedding_dim,
        split=split,
        graphdit=spec.graphdit,
        training_seed=spec.training.seed,
        evaluation_seed=spec.training.seed + 1000,
        branch_weights=spec.branch_weights,
        sampling_mode=spec.sampler_mode,
        max_groups_per_batch=spec.max_groups_per_batch,
        max_candidates_per_batch=spec.max_candidates_per_batch,
        max_open_graph_shards=spec.max_open_graph_shards,
        cpu_batch_workers=spec.cpu_batch_workers,
        max_padded_graph_cells_per_batch=spec.max_padded_graph_cells_per_batch,
        cost_balanced_rank_batching=spec.cost_balanced_rank_batching,
        include_shuffled=(
            spec.stage == "stage-a"
            and spec.stage_a_implementation == "projector_only_embedding_alignment_v1"
        ),
        virtual_microbatch_merge=spec.virtual_microbatch_merge,
    )


def preflight_pista_training(spec: PistaTrainingSpec, *, all_splits: bool = True) -> dict[str, object]:
    """Check cache/corpus compatibility with one bounded CPU batch per split."""

    if not isinstance(spec, PistaTrainingSpec):
        raise TypeError("spec must be PistaTrainingSpec")
    splits = ("train", "validation", "test") if all_splits else ("train",)
    report: dict[str, object] = {
        "schema_version": (
            f"{PISTA_TRAINING_SCHEMA}.preflight.v2"
            if spec.training.budget_mode == "epochs"
            else f"{PISTA_TRAINING_SCHEMA}.preflight.v1"
        ),
        "stage": spec.stage,
        "training_recipe": (
            None
            if spec.training_recipe is None
            else spec.training_recipe.resolved_contract(spec.selected_branch)
        ),
        "splits": {},
    }
    for split in splits:
        provider = _provider(spec, split)
        try:
            draw = provider.next_batch(epoch=0, draw_index=0, device="cpu", evaluation=True)
            report["splits"][split] = {
                "group_count": provider.runtime.group_count(split),
                "epoch_size_groups": provider.epoch_size_groups,
                "sampled_groups": len(draw.conditions),
                "sampled_candidates": int(draw.batch.graphs.p_X.shape[0]),
                "sampled_branches": [condition.group.branch for condition in draw.conditions],
                "next_draw_index": draw.next_draw_index,
            }
            if (
                split == "train"
                and spec.formal_training is not None
                and spec.formal_training.budget_mode == "epochs"
            ):
                training_plan = resolve_formal_training_plan(
                    provider, spec.formal_training
                )
                report["training_epoch_plan"] = training_plan.as_dict()
                if spec.stage == "stage-b" and training_plan.epoch_plan is not None:
                    report["resolved_epoch_schedules"] = {
                        "positive_recovery": (
                            spec.positive_recovery.resolved_schedule_dict(
                                optimizer_updates_per_epoch=(
                                    training_plan.epoch_plan.optimizer_updates_per_epoch
                                )
                            )
                        ),
                        "guidance_residual": (
                            spec.training_recipe.stage_b_guidance_residual_schedule()
                            if spec.training_recipe is not None
                            else GuidanceResidualScheduleConfig()
                        ).resolved_contract(
                            total_optimizer_updates=(
                                training_plan.max_optimizer_updates
                            )
                        ),
                    }
        finally:
            provider.close()
    return report


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint output: {path}")
    ensure_output_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _build_pista_stage(
    spec: PistaTrainingSpec,
) -> tuple[torch.nn.Module, Mapping[str, object]]:
    """Construct the requested Pista stage without coupling the runner to it."""

    if spec.stage == "stage-a":
        if spec.stage_a_implementation == "projector_only_embedding_alignment_v1":
            if spec.projector_only_stage_a is None:  # pragma: no cover - dataclass validates this
                raise AssertionError("projector-only Stage-A configuration is absent")
            model, initialization = build_projector_only_stage_a(spec.projector_only_stage_a)
        else:
            model, initialization = build_adapter_only_stage_a_from_checkpoint(
                StageAConfig(
                    graphdit=spec.graphdit,
                    guidance_text_dim=spec.guidance_text_dim,
                    semantic_dim=spec.semantic_route_embedding_dim,
                    condition_dim=spec.condition_dim,
                    guidance_architecture=spec.guidance_architecture,
                    objective=spec.objective,
                ),
                spec.base_checkpoint,
                trust_checkpoint=spec.trust_base_checkpoint,
            )
    else:
        if spec.stage_a_checkpoint is None:  # pragma: no cover - dataclass validates this
            raise AssertionError("Stage B checkpoint is absent")
        model, initialization = build_adapter_stage_b_from_checkpoint(
            StageBConfig(
                graphdit=spec.graphdit,
                guidance_text_dim=spec.guidance_text_dim,
                semantic_dim=spec.semantic_route_embedding_dim,
                condition_dim=spec.condition_dim,
                guidance_architecture=spec.guidance_architecture,
                objective=spec.objective,
                semantic_alignment=spec.semantic_alignment,
                teacher_semantic_loss_share_controller=(
                    spec.teacher_semantic_loss_share_controller
                ),
                positive_recovery=spec.positive_recovery,
                rollout_reference_kl=spec.rollout_reference_kl,
                guidance_residual_schedule=(
                    spec.training_recipe.stage_b_guidance_residual_schedule()
                    if spec.training_recipe is not None
                    else GuidanceResidualScheduleConfig()
                ),
            ),
            base_checkpoint_path=spec.base_checkpoint,
            stage_a_checkpoint_path=spec.stage_a_checkpoint,
            trust_base_checkpoint=spec.trust_base_checkpoint,
            trust_stage_a_checkpoint=spec.trust_stage_a_checkpoint,
        )
    # Preserve the complete non-mutating initialization report in formal
    # candidate metadata.  Stage B needs the transferred-projector and fresh
    # cross-attention name lists for an auditable handoff gate; omitting those
    # facts made a correct build indistinguishable from an accidental wider or
    # cross-width transfer after the run had started.
    return model, initialization.as_dict()


def _formal_result_payload(result: FormalTrainingResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "mean_loss": result.mean_loss,
        "candidate_checkpoints": [str(path) for path in result.candidate_checkpoints],
        "recovered": result.recovered,
        "distributed": {
            "rank": result.context.rank,
            "world_size": result.context.world_size,
            "local_rank": result.context.local_rank,
            "device": str(result.context.device),
        },
    }
    if result.completed_epochs is None:
        payload.update(
            {
                "completed_steps": result.completed_steps,
                "last_step": result.last_step.as_dict(),
            }
        )
    else:
        payload.update(
            {
                "completed_epochs": epoch_value_json(result.completed_epochs),
                "completed_optimizer_updates": result.completed_optimizer_updates,
                "last_update": result.last_step.as_dict(),
                "epoch_plan": result.plan.as_dict(),
            }
        )
    return payload


def run_pista_formal_training(spec: PistaTrainingSpec) -> dict[str, object]:
    """Run Pista through the generic recovery/DDP engine, never the legacy loop."""

    if not isinstance(spec, PistaTrainingSpec):
        raise TypeError("spec must be PistaTrainingSpec")
    if spec.formal_training is None:
        raise ValueError("run.formal is required for formal Pista training")
    # Model construction itself consumes random numbers (notably the fresh
    # Stage-A projector and fresh Stage-B cross-attention blocks).  Seed it
    # after CUDA/rank initialization and before `_build_pista_stage`, so A0/A1
    # with the same registered seed really share an initialization rather than
    # only sharing the subsequent corruption generator.  The generic runner
    # seeds again before the training loop to keep its recovery stream exact.
    configure_deterministic_execution(spec.formal_training.deterministic)
    initialization_context = initialize_distributed_training(spec.device)
    initialization_seed = spec.formal_training.seed + initialization_context.rank * 1_000_003
    torch.manual_seed(initialization_seed)
    if initialization_context.device.type == "cuda":
        torch.cuda.manual_seed(initialization_seed)
    model, initialization = _build_pista_stage(spec)
    provider = _provider(spec, "train")
    try:
        result = run_formal_training(
            model=model,
            provider=provider,
            config=spec.formal_training,
            device=spec.device,
            run_metadata={
                "dataset_adapter": "pista",
                "stage": spec.stage,
                "corpus_root": str(spec.corpus_root),
                "graph_cache_root": str(spec.graph_cache_root),
                "token_cache_root": str(spec.token_cache_root),
                "token_extension_root": (
                    None if spec.token_extension_root is None else str(spec.token_extension_root)
                ),
                "projector_only_metadata_root": (
                    None
                    if spec.projector_only_metadata_root is None
                    else str(spec.projector_only_metadata_root)
                ),
                "sampler_mode": spec.sampler_mode,
                "branch_weights": dict(spec.branch_weights),
                "cpu_batch_workers": spec.cpu_batch_workers,
                "rank_batching": (
                    "cost_balanced_global_window_v1"
                    if spec.cost_balanced_rank_batching
                    else "independent_rank_lanes"
                ),
                "virtual_microbatch_merge": (
                    None
                    if spec.virtual_microbatch_merge is None
                    else {
                        "lane_groups": spec.virtual_microbatch_merge.lane_groups,
                        "lane_max_candidates": (
                            spec.virtual_microbatch_merge.lane_max_candidates
                        ),
                        "lane_max_padded_graph_cells": (
                            spec.virtual_microbatch_merge.lane_max_padded_graph_cells
                        ),
                        "merge_lanes": spec.virtual_microbatch_merge.merge_lanes,
                    }
                ),
                "activation_checkpointing": spec.graphdit.activation_checkpointing,
                "initialization_seed": initialization_seed,
                "initialization": dict(initialization),
                "training_recipe": (
                    None
                    if spec.training_recipe is None
                    else spec.training_recipe.resolved_contract(
                        spec.selected_branch
                    )
                ),
            },
        )
    finally:
        provider.close()
    return {
        "schema_version": (
            f"{PISTA_TRAINING_SCHEMA}.formal_run.v2"
            if result.completed_epochs is not None
            else f"{PISTA_TRAINING_SCHEMA}.formal_run.v1"
        ),
        "stage": spec.stage,
        "formal_output_dir": str(spec.formal_training.output_dir),
        "initialization": dict(initialization),
        "training": _formal_result_payload(result),
    }


def run_pista_training(spec: PistaTrainingSpec) -> dict[str, object]:
    """Run the explicitly bounded training loop and save one new checkpoint."""

    if not isinstance(spec, PistaTrainingSpec):
        raise TypeError("spec must be PistaTrainingSpec")
    if spec.output_checkpoint is None:
        raise ValueError("paths.output_checkpoint is required to run training")
    report_path = spec.output_checkpoint.with_suffix(spec.output_checkpoint.suffix + ".report.json")
    if spec.output_checkpoint.exists() or spec.output_checkpoint.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint output: {spec.output_checkpoint}")
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite training report: {report_path}")
    provider = _provider(spec, "train")
    model, initialization = _build_pista_stage(spec)
    try:
        result = run_guided_training(
            model=model,
            provider=provider,
            config=spec.training,
            device=spec.device,
        )
    finally:
        provider.close()
    epoch_result = result.completed_epochs is not None
    checkpoint_payload = {
        "schema_version": f"{PISTA_TRAINING_SCHEMA}.checkpoint.v{2 if epoch_result else 1}",
        "stage": spec.stage,
        "state_dict": model.state_dict(),
        "training_state": {
            "next_draw_index": result.next_draw_index,
            "mean_loss": result.mean_loss,
        },
    }
    if result.completed_epochs is None:
        checkpoint_payload["training_state"]["completed_steps"] = result.completed_steps
    else:
        checkpoint_payload["training_state"].update(
            {
                "completed_epochs": epoch_value_json(result.completed_epochs),
                "completed_optimizer_updates": result.completed_steps,
            }
        )
    _atomic_torch_save(spec.output_checkpoint, checkpoint_payload)
    payload = {
        "schema_version": f"{PISTA_TRAINING_SCHEMA}.run.v{2 if epoch_result else 1}",
        "stage": spec.stage,
        "checkpoint": str(spec.output_checkpoint),
        "initialization": initialization.as_dict(),
        "training": {
            "next_draw_index": result.next_draw_index,
            "mean_loss": result.mean_loss,
            "last_update": {
                "optimizer_update": result.last_step.step,
                "draw_index": result.last_step.draw_index,
                "next_draw_index": result.last_step.next_draw_index,
                "group_count": result.last_step.group_count,
                "candidate_count": result.last_step.candidate_count,
                "loss": result.last_step.loss,
                "metrics": dict(result.last_step.metrics),
            },
        },
    }
    if result.completed_epochs is None:
        legacy_last_step = payload["training"].pop("last_update")
        legacy_last_step["step"] = legacy_last_step.pop("optimizer_update")
        payload["training"].update(
            {
                "completed_steps": result.completed_steps,
                "last_step": legacy_last_step,
            }
        )
    else:
        payload["training"].update(
            {
                "completed_epochs": epoch_value_json(result.completed_epochs),
                "completed_optimizer_updates": result.completed_steps,
            }
        )
    write_json(report_path, payload)
    return payload


__all__ = [
    "PISTA_TRAINING_SCHEMA",
    "PistaTrainingSpec",
    "load_pista_training_spec",
    "preflight_pista_training",
    "run_pista_formal_training",
    "run_pista_training",
]
