"""Composable training-stage assemblies."""

from .stage_a import (
    GraphConditionBatch,
    GuidedStageABatch,
    StageAConfig,
    build_adapter_only_stage_a_from_checkpoint,
)

from .projector_stage_a import (
    ProjectorOnlyStageA,
    ProjectorOnlyStageAConfig,
    build_projector_only_stage_a,
)

from .stage_b import (
    GuidanceResidualScheduleConfig,
    RolloutReferenceKLConfig,
    SemanticAlignmentConfig,
    TeacherSemanticLossShareControllerConfig,
    StageBConfig,
    build_adapter_stage_b_from_checkpoint,
    standard_stage_b_objective,
    standard_stage_b_positive_recovery,
    standard_stage_b_rollout_reference_kl,
    standard_stage_b_semantic_alignment,
    standard_stage_b_teacher_semantic_loss_share_controller,
)

__all__ = [
    "GraphConditionBatch",
    "GuidanceResidualScheduleConfig",
    "GuidedStageABatch",
    "ProjectorOnlyStageA",
    "ProjectorOnlyStageAConfig",
    "RolloutReferenceKLConfig",
    "SemanticAlignmentConfig",
    "StageAConfig",
    "StageBConfig",
    "TeacherSemanticLossShareControllerConfig",
    "build_adapter_only_stage_a_from_checkpoint",
    "build_adapter_stage_b_from_checkpoint",
    "build_projector_only_stage_a",
    "standard_stage_b_objective",
    "standard_stage_b_positive_recovery",
    "standard_stage_b_rollout_reference_kl",
    "standard_stage_b_semantic_alignment",
    "standard_stage_b_teacher_semantic_loss_share_controller",
]
