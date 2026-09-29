"""Discrete flow-matching primitives shared by guided model stages."""

from .coupled_corruption import (
    CorruptedGraphBatch,
    CoupledSelectorField,
    build_coupled_predefined_corruption,
    sample_coupled_selector_field,
)

from .losses import (
    full_dfm_per_graph,
    full_reference_kl_per_graph,
)

from .rollin import (
    PositiveRecoveryConfig,
    positive_recovery_time,
    sample_predefined_student_transition,
)

from .sampling import (
    PredefinedSamplingConfig,
    sample_predefined_graph,
)

__all__ = [
    "CorruptedGraphBatch",
    "CoupledSelectorField",
    "PositiveRecoveryConfig",
    "PredefinedSamplingConfig",
    "build_coupled_predefined_corruption",
    "full_dfm_per_graph",
    "full_reference_kl_per_graph",
    "positive_recovery_time",
    "sample_coupled_selector_field",
    "sample_predefined_graph",
    "sample_predefined_student_transition",
]
