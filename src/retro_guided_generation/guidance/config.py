"""Runtime sampling configuration for the released Stage A/B recipe.

Construction and annotation configurations are outside this runtime package.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Runtime true-guidance sampling policy for downstream training."""

    seed: int = 42
    evaluation_seed: int = 1042
    branch_weights: dict[str, float] = field(
        default_factory=lambda: {"metric": 0.5, "structure": 0.5}
    )
    axis_sampling: str = "uniform"
    semantic_family_sampling: str = "uniform"
    polarity_sampling: str = "uniform"
    training_wording: str = "all_train"
    evaluation_wording: str = "canonical"
