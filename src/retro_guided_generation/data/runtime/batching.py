"""Dataset-neutral physical-batch composition policies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VirtualMicrobatchMerge:
    """Merge exact historical draw lanes into one physical model batch.

    Each lane retains the group and capacity limits used before gradient
    accumulation was consolidated.  Providers must select the lanes in their
    original order and expose their group boundaries on the materialized
    batch; model stages can then preserve lane-local RNG and loss reduction
    while sharing the expensive dense forward/backward.
    """

    lane_groups: int
    lane_max_candidates: int
    lane_max_padded_graph_cells: int | None
    merge_lanes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("lane_groups", self.lane_groups),
            ("lane_max_candidates", self.lane_max_candidates),
            ("merge_lanes", self.merge_lanes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.lane_max_padded_graph_cells is not None and (
            isinstance(self.lane_max_padded_graph_cells, bool)
            or not isinstance(self.lane_max_padded_graph_cells, int)
            or self.lane_max_padded_graph_cells < 1
        ):
            raise ValueError("lane_max_padded_graph_cells must be positive or None")

    @property
    def merged_groups(self) -> int:
        return self.lane_groups * self.merge_lanes

    @property
    def merged_max_candidates(self) -> int:
        return self.lane_max_candidates * self.merge_lanes

    @property
    def merged_max_padded_graph_cells(self) -> int | None:
        if self.lane_max_padded_graph_cells is None:
            return None
        return self.lane_max_padded_graph_cells * self.merge_lanes


__all__ = ["VirtualMicrobatchMerge"]
