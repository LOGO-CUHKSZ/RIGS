"""Adapter-only Stage A assembly for frozen-reference GraphDiT training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from ...checkpoints import load_legacy_state_dict
from ..dfm import (
    CorruptedGraphBatch,
    CoupledSelectorField,
    build_coupled_predefined_corruption,
)
from ..graphdit import (
    GraphDiT,
    GraphDiTConfig,
    GraphDiTStateLoadReport,
    TokenCrossAttentionGraphDiT,
    load_graphdit_state_dict,
)
from ..guidance.architecture import GuidanceArchitectureSelector
from ..objectives import GuidanceObjective, GuidanceObjectiveConfig, GuidedObjectiveBatch, GuidedObjectiveOutput


@dataclass(frozen=True, slots=True)
class GraphConditionBatch:
    """Dense product/route graph tensors consumed by a categorical GraphDiT."""

    p_X: Tensor
    p_E: Tensor
    r_X: Tensor
    r_E: Tensor
    node_mask: Tensor
    p_mask: Tensor
    cond: Tensor


@dataclass(frozen=True, slots=True)
class GuidedStageABatch:
    """One group-contiguous Stage A batch with already encoded text tokens."""

    graphs: GraphConditionBatch
    supervision: GuidedObjectiveBatch
    group_index: Tensor
    guidance_tokens_group: Tensor
    guidance_token_mask_group: Tensor
    route_semantic_candidate: Tensor | None = None
    shuffled_guidance_tokens_group: Tensor | None = None
    shuffled_guidance_token_mask_group: Tensor | None = None
    # Optional CSR pointer over groups for a deterministically merged physical
    # batch.  Each interval is one historical microbatch lane.  Projector-only
    # Stage A uses it to preserve the original equal-lane loss denominator
    # after several lanes are materialized and forwarded together.
    virtual_microbatch_group_ptr: Tensor | None = None


@dataclass(frozen=True, slots=True)
class StageAConfig:
    """Public construction contract for adapter-only Stage A."""

    graphdit: GraphDiTConfig
    guidance_text_dim: int = 4096
    semantic_dim: int | None = None
    condition_dim: int | None = None
    guidance_architecture: GuidanceArchitectureSelector = None
    objective: GuidanceObjectiveConfig = GuidanceObjectiveConfig()

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
        if not isinstance(self.objective, GuidanceObjectiveConfig):
            raise TypeError("objective must be GuidanceObjectiveConfig")
        _validate_adapter_only_objective_config(self.objective)

    @property
    def resolved_semantic_dim(self) -> int:
        """Return the native Stage-A semantic-coordinate width."""

        return self.graphdit.x_dim if self.semantic_dim is None else self.semantic_dim

    @property
    def resolved_condition_dim(self) -> int:
        """Resolve generator conditioning width ``k`` from the backbone by default."""

        return self.graphdit.x_dim if self.condition_dim is None else self.condition_dim


@dataclass(frozen=True, slots=True)
class StageAInitializationReport:
    """Non-mutating initialization metadata without content-derived IDs."""

    reference_load: GraphDiTStateLoadReport
    student_load: GraphDiTStateLoadReport
    trainable_parameter_names: tuple[str, ...]
    architecture_contract: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "reference_load": self.reference_load.as_dict(),
            "student_load": self.student_load.as_dict(),
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "architecture_contract": dict(self.architecture_contract),
        }


@dataclass(frozen=True, slots=True)
class StageAForwardOutput:
    """One adapter-only Stage A forward on a shared corruption realization."""

    loss: Tensor
    objective: GuidedObjectiveOutput
    corruption: CorruptedGraphBatch
    student_logits: Mapping[str, Tensor]
    reference_logits: Mapping[str, Tensor]


def _validate_adapter_only_objective_config(
    objective: GuidanceObjectiveConfig,
) -> None:
    """Reject objective extensions that are outside the adapter-only Stage A.

    The generic tensor objective intentionally retains a few V4 research
    variants so a future stage can make an explicit decision about them.  The
    public Stage A assembly must not silently enable any of those variants:
    this stage is the frozen-reference, contrastive adapter fit only.
    """

    preservation_mode = str(objective.preservation_mode).strip().lower()
    if preservation_mode != "legacy":
        raise ValueError(
            "adapter-only Stage A requires objective.preservation_mode='legacy'; "
            "non-legacy preservation is outside the current Stage A scope"
        )
    candidate_mode = str(objective.candidate_supervision_mode).strip().lower()
    if candidate_mode != "contrastive":
        raise ValueError(
            "adapter-only Stage A requires "
            "objective.candidate_supervision_mode='contrastive'; "
            "alternative candidate-supervision modes are outside the current "
            "Stage A scope"
        )


def _validate_graphs(graphs: GraphConditionBatch) -> int:
    if not isinstance(graphs, GraphConditionBatch):
        raise TypeError("graphs must be GraphConditionBatch")
    p_x, p_e, r_x, r_e = graphs.p_X, graphs.p_E, graphs.r_X, graphs.r_E
    if p_x.ndim != 3 or p_x.shape != r_x.shape:
        raise ValueError("p_X and r_X must share shape [B,N,Kx]")
    if p_e.ndim != 4 or p_e.shape != r_e.shape:
        raise ValueError("p_E and r_E must share shape [B,N,N,Ke]")
    batch_size, nodes = p_x.shape[:2]
    if p_e.shape[:3] != (batch_size, nodes, nodes):
        raise ValueError("node and edge tensors disagree on batch or node dimensions")
    if graphs.node_mask.shape != (batch_size, nodes) or graphs.p_mask.shape != (batch_size, nodes):
        raise ValueError("node_mask and p_mask must have shape [B,N]")
    if graphs.cond.ndim != 2 or graphs.cond.shape[0] != batch_size:
        raise ValueError("cond must have shape [B,C]")
    values = (p_x, p_e, r_x, r_e, graphs.node_mask, graphs.p_mask, graphs.cond)
    if any(value.device != p_x.device for value in values):
        raise ValueError("all graph-condition tensors must share one device")
    return batch_size


def _expected_group_index(group_ptr: Tensor, candidates: int, device: torch.device) -> tuple[Tensor, int]:
    if not isinstance(group_ptr, Tensor) or group_ptr.ndim != 1 or group_ptr.numel() < 2:
        raise ValueError("group_ptr must be a one-dimensional CSR pointer")
    if group_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_ptr must use an integer dtype")
    ptr = group_ptr.to(device=device, dtype=torch.long)
    sizes = ptr[1:] - ptr[:-1]
    valid = (ptr[0] == 0) & (ptr[-1] == candidates) & torch.all(sizes > 0)
    if ptr.device.type == "cuda":
        torch._assert_async(valid, "group_ptr must contain positive contiguous segments")
    elif not bool(valid):
        raise ValueError("group_ptr must contain positive contiguous segments covering candidates")
    groups = int(ptr.numel() - 1)
    index = torch.repeat_interleave(
        torch.arange(groups, device=device, dtype=torch.long), sizes, output_size=candidates
    )
    return index, groups


class AdapterOnlyStageA(nn.Module):
    """Frozen base GraphDiT reference plus trainable token-guidance adapter.

    There is deliberately no reward network and no other trainable model in
    this stage.  The reference and every student-backbone parameter stay
    frozen; only ``student.guidance_adapter`` participates in optimization.
    """

    def __init__(
        self,
        *,
        reference: GraphDiT,
        student: TokenCrossAttentionGraphDiT,
        objective: GuidanceObjective,
    ) -> None:
        super().__init__()
        if not isinstance(reference, GraphDiT) or not isinstance(student, TokenCrossAttentionGraphDiT):
            raise TypeError("reference and student must be GraphDiT model instances")
        if not isinstance(objective, GuidanceObjective):
            raise TypeError("objective must be GuidanceObjective")
        _validate_adapter_only_objective_config(objective.config)
        if reference.config != student.config:
            raise ValueError("reference and student must have identical GraphDiT configs")
        self.reference = reference
        self.student = student
        self.objective = objective
        self._freeze_adapter_only()

    @property
    def graphdit_config(self) -> GraphDiTConfig:
        return self.reference.config

    def _freeze_adapter_only(self) -> None:
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)
        for parameter in self.student.parameters():
            parameter.requires_grad_(False)
        for parameter in self.student.guidance_adapter.parameters():
            parameter.requires_grad_(True)
        self.assert_adapter_only_trainable()

    def assert_adapter_only_trainable(self) -> None:
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("student.guidance_adapter.")
        ]
        if unexpected:
            raise RuntimeError(f"Stage A unexpectedly exposes non-adapter parameters: {unexpected}")
        frozen_adapter = [
            name
            for name, parameter in self.student.guidance_adapter.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen_adapter:
            raise RuntimeError(f"Stage A unexpectedly freezes adapter parameters: {frozen_adapter}")

    def trainable_parameter_names(self) -> tuple[str, ...]:
        self.assert_adapter_only_trainable()
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def train(self, mode: bool = True) -> "AdapterOnlyStageA":
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
        if batch.guidance_token_mask_group.dtype != torch.bool or batch.guidance_token_mask_group.shape != batch.guidance_tokens_group.shape[:2]:
            raise ValueError("guidance_token_mask_group must be bool [G,L]")
        if batch.guidance_tokens_group.device != batch.graphs.p_X.device or batch.guidance_token_mask_group.device != batch.graphs.p_X.device:
            raise ValueError("guidance token tensors must share the graph device")
        if batch.graphs.cond.shape[1] != self.graphdit_config.condition_dim:
            raise ValueError("batch cond width differs from GraphDiT config")
        return candidates, groups

    def forward(
        self,
        batch: GuidedStageABatch,
        *,
        generator: torch.Generator,
        t_group: Tensor | None = None,
        selector_field: CoupledSelectorField | None = None,
    ) -> StageAForwardOutput:
        """Run frozen-reference and trainable-adapter forwards on one corruption."""

        _candidates, groups = self._validate_batch(batch)
        corruption = build_coupled_predefined_corruption(
            batch.graphs,
            batch.group_index,
            groups,
            generator,
            t_group=t_group,
            selector_field=selector_field,
        )
        with torch.no_grad():
            reference_logits = self.reference(
                t=corruption.t_graph,
                cond=batch.graphs.cond,
                **corruption.model_inputs(),
            )
        student_logits = self.student(
            t=corruption.t_graph,
            cond=batch.graphs.cond,
            guidance_tokens_group=batch.guidance_tokens_group,
            guidance_token_mask_group=batch.guidance_token_mask_group,
            guidance_group_index=batch.group_index,
            **corruption.model_inputs(),
        )
        objective = self.objective(student_logits, reference_logits, corruption, batch.supervision)
        return StageAForwardOutput(
            loss=objective.loss,
            objective=objective,
            corruption=corruption,
            student_logits=student_logits,
            reference_logits=reference_logits,
        )


def build_adapter_only_stage_a_from_checkpoint(
    config: StageAConfig,
    checkpoint_path: str | Path,
    *,
    trust_checkpoint: bool = False,
    map_location: str = "cpu",
) -> tuple[AdapterOnlyStageA, StageAInitializationReport]:
    """Initialize frozen reference and adapter student from one trusted base checkpoint."""

    if not isinstance(config, StageAConfig):
        raise TypeError("config must be StageAConfig")
    state_dict = load_legacy_state_dict(
        checkpoint_path,
        map_location=map_location,
        trust_checkpoint=trust_checkpoint,
    )
    reference = GraphDiT(config.graphdit)
    student = TokenCrossAttentionGraphDiT(
        config.graphdit,
        text_dim=config.guidance_text_dim,
        semantic_dim=config.resolved_semantic_dim,
        condition_dim=config.condition_dim,
        architecture=config.guidance_architecture,
    )
    reference_report = load_graphdit_state_dict(reference, state_dict, strict=True)
    student_report = load_graphdit_state_dict(student, state_dict, strict=False)
    unexpected = student_report.unexpected_keys
    non_adapter_missing = tuple(
        name for name in student_report.missing_keys if not name.startswith("guidance_adapter.")
    )
    if unexpected or non_adapter_missing:
        raise RuntimeError(
            "base checkpoint did not initialize the complete Stage A backbone: "
            f"missing={non_adapter_missing}, unexpected={unexpected}"
        )
    stage = AdapterOnlyStageA(
        reference=reference,
        student=student,
        objective=GuidanceObjective(config.objective),
    )
    report = StageAInitializationReport(
        reference_load=reference_report,
        student_load=student_report,
        trainable_parameter_names=stage.trainable_parameter_names(),
        architecture_contract=student.architecture_contract(),
    )
    return stage, report


__all__ = [
    "AdapterOnlyStageA",
    "GraphConditionBatch",
    "GuidedStageABatch",
    "StageAConfig",
    "StageAForwardOutput",
    "StageAInitializationReport",
    "build_adapter_only_stage_a_from_checkpoint",
]
