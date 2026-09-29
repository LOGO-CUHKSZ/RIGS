"""Projector-only instruction--route grounding for the new Stage-A boundary.

This module intentionally contains neither a GraphDiT instance nor a route--text
cross-attention block.  The language and route representations arrive as frozen
cached tensors in :class:`GuidedStageABatch`; only one shared text projector is
trainable.  Its output width is the target generator's conditioning width ``k``,
so the formal checkpoint can be handed to Stage B without an intermediate lift,
zero padding, optimizer state, scalar head, route encoder, or cross-attention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..graphdit.token_guidance import TokenTextProjector
from ..guidance.architecture import resolve_guidance_architecture
from ..objectives import SemanticAlignmentObjective, SemanticAlignmentOutput
from ..objectives.semantic import (
    _virtual_lane_layout,
    _virtual_lane_mean,
    _virtual_lane_selected_count,
)
from .stage_a import GuidedStageABatch, _expected_group_index


@dataclass(frozen=True, slots=True)
class ProjectorOnlyStageAConfig:
    """Immutable projector-only Stage-A optimization contract.

    ``semantic_dim`` is the historical configuration name for the shared
    projector's output width.  In the current contract it is not a fixed
    teacher width: it must equal the target generator's conditioning width
    ``k`` and the native frozen route-embedding width used for alignment.

    ``logit_scale`` is deliberately a plain float rather than a parameter: it
    is the fixed similarity temperature described by the standard protocol.
    The route embedding is supplied by the frozen cache and cannot become a
    trainable module through this configuration.
    """

    semantic_dim: int
    guidance_architecture: str
    guidance_text_dim: int = 4096
    selected_branch: str = "structure"
    listwise_weight: float = 1.0
    pair_weight: float = 0.5
    pair_margin: float = 0.1
    identity_weight: float = 0.05
    identity_margin: float = 0.1
    endpoint_probability_floor: float = 1.0e-8
    logit_scale: float = 10.0

    def __post_init__(self) -> None:
        for name in ("guidance_text_dim", "semantic_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.selected_branch not in {"structure", "metric"}:
            raise ValueError("projector-only Stage A selected_branch must be structure or metric")
        if not isinstance(self.guidance_architecture, str) or not self.guidance_architecture.strip():
            raise ValueError(
                "projector-only Stage A requires an explicitly confirmed registered "
                "guidance_architecture ID"
            )
        for name, value, positive in (
            ("listwise_weight", self.listwise_weight, True),
            ("pair_weight", self.pair_weight, False),
            ("pair_margin", self.pair_margin, False),
            ("identity_weight", self.identity_weight, False),
            ("identity_margin", self.identity_margin, False),
            ("endpoint_probability_floor", self.endpoint_probability_floor, True),
            ("logit_scale", self.logit_scale, True),
        ):
            number = float(value)
            if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not math.isclose(float(self.listwise_weight), 1.0, rel_tol=0.0, abs_tol=1.0e-8):
            raise ValueError("projector-only Stage A fixes listwise_weight=1.0")
        # Resolve now so an unregistered Stage-B topology cannot silently be
        # recorded in an otherwise valid Stage-A checkpoint.
        architecture = resolve_guidance_architecture(self.guidance_architecture)
        if not architecture.projector_groups:
            raise ValueError(
                "projector-only Stage A requires a projected-token Stage-B "
                "architecture with transferable projector groups"
            )

    @property
    def conditioning_width_k(self) -> int:
        """Return the generator-specific Stage-A projector output width."""

        return self.semantic_dim


@dataclass(frozen=True, slots=True)
class ProjectorOnlyStageAInitializationReport:
    """Checkpoint handoff facts with no mutable route/generation state."""

    trainable_parameter_names: tuple[str, ...]
    architecture_contract: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "implementation_contract": "projector_only_embedding_alignment_v1",
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "architecture_contract": dict(self.architecture_contract),
        }


@dataclass(frozen=True, slots=True)
class ProjectorOnlyStageAForwardOutput:
    """Embedding-level Stage-A loss and optional identity diagnostics."""

    loss: Tensor
    semantic: SemanticAlignmentOutput
    identity_loss: Tensor
    identity_accuracy: Tensor
    identity_margin: Tensor
    identity_selected_group_count: Tensor
    identity_correct_by_group: Tensor
    identity_margin_by_group: Tensor
    identity_selected_group_mask: Tensor

    @property
    def metrics(self) -> dict[str, Tensor]:
        result = self.semantic.metrics()
        result.update(
            {
                "stage_a_projector_loss": self.loss,
                "teacher/listwise_kl": self.semantic.listwise_kl,
                "teacher/pairwise_loss": self.semantic.pairwise_loss,
                "teacher/selected_group_count": self.semantic.selected_group_count,
                "teacher/identity/pairwise_loss": self.identity_loss,
                "teacher/identity/accuracy": self.identity_accuracy,
                "teacher/identity/score_margin": self.identity_margin,
                "teacher/identity/selected_group_count": self.identity_selected_group_count,
            }
        )
        return result


def _validate_group_tokens(
    tokens: Tensor | None,
    mask: Tensor | None,
    *,
    groups: int,
    width: int,
    name: str,
) -> tuple[Tensor, Tensor]:
    if (
        not isinstance(tokens, Tensor)
        or tokens.ndim != 3
        or tokens.shape[0] != groups
        or tokens.shape[-1] != width
        or not tokens.is_floating_point()
    ):
        raise ValueError(f"{name} tokens must be floating [groups,length,{width}]")
    if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
        raise ValueError(f"{name} token mask must be bool[groups,length]")
    if tokens.device != mask.device:
        raise ValueError(f"{name} tokens and mask must share one device")
    valid = mask.unsqueeze(-1)
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"every {name} guidance group needs one valid token")
    if not bool(torch.isfinite(tokens.masked_select(valid)).all()):
        raise ValueError(f"{name} guidance tokens contain NaN or Inf")
    return tokens, mask


class ProjectorOnlyStageA(nn.Module):
    """Train the one embedding-model-to-generator projector of width ``k``."""

    def __init__(self, config: ProjectorOnlyStageAConfig) -> None:
        super().__init__()
        if not isinstance(config, ProjectorOnlyStageAConfig):
            raise TypeError("config must be ProjectorOnlyStageAConfig")
        self.config = config
        self.projector = TokenTextProjector(
            config.guidance_text_dim, config.conditioning_width_k
        )
        self.semantic_objective = SemanticAlignmentObjective(
            selected_branch=config.selected_branch,
            pair_weight=config.pair_weight,
            pair_margin=config.pair_margin,
            endpoint_probability_floor=config.endpoint_probability_floor,
        )
        self.assert_projector_only_trainable()

    def assert_projector_only_trainable(self) -> None:
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("projector.")
        ]
        if unexpected:
            raise RuntimeError(
                "projector-only Stage A exposes non-projector parameters: "
                f"{unexpected}"
            )
        frozen = [
            name
            for name, parameter in self.projector.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen:
            raise RuntimeError(f"projector-only Stage A freezes projector parameters: {frozen}")

    def trainable_parameter_names(self) -> tuple[str, ...]:
        self.assert_projector_only_trainable()
        return tuple(name for name, parameter in self.named_parameters() if parameter.requires_grad)

    def architecture_contract(self) -> dict[str, object]:
        spec = resolve_guidance_architecture(self.config.guidance_architecture)
        contract = spec.contract(
            text_dim=self.config.guidance_text_dim,
            semantic_dim=self.config.semantic_dim,
            condition_dim=self.config.conditioning_width_k,
            hidden_dim=self.config.conditioning_width_k,
        )
        contract.update(
            {
                "implementation_contract": "projector_only_embedding_alignment_v1",
                "stage_a_projector_groups": [["shared"]],
                "stage_a_trainable_module": "projector",
                "stage_a_trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
                "semantic_dim": self.config.semantic_dim,
                "condition_dim": self.config.conditioning_width_k,
                "conditioning_width_k": self.config.conditioning_width_k,
                "stage_a_output_dim": self.config.conditioning_width_k,
                "condition_dim_source": "generator_k_validated_by_training_spec",
                "condition_adapter": {
                    "input_dim": self.config.guidance_text_dim,
                    "output_dim": self.config.conditioning_width_k,
                    "identity": False,
                    "implemented_by": "shared_text_projector",
                },
                "cross_attention": {
                    "query_dim": self.config.conditioning_width_k,
                    "key_dim": self.config.conditioning_width_k,
                    "value_dim": self.config.conditioning_width_k,
                },
                "fixed_logit_scale": float(self.config.logit_scale),
            }
        )
        return contract

    def initialization_report(self) -> ProjectorOnlyStageAInitializationReport:
        return ProjectorOnlyStageAInitializationReport(
            trainable_parameter_names=self.trainable_parameter_names(),
            architecture_contract=self.architecture_contract(),
        )

    def _embedding(self, tokens: Tensor, mask: Tensor) -> Tensor:
        projected = self.projector(tokens).float()
        valid = mask.unsqueeze(-1)
        pooled = projected.mul(valid).sum(dim=1)
        pooled = pooled / valid.sum(dim=1).clamp_min(1)
        return F.normalize(pooled, dim=-1, eps=1.0e-8)

    def _embedding_by_virtual_lane(
        self,
        tokens: Tensor,
        mask: Tensor,
        group_ptr: Tensor | None,
    ) -> Tensor:
        """Project lanes in the same floating-point order as old microbatches."""

        if group_ptr is None:
            return self._embedding(tokens, mask)
        layout = _virtual_lane_layout(
            group_ptr,
            groups=int(tokens.shape[0]),
            device=tokens.device,
        )
        if layout is None:  # pragma: no cover - group_ptr is non-null here
            raise AssertionError("virtual lane layout is absent")
        pointer, _ = layout
        return torch.cat(
            [
                self._embedding(tokens[start:end], mask[start:end])
                for start, end in zip(pointer[:-1], pointer[1:], strict=True)
            ],
            dim=0,
        )

    def _layout_and_selection(
        self, batch: GuidedStageABatch, *, candidates: int, device: torch.device
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        expected_index, groups = _expected_group_index(
            batch.supervision.group_ptr, candidates, device
        )
        if (
            not isinstance(batch.group_index, Tensor)
            or batch.group_index.dtype not in {torch.int32, torch.int64}
            or batch.group_index.shape != (candidates,)
            or batch.group_index.device != device
            or not bool(torch.equal(batch.group_index.to(dtype=torch.long), expected_index))
        ):
            raise ValueError("projector-only Stage A group_index disagrees with group_ptr")
        labels = batch.supervision.structural_label
        if not isinstance(labels, Tensor) or labels.shape != (candidates,):
            raise ValueError("projector-only Stage A needs binary candidate structural labels")
        labels = labels.to(device=device, dtype=torch.long)
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("projector-only Stage A structural labels must be binary")
        eligible = batch.supervision.listwise_eligible
        if not isinstance(eligible, Tensor) or eligible.dtype != torch.bool or eligible.shape != (groups,):
            raise ValueError("projector-only Stage A listwise eligibility has an invalid shape")
        branches = batch.supervision.guidance_branch
        if not isinstance(branches, (tuple, list)) or len(branches) != groups:
            raise ValueError("projector-only Stage A requires branch-aware supervision")
        if set(branches) - {"structure", "metric"}:
            raise ValueError("projector-only Stage A has unsupported guidance branches")
        sizes = batch.supervision.group_ptr[1:].to(device=device, dtype=torch.long) - batch.supervision.group_ptr[:-1].to(device=device, dtype=torch.long)
        positive = torch.segment_reduce((labels == 1).float(), "sum", lengths=sizes)
        negative = torch.segment_reduce((labels == 0).float(), "sum", lengths=sizes)
        if self.config.selected_branch == "all":
            branch_selected = torch.ones(groups, dtype=torch.bool, device=device)
        else:
            branch_selected = torch.tensor(
                [branch == self.config.selected_branch for branch in branches],
                dtype=torch.bool,
                device=device,
            )
        selected = (
            branch_selected
            & eligible.to(device=device)
            & (positive > 0.0)
            & (negative > 0.0)
        )
        return expected_index, sizes, labels, selected

    def forward(
        self,
        batch: GuidedStageABatch,
        *,
        generator: torch.Generator | None = None,
    ) -> ProjectorOnlyStageAForwardOutput:
        """Evaluate the registered embedding-level Stage-A terms.

        ``generator`` is accepted only to share the formal runner interface;
        this loss is deterministic once the canonical true condition has been
        materialized by the runtime provider.  A zero identity weight is a hard
        disable: shuffled controls are neither required nor evaluated.
        """

        del generator
        if not isinstance(batch, GuidedStageABatch):
            raise TypeError("batch must be GuidedStageABatch")
        route = batch.route_semantic_candidate
        if route is None:
            raise RuntimeError("projector-only Stage A requires a frozen route semantic cache")
        if route.requires_grad or route.ndim != 2 or route.shape[1] != self.config.semantic_dim:
            raise ValueError(
                "route semantic embeddings must be detached [candidate,semantic_dim] tensors"
            )
        if not torch.isfinite(route).all():
            raise ValueError("route semantic embeddings contain NaN or Inf")
        candidates = int(route.shape[0])
        group_index, sizes, labels, selected = self._layout_and_selection(
            batch, candidates=candidates, device=route.device
        )
        groups = int(sizes.numel())
        tokens, mask = _validate_group_tokens(
            batch.guidance_tokens_group,
            batch.guidance_token_mask_group,
            groups=groups,
            width=self.config.guidance_text_dim,
            name="true",
        )
        if tokens.device != route.device:
            raise ValueError("guidance tokens and route embeddings must share one device")
        true_embedding = self._embedding_by_virtual_lane(
            tokens,
            mask,
            batch.virtual_microbatch_group_ptr,
        )
        semantic = self.semantic_objective(
            group_embeddings={"shared": true_embedding},
            route_embedding=route.detach(),
            group_ptr=batch.supervision.group_ptr,
            group_index=group_index,
            teacher_probability=batch.supervision.teacher_probability,
            endpoint_probability=batch.supervision.endpoint_probability,
            structural_label=labels,
            guidance_branch=batch.supervision.guidance_branch or (),
            listwise_eligible=batch.supervision.listwise_eligible,
            logit_scale=self.config.logit_scale,
            virtual_microbatch_group_ptr=batch.virtual_microbatch_group_ptr,
        )

        zero = true_embedding.sum() * 0.0
        virtual_lane_layout = _virtual_lane_layout(
            batch.virtual_microbatch_group_ptr,
            groups=groups,
            device=route.device,
        )
        if float(self.config.identity_weight) == 0.0:
            identity_selected = torch.zeros_like(selected)
            return ProjectorOnlyStageAForwardOutput(
                loss=semantic.loss,
                semantic=semantic,
                identity_loss=zero,
                identity_accuracy=zero,
                identity_margin=zero,
                identity_selected_group_count=_virtual_lane_selected_count(
                    identity_selected,
                    layout=virtual_lane_layout,
                ),
                identity_correct_by_group=torch.zeros(
                    groups,
                    dtype=torch.bool,
                    device=route.device,
                ),
                identity_margin_by_group=torch.zeros(
                    groups,
                    dtype=true_embedding.dtype,
                    device=route.device,
                ),
                identity_selected_group_mask=identity_selected,
            )

        shuffled_tokens, shuffled_mask = _validate_group_tokens(
            batch.shuffled_guidance_tokens_group,
            batch.shuffled_guidance_token_mask_group,
            groups=groups,
            width=self.config.guidance_text_dim,
            name="shuffled",
        )
        if shuffled_tokens.device != route.device:
            raise ValueError("shuffled tokens and route embeddings must share one device")
        # The donor is a counterfactual condition, not a negative label.  Its
        # embedding is detached so identity gradients can only improve the
        # true instruction's projector coordinate rather than making an
        # unrelated donor deliberately worse.
        with torch.no_grad():
            shuffled_embedding = self._embedding_by_virtual_lane(
                shuffled_tokens,
                shuffled_mask,
                batch.virtual_microbatch_group_ptr,
            )
        frozen_route = F.normalize(route.detach().float(), dim=-1, eps=1.0e-8)
        true_score = float(self.config.logit_scale) * (
            true_embedding.index_select(0, group_index) * frozen_route
        ).sum(dim=-1)
        shuffled_score = float(self.config.logit_scale) * (
            shuffled_embedding.index_select(0, group_index) * frozen_route
        ).sum(dim=-1)
        positive = labels == 1
        positive_count = torch.segment_reduce(positive.float(), "sum", lengths=sizes)
        true_positive = torch.segment_reduce(
            torch.where(positive, true_score, torch.zeros_like(true_score)),
            "sum",
            lengths=sizes,
        ) / positive_count.clamp_min(1.0)
        shuffled_positive = torch.segment_reduce(
            torch.where(positive, shuffled_score, torch.zeros_like(shuffled_score)),
            "sum",
            lengths=sizes,
        ) / positive_count.clamp_min(1.0)
        identity_delta = true_positive - shuffled_positive.detach()
        per_group_identity = F.softplus(
            identity_delta.new_tensor(float(self.config.identity_margin)) - identity_delta
        )
        identity_loss = _virtual_lane_mean(
            per_group_identity,
            selected,
            zero=zero,
            layout=virtual_lane_layout,
        )
        identity_accuracy = _virtual_lane_mean(
            (identity_delta > 0.0).float(),
            selected,
            zero=zero,
            layout=virtual_lane_layout,
        )
        identity_margin = _virtual_lane_mean(
            identity_delta,
            selected,
            zero=zero,
            layout=virtual_lane_layout,
        )
        loss = semantic.loss + float(self.config.identity_weight) * identity_loss
        return ProjectorOnlyStageAForwardOutput(
            loss=loss,
            semantic=semantic,
            identity_loss=identity_loss,
            identity_accuracy=identity_accuracy,
            identity_margin=identity_margin,
            identity_selected_group_count=_virtual_lane_selected_count(
                selected, layout=virtual_lane_layout
            ),
            identity_correct_by_group=(identity_delta > 0.0).detach(),
            identity_margin_by_group=identity_delta.detach(),
            identity_selected_group_mask=selected.detach(),
        )


def build_projector_only_stage_a(
    config: ProjectorOnlyStageAConfig,
) -> tuple[ProjectorOnlyStageA, ProjectorOnlyStageAInitializationReport]:
    """Construct the new Stage-A model without a hidden GraphDiT teacher."""

    stage = ProjectorOnlyStageA(config)
    return stage, stage.initialization_report()


__all__ = [
    "ProjectorOnlyStageA",
    "ProjectorOnlyStageAConfig",
    "ProjectorOnlyStageAForwardOutput",
    "ProjectorOnlyStageAInitializationReport",
    "build_projector_only_stage_a",
]
