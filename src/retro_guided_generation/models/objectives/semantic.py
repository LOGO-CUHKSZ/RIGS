"""Frozen-route semantic alignment for grouped text-guidance projectors.

The route embeddings come from a separately prepared, read-only Qwen semantic
teacher cache.  This objective trains only the guidance text projectors: it
does not turn negative candidates into generative targets and it does not
recompute, rewrite, or identify the teacher cache with content hashes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _assert(condition: Tensor, message: str) -> None:
    if condition.numel() != 1:
        raise ValueError("semantic alignment assertion must be scalar")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class SemanticAlignmentOutput:
    """Scalar and groupwise diagnostics from one semantic teacher alignment."""

    loss: Tensor
    listwise_kl: Tensor
    pairwise_loss: Tensor
    pair_accuracy: Tensor
    score_margin: Tensor
    top1_accuracy: Tensor
    selected_group_count: Tensor
    projector_losses: Mapping[str, Tensor]
    # These detached groupwise values are evaluation evidence, not additional
    # loss terms.  Keeping them in the typed output lets frozen validation
    # aggregate and bootstrap over actual semantic groups instead of treating
    # variable-size batches as independent observations.
    group_listwise_kl: Tensor
    group_pairwise_loss: Tensor
    group_pair_accuracy: Tensor
    group_score_margin: Tensor
    group_top1_accuracy: Tensor
    selected_group_mask: Tensor

    def metrics(self) -> dict[str, Tensor]:
        return {
            "semantic_loss": self.loss,
            "semantic_listwise_kl": self.listwise_kl,
            "semantic_pairwise_loss": self.pairwise_loss,
            "semantic_pair_accuracy": self.pair_accuracy,
            "semantic_score_margin": self.score_margin,
            "semantic_top1_accuracy": self.top1_accuracy,
            "semantic_selected_group_count": self.selected_group_count,
        }


@dataclass(frozen=True, slots=True)
class _Layout:
    sizes: Tensor
    index: Tensor
    groups: int
    candidates: int


def _layout(group_ptr: Tensor, candidates: int, device: torch.device) -> _Layout:
    if (
        not isinstance(group_ptr, Tensor)
        or group_ptr.ndim != 1
        or group_ptr.numel() < 2
        or group_ptr.dtype not in {torch.int32, torch.int64}
    ):
        raise TypeError("group_ptr must be a non-empty integer CSR pointer")
    pointer = group_ptr.to(device=device, dtype=torch.long)
    sizes = pointer[1:] - pointer[:-1]
    _assert(
        (pointer[0] == 0) & (pointer[-1] == candidates) & (sizes > 0).all(),
        "group_ptr must cover all candidates with non-empty contiguous groups",
    )
    groups = int(sizes.numel())
    return _Layout(
        sizes=sizes,
        index=torch.repeat_interleave(
            torch.arange(groups, device=device, dtype=torch.long),
            sizes,
            output_size=candidates,
        ),
        groups=groups,
        candidates=candidates,
    )


def _group_log_softmax(logits: Tensor, layout: _Layout) -> Tensor:
    maxima = torch.segment_reduce(logits, "max", lengths=layout.sizes)
    centered = logits - maxima.index_select(0, layout.index)
    totals = torch.segment_reduce(centered.exp(), "sum", lengths=layout.sizes)
    return centered - totals.clamp_min(torch.finfo(torch.float32).tiny).log().index_select(
        0, layout.index
    )


def _group_mean(value: Tensor, selected: Tensor, *, zero: Tensor) -> Tensor:
    if value.shape != selected.shape:
        raise ValueError("semantic group values and selection mask disagree")
    chosen = value[selected]
    return chosen.mean() if chosen.numel() else zero.sum() * 0.0


def _virtual_lane_layout(
    group_ptr: Tensor | None,
    *,
    groups: int,
    device: torch.device,
) -> tuple[tuple[int, ...], int] | None:
    if group_ptr is None:
        return None
    if (
        not isinstance(group_ptr, Tensor)
        or group_ptr.ndim != 1
        or group_ptr.numel() < 2
        or group_ptr.dtype not in {torch.int32, torch.int64}
    ):
        raise TypeError("virtual_microbatch_group_ptr must be a non-empty integer CSR pointer")
    # This is tiny control metadata, not a model tensor. Keep the offsets on
    # CPU so repeated loss-component reductions do not introduce CUDA scalar
    # synchronizations.
    pointer = group_ptr.detach().to(device="cpu", dtype=torch.long)
    sizes = pointer[1:] - pointer[:-1]
    _assert(
        (pointer[0] == 0) & (pointer[-1] == groups) & (sizes > 0).all(),
        "virtual_microbatch_group_ptr must cover all groups with non-empty lanes",
    )
    lanes = int(pointer.numel() - 1)
    return tuple(pointer.tolist()), lanes


def _virtual_lane_mean(
    value: Tensor,
    selected: Tensor,
    *,
    zero: Tensor,
    layout: tuple[tuple[int, ...], int] | None,
) -> Tensor:
    if layout is None:
        return _group_mean(value, selected, zero=zero)
    if value.shape != selected.shape:
        raise ValueError("semantic group values and selection mask disagree")
    pointer, lanes = layout
    lane_means: list[Tensor] = []
    for lane in range(lanes):
        start = pointer[lane]
        end = pointer[lane + 1]
        chosen = value[start:end][selected[start:end]]
        # A historical microbatch with no eligible group contributed a
        # differentiable zero before accumulation. Retain that behavior.
        lane_means.append(chosen.mean() if chosen.numel() else zero.sum() * 0.0)
    return torch.stack(lane_means).mean() if lane_means else zero.sum() * 0.0


def _virtual_lane_selected_count(
    selected: Tensor,
    *,
    layout: tuple[tuple[int, ...], int] | None,
) -> Tensor:
    if layout is None:
        return selected.float().sum()
    pointer, lanes = layout
    counts = [
        selected[pointer[lane] : pointer[lane + 1]].float().sum()
        for lane in range(lanes)
    ]
    return torch.stack(counts).mean()


class SemanticAlignmentObjective(nn.Module):
    """Align grouped guidance text with frozen route semantic embeddings."""

    def __init__(
        self,
        *,
        selected_branch: str = "structure",
        pair_weight: float = 0.5,
        pair_margin: float = 0.1,
        endpoint_probability_floor: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if selected_branch not in {"structure", "metric", "all"}:
            raise ValueError("selected_branch must be structure, metric, or all")
        for name, value, positive in (
            ("pair_weight", pair_weight, False),
            ("pair_margin", pair_margin, False),
            ("endpoint_probability_floor", endpoint_probability_floor, True),
        ):
            number = float(value)
            if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        self.selected_branch = selected_branch
        self.pair_weight = float(pair_weight)
        self.pair_margin = float(pair_margin)
        self.endpoint_probability_floor = float(endpoint_probability_floor)

    def forward(
        self,
        *,
        group_embeddings: Mapping[str, Tensor],
        route_embedding: Tensor,
        group_ptr: Tensor,
        group_index: Tensor,
        teacher_probability: Tensor,
        endpoint_probability: Tensor,
        structural_label: Tensor,
        guidance_branch: Sequence[str],
        listwise_eligible: Tensor,
        logit_scale: float,
        virtual_microbatch_group_ptr: Tensor | None = None,
    ) -> SemanticAlignmentOutput:
        if not isinstance(route_embedding, Tensor) or route_embedding.ndim != 2:
            raise TypeError("route_embedding must be a [candidate, semantic_dim] tensor")
        if route_embedding.requires_grad:
            raise ValueError("route_embedding must remain frozen")
        candidates, semantic_dim = route_embedding.shape
        if candidates < 1 or semantic_dim < 1:
            raise ValueError("route_embedding must contain candidates and semantic features")
        device = route_embedding.device
        route = F.normalize(route_embedding.detach().float(), dim=-1, eps=1.0e-8)
        _assert(torch.isfinite(route).all(), "route_embedding contains invalid values")
        layout = _layout(group_ptr, candidates, device)
        if (
            not isinstance(group_index, Tensor)
            or group_index.shape != (candidates,)
            or group_index.dtype not in {torch.int32, torch.int64}
        ):
            raise TypeError(f"group_index must be integer [{candidates}]")
        _assert(
            (group_index.to(device=device, dtype=torch.long) == layout.index).all(),
            "group_index disagrees with group_ptr",
        )
        if not isinstance(teacher_probability, Tensor) or teacher_probability.shape != (candidates,):
            raise ValueError(f"teacher_probability must have shape [{candidates}]")
        teacher = teacher_probability.to(device=device, dtype=torch.float32)
        _assert(torch.isfinite(teacher).all() & (teacher >= 0.0).all(), "teacher probabilities must be finite")
        _assert(
            torch.isclose(
                torch.segment_reduce(teacher, "sum", lengths=layout.sizes),
                torch.ones(layout.groups, device=device),
                atol=1.0e-6,
                rtol=0.0,
            ).all(),
            "teacher probabilities must sum to one within each group",
        )
        if not isinstance(endpoint_probability, Tensor) or endpoint_probability.shape != (candidates,):
            raise ValueError(f"endpoint_probability must have shape [{candidates}]")
        endpoint = endpoint_probability.to(device=device, dtype=torch.float32)
        _assert(torch.isfinite(endpoint).all() & (endpoint >= 0.0).all(), "endpoint probabilities must be finite")
        _assert(
            torch.isclose(
                torch.segment_reduce(endpoint, "sum", lengths=layout.sizes),
                torch.ones(layout.groups, device=device),
                atol=1.0e-6,
                rtol=0.0,
            ).all(),
            "endpoint probabilities must sum to one within each group",
        )
        if not isinstance(structural_label, Tensor) or structural_label.shape != (candidates,):
            raise ValueError(f"structural_label must have shape [{candidates}]")
        label = structural_label.to(device=device, dtype=torch.long)
        _assert(((label == 0) | (label == 1)).all(), "structural labels must be binary")
        if not isinstance(listwise_eligible, Tensor) or listwise_eligible.shape != (layout.groups,):
            raise ValueError(f"listwise_eligible must have shape [{layout.groups}]")
        if listwise_eligible.dtype != torch.bool:
            raise TypeError("listwise_eligible must use bool dtype")
        if not isinstance(guidance_branch, (tuple, list)) or len(guidance_branch) != layout.groups:
            raise ValueError(f"guidance_branch must contain {layout.groups} values")
        branch = tuple(str(value) for value in guidance_branch)
        if set(branch) - {"metric", "structure"}:
            raise ValueError("guidance_branch contains unsupported values")
        if self.selected_branch == "all":
            branch_selected = torch.ones(layout.groups, dtype=torch.bool, device=device)
        else:
            branch_selected = torch.tensor(
                [value == self.selected_branch for value in branch],
                dtype=torch.bool,
                device=device,
            )
        positive_count = torch.segment_reduce((label == 1).float(), "sum", lengths=layout.sizes)
        negative_count = torch.segment_reduce((label == 0).float(), "sum", lengths=layout.sizes)
        selected = branch_selected & listwise_eligible.to(device=device) & (positive_count > 0) & (negative_count > 0)
        virtual_lane_layout = _virtual_lane_layout(
            virtual_microbatch_group_ptr,
            groups=layout.groups,
            device=device,
        )
        scale = float(logit_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("logit_scale must be finite and positive")
        same_group = layout.index[:, None] == layout.index[None, :]
        pair_rows, pair_cols = torch.nonzero(
            (label == 1)[:, None] & (label == 0)[None, :] & same_group,
            as_tuple=True,
        )
        pair_group = layout.index.index_select(0, pair_rows)
        projector_losses: dict[str, Tensor] = {}
        group_kls: list[Tensor] = []
        group_pairs: list[Tensor] = []
        group_accuracies: list[Tensor] = []
        group_margins: list[Tensor] = []
        group_top1s: list[Tensor] = []
        for name, embedding in group_embeddings.items():
            if not isinstance(name, str) or not name:
                raise ValueError("semantic projector names must be non-empty strings")
            if not isinstance(embedding, Tensor) or embedding.shape != (layout.groups, semantic_dim):
                raise ValueError(
                    f"semantic projector {name!r} must have shape [{layout.groups},{semantic_dim}]"
                )
            text = F.normalize(embedding.float(), dim=-1, eps=1.0e-8)
            _assert(torch.isfinite(text).all(), f"semantic projector {name!r} contains invalid values")
            score = scale * (text.index_select(0, layout.index) * route).sum(dim=-1)
            logits = endpoint.clamp_min(self.endpoint_probability_floor).log() + score
            log_probability = _group_log_softmax(logits, layout)
            candidate_kl = torch.special.xlogy(teacher, teacher) - teacher * log_probability
            group_kl = torch.segment_reduce(candidate_kl, "sum", lengths=layout.sizes)
            pair_difference = score.index_select(0, pair_rows) - score.index_select(0, pair_cols)
            pair_value = F.softplus(score.new_tensor(self.pair_margin) - pair_difference)
            pair_sum = torch.zeros(layout.groups, dtype=torch.float32, device=device)
            pair_count = torch.zeros_like(pair_sum)
            pair_correct = torch.zeros_like(pair_sum)
            pair_margin_sum = torch.zeros_like(pair_sum)
            pair_sum.index_add_(0, pair_group, pair_value)
            pair_count.index_add_(0, pair_group, torch.ones_like(pair_value))
            pair_correct.index_add_(0, pair_group, (pair_difference > 0).float())
            pair_margin_sum.index_add_(0, pair_group, pair_difference)
            denominator = pair_count.clamp_min(1.0)
            group_pair = pair_sum / denominator
            group_accuracy = pair_correct / denominator
            group_margin = pair_margin_sum / denominator
            predicted_max = torch.segment_reduce(logits, "max", lengths=layout.sizes)
            teacher_max = torch.segment_reduce(teacher, "max", lengths=layout.sizes)
            predicted_top = logits == predicted_max.index_select(0, layout.index)
            teacher_top = teacher == teacher_max.index_select(0, layout.index)
            group_top1 = torch.segment_reduce(
                (predicted_top & teacher_top).float(), "max", lengths=layout.sizes
            )
            group_loss = group_kl + self.pair_weight * group_pair
            projector_losses[name] = _virtual_lane_mean(
                group_loss,
                selected,
                zero=score,
                layout=virtual_lane_layout,
            )
            group_kls.append(group_kl)
            group_pairs.append(group_pair)
            group_accuracies.append(group_accuracy)
            group_margins.append(group_margin)
            group_top1s.append(group_top1)
        if not projector_losses:
            raise ValueError("semantic alignment needs at least one projector embedding")
        loss = torch.stack(tuple(projector_losses.values())).mean()
        group_kl = torch.stack(group_kls).mean(dim=0)
        group_pair = torch.stack(group_pairs).mean(dim=0)
        group_accuracy = torch.stack(group_accuracies).mean(dim=0)
        group_margin = torch.stack(group_margins).mean(dim=0)
        group_top1 = torch.stack(group_top1s).mean(dim=0)
        return SemanticAlignmentOutput(
            loss=loss,
            listwise_kl=_virtual_lane_mean(
                group_kl, selected, zero=loss, layout=virtual_lane_layout
            ),
            pairwise_loss=_virtual_lane_mean(
                group_pair, selected, zero=loss, layout=virtual_lane_layout
            ),
            pair_accuracy=_virtual_lane_mean(
                group_accuracy, selected, zero=loss, layout=virtual_lane_layout
            ),
            score_margin=_virtual_lane_mean(
                group_margin, selected, zero=loss, layout=virtual_lane_layout
            ),
            top1_accuracy=_virtual_lane_mean(
                group_top1, selected, zero=loss, layout=virtual_lane_layout
            ),
            selected_group_count=_virtual_lane_selected_count(
                selected, layout=virtual_lane_layout
            ),
            projector_losses=projector_losses,
            group_listwise_kl=group_kl.detach(),
            group_pairwise_loss=group_pair.detach(),
            group_pair_accuracy=group_accuracy.detach(),
            group_score_margin=group_margin.detach(),
            group_top1_accuracy=group_top1.detach(),
            selected_group_mask=selected.detach(),
        )


__all__ = ["SemanticAlignmentObjective", "SemanticAlignmentOutput"]
