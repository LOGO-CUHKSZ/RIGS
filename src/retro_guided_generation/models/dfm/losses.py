"""Numerically stable per-graph DFM loss reductions.

Legacy GraphDiT training reduces all valid positions to one scalar.  Guided
training additionally requires a loss per candidate so routes can be compared
within a semantic group.  The functions here keep those reductions explicit
and perform cross entropy and reductions in float32 under mixed precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def _assert_condition(condition: Tensor, message: str) -> None:
    if condition.numel() != 1:
        raise ValueError("validation condition must be scalar")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class DfmLossParts:
    """Node, edge, and weighted-total losses, scalar or per graph."""

    node: Tensor
    edge: Tensor
    total: Tensor

    @property
    def loss_X(self) -> Tensor:
        return self.node

    @property
    def loss_E(self) -> Tensor:
        return self.edge

    @property
    def loss(self) -> Tensor:
        return self.total


@dataclass(frozen=True, slots=True)
class GroupPositionMasks:
    """Candidate-shaped preference/common position partitions."""

    preference_node: Tensor
    preference_edge: Tensor
    common_node: Tensor
    common_edge: Tensor
    eligible: Tensor


def categorical_position_ce(logits: Tensor, target_onehot: Tensor) -> Tensor:
    """Compute hard-target categorical cross entropy at every graph position."""

    if logits.ndim < 2:
        raise ValueError(f"logits must have a class dimension, got {logits.shape}")
    if logits.shape != target_onehot.shape:
        raise ValueError(
            "logits and target_onehot must have identical shapes, got "
            f"{tuple(logits.shape)} and {tuple(target_onehot.shape)}"
        )
    if logits.size(-1) < 2:
        raise ValueError("categorical logits must contain at least two classes")
    flat_logits = logits.float().reshape(-1, logits.size(-1))
    target_ids = target_onehot.argmax(dim=-1).reshape(-1)
    position_loss = F.cross_entropy(flat_logits, target_ids, reduction="none")
    return position_loss.reshape(logits.shape[:-1])


def reduce_masked_per_graph(position_loss: Tensor, mask: Tensor) -> Tensor:
    """Average valid positions independently for every graph in a batch."""

    if position_loss.ndim < 1:
        raise ValueError("position_loss must include a batch dimension")
    if position_loss.shape != mask.shape:
        raise ValueError(
            f"position_loss and mask shapes differ: {position_loss.shape} vs {mask.shape}"
        )
    loss32 = position_loss.float()
    mask32 = mask.to(device=loss32.device, dtype=torch.float32)
    reduce_dims = tuple(range(1, loss32.ndim))
    if not reduce_dims:
        return loss32 * mask32 / mask32.clamp_min(1.0)
    return (loss32 * mask32).sum(dim=reduce_dims) / mask32.sum(dim=reduce_dims).clamp_min(1.0)


def legacy_masked_reduce(position_loss: Tensor, mask: Tensor) -> Tensor:
    """Reproduce the legacy all-position scalar DFM reduction."""

    if position_loss.shape != mask.shape:
        raise ValueError(
            f"position_loss and mask shapes differ: {position_loss.shape} vs {mask.shape}"
        )
    loss32 = position_loss.float()
    mask32 = mask.to(device=loss32.device, dtype=torch.float32)
    return (loss32 * mask32).sum() / mask32.sum().clamp_min(1.0)


def _validate_edge_weight(edge_weight: float) -> float:
    value = float(edge_weight)
    if not torch.isfinite(torch.tensor(value)) or value < 0.0:
        raise ValueError(f"edge_weight must be finite and non-negative, got {edge_weight}")
    return value


def weighted_position_reduce(
    node_position_loss: Tensor,
    edge_position_loss: Tensor,
    node_mask: Tensor,
    edge_mask: Tensor,
    edge_weight: float = 10.0,
) -> DfmLossParts:
    """Reduce nodes and edges with one weighted denominator per graph."""

    if node_position_loss.ndim < 2 or edge_position_loss.ndim < 2:
        raise ValueError("node and edge position losses must have shape [B,...]")
    if node_position_loss.shape != node_mask.shape:
        raise ValueError("node_position_loss and node_mask shapes differ")
    if edge_position_loss.shape != edge_mask.shape:
        raise ValueError("edge_position_loss and edge_mask shapes differ")
    if node_position_loss.size(0) != edge_position_loss.size(0):
        raise ValueError("node and edge position losses disagree on batch size")
    if node_position_loss.device != edge_position_loss.device:
        raise ValueError("node and edge position losses must be on one device")
    weight = _validate_edge_weight(edge_weight)
    node_loss32 = node_position_loss.float()
    edge_loss32 = edge_position_loss.float()
    node_mask32 = node_mask.to(device=node_loss32.device, dtype=torch.float32)
    edge_mask32 = edge_mask.to(device=edge_loss32.device, dtype=torch.float32)
    node_dims = tuple(range(1, node_loss32.ndim))
    edge_dims = tuple(range(1, edge_loss32.ndim))
    node_sum = (node_loss32 * node_mask32).sum(dim=node_dims)
    edge_sum = (edge_loss32 * edge_mask32).sum(dim=edge_dims)
    node_count = node_mask32.sum(dim=node_dims)
    edge_count = edge_mask32.sum(dim=edge_dims)
    node = node_sum / node_count.clamp_min(1.0)
    edge = edge_sum / edge_count.clamp_min(1.0)
    total = (node_sum + weight * edge_sum) / (node_count + weight * edge_count).clamp_min(1.0)
    return DfmLossParts(node=node, edge=edge, total=total)


def _resolve_masks(
    masks: Mapping[str, Tensor] | Tensor,
    *,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if isinstance(masks, Tensor):
        node_mask = masks
        edge_mask = masks.unsqueeze(-1) & masks.unsqueeze(-2)
    elif isinstance(masks, Mapping):
        node_mask = masks.get("X", masks.get("node"))
        edge_mask = masks.get("E", masks.get("edge"))
        if node_mask is None:
            raise KeyError("masks must contain 'X' (or 'node')")
        if edge_mask is None:
            edge_mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
    else:
        raise TypeError("masks must be a Tensor or mapping of node/edge masks")
    expected_node = (batch_size, num_nodes)
    expected_edge = (batch_size, num_nodes, num_nodes)
    if tuple(node_mask.shape) != expected_node:
        raise ValueError(f"node mask must be {expected_node}, got {tuple(node_mask.shape)}")
    if tuple(edge_mask.shape) != expected_edge:
        raise ValueError(f"edge mask must be {expected_edge}, got {tuple(edge_mask.shape)}")
    return node_mask.to(device=device, dtype=torch.bool), edge_mask.to(device=device, dtype=torch.bool)


def _validate_target_tensors(targets: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    try:
        target_x, target_e = targets["X"], targets["E"]
    except KeyError as error:
        raise KeyError("targets must contain 'X' and 'E'") from error
    if target_x.ndim != 3 or target_x.size(-1) < 2:
        raise ValueError("node targets must have shape [B,N,Kx] with Kx >= 2")
    if target_e.ndim != 4 or target_e.size(-1) < 2:
        raise ValueError("edge targets must have shape [B,N,N,Ke] with Ke >= 2")
    if target_e.shape[:3] != (target_x.size(0), target_x.size(1), target_x.size(1)):
        raise ValueError("node and edge targets disagree on batch or node dimensions")
    if target_e.device != target_x.device:
        raise ValueError("node and edge targets must be on one device")
    return target_x, target_e


def _validate_graph_tensors(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    try:
        logits_x, logits_e = logits["X"], logits["E"]
    except KeyError as error:
        raise KeyError("logits must contain 'X' and 'E'") from error
    target_x, target_e = _validate_target_tensors(targets)
    if logits_x.ndim != 3 or target_x.shape != logits_x.shape:
        raise ValueError("node logits/targets must have the same [B,N,Kx] shape")
    if logits_e.ndim != 4 or target_e.shape != logits_e.shape:
        raise ValueError("edge logits/targets must have the same [B,N,N,Ke] shape")
    if logits_e.shape[:3] != (logits_x.size(0), logits_x.size(1), logits_x.size(1)):
        raise ValueError("node and edge tensors disagree on batch or node dimensions")
    if any(value.device != logits_x.device for value in (logits_e, target_x, target_e)):
        raise ValueError("all logits and targets must be on one device")
    return logits_x, logits_e, target_x, target_e


def full_dfm_per_graph(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | Tensor,
    edge_weight: float = 10.0,
) -> DfmLossParts:
    """Compute the legacy full node/edge DFM objective for every graph."""

    logits_x, logits_e, target_x, target_e = _validate_graph_tensors(logits, targets)
    node_mask, edge_mask = _resolve_masks(
        masks,
        batch_size=logits_x.size(0),
        num_nodes=logits_x.size(1),
        device=logits_x.device,
    )
    weight = _validate_edge_weight(edge_weight)
    node = reduce_masked_per_graph(categorical_position_ce(logits_x, target_x), node_mask)
    edge = reduce_masked_per_graph(categorical_position_ce(logits_e, target_e), edge_mask)
    return DfmLossParts(node=node, edge=edge, total=node + weight * edge)


def _group_layout(
    group_ptr: Tensor,
    batch_size: int,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(group_ptr, Tensor) or group_ptr.ndim != 1:
        raise ValueError("group_ptr must be a one-dimensional Tensor")
    if group_ptr.numel() < 2:
        raise ValueError("group_ptr must describe at least one group")
    if group_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_ptr must use an integer dtype")
    ptr = group_ptr.to(device=device, dtype=torch.long)
    sizes = ptr[1:] - ptr[:-1]
    valid = (ptr[0] == 0) & (ptr[-1] == batch_size) & torch.all(sizes > 0)
    if ptr.device.type == "cuda":
        _assert_condition(valid, "group_ptr must contain positive contiguous segments")
    elif not bool(valid):
        values = [int(value) for value in ptr.detach().cpu().tolist()]
        if values[0] != 0 or values[-1] != batch_size:
            raise ValueError(f"group_ptr must start at 0 and end at {batch_size}, got {values}")
        raise ValueError("every group must contain at least one contiguous candidate")
    num_groups = int(ptr.numel() - 1)
    group_index = torch.repeat_interleave(
        torch.arange(num_groups, device=device, dtype=torch.long),
        sizes,
        output_size=batch_size,
    )
    return sizes, group_index, ptr[:-1]


def positive_full_dfm_per_graph(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | Tensor,
    group_ptr: Tensor,
    structural_label: Tensor,
    teacher_probability: Tensor,
    edge_weight: float = 10.0,
) -> tuple[DfmLossParts, Tensor, Tensor]:
    """Teacher-weight full DFM loss over structural positive candidates."""

    logits_x, logits_e, target_x, target_e = _validate_graph_tensors(logits, targets)
    batch_size = logits_x.size(0)
    if not isinstance(structural_label, Tensor) or structural_label.shape != (batch_size,):
        raise ValueError(f"structural_label must have shape [{batch_size}]")
    labels = structural_label.to(device=logits_x.device, dtype=torch.float32)
    _assert_condition(((labels == 0) | (labels == 1)).all(), "structural_label must contain only 0 and 1")
    if not isinstance(teacher_probability, Tensor) or teacher_probability.shape != (batch_size,):
        raise ValueError(f"teacher_probability must have shape [{batch_size}]")
    probability = teacher_probability.to(device=logits_x.device, dtype=torch.float32)
    _assert_condition(
        torch.isfinite(probability).all() & torch.all(probability >= 0.0),
        "teacher_probability must contain finite non-negative values",
    )
    group_sizes, group_index, _group_first = _group_layout(group_ptr, batch_size, device=logits_x.device)
    group_probability_sum = torch.segment_reduce(probability, "sum", lengths=group_sizes)
    _assert_condition(
        torch.isclose(group_probability_sum, torch.ones_like(group_probability_sum), atol=1e-6, rtol=0.0).all(),
        "teacher_probability must be normalized by group",
    )
    positive = labels == 1
    positive_mass = torch.where(positive, probability, torch.zeros_like(probability))
    positive_group_mass = torch.segment_reduce(positive_mass, "sum", lengths=group_sizes)
    positive_count = torch.segment_reduce(positive.to(torch.float32), "sum", lengths=group_sizes)
    _assert_condition(
        ((positive_count <= 0.0) | (positive_group_mass > torch.finfo(torch.float32).tiny)).all(),
        "groups with positive candidates must have positive teacher mass",
    )
    positive_probability = positive_mass / positive_group_mass.index_select(0, group_index).clamp_min(torch.finfo(torch.float32).tiny)
    candidate_parts = full_dfm_per_graph(
        {"X": logits_x, "E": logits_e},
        {"X": target_x, "E": target_e},
        masks,
        edge_weight=edge_weight,
    )
    return (
        DfmLossParts(
            node=torch.segment_reduce(candidate_parts.node * positive_probability, "sum", lengths=group_sizes),
            edge=torch.segment_reduce(candidate_parts.edge * positive_probability, "sum", lengths=group_sizes),
            total=torch.segment_reduce(candidate_parts.total * positive_probability, "sum", lengths=group_sizes),
        ),
        positive_probability,
        positive_count,
    )


def squared_hinge(value: Tensor, margin: float) -> Tensor:
    """Return ``max(value-margin, 0)^2`` elementwise in float32."""

    if not isinstance(value, Tensor):
        raise TypeError("value must be a Tensor")
    budget = float(margin)
    if not torch.isfinite(torch.tensor(budget)) or budget < 0.0:
        raise ValueError(f"margin must be finite and non-negative, got {margin}")
    return (value.float() - budget).clamp_min(0.0).square()


def _validate_position_masks(
    position_masks: GroupPositionMasks,
    *,
    batch_size: int,
    num_nodes: int,
    device: torch.device,
) -> None:
    if not isinstance(position_masks, GroupPositionMasks):
        raise TypeError("position_masks must be a GroupPositionMasks")
    expected = {
        "preference_node": (batch_size, num_nodes),
        "preference_edge": (batch_size, num_nodes, num_nodes),
        "common_node": (batch_size, num_nodes),
        "common_edge": (batch_size, num_nodes, num_nodes),
    }
    for name, shape in expected.items():
        value = getattr(position_masks, name)
        if value.shape != shape or value.dtype != torch.bool or value.device != device:
            raise ValueError(f"position_masks.{name} must be bool{list(shape)} on {device}")
    if (
        position_masks.eligible.ndim != 1
        or position_masks.eligible.dtype != torch.bool
        or position_masks.eligible.device != device
    ):
        raise ValueError("position_masks.eligible must be a one-dimensional bool Tensor")


def build_group_position_masks(
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | Tensor,
    group_ptr: Tensor,
) -> GroupPositionMasks:
    """Partition valid positions into within-group preference and common masks."""

    target_x, target_e = _validate_target_tensors(targets)
    batch_size, num_nodes = target_x.shape[:2]
    node_mask, edge_mask = _resolve_masks(
        masks,
        batch_size=batch_size,
        num_nodes=num_nodes,
        device=target_x.device,
    )
    group_sizes, group_index, group_first = _group_layout(group_ptr, batch_size, device=target_x.device)
    target_x_ids = target_x.argmax(dim=-1)
    target_e_ids = target_e.argmax(dim=-1)
    strict_upper = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=target_x.device).triu(diagonal=1)
    valid_upper_edge = edge_mask & strict_upper.unsqueeze(0)
    group_node_mask = node_mask.index_select(0, group_first)
    group_edge_mask = edge_mask.index_select(0, group_first)
    expected_node_mask = group_node_mask.index_select(0, group_index)
    expected_edge_mask = group_edge_mask.index_select(0, group_index)
    masks_match = torch.all(node_mask == expected_node_mask) & torch.all(edge_mask == expected_edge_mask)
    if masks_match.device.type == "cuda":
        _assert_condition(masks_match, "node or edge masks differ inside a group")
    elif not bool(masks_match):
        node_mismatch = (node_mask != expected_node_mask).flatten(1).any(1)
        edge_mismatch = (edge_mask != expected_edge_mask).flatten(1).any(1)
        if bool(node_mismatch.any()):
            candidate = int(torch.nonzero(node_mismatch, as_tuple=False)[0])
            raise ValueError(f"node masks differ inside group {int(group_index[candidate])}")
        candidate = int(torch.nonzero(edge_mismatch, as_tuple=False)[0])
        raise ValueError(f"edge masks differ inside group {int(group_index[candidate])}")
    first_node_ids = target_x_ids.index_select(0, group_first).index_select(0, group_index)
    first_edge_ids = target_e_ids.index_select(0, group_first).index_select(0, group_index)
    node_disagreement = torch.segment_reduce(
        (target_x_ids != first_node_ids).to(torch.float32), "max", lengths=group_sizes
    ).bool()
    edge_disagreement = torch.segment_reduce(
        (target_e_ids != first_edge_ids).to(torch.float32), "max", lengths=group_sizes
    ).bool()
    group_preference_node = group_node_mask & node_disagreement
    group_preference_edge = group_edge_mask & strict_upper.unsqueeze(0) & edge_disagreement
    preference_node = group_preference_node.index_select(0, group_index)
    preference_edge = group_preference_edge.index_select(0, group_index)
    eligible = group_preference_node.any(dim=1) | group_preference_edge.flatten(1).any(dim=1)
    return GroupPositionMasks(
        preference_node=preference_node,
        preference_edge=preference_edge,
        common_node=node_mask & ~preference_node,
        common_edge=valid_upper_edge & ~preference_edge,
        eligible=eligible,
    )


def preference_dfm_per_graph(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | Tensor,
    group_ptr: Tensor,
    edge_weight: float = 10.0,
    *,
    position_masks: GroupPositionMasks | None = None,
) -> tuple[DfmLossParts, Tensor]:
    """Compute per-graph DFM over within-group disagreement positions."""

    logits_x, logits_e, target_x, target_e = _validate_graph_tensors(logits, targets)
    batch_size, num_nodes = logits_x.shape[:2]
    if position_masks is None:
        position_masks = build_group_position_masks(targets, masks, group_ptr)
    _validate_position_masks(position_masks, batch_size=batch_size, num_nodes=num_nodes, device=logits_x.device)
    parts = weighted_position_reduce(
        categorical_position_ce(logits_x, target_x),
        categorical_position_ce(logits_e, target_e),
        position_masks.preference_node,
        position_masks.preference_edge,
        edge_weight=edge_weight,
    )
    return parts, position_masks.eligible


def common_dfm_per_graph(
    logits: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | Tensor,
    group_ptr: Tensor,
    edge_weight: float = 10.0,
    *,
    position_masks: GroupPositionMasks | None = None,
) -> DfmLossParts:
    """Compute hard-target DFM on within-group common positions."""

    logits_x, logits_e, target_x, target_e = _validate_graph_tensors(logits, targets)
    batch_size, num_nodes = logits_x.shape[:2]
    if position_masks is None:
        position_masks = build_group_position_masks(targets, masks, group_ptr)
    _validate_position_masks(position_masks, batch_size=batch_size, num_nodes=num_nodes, device=logits_x.device)
    return weighted_position_reduce(
        categorical_position_ce(logits_x, target_x),
        categorical_position_ce(logits_e, target_e),
        position_masks.common_node,
        position_masks.common_edge,
        edge_weight=edge_weight,
    )


def _validate_logit_pair(
    student_logits: Mapping[str, Tensor],
    reference_logits: Mapping[str, Tensor],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    try:
        student_x, student_e = student_logits["X"], student_logits["E"]
        reference_x, reference_e = reference_logits["X"], reference_logits["E"]
    except KeyError as error:
        raise KeyError("student and reference logits must contain 'X' and 'E'") from error
    if student_x.ndim != 3 or reference_x.shape != student_x.shape:
        raise ValueError("student/reference node logits must share a [B,N,Kx] shape")
    if student_e.ndim != 4 or reference_e.shape != student_e.shape:
        raise ValueError("student/reference edge logits must share a [B,N,N,Ke] shape")
    if student_e.shape[:3] != (student_x.size(0), student_x.size(1), student_x.size(1)):
        raise ValueError("node and edge logits disagree on batch or node dimensions")
    if any(value.device != student_x.device for value in (student_e, reference_x, reference_e)):
        raise ValueError("student and reference logits must be on one device")
    return student_x, student_e, reference_x, reference_e


def _categorical_reference_kl(student_logits: Tensor, reference_logits: Tensor) -> Tensor:
    reference_log_probability = F.log_softmax(reference_logits.detach().float(), dim=-1)
    reference_probability = reference_log_probability.exp()
    student_log_probability = F.log_softmax(student_logits.float(), dim=-1)
    return (reference_probability * (reference_log_probability - student_log_probability)).sum(dim=-1)


def common_reference_kl_per_graph(
    student_logits: Mapping[str, Tensor],
    reference_logits: Mapping[str, Tensor],
    position_masks: GroupPositionMasks,
    edge_weight: float = 10.0,
) -> DfmLossParts:
    """Compute normalized KL(reference || student) on common positions."""

    student_x, student_e, reference_x, reference_e = _validate_logit_pair(student_logits, reference_logits)
    _validate_position_masks(
        position_masks,
        batch_size=student_x.size(0),
        num_nodes=student_x.size(1),
        device=student_x.device,
    )
    return weighted_position_reduce(
        _categorical_reference_kl(student_x, reference_x),
        _categorical_reference_kl(student_e, reference_e),
        position_masks.common_node,
        position_masks.common_edge,
        edge_weight=edge_weight,
    )


def full_reference_kl_per_graph(
    student_logits: Mapping[str, Tensor],
    reference_logits: Mapping[str, Tensor],
    masks: Mapping[str, Tensor] | Tensor,
    edge_weight: float = 10.0,
) -> DfmLossParts:
    """Compute KL(reference || student) on every valid graph position."""

    student_x, student_e, reference_x, reference_e = _validate_logit_pair(student_logits, reference_logits)
    node_mask, edge_mask = _resolve_masks(
        masks,
        batch_size=student_x.size(0),
        num_nodes=student_x.size(1),
        device=student_x.device,
    )
    return weighted_position_reduce(
        _categorical_reference_kl(student_x, reference_x),
        _categorical_reference_kl(student_e, reference_e),
        node_mask,
        edge_mask,
        edge_weight=edge_weight,
    )


__all__ = [
    "DfmLossParts",
    "GroupPositionMasks",
    "build_group_position_masks",
    "categorical_position_ce",
    "common_dfm_per_graph",
    "common_reference_kl_per_graph",
    "full_dfm_per_graph",
    "full_reference_kl_per_graph",
    "legacy_masked_reduce",
    "positive_full_dfm_per_graph",
    "preference_dfm_per_graph",
    "reduce_masked_per_graph",
    "squared_hinge",
    "weighted_position_reduce",
]
