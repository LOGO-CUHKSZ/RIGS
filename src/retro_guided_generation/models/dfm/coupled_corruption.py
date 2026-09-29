"""Group-coupled predefined corruption for guidance preference training.

The selector field is intentionally supplied as tensors or sampled from an
explicit ``torch.Generator``.  It is never identified by a content hash: the
stage manifest records the seed and explicit traversal coordinates instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _assert_condition(condition: Tensor, message: str) -> None:
    if condition.numel() != 1:
        raise ValueError("validation condition must be scalar")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class CorruptedGraphBatch:
    """One immutable noisy batch shared by student and frozen reference."""

    t_group: Tensor
    t_graph: Tensor
    t_X: Tensor
    t_E: Tensor
    target_X: Tensor
    target_E: Tensor
    p_X: Tensor
    p_E: Tensor
    node_mask: Tensor
    p_mask: Tensor

    def model_inputs(self) -> dict[str, Tensor]:
        return {
            "t_X": self.t_X,
            "t_E": self.t_E,
            "p_X": self.p_X,
            "p_E": self.p_E,
            "p_mask": self.p_mask,
            "node_mask": self.node_mask,
        }


@dataclass(frozen=True, slots=True)
class CoupledSelectorField:
    """Reusable group-level uniforms defining one monotone DFM coupling."""

    node_uniform_group: Tensor
    edge_uniform_group: Tensor

    def audit_summary(self) -> dict[str, object]:
        """Report shape/dtype metadata without deriving an identifier from data."""

        return {
            "node_shape": list(self.node_uniform_group.shape),
            "edge_shape": list(self.edge_uniform_group.shape),
            "node_dtype": str(self.node_uniform_group.dtype),
            "edge_dtype": str(self.edge_uniform_group.dtype),
        }


def _require_tensor(batch: Any, name: str) -> Tensor:
    value = getattr(batch, name, None)
    if not isinstance(value, Tensor):
        raise TypeError(f"batch.{name} must be a Tensor")
    return value


def _validate_inputs(
    batch: Any,
    group_index: Tensor,
    num_groups: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    p_x = _require_tensor(batch, "p_X")
    p_e = _require_tensor(batch, "p_E")
    target_x = _require_tensor(batch, "r_X")
    target_e = _require_tensor(batch, "r_E")
    node_mask = _require_tensor(batch, "node_mask")
    p_mask = _require_tensor(batch, "p_mask")
    if p_x.ndim != 3 or p_x.shape != target_x.shape:
        raise ValueError("p_X and r_X must share a [M,N,Kx] shape")
    if p_e.ndim != 4 or p_e.shape != target_e.shape:
        raise ValueError("p_E and r_E must share a [M,N,N,Ke] shape")
    batch_size, num_nodes = p_x.shape[:2]
    if p_e.shape[:3] != (batch_size, num_nodes, num_nodes):
        raise ValueError("node and edge tensors disagree on batch or node dimensions")
    if node_mask.shape != (batch_size, num_nodes):
        raise ValueError("node_mask must have shape [M,N]")
    if p_mask.shape != (batch_size, num_nodes):
        raise ValueError("p_mask must have shape [M,N]")
    tensors = (p_x, p_e, target_x, target_e, node_mask, p_mask)
    if any(value.device != p_x.device for value in tensors):
        raise ValueError("all graph tensors must be on the same device")
    if not isinstance(group_index, Tensor) or group_index.shape != (batch_size,):
        raise ValueError(f"group_index must have shape [{batch_size}]")
    if group_index.dtype not in (torch.int32, torch.int64):
        raise TypeError("group_index must use an integer dtype")
    if group_index.device != p_x.device:
        raise ValueError("group_index and graph tensors must be on the same device")
    if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 1:
        raise ValueError("num_groups must be a positive integer")
    if batch_size < num_groups:
        raise ValueError("num_groups cannot exceed the number of candidate graphs")
    _assert_condition(
        torch.all((group_index >= 0) & (group_index < num_groups)),
        "group_index contains an out-of-range group id",
    )
    counts = torch.bincount(group_index.long(), minlength=num_groups)
    _assert_condition(
        torch.all(counts > 0),
        "every group id in [0,num_groups) must be represented",
    )
    expected = torch.repeat_interleave(
        torch.arange(num_groups, device=p_x.device, dtype=torch.long),
        counts,
        output_size=batch_size,
    )
    _assert_condition(
        torch.all(group_index.long() == expected),
        "group_index must describe contiguous groups in ascending order",
    )
    _assert_condition(torch.all(p_e == p_e.transpose(1, 2)), "p_E must be symmetric")
    _assert_condition(
        torch.all(target_e == target_e.transpose(1, 2)), "r_E must be symmetric"
    )
    graph_groups = group_index.long()
    group_first = counts.cumsum(0) - counts
    values_and_names = (
        (p_x, "p_X"),
        (p_e, "p_E"),
        (node_mask, "node_mask"),
        (p_mask, "p_mask"),
    )
    mismatch_by_field = []
    for value, _name in values_and_names:
        expected_value = value.index_select(0, group_first).index_select(0, graph_groups)
        mismatch_by_field.append((value != expected_value).flatten(1).any(1))
    any_mismatch = torch.stack(mismatch_by_field).any()
    if any_mismatch.device.type == "cuda":
        _assert_condition(~any_mismatch, "product graph or masks differ inside a candidate group")
    elif bool(any_mismatch):
        for mismatch, (_value, name) in zip(mismatch_by_field, values_and_names, strict=True):
            if bool(mismatch.any()):
                candidate = int(torch.nonzero(mismatch, as_tuple=False)[0])
                group_id = int(graph_groups[candidate])
                raise ValueError(f"batch.{name} differs inside group {group_id}")
        raise AssertionError("internal error while locating group-condition mismatch")
    return p_x, p_e, target_x, target_e, node_mask.bool(), p_mask.bool()


def sample_coupled_selector_field(
    *,
    num_groups: int,
    num_nodes: int,
    device: torch.device,
    generator: torch.Generator,
) -> CoupledSelectorField:
    """Sample one group selector field from an explicit device-local generator."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    if isinstance(num_groups, bool) or not isinstance(num_groups, int) or num_groups < 1:
        raise ValueError("num_groups must be a positive integer")
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 1:
        raise ValueError("num_nodes must be a positive integer")
    try:
        node_uniform = torch.rand(
            (num_groups, num_nodes),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        edge_uniform = torch.rand(
            (num_groups, num_nodes, num_nodes),
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    except RuntimeError as error:
        raise ValueError(
            "generator must be device-local to the graph tensors "
            f"(graph device: {device})"
        ) from error
    return CoupledSelectorField(
        node_uniform_group=node_uniform,
        edge_uniform_group=edge_uniform,
    )


def _validate_selector_field(
    selector_field: CoupledSelectorField,
    *,
    num_groups: int,
    num_nodes: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if not isinstance(selector_field, CoupledSelectorField):
        raise TypeError("selector_field must be a CoupledSelectorField")
    node_uniform = selector_field.node_uniform_group
    edge_uniform = selector_field.edge_uniform_group
    if (
        node_uniform.shape != (num_groups, num_nodes)
        or edge_uniform.shape != (num_groups, num_nodes, num_nodes)
        or node_uniform.dtype != torch.float32
        or edge_uniform.dtype != torch.float32
        or node_uniform.device != device
        or edge_uniform.device != device
    ):
        raise ValueError("selector_field shape/dtype/device differs from graph batch")
    _assert_condition(
        torch.isfinite(node_uniform).all()
        & torch.isfinite(edge_uniform).all()
        & (node_uniform >= 0.0).all()
        & (node_uniform < 1.0).all()
        & (edge_uniform >= 0.0).all()
        & (edge_uniform < 1.0).all(),
        "selector_field uniforms must be finite in [0,1)",
    )
    return node_uniform, edge_uniform


def build_coupled_predefined_corruption(
    batch: Any,
    group_index: Tensor,
    num_groups: int,
    generator: torch.Generator,
    *,
    t_group: Tensor | None = None,
    selector_field: CoupledSelectorField | None = None,
) -> CorruptedGraphBatch:
    """Build a group-coupled corruption shared by student and reference paths."""

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    p_x, p_e, target_x, target_e, node_mask, p_mask = _validate_inputs(
        batch,
        group_index,
        num_groups,
    )
    device = p_x.device
    batch_size, num_nodes = p_x.shape[:2]
    if t_group is not None:
        if (
            not isinstance(t_group, Tensor)
            or t_group.shape != (num_groups,)
            or not t_group.is_floating_point()
            or t_group.device != device
        ):
            raise ValueError("explicit t_group must be floating-point [num_groups] on the graph device")
        t_group = t_group.to(dtype=torch.float32).clone()
        _assert_condition(
            torch.isfinite(t_group).all() & (t_group >= 0.02).all() & (t_group <= 0.98).all(),
            "explicit t_group must be finite in [0.02,0.98]",
        )
    try:
        if t_group is None:
            t_group = 0.02 + 0.96 * torch.rand(
                num_groups,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
    except RuntimeError as error:
        raise ValueError(
            "generator must be device-local to the graph tensors "
            f"(graph device: {device})"
        ) from error
    if selector_field is None:
        selector_field = sample_coupled_selector_field(
            num_groups=num_groups,
            num_nodes=num_nodes,
            device=device,
            generator=generator,
        )
    node_uniform, edge_uniform = _validate_selector_field(
        selector_field,
        num_groups=num_groups,
        num_nodes=num_nodes,
        device=device,
    )
    node_selector_group = node_uniform < t_group[:, None]
    strict_upper = torch.ones((num_nodes, num_nodes), dtype=torch.bool, device=device).triu(diagonal=1)
    edge_selector_upper = (edge_uniform < t_group[:, None, None]) & strict_upper
    edge_selector_group = edge_selector_upper | edge_selector_upper.transpose(1, 2)
    graph_groups = group_index.long()
    node_selector = node_selector_group.index_select(0, graph_groups)
    edge_selector = edge_selector_group.index_select(0, graph_groups)
    t_x = torch.where(node_selector.unsqueeze(-1), target_x, p_x)
    t_e = torch.where(edge_selector.unsqueeze(-1), target_e, p_e)
    diagonal = torch.arange(num_nodes, device=device)
    t_e = t_e.clone()
    t_e[:, diagonal, diagonal, :] = 0
    t_e[:, diagonal, diagonal, 0] = 1
    symmetric = torch.all(t_e == t_e.transpose(1, 2))
    if t_e.device.type == "cuda":
        torch._assert_async(symmetric, "internal error: coupled edge corruption is not symmetric")
    elif not bool(symmetric):
        raise AssertionError("internal error: coupled edge corruption is not symmetric")
    t_graph = t_group.index_select(0, graph_groups)
    if t_graph.shape != (batch_size,):
        raise AssertionError("internal error: graph time expansion has the wrong shape")
    return CorruptedGraphBatch(
        t_group=t_group,
        t_graph=t_graph,
        t_X=t_x,
        t_E=t_e,
        target_X=target_x,
        target_E=target_e,
        p_X=p_x,
        p_E=p_e,
        node_mask=node_mask,
        p_mask=p_mask,
    )


__all__ = [
    "CorruptedGraphBatch",
    "CoupledSelectorField",
    "build_coupled_predefined_corruption",
    "sample_coupled_selector_field",
]
