"""Explicit, dataset-independent GraphDiT backbone dimensions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphDiTConfig:
    """Architecture contract for a categorical dense-graph diffusion backbone.

    ``node_classes`` and ``edge_classes`` describe the already prepared graph
    representation, not a named dataset or a vocabulary file.  This keeps the
    model reusable for any adapter that emits the same tensor contract.
    """

    node_classes: int
    edge_classes: int
    condition_dim: int
    x_dim: int
    e_dim: int
    n_layers: int
    n_heads: int
    dropout: float = 0.0
    n_dummy: int = 0
    max_n_len: int = 100
    symmetry_edges: bool = True
    rotary_embedding: bool = True
    positional_embedding: bool = False
    fix_product_nodes: bool = False
    activation_checkpointing: bool = False

    def __post_init__(self) -> None:
        for name in (
            "node_classes",
            "edge_classes",
            "x_dim",
            "e_dim",
            "n_layers",
            "n_heads",
            "max_n_len",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.condition_dim, bool)
            or not isinstance(self.condition_dim, int)
            or self.condition_dim < 0
        ):
            raise ValueError("condition_dim must be a non-negative integer")
        if (
            isinstance(self.n_dummy, bool)
            or not isinstance(self.n_dummy, int)
            or self.n_dummy < 0
        ):
            raise ValueError("n_dummy must be a non-negative integer")
        if not isinstance(self.dropout, (int, float)) or not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if self.x_dim % self.n_heads:
            raise ValueError("x_dim must be divisible by n_heads")
        if self.rotary_embedding and (self.x_dim // self.n_heads) % 2:
            raise ValueError("rotary embedding requires an even attention head dimension")
        if not isinstance(self.activation_checkpointing, bool):
            raise TypeError("activation_checkpointing must be a bool")

    @property
    def node_input_dim(self) -> int:
        return self.node_classes * 2

    @property
    def edge_input_dim(self) -> int:
        return self.edge_classes * 2

    def as_dict(self) -> dict[str, object]:
        return {
            "node_classes": self.node_classes,
            "edge_classes": self.edge_classes,
            "condition_dim": self.condition_dim,
            "x_dim": self.x_dim,
            "e_dim": self.e_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": float(self.dropout),
            "n_dummy": self.n_dummy,
            "max_n_len": self.max_n_len,
            "symmetry_edges": self.symmetry_edges,
            "rotary_embedding": self.rotary_embedding,
            "positional_embedding": self.positional_embedding,
            "fix_product_nodes": self.fix_product_nodes,
            "activation_checkpointing": self.activation_checkpointing,
        }


__all__ = ["GraphDiTConfig"]
