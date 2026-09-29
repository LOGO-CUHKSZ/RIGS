"""Dataset-independent GraphDiT dense-graph diffusion backbone."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .blocks import DDiTBlock
from .config import GraphDiTConfig
from .embeddings import PositionalEmbedding, Rotary, TimestepEmbedder


class GraphDiT(nn.Module):
    """The V4 GraphDiT core with explicit tensor dimensions.

    The class keeps the original V4 parameter names so a compatibility loader
    can initialize it from an existing RetroDiT checkpoint.  It deliberately
    receives prepared categorical graph tensors rather than resolving a
    dataset-specific vocabulary at model construction time.
    """

    def __init__(self, config: GraphDiTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_layers = config.n_layers
        self.n_dummy = config.n_dummy
        self.max_n_len = config.max_n_len
        self.symmetry_E = config.symmetry_edges
        self.fix_product_nodes = config.fix_product_nodes
        self.node_input = nn.Linear(config.node_input_dim, config.x_dim)
        self.edge_input = nn.Linear(config.edge_input_dim, config.e_dim)
        self.t_embed = TimestepEmbedder(config.x_dim)
        self.y_embed = nn.Linear(config.condition_dim, config.x_dim)
        self.rotary_emb = Rotary(config.x_dim // config.n_heads)
        self.pos_emb = (
            PositionalEmbedding(config.max_n_len + config.n_dummy, config.node_input_dim)
            if config.positional_embedding
            else None
        )
        self.layers = nn.ModuleList(
            [
                DDiTBlock(
                    node_dim=config.x_dim,
                    edge_dim=config.e_dim,
                    n_heads=config.n_heads,
                    dropout=config.dropout,
                    rotary_embedding=config.rotary_embedding,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.out_norm_node = nn.LayerNorm(config.x_dim)
        self.out_norm_edge = nn.LayerNorm(config.e_dim)
        self.output_proj_node = nn.Linear(config.x_dim, config.node_classes)
        self.output_proj_edge = nn.Linear(config.e_dim, config.edge_classes)

    def node_condition_residual(
        self,
        hidden_x: Tensor,
        **kwargs: object,
    ) -> Tensor | None:
        """Extension point for guidance adapters; base GraphDiT returns none."""

        del hidden_x, kwargs
        return None

    def pre_block_hook(
        self,
        block_index: int,
        hidden_x: Tensor,
        hidden_e: Tensor,
        attention_mask: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        """Identity hook retained for the V4 guidance-adapter insertion point."""

        del block_index, attention_mask, kwargs
        return hidden_x, hidden_e

    def model_forward(
        self,
        node_features: Tensor,
        adjacency: Tensor,
        time: Tensor,
        condition: Tensor,
        attention_mask: Tensor,
        **kwargs: object,
    ) -> dict[str, Tensor]:
        """Evaluate the original dense GraphDiT path after input preparation."""

        self._validate_model_inputs(
            node_features=node_features,
            adjacency=adjacency,
            time=time,
            condition=condition,
            attention_mask=attention_mask,
        )
        if self.symmetry_E:
            adjacency = 0.5 * (adjacency + adjacency.transpose(-3, -2))
        condition_embedding = self.t_embed(time) + self.y_embed(condition)
        hidden_x = self.node_input(node_features)
        hidden_e = self.edge_input(adjacency)
        residual = self.node_condition_residual(hidden_x=hidden_x, **kwargs)
        if residual is not None:
            expected_shape = (hidden_x.size(0), hidden_x.size(-1))
            if residual.shape != expected_shape:
                raise ValueError(
                    f"node condition residual has shape {tuple(residual.shape)}; "
                    f"expected {expected_shape}"
                )
            condition_embedding = condition_embedding + residual
        hidden_x = hidden_x + condition_embedding.unsqueeze(1)
        rotary_cos_sin = self.rotary_emb(hidden_x)
        for block_index, layer in enumerate(self.layers):
            if self.config.activation_checkpointing and self.training and torch.is_grad_enabled():
                # Pass the tensor inputs explicitly so the non-reentrant
                # checkpoint can reconstruct the dense edge attention without
                # retaining each block's intermediate activations.  Adapter
                # context is captured by the closure; guided students bind it
                # through explicit kwargs that remain alive until backward.
                def run_block(
                    block_x: Tensor,
                    block_e: Tensor,
                    block_mask: Tensor,
                    rotary_cos: Tensor,
                    rotary_sin: Tensor,
                    *,
                    _block_index: int = block_index,
                    _layer: nn.Module = layer,
                    _hook_kwargs: dict[str, object] = kwargs,
                ) -> tuple[Tensor, Tensor]:
                    hooked_x, hooked_e = self.pre_block_hook(
                        _block_index,
                        block_x,
                        block_e,
                        block_mask,
                        **_hook_kwargs,
                    )
                    return _layer(
                        hooked_x,
                        hooked_e,
                        block_mask,
                        (rotary_cos, rotary_sin),
                        symmetric_edges=self.symmetry_E,
                    )

                hidden_x, hidden_e = activation_checkpoint(
                    run_block,
                    hidden_x,
                    hidden_e,
                    attention_mask,
                    rotary_cos_sin[0],
                    rotary_cos_sin[1],
                    use_reentrant=False,
                )
            else:
                hidden_x, hidden_e = self.pre_block_hook(
                    block_index,
                    hidden_x,
                    hidden_e,
                    attention_mask,
                    **kwargs,
                )
                hidden_x, hidden_e = layer(
                    hidden_x,
                    hidden_e,
                    attention_mask,
                    rotary_cos_sin,
                    symmetric_edges=self.symmetry_E,
                )
        node_embeddings = self.output_proj_node(hidden_x)
        node_mask = attention_mask.unsqueeze(-1)
        node_embeddings = node_embeddings * node_mask
        edge_embeddings = self.output_proj_edge(hidden_e)
        if self.symmetry_E:
            edge_embeddings = 0.5 * (
                edge_embeddings + edge_embeddings.transpose(-3, -2)
            )
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        edge_embeddings = edge_embeddings * edge_mask
        return {"X": node_embeddings, "E": edge_embeddings}

    def forward(
        self,
        *,
        t: Tensor,
        node_mask: Tensor,
        cond: Tensor,
        t_X: Tensor,
        t_E: Tensor,
        p_X: Tensor,
        p_E: Tensor,
        p_mask: Tensor,
        **kwargs: object,
    ) -> dict[str, Tensor]:
        """Run the categorical diffusion model on product/route graph tensors."""

        input_x = torch.cat((p_X, t_X), dim=-1)
        input_e = torch.cat((p_E, t_E), dim=-1)
        if self.pos_emb is not None:
            input_x = self.pos_emb(input_x)
        prediction = self.model_forward(
            input_x,
            input_e,
            t,
            cond,
            node_mask,
            **kwargs,
        )
        if self.fix_product_nodes:
            prediction["X"] = self.fix_nodes(
                prediction["X"], node_mask, p_X, p_mask
            )
        return prediction

    @staticmethod
    def fix_nodes(
        prediction: Tensor,
        node_mask: Tensor,
        product_nodes: Tensor,
        product_mask: Tensor,
    ) -> Tensor:
        mask = product_mask.unsqueeze(-1)
        prediction = prediction * (~mask) + product_nodes * mask
        return prediction * node_mask.unsqueeze(-1)

    def _validate_model_inputs(
        self,
        *,
        node_features: Tensor,
        adjacency: Tensor,
        time: Tensor,
        condition: Tensor,
        attention_mask: Tensor,
    ) -> None:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [B,N,F]")
        batch_size, node_count, node_dim = node_features.shape
        if node_dim != self.config.node_input_dim:
            raise ValueError(
                f"node_features last dimension must be {self.config.node_input_dim}"
            )
        if adjacency.shape != (
            batch_size,
            node_count,
            node_count,
            self.config.edge_input_dim,
        ):
            raise ValueError(
                "adjacency must have shape [B,N,N,edge_input_dim] matching node features"
            )
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [B]")
        if condition.shape != (batch_size, self.config.condition_dim):
            raise ValueError("condition has an incompatible shape")
        if attention_mask.shape != (batch_size, node_count):
            raise ValueError("attention_mask must have shape [B,N]")


__all__ = ["GraphDiT"]
