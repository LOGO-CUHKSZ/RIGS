"""The dense diffusion transformer block used by GraphDiT."""

from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn

from .embeddings import apply_rotary_pos_emb


class DDiTBlock(nn.Module):
    """V4 DDiT block retained with its original tensor equations."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        rotary_embedding: bool = True,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = node_dim // n_heads
        self.attn_qkv = nn.Linear(node_dim, 3 * node_dim, bias=True)
        self.att_out_dense = nn.Linear(node_dim, node_dim, bias=True)
        self.x_emb_to_e = nn.Linear(node_dim, edge_dim)
        self.e_emb_to_x_add = nn.Linear(edge_dim, node_dim)
        self.e_emb_to_x_mul = nn.Linear(edge_dim, node_dim)
        self.x_att_input_norm = nn.LayerNorm(node_dim)
        self.x_mlp_norm = nn.LayerNorm(node_dim)
        self.x_intermediate_mlp = nn.Sequential(
            nn.Linear(node_dim, mlp_ratio * node_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_ratio * node_dim, node_dim, bias=True),
        )
        self.e_att_input_norm = nn.LayerNorm(edge_dim)
        self.e_mlp_norm = nn.LayerNorm(edge_dim)
        self.e_intermediate_mlp = nn.Sequential(
            nn.Linear(edge_dim, mlp_ratio * edge_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_ratio * edge_dim, edge_dim, bias=True),
        )
        self.dropout = dropout
        self.rotary_embedding = rotary_embedding

    def forward(
        self,
        hidden_states: Tensor,
        adjacency: Tensor,
        attention_mask: Tensor | None = None,
        rotary_cos_sin: tuple[Tensor, Tensor] | None = None,
        symmetric_edges: bool = False,
    ) -> tuple[Tensor, Tensor]:
        if attention_mask is None:
            raise ValueError("DDiTBlock requires an attention mask")
        node_mask = attention_mask.unsqueeze(-1)
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        node_skip = hidden_states
        qkv = self.attn_qkv(self.x_att_input_norm(hidden_states))
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.n_heads)
        if self.rotary_embedding:
            if rotary_cos_sin is None:
                raise ValueError("rotary embedding is enabled but no rotary values were supplied")
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        query, key, value = qkv.unbind(dim=2)
        query = query / (query.shape[-1] ** 0.5)
        edge_attention = torch.einsum("bihd,bjhd->bijhd", query, key)
        edge_attention = edge_attention.flatten(3)
        edge_to_node_add = self.e_emb_to_x_add(adjacency) * edge_mask
        edge_to_node_add = edge_to_node_add.view(edge_attention.shape)
        edge_to_node_mul = self.e_emb_to_x_mul(adjacency) * edge_mask
        edge_to_node_mul = edge_to_node_mul.view(edge_attention.shape)
        edge_attention = edge_attention * (1 + edge_to_node_mul) + edge_to_node_add
        adjacency = adjacency + self.x_emb_to_e(edge_attention)
        if symmetric_edges:
            adjacency = 0.5 * (adjacency + adjacency.transpose(1, 2))
        edge_attention = edge_attention.view(*edge_attention.shape[:3], self.n_heads, -1)
        scores = edge_attention.sum(-1).permute(0, 3, 1, 2)
        mask = attention_mask[:, None, None, :].float()
        scores = scores + (1.0 - mask) * torch.finfo(scores.dtype).min
        attention_weights = torch.softmax(scores, dim=-1)
        node_value = torch.einsum("bhqk,bkhd->bqhd", attention_weights, value)
        node_value = rearrange(node_value, "b s h d -> b s (h d)")
        hidden_states = self.att_out_dense(node_value) + node_skip
        hidden_states = self.x_intermediate_mlp(self.x_mlp_norm(hidden_states)) + hidden_states
        adjacency = self.e_intermediate_mlp(self.e_mlp_norm(adjacency)) + adjacency
        adjacency = adjacency * edge_mask
        return hidden_states, adjacency


__all__ = ["DDiTBlock"]
