"""Positional, time, and rotary embeddings used by the GraphDiT backbone."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


try:
    from flash_attn.layers.rotary import apply_rotary_emb_qkv_
except ImportError:  # pragma: no cover - depends on the deployment image
    apply_rotary_emb_qkv_ = None


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, n_dim: int) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embedding = nn.Embedding(max_seq_len, n_dim)

    def forward(self, value: Tensor) -> Tensor:
        batch_size, sequence_length = value.shape[:2]
        if sequence_length > self.max_seq_len:
            raise ValueError("sequence length exceeds configured positional embedding")
        positions = torch.arange(sequence_length, device=value.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        return value + self.pos_embedding(positions)


class TimestepEmbedder(nn.Module):
    """Embed scalar diffusion times using the selected GraphDiT formulation."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        arguments = time[..., None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        frequency_embedding = self.timestep_embedding(time, self.frequency_embedding_size)
        frequency_embedding = frequency_embedding.to(dtype=self.mlp[0].weight.dtype)
        return self.mlp(frequency_embedding)


class Rotary(nn.Module):
    """Cache the DDiT rotary embeddings by sequence length."""

    def __init__(self, dim: int, base: int = 10000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached: int | None = None
        self.cos_cached: Tensor | None = None
        self.sin_cached: Tensor | None = None

    def forward(self, value: Tensor, seq_dim: int = 1) -> tuple[Tensor, Tensor]:
        sequence_length = value.shape[seq_dim]
        cached_inference_tensor = (
            (self.cos_cached is not None and self.cos_cached.is_inference())
            or (self.sin_cached is not None and self.sin_cached.is_inference())
        )
        if sequence_length != self.seq_len_cached or cached_inference_tensor:
            self.seq_len_cached = sequence_length
            # Rollout replay invokes the student under inference mode before
            # the differentiable train forward.  Cache construction must use
            # normal tensors so an equal-length subsequent forward can pass
            # the rotary values through activation checkpointing.
            with torch.inference_mode(False), torch.no_grad():
                positions = torch.arange(sequence_length, device=value.device).type_as(self.inv_freq)
                frequencies = torch.outer(positions, self.inv_freq)
                embedding = torch.cat((frequencies, frequencies), dim=-1).to(value.device)
                self.cos_cached = embedding.cos()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
                self.sin_cached = embedding.sin()[None, :, None, None, :].repeat(1, 1, 3, 1, 1)
                self.cos_cached[:, :, 2, :, :].fill_(1.0)
                self.sin_cached[:, :, 2, :, :].fill_(0.0)
        assert self.cos_cached is not None and self.sin_cached is not None
        return self.cos_cached, self.sin_cached


def apply_rotary_pos_emb(qkv: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply the same query/key-only rotary transform as the V4 backbone."""

    cos_half = cos[0, :, 0, 0, : cos.shape[-1] // 2]
    sin_half = sin[0, :, 0, 0, : sin.shape[-1] // 2]
    if apply_rotary_emb_qkv_ is not None and qkv.is_cuda:
        return apply_rotary_emb_qkv_(qkv, cos_half, sin_half)
    result = qkv.clone()
    rotary_dim = cos_half.size(-1) * 2
    cos_full = torch.cat((cos_half, cos_half), dim=-1)[None, :, None, :]
    sin_full = torch.cat((sin_half, sin_half), dim=-1)[None, :, None, :]
    for qk_index in (0, 1):
        value = qkv[:, :, qk_index, :, :rotary_dim]
        half = rotary_dim // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        result[:, :, qk_index, :, :rotary_dim] = (
            value * cos_full.to(value) + rotated * sin_full.to(value)
        )
    return result


__all__ = ["PositionalEmbedding", "Rotary", "TimestepEmbedder", "apply_rotary_pos_emb"]
