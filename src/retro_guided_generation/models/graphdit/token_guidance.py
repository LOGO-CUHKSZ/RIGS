"""Token-level guidance adapters for the dataset-independent GraphDiT core."""

from __future__ import annotations

import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..guidance.architecture import (
    DEFAULT_GUIDANCE_ARCHITECTURE_ID,
    NATIVE_KV_MODE,
    GuidanceArchitectureSelector,
    GuidanceArchitectureSpec,
    resolve_guidance_architecture,
)
from .core import GraphDiT


TEXT_DIM = 4096
HIDDEN_DIM = 256
INJECTION_LAYERS = (2, 4, 6, 8)
INJECTION_BLOCKS = tuple(layer - 1 for layer in INJECTION_LAYERS)
NativeKVMemory = tuple[Tensor, Tensor]
CandidateMemory = Tensor | NativeKVMemory
GuidanceResidualScale = float | Tensor


def _assert_condition(condition: Tensor, message: str) -> None:
    if condition.numel() != 1:
        raise ValueError("runtime validation condition must be scalar")
    if condition.device.type == "cuda":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def _fp32_layer_norm(value: Tensor, module: nn.LayerNorm) -> Tensor:
    """Evaluate LayerNorm in float32 without changing the public tensor dtype."""

    with torch.autocast(device_type=value.device.type, enabled=False):
        return F.layer_norm(
            value.float(),
            module.normalized_shape,
            module.weight.float(),
            module.bias.float(),
            module.eps,
        )


def _normalize_guidance_residual_scale(
    value: GuidanceResidualScale,
    *,
    node_mask: Tensor,
) -> GuidanceResidualScale:
    """Validate a scalar or per-candidate residual attenuation in ``[0, 1]``."""

    if isinstance(value, bool):
        raise TypeError("guidance_residual_scale must be a float or floating Tensor")
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise TypeError("guidance_residual_scale Tensor must be floating point")
        if value.device != node_mask.device:
            raise ValueError("guidance_residual_scale Tensor must be on the graph device")
        if value.ndim == 0:
            normalized = value
        elif value.shape == (int(node_mask.shape[0]),):
            normalized = value
        else:
            raise ValueError(
                "guidance_residual_scale Tensor must be scalar or have shape [candidates]"
            )
        _assert_condition(
            torch.isfinite(normalized).all()
            & torch.all(normalized >= 0.0)
            & torch.all(normalized <= 1.0),
            "guidance_residual_scale must be finite and lie in [0, 1]",
        )
        return normalized
    if not isinstance(value, (int, float)):
        raise TypeError("guidance_residual_scale must be a float or floating Tensor")
    normalized_float = float(value)
    if not math.isfinite(normalized_float) or not 0.0 <= normalized_float <= 1.0:
        raise ValueError("guidance_residual_scale must be finite and lie in [0, 1]")
    return normalized_float


def _scale_guidance_delta(
    delta: Tensor,
    residual_scale: GuidanceResidualScale,
) -> Tensor:
    """Scale only the guidance residual, preserving the backbone hidden state."""

    if isinstance(residual_scale, Tensor):
        scale = residual_scale.to(dtype=delta.dtype)
        if scale.ndim == 1:
            if scale.shape != (int(delta.shape[0]),):
                raise ValueError(
                    "per-candidate guidance_residual_scale differs from the hidden batch"
                )
            scale = scale[:, None, None]
        return delta * scale
    if float(residual_scale) == 1.0:
        return delta
    return delta * float(residual_scale)


class NodeTextCrossAttention(nn.Module):
    """Zero-initialized node-to-projected-text residual attention."""

    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = 4,
        condition_dim: int | None = None,
    ) -> None:
        super().__init__()
        if condition_dim is None:
            condition_dim = hidden_dim
        if isinstance(condition_dim, bool) or not isinstance(condition_dim, int) or condition_dim < 1:
            raise ValueError("condition_dim must be a positive integer")
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
            kdim=condition_dim,
            vdim=condition_dim,
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def _forward_with_optional_attention(
        self,
        hidden_x: Tensor,
        memory: Tensor,
        text_mask: Tensor,
        node_mask: Tensor,
        *,
        need_weights: bool,
        residual_scale: GuidanceResidualScale = 1.0,
    ) -> tuple[Tensor, Tensor | None]:
        if memory.ndim != 3 or memory.shape[-1] != self.condition_dim:
            observed = memory.shape[-1] if memory.ndim else "scalar"
            raise ValueError(
                "cross-attention memory width differs from condition_dim: "
                f"{observed} != {self.condition_dim}"
            )
        query_weight = (
            self.attn.in_proj_weight
            if self.attn.in_proj_weight is not None
            else self.attn.q_proj_weight
        )
        if query_weight is None:  # pragma: no cover - guaranteed by MultiheadAttention
            raise RuntimeError("cross-attention has no query projection weight")
        query = _fp32_layer_norm(hidden_x, self.query_norm).to(dtype=query_weight.dtype)
        delta, attention = self.attn(
            query=query,
            key=memory,
            value=memory,
            key_padding_mask=~text_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        scaled_delta = _scale_guidance_delta(
            delta.to(hidden_x.dtype), residual_scale
        )
        result = hidden_x + scaled_delta * node_mask.unsqueeze(-1)
        if attention is not None:
            attention = attention * node_mask[:, None, :, None]
        return result, attention

    def forward(
        self,
        hidden_x: Tensor,
        memory: Tensor,
        text_mask: Tensor,
        node_mask: Tensor,
        *,
        residual_scale: GuidanceResidualScale = 1.0,
    ) -> Tensor:
        result, _ = self._forward_with_optional_attention(
            hidden_x,
            memory,
            text_mask,
            node_mask,
            need_weights=False,
            residual_scale=residual_scale,
        )
        return result

    def forward_with_attention(
        self,
        hidden_x: Tensor,
        memory: Tensor,
        text_mask: Tensor,
        node_mask: Tensor,
        *,
        residual_scale: GuidanceResidualScale = 1.0,
    ) -> tuple[Tensor, Tensor]:
        result, attention = self._forward_with_optional_attention(
            hidden_x,
            memory,
            text_mask,
            node_mask,
            need_weights=True,
            residual_scale=residual_scale,
        )
        if attention is None:
            raise AssertionError("attention weights were requested but absent")
        return result, attention


class AsymmetricNativeKVMultiheadAttention(nn.Module):
    """Node queries with text-native key/value projections."""

    def __init__(self, *, text_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(text_dim, hidden_dim)
        self.v_proj = nn.Linear(text_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _split_heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        projected_key: Tensor,
        projected_value: Tensor,
        text_mask: Tensor,
    ) -> Tensor:
        q = self._split_heads(self.q_proj(query))
        key = self._split_heads(projected_key)
        value = self._split_heads(projected_value)
        scores = torch.matmul(q.float(), key.float().transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~text_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value.float()).to(dtype=q.dtype)
        context = context.transpose(1, 2).flatten(start_dim=2)
        return self.out_proj(context)

    def project_memory(self, memory: Tensor) -> NativeKVMemory:
        return self.k_proj(memory), self.v_proj(memory)


class NativeKVNodeTextCrossAttention(nn.Module):
    """Block-local cross attention that accepts native-width text tokens."""

    def __init__(self, *, text_dim: int = TEXT_DIM, hidden_dim: int = HIDDEN_DIM, num_heads: int = 4) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(text_dim)
        self.attn = AsymmetricNativeKVMultiheadAttention(
            text_dim=text_dim, hidden_dim=hidden_dim, num_heads=num_heads
        )

    def forward(
        self,
        hidden_x: Tensor,
        memory: NativeKVMemory,
        text_mask: Tensor,
        node_mask: Tensor,
        *,
        residual_scale: GuidanceResidualScale = 1.0,
    ) -> Tensor:
        if not isinstance(memory, tuple) or len(memory) != 2 or any(not isinstance(value, Tensor) for value in memory):
            raise TypeError("native K/V memory must be a (key, value) Tensor tuple")
        projected_key, projected_value = memory
        query = _fp32_layer_norm(hidden_x, self.query_norm).to(dtype=self.attn.q_proj.weight.dtype)
        delta = self.attn(query, projected_key, projected_value, text_mask)
        scaled_delta = _scale_guidance_delta(
            delta.to(hidden_x.dtype), residual_scale
        )
        return hidden_x + scaled_delta * node_mask.unsqueeze(-1)

    def project_group_memory(self, tokens: Tensor) -> NativeKVMemory:
        memory = _fp32_layer_norm(tokens, self.memory_norm).to(dtype=self.attn.k_proj.weight.dtype)
        return self.attn.project_memory(memory)


class SemanticConditionAdapter(nn.Module):
    """Map native route semantics into generator-specific conditioning width.

    ``semantic_dim`` is the native frozen-teacher coordinate used by Stage A.
    ``condition_dim`` is the generator adapter width ``k`` consumed by cross
    attention.  Keeping this bridge explicit avoids padding a smaller semantic
    teacher with zeros merely to satisfy the generation model's API.
    Equal-width configurations retain an exact identity path for backward
    compatibility; unequal widths use a trainable orthogonally initialized
    projection.
    """

    def __init__(self, semantic_dim: int, condition_dim: int) -> None:
        super().__init__()
        for name, value in (("semantic_dim", semantic_dim), ("condition_dim", condition_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.semantic_dim = semantic_dim
        self.condition_dim = condition_dim
        if semantic_dim == condition_dim:
            self.projection: nn.Module = nn.Identity()
        else:
            projection = nn.Linear(semantic_dim, condition_dim)
            nn.init.orthogonal_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.projection = projection

    @property
    def is_identity(self) -> bool:
        return isinstance(self.projection, nn.Identity)

    def forward(self, semantic_memory: Tensor) -> Tensor:
        if semantic_memory.shape[-1] != self.semantic_dim:
            raise ValueError(
                "semantic memory width differs from the condition adapter: "
                f"{semantic_memory.shape[-1]} != {self.semantic_dim}"
            )
        return self.projection(semantic_memory)


class _TokenGuidanceAdapterBase(nn.Module):
    """Common validation and memory-binding API for registered adapters."""

    def __init__(
        self,
        *,
        text_dim: int,
        semantic_dim: int,
        condition_dim: int,
        hidden_dim: int,
        architecture_spec: GuidanceArchitectureSpec,
        learned_null_guidance: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(text_dim, bool) or not isinstance(text_dim, int) or text_dim < 1:
            raise ValueError("text_dim must be a positive integer")
        if isinstance(semantic_dim, bool) or not isinstance(semantic_dim, int) or semantic_dim < 1:
            raise ValueError("semantic_dim must be a positive integer")
        if isinstance(condition_dim, bool) or not isinstance(condition_dim, int) or condition_dim < 1:
            raise ValueError("condition_dim must be a positive integer")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError("hidden_dim must be a positive integer")
        if hidden_dim % architecture_spec.num_heads:
            raise ValueError("hidden_dim must be divisible by the architecture head count")
        if not isinstance(learned_null_guidance, bool):
            raise TypeError("learned_null_guidance must be a bool")
        self.text_dim = text_dim
        self.semantic_dim = semantic_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.architecture_spec = architecture_spec
        self.architecture_id = architecture_spec.architecture_id
        self.injection_layers = architecture_spec.injection_layers
        self.injection_blocks = architecture_spec.injection_blocks
        self.learned_null_guidance = learned_null_guidance

    def validate_group_index(
        self,
        group_index: Tensor | None,
        num_candidates: int,
        *,
        num_groups: int | None = None,
    ) -> Tensor:
        if group_index is None:
            raise ValueError("guidance_group_index is required")
        if group_index.dtype != torch.int64 or group_index.ndim != 1:
            raise TypeError("guidance_group_index must be one-dimensional int64")
        if group_index.numel() != num_candidates:
            raise ValueError("guidance_group_index length must equal candidate count")
        group_count = int(group_index.max().item()) + 1 if num_groups is None and group_index.numel() else int(num_groups or 0)
        if group_count < 1 and num_candidates:
            raise ValueError("guidance group count must be positive")
        _assert_condition(
            ((group_index >= 0) & (group_index < group_count)).all(),
            "guidance_group_index contains an out-of-range group id",
        )
        return group_index

    def validate(
        self,
        tokens: Tensor | None,
        mask: Tensor | None,
        group_index: Tensor | None,
        num_candidates: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if tokens is None or mask is None or group_index is None:
            raise ValueError("guided student requires tokens, mask, and group index")
        if tokens.ndim != 3 or tokens.shape[-1] != self.text_dim or not tokens.is_floating_point():
            raise ValueError(f"guidance tokens must be floating [G,L,{self.text_dim}]")
        if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
            raise ValueError("guidance token mask must be bool [G,L]")
        if tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("guidance tokens must contain at least one group and token")
        _assert_condition(mask.any(dim=1).all(), "every guidance group needs one valid token")
        _assert_condition(torch.isfinite(tokens.masked_select(mask.unsqueeze(-1))).all(), "valid guidance tokens contain NaN or Inf")
        return tokens, mask, self.validate_group_index(
            group_index, num_candidates, num_groups=int(tokens.shape[0])
        )

    def null_candidate_memories(self, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        raise NotImplementedError

    def candidate_memories(
        self, tokens: Tensor, mask: Tensor, group_index: Tensor
    ) -> tuple[dict[int, CandidateMemory], Tensor]:
        raise NotImplementedError

    def architecture_contract(self) -> dict[str, object]:
        return self.architecture_spec.contract(
            text_dim=self.text_dim,
            semantic_dim=self.semantic_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
        )

    def expected_parameter_count(self) -> int:
        """Return the topology-determined adapter parameter count.

        This excludes the frozen GraphDiT backbone and lets recipes report the
        exact sharing choice before a trainer is constructed.
        """

        if self.architecture_spec.kv_mode == NATIVE_KV_MODE:
            per_block = (
                2 * self.text_dim
                + 2 * self.text_dim * self.hidden_dim
                + 2 * self.hidden_dim**2
                + 6 * self.hidden_dim
            )
            result = len(self.injection_blocks) * per_block
            if self.learned_null_guidance:
                result += len(self.injection_blocks) * self.text_dim
            return result
        projector = 2 * self.text_dim + self.text_dim * self.semantic_dim + self.semantic_dim
        condition_adapter = (
            0
            if self.semantic_dim == self.condition_dim
            else self.semantic_dim * self.condition_dim + self.condition_dim
        )
        cross_attention = (
            2 * self.hidden_dim**2
            + 2 * self.hidden_dim * self.condition_dim
            + 6 * self.hidden_dim
        )
        result = len(self.architecture_spec.projector_groups) * projector
        result += len(self.architecture_spec.projector_groups) * condition_adapter
        result += len(self.injection_blocks) * cross_attention
        if self.learned_null_guidance:
            null_count = 1 if self.architecture_id == DEFAULT_GUIDANCE_ARCHITECTURE_ID else len(self.architecture_spec.projector_groups)
            result += null_count * self.text_dim
        return result

    def iter_trainable_parameter_names(self) -> Iterator[str]:
        yield from (name for name, _ in self.named_parameters())

    def iter_cross_attention_blocks(self) -> Iterator[tuple[int, NodeTextCrossAttention | NativeKVNodeTextCrossAttention]]:
        for block_index in self.injection_blocks:
            yield block_index, self.cross_attention[str(block_index)]


class TokenGuidanceAdapter(_TokenGuidanceAdapterBase):
    """Shared native-text projector with independent layer cross attention."""

    def __init__(
        self,
        *,
        text_dim: int = TEXT_DIM,
        semantic_dim: int = HIDDEN_DIM,
        condition_dim: int = HIDDEN_DIM,
        hidden_dim: int = HIDDEN_DIM,
        injection_blocks: tuple[int, ...] = INJECTION_BLOCKS,
        architecture_spec: GuidanceArchitectureSpec | None = None,
        learned_null_guidance: bool = False,
    ) -> None:
        if architecture_spec is None:
            architecture_spec = resolve_guidance_architecture(DEFAULT_GUIDANCE_ARCHITECTURE_ID)
        if architecture_spec.injection_blocks != injection_blocks:
            raise ValueError("injection_blocks must match the registered architecture")
        super().__init__(
            text_dim=text_dim,
            semantic_dim=semantic_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            architecture_spec=architecture_spec,
            learned_null_guidance=learned_null_guidance,
        )
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_projection = nn.Linear(text_dim, semantic_dim)
        self.condition_adapter = SemanticConditionAdapter(semantic_dim, condition_dim)
        if learned_null_guidance:
            self.null_token = nn.Parameter(torch.zeros(text_dim))
        self.cross_attention = nn.ModuleDict(
            {
                str(index): NodeTextCrossAttention(
                    hidden_dim,
                    architecture_spec.num_heads,
                    condition_dim,
                )
                for index in injection_blocks
            }
        )

    def group_semantic_embeddings(self, tokens: Tensor, mask: Tensor) -> dict[str, Tensor]:
        """Return the one semantic embedding owned by the shared projector.

        The shared topology has one projector group covering all of its
        injection layers.  It therefore uses the same masked-pool-and-normalize
        semantic contract as the multi-projector adapter, without introducing
        another projector or changing candidate-memory construction.
        """

        if tokens.ndim != 3 or tokens.shape[-1] != self.text_dim:
            raise ValueError("tokens must be [G,L,text_dim]")
        if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
            raise ValueError("mask must be bool [G,L]")
        _assert_condition(mask.any(dim=1).all(), "every guidance group needs one valid token")
        keys = self.architecture_spec.projector_group_keys
        if len(keys) != 1:
            raise RuntimeError("shared token guidance requires exactly one projector group")
        valid = mask.unsqueeze(-1)
        normalized = _fp32_layer_norm(tokens, self.text_norm)
        projected = self.text_projection(normalized.to(dtype=self.text_projection.weight.dtype))
        pooled = projected.float().mul(valid).sum(dim=1)
        pooled = pooled / valid.sum(dim=1).clamp_min(1)
        return {keys[0]: F.normalize(pooled, dim=-1, eps=1.0e-8)}

    def candidate_memory(self, tokens: Tensor, mask: Tensor, group_index: Tensor) -> tuple[Tensor, Tensor]:
        normalized = _fp32_layer_norm(tokens, self.text_norm)
        semantic_memory = self.text_projection(
            normalized.to(dtype=self.text_projection.weight.dtype)
        )
        group_memory = self.condition_adapter(semantic_memory)
        return group_memory.index_select(0, group_index), mask.index_select(0, group_index)

    def candidate_memories(self, tokens: Tensor, mask: Tensor, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        memory, candidate_mask = self.candidate_memory(tokens, mask, group_index)
        return {index: memory for index in self.injection_blocks}, candidate_mask

    def null_candidate_memories(self, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        if not self.learned_null_guidance:
            raise ValueError("learned null guidance is disabled")
        num_candidates = int(group_index.numel())
        tokens = self.null_token.view(1, 1, -1).expand(num_candidates, -1, -1)
        normalized = _fp32_layer_norm(tokens, self.text_norm)
        semantic_memory = self.text_projection(
            normalized.to(dtype=self.text_projection.weight.dtype)
        )
        memory = self.condition_adapter(semantic_memory)
        mask = torch.ones((num_candidates, 1), device=group_index.device, dtype=torch.bool)
        return {index: memory for index in self.injection_blocks}, mask


class TokenTextProjector(nn.Module):
    """One independently trainable native-text-to-hidden projector."""

    def __init__(self, text_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = _fp32_layer_norm(tokens, self.text_norm)
        return self.text_projection(normalized.to(dtype=self.text_projection.weight.dtype))


class GroupedTokenGuidanceAdapter(_TokenGuidanceAdapterBase):
    """One independent projector for each declared set of injection layers."""

    def __init__(
        self,
        *,
        architecture_spec: GuidanceArchitectureSpec,
        text_dim: int = TEXT_DIM,
        semantic_dim: int = HIDDEN_DIM,
        condition_dim: int = HIDDEN_DIM,
        hidden_dim: int = HIDDEN_DIM,
        learned_null_guidance: bool = False,
    ) -> None:
        if len(architecture_spec.projector_groups) < 2:
            raise ValueError("grouped adapter requires at least two projector groups")
        super().__init__(
            text_dim=text_dim,
            semantic_dim=semantic_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            architecture_spec=architecture_spec,
            learned_null_guidance=learned_null_guidance,
        )
        self.projector_groups = nn.ModuleDict(
            {
                key: TokenTextProjector(text_dim, semantic_dim)
                for key in architecture_spec.projector_group_keys
            }
        )
        self.condition_adapters = nn.ModuleDict(
            {
                key: SemanticConditionAdapter(semantic_dim, condition_dim)
                for key in architecture_spec.projector_group_keys
            }
        )
        if learned_null_guidance:
            self.null_tokens = nn.ParameterDict(
                {key: nn.Parameter(torch.zeros(text_dim)) for key in architecture_spec.projector_group_keys}
            )
        self.cross_attention = nn.ModuleDict(
            {
                str(index): NodeTextCrossAttention(
                    hidden_dim,
                    architecture_spec.num_heads,
                    condition_dim,
                )
                for index in self.injection_blocks
            }
        )

    def group_semantic_embeddings(self, tokens: Tensor, mask: Tensor) -> dict[str, Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.text_dim:
            raise ValueError("tokens must be [G,L,text_dim]")
        if mask.dtype != torch.bool or mask.shape != tokens.shape[:2]:
            raise ValueError("mask must be bool [G,L]")
        _assert_condition(mask.any(dim=1).all(), "every guidance group needs one valid token")
        valid = mask.unsqueeze(-1)
        result: dict[str, Tensor] = {}
        for key, projector in self.projector_groups.items():
            pooled = projector(tokens).float().mul(valid).sum(dim=1)
            pooled = pooled / valid.sum(dim=1).clamp_min(1)
            result[key] = F.normalize(pooled, dim=-1, eps=1.0e-8)
        return result

    def candidate_memories(self, tokens: Tensor, mask: Tensor, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        memories: dict[int, CandidateMemory] = {}
        for key, block_group in zip(
            self.architecture_spec.projector_group_keys,
            self.architecture_spec.projector_group_blocks,
            strict=True,
        ):
            semantic_memory = self.projector_groups[key](tokens)
            candidate_memory = self.condition_adapters[key](semantic_memory).index_select(
                0, group_index
            )
            for block_index in block_group:
                memories[block_index] = candidate_memory
        return memories, mask.index_select(0, group_index)

    def null_candidate_memories(self, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        if not self.learned_null_guidance:
            raise ValueError("learned null guidance is disabled")
        num_candidates = int(group_index.numel())
        memories: dict[int, CandidateMemory] = {}
        for key, block_group in zip(
            self.architecture_spec.projector_group_keys,
            self.architecture_spec.projector_group_blocks,
            strict=True,
        ):
            token = self.null_tokens[key].view(1, 1, -1)
            semantic_memory = self.projector_groups[key](token)
            memory = self.condition_adapters[key](semantic_memory).expand(
                num_candidates, -1, -1
            )
            for block_index in block_group:
                memories[block_index] = memory
        mask = torch.ones((num_candidates, 1), device=group_index.device, dtype=torch.bool)
        return memories, mask


class NativeKVTokenGuidanceAdapter(_TokenGuidanceAdapterBase):
    """Block-local native-width K/V projections with no shared projector."""

    def __init__(
        self,
        *,
        architecture_spec: GuidanceArchitectureSpec,
        text_dim: int = TEXT_DIM,
        semantic_dim: int = HIDDEN_DIM,
        condition_dim: int = HIDDEN_DIM,
        hidden_dim: int = HIDDEN_DIM,
        learned_null_guidance: bool = False,
    ) -> None:
        if architecture_spec.kv_mode != NATIVE_KV_MODE or architecture_spec.projector_groups:
            raise ValueError("native K/V adapter requires a native registered architecture")
        super().__init__(
            text_dim=text_dim,
            semantic_dim=semantic_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            architecture_spec=architecture_spec,
            learned_null_guidance=learned_null_guidance,
        )
        if learned_null_guidance:
            self.null_tokens = nn.ParameterDict(
                {f"layer_{layer}": nn.Parameter(torch.zeros(text_dim)) for layer in architecture_spec.injection_layers}
            )
        self.cross_attention = nn.ModuleDict(
            {
                str(index): NativeKVNodeTextCrossAttention(
                    text_dim=text_dim, hidden_dim=hidden_dim, num_heads=architecture_spec.num_heads
                )
                for index in self.injection_blocks
            }
        )

    def candidate_memories(self, tokens: Tensor, mask: Tensor, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        memories: dict[int, CandidateMemory] = {}
        for block_index, block in self.iter_cross_attention_blocks():
            key, value = block.project_group_memory(tokens)
            memories[block_index] = (key.index_select(0, group_index), value.index_select(0, group_index))
        return memories, mask.index_select(0, group_index)

    def null_candidate_memories(self, group_index: Tensor) -> tuple[dict[int, CandidateMemory], Tensor]:
        if not self.learned_null_guidance:
            raise ValueError("learned null guidance is disabled")
        num_candidates = int(group_index.numel())
        memories: dict[int, CandidateMemory] = {}
        for block_index, block in self.iter_cross_attention_blocks():
            token = self.null_tokens[f"layer_{block_index + 1}"].view(1, 1, -1)
            key, value = block.project_group_memory(token)
            memories[block_index] = (
                key.expand(num_candidates, -1, -1),
                value.expand(num_candidates, -1, -1),
            )
        mask = torch.ones((num_candidates, 1), device=group_index.device, dtype=torch.bool)
        return memories, mask


def build_token_guidance_adapter(
    architecture: GuidanceArchitectureSelector = None,
    *,
    text_dim: int = TEXT_DIM,
    semantic_dim: int | None = None,
    condition_dim: int | None = None,
    hidden_dim: int = HIDDEN_DIM,
    learned_null_guidance: bool = False,
) -> _TokenGuidanceAdapterBase:
    """Build the registered adapter requested by a public topology selector."""

    spec = resolve_guidance_architecture(architecture)
    resolved_semantic_dim = hidden_dim if semantic_dim is None else semantic_dim
    resolved_condition_dim = hidden_dim if condition_dim is None else condition_dim
    if spec.architecture_id == DEFAULT_GUIDANCE_ARCHITECTURE_ID:
        return TokenGuidanceAdapter(
            text_dim=text_dim,
            semantic_dim=resolved_semantic_dim,
            condition_dim=resolved_condition_dim,
            hidden_dim=hidden_dim,
            injection_blocks=spec.injection_blocks,
            architecture_spec=spec,
            learned_null_guidance=learned_null_guidance,
        )
    if spec.kv_mode == NATIVE_KV_MODE:
        return NativeKVTokenGuidanceAdapter(
            architecture_spec=spec,
            text_dim=text_dim,
            semantic_dim=resolved_semantic_dim,
            condition_dim=resolved_condition_dim,
            hidden_dim=hidden_dim,
            learned_null_guidance=learned_null_guidance,
        )
    return GroupedTokenGuidanceAdapter(
        architecture_spec=spec,
        text_dim=text_dim,
        semantic_dim=resolved_semantic_dim,
        condition_dim=resolved_condition_dim,
        hidden_dim=hidden_dim,
        learned_null_guidance=learned_null_guidance,
    )


class TokenCrossAttentionGraphDiT(GraphDiT):
    """GraphDiT student that requires text guidance at each registered injection."""

    def __init__(
        self,
        *args: object,
        text_dim: int = TEXT_DIM,
        semantic_dim: int | None = None,
        condition_dim: int | None = None,
        architecture: GuidanceArchitectureSelector = None,
        injection_layers: tuple[int, ...] | None = None,
        learned_null_guidance: bool = False,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)
        architecture_spec = resolve_guidance_architecture(architecture)
        if max(architecture_spec.injection_layers) > self.n_layers:
            raise ValueError("guidance injection layers exceed the GraphDiT depth")
        if injection_layers is not None and injection_layers != architecture_spec.injection_layers:
            raise ValueError("injection_layers must match the registered architecture")
        self.architecture_spec = architecture_spec
        self.architecture_id = architecture_spec.architecture_id
        self.injection_layers = architecture_spec.injection_layers
        self.injection_blocks = architecture_spec.injection_blocks
        self.text_dim = text_dim
        self.hidden_dim = self.node_input.out_features
        self.semantic_dim = self.hidden_dim if semantic_dim is None else semantic_dim
        self.condition_dim = self.hidden_dim if condition_dim is None else condition_dim
        self.guidance_adapter = build_token_guidance_adapter(
            architecture_spec,
            text_dim=text_dim,
            semantic_dim=self.semantic_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
            learned_null_guidance=learned_null_guidance,
        )
        self._candidate_memory_by_block: dict[int, CandidateMemory] | None = None
        self._candidate_text_mask: Tensor | None = None
        self._candidate_residual_scale: GuidanceResidualScale = 1.0

    def architecture_contract(self) -> dict[str, object]:
        return self.architecture_spec.contract(
            text_dim=self.text_dim,
            semantic_dim=self.semantic_dim,
            condition_dim=self.condition_dim,
            hidden_dim=self.hidden_dim,
        )

    def iter_cross_attention_blocks(self) -> Iterator[tuple[int, NodeTextCrossAttention | NativeKVNodeTextCrossAttention]]:
        yield from self.guidance_adapter.iter_cross_attention_blocks()

    def forward(
        self,
        *args: object,
        guidance_tokens_group: Tensor | None = None,
        guidance_token_mask_group: Tensor | None = None,
        guidance_group_index: Tensor | None = None,
        guidance_null: bool = False,
        guidance_residual_scale: GuidanceResidualScale = 1.0,
        **kwargs: object,
    ) -> dict[str, Tensor]:
        del args
        node_mask = kwargs.get("node_mask")
        if not isinstance(node_mask, Tensor):
            raise ValueError("node_mask is required for guided GraphDiT")
        if not isinstance(guidance_null, bool):
            raise TypeError("guidance_null must be a bool")
        residual_scale = _normalize_guidance_residual_scale(
            guidance_residual_scale,
            node_mask=node_mask,
        )
        if guidance_null:
            if guidance_tokens_group is not None or guidance_token_mask_group is not None:
                raise ValueError("guidance_null cannot receive explicit guidance tokens")
            if not self.guidance_adapter.learned_null_guidance:
                raise ValueError("guidance_null requires learned_null_guidance=True")
            index = self.guidance_adapter.validate_group_index(
                guidance_group_index, int(node_mask.shape[0])
            )
            memories, mask = self.guidance_adapter.null_candidate_memories(index)
        else:
            tokens, mask, index = self.guidance_adapter.validate(
                guidance_tokens_group,
                guidance_token_mask_group,
                guidance_group_index,
                int(node_mask.shape[0]),
            )
            if tokens.device != node_mask.device:
                raise ValueError("guidance tokens must be on the graph device")
            memories, mask = self.guidance_adapter.candidate_memories(tokens, mask, index)
        if index.device != node_mask.device or mask.device != node_mask.device:
            raise ValueError("guidance index and mask must be on the graph device")
        self._candidate_memory_by_block = memories
        self._candidate_text_mask = mask
        self._candidate_residual_scale = residual_scale
        try:
            # Keep an explicit tensor-bearing context in GraphDiT's block
            # closure.  Activation-checkpoint recomputation happens after
            # this method's cleanup, so relying only on the transient module
            # attributes would make the recomputed guided blocks lose their
            # text memory.
            return super().forward(
                _candidate_memory_by_block=memories,
                _candidate_text_mask=mask,
                _guidance_residual_scale=residual_scale,
                **kwargs,
            )
        finally:
            self._candidate_memory_by_block = None
            self._candidate_text_mask = None
            self._candidate_residual_scale = 1.0

    def pre_block_hook(
        self,
        block_index: int,
        hidden_x: Tensor,
        hidden_e: Tensor,
        attention_mask: Tensor,
        **kwargs: object,
    ) -> tuple[Tensor, Tensor]:
        if block_index in self.injection_blocks:
            memory_by_block = kwargs.get(
                "_candidate_memory_by_block", self._candidate_memory_by_block
            )
            text_mask = kwargs.get("_candidate_text_mask", self._candidate_text_mask)
            residual_scale = kwargs.get(
                "_guidance_residual_scale", self._candidate_residual_scale
            )
            if not isinstance(memory_by_block, dict) or not isinstance(text_mask, Tensor):
                raise RuntimeError("guided forward did not bind candidate text memory")
            hidden_x = self.guidance_adapter.cross_attention[str(block_index)](
                hidden_x,
                memory_by_block[block_index],
                text_mask,
                attention_mask,
                residual_scale=residual_scale,
            )
        return hidden_x, hidden_e


__all__ = [
    "AsymmetricNativeKVMultiheadAttention",
    "GroupedTokenGuidanceAdapter",
    "HIDDEN_DIM",
    "INJECTION_BLOCKS",
    "INJECTION_LAYERS",
    "NativeKVNodeTextCrossAttention",
    "NativeKVTokenGuidanceAdapter",
    "NodeTextCrossAttention",
    "SemanticConditionAdapter",
    "TEXT_DIM",
    "TokenCrossAttentionGraphDiT",
    "TokenGuidanceAdapter",
    "TokenTextProjector",
    "build_token_guidance_adapter",
]
