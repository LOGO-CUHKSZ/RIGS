"""Registered topology contracts for token-to-GraphDiT guidance adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias


TOKEN_CROSS_ATTENTION_FAMILY = "token_cross_attention"
DEFAULT_GUIDANCE_ARCHITECTURE_ID = "token_shared_2468_v1"
TOKEN_INPUT_KIND = "tokens"
PROJECTED_KV_MODE = "projected"
NATIVE_KV_MODE = "native"
SHARED_ADAPTER_STATE_SCHEMA = "token_guidance_adapter_shared_state_v1"
GROUPED_ADAPTER_STATE_SCHEMA = "token_guidance_adapter_grouped_state_v1"
NATIVE_KV_ADAPTER_STATE_SCHEMA = "token_guidance_adapter_native_kv_state_v1"


@dataclass(frozen=True, slots=True)
class GuidanceArchitectureSpec:
    """Immutable description of one registered adapter topology.

    Projected K/V adapters share a native-text projector among the one-based
    GraphDiT layers in each ``projector_groups`` entry.  Native K/V adapters
    instead own a block-local text normalization and K/V projections.
    """

    architecture_id: str
    projector_groups: tuple[tuple[int, ...], ...]
    family: str = TOKEN_CROSS_ATTENTION_FAMILY
    version: int = 1
    num_heads: int = 4
    kv_mode: str = PROJECTED_KV_MODE
    native_kv_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.architecture_id, str) or not self.architecture_id:
            raise ValueError("architecture_id must be a non-empty string")
        if self.family != TOKEN_CROSS_ATTENTION_FAMILY:
            raise ValueError(f"unsupported guidance family: {self.family!r}")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        if isinstance(self.num_heads, bool) or not isinstance(self.num_heads, int) or self.num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if self.kv_mode not in {PROJECTED_KV_MODE, NATIVE_KV_MODE}:
            raise ValueError(f"unsupported K/V mode: {self.kv_mode!r}")
        if not isinstance(self.projector_groups, tuple) or any(
            not isinstance(group, tuple) or not group for group in self.projector_groups
        ):
            raise TypeError("projector_groups must be a tuple of non-empty tuples")
        if not isinstance(self.native_kv_layers, tuple):
            raise TypeError("native_kv_layers must be a tuple")
        if self.kv_mode == PROJECTED_KV_MODE:
            if not self.projector_groups or self.native_kv_layers:
                raise ValueError("projected K/V requires projector groups and no native layers")
        elif self.projector_groups or not self.native_kv_layers:
            raise ValueError("native K/V requires native layers and no projector groups")

        projected_layers: list[int] = []
        for group in self.projector_groups:
            if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in group):
                raise TypeError("projector group layers must be integers")
            if any(layer < 1 for layer in group):
                raise ValueError("projector group layers must be positive and one-based")
            if tuple(sorted(group)) != group:
                raise ValueError("layers in each projector group must be sorted")
            projected_layers.extend(group)
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in self.native_kv_layers):
            raise TypeError("native K/V layers must be integers")
        if any(layer < 1 for layer in self.native_kv_layers):
            raise ValueError("native K/V layers must be positive and one-based")
        if tuple(sorted(self.native_kv_layers)) != self.native_kv_layers:
            raise ValueError("native K/V layers must be sorted")
        if len(set(self.native_kv_layers)) != len(self.native_kv_layers):
            raise ValueError("native K/V layers cannot contain duplicates")
        if len(set(projected_layers)) != len(projected_layers):
            raise ValueError("one injection layer may occur in only one projector group")
        if tuple(sorted(projected_layers)) != tuple(projected_layers):
            raise ValueError("projector groups must be ordered by injection layer")

    @property
    def injection_layers(self) -> tuple[int, ...]:
        if self.kv_mode == NATIVE_KV_MODE:
            return self.native_kv_layers
        return tuple(layer for group in self.projector_groups for layer in group)

    @property
    def injection_blocks(self) -> tuple[int, ...]:
        return tuple(layer - 1 for layer in self.injection_layers)

    @property
    def projector_group_blocks(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(layer - 1 for layer in group) for group in self.projector_groups)

    @property
    def projector_group_keys(self) -> tuple[str, ...]:
        return tuple("layers_" + "_".join(str(layer) for layer in group) for group in self.projector_groups)

    @property
    def adapter_state_schema_version(self) -> str:
        if self.architecture_id == DEFAULT_GUIDANCE_ARCHITECTURE_ID:
            return SHARED_ADAPTER_STATE_SCHEMA
        if self.kv_mode == NATIVE_KV_MODE:
            return NATIVE_KV_ADAPTER_STATE_SCHEMA
        return GROUPED_ADAPTER_STATE_SCHEMA

    def contract(
        self,
        *,
        text_dim: int,
        hidden_dim: int,
        semantic_dim: int | None = None,
        condition_dim: int | None = None,
    ) -> dict[str, Any]:
        _validate_positive_dimension("text_dim", text_dim)
        _validate_positive_dimension("hidden_dim", hidden_dim)
        if semantic_dim is None:
            semantic_dim = hidden_dim
        if condition_dim is None:
            condition_dim = hidden_dim
        _validate_positive_dimension("semantic_dim", semantic_dim)
        _validate_positive_dimension("condition_dim", condition_dim)
        if hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by the architecture head count")
        return {
            "architecture_id": self.architecture_id,
            "family": self.family,
            "version": self.version,
            "input_kind": TOKEN_INPUT_KIND,
            "kv_mode": self.kv_mode,
            "adapter_state_schema_version": self.adapter_state_schema_version,
            "text_dim": text_dim,
            "semantic_dim": semantic_dim,
            "condition_dim": condition_dim,
            "conditioning_width_k": condition_dim,
            "condition_dim_source": (
                "graphdit.x_dim" if condition_dim == hidden_dim else "explicit_adapter_contract"
            ),
            "hidden_dim": hidden_dim,
            "condition_adapter": {
                "input_dim": semantic_dim,
                "output_dim": condition_dim,
                "identity": semantic_dim == condition_dim,
            },
            "cross_attention": {
                "query_dim": hidden_dim,
                "key_dim": condition_dim,
                "value_dim": condition_dim,
            },
            "num_heads": self.num_heads,
            "injection_layers": list(self.injection_layers),
            "injection_blocks": list(self.injection_blocks),
            "projector_groups": [list(group) for group in self.projector_groups],
            "projector_group_blocks": [list(group) for group in self.projector_group_blocks],
        }


def _validate_positive_dimension(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


_ARCHITECTURE_SPECS = (
    GuidanceArchitectureSpec(
        architecture_id=DEFAULT_GUIDANCE_ARCHITECTURE_ID,
        projector_groups=((2, 4, 6, 8),),
    ),
    GuidanceArchitectureSpec(
        architecture_id="token_grouped_24_68_v1",
        projector_groups=((2, 4), (6, 8)),
    ),
    GuidanceArchitectureSpec(
        architecture_id="token_grouped_24_68_1012_1416_h8_v1",
        projector_groups=((2, 4), (6, 8), (10, 12), (14, 16)),
        num_heads=8,
    ),
    GuidanceArchitectureSpec(
        architecture_id="token_independent_2468_v1",
        projector_groups=((2,), (4,), (6,), (8,)),
    ),
    GuidanceArchitectureSpec(
        architecture_id="token_independent_68_v1",
        projector_groups=((6,), (8,)),
    ),
    GuidanceArchitectureSpec(
        architecture_id="token_native_kv_68_v1",
        projector_groups=(),
        kv_mode=NATIVE_KV_MODE,
        native_kv_layers=(6, 8),
    ),
)

GUIDANCE_ARCHITECTURES: Mapping[str, GuidanceArchitectureSpec] = MappingProxyType(
    {spec.architecture_id: spec for spec in _ARCHITECTURE_SPECS}
)
GuidanceArchitectureSelector: TypeAlias = (
    str | Mapping[str, object] | GuidanceArchitectureSpec | None
)


def guidance_architecture_ids() -> tuple[str, ...]:
    """Return the registered canonical IDs in declaration order."""

    return tuple(GUIDANCE_ARCHITECTURES)


def resolve_guidance_architecture(
    value: GuidanceArchitectureSelector = None,
) -> GuidanceArchitectureSpec:
    """Resolve one registered architecture without silently accepting extras."""

    if value is None:
        architecture_id = DEFAULT_GUIDANCE_ARCHITECTURE_ID
    elif isinstance(value, GuidanceArchitectureSpec):
        registered = GUIDANCE_ARCHITECTURES.get(value.architecture_id)
        if registered is None or registered != value:
            raise ValueError("guidance architecture spec is not exactly registered")
        return registered
    elif isinstance(value, str):
        architecture_id = value
    elif isinstance(value, Mapping):
        if set(value) != {"architecture_id"}:
            raise ValueError("guidance architecture mapping must contain only architecture_id")
        architecture_id = value["architecture_id"]
        if not isinstance(architecture_id, str):
            raise TypeError("architecture_id must be a string")
    else:
        raise TypeError("guidance architecture must be an id, mapping, spec, or None")
    try:
        return GUIDANCE_ARCHITECTURES[architecture_id]
    except KeyError as error:
        options = ", ".join(guidance_architecture_ids())
        raise ValueError(f"unknown guidance architecture {architecture_id!r}; choices: {options}") from error


__all__ = [
    "DEFAULT_GUIDANCE_ARCHITECTURE_ID",
    "GROUPED_ADAPTER_STATE_SCHEMA",
    "GUIDANCE_ARCHITECTURES",
    "GuidanceArchitectureSelector",
    "GuidanceArchitectureSpec",
    "NATIVE_KV_ADAPTER_STATE_SCHEMA",
    "NATIVE_KV_MODE",
    "PROJECTED_KV_MODE",
    "SHARED_ADAPTER_STATE_SCHEMA",
    "TOKEN_CROSS_ATTENTION_FAMILY",
    "guidance_architecture_ids",
    "resolve_guidance_architecture",
]
