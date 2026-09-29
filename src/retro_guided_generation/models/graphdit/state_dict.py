"""State-dict mapping for the standalone GraphDiT backbone."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from torch import Tensor

from .core import GraphDiT


class GraphDiTStateCompatibilityError(RuntimeError):
    """Raised when a supplied state dictionary cannot initialize GraphDiT."""


@dataclass(frozen=True, slots=True)
class GraphDiTStateLoadReport:
    """Explicit record of a non-mutating legacy state-key transformation."""

    source_prefix: str | None
    loaded_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source_prefix": self.source_prefix,
            "loaded_key_count": self.loaded_key_count,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }


def _strip_shared_prefix(
    state_dict: Mapping[str, Tensor],
    prefix: str,
) -> tuple[dict[str, Tensor], str | None]:
    if not prefix:
        return dict(state_dict), None
    keys = tuple(state_dict)
    prefixed = tuple(key.startswith(prefix) for key in keys)
    if all(prefixed):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}, prefix
    if not any(prefixed):
        return dict(state_dict), None
    raise GraphDiTStateCompatibilityError(
        f"state dictionary mixes keys with and without prefix {prefix!r}"
    )


def load_graphdit_state_dict(
    model: GraphDiT,
    state_dict: Mapping[str, Tensor],
    *,
    legacy_prefix: str = "net.",
    strict: bool = True,
) -> GraphDiTStateLoadReport:
    """Load V4 ``net.`` weights into the standalone GraphDiT module.

    Only parameter key names are transformed.  Tensor values, dtypes, and
    shapes are delegated unchanged to ``torch.nn.Module.load_state_dict``.
    """

    if not isinstance(model, GraphDiT):
        raise TypeError("model must be a GraphDiT instance")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise GraphDiTStateCompatibilityError("state_dict must be a non-empty mapping")
    if not all(isinstance(key, str) and isinstance(value, Tensor) for key, value in state_dict.items()):
        raise GraphDiTStateCompatibilityError(
            "state_dict must map string names to torch tensors"
        )
    normalized, source_prefix = _strip_shared_prefix(state_dict, legacy_prefix)
    incompatible = model.load_state_dict(normalized, strict=strict)
    report = GraphDiTStateLoadReport(
        source_prefix=source_prefix,
        loaded_key_count=len(normalized),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
    )
    if strict and (report.missing_keys or report.unexpected_keys):
        raise GraphDiTStateCompatibilityError(
            "strict GraphDiT state load left incompatible keys: "
            f"missing={report.missing_keys}, unexpected={report.unexpected_keys}"
        )
    return report


__all__ = [
    "GraphDiTStateCompatibilityError",
    "GraphDiTStateLoadReport",
    "load_graphdit_state_dict",
]
