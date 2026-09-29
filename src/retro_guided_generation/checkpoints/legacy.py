"""Read trusted historical Lightning checkpoints without legacy runtime imports.

Historical RetroDiT checkpoints may pickle a small number of scheduler or
sampling-store classes from the original source tree.  Those objects are not
needed to recover a model state dictionary.  This module replaces only those
known metadata classes with opaque containers while retaining PyTorch's normal
tensor reconstruction logic.

This is intentionally not a generic safe pickle loader.  Callers must opt in
with ``trust_checkpoint=True`` and should use it only for checkpoints obtained
from a trusted source.  The new repository never imports the historical source
tree at runtime and does not calculate content hashes.
"""

from __future__ import annotations

import pickle
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class LegacyCheckpointCompatibilityError(RuntimeError):
    """Raised when a historical checkpoint cannot satisfy this contract."""


class _OpaqueLegacyMetadata:
    """Accept serialized metadata that is irrelevant to model restoration."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = tuple(args)
        self.kwargs = dict(kwargs)
        self.state: object | None = None

    def __setstate__(self, state: object) -> None:
        self.state = state


_OPAQUE_LEGACY_GLOBALS = frozenset(
    {
        ("src.models.lr_scheduler", "PolyNomialLRScheduler"),
        ("src.utils.sampling_store", "ResultStore"),
    }
)


class _LegacyMetadataUnpickler(pickle.Unpickler):
    """Resolve the narrow known historical metadata surface without imports."""

    def find_class(self, module: str, name: str) -> object:
        if (module, name) in _OPAQUE_LEGACY_GLOBALS:
            return _OpaqueLegacyMetadata
        if module == "src" or module.startswith("src."):
            raise pickle.UnpicklingError(
                "unsupported historical runtime global "
                f"{module}.{name}; add an explicit state-dict-only adapter"
            )
        return super().find_class(module, name)


def _legacy_pickle_module() -> types.ModuleType:
    module = types.ModuleType("retro_guided_generation_legacy_pickle")
    module.Unpickler = _LegacyMetadataUnpickler
    module.dump = pickle.dump
    module.dumps = pickle.dumps
    module.load = pickle.load
    module.loads = pickle.loads
    return module


def _torch_module() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by installation
        raise LegacyCheckpointCompatibilityError(
            "PyTorch is required to inspect or load a legacy checkpoint"
        ) from error
    return torch


def _require_trusted_checkpoint(trust_checkpoint: bool) -> None:
    if trust_checkpoint is not True:
        raise LegacyCheckpointCompatibilityError(
            "legacy checkpoints are pickle containers; pass "
            "trust_checkpoint=True only for a trusted local artifact"
        )


def _load_payload(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    trust_checkpoint: bool,
) -> Mapping[str, Any]:
    _require_trusted_checkpoint(trust_checkpoint)
    path = Path(checkpoint_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"legacy checkpoint does not exist: {path}")
    torch = _torch_module()
    try:
        payload = torch.load(
            path,
            map_location=map_location,
            pickle_module=_legacy_pickle_module(),
            weights_only=False,
        )
    except (OSError, pickle.PickleError, RuntimeError, ValueError) as error:
        raise LegacyCheckpointCompatibilityError(
            f"cannot load legacy checkpoint {path}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise LegacyCheckpointCompatibilityError(
            "legacy checkpoint payload must be a mapping"
        )
    return payload


def _state_dict_from_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state_dict = payload.get("state_dict", payload)
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise LegacyCheckpointCompatibilityError(
            "legacy checkpoint must contain a non-empty mapping state_dict"
        )
    if not all(isinstance(name, str) for name in state_dict):
        raise LegacyCheckpointCompatibilityError(
            "legacy checkpoint state_dict keys must be strings"
        )
    torch = _torch_module()
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise LegacyCheckpointCompatibilityError(
            "legacy checkpoint state_dict values must all be tensors"
        )
    return state_dict


def _observed_prefixes(state_dict: Mapping[str, Any]) -> tuple[str, ...]:
    prefixes = {
        ".".join(name.split(".")[:2]) if "." in name else name
        for name in state_dict
    }
    return tuple(sorted(prefixes))


@dataclass(frozen=True)
class LegacyCheckpointSummary:
    """Portable metadata extracted from a trusted historical checkpoint."""

    path: Path
    global_step: int | None
    epoch: int | None
    top_level_keys: tuple[str, ...]
    state_dict_entries: int
    tensor_entries: int
    observed_state_prefixes: tuple[str, ...]

    @classmethod
    def from_payload(
        cls,
        checkpoint_path: str | Path,
        payload: Mapping[str, Any],
    ) -> "LegacyCheckpointSummary":
        state_dict = _state_dict_from_payload(payload)
        raw_step = payload.get("global_step")
        raw_epoch = payload.get("epoch")
        if raw_step is not None and (
            isinstance(raw_step, bool) or not isinstance(raw_step, int) or raw_step < 0
        ):
            raise LegacyCheckpointCompatibilityError(
                "legacy checkpoint global_step must be a non-negative integer or null"
            )
        if raw_epoch is not None and (
            isinstance(raw_epoch, bool) or not isinstance(raw_epoch, int) or raw_epoch < 0
        ):
            raise LegacyCheckpointCompatibilityError(
                "legacy checkpoint epoch must be a non-negative integer or null"
            )
        return cls(
            path=Path(checkpoint_path).expanduser(),
            global_step=raw_step,
            epoch=raw_epoch,
            top_level_keys=tuple(sorted(str(key) for key in payload)),
            state_dict_entries=len(state_dict),
            tensor_entries=len(state_dict),
            observed_state_prefixes=_observed_prefixes(state_dict),
        )

    def as_dict(self) -> dict[str, object]:
        """Return JSON-ready metadata without content-derived identifiers."""

        return {
            "path": str(self.path),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "top_level_keys": list(self.top_level_keys),
            "state_dict_entries": self.state_dict_entries,
            "tensor_entries": self.tensor_entries,
            "observed_state_prefixes": list(self.observed_state_prefixes),
            "content_hashes_computed": False,
        }


def inspect_legacy_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    trust_checkpoint: bool = False,
) -> LegacyCheckpointSummary:
    """Inspect trusted historical checkpoint metadata without old imports."""

    payload = _load_payload(
        checkpoint_path,
        map_location=map_location,
        trust_checkpoint=trust_checkpoint,
    )
    return LegacyCheckpointSummary.from_payload(checkpoint_path, payload)


def load_legacy_state_dict(
    checkpoint_path: str | Path,
    *,
    map_location: str = "cpu",
    trust_checkpoint: bool = False,
) -> dict[str, Any]:
    """Return a tensor-only legacy state dictionary without old imports."""

    payload = _load_payload(
        checkpoint_path,
        map_location=map_location,
        trust_checkpoint=trust_checkpoint,
    )
    state_dict = _state_dict_from_payload(payload)
    return dict(state_dict)


__all__ = [
    "LegacyCheckpointCompatibilityError",
    "LegacyCheckpointSummary",
    "inspect_legacy_checkpoint",
    "load_legacy_state_dict",
]
