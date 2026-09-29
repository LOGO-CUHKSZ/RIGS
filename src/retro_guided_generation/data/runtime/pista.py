"""Read-only Pista cache adapters for canonical Stage A/Stage B batches.

The Pista graph and Qwen token caches predate the canonical guidance corpus.
This module binds them by explicit route IDs and exact stored text strings,
never by recomputing a content digest.  It intentionally does not encode new
text, rebuild graph caches, or verify whole-file digests; those are separate
materialization jobs with their own resource contracts.
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from retro_guided_generation.data.runtime.batching import VirtualMicrobatchMerge
from retro_guided_generation.data.runtime.canonical import (
    GuidanceRuntimeIndex,
    RuntimeGuidanceCondition,
)
from retro_guided_generation.data.runtime.graphdit import (
    GuidedStageABatchMaterializer,
    RouteGraphKey,
    RouteGraphRecord,
    move_guided_stage_a_batch,
)
from retro_guided_generation.data.runtime.projector_only import (
    ProjectorOnlyGraphFreeBatchMaterializer,
    ProjectorOnlyMetadataStore,
)
from retro_guided_generation.guidance.io import iter_jsonl, iter_models, read_json
from retro_guided_generation.guidance.schema import SemanticGroup, TextPool
from retro_guided_generation.models.graphdit import GraphDiTConfig
from retro_guided_generation.models.stages import GuidedStageABatch


_SPLIT_TO_CACHE_SPLIT = {"train": "train", "validation": "val", "test": "test"}
PISTA_TOKEN_EXTENSION_SCHEMA = "retro_guided_generation.pista_token_extension.v1"
PISTA_NODE_COUNT_SIDECAR_SCHEMA = "retro_guided_generation.pista_node_count_sidecar.v1"
# These defaults are per DDP rank. One record builder avoids the measured
# PyG/Torch intra-rank contention while the ordered next-batch producer
# overlaps it with GPU work; two open shards avoid immediate GPFS churn.
DEFAULT_CPU_BATCH_WORKERS = 1
DEFAULT_MAX_OPEN_GRAPH_SHARDS = 2


class PistaCacheError(RuntimeError):
    """Raised when an existing Pista cache cannot satisfy the runtime contract."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PistaCacheError(f"{name} must be an object")
    return value


def _path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PistaCacheError(f"{name} must be a non-empty relative path")
    try:
        candidate = (root / value).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError) as error:
        raise PistaCacheError(f"{name} is unavailable under {root}") from error
    return candidate


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PistaCacheError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PistaCacheError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "non-negative finite"
        raise PistaCacheError(f"{name} must be {qualifier}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PistaCacheError(f"{name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class _TokenRow:
    text_idx: int
    shard_id: int
    local_start: int
    length: int
    query_start: int
    query_length: int


@dataclass(frozen=True, slots=True)
class _TokenLocation:
    """Locate one canonical surface in a read-only token source."""

    source_name: str
    text_idx: int


class PistaTokenCache:
    """Lazily slice a legacy Qwen token cache through its explicit text table."""

    def __init__(
        self,
        *,
        token_cache_root: str | Path,
        guidance_text_table: str | Path,
        expected_hidden_dim: int = 4096,
    ) -> None:
        self.root = Path(token_cache_root).expanduser().resolve(strict=True)
        self.guidance_text_table = Path(guidance_text_table).expanduser().resolve(strict=True)
        manifest = read_json(self.root / "manifest.json")
        hidden_dim = _integer(manifest.get("hidden_dim"), "token manifest.hidden_dim", minimum=1)
        if hidden_dim != expected_hidden_dim:
            raise PistaCacheError(
                f"Pista token width {hidden_dim} differs from expected {expected_hidden_dim}"
            )
        if manifest.get("dtype") != "bfloat16":
            raise PistaCacheError("Pista token cache must use bfloat16 states")
        prompt_count = _integer(manifest.get("prompt_count"), "token manifest.prompt_count", minimum=1)
        prompt_template = manifest.get("prompt_template")
        if not isinstance(prompt_template, str) or "{guidance_text}" not in prompt_template:
            raise PistaCacheError(
                "Pista token cache prompt_template must contain '{guidance_text}'"
            )
        self.hidden_dim = hidden_dim
        self.prompt_template = prompt_template
        self._rows = self._load_rows(prompt_count)
        self._text_to_index = self._load_guidance_text_index(prompt_count)
        self._row_cache_capacity = 256
        self._row_cache: OrderedDict[int, Tensor] = OrderedDict()

    def _load_rows(self, expected_count: int) -> tuple[_TokenRow, ...]:
        path = self.root / "table.jsonl"
        if not path.is_file():
            raise PistaCacheError(f"Pista token table is missing: {path}")
        rows: list[_TokenRow] = []
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            text_idx = _integer(row.get("text_idx"), f"token row {line_number}.text_idx")
            if text_idx != len(rows):
                raise PistaCacheError("Pista token table text_idx values must be contiguous")
            shard_id = _integer(row.get("shard_id"), f"token row {line_number}.shard_id")
            local_start = _integer(row.get("local_start"), f"token row {line_number}.local_start")
            length = _integer(row.get("length"), f"token row {line_number}.length", minimum=1)
            query_start = _integer(row.get("query_start"), f"token row {line_number}.query_start")
            query_length = _integer(
                row.get("query_length"), f"token row {line_number}.query_length", minimum=1
            )
            if query_start + query_length > length:
                raise PistaCacheError("Pista token query span exceeds its row length")
            shard = self.root / "shards" / f"tokens-{shard_id:05d}.safetensors"
            if not shard.is_file():
                raise PistaCacheError(f"Pista token row references a missing shard: {shard}")
            rows.append(
                _TokenRow(text_idx, shard_id, local_start, length, query_start, query_length)
            )
        if len(rows) != expected_count:
            raise PistaCacheError(
                f"Pista token table has {len(rows)} rows, manifest declares {expected_count}"
            )
        return tuple(rows)

    def _load_guidance_text_index(self, expected_count: int) -> dict[str, int]:
        result: dict[str, int] = {}
        for line_number, row in enumerate(iter_jsonl(self.guidance_text_table), start=1):
            text = _text(row.get("guidance_text"), f"guidance text row {line_number}.guidance_text")
            text_idx = _integer(row.get("text_idx"), f"guidance text row {line_number}.text_idx")
            if text_idx >= expected_count:
                raise PistaCacheError("guidance text table references an out-of-range token row")
            previous = result.setdefault(text, text_idx)
            if previous != text_idx:
                raise PistaCacheError(
                    "one exact guidance string maps to more than one legacy token row"
                )
        if not result:
            raise PistaCacheError("Pista guidance text table is empty")
        return result

    @property
    def known_texts(self) -> frozenset[str]:
        """Exact guidance surfaces available from the immutable legacy cache."""

        return frozenset(self._text_to_index)

    def text_index(self, text: str) -> int | None:
        """Return the legacy cache coordinate for one exact surface, if present."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._text_to_index.get(text)

    def token_row(self, text_idx: int) -> Tensor:
        if isinstance(text_idx, bool) or not isinstance(text_idx, int):
            raise TypeError("text_idx must be an int")
        try:
            row = self._rows[text_idx]
        except IndexError as error:
            raise IndexError(f"Pista token text_idx is outside the table: {text_idx}") from error
        cached = self._row_cache.pop(text_idx, None)
        if cached is not None:
            self._row_cache[text_idx] = cached
            return cached
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise RuntimeError("safetensors is required to read Pista token memories") from error
        path = self.root / "shards" / f"tokens-{row.shard_id:05d}.safetensors"
        start = row.local_start + row.query_start
        end = start + row.query_length
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"tokens"}:
                    raise PistaCacheError("Pista token shard must contain only a 'tokens' tensor")
                view = handle.get_slice("tokens")
                shape = view.get_shape()
                if len(shape) != 2 or shape[1] != self.hidden_dim or start < 0 or end > shape[0]:
                    raise PistaCacheError("Pista token span exceeds its declared shard")
                value = view[start:end].clone().contiguous()
        except PistaCacheError:
            raise
        except Exception as error:
            raise PistaCacheError(f"cannot read Pista token shard {path}: {error}") from error
        if value.dtype != torch.bfloat16 or value.shape != (row.query_length, self.hidden_dim):
            raise PistaCacheError("Pista token row has an unexpected dtype or shape")
        if not bool(torch.isfinite(value.float()).all()):
            raise PistaCacheError("Pista token row contains NaN or Inf")
        self._row_cache[text_idx] = value
        while len(self._row_cache) > self._row_cache_capacity:
            self._row_cache.popitem(last=False)
        return value

    def bind_text_pools(self, pools: Sequence[TextPool]) -> "PistaTextTokenMapping":
        """Bind canonical text IDs to cached token rows without re-encoding text."""
        return PistaTokenMemory(base=self).bind_text_pools(pools)


class PistaTokenExtensionCache:
    """Read a compact no-rewrite supplement for missing Pista text surfaces.

    A supplement is intentionally separate from the historical token cache. It
    contains only exact surfaces that were absent from that cache (normally the
    verified counterfactual inverses) and has the same token-row layout.  The
    reader performs bounded schema and span checks but does not derive or
    validate content hashes.
    """

    def __init__(
        self,
        *,
        token_extension_root: str | Path,
        expected_hidden_dim: int = 4096,
    ) -> None:
        self.root = Path(token_extension_root).expanduser().resolve(strict=True)
        manifest = read_json(self.root / "manifest.json")
        if manifest.get("schema_version") != PISTA_TOKEN_EXTENSION_SCHEMA:
            raise PistaCacheError("unsupported Pista token extension schema")
        if manifest.get("status") != "complete":
            raise PistaCacheError("Pista token extension is not complete")
        hidden_dim = _integer(
            manifest.get("hidden_dim"), "token extension manifest.hidden_dim", minimum=1
        )
        if hidden_dim != expected_hidden_dim:
            raise PistaCacheError(
                f"Pista token extension width {hidden_dim} differs from expected "
                f"{expected_hidden_dim}"
            )
        if manifest.get("dtype") != "bfloat16":
            raise PistaCacheError("Pista token extension must use bfloat16 states")
        prompt_count = _integer(
            manifest.get("prompt_count"),
            "token extension manifest.prompt_count",
            minimum=1,
        )
        prompt_template = manifest.get("prompt_template")
        if not isinstance(prompt_template, str) or "{guidance_text}" not in prompt_template:
            raise PistaCacheError(
                "Pista token extension prompt_template must contain '{guidance_text}'"
            )
        self.hidden_dim = hidden_dim
        self.prompt_template = prompt_template
        self._rows = self._load_rows(prompt_count)
        self._text_to_index = self._load_text_index(prompt_count)
        self._row_cache_capacity = 256
        self._row_cache: OrderedDict[int, Tensor] = OrderedDict()

    def _load_rows(self, expected_count: int) -> tuple[_TokenRow, ...]:
        path = self.root / "table.jsonl"
        if not path.is_file():
            raise PistaCacheError(f"Pista token extension table is missing: {path}")
        rows: list[_TokenRow] = []
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            text_idx = _integer(
                row.get("text_idx"), f"token extension row {line_number}.text_idx"
            )
            if text_idx != len(rows):
                raise PistaCacheError(
                    "Pista token extension text_idx values must be contiguous"
                )
            shard_id = _integer(
                row.get("shard_id"), f"token extension row {line_number}.shard_id"
            )
            local_start = _integer(
                row.get("local_start"),
                f"token extension row {line_number}.local_start",
            )
            length = _integer(
                row.get("length"), f"token extension row {line_number}.length", minimum=1
            )
            query_start = _integer(
                row.get("query_start"),
                f"token extension row {line_number}.query_start",
            )
            query_length = _integer(
                row.get("query_length"),
                f"token extension row {line_number}.query_length",
                minimum=1,
            )
            if query_start + query_length > length:
                raise PistaCacheError("Pista token extension query span exceeds its row length")
            shard = self.root / "shards" / f"tokens-{shard_id:05d}.safetensors"
            if not shard.is_file():
                raise PistaCacheError(
                    f"Pista token extension row references a missing shard: {shard}"
                )
            rows.append(
                _TokenRow(text_idx, shard_id, local_start, length, query_start, query_length)
            )
        if len(rows) != expected_count:
            raise PistaCacheError(
                f"Pista token extension table has {len(rows)} rows, manifest declares "
                f"{expected_count}"
            )
        return tuple(rows)

    def _load_text_index(self, expected_count: int) -> dict[str, int]:
        path = self.root / "texts.jsonl"
        if not path.is_file():
            raise PistaCacheError(f"Pista token extension texts are missing: {path}")
        result: dict[str, int] = {}
        seen_indices: set[int] = set()
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            text = _text(
                row.get("guidance_text"),
                f"token extension text {line_number}.guidance_text",
            )
            text_idx = _integer(
                row.get("text_idx"), f"token extension text {line_number}.text_idx"
            )
            if text_idx >= expected_count:
                raise PistaCacheError(
                    "Pista token extension text references an out-of-range token row"
                )
            if text in result:
                raise PistaCacheError(
                    "Pista token extension contains one exact guidance string more than once"
                )
            if text_idx in seen_indices:
                raise PistaCacheError(
                    "Pista token extension text index maps to more than one surface"
                )
            result[text] = text_idx
            seen_indices.add(text_idx)
        if len(result) != expected_count or seen_indices != set(range(expected_count)):
            raise PistaCacheError(
                "Pista token extension text table must cover every contiguous token row"
            )
        return result

    @property
    def known_texts(self) -> frozenset[str]:
        return frozenset(self._text_to_index)

    def text_index(self, text: str) -> int | None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self._text_to_index.get(text)

    def token_row(self, text_idx: int) -> Tensor:
        if isinstance(text_idx, bool) or not isinstance(text_idx, int):
            raise TypeError("text_idx must be an int")
        try:
            row = self._rows[text_idx]
        except IndexError as error:
            raise IndexError(
                f"Pista token extension text_idx is outside the table: {text_idx}"
            ) from error
        cached = self._row_cache.pop(text_idx, None)
        if cached is not None:
            self._row_cache[text_idx] = cached
            return cached
        try:
            from safetensors import safe_open
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise RuntimeError("safetensors is required to read Pista token memories") from error
        path = self.root / "shards" / f"tokens-{row.shard_id:05d}.safetensors"
        start = row.local_start + row.query_start
        end = start + row.query_length
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                if set(handle.keys()) != {"tokens"}:
                    raise PistaCacheError(
                        "Pista token extension shard must contain only a 'tokens' tensor"
                    )
                view = handle.get_slice("tokens")
                shape = view.get_shape()
                if (
                    len(shape) != 2
                    or shape[1] != self.hidden_dim
                    or start < 0
                    or end > shape[0]
                ):
                    raise PistaCacheError(
                        "Pista token extension span exceeds its declared shard"
                    )
                value = view[start:end].clone().contiguous()
        except PistaCacheError:
            raise
        except Exception as error:
            raise PistaCacheError(
                f"cannot read Pista token extension shard {path}: {error}"
            ) from error
        if value.dtype != torch.bfloat16 or value.shape != (row.query_length, self.hidden_dim):
            raise PistaCacheError("Pista token extension row has an unexpected dtype or shape")
        if not bool(torch.isfinite(value.float()).all()):
            raise PistaCacheError("Pista token extension row contains NaN or Inf")
        self._row_cache[text_idx] = value
        while len(self._row_cache) > self._row_cache_capacity:
            self._row_cache.popitem(last=False)
        return value


class PistaSemanticRouteCache:
    """Read frozen reaction embeddings indexed by the stored graph index.

    The cache is a model-ready teacher asset: it is read-only, has no trainable
    parameters, and is never rewritten by a Stage A/B run.  Compatibility is
    established from its declared schema, split, dimension, and graph-index
    range; no content digest is computed or checked.
    """

    def __init__(
        self,
        *,
        semantic_cache_root: str | Path,
        split: str,
        expected_dim: int | None = None,
    ) -> None:
        if split not in _SPLIT_TO_CACHE_SPLIT:
            raise ValueError(f"unsupported canonical Pista split {split!r}")
        self.root = Path(semantic_cache_root).expanduser().resolve(strict=True)
        manifest = read_json(self.root / "manifest.json")
        if manifest.get("schema_version") != "route_semantic_cache_v1":
            raise PistaCacheError("unsupported Pista semantic-route cache schema")
        if manifest.get("status") != "complete":
            raise PistaCacheError("Pista semantic-route cache is not complete")
        semantic_dim = _integer(manifest.get("semantic_dim"), "semantic cache.semantic_dim", minimum=1)
        source_semantic_dim = manifest.get("source_semantic_dim", semantic_dim)
        if source_semantic_dim != semantic_dim or "zero_pad" in str(
            manifest.get("embedding_lift", "")
        ).lower():
            raise PistaCacheError(
                "Pista semantic-route cache must contain native-width embeddings; "
                "fixed zero-padding is not a valid teacher coordinate"
            )
        if expected_dim is not None and semantic_dim != expected_dim:
            raise PistaCacheError(
                f"Pista semantic-route width {semantic_dim} differs from expected {expected_dim}"
            )
        splits = _mapping(manifest.get("splits"), "semantic cache.splits")
        entry = _mapping(splits.get(_SPLIT_TO_CACHE_SPLIT[split]), f"semantic cache split {split}")
        path = _path(self.root, entry.get("path"), f"semantic cache split {split}.path")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise PistaCacheError(f"cannot read Pista semantic-route tensor {path}: {error}") from error
        if not isinstance(payload, Mapping) or set(payload) != {"embedding", "embedding_valid"}:
            raise PistaCacheError(
                "Pista semantic-route payload must contain exactly embedding and embedding_valid"
            )
        embeddings = payload["embedding"]
        embedding_valid = payload["embedding_valid"]
        expected_dtype = torch.float16 if manifest.get("dtype") == "float16" else torch.bfloat16
        if (
            not isinstance(embeddings, Tensor)
            or embeddings.ndim != 2
            or embeddings.shape[0] < 1
            or embeddings.shape[1] != semantic_dim
            or embeddings.dtype != expected_dtype
        ):
            raise PistaCacheError("Pista semantic-route tensor has an unexpected shape or dtype")
        if (
            not isinstance(embedding_valid, Tensor)
            or embedding_valid.dtype != torch.bool
            or embedding_valid.shape != (embeddings.shape[0],)
        ):
            raise PistaCacheError("Pista semantic-route embedding_valid has an unexpected shape or dtype")
        if not bool(embedding_valid.any()):
            raise PistaCacheError("Pista semantic-route cache contains no reachable embeddings")
        declared_count = _integer(
            entry.get("num_embeddings"),
            f"semantic cache split {split}.num_embeddings",
            minimum=1,
        )
        if declared_count != embeddings.shape[0]:
            raise PistaCacheError("Pista semantic-route tensor count differs from its manifest")
        value = embeddings.detach().to(device="cpu").contiguous()
        valid_value = value[embedding_valid]
        if value.requires_grad or not bool(torch.isfinite(valid_value.float()).all()):
            raise PistaCacheError("Pista semantic-route tensor contains invalid values")
        norm = valid_value.float().norm(dim=-1)
        if not bool(torch.allclose(norm, torch.ones_like(norm), atol=5.0e-3, rtol=0.0)):
            raise PistaCacheError("Pista semantic-route tensor is not unit normalized")
        self.split = split
        self.semantic_dim = semantic_dim
        self._embeddings = value
        self._embedding_valid = embedding_valid.detach().to(device="cpu", dtype=torch.bool).contiguous()

    def embedding(self, graph_idx: int) -> Tensor:
        if isinstance(graph_idx, bool) or not isinstance(graph_idx, int):
            raise TypeError("graph_idx must be an integer")
        if not 0 <= graph_idx < self._embeddings.shape[0]:
            raise KeyError(f"Pista semantic-route cache lacks graph index {graph_idx}")
        if not bool(self._embedding_valid[graph_idx]):
            raise KeyError(
                "Pista semantic-route cache has no reachable route embedding "
                f"for graph index {graph_idx}"
            )
        return self._embeddings[graph_idx].detach()


class PistaTokenMemory:
    """Resolve canonical text surfaces across a base cache and one supplement."""

    def __init__(
        self,
        *,
        base: PistaTokenCache,
        extension: PistaTokenExtensionCache | None = None,
    ) -> None:
        if not isinstance(base, PistaTokenCache):
            raise TypeError("base must be PistaTokenCache")
        if extension is not None and not isinstance(extension, PistaTokenExtensionCache):
            raise TypeError("extension must be PistaTokenExtensionCache or None")
        if extension is not None and extension.hidden_dim != base.hidden_dim:
            raise PistaCacheError("Pista token base and extension hidden widths differ")
        if extension is not None:
            if extension.prompt_template != base.prompt_template:
                raise PistaCacheError(
                    "Pista token base and extension prompt templates differ"
                )
            overlaps = base.known_texts.intersection(extension.known_texts)
            if overlaps:
                preview = ", ".join(sorted(overlaps)[:3])
                raise PistaCacheError(
                    "Pista token extension must contain only base-cache-missing surfaces; "
                    f"found overlap: {preview}"
                )
        self.base = base
        self.extension = extension
        self.hidden_dim = base.hidden_dim

    @classmethod
    def from_paths(
        cls,
        *,
        token_cache_root: str | Path,
        guidance_text_table: str | Path,
        token_extension_root: str | Path | None = None,
        expected_hidden_dim: int = 4096,
    ) -> "PistaTokenMemory":
        base = PistaTokenCache(
            token_cache_root=token_cache_root,
            guidance_text_table=guidance_text_table,
            expected_hidden_dim=expected_hidden_dim,
        )
        extension = (
            None
            if token_extension_root is None
            else PistaTokenExtensionCache(
                token_extension_root=token_extension_root,
                expected_hidden_dim=expected_hidden_dim,
            )
        )
        return cls(base=base, extension=extension)

    def _location(self, text: str) -> _TokenLocation | None:
        base_index = self.base.text_index(text)
        if base_index is not None:
            return _TokenLocation("base", base_index)
        if self.extension is not None:
            extension_index = self.extension.text_index(text)
            if extension_index is not None:
                return _TokenLocation("extension", extension_index)
        return None

    def token_row(self, location: _TokenLocation) -> Tensor:
        if not isinstance(location, _TokenLocation):
            raise TypeError("location must be a token location")
        if location.source_name == "base":
            return self.base.token_row(location.text_idx)
        if location.source_name == "extension" and self.extension is not None:
            return self.extension.token_row(location.text_idx)
        raise PistaCacheError(f"unknown Pista token source {location.source_name!r}")

    def bind_text_pools(self, pools: Sequence[TextPool]) -> "PistaTextTokenMapping":
        """Map every pool text ID to its exact cached token sequence."""

        locations: dict[str, _TokenLocation] = {}
        missing: list[str] = []
        for pool in pools:
            for variant in pool.variants:
                location = self._location(variant.text)
                if location is None:
                    missing.append(variant.text_id)
                    continue
                previous = locations.setdefault(variant.text_id, location)
                if previous != location:
                    raise PistaCacheError(
                        f"canonical text ID {variant.text_id!r} maps to conflicting cache rows"
                    )
        if missing:
            preview = ", ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f" (+{len(missing) - 8} more)"
            raise PistaCacheError(
                "the configured Pista token sources do not contain all canonical text "
                f"surfaces: {preview}{suffix}; materialize a compatible token extension first"
            )
        return PistaTextTokenMapping(memory=self, text_locations=locations)


class PistaTextTokenMapping(Mapping[str, Tensor]):
    """A lazy Mapping accepted directly by ``GuidedStageABatchMaterializer``."""

    def __init__(
        self,
        *,
        memory: PistaTokenMemory,
        text_locations: Mapping[str, _TokenLocation],
    ) -> None:
        if not isinstance(memory, PistaTokenMemory):
            raise TypeError("memory must be PistaTokenMemory")
        self.memory = memory
        self._text_locations = dict(text_locations)
        if not self._text_locations:
            raise ValueError("text_locations must not be empty")

    def __getitem__(self, text_id: str) -> Tensor:
        try:
            location = self._text_locations[text_id]
        except KeyError as error:
            raise KeyError(text_id) from error
        return self.memory.token_row(location)

    def __iter__(self) -> Iterator[str]:
        return iter(self._text_locations)

    def __len__(self) -> int:
        return len(self._text_locations)

    @property
    def hidden_dim(self) -> int:
        return self.memory.hidden_dim


def load_corpus_text_pools(
    corpus_root: str | Path,
    *,
    branches: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
) -> tuple[TextPool, ...]:
    """Load requested text pools once, preserving global metric-pool reuse.

    Omitting both filters preserves the historical all-branch/all-split API.
    Evaluation can request only the frozen manifest's branch and split, which
    avoids opening unrelated text-pool artifacts without changing any text ID.
    """

    root = Path(corpus_root).expanduser().resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    artifacts = manifest.get("artifacts")
    manifest_splits = manifest.get("splits")
    if not isinstance(artifacts, Mapping) or not isinstance(manifest_splits, list):
        raise PistaCacheError("canonical corpus manifest is missing artifacts or splits")
    selected_branches = ("metric", "structure") if branches is None else tuple(branches)
    if (
        not selected_branches
        or len(set(selected_branches)) != len(selected_branches)
        or set(selected_branches) - {"metric", "structure"}
    ):
        raise PistaCacheError("text-pool branches must be unique metric/structure names")
    selected_splits = tuple(manifest_splits) if splits is None else tuple(splits)
    if (
        not selected_splits
        or len(set(selected_splits)) != len(selected_splits)
        or any(split not in manifest_splits for split in selected_splits)
    ):
        raise PistaCacheError("text-pool splits must be unique members of the corpus")
    paths: list[Path] = []
    if "metric" in selected_branches:
        paths.append(_path(root, artifacts.get("metric_text_pools"), "metric_text_pools"))
    for split in selected_splits if "structure" in selected_branches else ():
        if split not in _SPLIT_TO_CACHE_SPLIT:
            raise PistaCacheError(f"canonical corpus has unsupported split {split!r}")
        paths.append(
            _path(root, artifacts.get(f"structure_text_pools.{split}"), f"structure_text_pools.{split}")
        )
    result: dict[str, TextPool] = {}
    for path in paths:
        for pool in iter_models(path, TextPool):
            previous = result.setdefault(pool.text_pool_id, pool)
            if previous != pool:
                raise PistaCacheError(
                    f"canonical text pool {pool.text_pool_id!r} differs across artifacts"
                )
    if not result:
        raise PistaCacheError("canonical corpus contains no text pools")
    return tuple(result.values())


@dataclass(frozen=True, slots=True)
class _RouteLocator:
    graph_idx: int
    endpoint_weight: float


def _canonical_route_id_for_sidecar_row(
    *,
    row: Mapping[str, object],
    case_id: str,
    stored_route_id: str,
    expected_route_ids: frozenset[str] | None,
) -> str | None:
    """Resolve old source IDs and new candidate UIDs to one corpus route ID."""

    if expected_route_ids is None:
        return stored_route_id
    raw_aliases = (
        stored_route_id,
        row.get("candidate_uid"),
        f"{case_id}::{stored_route_id}",
        row.get("source_id"),
        row.get("source_route_id"),
    )
    matches = tuple(
        dict.fromkeys(
            alias
            for alias in raw_aliases
            if isinstance(alias, str) and alias in expected_route_ids
        )
    )
    if len(matches) > 1:
        raise PistaCacheError(
            "Pista route row matches more than one canonical route identity: "
            f"{matches!r}"
        )
    return None if not matches else matches[0]


class PistaRouteGraphStore(Mapping[RouteGraphKey, RouteGraphRecord]):
    """Lazy route-key adapter over read-only Pista-compatible graph shards.

    Only selected Pista cases are indexed, and at most ``max_open_shards``
    collated PyG shard payloads are retained.  The default of one keeps memory
    bounded on a 24-CPU worker while preserving random group sampling.
    """

    def __init__(
        self,
        *,
        graph_cache_root: str | Path,
        split: str,
        graphdit: GraphDiTConfig,
        case_ids: Collection[str] | None = None,
        candidate_pool_ids_by_case: Mapping[str, str] | None = None,
        canonical_route_ids_by_case: Mapping[str, Collection[str]] | None = None,
        max_open_shards: int = DEFAULT_MAX_OPEN_GRAPH_SHARDS,
        record_cache_capacity: int = 4096,
        record_materialize_workers: int = DEFAULT_CPU_BATCH_WORKERS,
    ) -> None:
        if split not in _SPLIT_TO_CACHE_SPLIT:
            raise ValueError(f"unsupported canonical Pista split {split!r}")
        if not isinstance(graphdit, GraphDiTConfig):
            raise TypeError("graphdit must be GraphDiTConfig")
        if isinstance(max_open_shards, bool) or not isinstance(max_open_shards, int) or max_open_shards < 1:
            raise ValueError("max_open_shards must be a positive integer")
        if (
            isinstance(record_cache_capacity, bool)
            or not isinstance(record_cache_capacity, int)
            or record_cache_capacity < 1
        ):
            raise ValueError("record_cache_capacity must be a positive integer")
        if (
            isinstance(record_materialize_workers, bool)
            or not isinstance(record_materialize_workers, int)
            or record_materialize_workers < 1
        ):
            raise ValueError("record_materialize_workers must be a positive integer")
        self.root = Path(graph_cache_root).expanduser().resolve(strict=True)
        self.split = split
        self.cache_split = _SPLIT_TO_CACHE_SPLIT[split]
        self.graphdit = graphdit
        self.max_open_shards = max_open_shards
        self.record_cache_capacity = record_cache_capacity
        self.record_materialize_workers = record_materialize_workers
        selected_cases = None if case_ids is None else set(case_ids)
        if selected_cases is not None and any(not isinstance(case, str) or not case for case in selected_cases):
            raise ValueError("case_ids must contain non-empty strings")
        pool_ids: dict[str, str] = {}
        if candidate_pool_ids_by_case is not None:
            for case_id, candidate_pool_id in candidate_pool_ids_by_case.items():
                pool_ids[_text(case_id, "candidate pool mapping.case_id")] = _text(
                    candidate_pool_id,
                    "candidate pool mapping.candidate_pool_id",
                )
        if selected_cases is not None and not set(pool_ids).issubset(selected_cases):
            raise ValueError("candidate_pool_ids_by_case contains a case outside case_ids")
        self._candidate_pool_ids_by_case = pool_ids
        route_ids: dict[str, frozenset[str]] = {}
        if canonical_route_ids_by_case is not None:
            for case_id, values in canonical_route_ids_by_case.items():
                normalized_case_id = _text(case_id, "canonical route mapping.case_id")
                if isinstance(values, str) or not isinstance(values, Collection):
                    raise TypeError("canonical route mapping values must be collections of route IDs")
                normalized_values = frozenset(
                    _text(route_id, f"canonical route mapping {normalized_case_id!r}.route_id")
                    for route_id in values
                )
                if not normalized_values:
                    raise ValueError("canonical route mapping values must not be empty")
                route_ids[normalized_case_id] = normalized_values
        if selected_cases is not None and not set(route_ids).issubset(selected_cases):
            raise ValueError("canonical_route_ids_by_case contains a case outside case_ids")
        self._canonical_route_ids_by_case = route_ids
        self._shard_id, self._local_idx, self._num_nodes = self._load_index()
        self._manifest_shard_paths = self._load_manifest_shard_paths()
        self._routes = self._load_routes(selected_cases)
        self._shard_cache: OrderedDict[int, tuple[Any, Mapping[str, Tensor] | None]] = OrderedDict()
        self._record_cache: OrderedDict[RouteGraphKey, RouteGraphRecord] = OrderedDict()
        self._node_count_cache: dict[RouteGraphKey, int] = {}
        self._record_materialize_executor: ThreadPoolExecutor | None = None

    def _load_node_count_sidecar(self, expected_count: int) -> Tensor | None:
        """Load an optional immutable node-count index beside the graph shards.

        Historical Pista endpoint indexes contain only ``(shard_id, local_idx)``.
        Without this derived sidecar, cost-balanced batching has to load a full
        PyG shard merely to inspect one graph's node count.  The sidecar is
        optional so old caches retain their established fallback behavior, but
        when present it removes that repeated GPFS read without changing route
        selection or graph materialization.
        """

        path = self.root / "endpoint_graphs" / f"node_counts_{self.cache_split}.pt"
        if not path.is_file():
            return None
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise PistaCacheError(f"cannot load Pista node-count sidecar {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise PistaCacheError("Pista node-count sidecar must be a mapping")
        if payload.get("schema_version") != PISTA_NODE_COUNT_SIDECAR_SCHEMA:
            raise PistaCacheError("unsupported Pista node-count sidecar schema")
        if payload.get("cache_split") != self.cache_split:
            raise PistaCacheError("Pista node-count sidecar split differs from graph index")
        if payload.get("index_entries") != expected_count:
            raise PistaCacheError("Pista node-count sidecar length differs from graph index")
        value = payload.get("num_nodes")
        if (
            not isinstance(value, Tensor)
            or value.ndim != 1
            or value.numel() != expected_count
            or value.dtype not in {torch.int16, torch.int32, torch.int64}
            or value.numel() == 0
            or bool((value <= 0).any())
        ):
            raise PistaCacheError("Pista node-count sidecar has an invalid num_nodes vector")
        return value.to(device="cpu", dtype=torch.long).contiguous()

    def _load_index(self) -> tuple[Tensor, Tensor, Tensor | None]:
        path = self.root / "endpoint_graphs" / f"index_{self.cache_split}.pt"
        if not path.is_file():
            raise PistaCacheError(f"Pista endpoint index is missing: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise PistaCacheError(f"cannot load Pista endpoint index {path}: {error}") from error
        if not isinstance(payload, Mapping):
            raise PistaCacheError("Pista endpoint index must be a mapping")
        values: list[Tensor] = []
        for name in ("shard_id", "local_idx"):
            value = payload.get(name)
            if not isinstance(value, Tensor) or value.ndim != 1 or value.dtype not in {torch.int16, torch.int32, torch.int64}:
                raise PistaCacheError(f"Pista endpoint index field {name!r} must be an integer vector")
            values.append(value.to(device="cpu", dtype=torch.long).contiguous())
        shard_id, local_idx = values
        raw_num_nodes = payload.get("num_nodes")
        num_nodes: Tensor | None
        if raw_num_nodes is None:
            # Historical Pista indexes encode only shard/local coordinates.
            # Prefer the derived sidecar when available; otherwise the graph's
            # node count is read lazily from its stored p_x tensor.
            num_nodes = self._load_node_count_sidecar(int(shard_id.numel()))
        elif (
            isinstance(raw_num_nodes, Tensor)
            and raw_num_nodes.ndim == 1
            and raw_num_nodes.dtype in {torch.int16, torch.int32, torch.int64}
        ):
            num_nodes = raw_num_nodes.to(device="cpu", dtype=torch.long).contiguous()
        else:
            raise PistaCacheError("Pista endpoint index field 'num_nodes' must be an integer vector")
        if shard_id.numel() != local_idx.numel() or (
            num_nodes is not None and num_nodes.numel() != shard_id.numel()
        ):
            raise PistaCacheError("Pista endpoint index fields have different lengths")
        if (
            shard_id.numel() == 0
            or bool((shard_id < 0).any())
            or bool((local_idx < 0).any())
            or (num_nodes is not None and bool((num_nodes <= 0).any()))
        ):
            raise PistaCacheError("Pista endpoint index contains invalid coordinates")
        return shard_id, local_idx, num_nodes

    def _load_manifest_shard_paths(self) -> dict[int, Path]:
        """Resolve optional external shard records from a Top60 cache manifest.

        Legacy Pista caches retain shards directly beneath ``endpoint_graphs``.
        The Top60 GraphDiT cache instead records immutable Top30 and Rank31-60
        shard locations in its manifest.  Both layouts use the same endpoint
        index coordinates, so this only resolves an existing stored path; it
        never copies, writes, or re-encodes graph data.
        """

        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            return {}
        manifest = _mapping(read_json(manifest_path), "Pista graph manifest")
        components = manifest.get("components")
        if not isinstance(components, Mapping):
            return {}
        guided = components.get("guided")
        if not isinstance(guided, Mapping):
            return {}
        endpoint_graphs = guided.get("endpoint_graphs")
        if not isinstance(endpoint_graphs, Mapping):
            return {}
        split_entry = endpoint_graphs.get(self.cache_split)
        if not isinstance(split_entry, Mapping):
            return {}
        artifacts = split_entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            return {}
        shard_records = artifacts.get("shards")
        if not isinstance(shard_records, list):
            return {}

        resolved: dict[int, Path] = {}
        for ordinal, raw_record in enumerate(shard_records):
            record = _mapping(raw_record, f"Pista graph manifest shard {ordinal}")
            raw_shard_id = record.get("shard_id")
            if raw_shard_id is None:
                # Standard local Pista caches record direct endpoint shards in
                # the manifest only as provenance (path, size, digest).  Their
                # immutable numeric coordinate is supplied by index_<split>.pt
                # and _shard_path resolves the local filename directly.
                continue
            shard_id = _integer(
                raw_shard_id,
                f"Pista graph manifest shard {ordinal}.shard_id",
            )
            raw_path = record.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                raise PistaCacheError(f"Pista graph manifest shard {ordinal}.path must be a path string")
            candidate = Path(raw_path).expanduser()
            if not candidate.is_absolute():
                candidate = self.root / candidate
            try:
                candidate = candidate.resolve(strict=True)
            except OSError as error:
                raise PistaCacheError(
                    f"Pista graph manifest shard {ordinal} is unavailable: {raw_path}"
                ) from error
            if not candidate.is_file():
                raise PistaCacheError(f"Pista graph manifest shard {ordinal} is not a file: {candidate}")
            if shard_id in resolved:
                raise PistaCacheError(f"Pista graph manifest declares shard {shard_id} more than once")
            resolved[shard_id] = candidate
        return resolved

    def _load_routes(self, selected_cases: set[str] | None) -> dict[RouteGraphKey, _RouteLocator]:
        path = self.root / "endpoint_graphs" / f"routes_{self.cache_split}.jsonl"
        if not path.is_file():
            raise PistaCacheError(f"Pista route sidecar is missing: {path}")
        result: dict[RouteGraphKey, _RouteLocator] = {}
        for line_number, row in enumerate(iter_jsonl(path), start=1):
            case_id = _text(row.get("case_id"), f"Pista route row {line_number}.case_id")
            if selected_cases is not None and case_id not in selected_cases:
                continue
            stored_route_id = _text(
                row.get("route_id", row.get("candidate_id")),
                f"Pista route row {line_number}.route_id",
            )
            expected_route_ids = self._canonical_route_ids_by_case.get(case_id)
            route_id = _canonical_route_id_for_sidecar_row(
                row=row,
                case_id=case_id,
                stored_route_id=stored_route_id,
                expected_route_ids=expected_route_ids,
            )
            if route_id is None:
                # The cache can contain graphable routes outside this
                # canonical corpus.  They are irrelevant to this view;
                # preflight later proves that every selected contrast has the
                # route graph it requires.
                continue
            declared_pool_id = row.get("candidate_pool_id")
            mapped_pool_id = self._candidate_pool_ids_by_case.get(case_id)
            if declared_pool_id is not None:
                candidate_pool_id = _text(
                    declared_pool_id,
                    f"Pista route row {line_number}.candidate_pool_id",
                )
                if mapped_pool_id is not None and mapped_pool_id != candidate_pool_id:
                    raise PistaCacheError(
                        f"Pista route row {line_number} conflicts with the corpus candidate-pool mapping"
                    )
            elif mapped_pool_id is not None:
                candidate_pool_id = mapped_pool_id
            else:
                # Historical Top15 cache sidecars omit the pool identifier.
                # Keep their established key convention when no canonical
                # corpus mapping was supplied.
                candidate_pool_id = f"pista_top15::{case_id}"
            graph_idx = _integer(row.get("graph_idx"), f"Pista route row {line_number}.graph_idx")
            if graph_idx >= self._shard_id.numel():
                raise PistaCacheError("Pista route sidecar has an out-of-range graph_idx")
            observed = row.get("observed_record_count")
            endpoint_weight = (
                # A zero observation count is a legitimate candidate prior: it
                # means this endpoint was retained by the pool but has no
                # observed source records.  The materializer preserves such a
                # zero when another route has mass and falls back to a uniform
                # group prior only when every selected endpoint is zero.
                _finite(observed, f"Pista route row {line_number}.observed_record_count")
                if observed is not None
                else 1.0
            )
            key = (case_id, candidate_pool_id, route_id)
            if key in result:
                raise PistaCacheError(f"duplicate Pista route identity {key!r}")
            result[key] = _RouteLocator(graph_idx, endpoint_weight)
        if not result:
            scope = "the selected Pista cases" if selected_cases is not None else "the Pista route sidecar"
            raise PistaCacheError(f"no endpoint graphs were found for {scope}")
        return result

    def __len__(self) -> int:
        return len(self._routes)

    def __iter__(self) -> Iterator[RouteGraphKey]:
        return iter(self._routes)

    @property
    def graph_count(self) -> int:
        """Return the immutable index-space size, including every graph row."""

        return int(self._shard_id.numel())

    def graph_index(self, key: RouteGraphKey) -> int:
        """Return the immutable cache coordinate for one canonical route."""

        try:
            return self._routes[key].graph_idx
        except KeyError as error:
            raise KeyError(f"Pista graph cache lacks canonical route {key!r}") from error

    def _record_materializer_executor(self) -> ThreadPoolExecutor:
        """Return the process-local pool used for bounded record construction."""

        executor = self._record_materialize_executor
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=self.record_materialize_workers,
                thread_name_prefix="pista-record-build",
            )
            self._record_materialize_executor = executor
        return executor

    def close(self) -> None:
        """Release the optional record-materialization workers deterministically."""

        executor = self._record_materialize_executor
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._record_materialize_executor = None

    def prefetch_records(
        self, keys: Sequence[RouteGraphKey]
    ) -> Mapping[RouteGraphKey, RouteGraphRecord]:
        """Materialize a batch's routes in shard order, preserving key order.

        Runtime sampling is intentionally random, but the legacy endpoint
        cache is organized into large GPFS-backed PyG shards.  Resolving the
        same batch in route order can evict a shard and immediately reopen it.
        Sorting only this bounded batch by immutable shard coordinates removes
        that I/O churn; the returned mapping leaves all supervision and
        candidate ordering unchanged.
        """

        unique_keys = tuple(dict.fromkeys(keys))
        try:
            ordered_keys = tuple(
                sorted(
                    unique_keys,
                    key=lambda key: (
                        int(self._shard_id[self._routes[key].graph_idx]),
                        int(self._local_idx[self._routes[key].graph_idx]),
                    ),
                )
            )
        except KeyError as error:
            raise KeyError(f"Pista graph cache lacks canonical route {error.args[0]!r}") from error
        if self.record_materialize_workers == 1:
            # Preserve the original bounded-LRU behavior for generic callers
            # that have not opted into the Pista batch-parallel path.
            for key in ordered_keys:
                self[key]
            return {key: self[key] for key in unique_keys}
        locators = {key: self._routes[key] for key in unique_keys}
        # Keep every in-batch cache hit alive in a transient mapping.  Adding
        # a newly built record can evict an older hit from the bounded LRU
        # before its position in ``ordered_keys`` is visited; the result still
        # needs that immutable record for this batch.  The transient map is
        # batch-local and does not change the final LRU capacity or order.
        cached_records = {
            key: self._record_cache[key]
            for key in unique_keys
            if key in self._record_cache
        }
        missing_keys = tuple(key for key in ordered_keys if key not in self._record_cache)

        # Resolve one immutable shard at a time, preserving the bounded-LRU
        # contract.  Do not retain every distinct payload in a local mapping:
        # a Top60 batch can span many GPFS shards, and retaining all of them
        # defeats ``max_open_shards`` precisely when CPU materialization is
        # meant to relieve the accelerator.  Record construction is parallel
        # only among the keys that share the currently open shard; its inputs
        # are then read-only and no shard-cache mutation occurs in worker
        # threads.
        keys_by_shard: OrderedDict[int, list[RouteGraphKey]] = OrderedDict()
        for key in missing_keys:
            locator = locators[key]
            shard_id = int(self._shard_id[locator.graph_idx])
            keys_by_shard.setdefault(shard_id, []).append(key)

        new_records: dict[RouteGraphKey, RouteGraphRecord] = {}
        executor = (
            None
            if len(missing_keys) < 2
            else self._record_materializer_executor()
        )
        for shard_id, shard_keys in keys_by_shard.items():
            loaded_shard = self._load_shard(shard_id)

            def build_record(key: RouteGraphKey) -> RouteGraphRecord:
                return self._to_record_from_loaded_shard(
                    key,
                    locators[key],
                    loaded_shard,
                )

            if executor is None or len(shard_keys) == 1:
                for key in shard_keys:
                    new_records[key] = build_record(key)
            else:
                futures = {key: executor.submit(build_record, key) for key in shard_keys}
                for key in shard_keys:
                    new_records[key] = futures[key].result()

        # Preserve the historical two-pass LRU touch order: first shard order
        # for efficient source access, then the caller's candidate order.  The
        # transient maps handle a deliberately tiny generic LRU without
        # re-materializing a record that was just built or temporarily evicted
        # while resolving this batch.
        def touch_record(key: RouteGraphKey, record: RouteGraphRecord) -> None:
            self._record_cache.pop(key, None)
            self._record_cache[key] = record
            while len(self._record_cache) > self.record_cache_capacity:
                self._record_cache.popitem(last=False)

        resolved: dict[RouteGraphKey, RouteGraphRecord] = {}
        for key in ordered_keys:
            record = cached_records.get(key)
            if record is None:
                record = new_records[key]
            touch_record(key, record)
            resolved[key] = record
        result: dict[RouteGraphKey, RouteGraphRecord] = {}
        for key in unique_keys:
            record = resolved[key]
            touch_record(key, record)
            result[key] = record
        return result

    def _shard_path(self, shard_id: int) -> Path:
        direct = self.root / "endpoint_graphs" / f"{self.cache_split}-{shard_id:05d}.pt"
        if direct.is_file():
            return direct
        try:
            return self._manifest_shard_paths[shard_id]
        except KeyError as error:
            raise PistaCacheError(
                f"Pista endpoint index references a missing shard {shard_id} for {self.cache_split!r}"
            ) from error

    @staticmethod
    def _safe_load_pyg_shard(path: Path) -> tuple[Any, Mapping[str, Tensor] | None]:
        try:
            from torch_geometric.data import Data
            from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
            from torch_geometric.data.storage import GlobalStorage
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise RuntimeError("torch_geometric is required to read Pista endpoint graphs") from error
        try:
            with torch.serialization.safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            raise PistaCacheError(
                f"cannot safely load Pista endpoint shard {path}: {error}"
            ) from error
        if not isinstance(payload, tuple) or len(payload) != 2:
            raise PistaCacheError("Pista endpoint shard must contain a (data, slices) tuple")
        data, slices = payload
        if slices is not None and not isinstance(slices, Mapping):
            raise PistaCacheError("Pista endpoint shard slices must be a mapping or null")
        return data, slices

    def _load_shard(self, shard_id: int) -> tuple[Any, Mapping[str, Tensor] | None]:
        cached = self._shard_cache.pop(shard_id, None)
        if cached is not None:
            self._shard_cache[shard_id] = cached
            return cached
        path = self._shard_path(shard_id)
        if not path.is_file():
            raise PistaCacheError(f"Pista endpoint index references a missing shard: {path}")
        loaded = self._safe_load_pyg_shard(path)
        self._shard_cache[shard_id] = loaded
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return loaded

    @staticmethod
    def _separate(data: Any, slices: Mapping[str, Tensor] | None, local_idx: int) -> Any:
        if slices is None:
            if local_idx != 0:
                raise PistaCacheError("single-graph Pista shard cannot serve a nonzero local_idx")
            return data.clone()
        try:
            from torch_geometric.data.separate import separate
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise RuntimeError("torch_geometric is required to separate Pista endpoint graphs") from error
        try:
            return separate(
                cls=data.__class__, batch=data, idx=local_idx, slice_dict=slices, decrement=False
            )
        except (IndexError, KeyError, RuntimeError, TypeError) as error:
            raise PistaCacheError(f"cannot separate Pista endpoint graph {local_idx}") from error

    @staticmethod
    def _dense_edges(graph: Any, prefix: str, *, nodes: int, classes: int) -> Tensor:
        edge_index = getattr(graph, f"{prefix}_edge_index", None)
        edge_attr = getattr(graph, f"{prefix}_edge_attr", None)
        if (
            not isinstance(edge_index, Tensor)
            or edge_index.dtype not in {torch.int32, torch.int64}
            or edge_index.shape[0] != 2
            or not isinstance(edge_attr, Tensor)
            or edge_attr.ndim != 2
            or edge_attr.shape != (edge_index.shape[1], classes)
        ):
            raise PistaCacheError(f"Pista graph has invalid {prefix}_edge tensors")
        if edge_index.numel() and (bool((edge_index < 0).any()) or bool((edge_index >= nodes).any())):
            raise PistaCacheError(f"Pista graph has out-of-range {prefix}_edge_index values")
        dense = torch.zeros((nodes, nodes, classes), dtype=edge_attr.dtype)
        if edge_index.numel():
            dense[edge_index[0].long(), edge_index[1].long()] = edge_attr
        return dense

    def _to_record_from_loaded_shard(
        self,
        key: RouteGraphKey,
        locator: _RouteLocator,
        loaded_shard: tuple[Any, Mapping[str, Tensor] | None],
    ) -> RouteGraphRecord:
        shard_id = int(self._shard_id[locator.graph_idx])
        local_idx = int(self._local_idx[locator.graph_idx])
        expected_nodes = (
            None
            if self._num_nodes is None
            else int(self._num_nodes[locator.graph_idx])
        )
        data, slices = loaded_shard
        graph = self._separate(data, slices, local_idx)
        p_x = getattr(graph, "p_x", None)
        r_x = getattr(graph, "r_x", None)
        if (
            not isinstance(p_x, Tensor)
            or not isinstance(r_x, Tensor)
            or p_x.ndim != 2
            or r_x.shape != p_x.shape
            or p_x.shape[1] != self.graphdit.node_classes
            or not p_x.is_floating_point()
            or not r_x.is_floating_point()
        ):
            raise PistaCacheError("Pista graph node tensors differ from the GraphDiT contract")
        nodes = p_x.shape[0]
        if expected_nodes is not None and nodes != expected_nodes:
            raise PistaCacheError("Pista graph node count differs from its endpoint index")
        if nodes > self.graphdit.max_n_len + self.graphdit.n_dummy:
            raise PistaCacheError("Pista graph exceeds the configured GraphDiT node capacity")
        p_e = self._dense_edges(
            graph, "p", nodes=nodes, classes=self.graphdit.edge_classes
        )
        r_e = self._dense_edges(
            graph, "r", nodes=nodes, classes=self.graphdit.edge_classes
        )
        p_mask = ~p_x[:, -1].bool()
        node_mask = torch.ones(nodes, dtype=torch.bool)
        if self.graphdit.condition_dim != 0:
            raise PistaCacheError(
                "the current Pista endpoint cache has no condition payload; "
                "use a condition_dim=0 GraphDiTConfig"
            )
        return RouteGraphRecord(
            case_id=key[0],
            candidate_pool_id=key[1],
            route_id=key[2],
            p_X=p_x.detach().cpu().contiguous(),
            p_E=p_e.detach().cpu().contiguous(),
            r_X=r_x.detach().cpu().contiguous(),
            r_E=r_e.detach().cpu().contiguous(),
            node_mask=node_mask,
            p_mask=p_mask,
            cond=torch.zeros(0, dtype=torch.float32),
            endpoint_weight=locator.endpoint_weight,
        )

    def _to_record(self, key: RouteGraphKey, locator: _RouteLocator) -> RouteGraphRecord:
        shard_id = int(self._shard_id[locator.graph_idx])
        return self._to_record_from_loaded_shard(
            key,
            locator,
            self._load_shard(shard_id),
        )

    def __getitem__(self, key: RouteGraphKey) -> RouteGraphRecord:
        if not isinstance(key, tuple) or len(key) != 3 or not all(isinstance(part, str) for part in key):
            raise KeyError("Pista route keys must be (case_id, candidate_pool_id, route_id)")
        cached = self._record_cache.pop(key, None)
        if cached is not None:
            self._record_cache[key] = cached
            return cached
        try:
            locator = self._routes[key]
        except KeyError as error:
            raise KeyError(f"Pista graph cache lacks canonical route {key!r}") from error
        record = self._to_record(key, locator)
        self._record_cache[key] = record
        while len(self._record_cache) > self.record_cache_capacity:
            self._record_cache.popitem(last=False)
        return record

    def node_count(self, key: RouteGraphKey) -> int:
        """Return one route's node count without building dense graph tensors.

        Cost-balanced rank batching needs only the padded node dimension while
        choosing a lane.  The actual materializer still resolves the full
        ``RouteGraphRecord`` and retains all graph validation.  When the
        historical index lacks an explicit ``num_nodes`` vector, derive the
        count from the PyG shard's ``p_x`` slice boundaries.
        """

        if not isinstance(key, tuple) or len(key) != 3 or not all(isinstance(part, str) for part in key):
            raise KeyError("Pista route keys must be (case_id, candidate_pool_id, route_id)")
        cached = self._node_count_cache.get(key)
        if cached is not None:
            return cached
        try:
            locator = self._routes[key]
        except KeyError as error:
            raise KeyError(f"Pista graph cache lacks canonical route {key!r}") from error

        if self._num_nodes is not None:
            nodes = int(self._num_nodes[locator.graph_idx])
        else:
            shard_id = int(self._shard_id[locator.graph_idx])
            local_idx = int(self._local_idx[locator.graph_idx])
            data, slices = self._load_shard(shard_id)
            raw_p_x = getattr(data, "p_x", None)
            nodes: int | None = None
            if slices is None:
                if local_idx == 0 and isinstance(raw_p_x, Tensor) and raw_p_x.ndim >= 1:
                    nodes = int(raw_p_x.shape[0])
            else:
                raw_slices = slices.get("p_x") if isinstance(slices, Mapping) else None
                if (
                    isinstance(raw_slices, Tensor)
                    and raw_slices.ndim == 1
                    and 0 <= local_idx < raw_slices.numel() - 1
                ):
                    nodes = int(raw_slices[local_idx + 1] - raw_slices[local_idx])
            if nodes is None:
                # Preserve the established validation/error path for an
                # unusual shard layout that has no usable p_x slice metadata.
                nodes = self._to_record(key, locator).p_X.shape[0]

        if nodes < 1 or nodes > self.graphdit.max_n_len + self.graphdit.n_dummy:
            raise PistaCacheError(
                f"Pista graph node count is outside GraphDiT capacity for route {key!r}: {nodes}"
            )
        self._node_count_cache[key] = nodes
        return nodes


class PistaRouteSemanticMemory(Mapping[RouteGraphKey, Tensor]):
    """Resolve canonical route identities into frozen semantic teacher vectors."""

    def __init__(self, *, graph_store: PistaRouteGraphStore, cache: PistaSemanticRouteCache) -> None:
        if not isinstance(graph_store, PistaRouteGraphStore):
            raise TypeError("graph_store must be PistaRouteGraphStore")
        if not isinstance(cache, PistaSemanticRouteCache):
            raise TypeError("cache must be PistaSemanticRouteCache")
        if graph_store.split != cache.split:
            raise ValueError("Pista graph and semantic-route caches must use one split")
        self.graph_store = graph_store
        self.cache = cache

    def __getitem__(self, key: RouteGraphKey) -> Tensor:
        return self.cache.embedding(self.graph_store.graph_index(key))

    def __iter__(self) -> Iterator[RouteGraphKey]:
        return iter(self.graph_store)

    def __len__(self) -> int:
        return len(self.graph_store)


@dataclass(frozen=True, slots=True)
class PistaBatchDraw:
    """One materialized batch and the next deterministic runtime draw ordinal."""

    batch: GuidedStageABatch
    conditions: tuple[RuntimeGuidanceCondition, ...]
    next_draw_index: int
    # The graph-free projector path retains a one-node placeholder graph to
    # keep the established outer batch type.  Preserve the real graph-size
    # cost used by the sampler separately so formal telemetry and capacity
    # reporting continue to describe the unchanged canonical draw.
    logical_padded_graph_cells: int | None = None

    def to_device(self, device: torch.device | str) -> "PistaBatchDraw":
        return PistaBatchDraw(
            batch=move_guided_stage_a_batch(self.batch, device),
            conditions=self.conditions,
            next_draw_index=self.next_draw_index,
            logical_padded_graph_cells=self.logical_padded_graph_cells,
        )


# Backward-compatible public name for existing Pista configs and imports.
PistaVirtualMicrobatchMerge = VirtualMicrobatchMerge


class PistaStageABatchProvider:
    """Deterministically sample canonical Pista groups into bounded dense batches."""

    # Opt in to the formal runner's one-batch CPU producer.  The producer
    # emits a complete CPU batch; PistaBatchDraw.to_device performs the single
    # staged transfer after the future is ready.
    supports_cpu_prefetch = True

    def __init__(
        self,
        *,
        runtime: GuidanceRuntimeIndex,
        split: str,
        materializer: GuidedStageABatchMaterializer,
        include_shuffled: bool = False,
        max_groups_per_batch: int = 4,
        max_candidates_per_batch: int = 120,
        max_padded_graph_cells_per_batch: int | None = None,
        cost_balanced_rank_batching: bool = False,
        virtual_microbatch_merge: PistaVirtualMicrobatchMerge | None = None,
    ) -> None:
        if not isinstance(runtime, GuidanceRuntimeIndex):
            raise TypeError("runtime must be GuidanceRuntimeIndex")
        if split not in runtime.splits:
            raise ValueError(f"runtime does not expose split {split!r}")
        if not isinstance(materializer, GuidedStageABatchMaterializer):
            raise TypeError("materializer must be GuidedStageABatchMaterializer")
        for name, value in (
            ("max_groups_per_batch", max_groups_per_batch),
            ("max_candidates_per_batch", max_candidates_per_batch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_padded_graph_cells_per_batch is not None and (
            isinstance(max_padded_graph_cells_per_batch, bool)
            or not isinstance(max_padded_graph_cells_per_batch, int)
            or max_padded_graph_cells_per_batch < 1
        ):
            raise ValueError("max_padded_graph_cells_per_batch must be positive or None")
        if not isinstance(cost_balanced_rank_batching, bool):
            raise TypeError("cost_balanced_rank_batching must be a bool")
        if not isinstance(include_shuffled, bool):
            raise TypeError("include_shuffled must be a bool")
        if virtual_microbatch_merge is not None:
            if not isinstance(virtual_microbatch_merge, PistaVirtualMicrobatchMerge):
                raise TypeError(
                    "virtual_microbatch_merge must be PistaVirtualMicrobatchMerge or None"
                )
            if cost_balanced_rank_batching:
                raise ValueError(
                    "virtual microbatch merge is incompatible with cost-balanced rank batching"
                )
            if max_groups_per_batch != virtual_microbatch_merge.merged_groups:
                raise ValueError(
                    "max_groups_per_batch must equal virtual lane groups times merge_lanes"
                )
            if max_candidates_per_batch != virtual_microbatch_merge.merged_max_candidates:
                raise ValueError(
                    "max_candidates_per_batch must equal the scaled virtual-lane cap"
                )
            if (
                max_padded_graph_cells_per_batch
                != virtual_microbatch_merge.merged_max_padded_graph_cells
            ):
                raise ValueError(
                    "max_padded_graph_cells_per_batch must equal the scaled virtual-lane cap"
                )
        self.runtime = runtime
        self.split = split
        self.materializer = materializer
        self.include_shuffled = include_shuffled
        self.max_groups_per_batch = max_groups_per_batch
        self.max_candidates_per_batch = max_candidates_per_batch
        self.max_padded_graph_cells_per_batch = max_padded_graph_cells_per_batch
        self.cost_balanced_rank_batching = cost_balanced_rank_batching
        self.supports_cost_balanced_rank_batching = cost_balanced_rank_batching
        self.virtual_microbatch_merge = virtual_microbatch_merge

    @property
    def epoch_size_groups(self) -> int:
        """Global accepted-group target for one deterministic train epoch."""

        return self.runtime.epoch_group_count(self.split)

    @staticmethod
    def _signed_candidate_count(group: SemanticGroup) -> int:
        count = sum(
            item.eligible and item.label in {-1, 1}
            for item in group.candidate_preferences
        )
        if count < 2:
            raise PistaCacheError(
                f"semantic group {group.semantic_group_uid!r} lacks a signed contrast"
            )
        return count

    @classmethod
    def from_pista_paths(
        cls,
        *,
        corpus_root: str | Path,
        graph_cache_root: str | Path,
        token_cache_root: str | Path,
        guidance_text_table: str | Path,
        token_extension_root: str | Path | None = None,
        semantic_route_cache_root: str | Path | None = None,
        projector_only_metadata_root: str | Path | None = None,
        semantic_route_embedding_dim: int | None = None,
        split: str,
        graphdit: GraphDiTConfig,
        training_seed: int = 42,
        evaluation_seed: int = 1042,
        branch_weights: Mapping[str, float] | None = None,
        sampling_mode: str = "branch_cycle",
        max_groups_per_batch: int = 4,
        max_candidates_per_batch: int = 120,
        max_open_graph_shards: int = DEFAULT_MAX_OPEN_GRAPH_SHARDS,
        max_padded_graph_cells_per_batch: int | None = None,
        cost_balanced_rank_batching: bool = False,
        cpu_batch_workers: int = DEFAULT_CPU_BATCH_WORKERS,
        record_cache_capacity: int = 4096,
        include_shuffled: bool = False,
        virtual_microbatch_merge: PistaVirtualMicrobatchMerge | None = None,
    ) -> "PistaStageABatchProvider":
        """Bind an existing joint corpus to existing Pista graph/token caches."""

        weights = {"metric": 0.5, "structure": 0.5}
        if branch_weights is not None:
            weights.update({key: float(value) for key, value in branch_weights.items()})
        runtime = GuidanceRuntimeIndex.from_corpus(
            corpus_root,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
            branch_weights=weights,
            sampling_mode=sampling_mode,
        )
        pools = load_corpus_text_pools(corpus_root)
        token_memory = PistaTokenMemory.from_paths(
            token_cache_root=token_cache_root,
            guidance_text_table=guidance_text_table,
            expected_hidden_dim=4096,
            token_extension_root=token_extension_root,
        )
        text_tokens = token_memory.bind_text_pools(pools)
        if projector_only_metadata_root is not None:
            if split != "train":
                raise ValueError(
                    "projector_only_metadata_root may only be used for the train split"
                )
            if semantic_route_cache_root is None:
                raise ValueError(
                    "projector_only_metadata_root requires semantic_route_cache_root"
                )
            metadata = ProjectorOnlyMetadataStore(projector_only_metadata_root, split=split)
            semantic_cache = PistaSemanticRouteCache(
                semantic_cache_root=semantic_route_cache_root,
                split=split,
                expected_dim=semantic_route_embedding_dim,
            )
            materializer = ProjectorOnlyGraphFreeBatchMaterializer(
                graphdit=graphdit,
                guidance_text_dim=token_memory.hidden_dim,
                metadata=metadata,
                text_tokens=text_tokens,
                route_semantic_embedding=semantic_cache.embedding,
            )
            return cls(
                runtime=runtime,
                split=split,
                materializer=materializer,
                include_shuffled=include_shuffled,
                max_groups_per_batch=max_groups_per_batch,
                max_candidates_per_batch=max_candidates_per_batch,
                max_padded_graph_cells_per_batch=max_padded_graph_cells_per_batch,
                cost_balanced_rank_batching=cost_balanced_rank_batching,
                virtual_microbatch_merge=virtual_microbatch_merge,
            )
        candidate_pool_ids_by_case: dict[str, str] = {}
        canonical_route_ids_by_case: dict[str, set[str]] = {}
        for ordinal in range(runtime.group_count(split)):
            group = runtime.group(split, ordinal)
            existing = candidate_pool_ids_by_case.setdefault(group.case_id, group.candidate_pool_id)
            if existing != group.candidate_pool_id:
                raise PistaCacheError(
                    f"canonical corpus case {group.case_id!r} has more than one candidate_pool_id"
                )
            route_ids = canonical_route_ids_by_case.setdefault(group.case_id, set())
            route_ids.update(item.route_id for item in group.candidate_preferences)
        case_ids = set(candidate_pool_ids_by_case)
        graph_store = PistaRouteGraphStore(
            graph_cache_root=graph_cache_root,
            split=split,
            graphdit=graphdit,
            case_ids=case_ids,
            candidate_pool_ids_by_case=candidate_pool_ids_by_case,
            canonical_route_ids_by_case=canonical_route_ids_by_case,
            max_open_shards=max_open_graph_shards,
            record_cache_capacity=record_cache_capacity,
            record_materialize_workers=cpu_batch_workers,
        )
        semantic_memory = None
        if semantic_route_cache_root is not None:
            semantic_memory = PistaRouteSemanticMemory(
                graph_store=graph_store,
                cache=PistaSemanticRouteCache(
                    semantic_cache_root=semantic_route_cache_root,
                    split=split,
                    expected_dim=semantic_route_embedding_dim,
                ),
            )
        materializer = GuidedStageABatchMaterializer(
            graphdit=graphdit,
            guidance_text_dim=token_memory.hidden_dim,
            route_graphs=graph_store,
            text_tokens=text_tokens,
            route_semantic_embeddings=semantic_memory,
            cpu_materialize_workers=cpu_batch_workers,
        )
        return cls(
            runtime=runtime,
            split=split,
            materializer=materializer,
            include_shuffled=include_shuffled,
            max_groups_per_batch=max_groups_per_batch,
            max_candidates_per_batch=max_candidates_per_batch,
            max_padded_graph_cells_per_batch=max_padded_graph_cells_per_batch,
            cost_balanced_rank_batching=cost_balanced_rank_batching,
            virtual_microbatch_merge=virtual_microbatch_merge,
        )

    def close(self) -> None:
        """Release graph-store workers after a bounded preflight or run."""

        close = getattr(self.materializer.route_graphs, "close", None)
        if callable(close):
            close()

    def _exact_two_rank_partition(
        self,
        conditions: Sequence[RuntimeGuidanceCondition],
        costs: Sequence[tuple[int, int]],
        *,
        order: Sequence[int],
        branch_targets: Mapping[str, Sequence[int]],
        per_rank_batch_groups: int,
    ) -> tuple[tuple[RuntimeGuidanceCondition, ...], ...] | None:
        """Find a feasible two-rank assignment for one fixed global window.

        The first assignment tried is the same cheapest-lane choice as the
        historical greedy implementation.  If that local choice paints the
        remaining lane into a corner, the memoized search backtracks over only
        the bounded rank assignment; it never changes the draw window or
        branch counts.  This keeps the Pista batch contract exact while
        avoiding false failures caused by greedy packing.
        """

        if len(order) != len(conditions) or len(conditions) != 2 * per_rank_batch_groups:
            raise ValueError("two-rank partition received an incomplete global window")
        if set(branch_targets) != set(condition.group.branch for condition in conditions):
            raise ValueError("two-rank branch targets do not cover the global window")

        branch_names = tuple(sorted(branch_targets))
        branch_index = {name: index for index, name in enumerate(branch_names)}
        branches = tuple(condition.group.branch for condition in conditions)
        ordered_branches = tuple(branches[index] for index in order)
        ordered_costs = tuple(costs[index] for index in order)
        total_branch_counts = tuple(ordered_branches.count(name) for name in branch_names)

        prefix_candidates = [0]
        prefix_branches = [[0 for _ in branch_names]]
        for branch, (_, candidates) in zip(ordered_branches, ordered_costs, strict=True):
            prefix_candidates.append(prefix_candidates[-1] + candidates)
            counts = prefix_branches[-1].copy()
            counts[branch_index[branch]] += 1
            prefix_branches.append(counts)

        # A deliberately bounded state budget keeps a malformed pathological
        # cost table from turning a worker into an unbounded CPU search.  The
        # normal Pista window is solved by the first few branches of this
        # search; exceeding the budget falls back to the caller's explicit
        # hard failure rather than changing the batch contract.
        state_counter = [0]
        state_limit = 500_000

        @lru_cache(maxsize=None)
        def search(
            position: int,
            rank0_count: int,
            rank0_branch_counts: tuple[int, ...],
            rank0_candidates: int,
            rank0_max_nodes: int,
            rank1_max_nodes: int,
        ) -> tuple[int, ...] | None:
            state_counter[0] += 1
            if state_counter[0] > state_limit:
                return None
            if position == len(order):
                if rank0_count != per_rank_batch_groups:
                    return None
                rank1_count = position - rank0_count
                if rank1_count != per_rank_batch_groups:
                    return None
                for index, name in enumerate(branch_names):
                    if rank0_branch_counts[index] != branch_targets[name][0]:
                        return None
                    rank1_branch_count = total_branch_counts[index] - rank0_branch_counts[index]
                    if rank1_branch_count != branch_targets[name][1]:
                        return None
                return ()

            branch = ordered_branches[position]
            group_nodes, group_candidates = ordered_costs[position]
            options: list[tuple[tuple[int, int, int, int], int, tuple[int, ...], int, int, int, int]] = []
            for rank in (0, 1):
                next_rank0_count = rank0_count + (rank == 0)
                next_rank1_count = position + 1 - next_rank0_count
                if next_rank0_count > per_rank_batch_groups or next_rank1_count > per_rank_batch_groups:
                    continue
                next_rank0_branches = list(rank0_branch_counts)
                if rank == 0:
                    next_rank0_branches[branch_index[branch]] += 1
                branch_slot = branch_index[branch]
                next_rank1_branch = (
                    prefix_branches[position + 1][branch_slot]
                    - next_rank0_branches[branch_slot]
                )
                if next_rank0_branches[branch_slot] > branch_targets[branch][0]:
                    continue
                if next_rank1_branch > branch_targets[branch][1]:
                    continue
                next_rank0_candidates = rank0_candidates + (group_candidates if rank == 0 else 0)
                next_rank1_candidates = prefix_candidates[position + 1] - next_rank0_candidates
                if next_rank0_candidates > self.max_candidates_per_batch:
                    continue
                if next_rank1_candidates > self.max_candidates_per_batch:
                    continue
                next_rank0_max_nodes = max(rank0_max_nodes, group_nodes) if rank == 0 else rank0_max_nodes
                next_rank1_max_nodes = max(rank1_max_nodes, group_nodes) if rank == 1 else rank1_max_nodes
                if self.max_padded_graph_cells_per_batch is not None:
                    if next_rank0_candidates * next_rank0_max_nodes * next_rank0_max_nodes > self.max_padded_graph_cells_per_batch:
                        continue
                    if next_rank1_candidates * next_rank1_max_nodes * next_rank1_max_nodes > self.max_padded_graph_cells_per_batch:
                        continue
                remaining_groups = len(order) - (position + 1)
                if (per_rank_batch_groups - next_rank0_count) + (per_rank_batch_groups - next_rank1_count) != remaining_groups:
                    continue
                feasible_remaining = True
                for index, name in enumerate(branch_names):
                    remaining_branch = total_branch_counts[index] - prefix_branches[position + 1][index]
                    need0 = branch_targets[name][0] - next_rank0_branches[index]
                    need1 = branch_targets[name][1] - (
                        prefix_branches[position + 1][index] - next_rank0_branches[index]
                    )
                    if need0 < 0 or need1 < 0 or need0 + need1 != remaining_branch:
                        feasible_remaining = False
                        break
                if not feasible_remaining:
                    continue
                padded0 = next_rank0_candidates * next_rank0_max_nodes * next_rank0_max_nodes
                padded1 = next_rank1_candidates * next_rank1_max_nodes * next_rank1_max_nodes
                score = (
                    padded0 if rank == 0 else padded1,
                    next_rank0_candidates if rank == 0 else next_rank1_candidates,
                    next_rank0_count if rank == 0 else next_rank1_count,
                    rank,
                )
                options.append(
                    (
                        score,
                        rank,
                        tuple(next_rank0_branches),
                        next_rank0_candidates,
                        next_rank0_max_nodes,
                        next_rank1_max_nodes,
                        next_rank1_candidates,
                    )
                )
            options.sort(key=lambda option: option[0])
            for _, rank, next_branches, next_candidates, next_max0, next_max1, _ in options:
                tail = search(
                    position + 1,
                    rank0_count + (rank == 0),
                    next_branches,
                    next_candidates,
                    next_max0,
                    next_max1,
                )
                if tail is not None:
                    return (rank,) + tail
            return None

        assignment = search(0, 0, tuple(0 for _ in branch_names), 0, 0, 0)
        if assignment is None:
            return None
        lanes: list[list[RuntimeGuidanceCondition]] = [[], []]
        for position, rank in enumerate(assignment):
            lanes[rank].append(conditions[order[position]])
        if any(len(lane) != per_rank_batch_groups for lane in lanes):
            return None
        return tuple(tuple(lane) for lane in lanes)

    def _cost_balanced_partition(
        self,
        conditions: Sequence[RuntimeGuidanceCondition],
        costs: Sequence[tuple[int, int]],
        *,
        world_size: int,
        per_rank_batch_groups: int,
    ) -> tuple[tuple[RuntimeGuidanceCondition, ...], ...]:
        """Partition one global draw window into deterministic cost-balanced lanes.

        Conditions are sorted by dense padded cost and assigned to the
        currently cheapest feasible lane.  For the active two-rank profile a
        bounded deterministic backtracking pass repairs greedy dead ends.
        Branch quotas are enforced per lane so the contracted 1:1
        metric/structure mix is not changed by the partitioning.  The
        assignment depends only on resolved cache metadata, making every DDP
        rank able to reproduce its own lane without a coordination collective.
        """

        if len(conditions) != world_size * per_rank_batch_groups:
            raise ValueError("global condition count does not match rank topology")
        if len(costs) != len(conditions):
            raise ValueError("condition costs must align with conditions")
        branches = tuple(condition.group.branch for condition in conditions)
        branch_targets: dict[str, list[int]] = {}
        for branch in sorted(set(branches)):
            count = branches.count(branch)
            base, remainder = divmod(count, world_size)
            branch_targets[branch] = [base + (rank < remainder) for rank in range(world_size)]

        order = sorted(
            range(len(conditions)),
            key=lambda index: (
                -(costs[index][1] * costs[index][0] * costs[index][0]),
                -costs[index][0],
                -costs[index][1],
                index,
            ),
        )

        def greedy_partition() -> tuple[tuple[RuntimeGuidanceCondition, ...], ...]:
            lanes: list[list[RuntimeGuidanceCondition]] = [[] for _ in range(world_size)]
            lane_branch_counts: list[dict[str, int]] = [dict() for _ in range(world_size)]
            lane_candidates = [0 for _ in range(world_size)]
            lane_max_nodes = [0 for _ in range(world_size)]

            def projected(
                rank: int, group_nodes: int, group_candidates: int
            ) -> tuple[int, int, int]:
                proposed_candidates = lane_candidates[rank] + group_candidates
                proposed_max_nodes = max(lane_max_nodes[rank], group_nodes)
                padded_cells = proposed_candidates * proposed_max_nodes * proposed_max_nodes
                return padded_cells, proposed_candidates, proposed_max_nodes

            for index in order:
                condition = conditions[index]
                group_nodes, group_candidates = costs[index]
                branch = condition.group.branch
                feasible: list[int] = []
                for rank in range(world_size):
                    if len(lanes[rank]) >= per_rank_batch_groups:
                        continue
                    if lane_branch_counts[rank].get(branch, 0) >= branch_targets[branch][rank]:
                        continue
                    padded_cells, proposed_candidates, _ = projected(
                        rank, group_nodes, group_candidates
                    )
                    if proposed_candidates > self.max_candidates_per_batch:
                        continue
                    if (
                        self.max_padded_graph_cells_per_batch is not None
                        and padded_cells > self.max_padded_graph_cells_per_batch
                    ):
                        continue
                    feasible.append(rank)
                if not feasible:
                    raise PistaCacheError(
                        "cost-balanced global batch cannot satisfy per-rank candidate/"
                        "padded-cell budgets while preserving branch quotas"
                    )
                rank = min(
                    feasible,
                    key=lambda candidate_rank: (
                        projected(candidate_rank, group_nodes, group_candidates)[0],
                        projected(candidate_rank, group_nodes, group_candidates)[1],
                        len(lanes[candidate_rank]),
                        candidate_rank,
                    ),
                )
                lanes[rank].append(condition)
                lane_branch_counts[rank][branch] = lane_branch_counts[rank].get(branch, 0) + 1
                lane_candidates[rank] += group_candidates
                lane_max_nodes[rank] = max(lane_max_nodes[rank], group_nodes)

            if any(len(lane) != per_rank_batch_groups for lane in lanes):
                raise PistaCacheError("cost-balanced partition produced an incomplete rank lane")
            return tuple(tuple(lane) for lane in lanes)

        # The greedy assignment is the historical deterministic policy and is
        # valid for the overwhelmingly common feasible window.  The exact
        # search is a repair path for greedy dead ends, not a prerequisite for
        # every optimizer step.  Keeping it behind the cheap path preserves the
        # contracted branch/candidate/padded-cell constraints while avoiding a
        # potentially large memoized search on every batch.
        try:
            return greedy_partition()
        except PistaCacheError as greedy_error:
            if world_size == 2:
                exact = self._exact_two_rank_partition(
                    conditions,
                    costs,
                    order=order,
                    branch_targets=branch_targets,
                    per_rank_batch_groups=per_rank_batch_groups,
                )
                if exact is not None:
                    return exact
            raise greedy_error

    def next_rank_balanced_batch(
        self,
        *,
        epoch: int,
        draw_index: int,
        rank: int,
        world_size: int,
        per_rank_batch_groups: int,
        device: torch.device | str | None = None,
        evaluation: bool = False,
    ) -> PistaBatchDraw:
        """Resolve one global draw window, then return this rank's cost lane.

        Recovery stores the legacy lane ordinals as ``global_cursor + rank``.
        Reconstructing the shared cursor here lets an optimized continuation
        resume an old checkpoint without replaying or dropping a rank-specific
        draw lane.
        """

        if not self.cost_balanced_rank_batching:
            raise RuntimeError("cost-balanced rank batching is disabled")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise ValueError("draw_index must be a non-negative integer")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer")
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if rank >= world_size:
            raise ValueError("rank must be smaller than world_size")
        if (
            isinstance(per_rank_batch_groups, bool)
            or not isinstance(per_rank_batch_groups, int)
            or per_rank_batch_groups < 1
        ):
            raise ValueError("per_rank_batch_groups must be a positive integer")
        if draw_index < rank:
            raise ValueError("rank draw index precedes the shared draw cursor")

        total_groups = world_size * per_rank_batch_groups
        cursor = draw_index - rank
        conditions: list[RuntimeGuidanceCondition] = []
        costs: list[tuple[int, int]] = []
        # A few groups can be individually valid but make the whole dense
        # 2-rank window impossible under the contracted padded-cell cap. Keep
        # the original branch composition, reject the deterministic highest
        # cost group, and refill it from subsequent draws. This is the same
        # explicit resource-filtering policy as next_batch's candidate-budget
        # skip, and keeps every accepted optimizer step complete and balanced.
        target_branch_counts: dict[str, int] | None = None
        maximum_attempts = max(
            total_groups * 16,
            min(self.runtime.group_count(self.split) * 2, total_groups * 128),
        )
        attempts = 0
        while True:
            while len(conditions) < total_groups and attempts < maximum_attempts:
                draw = self.runtime.sample_true(
                    split=self.split,
                    epoch=epoch,
                    draw_index=cursor,
                    evaluation=evaluation,
                )
                condition = self.runtime.condition_for_true_draw(
                    draw,
                    epoch=epoch,
                    draw_index=cursor,
                    include_shuffled=self.include_shuffled,
                    evaluation=evaluation,
                )
                cursor += 1
                attempts += 1
                branch = condition.group.branch
                if (
                    target_branch_counts is not None
                    and sum(item.group.branch == branch for item in conditions)
                    >= target_branch_counts.get(branch, 0)
                ):
                    continue
                group_candidates = self._signed_candidate_count(condition.group)
                if group_candidates > self.max_candidates_per_batch:
                    continue
                group_nodes, observed_candidates = self.materializer.condition_length_cost(condition)
                if observed_candidates != group_candidates:
                    raise PistaCacheError(
                        f"length-cost candidate count disagrees for "
                        f"{condition.group.semantic_group_uid!r}"
                    )
                conditions.append(condition)
                costs.append((group_nodes, group_candidates))

            if len(conditions) != total_groups:
                raise PistaCacheError(
                    "could not resolve a complete global cost-balanced batch of "
                    f"{total_groups} groups within the deterministic resource-filter "
                    f"attempt budget ({attempts})"
                )
            if target_branch_counts is None:
                target_branch_counts = {
                    branch: sum(item.group.branch == branch for item in conditions)
                    for branch in sorted({item.group.branch for item in conditions})
                }
            try:
                lanes = self._cost_balanced_partition(
                    conditions,
                    costs,
                    world_size=world_size,
                    per_rank_batch_groups=per_rank_batch_groups,
                )
            except PistaCacheError:
                # Reject only the group with the largest dense contribution;
                # branch quotas remain fixed and the next loop refills that
                # same branch. The draw cursor is intentionally not rewound.
                drop_index = max(
                    range(len(conditions)),
                    key=lambda index: (
                        costs[index][1] * costs[index][0] * costs[index][0],
                        costs[index][0],
                        costs[index][1],
                        -index,
                    ),
                )
                conditions.pop(drop_index)
                costs.pop(drop_index)
                continue
            selected = lanes[rank]
            logical_padded_graph_cells = None
            if bool(getattr(self.materializer, "graph_free_projector_only", False)):
                selected_costs = tuple(
                    self.materializer.condition_length_cost(condition)
                    for condition in selected
                )
                logical_padded_graph_cells = (
                    sum(cost[1] for cost in selected_costs)
                    * max(cost[0] for cost in selected_costs) ** 2
                )
            return PistaBatchDraw(
                batch=self.materializer.materialize(
                    selected,
                    include_shuffled=self.include_shuffled,
                    device=device,
                ),
                conditions=selected,
                next_draw_index=cursor + rank,
                logical_padded_graph_cells=logical_padded_graph_cells,
            )

    def _select_bounded_conditions(
        self,
        *,
        epoch: int,
        draw_index: int,
        draw_stride: int,
        evaluation: bool,
        groups_per_lane: int,
        max_candidates_per_lane: int,
        max_padded_graph_cells_per_lane: int | None,
    ) -> tuple[tuple[RuntimeGuidanceCondition, ...], int]:
        """Select one historical resource-bounded lane without materializing it."""

        selected: list[RuntimeGuidanceCondition] = []
        candidates = 0
        max_nodes = 0
        cursor = draw_index
        attempts = 0
        maximum_attempts = max(
            self.runtime.group_count(self.split) * 2,
            groups_per_lane * 8,
        )
        while len(selected) < groups_per_lane and attempts < maximum_attempts:
            draw = self.runtime.sample_true(
                split=self.split,
                epoch=epoch,
                draw_index=cursor,
                evaluation=evaluation,
            )
            condition = self.runtime.condition_for_true_draw(
                draw,
                epoch=epoch,
                draw_index=cursor,
                include_shuffled=self.include_shuffled,
                evaluation=evaluation,
            )
            cursor += draw_stride
            attempts += 1
            group_candidates = self._signed_candidate_count(condition.group)
            if group_candidates > max_candidates_per_lane:
                raise PistaCacheError(
                    f"semantic group {condition.group.semantic_group_uid!r} has "
                    f"{group_candidates} signed candidates, exceeding the configured "
                    "virtual-lane candidate cap"
                )
            group_nodes, observed_candidates = self.materializer.condition_length_cost(condition)
            if observed_candidates != group_candidates:
                raise PistaCacheError(
                    f"length-cost candidate count disagrees for "
                    f"{condition.group.semantic_group_uid!r}"
                )
            proposed_candidates = candidates + group_candidates
            proposed_max_nodes = max(max_nodes, group_nodes)
            proposed_padded_cells = proposed_candidates * proposed_max_nodes * proposed_max_nodes
            if proposed_candidates > max_candidates_per_lane:
                continue
            if (
                max_padded_graph_cells_per_lane is not None
                and selected
                and proposed_padded_cells > max_padded_graph_cells_per_lane
            ):
                continue
            selected.append(condition)
            candidates = proposed_candidates
            max_nodes = proposed_max_nodes
        if len(selected) != groups_per_lane:
            budget = max_padded_graph_cells_per_lane
            detail = "" if budget is None else f" under padded_graph_cell_budget={budget}"
            raise PistaCacheError(
                "could not select a complete length-bucketed batch of "
                f"{groups_per_lane} groups{detail}"
            )
        if not selected:
            raise PistaCacheError(
                "could not select one complete canonical group within the candidate budget"
            )
        return tuple(selected), cursor

    def next_batch(
        self,
        *,
        epoch: int,
        draw_index: int,
        draw_stride: int = 1,
        device: torch.device | str | None = None,
        evaluation: bool = False,
    ) -> PistaBatchDraw:
        """Resolve one batch, optionally merging exact historical draw lanes."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise ValueError("draw_index must be a non-negative integer")
        if isinstance(draw_stride, bool) or not isinstance(draw_stride, int) or draw_stride < 1:
            raise ValueError("draw_stride must be a positive integer")

        merge = self.virtual_microbatch_merge
        lane_groups = self.max_groups_per_batch if merge is None else merge.lane_groups
        lane_candidates = (
            self.max_candidates_per_batch
            if merge is None
            else merge.lane_max_candidates
        )
        lane_cells = (
            self.max_padded_graph_cells_per_batch
            if merge is None
            else merge.lane_max_padded_graph_cells
        )
        lane_count = 1 if merge is None else merge.merge_lanes
        selected: list[RuntimeGuidanceCondition] = []
        virtual_group_ptr = [0]
        cursor = draw_index
        for _ in range(lane_count):
            lane, cursor = self._select_bounded_conditions(
                epoch=epoch,
                draw_index=cursor,
                draw_stride=draw_stride,
                evaluation=evaluation,
                groups_per_lane=lane_groups,
                max_candidates_per_lane=lane_candidates,
                max_padded_graph_cells_per_lane=lane_cells,
            )
            selected.extend(lane)
            virtual_group_ptr.append(len(selected))

        selected_costs = tuple(
            self.materializer.condition_length_cost(condition)
            for condition in selected
        )
        candidate_count = sum(cost[1] for cost in selected_costs)
        max_nodes = max(cost[0] for cost in selected_costs)
        padded_graph_cells = candidate_count * max_nodes * max_nodes
        if candidate_count > self.max_candidates_per_batch:
            raise PistaCacheError("merged virtual lanes exceed max_candidates_per_batch")
        if (
            self.max_padded_graph_cells_per_batch is not None
            and padded_graph_cells > self.max_padded_graph_cells_per_batch
        ):
            raise PistaCacheError(
                "merged virtual lanes exceed max_padded_graph_cells_per_batch"
            )

        batch = self.materializer.materialize(
            selected,
            include_shuffled=self.include_shuffled,
            device=device,
        )
        if merge is not None:
            batch = replace(
                batch,
                virtual_microbatch_group_ptr=torch.tensor(
                    virtual_group_ptr,
                    dtype=torch.long,
                    device=batch.supervision.group_ptr.device,
                ),
            )
        return PistaBatchDraw(
            batch=batch,
            conditions=tuple(selected),
            next_draw_index=cursor,
            logical_padded_graph_cells=(
                padded_graph_cells
                if bool(getattr(self.materializer, "graph_free_projector_only", False))
                else None
            ),
        )


__all__ = [
    "DEFAULT_CPU_BATCH_WORKERS",
    "DEFAULT_MAX_OPEN_GRAPH_SHARDS",
    "PistaBatchDraw",
    "PistaCacheError",
    "PistaRouteGraphStore",
    "PistaRouteSemanticMemory",
    "PistaSemanticRouteCache",
    "PistaStageABatchProvider",
    "PistaTextTokenMapping",
    "PistaTokenCache",
    "PistaTokenExtensionCache",
    "PistaTokenMemory",
    "PistaVirtualMicrobatchMerge",
    "PISTA_TOKEN_EXTENSION_SCHEMA",
    "load_corpus_text_pools",
]
