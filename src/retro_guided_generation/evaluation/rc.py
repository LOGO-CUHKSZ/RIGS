"""Frozen RC top-k plans shared by protocol-bound evaluation entry points.

The table format is intentionally small and model-agnostic: a product-condition
key maps to permutation keys and predictor scores.  Resolution is exact first,
then canonical-isomeric only when it is unambiguous.  No model is retrained,
no root is inferred from generated output, and no content hash is used.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import EvaluationError
from .rooted_product import canonical_product_smiles, product_atom_count


@dataclass(frozen=True, slots=True)
class FrozenRCResolution:
    """A canonical product's frozen ranked RC slots and their provenance."""

    canonical_product_smiles: str
    source_condition_key: str
    resolution_kind: str
    unique_available_rc_k: int
    rc_candidates: tuple[dict[str, Any], ...]


def _positive_score(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{name} must be a finite positive number")
    score = float(value)
    if not math.isfinite(score) or score <= 0.0:
        raise EvaluationError(f"{name} must be a finite positive number")
    return score


def _parse_permutation(value: object, *, atom_count: int, name: str) -> list[int]:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{name} must be a non-empty underscore-separated permutation")
    try:
        permutation = [int(part) for part in value.split("_")]
    except ValueError as error:
        raise EvaluationError(f"{name} is not an integer permutation") from error
    if len(permutation) != atom_count or sorted(permutation) != list(range(atom_count)):
        raise EvaluationError(f"{name} is not a bijection for the canonical product atom count")
    return permutation


class FrozenRCTable:
    """Resolve a read-only predictor output into protocol-ready top-k slots."""

    def __init__(self, *, source: str | Path, table: Mapping[str, Mapping[str, float]]) -> None:
        path = Path(source).expanduser().resolve(strict=True)
        normalized: dict[str, dict[str, float]] = {}
        aliases: dict[str, list[str]] = defaultdict(list)
        for raw_product, raw_weights in table.items():
            if not isinstance(raw_product, str) or not raw_product.strip():
                raise EvaluationError("frozen RC table has an empty product-condition key")
            if not isinstance(raw_weights, Mapping) or not raw_weights:
                raise EvaluationError(f"frozen RC table has no candidates for {raw_product!r}")
            product = raw_product.strip()
            weights: dict[str, float] = {}
            for raw_key, raw_score in raw_weights.items():
                if not isinstance(raw_key, str) or not raw_key:
                    raise EvaluationError(f"frozen RC table has an invalid permutation key for {product!r}")
                weights[raw_key] = _positive_score(
                    raw_score,
                    name=f"frozen RC score for {product!r}/{raw_key!r}",
                )
            normalized[product] = weights
            aliases[canonical_product_smiles(product)].append(product)
        if not normalized:
            raise EvaluationError("frozen RC table is empty")
        self.source = path
        self._table = normalized
        self._canonical_aliases = {
            canonical: tuple(sorted(keys)) for canonical, keys in aliases.items()
        }

    @classmethod
    def from_json(cls, source: str | Path) -> "FrozenRCTable":
        path = Path(source).expanduser().resolve(strict=True)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise EvaluationError(f"frozen RC prediction table is invalid JSON: {path}") from error
        if not isinstance(payload, Mapping) or set(payload) != {"predicted_rc"}:
            raise EvaluationError("frozen RC prediction table must contain only predicted_rc")
        raw_table = payload["predicted_rc"]
        if not isinstance(raw_table, Mapping):
            raise EvaluationError("frozen RC predicted_rc must be an object")
        return cls(source=path, table=raw_table)

    def _resolve_key(self, product_smiles: str) -> tuple[str, str, str]:
        canonical = canonical_product_smiles(product_smiles)
        raw = product_smiles.strip()
        if raw in self._table:
            return canonical, raw, "exact_product_condition"
        candidates = self._canonical_aliases.get(canonical, ())
        if not candidates:
            raise EvaluationError(
                "frozen RC table has no canonical product-condition entry for "
                f"{canonical!r}"
            )
        if len(candidates) == 1:
            return canonical, candidates[0], "canonical_isomeric_alias"
        reference = self._table[candidates[0]]
        if all(self._table[candidate] == reference for candidate in candidates[1:]):
            return canonical, candidates[0], "canonical_equivalent_alias"
        raise EvaluationError(
            "frozen RC table has ambiguous non-identical canonical aliases for "
            f"{canonical!r}; exact product-condition key is required"
        )

    def resolve(self, product_smiles: str, *, top_k: int = 5) -> FrozenRCResolution:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise EvaluationError("RC top_k must be a positive integer")
        canonical, source_key, resolution_kind = self._resolve_key(product_smiles)
        atom_count = product_atom_count(canonical)
        ranked = sorted(
            self._table[source_key].items(),
            key=lambda item: (-item[1], item[0]),
        )
        selected = ranked[:top_k]
        if not selected:
            raise EvaluationError(f"frozen RC table has no selectable roots for {source_key!r}")
        roots: list[dict[str, Any]] = []
        for rank, (key, score) in enumerate(selected, start=1):
            roots.append(
                {
                    "rc_slot_index": len(roots),
                    "predictor_rank": rank,
                    "rc_permutation_key": key,
                    "permutation": _parse_permutation(
                        key,
                        atom_count=atom_count,
                        name=f"RC permutation for {source_key!r}",
                    ),
                    "predictor_score": float(score),
                    "is_padding_duplicate": False,
                }
            )
        unique_count = len(roots)
        while len(roots) < top_k:
            duplicate = dict(roots[-1])
            duplicate["rc_slot_index"] = len(roots)
            duplicate["permutation"] = list(roots[-1]["permutation"])
            duplicate["is_padding_duplicate"] = True
            roots.append(duplicate)
        return FrozenRCResolution(
            canonical_product_smiles=canonical,
            source_condition_key=source_key,
            resolution_kind=resolution_kind,
            unique_available_rc_k=unique_count,
            rc_candidates=tuple(roots),
        )


__all__ = ["FrozenRCResolution", "FrozenRCTable"]
