"""Adapters from the canonical Guidance V3 corpus to evaluation manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from retro_guided_generation.guidance.io import iter_jsonl
from retro_guided_generation.guidance.schema import EvaluationMapping, GuidedMultiRouteSet

from .chem import CANONICALIZATION_POLICY, canonical_reaction_smiles
from .io import EvaluationError, INTEGRITY_POLICY, write_new_jsonl


def build_frozen_known_set_rows(
    *,
    guided_sets_path: str | Path,
    evaluation_mappings_path: str | Path,
) -> list[dict[str, Any]]:
    """Turn Guidance V3 positive/negative preferences into known-set rows.

    A semantic group owns route labels, while the enclosing guided set owns the
    route SMILES.  Keeping that join here avoids teaching the generic metric
    code about Guidance V3 internals.
    """

    mappings: dict[str, EvaluationMapping] = {}
    for raw in iter_jsonl(evaluation_mappings_path):
        mapping = EvaluationMapping.from_dict(raw)
        if mapping.semantic_group_uid in mappings:
            raise EvaluationError(
                f"duplicate EvaluationMapping for {mapping.semantic_group_uid!r}"
            )
        mappings[mapping.semantic_group_uid] = mapping

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in iter_jsonl(guided_sets_path):
        guided = GuidedMultiRouteSet.from_dict(raw)
        routes = {route.route_id: route.precursor_smiles for route in guided.case.routes}
        for group in guided.semantic_groups:
            mapping = mappings.get(group.semantic_group_uid)
            if mapping is None:
                raise EvaluationError(
                    f"missing EvaluationMapping for {group.semantic_group_uid!r}"
                )
            if group.semantic_group_uid in seen:
                raise EvaluationError(f"duplicate semantic group {group.semantic_group_uid!r}")
            seen.add(group.semantic_group_uid)
            positive: set[str] = set()
            negative: set[str] = set()
            for preference in group.candidate_preferences:
                if not preference.eligible or preference.label == 0:
                    continue
                precursor = routes.get(preference.route_id)
                canonical = canonical_reaction_smiles(precursor)
                if canonical is None:
                    raise EvaluationError(
                        f"cannot canonicalize route {preference.route_id!r} in "
                        f"{group.semantic_group_uid!r}"
                    )
                if preference.label == 1:
                    positive.add(canonical)
                elif preference.label == -1:
                    negative.add(canonical)
                else:  # Defensive: Guidance V3 validates this already.
                    raise EvaluationError("candidate preference has an unsupported label")
            if not positive or not negative:
                raise EvaluationError(
                    f"semantic group {group.semantic_group_uid!r} lacks a positive or negative known set"
                )
            overlap = positive.intersection(negative)
            if overlap:
                raise EvaluationError(
                    f"positive and negative known sets overlap for {group.semantic_group_uid!r}"
                )
            rows.append(
                {
                    "schema_version": "rgg_known_set_frozen_group.v1",
                    "group_uid": group.semantic_group_uid,
                    "case_id": group.case_id,
                    "evaluation_group_index": mapping.semantic_group_ordinal,
                    "product_smiles": guided.case.product_smiles,
                    "branch": group.branch,
                    "axis_id": group.axis_id,
                    "polarity": group.polarity,
                    "positive_target_smiles": sorted(positive),
                    "negative_target_smiles": sorted(negative),
                    "canonicalization_policy": CANONICALIZATION_POLICY,
                    "integrity_policy": INTEGRITY_POLICY,
                }
            )

    extra = set(mappings).difference(seen)
    if extra:
        raise EvaluationError(
            "EvaluationMapping contains groups absent from guided sets: "
            + repr(sorted(extra)[:5])
        )
    rows.sort(key=lambda row: int(row["evaluation_group_index"]))
    expected_indices = list(range(len(rows)))
    observed_indices = [int(row["evaluation_group_index"]) for row in rows]
    if observed_indices != expected_indices:
        raise EvaluationError("EvaluationMapping ordinals are not contiguous from zero")
    return rows


def write_frozen_known_set_manifest(
    *,
    guided_sets_path: str | Path,
    evaluation_mappings_path: str | Path,
    output_jsonl: str | Path,
) -> Path:
    rows = build_frozen_known_set_rows(
        guided_sets_path=guided_sets_path,
        evaluation_mappings_path=evaluation_mappings_path,
    )
    return write_new_jsonl(output_jsonl, rows)
