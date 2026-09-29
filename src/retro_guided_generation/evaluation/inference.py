"""Known-set exact-route and frequency-ranked Top-k evaluation.

This module is deliberately post-generation only.  It consumes frozen known
sets and completed arm outputs, recomputes every metric, and never selects a
checkpoint, root, seed, or generated candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .adapters import write_frozen_known_set_manifest
from .chem import CANONICALIZATION_POLICY, canonical_reaction_smiles
from .execution import (
    select_inference_execution_profile,
    validate_cpu_layout,
    validate_cuda_layout,
)
from .io import (
    EvaluationError,
    INTEGRITY_POLICY,
    load_document,
    now_utc,
    read_rows,
    require_complete,
    require_int,
    require_text,
    resolve_path,
    write_new_json,
    write_new_text,
)


DEFAULT_TOP_K = (1, 3, 5, 10)
BASE_METRICS = (
    "validity",
    "positive_exact_rate",
    "negative_exact_rate",
    "other_valid_rate",
    "invalid_rate",
    "any_positive",
    "any_negative",
)
CATEGORIES = ("positive", "negative", "other_valid", "invalid")


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - dependency boundary
        raise ImportError(
            "known-set aggregation and paired bootstrap require numpy; "
            "install the evaluation optional dependency"
        ) from error
    return np


def normalize_top_k(values: Sequence[int] = DEFAULT_TOP_K) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EvaluationError("top_k must be a sequence of positive integers")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise EvaluationError("top_k must contain positive integers")
        result.append(value)
    if not result:
        raise EvaluationError("top_k cannot be empty")
    return tuple(sorted(set(result)))


def metric_names(top_k: Sequence[int] = DEFAULT_TOP_K) -> tuple[str, ...]:
    return BASE_METRICS + tuple(
        name
        for cutoff in normalize_top_k(top_k)
        for name in (f"freq_positive_at_{cutoff}", f"freq_negative_at_{cutoff}")
    )


def _canonical_known_set(
    values: Any,
    *,
    label: str,
    require_canonical: bool,
) -> set[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise EvaluationError(f"{label} known set must be a sequence")
    result: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise EvaluationError(f"{label} known set contains a non-string value")
        canonical = canonical_reaction_smiles(raw)
        if canonical is None:
            raise EvaluationError(f"{label} known set contains invalid SMILES {raw!r}")
        if require_canonical and canonical != raw:
            raise EvaluationError(f"{label} known set contains non-canonical SMILES")
        result.add(canonical)
    if not result:
        raise EvaluationError(f"{label} known set cannot be empty")
    return result


def score_generated_group(
    generated_smiles: Sequence[str | None],
    positive_target_smiles: Sequence[str],
    negative_target_smiles: Sequence[str],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    expected_draw_count: int | None = None,
    require_canonical_known_sets: bool = False,
    require_canonical_generated: bool = False,
) -> dict[str, Any]:
    """Recompute exact-route and frequency-ranked metrics for one group."""

    if isinstance(generated_smiles, (str, bytes)) or not isinstance(generated_smiles, Sequence):
        raise EvaluationError("generated_smiles must be a sequence")
    if not generated_smiles:
        raise EvaluationError("generated_smiles cannot be empty")
    if expected_draw_count is not None:
        require_int(expected_draw_count, "expected_draw_count", minimum=1)
        if len(generated_smiles) != expected_draw_count:
            raise EvaluationError(
                f"trajectory count {len(generated_smiles)} differs from {expected_draw_count}"
            )
    positives = _canonical_known_set(
        positive_target_smiles,
        label="positive",
        require_canonical=require_canonical_known_sets,
    )
    negatives = _canonical_known_set(
        negative_target_smiles,
        label="negative",
        require_canonical=require_canonical_known_sets,
    )
    if positives.intersection(negatives):
        raise EvaluationError("positive and negative known sets overlap")

    canonical_generated: list[str | None] = []
    for raw in generated_smiles:
        if raw is not None and not isinstance(raw, str):
            raise EvaluationError("each generated trajectory must be a SMILES string or null")
        canonical = canonical_reaction_smiles(raw)
        if require_canonical_generated and raw is not None and canonical != raw:
            raise EvaluationError("generated output is not canonical isomeric SMILES or null")
        canonical_generated.append(canonical)

    return _score_precanonical_generated_group(
        canonical_generated,
        positives,
        negatives,
        top_k=top_k,
        include_unique_support_recall=False,
    )


def _score_precanonical_generated_group(
    canonical_generated: Sequence[str | None],
    positives: set[str],
    negatives: set[str],
    *,
    top_k: Sequence[int],
    include_unique_support_recall: bool,
) -> dict[str, Any]:
    """Score already-canonical generated endpoints against canonical known sets."""

    categories = [
        (
            "invalid"
            if canonical is None
            else "positive"
            if canonical in positives
            else "negative"
            if canonical in negatives
            else "other_valid"
        )
        for canonical in canonical_generated
    ]

    counts = Counter(categories)
    draws = len(canonical_generated)
    endpoint_counts = Counter(value for value in canonical_generated if value is not None)
    ranked = sorted(endpoint_counts.items(), key=lambda item: (-item[1], item[0]))
    ranked_smiles = [smiles for smiles, _count in ranked]
    metrics: dict[str, float] = {
        "validity": 1.0 - counts["invalid"] / draws,
        "positive_exact_rate": counts["positive"] / draws,
        "negative_exact_rate": counts["negative"] / draws,
        "other_valid_rate": counts["other_valid"] / draws,
        "invalid_rate": counts["invalid"] / draws,
        "any_positive": float(counts["positive"] > 0),
        "any_negative": float(counts["negative"] > 0),
    }
    for cutoff in normalize_top_k(top_k):
        prefix = ranked_smiles[:cutoff]
        metrics[f"freq_positive_at_{cutoff}"] = float(any(value in positives for value in prefix))
        metrics[f"freq_negative_at_{cutoff}"] = float(any(value in negatives for value in prefix))
    if include_unique_support_recall:
        generated_endpoints = set(endpoint_counts)
        metrics["positive_unique_support_recall"] = len(
            generated_endpoints.intersection(positives)
        ) / len(positives)
        metrics["negative_unique_support_recall"] = len(
            generated_endpoints.intersection(negatives)
        ) / len(negatives)
    return {
        "draw_count": draws,
        "canonical_generated_smiles": canonical_generated,
        "categories": categories,
        "category_counts": {name: int(counts[name]) for name in CATEGORIES},
        "ranked_valid_endpoints": [
            {"canonical_smiles": smiles, "count": count} for smiles, count in ranked
        ],
        "metrics": metrics,
    }


def score_precanonical_generated_group(
    canonical_generated_smiles: Sequence[str | None],
    positive_target_smiles: Sequence[str],
    negative_target_smiles: Sequence[str],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    expected_draw_count: int | None = None,
    require_canonical_known_sets: bool = True,
    include_unique_support_recall: bool = False,
) -> dict[str, Any]:
    """Score outputs that a caller has already validated as canonical.

    This is intended for audited multi-view rescoring: generated endpoints are
    canonicalized once, while each frozen target view is still independently
    validated and classified.  Passing non-canonical generated strings is a
    caller error; target sets retain the normal canonical validation gate.
    """

    if isinstance(canonical_generated_smiles, (str, bytes)) or not isinstance(
        canonical_generated_smiles, Sequence
    ):
        raise EvaluationError("canonical_generated_smiles must be a sequence")
    if not canonical_generated_smiles:
        raise EvaluationError("canonical_generated_smiles cannot be empty")
    if expected_draw_count is not None:
        require_int(expected_draw_count, "expected_draw_count", minimum=1)
        if len(canonical_generated_smiles) != expected_draw_count:
            raise EvaluationError(
                f"trajectory count {len(canonical_generated_smiles)} differs from "
                f"{expected_draw_count}"
            )
    canonical_generated: list[str | None] = []
    for raw in canonical_generated_smiles:
        if raw is not None and not isinstance(raw, str):
            raise EvaluationError(
                "each canonical generated trajectory must be a SMILES string or null"
            )
        canonical_generated.append(raw)
    positives = _canonical_known_set(
        positive_target_smiles,
        label="positive",
        require_canonical=require_canonical_known_sets,
    )
    negatives = _canonical_known_set(
        negative_target_smiles,
        label="negative",
        require_canonical=require_canonical_known_sets,
    )
    if positives.intersection(negatives):
        raise EvaluationError("positive and negative known sets overlap")
    return _score_precanonical_generated_group(
        canonical_generated,
        positives,
        negatives,
        top_k=top_k,
        include_unique_support_recall=include_unique_support_recall,
    )


def _group_key(row: Mapping[str, Any]) -> tuple[str, str | int]:
    group_uid = row.get("group_uid")
    if isinstance(group_uid, str) and group_uid:
        return "group_uid", group_uid
    index = row.get("evaluation_group_index")
    if isinstance(index, int) and not isinstance(index, bool):
        return "evaluation_group_index", index
    raise EvaluationError("group record needs group_uid or evaluation_group_index")


def _index_unique(rows: Sequence[Mapping[str, Any]], *, context: str) -> dict[tuple[str, str | int], Mapping[str, Any]]:
    indexed: dict[tuple[str, str | int], Mapping[str, Any]] = {}
    for row in rows:
        key = _group_key(row)
        if key in indexed:
            raise EvaluationError(f"duplicate {context} group identity: {key}")
        indexed[key] = row
    return indexed


def _validate_schedule(
    frozen: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    arm: str,
    key: tuple[str, str | int],
    require_complete_schedule: bool,
) -> None:
    for field in (
        "evaluation_group_index",
        "legacy_group_idx",
        "root_draw_seeds",
        "root_choice_indices",
        "trajectory_seeds",
    ):
        if field in frozen or field in output:
            if field not in frozen or field not in output:
                raise EvaluationError(f"paired field {field} is missing for {arm}/{key}")
            if frozen[field] != output[field]:
                raise EvaluationError(f"paired field {field} drift for {arm}/{key}")
    if require_complete_schedule:
        for field in (
            "evaluation_group_index",
            "root_draw_seeds",
            "root_choice_indices",
            "trajectory_seeds",
        ):
            if field not in frozen or field not in output:
                raise EvaluationError(f"formal paired schedule field {field} is missing for {arm}/{key}")
        if "rc_candidates" not in frozen:
            raise EvaluationError(f"formal frozen RC candidates are missing for {arm}/{key}")
        if "rc_permutation_keys" not in output:
            raise EvaluationError(f"formal RC permutation schedule is missing for {arm}/{key}")
    choices = frozen.get("root_choice_indices")
    root_draws = frozen.get("root_draw_seeds")
    seeds = frozen.get("trajectory_seeds")
    candidates = frozen.get("rc_candidates")
    if choices is not None or root_draws is not None or seeds is not None or candidates is not None:
        if (
            not isinstance(choices, list)
            or not isinstance(root_draws, list)
            or not isinstance(seeds, list)
            or not isinstance(candidates, list)
        ):
            raise EvaluationError(f"incomplete frozen RC schedule for {arm}/{key}")
        if len(choices) != len(root_draws) or len(choices) != len(seeds) or not choices:
            raise EvaluationError(f"frozen RC draw/seed schedule is invalid for {arm}/{key}")
        generated = output.get("generated_smiles")
        if isinstance(generated, (str, bytes)) or not isinstance(generated, Sequence) or len(choices) != len(generated):
            raise EvaluationError(f"frozen RC schedule length differs from trajectories for {arm}/{key}")
        for label, values in (("root draw", root_draws), ("trajectory", seeds)):
            if len(set(values)) != len(values):
                raise EvaluationError(f"frozen {label} seeds are not unique for {arm}/{key}")
            if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in values):
                raise EvaluationError(f"frozen {label} seeds are not integers for {arm}/{key}")
        if not candidates:
            raise EvaluationError(f"frozen RC candidate list is empty for {arm}/{key}")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise EvaluationError(f"frozen RC candidate is not an object for {arm}/{key}")
            if not isinstance(candidate.get("rc_permutation_key"), str) or not candidate["rc_permutation_key"]:
                raise EvaluationError(f"frozen RC candidate key is invalid for {arm}/{key}")
            if require_complete_schedule:
                if candidate.get("rc_slot_index") != candidate_index:
                    raise EvaluationError(f"frozen RC candidate slot drift for {arm}/{key}")
                rank = candidate.get("predictor_rank")
                score = candidate.get("predictor_score")
                permutation = candidate.get("permutation")
                if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
                    raise EvaluationError(f"frozen RC candidate rank is invalid for {arm}/{key}")
                if (
                    isinstance(score, bool)
                    or not isinstance(score, (int, float))
                    or not math.isfinite(float(score))
                ):
                    raise EvaluationError(f"frozen RC candidate score is invalid for {arm}/{key}")
                if (
                    not isinstance(permutation, list)
                    or not permutation
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in permutation)
                    or sorted(permutation) != list(range(len(permutation)))
                ):
                    raise EvaluationError(f"frozen RC candidate permutation is invalid for {arm}/{key}")
                if not isinstance(candidate.get("is_padding_duplicate"), bool):
                    raise EvaluationError(f"frozen RC padding marker is invalid for {arm}/{key}")
        for choice in choices:
            if isinstance(choice, bool) or not isinstance(choice, int) or choice < 0 or choice >= len(candidates):
                raise EvaluationError(f"frozen root choice is outside RC candidates for {arm}/{key}")
    if "rc_permutation_keys" in output:
        if not isinstance(choices, list) or not isinstance(candidates, list):
            raise EvaluationError(f"cannot reconstruct RC schedule for {arm}/{key}")
        try:
            expected = [candidates[index]["rc_permutation_key"] for index in choices]
        except (IndexError, KeyError, TypeError) as error:
            raise EvaluationError(f"invalid frozen RC schedule for {arm}/{key}") from error
        if output["rc_permutation_keys"] != expected:
            raise EvaluationError(f"RC permutation schedule drift for {arm}/{key}")


def score_arm_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    generated_rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    expected_draw_count: int | None = None,
    require_reported_categories: bool = False,
    require_canonical_known_sets: bool = False,
    require_canonical_generated: bool = False,
    validate_identity_and_schedule: bool = True,
    require_complete_schedule: bool = False,
) -> list[dict[str, Any]]:
    """Join a single arbitrary arm to every frozen group and score it."""

    frozen_by_key = _index_unique(frozen_rows, context="frozen")
    generated_by_key = _index_unique(generated_rows, context=f"{arm} generated")
    if set(frozen_by_key) != set(generated_by_key):
        missing = set(frozen_by_key).difference(generated_by_key)
        extra = set(generated_by_key).difference(frozen_by_key)
        raise EvaluationError(
            f"{arm} group coverage differs from frozen manifest; "
            f"missing={list(sorted(missing))[:3]}, extra={list(sorted(extra))[:3]}"
        )
    scored: list[dict[str, Any]] = []
    for frozen in frozen_rows:
        key = _group_key(frozen)
        output = generated_by_key[key]
        if output.get("arm") != arm:
            raise EvaluationError(f"generated row has incorrect arm for {key}: {output.get('arm')!r}")
        for field in ("case_id",):
            if output.get(field) != frozen.get(field):
                raise EvaluationError(f"{arm} row mismatches {field} for {key}")
        if validate_identity_and_schedule:
            _validate_schedule(
                frozen,
                output,
                arm=arm,
                key=key,
                require_complete_schedule=require_complete_schedule,
            )
        score = score_generated_group(
            output.get("generated_smiles"),
            frozen.get("positive_target_smiles"),
            frozen.get("negative_target_smiles"),
            top_k=top_k,
            expected_draw_count=expected_draw_count,
            require_canonical_known_sets=require_canonical_known_sets,
            require_canonical_generated=require_canonical_generated,
        )
        if require_reported_categories and output.get("categories") is None:
            raise EvaluationError(f"reported categories are missing for {arm}/{key}")
        if output.get("categories") is not None and output["categories"] != score["categories"]:
            raise EvaluationError(f"reported categories drift for {arm}/{key}")
        if require_reported_categories and output.get("category_counts") is None:
            raise EvaluationError(f"reported category counts are missing for {arm}/{key}")
        if output.get("category_counts") is not None and output["category_counts"] != score["category_counts"]:
            raise EvaluationError(f"category counts drift for {arm}/{key}")
        scored.append(
            {
                "group_uid": frozen.get("group_uid"),
                "case_id": frozen.get("case_id"),
                "metrics": score["metrics"],
                "draw_count": score["draw_count"],
            }
        )
    return scored


def _comparison_specs(
    comparisons: Sequence[Mapping[str, Any]], arms: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[tuple[str, str, str]]:
    result: list[tuple[str, str, str]] = []
    for row in comparisons:
        if not isinstance(row, Mapping):
            raise EvaluationError("each comparison must be an object")
        name = require_text(row.get("name"), "comparison.name")
        left = require_text(row.get("left"), f"comparison {name}.left")
        right = require_text(row.get("right"), f"comparison {name}.right")
        if name in {item[0] for item in result}:
            raise EvaluationError(f"duplicate comparison name {name!r}")
        if left not in arms or right not in arms:
            raise EvaluationError(f"comparison {name!r} references an unknown arm")
        result.append((name, left, right))
    return result


def _aggregate_case_matrices(
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    trajectory_counts: Mapping[str, int],
    names: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Build group means and per-case metric matrices for one fixed cohort."""

    group_count = len(frozen_rows)
    arm_names = list(scored_by_arm)
    case_order: list[str] = []
    for frozen in frozen_rows:
        case_id = require_text(frozen.get("case_id"), "frozen case_id")
        if case_id not in case_order:
            case_order.append(case_id)
    case_position = {case_id: index for index, case_id in enumerate(case_order)}
    np = _numpy()
    absolute: dict[str, Any] = {}
    case_matrices: dict[str, Any] = {}
    for arm in arm_names:
        rows = list(scored_by_arm[arm])
        if len(rows) != group_count:
            raise EvaluationError(f"scored row count differs for arm {arm!r}")
        by_group = _index_unique(rows, context=f"{arm} scored")
        ordered_rows: list[Mapping[str, Any]] = []
        for frozen in frozen_rows:
            key = _group_key(frozen)
            try:
                ordered_rows.append(by_group[key])
            except KeyError as error:
                raise EvaluationError(f"scored output missing frozen group {key}") from error
        values = np.asarray(
            [[float(row["metrics"][name]) for name in names] for row in ordered_rows],
            dtype=np.float64,
        )
        absolute[arm] = {
            "trajectory_count": int(trajectory_counts[arm]),
            "group_count": group_count,
            "metrics": {name: float(values[:, index].mean()) for index, name in enumerate(names)},
        }
        sums = np.zeros((len(case_order), len(names)), dtype=np.float64)
        counts = np.zeros(len(case_order), dtype=np.int64)
        for row, frozen in zip(ordered_rows, frozen_rows, strict=True):
            position = case_position[require_text(frozen.get("case_id"), "frozen case_id")]
            sums[position] += np.asarray([float(row["metrics"][name]) for name in names])
            counts[position] += 1
        if any(count == 0 for count in counts):
            raise EvaluationError("at least one case has no scored groups")
        case_matrices[arm] = sums / counts[:, None]
    return absolute, case_matrices, case_order


def _branch_partitions(
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]]:
    """Partition one fully paired cohort by its declared branch label."""

    frozen_by_branch: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for frozen in frozen_rows:
        frozen_by_branch[require_text(frozen.get("branch"), "frozen branch")].append(frozen)
    by_arm = {arm: _index_unique(rows, context=f"{arm} scored") for arm, rows in scored_by_arm.items()}
    result: dict[str, tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]] = {}
    for branch in sorted(frozen_by_branch):
        branch_frozen = frozen_by_branch[branch]
        branch_scored = {
            arm: [indexed[_group_key(row)] for row in branch_frozen]
            for arm, indexed in by_arm.items()
        }
        result[branch] = (branch_frozen, branch_scored)
    return result


def _branch_balanced_aggregate(
    partitions: Mapping[str, tuple[Sequence[Mapping[str, Any]], Mapping[str, Sequence[Mapping[str, Any]]]]],
    trajectory_counts: Mapping[str, int],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_batch_size: int,
    record_validation: Mapping[str, bool] | None,
) -> dict[str, Any]:
    """Equally weight branch means with a paired case bootstrap in each branch."""

    if not partitions:
        raise EvaluationError("branch-balanced aggregation needs at least one branch")
    names = metric_names(top_k)
    branch_names = list(partitions)
    first_scored = next(iter(partitions.values()))[1]
    arm_names = list(first_scored)
    comparison_specs = _comparison_specs(comparisons, first_scored)
    prepared: dict[str, tuple[dict[str, Any], Any, list[str]]] = {}
    for branch, (frozen_rows, scored_by_arm) in partitions.items():
        if list(scored_by_arm) != arm_names:
            raise EvaluationError("branch partitions disagree on arm ordering")
        prepared[branch] = _aggregate_case_matrices(
            frozen_rows,
            scored_by_arm,
            {arm: sum(int(row["draw_count"]) for row in rows) for arm, rows in scored_by_arm.items()},
            names,
        )
    absolute: dict[str, Any] = {}
    for arm in arm_names:
        absolute[arm] = {
            "trajectory_count": int(trajectory_counts[arm]),
            "group_count": sum(len(rows) for rows, _scored in partitions.values()),
            "metrics": {
                name: float(
                    sum(prepared[branch][0][arm]["metrics"][name] for branch in branch_names)
                    / len(branch_names)
                )
                for name in names
            },
        }
    comparison_report: dict[str, Any] = {}
    if comparison_specs:
        np = _numpy()
        differences: dict[str, Any] = {}
        for branch, (_absolute, case_matrices, _case_order) in prepared.items():
            differences[branch] = np.stack(
                [case_matrices[left] - case_matrices[right] for _name, left, right in comparison_specs],
                axis=0,
            )
        observed = sum(values.mean(axis=1) for values in differences.values()) / len(branch_names)
        rng = np.random.default_rng(bootstrap_seed)
        bootstrap = np.empty(
            (bootstrap_replicates, len(comparison_specs), len(names)), dtype=np.float64
        )
        for start in range(0, bootstrap_replicates, bootstrap_batch_size):
            stop = min(start + bootstrap_batch_size, bootstrap_replicates)
            sampled_total = np.zeros(
                (stop - start, len(comparison_specs), len(names)), dtype=np.float64
            )
            for branch in branch_names:
                difference = differences[branch]
                draw = rng.integers(
                    0,
                    difference.shape[1],
                    size=(stop - start, difference.shape[1]),
                )
                sampled = np.take(difference, draw, axis=1).mean(axis=2)
                sampled_total += np.moveaxis(sampled, 0, 1)
            bootstrap[start:stop] = sampled_total / len(branch_names)
        lower = np.percentile(bootstrap, 2.5, axis=0)
        upper = np.percentile(bootstrap, 97.5, axis=0)
        for index, (name, left, right) in enumerate(comparison_specs):
            comparison_report[name] = {
                "left": left,
                "right": right,
                "case_count_by_branch": {
                    branch: len(prepared[branch][2]) for branch in branch_names
                },
                "observed": {metric: float(observed[index, column]) for column, metric in enumerate(names)},
                "ci95_low": {metric: float(lower[index, column]) for column, metric in enumerate(names)},
                "ci95_high": {metric: float(upper[index, column]) for column, metric in enumerate(names)},
            }
    return {
        "schema_version": "rgg_known_set_branch_balanced_evaluation.v1",
        "canonical_smiles_policy": CANONICALIZATION_POLICY,
        "top_k": list(normalize_top_k(top_k)),
        "metric_names": list(names),
        "group_count": sum(len(rows) for rows, _scored in partitions.values()),
        "branch_names": branch_names,
        "arm_names": arm_names,
        "absolute": absolute,
        "comparisons": comparison_report,
        "aggregation": {
            "branch_weighting": "equal_branch_group_mean",
            "difference_weighting": "equal_branch_case_cluster_mean",
        },
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "batch_size": bootstrap_batch_size,
            "cluster_unit": "case_id",
            "paired": True,
            "resampling": "within_branch_case_clusters",
            "shared_resamples_across_comparisons": True,
        },
        "record_validation": dict(record_validation or {}),
        "integrity_policy": INTEGRITY_POLICY,
    }


def aggregate_scored_arms(
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    trajectory_counts: Mapping[str, int],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    bootstrap_batch_size: int = 64,
    record_validation: Mapping[str, bool] | None = None,
    include_branch_summaries: bool = False,
) -> dict[str, Any]:
    """Produce group means and case-clustered paired bootstrap comparisons."""

    if not frozen_rows or not scored_by_arm:
        raise EvaluationError("frozen rows and scored arms cannot be empty")
    require_int(bootstrap_replicates, "bootstrap.replicates", minimum=1)
    require_int(bootstrap_batch_size, "bootstrap.batch_size", minimum=1)
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise EvaluationError("bootstrap.seed must be an integer")
    names = metric_names(top_k)
    arm_names = list(scored_by_arm)
    group_count = len(frozen_rows)
    np = _numpy()
    absolute, case_matrices, case_order = _aggregate_case_matrices(
        frozen_rows,
        scored_by_arm,
        trajectory_counts,
        names,
    )

    comparison_specs = _comparison_specs(comparisons, scored_by_arm)
    comparison_report: dict[str, Any] = {}
    if comparison_specs:
        difference_stack = np.stack(
            [case_matrices[left] - case_matrices[right] for _name, left, right in comparison_specs],
            axis=0,
        )
        rng = np.random.default_rng(bootstrap_seed)
        bootstrap = np.empty(
            (bootstrap_replicates, len(comparison_specs), len(names)), dtype=np.float64
        )
        for start in range(0, bootstrap_replicates, bootstrap_batch_size):
            stop = min(start + bootstrap_batch_size, bootstrap_replicates)
            draw = rng.integers(0, len(case_order), size=(stop - start, len(case_order)))
            # take() yields comparison x replicate x case x metric.
            sampled = np.take(difference_stack, draw, axis=1).mean(axis=2)
            bootstrap[start:stop] = np.moveaxis(sampled, 0, 1)
        observed = difference_stack.mean(axis=1)
        lower = np.percentile(bootstrap, 2.5, axis=0)
        upper = np.percentile(bootstrap, 97.5, axis=0)
        for index, (name, left, right) in enumerate(comparison_specs):
            comparison_report[name] = {
                "left": left,
                "right": right,
                "case_count": len(case_order),
                "observed": {metric: float(observed[index, column]) for column, metric in enumerate(names)},
                "ci95_low": {metric: float(lower[index, column]) for column, metric in enumerate(names)},
                "ci95_high": {metric: float(upper[index, column]) for column, metric in enumerate(names)},
            }
    result = {
        "schema_version": "rgg_known_set_evaluation.v1",
        "canonical_smiles_policy": CANONICALIZATION_POLICY,
        "top_k": list(normalize_top_k(top_k)),
        "metric_names": list(names),
        "group_count": group_count,
        "case_count": len(case_order),
        "arm_names": arm_names,
        "absolute": absolute,
        "comparisons": comparison_report,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "batch_size": bootstrap_batch_size,
            "cluster_unit": "case_id",
            "paired": True,
            "shared_resamples_across_comparisons": True,
        },
        "record_validation": dict(record_validation or {}),
        "integrity_policy": INTEGRITY_POLICY,
    }
    if include_branch_summaries:
        partitions = _branch_partitions(frozen_rows, scored_by_arm)
        branch_reports: dict[str, Any] = {}
        for branch, (branch_frozen, branch_scored) in partitions.items():
            branch_reports[branch] = aggregate_scored_arms(
                branch_frozen,
                branch_scored,
                {arm: sum(int(row["draw_count"]) for row in rows) for arm, rows in branch_scored.items()},
                comparisons,
                top_k=top_k,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
                bootstrap_batch_size=bootstrap_batch_size,
                record_validation=record_validation,
                include_branch_summaries=False,
            )
        result["branch_summaries"] = branch_reports
        result["branch_balanced"] = _branch_balanced_aggregate(
            partitions,
            trajectory_counts,
            comparisons,
            top_k=top_k,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            bootstrap_batch_size=bootstrap_batch_size,
            record_validation=record_validation,
        )
    return result


def _axis_partitions(
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    axis_fields: Sequence[str],
    branch: str | None,
) -> dict[str, tuple[list[Mapping[str, Any]], dict[str, list[Mapping[str, Any]]]]]:
    """Partition a paired cohort by one or more frozen categorical fields."""

    def category(row: Mapping[str, Any], field: str) -> str:
        value = row.get(field)
        if field == "polarity":
            if value in {1, "+1", "positive"}:
                return "positive"
            if value in {-1, "-1", "negative"}:
                return "negative"
        if isinstance(value, str):
            return require_text(value, f"frozen {field}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationError(
                f"frozen categorical field {field!r} must be text or numeric"
            )
        return str(value)

    def partition_name(row: Mapping[str, Any]) -> str:
        values = [category(row, field) for field in axis_fields]
        return values[0] if len(values) == 1 else "::".join(values)

    frozen_by_axis: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for frozen in frozen_rows:
        if branch is not None and require_text(frozen.get("branch"), "frozen branch") != branch:
            continue
        frozen_by_axis[partition_name(frozen)].append(frozen)
    if not frozen_by_axis:
        scope = f"branch {branch!r}" if branch is not None else "the complete cohort"
        raise EvaluationError(f"axis reporting found no groups in {scope}")
    by_arm = {arm: _index_unique(rows, context=f"{arm} scored") for arm, rows in scored_by_arm.items()}
    return {
        axis: (
            axis_frozen,
            {
                arm: [indexed[_group_key(row)] for row in axis_frozen]
                for arm, indexed in by_arm.items()
            },
        )
        for axis, axis_frozen in sorted(frozen_by_axis.items())
    }


def _axis_reporting_options(value: Any) -> dict[str, Any] | None:
    """Validate the optional, reusable axis-diagnostic configuration block."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise EvaluationError("axis_reporting must be an object")
    allowed = {
        "axis_field",
        "axis_fields",
        "branch",
        "primary_comparison",
        "positive_metric",
        "negative_metric",
        "minimum_group_count",
        "minimum_case_count",
    }
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise EvaluationError("axis_reporting has unknown field(s): " + ", ".join(unknown))
    raw_branch = value.get("branch")
    if raw_branch is not None and not isinstance(raw_branch, str):
        raise EvaluationError("axis_reporting.branch must be a string or null")
    raw_axis_field = value.get("axis_field")
    raw_axis_fields = value.get("axis_fields")
    if raw_axis_field is not None and raw_axis_fields is not None:
        raise EvaluationError(
            "axis_reporting.axis_field and axis_fields are mutually exclusive"
        )
    if raw_axis_fields is None:
        axis_field = require_text(
            "axis_id" if raw_axis_field is None else raw_axis_field,
            "axis_reporting.axis_field",
        )
        axis_fields = None
    else:
        if isinstance(raw_axis_fields, (str, bytes)) or not isinstance(
            raw_axis_fields, Sequence
        ):
            raise EvaluationError("axis_reporting.axis_fields must be a sequence")
        resolved_fields = tuple(
            require_text(item, "axis_reporting.axis_fields item")
            for item in raw_axis_fields
        )
        if not resolved_fields or len(set(resolved_fields)) != len(resolved_fields):
            raise EvaluationError(
                "axis_reporting.axis_fields must be non-empty and unique"
            )
        axis_field = None
        axis_fields = resolved_fields
    return {
        "axis_field": axis_field,
        "axis_fields": axis_fields,
        "branch": None if raw_branch is None else require_text(raw_branch, "axis_reporting.branch"),
        "primary_comparison": require_text(
            value.get("primary_comparison", "true_minus_shuffled"),
            "axis_reporting.primary_comparison",
        ),
        "positive_metric": require_text(
            value.get("positive_metric", "freq_positive_at_10"),
            "axis_reporting.positive_metric",
        ),
        "negative_metric": require_text(
            value.get("negative_metric", "freq_negative_at_10"),
            "axis_reporting.negative_metric",
        ),
        "minimum_group_count": require_int(
            value.get("minimum_group_count", 1),
            "axis_reporting.minimum_group_count",
            minimum=1,
        ),
        "minimum_case_count": require_int(
            value.get("minimum_case_count", 1),
            "axis_reporting.minimum_case_count",
            minimum=1,
        ),
    }


def _axis_interpretation(
    comparison: Mapping[str, Any],
    *,
    positive_metric: str,
    negative_metric: str,
) -> dict[str, Any]:
    """Classify a text-specific axis effect without collapsing its two endpoints."""

    positive_observed = float(comparison["observed"][positive_metric])
    positive_low = float(comparison["ci95_low"][positive_metric])
    positive_high = float(comparison["ci95_high"][positive_metric])
    negative_observed = float(comparison["observed"][negative_metric])
    negative_low = float(comparison["ci95_low"][negative_metric])
    negative_high = float(comparison["ci95_high"][negative_metric])
    positive_improves = positive_low > 0.0
    negative_improves = negative_high < 0.0
    positive_harms = positive_high < 0.0
    negative_harms = negative_low > 0.0
    if positive_harms or negative_harms:
        label = "learned_poorly"
    elif positive_improves and negative_improves:
        label = "learned_well"
    elif positive_improves or negative_improves:
        label = "partial_signal"
    else:
        label = "inconclusive"
    return {
        "label": label,
        "decision_rule": (
            "unadjusted_paired_ci95: learned_well requires positive CI entirely above 0 "
            "and negative CI entirely below 0; learned_poorly requires either opposite sign"
        ),
        "positive_metric": positive_metric,
        "negative_metric": negative_metric,
        "positive_observed": positive_observed,
        "positive_ci95_low": positive_low,
        "positive_ci95_high": positive_high,
        "negative_observed": negative_observed,
        "negative_ci95_low": negative_low,
        "negative_ci95_high": negative_high,
    }


def aggregate_axis_diagnostics(
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    trajectory_counts: Mapping[str, int],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    bootstrap_batch_size: int = 64,
    record_validation: Mapping[str, bool] | None = None,
    axis_field: str | None = None,
    axis_fields: Sequence[str] | None = None,
    branch: str | None = None,
    primary_comparison: str = "true_minus_shuffled",
    positive_metric: str = "freq_positive_at_10",
    negative_metric: str = "freq_negative_at_10",
    minimum_group_count: int = 1,
    minimum_case_count: int = 1,
) -> dict[str, Any]:
    """Aggregate paired results by a frozen axis without rerunning inference.

    ``primary_comparison`` is normally ``true_minus_shuffled`` so the
    classification isolates the conditional text effect rather than comparing
    a guided student directly to the frozen unconditional reference arm.
    """

    if axis_fields is None:
        resolved_axis_fields = (
            require_text(
                "axis_id" if axis_field is None else axis_field,
                "axis_reporting.axis_field",
            ),
        )
    else:
        if axis_field is not None:
            raise EvaluationError(
                "axis_reporting.axis_field and axis_fields are mutually exclusive"
            )
        if isinstance(axis_fields, (str, bytes)) or not isinstance(
            axis_fields, Sequence
        ):
            raise EvaluationError("axis_reporting.axis_fields must be a sequence")
        resolved_axis_fields = tuple(
            require_text(item, "axis_reporting.axis_fields item")
            for item in axis_fields
        )
        if not resolved_axis_fields or len(set(resolved_axis_fields)) != len(
            resolved_axis_fields
        ):
            raise EvaluationError(
                "axis_reporting.axis_fields must be non-empty and unique"
            )
    if branch is not None:
        branch = require_text(branch, "axis_reporting.branch")
    primary_comparison = require_text(primary_comparison, "axis_reporting.primary_comparison")
    positive_metric = require_text(positive_metric, "axis_reporting.positive_metric")
    negative_metric = require_text(negative_metric, "axis_reporting.negative_metric")
    require_int(minimum_group_count, "axis_reporting.minimum_group_count", minimum=1)
    require_int(minimum_case_count, "axis_reporting.minimum_case_count", minimum=1)
    names = metric_names(top_k)
    for metric in (positive_metric, negative_metric):
        if metric not in names:
            raise EvaluationError(f"axis_reporting metric is not enabled by top_k: {metric}")
    comparison_specs = _comparison_specs(comparisons, scored_by_arm)
    comparison_by_name = {
        name: {"left": left, "right": right}
        for name, left, right in comparison_specs
    }
    comparison_names = set(comparison_by_name)
    if primary_comparison not in comparison_names:
        raise EvaluationError(
            f"axis_reporting.primary_comparison is absent from comparisons: {primary_comparison!r}"
        )
    partitions = _axis_partitions(
        frozen_rows,
        scored_by_arm,
        axis_fields=resolved_axis_fields,
        branch=branch,
    )
    axis_balanced = _branch_balanced_aggregate(
        partitions,
        trajectory_counts,
        comparisons,
        top_k=top_k,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_batch_size=bootstrap_batch_size,
        record_validation=record_validation,
    )
    axis_balanced["schema_version"] = "rgg_known_set_axis_balanced_evaluation.v1"
    axis_balanced["axis_names"] = axis_balanced.pop("branch_names")
    axis_balanced["aggregation"] = {
        "axis_weighting": "equal_axis_group_mean",
        "difference_weighting": "equal_axis_case_cluster_mean",
    }
    for comparison in axis_balanced["comparisons"].values():
        comparison["case_count_by_axis"] = comparison.pop("case_count_by_branch")
    axes: dict[str, Any] = {}
    for axis, (axis_frozen, axis_scored) in partitions.items():
        axis_trajectory_counts = {
            arm: sum(int(row["draw_count"]) for row in rows)
            for arm, rows in axis_scored.items()
        }
        summary = aggregate_scored_arms(
            axis_frozen,
            axis_scored,
            axis_trajectory_counts,
            comparisons,
            top_k=top_k,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            bootstrap_batch_size=bootstrap_batch_size,
            record_validation=record_validation,
            include_branch_summaries=False,
        )
        supported = (
            summary["group_count"] >= minimum_group_count
            and summary["case_count"] >= minimum_case_count
        )
        if supported:
            interpretation = _axis_interpretation(
                summary["comparisons"][primary_comparison],
                positive_metric=positive_metric,
                negative_metric=negative_metric,
            )
        else:
            interpretation = {
                "label": "insufficient_support",
                "decision_rule": "not classified below the configured group/case support thresholds",
            }
        axes[axis] = {
            "group_count": summary["group_count"],
            "case_count": summary["case_count"],
            "trajectory_count_by_arm": axis_trajectory_counts,
            "support_status": "eligible" if supported else "insufficient_support",
            "interpretation": interpretation,
            "absolute": summary["absolute"],
            "comparisons": summary["comparisons"],
            "bootstrap": summary["bootstrap"],
        }
    priority = {
        "learned_well": 0,
        "partial_signal": 1,
        "inconclusive": 2,
        "learned_poorly": 3,
        "insufficient_support": 4,
    }
    display_order = sorted(
        axes,
        key=lambda axis: (
            priority[axes[axis]["interpretation"]["label"]],
            -float(axes[axis]["interpretation"].get("positive_observed", 0.0)),
            float(axes[axis]["interpretation"].get("negative_observed", 0.0)),
            axis,
        ),
    )
    return {
        "schema_version": "rgg_known_set_axis_diagnostics.v1",
        "axis_field": (
            resolved_axis_fields[0] if len(resolved_axis_fields) == 1 else None
        ),
        "axis_fields": list(resolved_axis_fields),
        "branch_filter": branch,
        "primary_comparison": primary_comparison,
        "primary_arms": comparison_by_name[primary_comparison],
        "decision_metrics": {
            "positive": positive_metric,
            "negative": negative_metric,
        },
        "minimum_support": {
            "group_count": minimum_group_count,
            "case_count": minimum_case_count,
        },
        "axis_count": len(axes),
        "axis_balanced": axis_balanced,
        "display_order": display_order,
        "axes": axes,
        "integrity_policy": INTEGRITY_POLICY,
    }


def evaluate_known_set(
    frozen_rows: Sequence[Mapping[str, Any]],
    arm_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    top_k: Sequence[int] = DEFAULT_TOP_K,
    expected_draw_count: int | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 42,
    bootstrap_batch_size: int = 64,
    require_reported_categories: bool = False,
    require_canonical_known_sets: bool = False,
    require_canonical_generated: bool = False,
    validate_identity_and_schedule: bool = True,
    require_complete_schedule: bool = False,
    include_branch_summaries: bool = False,
    axis_reporting: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """In-memory convenience API for bounded tests and small smoke runs."""

    scored: dict[str, list[dict[str, Any]]] = {}
    trajectory_counts: dict[str, int] = {}
    for arm, rows in arm_rows.items():
        require_text(arm, "arm name")
        scored[arm] = score_arm_rows(
            frozen_rows,
            rows,
            arm=arm,
            top_k=top_k,
            expected_draw_count=expected_draw_count,
            require_reported_categories=require_reported_categories,
            require_canonical_known_sets=require_canonical_known_sets,
            require_canonical_generated=require_canonical_generated,
            validate_identity_and_schedule=validate_identity_and_schedule,
            require_complete_schedule=require_complete_schedule,
        )
        trajectory_counts[arm] = sum(int(row["draw_count"]) for row in scored[arm])
    report = aggregate_scored_arms(
        frozen_rows,
        scored,
        trajectory_counts,
        comparisons,
        top_k=top_k,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_batch_size=bootstrap_batch_size,
        record_validation={
            "reported_categories_required_and_recomputed": require_reported_categories,
            "canonical_known_sets_required": require_canonical_known_sets,
            "canonical_generated_outputs_or_null_required": require_canonical_generated,
            "paired_identity_and_schedule_fields_checked": validate_identity_and_schedule,
            "complete_paired_schedule_fields_required": require_complete_schedule,
        },
        include_branch_summaries=include_branch_summaries,
    )
    options = _axis_reporting_options(axis_reporting)
    if options is not None:
        report["axis_diagnostics"] = aggregate_axis_diagnostics(
            frozen_rows,
            scored,
            trajectory_counts,
            comparisons,
            top_k=top_k,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            bootstrap_batch_size=bootstrap_batch_size,
            record_validation=report["record_validation"],
            **options,
        )
    return report


def _append_markdown_summary(
    lines: list[str],
    report: Mapping[str, Any],
    *,
    heading: str,
) -> None:
    lines.extend([heading, "", "### Absolute metrics", ""])
    metrics = report["metric_names"]
    header = "| arm | " + " | ".join(metrics) + " |"
    lines.extend([header, "|" + " --- |" * (len(metrics) + 1)])
    for arm in report["arm_names"]:
        values = report["absolute"][arm]["metrics"]
        lines.append("| " + arm + " | " + " | ".join(f"{values[name]:.6f}" for name in metrics) + " |")
    if report["comparisons"]:
        lines.extend(["", "### Paired case-bootstrap comparisons", ""])
        for name, value in report["comparisons"].items():
            lines.append(f"- {name}: {value['left']} − {value['right']}")


def _append_axis_diagnostics(lines: list[str], diagnostics: Mapping[str, Any]) -> None:
    """Render the high-signal axis table while retaining full values in JSON."""

    branch = diagnostics.get("branch_filter")
    scope = "all branches" if branch is None else f"branch `{branch}`"
    primary = diagnostics["primary_comparison"]
    primary_arms = diagnostics["primary_arms"]
    primary_label = f"{primary_arms['left']}−{primary_arms['right']}"
    positive = diagnostics["decision_metrics"]["positive"]
    negative = diagnostics["decision_metrics"]["negative"]
    lines.extend(
        [
            "## Axis diagnostics",
            "",
            f"Scope: {scope}; primary causal comparison: `{primary}`.",
            "",
            f"| axis | groups | cases | {primary_label} positive delta [95% CI] | {primary_label} negative delta [95% CI] | interpretation |",
            "| --- | ---: | ---: | --- | --- | --- |",
        ]
    )
    for axis in diagnostics["display_order"]:
        entry = diagnostics["axes"][axis]
        interpretation = entry["interpretation"]
        if interpretation["label"] == "insufficient_support":
            positive_text = "n/a"
            negative_text = "n/a"
        else:
            positive_text = (
                f"{interpretation['positive_observed']:+.6f} "
                f"[{interpretation['positive_ci95_low']:+.6f}, {interpretation['positive_ci95_high']:+.6f}]"
            )
            negative_text = (
                f"{interpretation['negative_observed']:+.6f} "
                f"[{interpretation['negative_ci95_low']:+.6f}, {interpretation['negative_ci95_high']:+.6f}]"
            )
        lines.append(
            "| "
            + " | ".join(
                (
                    axis,
                    str(entry["group_count"]),
                    str(entry["case_count"]),
                    positive_text,
                    negative_text,
                    interpretation["label"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Decision metrics: `{positive}` higher is better; `{negative}` lower is better.",
            "`learned_well` requires both unadjusted paired 95% CIs to have the desired sign; "
            "`learned_poorly` requires either CI to have the opposite sign. Other entries are not conclusive.",
        ]
    )


def _render_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Known-set frequency Top-k evaluation", ""]
    axis_diagnostics = report.get("axis_diagnostics")
    if isinstance(axis_diagnostics, Mapping):
        _append_axis_diagnostics(lines, axis_diagnostics)
        lines.append("")
    _append_markdown_summary(lines, report, heading="## All groups")
    branch_summaries = report.get("branch_summaries")
    if isinstance(branch_summaries, Mapping):
        lines.extend(["", "## Per-branch summaries"])
        for branch, summary in branch_summaries.items():
            if isinstance(summary, Mapping):
                lines.append("")
                _append_markdown_summary(lines, summary, heading=f"### {branch}")
    balanced = report.get("branch_balanced")
    if isinstance(balanced, Mapping):
        lines.append("")
        _append_markdown_summary(lines, balanced, heading="## Branch-balanced aggregate")
    lines.extend(["", f"Integrity policy: `{INTEGRITY_POLICY}`.", ""])
    return "\n".join(lines)


def _contract_paths(
    config: Mapping[str, Any], config_path: Path
) -> tuple[Path, dict[str, Path], Any | None]:
    root = config_path.parent
    bound_split = None
    raw_dataset_manifest = config.get("dataset_manifest")
    if raw_dataset_manifest is None:
        frozen = resolve_path(
            require_text(config.get("frozen_manifest"), "frozen_manifest"), root=root
        )
    else:
        if config.get("frozen_manifest") is not None:
            raise EvaluationError(
                "dataset_manifest and frozen_manifest are mutually exclusive"
            )
        if config.get("metric_cohort_selection") is not None:
            raise EvaluationError(
                "a bound dataset cannot also bind a runtime cohort selection"
            )
        from retro_guided_generation.data.bound_dataset import (
            BoundDatasetError,
            resolve_bound_evaluation_split,
        )

        try:
            bound_split = resolve_bound_evaluation_split(
                resolve_path(
                    require_text(raw_dataset_manifest, "dataset_manifest"), root=root
                ),
                split=require_text(config.get("dataset_split"), "dataset_split"),
            )
        except BoundDatasetError as error:
            raise EvaluationError(str(error)) from error
        frozen = bound_split.frozen_manifest
    raw_arms = config.get("arms")
    if not isinstance(raw_arms, Mapping) or not raw_arms:
        raise EvaluationError("arms must be a non-empty object")
    arms = {
        require_text(name, "arm name"): resolve_path(require_text(value, f"arms.{name}"), root=root)
        for name, value in raw_arms.items()
    }
    if len(arms) != len(raw_arms):
        raise EvaluationError("arm names must be unique")
    return frozen, arms, bound_split


def run_inference_evaluation(
    *,
    config_path: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
    output_audit: str | Path,
    completion_json: str | Path,
    bootstrap_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run the post-generation evaluator and write the completion marker last."""

    config_file = Path(config_path).expanduser().resolve(strict=True)
    config = load_document(config_file)
    stage = str(config.get("stage", "standalone"))
    if stage not in {"standalone", "smoke", "formal"}:
        raise EvaluationError("stage must be standalone, smoke, or formal")
    raw_evaluation_branch = config.get("evaluation_branch")
    evaluation_branch = (
        None
        if raw_evaluation_branch is None
        else require_text(raw_evaluation_branch, "evaluation_branch")
    )
    if evaluation_branch not in {None, "metric", "structure"}:
        raise EvaluationError("evaluation_branch must be metric or structure")
    if stage == "formal":
        smoke = config.get("smoke_audit")
        if not isinstance(smoke, str) or not smoke:
            raise EvaluationError("formal stage requires smoke_audit")
        require_complete(resolve_path(smoke, root=config_file.parent), description="smoke audit")
    branch_reporting = config.get("branch_reporting", stage == "formal")
    if not isinstance(branch_reporting, bool):
        raise EvaluationError("branch_reporting must be a boolean")
    if stage == "formal" and not branch_reporting:
        raise EvaluationError("formal stage requires branch_reporting")
    targets = [Path(item).expanduser() for item in (output_json, output_markdown, output_audit, completion_json)]
    if bootstrap_json is not None:
        targets.append(Path(bootstrap_json).expanduser())
    existing = [str(item) for item in targets if item.exists()]
    if existing:
        raise EvaluationError("refusing to overwrite existing outputs: " + ", ".join(existing))
    frozen_path, arms, bound_split = _contract_paths(config, config_file)
    frozen_rows = read_rows(frozen_path)
    bound_dataset_summary: dict[str, Any] | None = None
    if bound_split is not None:
        from retro_guided_generation.data.bound_dataset import (
            BoundDatasetError,
            validate_bound_frozen_rows,
        )

        try:
            bound_dataset_summary = validate_bound_frozen_rows(
                frozen_rows,
                binding=bound_split,
            )
        except BoundDatasetError as error:
            raise EvaluationError(str(error)) from error
    observed_frozen_branches = {
        require_text(row.get("branch"), "frozen branch") for row in frozen_rows
    }
    if stage == "formal" and evaluation_branch is None and len(observed_frozen_branches) == 1:
        evaluation_branch = next(iter(observed_frozen_branches))
    metric_cohort_summary: dict[str, Any] | None = None
    metric_selection_path: Path | None = None
    if evaluation_branch == "metric":
        from .metric_cohort import METRIC_CONDITIONS

        raw_selection = config.get("metric_cohort_selection")
        if raw_selection is not None:
            if bound_split is not None:
                raise EvaluationError(
                    "bound dataset scoring forbids a second cohort selection manifest"
                )
            from .metric_cohort import validate_metric_frozen_rows

            metric_selection_path = resolve_path(
                require_text(raw_selection, "metric_cohort_selection"),
                root=config_file.parent,
            ).resolve(strict=True)
            metric_cohort_summary = validate_metric_frozen_rows(
                frozen_rows,
                selection=load_document(metric_selection_path),
            )
        if set(arms) != set(METRIC_CONDITIONS):
            raise EvaluationError(
                "metric evaluation arms must be exactly reference, null, true, and shuffled"
            )
    elif evaluation_branch == "structure":
        wrong_branches = {
            require_text(row.get("branch"), "frozen branch")
            for row in frozen_rows
            if row.get("branch") != "structure"
        }
        if wrong_branches:
            raise EvaluationError("structure evaluation frozen manifest contains non-structure rows")
    default_required_branches = (
        [evaluation_branch]
        if evaluation_branch is not None
        else ["metric", "structure"]
        if stage == "formal"
        else []
    )
    required_branches = config.get("required_branches", default_required_branches)
    if isinstance(required_branches, (str, bytes)) or not isinstance(required_branches, Sequence):
        raise EvaluationError("required_branches must be a sequence of branch names")
    required_branch_names = [require_text(value, "required_branches item") for value in required_branches]
    if len(set(required_branch_names)) != len(required_branch_names):
        raise EvaluationError("required_branches must not contain duplicates")
    if branch_reporting:
        missing_branches = set(required_branch_names).difference(observed_frozen_branches)
        if missing_branches:
            raise EvaluationError(
                "frozen manifest lacks required branches: " + ", ".join(sorted(missing_branches))
            )
    expected_groups = config.get("expected_group_count")
    if bound_split is not None:
        if expected_groups is None:
            expected_groups = bound_split.expected_group_count
        elif expected_groups != bound_split.expected_group_count:
            raise EvaluationError(
                "expected_group_count differs from the bound dataset manifest"
            )
    if expected_groups is not None and len(frozen_rows) != require_int(expected_groups, "expected_group_count", minimum=1):
        raise EvaluationError("frozen group count differs from expected_group_count")
    cases = {require_text(row.get("case_id"), "frozen case_id") for row in frozen_rows}
    expected_cases = config.get("expected_case_count")
    if bound_split is not None:
        if expected_cases is None:
            expected_cases = bound_split.expected_case_count
        elif expected_cases != bound_split.expected_case_count:
            raise EvaluationError(
                "expected_case_count differs from the bound dataset manifest"
            )
    if expected_cases is not None and len(cases) != require_int(expected_cases, "expected_case_count", minimum=1):
        raise EvaluationError("frozen case count differs from expected_case_count")
    top_k = normalize_top_k(config.get("top_k", list(DEFAULT_TOP_K)))
    expected_draw_count = config.get("expected_draw_count")
    if expected_draw_count is not None:
        expected_draw_count = require_int(expected_draw_count, "expected_draw_count", minimum=1)
    if metric_cohort_summary is not None:
        from .metric_cohort import METRIC_DRAW_COUNT_BY_SPLIT

        protocol_draw_count = METRIC_DRAW_COUNT_BY_SPLIT[
            str(metric_cohort_summary["source_split"])
        ]
        if expected_draw_count is None:
            expected_draw_count = protocol_draw_count
        elif expected_draw_count != protocol_draw_count:
            raise EvaluationError(
                "metric expected_draw_count differs from the validation/test cohort protocol"
            )
    if bound_split is not None:
        if expected_draw_count is None:
            expected_draw_count = bound_split.expected_draw_count
        elif expected_draw_count != bound_split.expected_draw_count:
            raise EvaluationError(
                "expected_draw_count differs from the bound dataset manifest"
            )
    bootstrap = config.get("bootstrap", {})
    if not isinstance(bootstrap, Mapping):
        raise EvaluationError("bootstrap must be an object")
    axis_reporting = _axis_reporting_options(config.get("axis_reporting"))
    if evaluation_branch == "metric":
        if axis_reporting is None:
            raise EvaluationError("metric evaluation requires axis_reporting")
        metric_axis_fields = (
            tuple(axis_reporting["axis_fields"])
            if axis_reporting["axis_fields"] is not None
            else (axis_reporting["axis_field"],)
        )
        if metric_axis_fields not in {
            ("axis_id",),
            ("axis_id", "polarity"),
        } or axis_reporting["branch"] != "metric":
            raise EvaluationError(
                "metric axis_reporting must use axis_id, optionally followed by "
                "polarity, and branch=metric"
            )
    options = config.get("record_validation", {})
    if not isinstance(options, Mapping):
        raise EvaluationError("record_validation must be an object")
    def bool_option(name: str, default: bool) -> bool:
        value = options.get(name, default)
        if not isinstance(value, bool):
            raise EvaluationError(f"record_validation.{name} must be a boolean")
        return value
    minimum_paired_arm_count = require_int(
        config.get(
            "minimum_paired_arm_count",
            4 if evaluation_branch == "metric" else 3 if stage == "formal" else 1,
        ),
        "minimum_paired_arm_count",
        minimum=1,
    )
    if len(arms) < minimum_paired_arm_count:
        raise EvaluationError(
            f"evaluation has {len(arms)} arm(s), fewer than minimum_paired_arm_count={minimum_paired_arm_count}"
        )
    reference_null_parity: dict[str, Any] | None = None
    if evaluation_branch == "metric":
        from .metric_cohort import validate_reference_null_trajectory_parity

        reference_null_parity = validate_reference_null_trajectory_parity(
            read_rows(arms["reference"]),
            read_rows(arms["null"]),
        )
    scored: dict[str, list[dict[str, Any]]] = {}
    trajectory_counts: dict[str, int] = {}
    arm_audits: dict[str, Any] = {}
    for arm, source in arms.items():
        generated = read_rows(source)
        scored_rows = score_arm_rows(
            frozen_rows,
            generated,
            arm=arm,
            top_k=top_k,
            expected_draw_count=expected_draw_count,
            require_reported_categories=bool_option("require_reported_categories", stage == "formal"),
            require_canonical_known_sets=bool_option("require_canonical_known_sets", stage == "formal"),
            require_canonical_generated=bool_option("require_canonical_generated", stage == "formal"),
            validate_identity_and_schedule=bool_option("validate_identity_and_schedule", True),
            require_complete_schedule=bool_option("require_complete_schedule", stage == "formal"),
        )
        scored[arm] = scored_rows
        trajectory_counts[arm] = sum(int(row["draw_count"]) for row in scored_rows)
        arm_audits[arm] = {
            "source": str(source.resolve()),
            "group_count": len(generated),
            "trajectory_count": trajectory_counts[arm],
            "status": "passed",
        }
        del generated
    report = aggregate_scored_arms(
        frozen_rows,
        scored,
        trajectory_counts,
        config.get("comparisons", []),
        top_k=top_k,
        bootstrap_replicates=require_int(bootstrap.get("replicates", 10_000), "bootstrap.replicates", minimum=1),
        bootstrap_seed=bootstrap.get("seed", 42),
        bootstrap_batch_size=require_int(bootstrap.get("batch_size", 64), "bootstrap.batch_size", minimum=1),
        record_validation={
            "reported_categories_required_and_recomputed": bool_option("require_reported_categories", stage == "formal"),
            "canonical_known_sets_required": bool_option("require_canonical_known_sets", stage == "formal"),
            "canonical_generated_outputs_or_null_required": bool_option("require_canonical_generated", stage == "formal"),
            "paired_identity_and_schedule_fields_checked": bool_option("validate_identity_and_schedule", True),
            "complete_paired_schedule_fields_required": bool_option("require_complete_schedule", stage == "formal"),
        },
        include_branch_summaries=branch_reporting,
    )
    if axis_reporting is not None:
        report["axis_diagnostics"] = aggregate_axis_diagnostics(
            frozen_rows,
            scored,
            trajectory_counts,
            config.get("comparisons", []),
            top_k=top_k,
            bootstrap_replicates=require_int(bootstrap.get("replicates", 10_000), "bootstrap.replicates", minimum=1),
            bootstrap_seed=bootstrap.get("seed", 42),
            bootstrap_batch_size=require_int(bootstrap.get("batch_size", 64), "bootstrap.batch_size", minimum=1),
            record_validation={
                "reported_categories_required_and_recomputed": bool_option("require_reported_categories", stage == "formal"),
                "canonical_known_sets_required": bool_option("require_canonical_known_sets", stage == "formal"),
                "canonical_generated_outputs_or_null_required": bool_option("require_canonical_generated", stage == "formal"),
                "paired_identity_and_schedule_fields_checked": bool_option("validate_identity_and_schedule", True),
                "complete_paired_schedule_fields_required": bool_option("require_complete_schedule", stage == "formal"),
            },
            **axis_reporting,
        )
    audit = {
        "schema_version": "rgg_known_set_completion_audit.v1",
        "status": "passed",
        "stage": stage,
        "frozen_manifest": str(frozen_path.resolve()),
        "group_count": len(frozen_rows),
        "case_count": len(cases),
        "arms": arm_audits,
        "checks": {
            "positive_negative_sets_disjoint": True,
            "all_metrics_recomputed": True,
            "frequency_tie_break_is_canonical_smiles_ascending": True,
            "paired_schedule_checked": bool_option("validate_identity_and_schedule", True),
            "complete_paired_schedule_required": bool_option("require_complete_schedule", stage == "formal"),
            "branch_summaries_recomputed": branch_reporting,
            "branch_balanced_aggregate_recomputed": branch_reporting,
            "required_branches_present": bool(required_branch_names),
        },
        "integrity_policy": INTEGRITY_POLICY,
    }
    if bound_dataset_summary is not None:
        audit["bound_dataset"] = {
            **bound_dataset_summary,
            "manifest": str(bound_split.manifest_path),
            "selection_manifest": str(bound_split.selection_manifest),
        }
    if axis_reporting is not None:
        audit["axis_diagnostics"] = {
            "status": "passed",
            "axis_field": report["axis_diagnostics"]["axis_field"],
            "axis_fields": report["axis_diagnostics"]["axis_fields"],
            "branch_filter": report["axis_diagnostics"]["branch_filter"],
            "axis_count": report["axis_diagnostics"]["axis_count"],
            "primary_comparison": report["axis_diagnostics"]["primary_comparison"],
        }
    if metric_cohort_summary is not None:
        audit["metric_cohort"] = {
            **metric_cohort_summary,
            "selection_manifest": str(metric_selection_path),
            "conditions": list(arms),
            "expected_draw_count": expected_draw_count,
            "reference_null_trajectory_parity": reference_null_parity,
        }
    report["workflow"] = {
        "stage": stage,
        "audit_status": "passed",
        "arm_loading": "sequential",
        "raw_arm_records_retained_after_scoring": False,
        "branch_reporting": branch_reporting,
        "required_branches": required_branch_names,
        "evaluation_branch": evaluation_branch,
        "integrity_policy": INTEGRITY_POLICY,
    }
    if bound_dataset_summary is not None:
        report["workflow"]["bound_dataset"] = {
            **bound_dataset_summary,
            "manifest": str(bound_split.manifest_path),
        }
    if metric_cohort_summary is not None:
        report["workflow"]["metric_cohort"] = {
            **metric_cohort_summary,
            "selection_manifest": str(metric_selection_path),
        }
    if axis_reporting is not None:
        report["workflow"]["axis_reporting"] = axis_reporting
    write_new_json(output_json, report)
    write_new_text(output_markdown, _render_markdown(report))
    write_new_json(output_audit, audit)
    bootstrap_output: Path | None = None
    if bootstrap_json is not None:
        bootstrap_payload: dict[str, Any] = {
                "schema_version": "rgg_known_set_paired_case_bootstrap.v1",
                "stage": stage,
                "comparisons": report["comparisons"],
                "branch_summaries": {
                    branch: summary["comparisons"]
                    for branch, summary in report.get("branch_summaries", {}).items()
                },
                "branch_balanced": report.get("branch_balanced", {}).get("comparisons", {}),
                "integrity_policy": INTEGRITY_POLICY,
        }
        if axis_reporting is not None:
            bootstrap_payload["axis_diagnostics"] = report["axis_diagnostics"]
        bootstrap_output = write_new_json(bootstrap_json, bootstrap_payload)
    completion = {
        "schema_version": "rgg_known_set_completion.v1",
        "status": "complete",
        "completed_at": now_utc(),
        "stage": stage,
        "outputs": {
            "json": str(Path(output_json).expanduser().resolve()),
            "markdown": str(Path(output_markdown).expanduser().resolve()),
            "audit": str(Path(output_audit).expanduser().resolve()),
        },
        "counts": {"arms": len(arms), "groups": len(frozen_rows), "cases": len(cases)},
        "integrity_policy": INTEGRITY_POLICY,
    }
    if bootstrap_output is not None:
        completion["outputs"]["paired_case_bootstrap"] = str(bootstrap_output)
    write_new_json(completion_json, completion)
    return {"report": report, "audit": audit, "completion": completion}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rgg evaluate inference")
    commands = parser.add_subparsers(dest="command", required=True)
    frozen = commands.add_parser("build-frozen", help="adapt Guidance V3 groups into a frozen known-set manifest")
    frozen.add_argument("--guided-sets", type=Path, required=True)
    frozen.add_argument("--evaluation-mappings", type=Path, required=True)
    frozen.add_argument("--output-jsonl", type=Path, required=True)
    score = commands.add_parser("score", help="score completed arms against a frozen known set")
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--output-json", type=Path, required=True)
    score.add_argument("--output-markdown", type=Path, required=True)
    score.add_argument("--output-audit", type=Path, required=True)
    score.add_argument("--completion-json", type=Path, required=True)
    score.add_argument("--bootstrap-json", type=Path)
    execution_select = commands.add_parser(
        "select-execution-profile",
        help="select the fastest fully validated portable execution profile",
    )
    execution_select.add_argument("--benchmark-report", type=Path, required=True)
    execution_select.add_argument("--output-json", type=Path, required=True)
    cuda_layout = commands.add_parser(
        "validate-cuda-layout",
        help="validate the CUDA device layout visible to this inference process",
    )
    cuda_layout.add_argument("--expected-visible-devices", type=int, required=True)
    cpu_layout = commands.add_parser(
        "validate-cpu-layout",
        help="validate the CPU affinity visible to this inference process",
    )
    cpu_layout.add_argument("--minimum-visible-cpus", type=int, required=True)
    pista_frozen = commands.add_parser(
        "build-pista-frozen",
        help="freeze a Pista test corpus into the generic known-set manifest schema",
    )
    pista_frozen.add_argument("--training-config", type=Path, required=True)
    pista_frozen.add_argument("--rc-prediction-table", type=Path, required=True)
    pista_frozen.add_argument("--rc-predictor-checkpoint", type=Path, required=True)
    pista_frozen.add_argument("--output-jsonl", type=Path, required=True)
    pista_frozen.add_argument("--split", choices=("train", "validation", "test"), default="test")
    pista_frozen.add_argument("--draw-count", type=int, default=100)
    pista_frozen.add_argument("--rc-top-k", type=int, default=5)
    pista_frozen.add_argument("--root-draw-seed-base", type=int, default=20_260_721)
    pista_frozen.add_argument("--trajectory-seed-base", type=int, default=42_000)
    pista_frozen.add_argument("--expected-group-count", type=int)
    pista_frozen.add_argument("--expected-case-count", type=int)
    pista_frozen.add_argument(
        "--sample-group-count",
        type=int,
        help="uniformly sample this many source-order-preserving Pista groups before freezing",
    )
    pista_frozen.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="seed for --sample-group-count; ignored for a full split",
    )
    pista_validate = commands.add_parser(
        "validate-pista-frozen",
        help="fail closed unless a Pista frozen manifest has the complete RC-aware schedule",
    )
    pista_validate.add_argument("--frozen-manifest", type=Path, required=True)
    pista_validate.add_argument("--expected-draw-count", type=int, default=100)
    pista_validate.add_argument("--expected-rc-top-k", type=int, default=5)
    pista_validate.add_argument("--root-draw-seed-base", type=int, default=20_260_721)
    pista_validate.add_argument("--trajectory-seed-base", type=int, default=42_000)
    pista_smoke = commands.add_parser(
        "select-pista-smoke",
        help="select a deterministic branch-stratified Pista smoke manifest",
    )
    pista_smoke.add_argument("--frozen-manifest", type=Path, required=True)
    pista_smoke.add_argument("--output-jsonl", type=Path, required=True)
    pista_smoke.add_argument("--metric-count", type=int, default=48)
    pista_smoke.add_argument("--structure-count", type=int, default=16)
    pista_partition = commands.add_parser(
        "partition-pista-frozen",
        help="write a deterministic Pista early cohort and its exact remainder",
    )
    pista_partition.add_argument("--frozen-manifest", type=Path, required=True)
    pista_partition.add_argument("--selected-output-jsonl", type=Path, required=True)
    pista_partition.add_argument("--remaining-output-jsonl", type=Path, required=True)
    pista_partition.add_argument("--metric-count", type=int, required=True)
    pista_partition.add_argument("--structure-count", type=int, required=True)
    pista_generate = commands.add_parser(
        "generate-pista-arm",
        help="generate one raw Pista arm shard for the generic scorer",
    )
    pista_generate.add_argument("--training-config", type=Path, required=True)
    pista_generate.add_argument("--checkpoint", type=Path, required=True)
    pista_generate.add_argument("--frozen-manifest", type=Path, required=True)
    pista_generate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    pista_generate.add_argument("--expected-draw-count", type=int, default=100)
    pista_generate.add_argument("--root-draw-seed-base", type=int, default=20_260_721)
    pista_generate.add_argument("--trajectory-seed-base", type=int, default=42_000)
    pista_generate.add_argument(
        "--arm",
        choices=("reference", "true", "shuffled", "null"),
        required=True,
    )
    pista_generate.add_argument("--output-jsonl", type=Path, required=True)
    pista_generate.add_argument("--sampling-steps", type=int, default=50)
    pista_generate.add_argument(
        "--guidance-alpha",
        type=float,
        default=1.0,
        help="scale the guided residual in [0,1]; reference arms ignore it",
    )
    pista_generate.add_argument("--execution-profile", type=Path)
    pista_generate.add_argument("--trajectory-batch", type=int)
    pista_generate.add_argument("--shard-index", type=int, default=0)
    pista_generate.add_argument("--shard-count", type=int, default=1)
    pista_generate.add_argument("--device", default="cuda")
    pista_generate.add_argument("--vocabulary", type=Path)
    pista_generate.add_argument("--vocabulary-id")
    pista_generate.add_argument("--progress-every-groups", type=int)
    pista_generate.add_argument("--progress-every-trajectory-batches", type=int)
    pista_generate_session = commands.add_parser(
        "generate-pista-arms-session",
        help="generate sequential Pista arms while reusing one loaded CUDA runtime/model",
    )
    pista_generate_session.add_argument("--training-config", type=Path, required=True)
    pista_generate_session.add_argument("--checkpoint", type=Path, required=True)
    pista_generate_session.add_argument("--frozen-manifest", type=Path, required=True)
    pista_generate_session.add_argument(
        "--split", choices=("train", "validation", "test"), default="test"
    )
    pista_generate_session.add_argument("--expected-draw-count", type=int, default=100)
    pista_generate_session.add_argument(
        "--root-draw-seed-base", type=int, default=20_260_721
    )
    pista_generate_session.add_argument(
        "--trajectory-seed-base", type=int, default=42_000
    )
    pista_generate_session.add_argument(
        "--arm-output",
        action="append",
        required=True,
        metavar="ARM=PATH",
        help=(
            "one distinct output for reference, true, shuffled, or null; "
            "repeat for each sequential arm"
        ),
    )
    pista_generate_session.add_argument("--sampling-steps", type=int, default=50)
    pista_generate_session.add_argument(
        "--guidance-alpha",
        type=float,
        default=1.0,
        help="scale every guided arm residual in [0,1]",
    )
    pista_generate_session.add_argument("--execution-profile", type=Path)
    pista_generate_session.add_argument("--trajectory-batch", type=int)
    pista_generate_session.add_argument("--shard-index", type=int, default=0)
    pista_generate_session.add_argument("--shard-count", type=int, default=1)
    pista_generate_session.add_argument("--device", default="cuda")
    pista_generate_session.add_argument("--vocabulary", type=Path)
    pista_generate_session.add_argument("--vocabulary-id")
    pista_generate_session.add_argument("--progress-every-groups", type=int)
    pista_generate_session.add_argument("--progress-every-trajectory-batches", type=int)
    pista_merge = commands.add_parser(
        "merge-pista-arm-shards",
        help="merge disjoint Pista arm shards in frozen group-index order",
    )
    pista_merge.add_argument(
        "--arm",
        choices=("reference", "true", "shuffled", "null"),
        required=True,
    )
    pista_merge.add_argument("--input-jsonl", type=Path, action="append", required=True)
    pista_merge.add_argument("--output-jsonl", type=Path, required=True)
    pista_merge.add_argument("--expected-group-count", type=int)
    pista_merge.add_argument(
        "--frozen-manifest",
        type=Path,
        help="validate exact index/group coverage against a (possibly sparse) frozen manifest",
    )
    pista_alpha_zero_parity = commands.add_parser(
        "audit-pista-alpha-zero-parity",
        help="require Reference/Null/True/Shuffled exact trajectory parity at alpha zero",
    )
    pista_alpha_zero_parity.add_argument("--reference-jsonl", type=Path, required=True)
    pista_alpha_zero_parity.add_argument("--null-jsonl", type=Path, required=True)
    pista_alpha_zero_parity.add_argument("--true-jsonl", type=Path, required=True)
    pista_alpha_zero_parity.add_argument("--shuffled-jsonl", type=Path, required=True)
    pista_alpha_zero_parity.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-frozen":
            output = write_frozen_known_set_manifest(
                guided_sets_path=args.guided_sets,
                evaluation_mappings_path=args.evaluation_mappings,
                output_jsonl=args.output_jsonl,
            )
            print(json.dumps({"output": str(output)}, ensure_ascii=False))
            return 0
        if args.command == "select-execution-profile":
            from .io import write_new_json

            selection = select_inference_execution_profile(load_document(args.benchmark_report))
            output = write_new_json(args.output_json, selection)
            print(json.dumps({"output": str(output)}, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "validate-cuda-layout":
            print(
                json.dumps(
                    validate_cuda_layout(args.expected_visible_devices),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "validate-cpu-layout":
            print(
                json.dumps(
                    validate_cpu_layout(args.minimum_visible_cpus),
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "build-pista-frozen":
            from retro_guided_generation.training.pista import load_pista_training_spec

            from .pista import write_pista_frozen_manifest

            spec = load_pista_training_spec(args.training_config)
            output = write_pista_frozen_manifest(
                corpus_root=spec.corpus_root,
                rc_prediction_table=args.rc_prediction_table,
                rc_predictor_checkpoint=args.rc_predictor_checkpoint,
                output_jsonl=args.output_jsonl,
                split=args.split,
                draw_count=args.draw_count,
                training_seed=spec.training.seed,
                evaluation_seed=spec.training.seed + 1000,
                rc_top_k=args.rc_top_k,
                root_draw_seed_base=args.root_draw_seed_base,
                trajectory_seed_base=args.trajectory_seed_base,
                expected_group_count=args.expected_group_count,
                expected_case_count=args.expected_case_count,
                sample_group_count=args.sample_group_count,
                sample_seed=args.sample_seed,
            )
            print(json.dumps({"output": str(output)}, ensure_ascii=False))
            return 0
        if args.command == "validate-pista-frozen":
            from .pista import validate_pista_frozen_rows

            result = validate_pista_frozen_rows(
                read_rows(args.frozen_manifest),
                expected_draw_count=args.expected_draw_count,
                expected_rc_top_k=args.expected_rc_top_k,
                root_draw_seed_base=args.root_draw_seed_base,
                trajectory_seed_base=args.trajectory_seed_base,
            )
            print(json.dumps({"status": "passed", **result}, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "select-pista-smoke":
            from .pista import write_pista_smoke_manifest

            output = write_pista_smoke_manifest(
                frozen_manifest=args.frozen_manifest,
                output_jsonl=args.output_jsonl,
                metric_count=args.metric_count,
                structure_count=args.structure_count,
            )
            print(json.dumps({"output": str(output)}, ensure_ascii=False))
            return 0
        if args.command == "partition-pista-frozen":
            from .pista import write_pista_frozen_partition

            outputs = write_pista_frozen_partition(
                frozen_manifest=args.frozen_manifest,
                selected_output_jsonl=args.selected_output_jsonl,
                remaining_output_jsonl=args.remaining_output_jsonl,
                metric_count=args.metric_count,
                structure_count=args.structure_count,
            )
            print(
                json.dumps(
                    {name: str(path) for name, path in outputs.items()},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "generate-pista-arm":
            from .pista import (
                DEFAULT_PISTA_VOCAB,
                DEFAULT_PISTA_VOCAB_ID,
                generate_pista_arm,
            )

            result = generate_pista_arm(
                training_config=args.training_config,
                checkpoint=args.checkpoint,
                frozen_manifest=args.frozen_manifest,
                split=args.split,
                expected_draw_count=args.expected_draw_count,
                root_draw_seed_base=args.root_draw_seed_base,
                trajectory_seed_base=args.trajectory_seed_base,
                arm=args.arm,
                output_jsonl=args.output_jsonl,
                sampling_steps=args.sampling_steps,
                trajectory_batch=args.trajectory_batch,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                device=args.device,
                execution_profile=args.execution_profile,
                vocabulary_path=DEFAULT_PISTA_VOCAB if args.vocabulary is None else args.vocabulary,
                vocabulary_id=DEFAULT_PISTA_VOCAB_ID if args.vocabulary_id is None else args.vocabulary_id,
                progress_every_groups=args.progress_every_groups,
                progress_every_trajectory_batches=args.progress_every_trajectory_batches,
                guidance_alpha=args.guidance_alpha,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "generate-pista-arms-session":
            from .pista import (
                DEFAULT_PISTA_VOCAB,
                DEFAULT_PISTA_VOCAB_ID,
                generate_pista_arms,
            )

            arm_outputs: dict[str, Path] = {}
            for raw in args.arm_output:
                arm, separator, output = raw.partition("=")
                if not separator or not arm or not output:
                    raise EvaluationError("--arm-output must use ARM=PATH")
                if arm in arm_outputs:
                    raise EvaluationError(f"duplicate Pista arm output for {arm!r}")
                arm_outputs[arm] = Path(output)
            result = generate_pista_arms(
                training_config=args.training_config,
                checkpoint=args.checkpoint,
                frozen_manifest=args.frozen_manifest,
                split=args.split,
                expected_draw_count=args.expected_draw_count,
                root_draw_seed_base=args.root_draw_seed_base,
                trajectory_seed_base=args.trajectory_seed_base,
                arm_outputs=arm_outputs,
                sampling_steps=args.sampling_steps,
                trajectory_batch=args.trajectory_batch,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
                device=args.device,
                execution_profile=args.execution_profile,
                vocabulary_path=DEFAULT_PISTA_VOCAB if args.vocabulary is None else args.vocabulary,
                vocabulary_id=DEFAULT_PISTA_VOCAB_ID if args.vocabulary_id is None else args.vocabulary_id,
                progress_every_groups=args.progress_every_groups,
                progress_every_trajectory_batches=args.progress_every_trajectory_batches,
                guidance_alpha=args.guidance_alpha,
            )
            print(json.dumps(result, ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "merge-pista-arm-shards":
            from .pista import merge_pista_arm_shards

            output = merge_pista_arm_shards(
                arm=args.arm,
                input_jsonl=args.input_jsonl,
                output_jsonl=args.output_jsonl,
                expected_group_count=args.expected_group_count,
                frozen_manifest=args.frozen_manifest,
            )
            print(json.dumps({"output": str(output)}, ensure_ascii=False))
            return 0
        if args.command == "audit-pista-alpha-zero-parity":
            from .pista import write_pista_alpha_zero_parity_audit

            output = write_pista_alpha_zero_parity_audit(
                reference_jsonl=args.reference_jsonl,
                null_jsonl=args.null_jsonl,
                true_jsonl=args.true_jsonl,
                shuffled_jsonl=args.shuffled_jsonl,
                output_json=args.output_json,
            )
            print(json.dumps({"status": "passed", "output": str(output)}, sort_keys=True))
            return 0
        result = run_inference_evaluation(
            config_path=args.config,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            output_audit=args.output_audit,
            completion_json=args.completion_json,
            bootstrap_json=args.bootstrap_json,
        )
        print(json.dumps(result["completion"], ensure_ascii=False, sort_keys=True))
        return 0
    except (EvaluationError, FileNotFoundError, ImportError) as error:
        print(f"rgg evaluate inference: {error}", file=sys.stderr)
        return 2


__all__ = [
    "DEFAULT_TOP_K",
    "aggregate_axis_diagnostics",
    "aggregate_scored_arms",
    "evaluate_known_set",
    "metric_names",
    "normalize_top_k",
    "run_inference_evaluation",
    "score_arm_rows",
    "score_generated_group",
]
