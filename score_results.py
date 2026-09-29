#!/usr/bin/env python3
"""Keyed post-generation scoring for the strict-Top100 common-support run."""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from retro_guided_generation.evaluation.chem import canonical_reaction_smiles  # noqa: E402
from retro_guided_generation.evaluation.inference import score_generated_group  # noqa: E402
from retro_guided_generation.evaluation.io import (  # noqa: E402
    INTEGRITY_POLICY,
    write_new_json,
    write_new_jsonl,
)
from retro_guided_generation.guidance.io import iter_jsonl  # noqa: E402


def configure_paths(run_root: Path) -> None:
    global CONFIG_PATH, FORMAL_MANIFEST_PATH, INVENTORY_PATH, OUTPUT_ROOT
    global REPORT_ROOT, GROUP_SCORE_ROOT, CONTROLLABILITY_PATH, REACHABILITY_PATH
    global BOOTSTRAP_PATH, FINAL_PATH, COMPLETE_PATH
    run_root = run_root.resolve(strict=True)
    prepared = json.loads((run_root / "prepared.json").read_text())
    CONFIG_PATH = run_root / "configs/formal_eval.json"
    model = json.loads(CONFIG_PATH.read_text())["models"]["top30"]
    if model["checkpoint"] != prepared["checkpoint"] or model["alpha"] != prepared["alpha"]:
        raise ValueError("Deployment and scoring configurations disagree")
    FORMAL_MANIFEST_PATH = Path(prepared["frozen_manifest"])
    INVENTORY_PATH = Path(prepared["endpoint_inventory"])
    OUTPUT_ROOT = run_root / "outputs/formal"
    REPORT_ROOT = run_root / "reports"
    GROUP_SCORE_ROOT = REPORT_ROOT / "group_scores"
    CONTROLLABILITY_PATH = REPORT_ROOT / "controllability_metrics.json"
    REACHABILITY_PATH = REPORT_ROOT / "reachability_curves.json"
    BOOTSTRAP_PATH = REPORT_ROOT / "paired_bootstrap.json"
    FINAL_PATH = REPORT_ROOT / "final_audit.json"
    COMPLETE_PATH = run_root / "runtime/SCORING_COMPLETE"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError("formal config must be an object")
    return config


def _rows(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def _write_json_equal(path: Path, payload: Mapping[str, Any]) -> Path:
    value = dict(payload)
    if path.exists():
        if _load_json(path) != value:
            raise RuntimeError(f"existing JSON differs from deterministic score: {path}")
        return path.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o775)
    return write_new_json(path, value)


def _write_rows_equal(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    values = [dict(row) for row in rows]
    if path.exists():
        if _rows(path) != values:
            raise RuntimeError(f"existing JSONL differs from deterministic score: {path}")
        return path.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o775)
    return write_new_jsonl(path, values)


def _write_text_equal(path: Path, value: str) -> Path:
    if path.exists():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"existing text differs: {path}")
        return path.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o775)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
    return path.resolve(strict=True)


def index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, raw in enumerate(rows):
        value = raw.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{label} row {position} lacks {key}")
        if value in indexed:
            raise RuntimeError(f"{label} repeats {key}={value!r}")
        indexed[value] = dict(raw)
    return indexed


def _canonical_generated(
    values: Any,
    *,
    expected_draw_count: int,
) -> list[str | None]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes))
        or len(values) != expected_draw_count
    ):
        raise RuntimeError("generated_smiles differs from the confirmed draw count")
    output: list[str | None] = []
    for raw in values:
        if raw is not None and not isinstance(raw, str):
            raise RuntimeError("generated endpoint must be a string or null")
        canonical = canonical_reaction_smiles(raw)
        if raw is not None and canonical != raw:
            raise RuntimeError("generated endpoint is not canonical or null")
        output.append(canonical)
    return output


def prefix_reachability_metrics(
    canonical_generated: Sequence[str | None],
    *,
    top100_endpoints: set[str],
    positive_endpoints: set[str],
    negative_endpoints: set[str],
    budgets: Sequence[int],
) -> dict[str, float]:
    if not top100_endpoints:
        raise RuntimeError("Top100 endpoint denominator is empty")
    if not positive_endpoints or not negative_endpoints:
        raise RuntimeError("positive or negative endpoint denominator is empty")
    if positive_endpoints.intersection(negative_endpoints):
        raise RuntimeError("positive and negative endpoint sets overlap")
    if not positive_endpoints.union(negative_endpoints).issubset(top100_endpoints):
        raise RuntimeError("semantic target endpoint is outside the Top100 inventory")
    metrics: dict[str, float] = {}
    for budget in budgets:
        if budget < 1 or budget > len(canonical_generated):
            raise RuntimeError(f"invalid prefix budget {budget}")
        prefix = canonical_generated[:budget]
        valid = [value for value in prefix if value is not None]
        unique_valid = set(valid)
        top100_hits = unique_valid.intersection(top100_endpoints)
        positive_hits = unique_valid.intersection(positive_endpoints)
        negative_hits = unique_valid.intersection(negative_endpoints)
        valid_in = sum(value in top100_endpoints for value in valid)
        valid_outside = len(valid) - valid_in
        prefix_name = f"budget_{budget}"
        metrics.update(
            {
                f"{prefix_name}_validity": len(valid) / budget,
                f"{prefix_name}_unique_valid_endpoints": float(len(unique_valid)),
                f"{prefix_name}_duplicate_rate": (
                    len(valid) - len(unique_valid)
                )
                / budget,
                f"{prefix_name}_unique_top100_hits": float(len(top100_hits)),
                f"{prefix_name}_top100_recall": len(top100_hits)
                / len(top100_endpoints),
                f"{prefix_name}_any_top100_hit": float(bool(top100_hits)),
                f"{prefix_name}_valid_in_top100_rate": valid_in / budget,
                f"{prefix_name}_valid_outside_top100_rate": valid_outside / budget,
                f"{prefix_name}_positive_unique_hits": float(len(positive_hits)),
                f"{prefix_name}_positive_recall": len(positive_hits)
                / len(positive_endpoints),
                f"{prefix_name}_negative_unique_hits": float(len(negative_hits)),
                f"{prefix_name}_negative_recall": len(negative_hits)
                / len(negative_endpoints),
            }
        )
    return metrics


def _validate_generated_identity(
    frozen: Mapping[str, Any],
    generated: Mapping[str, Any],
    *,
    condition: str,
    alpha: float,
    draw_count: int,
    sampling_steps: int,
    profile_id: str,
) -> list[str | None]:
    if generated.get("arm") != condition:
        raise RuntimeError(
            f"generated arm differs for group {frozen.get('group_uid')!r}"
        )
    for field in (
        "group_uid",
        "case_id",
        "evaluation_group_index",
        "legacy_group_idx",
        "root_draw_seeds",
        "root_choice_indices",
        "trajectory_seeds",
    ):
        if generated.get(field) != frozen.get(field):
            raise RuntimeError(
                f"generated {field} differs for group {frozen.get('group_uid')!r}"
            )
    if (
        generated.get("execution_profile_id") != profile_id
        or generated.get("sampling_steps") != sampling_steps
        or generated.get("per_sample_transition_kernel") != "stateless_counter_v1"
        or generated.get("sampling_rng_mode") != "paired_stateless_counter_v1"
    ):
        raise RuntimeError("generated row violates the formal execution profile")
    expected_alpha: float | None = None if condition == "reference" else alpha
    observed_alpha = generated.get("guidance_residual_scale")
    if expected_alpha is None:
        if observed_alpha is not None:
            raise RuntimeError("reference row unexpectedly has a guidance scale")
    elif (
        isinstance(observed_alpha, bool)
        or not isinstance(observed_alpha, (int, float))
        or not math.isclose(
            float(observed_alpha), expected_alpha, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise RuntimeError("generated guidance scale differs from frozen alpha")
    for field in ("categories", "rc_permutation_keys"):
        value = generated.get(field)
        if (
            not isinstance(value, Sequence)
            or isinstance(value, (str, bytes))
            or len(value) != draw_count
        ):
            raise RuntimeError(f"generated {field} differs from draw count")
    return _canonical_generated(
        generated.get("generated_smiles"),
        expected_draw_count=draw_count,
    )


def score_arm_rows(
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    inventory_by_case: Mapping[str, set[str]],
    generated_rows: Sequence[Mapping[str, Any]],
    model: str,
    condition: str,
    alpha: float,
    draw_count: int,
    sampling_steps: int,
    profile_id: str,
    top_k: Sequence[int],
    budgets: Sequence[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_index = index_rows(frozen_rows, key="group_uid", label="frozen manifest")
    generated_index = index_rows(generated_rows, key="group_uid", label="generated arm")
    if set(frozen_index) != set(generated_index):
        missing = sorted(set(frozen_index).difference(generated_index))
        extra = sorted(set(generated_index).difference(frozen_index))
        raise RuntimeError(
            f"generated group_uid coverage differs: missing={missing[:3]} extra={extra[:3]}"
        )

    pooled: dict[int, dict[str, set[str]]] = {
        budget: defaultdict(set) for budget in budgets
    }
    scored: list[dict[str, Any]] = []
    for raw_frozen in frozen_rows:
        frozen = dict(raw_frozen)
        uid = str(frozen["group_uid"])
        generated = generated_index[uid]
        canonical = _validate_generated_identity(
            frozen,
            generated,
            condition=condition,
            alpha=alpha,
            draw_count=draw_count,
            sampling_steps=sampling_steps,
            profile_id=profile_id,
        )
        case_id = str(frozen["case_id"])
        try:
            top100 = inventory_by_case[case_id]
        except KeyError as error:
            raise RuntimeError(f"missing Top100 inventory for {case_id}") from error
        positives = set(frozen["positive_target_smiles"])
        negatives = set(frozen["negative_target_smiles"])
        metrics = score_generated_group(
            canonical,
            sorted(positives),
            sorted(negatives),
            top_k=top_k,
            expected_draw_count=draw_count,
            require_canonical_known_sets=True,
            require_canonical_generated=True,
        )["metrics"]
        metrics.update(
            prefix_reachability_metrics(
                canonical,
                top100_endpoints=top100,
                positive_endpoints=positives,
                negative_endpoints=negatives,
                budgets=budgets,
            )
        )
        for budget in budgets:
            pooled[budget][case_id].update(
                value for value in canonical[:budget] if value is not None
            )
        scored.append(
            {
                "schema_version": "uspto_top100.common_support_group_score.v1",
                "arm": f"{model}.{condition}",
                "model": model,
                "condition": condition,
                "group_uid": uid,
                "case_id": case_id,
                "axis_id": frozen["axis_id"],
                "polarity": frozen["polarity"],
                "draw_count": draw_count,
                "metrics": metrics,
            }
        )

    selected_cases = sorted({str(row["case_id"]) for row in frozen_rows})
    denominator = sum(len(inventory_by_case[case_id]) for case_id in selected_cases)
    pooled_report: dict[str, Any] = {}
    for budget in budgets:
        generated_count = 0
        hit_count = 0
        outside_count = 0
        hit_cases = 0
        for case_id in selected_cases:
            generated_set = pooled[budget].get(case_id, set())
            hits = generated_set.intersection(inventory_by_case[case_id])
            outside = generated_set.difference(inventory_by_case[case_id])
            generated_count += len(generated_set)
            hit_count += len(hits)
            outside_count += len(outside)
            hit_cases += int(bool(hits))
        pooled_report[str(budget)] = {
            "interpretation": "descriptive_union_across_groups_with_budget_per_group",
            "case_count": len(selected_cases),
            "top100_unique_case_endpoint_denominator": denominator,
            "generated_unique_case_endpoints": generated_count,
            "unique_top100_case_endpoint_hits": hit_count,
            "unique_valid_outside_top100_case_endpoints": outside_count,
            "whole_pool_coverage": hit_count / denominator,
            "cases_with_any_top100_hit": hit_cases,
            "case_any_top100_hit_rate": hit_cases / len(selected_cases),
        }
    return scored, pooled_report


def _metric_matrix(
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    arm_names: Sequence[str],
    metric_names: Sequence[str],
    indices: Sequence[int] | None = None,
) -> tuple[Any, Any, list[str]]:
    import numpy as np

    positions = list(range(len(frozen_rows))) if indices is None else list(indices)
    by_arm = {
        arm: index_rows(scored_by_arm[arm], key="group_uid", label=f"scores {arm}")
        for arm in arm_names
    }
    group = np.empty(
        (len(positions), len(arm_names), len(metric_names)), dtype=np.float64
    )
    case_ids: list[str] = []
    for out_position, frozen_position in enumerate(positions):
        frozen = frozen_rows[frozen_position]
        uid = str(frozen["group_uid"])
        case_ids.append(str(frozen["case_id"]))
        for arm_index, arm in enumerate(arm_names):
            metrics = by_arm[arm][uid]["metrics"]
            group[out_position, arm_index] = [
                float(metrics[name]) for name in metric_names
            ]
    case_order = list(dict.fromkeys(case_ids))
    case_position = {case_id: index for index, case_id in enumerate(case_order)}
    sums = np.zeros(
        (len(case_order), len(arm_names), len(metric_names)), dtype=np.float64
    )
    counts = np.zeros(len(case_order), dtype=np.int64)
    for group_index, case_id in enumerate(case_ids):
        position = case_position[case_id]
        sums[position] += group[group_index]
        counts[position] += 1
    return group, sums / counts[:, None, None], case_order


def _estimand_specs(
    *,
    models: Sequence[str],
    conditions: Sequence[str],
    arm_names: Sequence[str],
) -> tuple[list[str], Any, dict[str, Any]]:
    import numpy as np

    position = {arm: index for index, arm in enumerate(arm_names)}
    names: list[str] = []
    coefficients: list[Any] = []
    metadata: dict[str, Any] = {}

    def add(name: str, terms: Sequence[tuple[str, float]], kind: str) -> None:
        vector = np.zeros(len(arm_names), dtype=np.float64)
        for arm, coefficient in terms:
            vector[position[arm]] += coefficient
        names.append(name)
        coefficients.append(vector)
        metadata[name] = {
            "kind": kind,
            "terms": [
                {"arm": arm, "coefficient": coefficient}
                for arm, coefficient in terms
            ],
        }

    for arm in arm_names:
        add(arm, ((arm, 1.0),), "raw_arm")
    for model in models:
        add(
            f"{model}.true_minus_shuffled",
            ((f"{model}.true", 1.0), (f"{model}.shuffled", -1.0)),
            "within_model_primary",
        )
        add(
            f"{model}.true_minus_reference",
            ((f"{model}.true", 1.0), (f"{model}.reference", -1.0)),
            "within_model_secondary",
        )
        add(
            f"{model}.shuffled_minus_reference",
            ((f"{model}.shuffled", 1.0), (f"{model}.reference", -1.0)),
            "within_model_diagnostic",
        )
    support_pairs = (list(zip(models[1:], models[:-1])) + [(models[-1], models[0])]) if len(models) > 1 else []
    for later, earlier in support_pairs:
        name = (
            f"{later}_minus_{earlier}.true_minus_shuffled_gap"
            if (later, earlier) != (models[-1], models[0])
            else f"{later}_minus_{earlier}.endpoint_true_minus_shuffled_gap"
        )
        add(
            name,
            (
                (f"{later}.true", 1.0),
                (f"{later}.shuffled", -1.0),
                (f"{earlier}.true", -1.0),
                (f"{earlier}.shuffled", 1.0),
            ),
            "support_expansion_difference_in_differences",
        )
    return names, np.stack(coefficients, axis=0), metadata


def _bootstrap(
    matrix: Any,
    *,
    replicates: int,
    seed: int,
    batch_size: int,
) -> tuple[Any, Any, Any]:
    import numpy as np

    if matrix.ndim != 3 or matrix.shape[0] < 2:
        raise RuntimeError("bootstrap matrix must be cases x estimands x metrics")
    observed = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    samples = np.empty(
        (replicates, matrix.shape[1], matrix.shape[2]), dtype=np.float32
    )
    flat = matrix.reshape(matrix.shape[0], -1)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        draw = rng.integers(
            0, matrix.shape[0], size=(stop - start, matrix.shape[0])
        )
        weights = np.zeros((stop - start, matrix.shape[0]), dtype=np.float64)
        rows = np.repeat(np.arange(stop - start), matrix.shape[0])
        np.add.at(
            weights,
            (rows, draw.reshape(-1)),
            1.0 / matrix.shape[0],
        )
        samples[start:stop] = (weights @ flat).reshape(
            stop - start, matrix.shape[1], matrix.shape[2]
        )
    return (
        observed,
        np.percentile(samples, 2.5, axis=0),
        np.percentile(samples, 97.5, axis=0),
    )


def _records(
    *,
    estimand_names: Sequence[str],
    metric_names: Sequence[str],
    observed: Any,
    low: Any,
    high: Any,
    metadata: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        name: {
            **dict(metadata[name]),
            **dict(extra),
            "observed": {
                metric: float(observed[e_index, m_index])
                for m_index, metric in enumerate(metric_names)
            },
            "ci95_low": {
                metric: float(low[e_index, m_index])
                for m_index, metric in enumerate(metric_names)
            },
            "ci95_high": {
                metric: float(high[e_index, m_index])
                for m_index, metric in enumerate(metric_names)
            },
        }
        for e_index, name in enumerate(estimand_names)
    }


def _bootstrap_report(
    *,
    frozen_rows: Sequence[Mapping[str, Any]],
    scored_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    models: Sequence[str],
    conditions: Sequence[str],
    axes: Sequence[str],
    all_metric_names: Sequence[str],
    bootstrap_metric_names: Sequence[str],
    replicates: int,
    seed: int,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    arm_names = [f"{model}.{condition}" for model in models for condition in conditions]
    estimand_names, coefficients, metadata = _estimand_specs(
        models=models,
        conditions=conditions,
        arm_names=arm_names,
    )
    group, cases, case_order = _metric_matrix(
        frozen_rows=frozen_rows,
        scored_by_arm=scored_by_arm,
        arm_names=arm_names,
        metric_names=all_metric_names,
    )
    metric_position = {name: index for index, name in enumerate(all_metric_names)}
    selected = [metric_position[name] for name in bootstrap_metric_names]
    case_estimands = np.einsum(
        "cam,ea->cem", cases[:, :, selected], coefficients, optimize=True
    )
    observed, low, high = _bootstrap(
        case_estimands,
        replicates=replicates,
        seed=seed,
        batch_size=batch_size,
    )
    overall = _records(
        estimand_names=estimand_names,
        metric_names=bootstrap_metric_names,
        observed=observed,
        low=low,
        high=high,
        metadata=metadata,
        extra={
            "case_count": len(case_order),
            "cluster_unit": "case_id",
            "bootstrap_seed": seed,
            "bootstrap_replicates": replicates,
        },
    )

    axis_case_matrices: dict[str, Any] = {}
    axis_case_counts: dict[str, int] = {}
    axis_point_all: dict[str, Any] = {}
    per_axis_bootstrap: dict[str, Any] = {}
    for axis_index, axis in enumerate(axes):
        indices = [
            index for index, row in enumerate(frozen_rows) if row["axis_id"] == axis
        ]
        axis_group, axis_cases, axis_case_order = _metric_matrix(
            frozen_rows=frozen_rows,
            scored_by_arm=scored_by_arm,
            arm_names=arm_names,
            metric_names=all_metric_names,
            indices=indices,
        )
        axis_case_matrices[axis] = np.einsum(
            "cam,ea->cem",
            axis_cases[:, :, selected],
            coefficients,
            optimize=True,
        )
        axis_case_counts[axis] = len(axis_case_order)
        axis_point_all[axis] = {
            arm: {
                metric: float(axis_cases[:, arm_index, metric_index].mean())
                for metric_index, metric in enumerate(all_metric_names)
            }
            for arm_index, arm in enumerate(arm_names)
        }
        axis_observed, axis_low, axis_high = _bootstrap(
            axis_case_matrices[axis],
            replicates=replicates,
            seed=seed,
            batch_size=batch_size,
        )
        per_axis_bootstrap[axis] = _records(
            estimand_names=estimand_names,
            metric_names=bootstrap_metric_names,
            observed=axis_observed,
            low=axis_low,
            high=axis_high,
            metadata=metadata,
            extra={
                "case_count": len(axis_case_order),
                "axis_id": axis,
                "cluster_unit": "case_id_within_axis",
                "bootstrap_seed": seed,
                "bootstrap_replicates": replicates,
            },
        )

    rng = np.random.default_rng(seed)
    macro_samples = np.empty(
        (replicates, len(estimand_names), len(bootstrap_metric_names)),
        dtype=np.float32,
    )
    macro_observed = np.stack(
        [matrix.mean(axis=0) for matrix in axis_case_matrices.values()], axis=0
    ).mean(axis=0)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        total = np.zeros(
            (stop - start, len(estimand_names) * len(bootstrap_metric_names)),
            dtype=np.float64,
        )
        for matrix in axis_case_matrices.values():
            case_count = matrix.shape[0]
            draw = rng.integers(
                0, case_count, size=(stop - start, case_count)
            )
            weights = np.zeros((stop - start, case_count), dtype=np.float64)
            rows = np.repeat(np.arange(stop - start), case_count)
            np.add.at(weights, (rows, draw.reshape(-1)), 1.0 / case_count)
            total += weights @ matrix.reshape(case_count, -1)
        macro_samples[start:stop] = (
            total / len(axis_case_matrices)
        ).reshape(
            stop - start, len(estimand_names), len(bootstrap_metric_names)
        )
    macro_low = np.percentile(macro_samples, 2.5, axis=0)
    macro_high = np.percentile(macro_samples, 97.5, axis=0)
    axis_macro = _records(
        estimand_names=estimand_names,
        metric_names=bootstrap_metric_names,
        observed=macro_observed,
        low=macro_low,
        high=macro_high,
        metadata=metadata,
        extra={
            "axis_count": len(axes),
            "case_count_by_axis": axis_case_counts,
            "cluster_unit": "case_id_within_axis",
            "axis_weighting": "equal_axis",
            "bootstrap_seed": seed,
            "bootstrap_replicates": replicates,
        },
    )

    point = {
        "group_micro": {
            arm: {
                metric: float(group[:, arm_index, metric_index].mean())
                for metric_index, metric in enumerate(all_metric_names)
            }
            for arm_index, arm in enumerate(arm_names)
        },
        "case_cluster_mean": {
            arm: {
                metric: float(cases[:, arm_index, metric_index].mean())
                for metric_index, metric in enumerate(all_metric_names)
            }
            for arm_index, arm in enumerate(arm_names)
        },
        "per_axis_case_cluster_mean": axis_point_all,
        "equal_axis_case_cluster_mean": {
            arm: {
                metric: sum(
                    axis_point_all[axis][arm][metric] for axis in axes
                )
                / len(axes)
                for metric in all_metric_names
            }
            for arm in arm_names
        },
    }
    bootstrap = {
        "schema_version": "uspto_top100.common_support_case_cluster_bootstrap.v1",
        "status": "passed",
        "bootstrap_metrics": list(bootstrap_metric_names),
        "shared_resamples_across_arms_and_estimands": True,
        "overall_case_cluster": overall,
        "equal_axis_macro": axis_macro,
        "per_axis": per_axis_bootstrap,
        "content_hashes_computed": False,
        "integrity_policy": INTEGRITY_POLICY,
    }
    return point, bootstrap


def _metric_subsets(
    metric_names: Sequence[str],
) -> tuple[list[str], list[str], list[str]]:
    controllability = [
        name
        for name in metric_names
        if name
        in {
            "validity",
            "positive_exact_rate",
            "negative_exact_rate",
            "other_valid_rate",
            "invalid_rate",
            "any_positive",
            "any_negative",
        }
        or name.startswith("freq_positive_at_")
        or name.startswith("freq_negative_at_")
    ]
    reachability = [name for name in metric_names if name.startswith("budget_")]
    bootstrap = list(controllability)
    for name in reachability:
        if (
            name.endswith("_top100_recall")
            or name.endswith("_unique_top100_hits")
            or name.endswith("_any_top100_hit")
            or name
            in {
                "budget_100_validity",
                "budget_100_unique_valid_endpoints",
                "budget_100_duplicate_rate",
                "budget_100_valid_in_top100_rate",
                "budget_100_valid_outside_top100_rate",
                "budget_100_positive_recall",
                "budget_100_negative_recall",
            }
        ):
            bootstrap.append(name)
    return controllability, reachability, bootstrap


def _subset_point(
    point: Mapping[str, Any],
    names: set[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for aggregation, payload in point.items():
        if aggregation == "per_axis_case_cluster_mean":
            result[aggregation] = {
                axis: {
                    arm: {
                        metric: value
                        for metric, value in metrics.items()
                        if metric in names
                    }
                    for arm, metrics in arms.items()
                }
                for axis, arms in payload.items()
            }
        else:
            result[aggregation] = {
                arm: {
                    metric: value
                    for metric, value in metrics.items()
                    if metric in names
                }
                for arm, metrics in payload.items()
            }
    return result


def main() -> int:
    if FINAL_PATH.exists() and COMPLETE_PATH.exists():
        report = _load_json(FINAL_PATH)
        if report.get("status") != "passed":
            raise RuntimeError("existing final score report is not passed")
        print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
        return 0

    config = _load_config()
    protocol = config["protocol"]
    expected = protocol["expected"]
    models = list(config["model_order"])
    conditions = list(config["conditions"])
    axes = list(protocol["axes"])
    frozen_rows = _rows(FORMAL_MANIFEST_PATH)
    if (
        len(frozen_rows) != expected["selected_groups"]
        or len({str(row["case_id"]) for row in frozen_rows})
        != expected["selected_cases"]
    ):
        raise RuntimeError("formal frozen manifest differs from confirmed counts")

    inventory_rows = _rows(INVENTORY_PATH)
    inventory_by_case = {
        str(row["case_id"]): set(row["unique_canonical_endpoints"])
        for row in inventory_rows
    }
    if len(inventory_by_case) != expected["selected_cases"]:
        raise RuntimeError("Top100 endpoint inventory case count differs")

    scored_by_arm: dict[str, list[dict[str, Any]]] = {}
    pooled_by_arm: dict[str, Any] = {}
    reconciliation: dict[str, Any] = {}
    for model in models:
        alpha = float(config["models"][model]["alpha"])
        for condition in conditions:
            arm = f"{model}.{condition}"
            generated_path = OUTPUT_ROOT / model / "arms" / f"{condition}.jsonl"
            generated_rows = _rows(generated_path)
            scored, pooled = score_arm_rows(
                frozen_rows=frozen_rows,
                inventory_by_case=inventory_by_case,
                generated_rows=generated_rows,
                model=model,
                condition=condition,
                alpha=alpha,
                draw_count=protocol["draw_count"],
                sampling_steps=protocol["sampling_steps"],
                profile_id=config["execution"]["profile_id"],
                top_k=protocol["top_k"],
                budgets=protocol["budgets"],
            )
            score_path = GROUP_SCORE_ROOT / f"{model}_{condition}.jsonl.gz"
            _write_rows_equal(score_path, scored)
            scored_by_arm[arm] = scored
            pooled_by_arm[arm] = pooled
            reconciliation[arm] = {
                "generated_path": str(generated_path.resolve(strict=True)),
                "score_path": str(score_path.resolve(strict=True)),
                "group_count": len(scored),
                "trajectory_count": len(scored) * protocol["draw_count"],
            }

    first_metrics = scored_by_arm[f"{models[0]}.{conditions[0]}"][0]["metrics"]
    all_metric_names = list(first_metrics)
    for arm, rows in scored_by_arm.items():
        for row in rows:
            if list(row["metrics"]) != all_metric_names:
                raise RuntimeError(f"metric schema differs in arm {arm}")
    controllability_names, reachability_names, bootstrap_names = _metric_subsets(
        all_metric_names
    )
    point, bootstrap = _bootstrap_report(
        frozen_rows=frozen_rows,
        scored_by_arm=scored_by_arm,
        models=models,
        conditions=conditions,
        axes=axes,
        all_metric_names=all_metric_names,
        bootstrap_metric_names=bootstrap_names,
        replicates=protocol["bootstrap_replicates"],
        seed=protocol["bootstrap_seed"],
        batch_size=protocol["bootstrap_batch_size"],
    )

    controllability_report = {
        "schema_version": "uspto_top100.common_support_controllability.v1",
        "status": "passed",
        "top_k": list(protocol["top_k"]),
        "point_estimates": _subset_point(point, set(controllability_names)),
        "primary_contrast": "true_minus_shuffled",
        "secondary_contrast": "true_minus_reference",
        "diagnostic_contrast": "shuffled_minus_reference",
        "bootstrap_report": str(BOOTSTRAP_PATH),
        "content_hashes_computed": False,
    }
    reachability_report = {
        "schema_version": "uspto_top100.common_support_reachability.v1",
        "status": "passed",
        "budgets": list(protocol["budgets"]),
        "support_unit": "canonical_precursor_endpoint",
        "prefix_semantics": "literal_first_B_trajectories",
        "point_estimates": _subset_point(point, set(reachability_names)),
        "pooled_descriptive": pooled_by_arm,
        "valid_outside_top100_is_not_invalid": True,
        "rank_shell_analysis_performed": False,
        "bootstrap_report": str(BOOTSTRAP_PATH),
        "content_hashes_computed": False,
    }
    _write_json_equal(CONTROLLABILITY_PATH, controllability_report)
    _write_json_equal(REACHABILITY_PATH, reachability_report)
    _write_json_equal(BOOTSTRAP_PATH, bootstrap)

    trajectory_total = sum(
        int(record["trajectory_count"]) for record in reconciliation.values()
    )
    if (
        len(reconciliation) != expected["total_arms"]
        or any(
            record["trajectory_count"] != expected["trajectories_per_arm"]
            for record in reconciliation.values()
        )
        or trajectory_total != expected["total_trajectories"]
    ):
        raise RuntimeError("formal trajectory reconciliation differs from contract")
    final = {
        "schema_version": "uspto_top100.common_support_final_audit.v1",
        "status": "passed",
        "experiment_id": config["experiment_id"],
        "selected_cases": expected["selected_cases"],
        "selected_groups": expected["selected_groups"],
        "arms": len(reconciliation),
        "trajectories_per_arm": expected["trajectories_per_arm"],
        "total_trajectories": trajectory_total,
        "reconciliation": reconciliation,
        "reports": {
            "controllability": str(CONTROLLABILITY_PATH.resolve(strict=True)),
            "reachability": str(REACHABILITY_PATH.resolve(strict=True)),
            "paired_bootstrap": str(BOOTSTRAP_PATH.resolve(strict=True)),
        },
        "scoring_join": "semantic_group_uid_keyed",
        "generated_row_order_used_for_scoring": False,
        "test_based_selection": False,
        "content_hashes_computed": False,
        "integrity_policy": INTEGRITY_POLICY,
    }
    _write_json_equal(FINAL_PATH, final)
    _write_text_equal(COMPLETE_PATH, "complete\n")
    print(json.dumps(final, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    configure_paths(args.run_dir)
    raise SystemExit(main())
