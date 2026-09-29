"""Read-only compatibility for historical cap-256 metric cohorts.

New USPTO Top30 4-/5-axis formal data is materialized offline through
``data.bound_dataset`` and loaded from its dataset manifest.  The cap-256
schema remains here so historical and Top60 19-axis manifests can still be
validated; it is not the canonical builder for new bound datasets.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from retro_guided_generation.guidance.io import iter_jsonl

from .io import (
    EvaluationError,
    INTEGRITY_POLICY,
    require_int,
    require_text,
    write_new_json,
    write_new_jsonl,
)


METRIC_COHORT_SCHEMA = "rgg_metric_axis_frozen_cohort_selection.v1"
METRIC_COHORT_PROTOCOL_ID = "metric_axis_paired_family_cap256_v1"
METRIC_MAX_FAMILIES_PER_AXIS = 256
METRIC_POLARITIES = (-1, 1)
METRIC_SELECTION_SEED = 1042
METRIC_CONDITIONS = ("reference", "null", "true", "shuffled")
METRIC_DRAW_COUNT_BY_SPLIT = {"validation": 1, "test": 100}


def _load_manifest(corpus_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(corpus_root).expanduser().resolve(strict=True)
    manifest_path = (root / "manifest.json").resolve(strict=True)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read metric execution-view manifest: {error}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError("metric execution-view manifest must be an object")
    return root, dict(value)


def _metric_axis_ids(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    branches = manifest.get("branches")
    if not isinstance(branches, Mapping):
        raise EvaluationError("metric execution view must declare branches")
    enabled = {str(name) for name, enabled in branches.items() if enabled is True}
    if enabled != {"metric"}:
        raise EvaluationError(
            "canonical metric cohort requires a metric-only execution view; "
            f"enabled branches are {sorted(enabled)}"
        )
    subset = manifest.get("axis_subset")
    if not isinstance(subset, Mapping):
        raise EvaluationError("metric execution view must declare axis_subset")
    raw_axis_ids = subset.get("axis_ids")
    if isinstance(raw_axis_ids, (str, bytes)) or not isinstance(raw_axis_ids, Sequence):
        raise EvaluationError("axis_subset.axis_ids must be an ordered sequence")
    axis_ids = tuple(
        require_text(axis_id, f"axis_subset.axis_ids[{index}]")
        for index, axis_id in enumerate(raw_axis_ids)
    )
    if len(axis_ids) < 2:
        raise EvaluationError(
            "metric cohort v1 requires at least two axes for a different-axis shuffled control"
        )
    if len(set(axis_ids)) != len(axis_ids):
        raise EvaluationError("axis_subset.axis_ids must be unique")
    return axis_ids


def _semantic_groups_path(
    *, root: Path, manifest: Mapping[str, Any], split: str
) -> Path:
    if split not in METRIC_DRAW_COUNT_BY_SPLIT:
        raise EvaluationError("canonical metric cohort split must be validation or test")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationError("metric execution view must declare artifacts")
    relative = require_text(
        artifacts.get(f"semantic_groups.{split}"),
        f"artifacts.semantic_groups.{split}",
    )
    path = Path(relative).expanduser()
    return path.resolve(strict=True) if path.is_absolute() else (root / path).resolve(strict=True)


def build_metric_cohort_selection(
    *,
    corpus_root: str | Path,
    split: str,
) -> dict[str, Any]:
    """Select up to 256 complete paired semantic families per metric axis.

    The registered axis order, source group order, fixed seed, and Python
    ``random.sample`` schedule reproduce the five-axis validation selection
    while extending it to any registered metric-axis count of at least two.
    """

    root, manifest = _load_manifest(corpus_root)
    axis_ids = _metric_axis_ids(manifest)
    source = _semantic_groups_path(root=root, manifest=manifest, split=split)
    families: dict[str, dict[str, dict[int, dict[str, Any]]]] = {
        axis_id: defaultdict(dict) for axis_id in axis_ids
    }
    seen_group_uids: set[str] = set()
    source_group_count = 0
    for ordinal, row in enumerate(iter_jsonl(source)):
        source_group_count += 1
        if row.get("split") != split:
            raise EvaluationError(
                f"metric source group ordinal {ordinal} does not belong to split {split!r}"
            )
        if row.get("branch") != "metric":
            raise EvaluationError(
                f"metric source group ordinal {ordinal} belongs to a non-metric branch"
            )
        axis_id = require_text(row.get("axis_id"), "axis_id")
        if axis_id not in families:
            raise EvaluationError(f"metric source exposes unregistered axis {axis_id!r}")
        polarity = row.get("polarity")
        if isinstance(polarity, bool) or polarity not in METRIC_POLARITIES:
            raise EvaluationError(f"metric source group ordinal {ordinal} has invalid polarity")
        family_uid = require_text(row.get("semantic_family_uid"), "semantic_family_uid")
        group_uid = require_text(row.get("semantic_group_uid"), "semantic_group_uid")
        case_id = require_text(row.get("case_id"), "case_id")
        if group_uid in seen_group_uids:
            raise EvaluationError(f"duplicate semantic_group_uid {group_uid!r}")
        seen_group_uids.add(group_uid)
        family = families[axis_id][family_uid]
        if int(polarity) in family:
            raise EvaluationError(
                f"duplicate polarity {polarity:+d} in metric family {family_uid!r}"
            )
        family[int(polarity)] = {
            "source_group_ordinal": ordinal,
            "semantic_group_uid": group_uid,
            "semantic_family_uid": family_uid,
            "case_id": case_id,
            "axis_id": axis_id,
            "polarity": int(polarity),
            "source_split": split,
            "branch": "metric",
        }

    rng = random.Random(METRIC_SELECTION_SEED)
    inventory: list[dict[str, Any]] = []
    available_family_counts: dict[str, int] = {}
    incomplete_family_counts: dict[str, int] = {}
    selected_family_counts: dict[str, int] = {}
    for axis_id in axis_ids:
        incomplete = {
            family_uid
            for family_uid, by_polarity in families[axis_id].items()
            if set(by_polarity) != set(METRIC_POLARITIES)
        }
        incomplete_family_counts[axis_id] = len(incomplete)
        ordered_families = sorted(
            set(families[axis_id]).difference(incomplete),
            key=lambda family_uid: (
                min(
                    int(row["source_group_ordinal"])
                    for row in families[axis_id][family_uid].values()
                ),
                family_uid,
            ),
        )
        available_family_counts[axis_id] = len(ordered_families)
        if not ordered_families:
            raise EvaluationError(f"metric axis {axis_id!r} has no complete semantic family")
        selected_family_count = min(
            METRIC_MAX_FAMILIES_PER_AXIS, len(ordered_families)
        )
        selected_family_counts[axis_id] = selected_family_count
        chosen_positions = sorted(
            rng.sample(range(len(ordered_families)), selected_family_count)
        )
        for position in chosen_positions:
            family_uid = ordered_families[position]
            for polarity in METRIC_POLARITIES:
                inventory.append(dict(families[axis_id][family_uid][polarity]))

    inventory.sort(key=lambda row: int(row["source_group_ordinal"]))
    expected_group_count = len(METRIC_POLARITIES) * sum(selected_family_counts.values())
    selected_indices = [int(row["source_group_ordinal"]) for row in inventory]
    if len(selected_indices) != expected_group_count:
        raise EvaluationError("metric cohort does not have the protocol group count")
    if len(set(selected_indices)) != expected_group_count:
        raise EvaluationError("metric cohort repeats a source group ordinal")
    if selected_indices != sorted(selected_indices):
        raise EvaluationError("metric cohort is not preserved in source order")

    dataset_id = require_text(manifest.get("dataset_id"), "dataset_id")
    return {
        "schema_version": METRIC_COHORT_SCHEMA,
        "protocol_id": METRIC_COHORT_PROTOCOL_ID,
        "status": "passed",
        "dataset_id": dataset_id,
        "corpus_root": str(root),
        "source_semantic_groups": str(source),
        "source_split": split,
        "source_group_count": source_group_count,
        "axis_ids": list(axis_ids),
        "axis_count": len(axis_ids),
        "polarities": list(METRIC_POLARITIES),
        "sampling": {
            "unit": "semantic_family_uid",
            "algorithm": "python_random_sample_without_replacement_source_order_v1",
            "seed": METRIC_SELECTION_SEED,
            "max_families_per_axis": METRIC_MAX_FAMILIES_PER_AXIS,
            "selected_family_rule": "min(max_families_per_axis, available_complete_families)",
            "paired_polarities_per_family": True,
        },
        "available_complete_family_counts": available_family_counts,
        "excluded_incomplete_family_counts": incomplete_family_counts,
        "selected_family_counts": selected_family_counts,
        "selected_group_count": expected_group_count,
        "selected_case_count": len({str(row["case_id"]) for row in inventory}),
        "selected_source_group_indices": selected_indices,
        "selected_groups": inventory,
        "conditions": list(METRIC_CONDITIONS),
        "draw_count_per_group_condition": METRIC_DRAW_COUNT_BY_SPLIT[split],
        "trajectories_per_model": (
            expected_group_count
            * len(METRIC_CONDITIONS)
            * METRIC_DRAW_COUNT_BY_SPLIT[split]
        ),
        "integrity_policy": INTEGRITY_POLICY,
    }


def validate_metric_cohort_selection(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a materialized canonical metric cohort selection report."""

    if selection.get("schema_version") != METRIC_COHORT_SCHEMA:
        raise EvaluationError("unsupported metric cohort selection schema")
    if selection.get("protocol_id") != METRIC_COHORT_PROTOCOL_ID:
        raise EvaluationError("metric cohort selection uses the wrong protocol")
    if selection.get("status") != "passed":
        raise EvaluationError("metric cohort selection is not passed")
    split = require_text(selection.get("source_split"), "source_split")
    if split not in METRIC_DRAW_COUNT_BY_SPLIT:
        raise EvaluationError("metric cohort source_split must be validation or test")
    raw_axes = selection.get("axis_ids")
    if isinstance(raw_axes, (str, bytes)) or not isinstance(raw_axes, Sequence):
        raise EvaluationError("metric cohort axis_ids must be a sequence")
    axis_ids = tuple(require_text(value, "axis_id") for value in raw_axes)
    if len(axis_ids) < 2 or len(set(axis_ids)) != len(axis_ids):
        raise EvaluationError("metric cohort axis_ids must be unique and contain at least two axes")
    sampling = selection.get("sampling")
    expected_sampling = {
        "unit": "semantic_family_uid",
        "algorithm": "python_random_sample_without_replacement_source_order_v1",
        "seed": METRIC_SELECTION_SEED,
        "max_families_per_axis": METRIC_MAX_FAMILIES_PER_AXIS,
        "selected_family_rule": "min(max_families_per_axis, available_complete_families)",
        "paired_polarities_per_family": True,
    }
    if sampling != expected_sampling:
        raise EvaluationError("metric cohort sampling contract differs from protocol v1")
    available_counts = selection.get("available_complete_family_counts")
    incomplete_counts = selection.get("excluded_incomplete_family_counts")
    selected_counts = selection.get("selected_family_counts")
    if (
        not isinstance(available_counts, Mapping)
        or not isinstance(incomplete_counts, Mapping)
        or not isinstance(selected_counts, Mapping)
    ):
        raise EvaluationError(
            "metric cohort must record complete, incomplete, and selected family counts by axis"
        )
    expected_selected_counts: dict[str, int] = {}
    for axis_id in axis_ids:
        available = require_int(
            available_counts.get(axis_id),
            f"available_complete_family_counts.{axis_id}",
            minimum=1,
        )
        require_int(
            incomplete_counts.get(axis_id),
            f"excluded_incomplete_family_counts.{axis_id}",
            minimum=0,
        )
        selected = require_int(
            selected_counts.get(axis_id),
            f"selected_family_counts.{axis_id}",
            minimum=1,
        )
        expected = min(METRIC_MAX_FAMILIES_PER_AXIS, available)
        if selected != expected:
            raise EvaluationError(
                f"metric axis {axis_id!r} selected family count differs from min(256, available)"
            )
        expected_selected_counts[axis_id] = expected
    if (
        set(available_counts) != set(axis_ids)
        or set(incomplete_counts) != set(axis_ids)
        or set(selected_counts) != set(axis_ids)
    ):
        raise EvaluationError("metric cohort family-count axes differ from axis_ids")
    expected_groups = len(METRIC_POLARITIES) * sum(expected_selected_counts.values())
    if selection.get("selected_group_count") != expected_groups:
        raise EvaluationError("metric cohort selected_group_count differs from per-axis capped counts")
    if selection.get("conditions") != list(METRIC_CONDITIONS):
        raise EvaluationError("metric cohort conditions differ from protocol v1")
    if selection.get("draw_count_per_group_condition") != METRIC_DRAW_COUNT_BY_SPLIT[split]:
        raise EvaluationError("metric cohort draw count differs from the split protocol")
    raw_groups = selection.get("selected_groups")
    raw_indices = selection.get("selected_source_group_indices")
    if isinstance(raw_groups, (str, bytes)) or not isinstance(raw_groups, Sequence):
        raise EvaluationError("metric cohort selected_groups must be a sequence")
    if isinstance(raw_indices, (str, bytes)) or not isinstance(raw_indices, Sequence):
        raise EvaluationError("metric cohort selected_source_group_indices must be a sequence")
    if len(raw_groups) != expected_groups or len(raw_indices) != expected_groups:
        raise EvaluationError("metric cohort inventory length differs from protocol")
    indices = tuple(
        require_int(value, "selected source group index", minimum=0) for value in raw_indices
    )
    if indices != tuple(sorted(indices)) or len(set(indices)) != expected_groups:
        raise EvaluationError("metric cohort source indices must be sorted and unique")

    cells: Counter[tuple[str, int]] = Counter()
    families: dict[tuple[str, int], set[str]] = defaultdict(set)
    inventory_indices: list[int] = []
    group_uids: set[str] = set()
    for row in raw_groups:
        if not isinstance(row, Mapping):
            raise EvaluationError("metric cohort inventory row must be an object")
        axis_id = require_text(row.get("axis_id"), "selected group axis_id")
        if axis_id not in axis_ids:
            raise EvaluationError("metric cohort inventory contains an unregistered axis")
        polarity = row.get("polarity")
        if isinstance(polarity, bool) or polarity not in METRIC_POLARITIES:
            raise EvaluationError("metric cohort inventory contains an invalid polarity")
        if row.get("source_split") != split or row.get("branch") != "metric":
            raise EvaluationError("metric cohort inventory contains a wrong split or branch")
        family_uid = require_text(row.get("semantic_family_uid"), "semantic_family_uid")
        group_uid = require_text(row.get("semantic_group_uid"), "semantic_group_uid")
        if group_uid in group_uids:
            raise EvaluationError("metric cohort inventory repeats semantic_group_uid")
        group_uids.add(group_uid)
        ordinal = require_int(row.get("source_group_ordinal"), "source_group_ordinal", minimum=0)
        inventory_indices.append(ordinal)
        key = (axis_id, int(polarity))
        cells[key] += 1
        families[key].add(family_uid)
    if tuple(inventory_indices) != indices:
        raise EvaluationError("metric cohort inventory differs from selected source indices")
    for axis_id in axis_ids:
        positive = families[(axis_id, 1)]
        negative = families[(axis_id, -1)]
        expected = expected_selected_counts[axis_id]
        if cells[(axis_id, 1)] != expected or cells[(axis_id, -1)] != expected:
            raise EvaluationError(
                "metric cohort cell count differs from min(256, available families)"
            )
        if positive != negative or len(positive) != expected:
            raise EvaluationError("metric cohort polarities are not paired by semantic family")
    return {
        "status": "passed",
        "protocol_id": METRIC_COHORT_PROTOCOL_ID,
        "source_split": split,
        "axis_ids": list(axis_ids),
        "axis_count": len(axis_ids),
        "selected_family_counts": expected_selected_counts,
        "group_count": expected_groups,
        "case_count": len(
            {require_text(row.get("case_id"), "case_id") for row in raw_groups}
        ),
    }


def validate_metric_frozen_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a Pista known-set manifest exactly to its canonical metric selection."""

    summary = validate_metric_cohort_selection(selection)
    selected_groups = selection["selected_groups"]
    if len(rows) != summary["group_count"]:
        raise EvaluationError("frozen metric manifest group count differs from canonical selection")
    expected_by_index = {
        require_int(row.get("source_group_ordinal"), "source_group_ordinal", minimum=0): row
        for row in selected_groups
    }
    seen_indices: list[int] = []
    for row in rows:
        if row.get("branch") != "metric" or row.get("source_split") != summary["source_split"]:
            raise EvaluationError("frozen metric manifest contains a wrong branch or split")
        index = require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
        try:
            expected = expected_by_index[index]
        except KeyError as error:
            raise EvaluationError("frozen metric manifest contains an unselected group") from error
        if (
            row.get("group_uid") != expected.get("semantic_group_uid")
            or row.get("semantic_family_uid") != expected.get("semantic_family_uid")
            or row.get("case_id") != expected.get("case_id")
            or row.get("axis_id") != expected.get("axis_id")
            or row.get("polarity") != expected.get("polarity")
        ):
            raise EvaluationError("frozen metric manifest identity differs from canonical selection")
        seen_indices.append(index)
    if seen_indices != list(selection["selected_source_group_indices"]):
        raise EvaluationError("frozen metric manifest order differs from canonical selection")
    return summary


def validate_reference_null_trajectory_parity(
    reference_rows: Sequence[Mapping[str, Any]],
    null_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int | str]:
    """Require exact generated-trajectory parity for Reference and Null."""

    if len(reference_rows) != len(null_rows):
        raise EvaluationError("Reference and Null group counts differ")
    paired_fields = (
        "group_uid",
        "case_id",
        "evaluation_group_index",
        "legacy_group_idx",
        "root_draw_seeds",
        "root_choice_indices",
        "trajectory_seeds",
        "rc_permutation_keys",
        "generated_smiles",
        "categories",
        "category_counts",
        "sampling_steps",
        "per_sample_transition_kernel",
        "sampling_rng_mode",
        "sampling_decode_mode",
    )
    trajectory_count = 0
    for row_index, (reference, null) in enumerate(zip(reference_rows, null_rows, strict=True)):
        if reference.get("arm") != "reference" or null.get("arm") != "null":
            raise EvaluationError("Reference/Null parity inputs contain incorrect arm labels")
        for field in paired_fields:
            if reference.get(field) != null.get(field):
                raise EvaluationError(
                    f"Reference and Null differ at row {row_index}, field {field}"
                )
        generated = reference.get("generated_smiles")
        if isinstance(generated, (str, bytes)) or not isinstance(generated, Sequence):
            raise EvaluationError("Reference parity row lacks generated_smiles")
        trajectory_count += len(generated)
        if null.get("guidance_residual_scale") != 0.0:
            raise EvaluationError("Null row must record guidance_residual_scale=0")
    return {
        "status": "passed",
        "group_count": len(reference_rows),
        "trajectory_count": trajectory_count,
    }


def write_metric_pista_frozen_cohort(
    *,
    training_config: str | Path,
    rc_prediction_table: str | Path,
    rc_predictor_checkpoint: str | Path,
    split: str,
    output_selection_json: str | Path,
    output_frozen_jsonl: str | Path,
    output_audit_json: str | Path,
) -> dict[str, Path]:
    """Build the complete metric-only cohort and Pista known-set manifest once."""

    targets = tuple(
        Path(path).expanduser()
        for path in (
            output_selection_json,
            output_frozen_jsonl,
            output_audit_json,
        )
    )
    existing = [str(path) for path in targets if path.exists() or path.is_symlink()]
    if existing:
        raise EvaluationError(
            "refusing to overwrite canonical metric cohort outputs: " + ", ".join(existing)
        )

    from retro_guided_generation.training.pista import load_pista_training_spec

    from .pista import (
        DEFAULT_RC_TOP_K,
        DEFAULT_ROOT_DRAW_SEED_BASE,
        DEFAULT_TRAJECTORY_SEED_BASE,
        build_pista_frozen_rows,
        validate_pista_frozen_rows,
    )

    spec = load_pista_training_spec(training_config)
    if spec.stage != "stage-b":
        raise EvaluationError("canonical metric inference cohort requires a stage-b config")
    if spec.selected_branch != "metric":
        raise EvaluationError(
            "canonical metric cohort requires training config model.selected_branch=metric"
        )
    metric_weight = float(spec.branch_weights.get("metric", 0.0))
    structure_weight = float(spec.branch_weights.get("structure", 0.0))
    if metric_weight <= 0.0 or structure_weight != 0.0:
        raise EvaluationError(
            "canonical metric cohort requires metric-only training branch weights"
        )

    selection = build_metric_cohort_selection(corpus_root=spec.corpus_root, split=split)
    validate_metric_cohort_selection(selection)
    draws = METRIC_DRAW_COUNT_BY_SPLIT[split]
    rows = build_pista_frozen_rows(
        corpus_root=spec.corpus_root,
        rc_prediction_table=rc_prediction_table,
        rc_predictor_checkpoint=rc_predictor_checkpoint,
        split=split,
        draw_count=draws,
        training_seed=spec.training.seed,
        evaluation_seed=spec.training.seed + 1000,
        rc_top_k=DEFAULT_RC_TOP_K,
        root_draw_seed_base=DEFAULT_ROOT_DRAW_SEED_BASE,
        trajectory_seed_base=DEFAULT_TRAJECTORY_SEED_BASE,
        expected_group_count=int(selection["selected_group_count"]),
        selected_group_indices=selection["selected_source_group_indices"],
        shuffle_donor_axis_ids=selection["axis_ids"],
    )
    schedule_summary = validate_pista_frozen_rows(
        rows,
        expected_draw_count=draws,
        expected_rc_top_k=DEFAULT_RC_TOP_K,
        root_draw_seed_base=DEFAULT_ROOT_DRAW_SEED_BASE,
        trajectory_seed_base=DEFAULT_TRAJECTORY_SEED_BASE,
    )
    cohort_summary = validate_metric_frozen_rows(rows, selection=selection)
    audit = {
        "schema_version": "rgg_metric_axis_frozen_cohort_audit.v1",
        "protocol_id": METRIC_COHORT_PROTOCOL_ID,
        "status": "passed",
        "dataset_id": selection["dataset_id"],
        "source_split": split,
        "axis_ids": selection["axis_ids"],
        "axis_count": selection["axis_count"],
        "group_count": selection["selected_group_count"],
        "case_count": selection["selected_case_count"],
        "max_families_per_axis": METRIC_MAX_FAMILIES_PER_AXIS,
        "excluded_incomplete_family_counts": selection[
            "excluded_incomplete_family_counts"
        ],
        "selected_family_counts": selection["selected_family_counts"],
        "conditions": list(METRIC_CONDITIONS),
        "draw_count_per_group_condition": draws,
        "trajectories_per_model": selection["trajectories_per_model"],
        "cohort_validation": cohort_summary,
        "pista_schedule_validation": schedule_summary,
        "integrity_policy": INTEGRITY_POLICY,
    }
    selection_path = write_new_json(targets[0], selection)
    frozen_path = write_new_jsonl(targets[1], rows)
    audit["selection_manifest"] = str(selection_path)
    audit["frozen_manifest"] = str(frozen_path)
    audit_path = write_new_json(targets[2], audit)
    return {
        "selection_manifest": selection_path,
        "frozen_manifest": frozen_path,
        "audit": audit_path,
    }


__all__ = [
    "METRIC_COHORT_PROTOCOL_ID",
    "METRIC_COHORT_SCHEMA",
    "METRIC_CONDITIONS",
    "METRIC_DRAW_COUNT_BY_SPLIT",
    "METRIC_MAX_FAMILIES_PER_AXIS",
    "METRIC_POLARITIES",
    "METRIC_SELECTION_SEED",
    "build_metric_cohort_selection",
    "validate_metric_cohort_selection",
    "validate_metric_frozen_rows",
    "validate_reference_null_trajectory_parity",
    "write_metric_pista_frozen_cohort",
]
