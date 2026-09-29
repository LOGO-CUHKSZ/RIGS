"""Read-only binding to existing evaluation splits; no dataset construction."""
from __future__ import annotations
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from retro_guided_generation.guidance.io import iter_jsonl

BOUND_DATASET_CONTRACT_SCHEMA = "rgg.bound_dataset_contract.v1"
BOUND_DATASET_PROTOCOL_ID = "metric_axis_balanced_train8_val1_test1_v2"
BOUND_SELECTION_SCHEMA = "rgg.bound_split_selection.v1"
BOUND_EVALUATION_SPLITS = frozenset({"validation", "test"})
BOUND_RUNTIME_SELECTION_POLICY = "forbidden"
INTEGRITY_POLICY = "no_hashes_calculated_or_validated"

class BoundDatasetError(ValueError):
    """Raised when an immutable dataset contract is incomplete or drifts."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BoundDatasetError(f"{name} must be a non-empty string")
    return value.strip()


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BoundDatasetError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BoundDatasetError(f"{name} must be an object")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BoundDatasetError(f"{name} must be a sequence")
    return value


def _load_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        else:
            value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BoundDatasetError(f"cannot read {resolved}: {error}") from error
    if not isinstance(value, Mapping):
        raise BoundDatasetError(f"{resolved} must contain an object")
    return resolved, dict(value)


def _relative_artifact(root: Path, manifest: Mapping[str, Any], key: str) -> Path:
    artifacts = _mapping(manifest.get("artifacts"), "manifest.artifacts")
    relative = Path(_text(artifacts.get(key), f"manifest.artifacts.{key}"))
    if relative.is_absolute():
        raise BoundDatasetError(f"artifact {key!r} must use a bundle-relative path")
    resolved = (root / relative).resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BoundDatasetError(f"artifact {key!r} escapes the dataset bundle") from error
    return resolved


@dataclass(frozen=True)
class BoundEvaluationSplit:
    """Resolved immutable inputs for one validation or test evaluation."""

    manifest_path: Path
    dataset_root: Path
    dataset_id: str
    split: str
    semantic_groups: Path
    evaluation_mappings: Path
    selection_manifest: Path
    frozen_manifest: Path
    expected_group_count: int
    expected_case_count: int
    expected_draw_count: int
    conditions: tuple[str, ...]
    protocol_id: str


def resolve_bound_evaluation_split(
    manifest_path: str | Path,
    *,
    split: str,
) -> BoundEvaluationSplit:
    """Resolve a pre-materialized split without any branch-specific sampling."""

    if split not in BOUND_EVALUATION_SPLITS:
        raise BoundDatasetError("bound evaluation split must be validation or test")
    resolved_manifest, manifest = _load_document(manifest_path)
    if manifest.get("status") != "complete":
        raise BoundDatasetError("bound dataset manifest is not complete")
    contract = _mapping(
        manifest.get("bound_dataset_contract"), "manifest.bound_dataset_contract"
    )
    if contract.get("schema_version") != BOUND_DATASET_CONTRACT_SCHEMA:
        raise BoundDatasetError("unsupported bound dataset contract schema")
    protocol_id = _text(contract.get("protocol_id"), "protocol_id")
    if contract.get("runtime_selection") != BOUND_RUNTIME_SELECTION_POLICY:
        raise BoundDatasetError("bound dataset must forbid runtime row selection")
    split_contract = _mapping(
        _mapping(contract.get("splits"), "bound_dataset_contract.splits").get(split),
        f"bound_dataset_contract.splits.{split}",
    )
    if split_contract.get("role") != "canonical_evaluation":
        raise BoundDatasetError(f"bound split {split!r} is not canonical evaluation data")
    root = resolved_manifest.parent.resolve(strict=True)
    artifacts = _mapping(split_contract.get("artifacts"), f"split {split} artifacts")

    def split_artifact(name: str) -> Path:
        relative = Path(_text(artifacts.get(name), f"split {split} artifact {name}"))
        if relative.is_absolute():
            raise BoundDatasetError("bound split artifacts must be bundle-relative")
        resolved = (root / relative).resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise BoundDatasetError("bound split artifact escapes the dataset bundle") from error
        return resolved

    raw_conditions = _sequence(split_contract.get("conditions"), "conditions")
    conditions = tuple(_text(value, "condition") for value in raw_conditions)
    if not conditions or len(set(conditions)) != len(conditions):
        raise BoundDatasetError("bound evaluation conditions must be non-empty and unique")
    return BoundEvaluationSplit(
        manifest_path=resolved_manifest,
        dataset_root=root,
        dataset_id=_text(manifest.get("dataset_id"), "dataset_id"),
        split=split,
        semantic_groups=split_artifact("semantic_groups"),
        evaluation_mappings=split_artifact("evaluation_mappings"),
        selection_manifest=split_artifact("selection_manifest"),
        frozen_manifest=split_artifact("frozen_manifest"),
        expected_group_count=_integer(split_contract.get("group_count"), "group_count", minimum=1),
        expected_case_count=_integer(split_contract.get("case_count"), "case_count", minimum=1),
        expected_draw_count=_integer(split_contract.get("draw_count_per_group_condition"), "draw_count", minimum=1),
        conditions=conditions,
        protocol_id=protocol_id,
    )


def validate_bound_frozen_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    binding: BoundEvaluationSplit,
) -> dict[str, Any]:
    """Require exact row/order identity between data, mappings, and schedule."""

    if len(rows) != binding.expected_group_count:
        raise BoundDatasetError("frozen schedule group count differs from dataset binding")
    schedule_cases = {_text(row.get("case_id"), "frozen case_id") for row in rows}
    if len(schedule_cases) != binding.expected_case_count:
        raise BoundDatasetError("frozen schedule case count differs from dataset binding")

    semantic_count = 0
    semantic_cases: set[str] = set()
    semantic_uids: set[str] = set()
    for ordinal, (semantic, frozen) in enumerate(
        zip(iter_jsonl(binding.semantic_groups), rows, strict=True)
    ):
        semantic_count += 1
        group_uid = _text(semantic.get("semantic_group_uid"), "semantic_group_uid")
        case_id = _text(semantic.get("case_id"), "case_id")
        if group_uid in semantic_uids:
            raise BoundDatasetError("bound semantic split repeats semantic_group_uid")
        semantic_uids.add(group_uid)
        semantic_cases.add(case_id)
        if frozen.get("evaluation_group_index") != ordinal:
            raise BoundDatasetError("frozen schedule index differs from materialized row order")
        identity_pairs = (
            ("group_uid", group_uid),
            ("case_id", case_id),
            ("semantic_family_uid", semantic.get("semantic_family_uid")),
            ("branch", semantic.get("branch")),
            ("axis_id", semantic.get("axis_id")),
            ("polarity", semantic.get("polarity")),
        )
        for frozen_key, expected in identity_pairs:
            if frozen.get(frozen_key) != expected:
                raise BoundDatasetError(
                    f"frozen schedule field {frozen_key!r} differs at row {ordinal}"
                )
    if semantic_count != binding.expected_group_count:
        raise BoundDatasetError("semantic split count differs from dataset binding")
    if semantic_cases != schedule_cases:
        raise BoundDatasetError("semantic and frozen schedule case identities differ")

    mapping_count = 0
    for ordinal, mapping in enumerate(iter_jsonl(binding.evaluation_mappings)):
        mapping_count += 1
        if mapping.get("semantic_group_ordinal") != ordinal:
            raise BoundDatasetError("evaluation mapping ordinal is not contiguous")
        if mapping.get("semantic_group_uid") != rows[ordinal].get("group_uid"):
            raise BoundDatasetError("evaluation mapping group identity differs from schedule")
    if mapping_count != binding.expected_group_count:
        raise BoundDatasetError("evaluation mapping count differs from dataset binding")

    _selection_path, selection = _load_document(binding.selection_manifest)
    if selection.get("schema_version") != BOUND_SELECTION_SCHEMA:
        raise BoundDatasetError("unsupported bound selection manifest schema")
    if selection.get("dataset_id") != binding.dataset_id or selection.get("split") != binding.split:
        raise BoundDatasetError("selection manifest identity differs from dataset binding")
    if selection.get("selected_group_count") != binding.expected_group_count:
        raise BoundDatasetError("selection manifest group count differs from dataset binding")
    selected = _sequence(selection.get("selected_groups"), "selected_groups")
    if len(selected) != binding.expected_group_count:
        raise BoundDatasetError("selection manifest inventory is incomplete")
    for ordinal, (selected_row, frozen) in enumerate(zip(selected, rows, strict=True)):
        selected_mapping = _mapping(selected_row, "selected group")
        if selected_mapping.get("materialized_group_ordinal") != ordinal:
            raise BoundDatasetError("selection materialized ordinals are not contiguous")
        if selected_mapping.get("semantic_group_uid") != frozen.get("group_uid"):
            raise BoundDatasetError("selection and frozen schedule group identities differ")

    return {
        "status": "passed",
        "dataset_id": binding.dataset_id,
        "protocol_id": binding.protocol_id,
        "split": binding.split,
        "group_count": binding.expected_group_count,
        "case_count": binding.expected_case_count,
        "draw_count_per_group_condition": binding.expected_draw_count,
        "conditions": list(binding.conditions),
        "runtime_selection": BOUND_RUNTIME_SELECTION_POLICY,
    }


