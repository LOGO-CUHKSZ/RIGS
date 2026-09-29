"""Pista-to-generic inference evaluation adapter.

The generic known-set evaluator deliberately scores completed arm records and
does not own a model-specific generation loop.  This module is the narrow
Pista bridge: it freezes the Pista runtime relations into the generic
manifest, emits raw paired arm records, and leaves metric computation to
``retro_guided_generation.evaluation.inference``.

No content hash is calculated or validated anywhere in this adapter.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from retro_guided_generation.data.runtime import GuidanceRuntimeIndex
from retro_guided_generation.data.runtime.pista import (
    PistaTextTokenMapping,
    PistaTokenMemory,
    load_corpus_text_pools,
)
from retro_guided_generation.data.runtime.pista_molecules import PistaGraphVocabulary
from retro_guided_generation.guidance.io import (
    JsonlStreamWriter,
    ensure_output_directory,
    iter_jsonl,
    iter_models,
)
from retro_guided_generation.guidance.schema import ShuffleDonorPool
from retro_guided_generation.guidance.shuffle import RuntimeShuffleSampler
from retro_guided_generation.training.pista import PistaTrainingSpec, load_pista_training_spec

from .chem import CANONICALIZATION_POLICY, canonical_reaction_smiles
from .execution import InferenceExecutionProfile, resolve_inference_execution_profile
from .inference import score_generated_group
from .io import (
    EvaluationError,
    INTEGRITY_POLICY,
    read_rows,
    require_int,
    write_new_json,
    write_new_jsonl,
)
from .rc import FrozenRCTable
from .rooted_product import RootedProductGraphBuilder, product_atom_count


PISTA_FROZEN_SCHEMA = "rgg_pista_known_set_frozen_group.v2"
PISTA_ARM_SCHEMA = "rgg_pista_inference_arm_row.v1"
_SPLIT_NAMES = frozenset({"train", "validation", "test"})
DEFAULT_RC_TOP_K = 5
DEFAULT_ROOT_DRAW_SEED_BASE = 20_260_721
DEFAULT_TRAJECTORY_SEED_BASE = 42_000
DEFAULT_PISTA_VOCAB = Path("graph_vocab.json")
DEFAULT_PISTA_VOCAB_ID = "pistachio_sa_ge4_graphdit_v1"
# Pista evaluation is driven by frozen per-trajectory seeds.  The stateless
# counter kernel consumes that schedule without a Python loop over categorical
# draws and is invariant to batch partitioning and row order.
DEFAULT_PISTA_PER_SAMPLE_TRANSITION_KERNEL = "stateless_counter_v1"


class PistaTerminalBatchObserver(Protocol):
    """Optional read-only instrumentation for sampled terminal graph batches.

    Primary inference never installs an observer.  Ablations may consume the
    exact product-aligned tensors after sampling and before SMILES decoding,
    but an observer cannot alter the sampler inputs, outputs, or arm record.
    """

    def observe_terminal_batch(
        self,
        *,
        chunk_rows: Sequence[Mapping[str, Any]],
        trajectory_start: int,
        trajectory_count: int,
        rooted_product: Any,
        terminal_state_X: Any,
        terminal_state_E: Any,
    ) -> None: ...


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a non-empty string")
    return value.strip()


def _guidance_alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError("guidance_alpha must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise EvaluationError("guidance_alpha must be a finite number in [0, 1]")
    return result


def deterministic_pista_sample_indices(
    *,
    population_group_count: int,
    sample_group_count: int,
    seed: int,
) -> tuple[int, ...]:
    """Return a source-order-preserving uniform group sample without hashes."""

    population = require_int(
        population_group_count,
        "population_group_count",
        minimum=1,
    )
    sample_count = require_int(
        sample_group_count,
        "sample_group_count",
        minimum=1,
    )
    if sample_count > population:
        raise EvaluationError("sample_group_count cannot exceed population_group_count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EvaluationError("Pista sample seed must be a non-negative integer")
    return tuple(sorted(random.Random(seed).sample(range(population), sample_count)))


def normalize_pista_selected_group_indices(
    *,
    population_group_count: int,
    selected_group_indices: Sequence[int],
) -> tuple[int, ...]:
    """Validate explicit source ordinals without calculating content hashes."""

    population = require_int(
        population_group_count,
        "population_group_count",
        minimum=1,
    )
    selected = tuple(selected_group_indices)
    if not selected:
        raise EvaluationError("selected_group_indices must not be empty")
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= population
        for index in selected
    ):
        raise EvaluationError(
            "selected_group_indices contains an out-of-range source ordinal"
        )
    if len(set(selected)) != len(selected):
        raise EvaluationError("selected_group_indices contains duplicates")
    if selected != tuple(sorted(selected)):
        raise EvaluationError("selected_group_indices must be in source order")
    return selected


def _resolve_pista_transition_kernel(pista_options: Mapping[str, Any]) -> str:
    """Require the single paired, vectorized Pista sampling implementation."""

    kernel = _text(
        pista_options.get(
            "per_sample_transition_kernel",
            DEFAULT_PISTA_PER_SAMPLE_TRANSITION_KERNEL,
        ),
        "execution profile pista.per_sample_transition_kernel",
    )
    if kernel != DEFAULT_PISTA_PER_SAMPLE_TRANSITION_KERNEL:
        raise EvaluationError(
            "Pista inference evaluation requires "
            "per_sample_transition_kernel='stateless_counter_v1'; all arms must use "
            "the same paired vectorized sampling kernel"
        )
    return kernel


def _decode_pista_sampled_terminal_state(
    vocabulary: PistaGraphVocabulary,
    *,
    terminal_state_X: Any,
    terminal_state_E: Any,
    node_mask: Any,
) -> str | None:
    """Decode the hard endpoint reached by the sampled diffusion trajectory.

    ``sample_predefined_graph`` also exposes a final model prediction at
    ``t=1`` for backward-compatible diagnostics.  That prediction is not the
    sampled molecule and must not be used for known-set scoring.
    """

    return vocabulary.decode(terminal_state_X, terminal_state_E, node_mask)


def _path_under(root: Path, value: object, name: str) -> Path:
    raw = _text(value, name)
    candidate = (root / raw).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise EvaluationError(f"{name} escapes corpus root") from error
    return candidate


def _manifest(corpus_root: str | Path) -> tuple[Path, dict[str, Any]]:
    root = Path(corpus_root).expanduser().resolve(strict=True)
    try:
        value = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read Pista corpus manifest under {root}: {error}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError("Pista corpus manifest must be an object")
    return root, dict(value)


def _artifact(root: Path, manifest: Mapping[str, Any], key: str) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationError("Pista corpus manifest has no artifacts mapping")
    return _path_under(root, artifacts.get(key), f"corpus artifacts.{key}")


def _candidate_route_map(
    corpus_root: str | Path,
    *,
    split: str,
    required_route_keys: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], str]:
    if split not in _SPLIT_NAMES:
        raise EvaluationError(f"unsupported Pista split {split!r}")
    root, manifest = _manifest(corpus_root)
    provenance = manifest.get("provenance")
    pools = provenance.get("candidate_pool_paths") if isinstance(provenance, Mapping) else None
    if isinstance(pools, Mapping) and pools.get(split):
        source = Path(_text(pools.get(split), f"candidate_pool_paths.{split}")).expanduser().resolve(strict=True)
        row_candidates = lambda row: row.get("candidates")
    else:
        # The strict-20 execution view stores the frozen candidate routes in
        # its corpus ``cases.<split>`` artifact rather than provenance pool
        # files.  Keep the same route-map semantics while accepting that
        # schema, without deriving or validating any content hashes.
        source = _artifact(root, manifest, f"cases.{split}")
        row_candidates = lambda row: row.get("routes")
    routes: dict[tuple[str, str], str] = {}
    for line_number, row in enumerate(iter_jsonl(source), start=1):
        case_id = _text(row.get("case_id"), f"candidate pool row {line_number}.case_id")
        candidates = row_candidates(row)
        if not isinstance(candidates, list):
            raise EvaluationError(f"candidate pool row {line_number}.candidates must be a list")
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise EvaluationError(
                    f"candidate pool row {line_number}.candidates[{candidate_index}] must be an object"
                )
            route_id = _text(
                candidate.get("candidate_id", candidate.get("route_id")),
                f"candidate pool route {line_number}:{candidate_index}.id",
            )
            key = (case_id, route_id)
            # A Pista frozen manifest only consumes the signed, eligible P/N
            # routes.  Strict-20 case artifacts may also retain neutral or
            # excluded candidates whose precursor strings are intentionally
            # not part of the evaluation contract.  Do not canonicalize those
            # unused candidates, but remain fail-closed for every required
            # route below.
            if required_route_keys is not None and key not in required_route_keys:
                continue
            precursor = canonical_reaction_smiles(
                candidate.get("canonical_precursor_smiles", candidate.get("precursor_smiles"))
            )
            if precursor is None:
                raise EvaluationError(
                    f"candidate pool route {case_id!r}/{route_id!r} has an invalid precursor SMILES"
                )
            previous = routes.setdefault(key, precursor)
            if previous != precursor:
                raise EvaluationError(f"candidate pool has conflicting routes for {key!r}")
    if required_route_keys is not None:
        missing = required_route_keys.difference(routes)
        if missing:
            case_id, route_id = min(missing)
            raise EvaluationError(
                f"Pista candidate route is absent from the frozen pool: {case_id!r}/{route_id!r}"
            )
    if not routes:
        raise EvaluationError(f"candidate route map is empty for Pista split {split!r}")
    return routes


def _products_by_case(corpus_root: str | Path, *, split: str) -> dict[str, str]:
    root, manifest = _manifest(corpus_root)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationError("corpus manifest has no artifacts mapping")
    binding_key = f"case_bindings.{split}"
    if isinstance(artifacts.get(binding_key), str) and artifacts.get(binding_key):
        source = _artifact(root, manifest, binding_key)
        product_key = "product_smiles"
        row_name = "case binding"
    else:
        # Metric-only guidance corpora keep the product on each cases row and
        # do not materialize the execution-view case_bindings artifact.  This
        # fallback preserves the same case->product identity without
        # canonicalizing or rebuilding any historical state.
        source = _artifact(root, manifest, f"cases.{split}")
        product_key = "product_smiles"
        row_name = "case artifact"
    products: dict[str, str] = {}
    for line_number, row in enumerate(iter_jsonl(source), start=1):
        case_id = _text(row.get("case_id"), f"{row_name} row {line_number}.case_id")
        product = _text(row.get(product_key), f"{row_name} row {line_number}.{product_key}")
        previous = products.setdefault(case_id, product)
        if previous != product:
            raise EvaluationError(f"case binding has conflicting products for {case_id!r}")
    if not products:
        raise EvaluationError(f"case binding map is empty for Pista split {split!r}")
    return products


def trajectory_seeds_for_group(
    *,
    evaluation_group_index: int,
    draw_count: int,
    seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
) -> list[int]:
    """Return deterministic, unique per-group trajectory identifiers.

    The Pista sampler uses the same ordered identifiers for every paired arm.
    They are intentionally arithmetic rather than content-derived, in keeping
    with the repository's no-hash integrity policy.
    """

    group_index = require_int(evaluation_group_index, "evaluation_group_index", minimum=0)
    draws = require_int(draw_count, "draw_count", minimum=1)
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise EvaluationError("seed_base must be a non-negative integer")
    start = seed_base + group_index * draws
    return list(range(start, start + draws))


def root_draw_seeds_for_group(
    *,
    evaluation_group_index: int,
    draw_count: int,
    seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
) -> list[int]:
    """Return the frozen Pista root-draw seed schedule for one group."""

    group_index = require_int(evaluation_group_index, "evaluation_group_index", minimum=0)
    draws = require_int(draw_count, "draw_count", minimum=1)
    if isinstance(seed_base, bool) or not isinstance(seed_base, int) or seed_base < 0:
        raise EvaluationError("root_draw_seed_base must be a non-negative integer")
    start = seed_base + group_index * draws
    return list(range(start, start + draws))


def root_choice_indices_for_group(
    *,
    root_draw_seeds: Sequence[int],
    rc_top_k: int = DEFAULT_RC_TOP_K,
) -> list[int]:
    """Uniformly draw frozen RC slots with replacement from per-draw seeds."""

    if isinstance(rc_top_k, bool) or not isinstance(rc_top_k, int) or rc_top_k < 1:
        raise EvaluationError("rc_top_k must be a positive integer")
    if not root_draw_seeds:
        raise EvaluationError("root_draw_seeds must not be empty")
    choices: list[int] = []
    for seed in root_draw_seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluationError("root_draw_seeds must contain integers")
        choices.append(random.Random(seed).randrange(rc_top_k))
    return choices


def _known_sets(
    *,
    group: Any,
    routes: Mapping[tuple[str, str], str],
) -> tuple[list[str], list[str]]:
    positive: set[str] = set()
    negative: set[str] = set()
    for preference in group.candidate_preferences:
        if not preference.eligible or preference.label == 0:
            continue
        try:
            route = routes[(group.case_id, preference.route_id)]
        except KeyError as error:
            raise EvaluationError(
                f"Pista candidate route is absent from the frozen pool: "
                f"{group.case_id!r}/{preference.route_id!r}"
            ) from error
        if preference.label == 1:
            positive.add(route)
        elif preference.label == -1:
            negative.add(route)
        else:  # SemanticGroup validates this, but retain a defensive boundary.
            raise EvaluationError(f"unsupported Pista preference label {preference.label!r}")
    if not positive or not negative:
        raise EvaluationError(
            f"Pista group {group.semantic_group_uid!r} lacks a positive or negative known set"
        )
    if positive.intersection(negative):
        raise EvaluationError(f"Pista group {group.semantic_group_uid!r} has overlapping P/N sets")
    return sorted(positive), sorted(negative)


def _eligible_signed_route_keys(
    runtime: GuidanceRuntimeIndex,
    *,
    split: str,
    group_indices: Sequence[int] | None = None,
) -> set[tuple[str, str]]:
    """Return precisely the corpus routes that Pista known sets can consume."""

    total_groups = runtime.group_count(split)
    if group_indices is None:
        selected_indices = tuple(range(total_groups))
    else:
        selected_indices = tuple(group_indices)
        if not selected_indices:
            raise EvaluationError("Pista selected group indices must not be empty")
        if len(set(selected_indices)) != len(selected_indices):
            raise EvaluationError("Pista selected group indices contain duplicates")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= total_groups
            for index in selected_indices
        ):
            raise EvaluationError("Pista selected group index lies outside the requested split")
    route_keys: set[tuple[str, str]] = set()
    for ordinal in selected_indices:
        group = runtime.group(split, ordinal)
        for preference in group.candidate_preferences:
            if preference.eligible and preference.label in {-1, 1}:
                route_keys.add((group.case_id, preference.route_id))
    if not route_keys:
        raise EvaluationError(f"Pista split {split!r} has no eligible signed candidate routes")
    return route_keys


def build_pista_frozen_rows(
    *,
    corpus_root: str | Path,
    rc_prediction_table: str | Path,
    rc_predictor_checkpoint: str | Path,
    split: str = "test",
    draw_count: int = 100,
    training_seed: int = 42,
    evaluation_seed: int = 1042,
    rc_top_k: int = DEFAULT_RC_TOP_K,
    root_draw_seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
    trajectory_seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
    expected_group_count: int | None = None,
    expected_case_count: int | None = None,
    sample_group_count: int | None = None,
    sample_seed: int = 42,
    selected_group_indices: Sequence[int] | None = None,
    shuffle_donor_axis_ids: Sequence[str] | None = None,
    known_set_corpus_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Freeze Pista P/N sets, RC roots, and the paired trajectory schedule.

    When ``sample_group_count`` is set, choose that many groups uniformly
    without replacement from the requested split, retain their source-order
    indices, and derive their paired seed schedules from those original
    indices.  This permits small train-split diagnostics without inventing a
    new seed namespace or requiring RC entries for unrelated groups.

    ``known_set_corpus_root`` may name the immutable backing cases corpus when
    an execution view deliberately stores only graph-bound semantic groups.
    Runtime guidance/text/shuffle identity still comes from ``corpus_root``.
    """

    if split not in _SPLIT_NAMES:
        raise EvaluationError(f"unsupported Pista split {split!r}")
    draws = require_int(draw_count, "draw_count", minimum=1)
    if isinstance(rc_top_k, bool) or not isinstance(rc_top_k, int) or rc_top_k < 1:
        raise EvaluationError("rc_top_k must be a positive integer")
    predictor_checkpoint = Path(rc_predictor_checkpoint).expanduser().resolve(strict=True)
    rc_table = FrozenRCTable.from_json(rc_prediction_table)
    _, corpus_manifest = _manifest(corpus_root)
    branches = corpus_manifest.get("branches")
    branch_weights = {"metric": 0.5, "structure": 0.5}
    if isinstance(branches, Mapping):
        enabled = [name for name in ("metric", "structure") if bool(branches.get(name))]
        if enabled:
            branch_weights = {name: (1.0 if name in enabled else 0.0) for name in ("metric", "structure")}
    runtime = GuidanceRuntimeIndex.from_corpus(
        corpus_root,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        branch_weights=branch_weights,
        sampling_mode="branch_cycle",
    )
    if shuffle_donor_axis_ids is not None:
        allowed_axes = tuple(str(axis_id).strip() for axis_id in shuffle_donor_axis_ids)
        if not allowed_axes or any(not axis_id for axis_id in allowed_axes):
            raise EvaluationError("shuffle_donor_axis_ids must contain non-empty strings")
        if len(set(allowed_axes)) != len(allowed_axes):
            raise EvaluationError("shuffle_donor_axis_ids must be unique")
        # Source rows may come from a larger metric-only corpus than the
        # checkpoint's token cache.  Restrict only the shuffled donor pool to
        # the explicitly configured axis universe; recipient groups and their
        # candidate supervision remain untouched.  This keeps the
        # different-axis control legal while preventing a frozen row from
        # naming a guidance surface that the model cannot embed.
        root, corpus_manifest = _manifest(corpus_root)
        donor_path = _artifact(root, corpus_manifest, f"shuffle_donor_pools.{split}")
        donors = tuple(iter_models(donor_path, ShuffleDonorPool))
        filtered_donors = tuple(
            donor for donor in donors if donor.axis_id in set(allowed_axes)
        )
        if not filtered_donors:
            raise EvaluationError(
                "shuffle_donor_axis_ids do not match any donor pools for the requested split"
            )
        shuffle_policy = corpus_manifest.get("shuffle_policy")
        metric_source_weights = {"metric": 1.0, "structure": 0.0}
        if isinstance(shuffle_policy, Mapping):
            raw_weights = shuffle_policy.get("metric_source_weights")
            if isinstance(raw_weights, Mapping):
                metric_source_weights = {
                    "metric": float(raw_weights.get("metric", 0.0)),
                    "structure": float(raw_weights.get("structure", 0.0)),
                }
        split_runtime = runtime._split_runtime[split]  # type: ignore[attr-defined]
        runtime._split_runtime[split] = replace(  # type: ignore[attr-defined]
            split_runtime,
            shuffle_sampler=RuntimeShuffleSampler(
                donor_pools=filtered_donors,
                seed=training_seed,
                evaluation_seed=evaluation_seed,
                metric_source_weights=metric_source_weights,
            ),
        )
    population_group_count = runtime.group_count(split)
    if sample_group_count is not None and selected_group_indices is not None:
        raise EvaluationError(
            "sample_group_count and selected_group_indices are mutually exclusive"
        )
    if selected_group_indices is not None:
        selected_ordinals = normalize_pista_selected_group_indices(
            population_group_count=population_group_count,
            selected_group_indices=selected_group_indices,
        )
        sample_selection: dict[str, Any] | None = {
            "mode": "explicit_source_ordinals_v1",
            "population_group_count": population_group_count,
            "sample_group_count": len(selected_ordinals),
            "content_hashes_computed": False,
        }
    elif sample_group_count is None:
        selected_ordinals = tuple(range(population_group_count))
        sample_selection: dict[str, Any] | None = None
    else:
        selected_ordinals = deterministic_pista_sample_indices(
            population_group_count=population_group_count,
            sample_group_count=sample_group_count,
            seed=sample_seed,
        )
        sample_selection = {
            "mode": "uniform_without_replacement_python_random_v1",
            "population_group_count": population_group_count,
            "sample_group_count": len(selected_ordinals),
            "sample_seed": sample_seed,
        }
    known_set_root = (
        corpus_root if known_set_corpus_root is None else known_set_corpus_root
    )
    routes = _candidate_route_map(
        known_set_root,
        split=split,
        required_route_keys=_eligible_signed_route_keys(
            runtime,
            split=split,
            group_indices=selected_ordinals,
        ),
    )
    products = _products_by_case(known_set_root, split=split)
    rows: list[dict[str, Any]] = []
    for ordinal in selected_ordinals:
        condition = runtime.condition(
            split=split,
            group_ordinal=ordinal,
            include_shuffled=True,
            evaluation=True,
        )
        if condition.shuffled is None:  # pragma: no cover - required above
            raise EvaluationError(f"Pista group {condition.group.semantic_group_uid!r} lacks shuffled guidance")
        group = condition.group
        try:
            product = _text(products[group.case_id], f"Pista product for {group.case_id!r}")
        except KeyError as error:
            raise EvaluationError(f"Pista group references a missing product: {group.case_id!r}") from error
        positive, negative = _known_sets(group=group, routes=routes)
        signed_candidate_count = sum(
            int(preference.eligible and preference.label in {-1, 1})
            for preference in group.candidate_preferences
        )
        if signed_candidate_count < 1:  # Defensive: _known_sets already rules this out.
            raise EvaluationError(f"Pista group {group.semantic_group_uid!r} has no signed candidates")
        rc_resolution = rc_table.resolve(product, top_k=rc_top_k)
        root_draw_seeds = root_draw_seeds_for_group(
            evaluation_group_index=ordinal,
            draw_count=draws,
            seed_base=root_draw_seed_base,
        )
        trajectory_seeds = trajectory_seeds_for_group(
            evaluation_group_index=ordinal,
            draw_count=draws,
            seed_base=trajectory_seed_base,
        )
        row: dict[str, Any] = {
                "schema_version": PISTA_FROZEN_SCHEMA,
                "source_split": split,
                "group_uid": group.semantic_group_uid,
                "semantic_family_uid": group.semantic_family_uid,
                "case_id": group.case_id,
                "evaluation_group_index": ordinal,
                "legacy_group_idx": ordinal,
                "product_smiles": product,
                "canonical_product_smiles": rc_resolution.canonical_product_smiles,
                "branch": group.branch,
                "axis_id": group.axis_id,
                "polarity": group.polarity,
                "positive_target_smiles": positive,
                "negative_target_smiles": negative,
                "signed_candidate_count": signed_candidate_count,
                "true_guidance_text_id": condition.true_text.text_id,
                "true_guidance_text": condition.true_text.text,
                "true_guidance_relation": condition.true_relation,
                "shuffled_guidance_text_id": condition.shuffled.donor_text_id,
                "shuffled_mapping": condition.shuffled.to_dict(),
                "rc_predictor": {
                    "checkpoint": str(predictor_checkpoint),
                    "prediction_table": str(rc_table.source),
                    "source_condition_key": rc_resolution.source_condition_key,
                    "resolution_kind": rc_resolution.resolution_kind,
                },
                "configured_rc_top_k": rc_top_k,
                "effective_rc_k": rc_top_k,
                "unique_available_rc_k": rc_resolution.unique_available_rc_k,
                "rc_candidates": [dict(candidate) for candidate in rc_resolution.rc_candidates],
                "root_draw_seeds": root_draw_seeds,
                "root_choice_indices": root_choice_indices_for_group(
                    root_draw_seeds=root_draw_seeds,
                    rc_top_k=rc_top_k,
                ),
                "trajectory_seeds": trajectory_seeds,
                "canonicalization_policy": CANONICALIZATION_POLICY,
                "integrity_policy": INTEGRITY_POLICY,
        }
        if sample_selection is not None:
            row["sample_selection"] = dict(sample_selection)
        rows.append(row)
    if expected_group_count is not None and len(rows) != require_int(
        expected_group_count, "expected_group_count", minimum=1
    ):
        raise EvaluationError("Pista frozen group count differs from expected_group_count")
    case_count = len({str(row["case_id"]) for row in rows})
    if expected_case_count is not None and case_count != require_int(
        expected_case_count, "expected_case_count", minimum=1
    ):
        raise EvaluationError("Pista frozen case count differs from expected_case_count")
    return rows


def write_pista_frozen_manifest(
    *,
    corpus_root: str | Path,
    rc_prediction_table: str | Path,
    rc_predictor_checkpoint: str | Path,
    output_jsonl: str | Path,
    split: str = "test",
    draw_count: int = 100,
    training_seed: int = 42,
    evaluation_seed: int = 1042,
    rc_top_k: int = DEFAULT_RC_TOP_K,
    root_draw_seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
    trajectory_seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
    expected_group_count: int | None = None,
    expected_case_count: int | None = None,
    sample_group_count: int | None = None,
    sample_seed: int = 42,
    selected_group_indices: Sequence[int] | None = None,
) -> Path:
    rows = build_pista_frozen_rows(
        corpus_root=corpus_root,
        rc_prediction_table=rc_prediction_table,
        rc_predictor_checkpoint=rc_predictor_checkpoint,
        split=split,
        draw_count=draw_count,
        training_seed=training_seed,
        evaluation_seed=evaluation_seed,
        rc_top_k=rc_top_k,
        root_draw_seed_base=root_draw_seed_base,
        trajectory_seed_base=trajectory_seed_base,
        expected_group_count=expected_group_count,
        expected_case_count=expected_case_count,
        sample_group_count=sample_group_count,
        sample_seed=sample_seed,
        selected_group_indices=selected_group_indices,
    )
    return write_new_jsonl(output_jsonl, rows)


def validate_pista_frozen_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_draw_count: int = 100,
    expected_rc_top_k: int = DEFAULT_RC_TOP_K,
    root_draw_seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
    trajectory_seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
) -> dict[str, int]:
    """Fail closed unless every row carries a complete paired Pista RC plan."""

    draws = require_int(expected_draw_count, "expected_draw_count", minimum=1)
    top_k = require_int(expected_rc_top_k, "expected_rc_top_k", minimum=1)
    if not rows:
        raise EvaluationError("Pista frozen manifest is empty")
    seen_indices: set[int] = set()
    cases: set[str] = set()
    padded_groups = 0
    for row_position, row in enumerate(rows):
        if row.get("schema_version") != PISTA_FROZEN_SCHEMA:
            raise EvaluationError(f"Pista frozen row {row_position} has an unsupported schema")
        index = require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
        if index in seen_indices:
            raise EvaluationError("Pista frozen manifest has duplicate evaluation_group_index values")
        seen_indices.add(index)
        case_id = _text(row.get("case_id"), "case_id")
        cases.add(case_id)
        _text(row.get("source_split"), "source_split")
        _text(row.get("group_uid"), "group_uid")
        _text(row.get("axis_id"), "axis_id")
        _text(row.get("true_guidance_text_id"), "true_guidance_text_id")
        _text(row.get("true_guidance_text"), "true_guidance_text")
        if _text(row.get("true_guidance_relation"), "true_guidance_relation") != "canonical_true":
            raise EvaluationError("Pista frozen row lacks canonical true guidance")
        shuffled = row.get("shuffled_mapping")
        if not isinstance(shuffled, Mapping):
            raise EvaluationError("Pista frozen row lacks a shuffled mapping")
        if _text(shuffled.get("recipient_group_uid"), "shuffled_mapping.recipient_group_uid") != row.get("group_uid"):
            raise EvaluationError("Pista frozen shuffled recipient differs from group identity")
        if _text(shuffled.get("relation"), "shuffled_mapping.relation") != "different_axis":
            raise EvaluationError("Pista frozen shuffled mapping is not a different-axis control")
        if _text(shuffled.get("donor_axis_id"), "shuffled_mapping.donor_axis_id") == row.get("axis_id"):
            raise EvaluationError("Pista frozen shuffled donor shares the recipient axis")
        if _text(shuffled.get("donor_text_id"), "shuffled_mapping.donor_text_id") != _text(
            row.get("shuffled_guidance_text_id"), "shuffled_guidance_text_id"
        ):
            raise EvaluationError("Pista frozen shuffled text ID differs from mapping")
        if "signed_candidate_count" in row:
            require_int(row.get("signed_candidate_count"), "signed_candidate_count", minimum=1)
        if row.get("configured_rc_top_k") != top_k or row.get("effective_rc_k") != top_k:
            raise EvaluationError("Pista frozen RC top-k differs from the formal contract")
        candidates = row.get("rc_candidates")
        if not isinstance(candidates, list) or len(candidates) != top_k:
            raise EvaluationError("Pista frozen RC candidate slots are incomplete")
        canonical_product = _text(row.get("canonical_product_smiles"), "canonical_product_smiles")
        atom_count = product_atom_count(canonical_product)
        for slot, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise EvaluationError("Pista frozen RC candidate must be an object")
            permutation = candidate.get("permutation")
            if (
                not isinstance(permutation, list)
                or len(permutation) != atom_count
                or any(isinstance(value, bool) or not isinstance(value, int) for value in permutation)
                or sorted(permutation) != list(range(len(permutation)))
            ):
                raise EvaluationError("Pista frozen RC candidate permutation is invalid")
            if candidate.get("rc_slot_index") != slot:
                raise EvaluationError("Pista frozen RC candidate slot index drifted")
            if not isinstance(candidate.get("predictor_rank"), int) or candidate["predictor_rank"] < 1:
                raise EvaluationError("Pista frozen RC candidate predictor rank is invalid")
            score = candidate.get("predictor_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)) or float(score) <= 0.0:
                raise EvaluationError("Pista frozen RC candidate predictor score is invalid")
            if not isinstance(candidate.get("rc_permutation_key"), str) or not candidate["rc_permutation_key"]:
                raise EvaluationError("Pista frozen RC candidate key is invalid")
        if any(bool(candidate.get("is_padding_duplicate")) for candidate in candidates):
            padded_groups += 1
        root_draw_seeds = row.get("root_draw_seeds")
        trajectory_seeds = row.get("trajectory_seeds")
        choices = row.get("root_choice_indices")
        if (
            not isinstance(root_draw_seeds, list)
            or not isinstance(trajectory_seeds, list)
            or not isinstance(choices, list)
            or len(root_draw_seeds) != draws
            or len(trajectory_seeds) != draws
            or len(choices) != draws
        ):
            raise EvaluationError("Pista frozen root/trajectory schedule is incomplete")
        expected_root_draws = root_draw_seeds_for_group(
            evaluation_group_index=index,
            draw_count=draws,
            seed_base=root_draw_seed_base,
        )
        expected_trajectories = trajectory_seeds_for_group(
            evaluation_group_index=index,
            draw_count=draws,
            seed_base=trajectory_seed_base,
        )
        if root_draw_seeds != expected_root_draws or trajectory_seeds != expected_trajectories:
            raise EvaluationError("Pista frozen seed schedule differs from the formal contract")
        if choices != root_choice_indices_for_group(root_draw_seeds=root_draw_seeds, rc_top_k=top_k):
            raise EvaluationError("Pista frozen root draws differ from the formal uniform schedule")
    return {
        "group_count": len(rows),
        "case_count": len(cases),
        "padded_group_count": padded_groups,
    }


def _evenly_spaced(rows: Sequence[Mapping[str, Any]], count: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if count < 1 or len(rows) < count:
        raise EvaluationError("cannot select the requested stratified Pista smoke rows")
    if count == 1:
        return [dict(rows[0])]
    positions = [(position * (len(rows) - 1)) // (count - 1) for position in range(count)]
    return [dict(rows[position]) for position in positions]


def select_pista_smoke_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    *,
    metric_count: int = 48,
    structure_count: int = 16,
) -> list[dict[str, Any]]:
    """Take a deterministic branch-stratified, order-preserving smoke cohort."""

    metric = [row for row in frozen_rows if row.get("branch") == "metric"]
    structure = [row for row in frozen_rows if row.get("branch") == "structure"]
    selected = _evenly_spaced(metric, metric_count) + _evenly_spaced(structure, structure_count)
    selected.sort(key=lambda row: require_int(row.get("evaluation_group_index"), "evaluation_group_index"))
    if len({row.get("group_uid") for row in selected}) != len(selected):
        raise EvaluationError("Pista smoke cohort contains duplicate groups")
    return selected


def write_pista_smoke_manifest(
    *,
    frozen_manifest: str | Path,
    output_jsonl: str | Path,
    metric_count: int = 48,
    structure_count: int = 16,
) -> Path:
    return write_new_jsonl(
        output_jsonl,
        select_pista_smoke_rows(
            read_rows(frozen_manifest),
            metric_count=metric_count,
            structure_count=structure_count,
        ),
    )


def partition_pista_frozen_rows(
    frozen_rows: Sequence[Mapping[str, Any]],
    *,
    metric_count: int,
    structure_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Make a deterministic early cohort and its exact non-overlapping remainder.

    The selected rows retain their original ``evaluation_group_index`` values,
    so their frozen RC/root/trajectory schedules are unchanged when their raw
    arm outputs are later merged with the remainder.
    """

    selected = select_pista_smoke_rows(
        frozen_rows,
        metric_count=metric_count,
        structure_count=structure_count,
    )
    selected_indices = {
        require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
        for row in selected
    }
    remaining = [
        dict(row)
        for row in frozen_rows
        if require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
        not in selected_indices
    ]
    if len(selected) + len(remaining) != len(frozen_rows):
        raise EvaluationError("Pista early cohort partition does not cover the frozen manifest")
    if not remaining:
        raise EvaluationError("Pista early cohort cannot consume the complete frozen manifest")
    return selected, remaining


def write_pista_frozen_partition(
    *,
    frozen_manifest: str | Path,
    selected_output_jsonl: str | Path,
    remaining_output_jsonl: str | Path,
    metric_count: int,
    structure_count: int,
) -> dict[str, Path]:
    """Persist an immutable selected/remainder partition of a frozen manifest."""

    selected_output = Path(selected_output_jsonl).expanduser().resolve()
    remaining_output = Path(remaining_output_jsonl).expanduser().resolve()
    if selected_output == remaining_output:
        raise EvaluationError("Pista partition outputs must be distinct")
    collisions = [
        str(path)
        for path in (selected_output, remaining_output)
        if path.exists() or path.is_symlink()
    ]
    if collisions:
        raise EvaluationError("refusing Pista partition output collision: " + ", ".join(collisions))
    selected, remaining = partition_pista_frozen_rows(
        read_rows(frozen_manifest),
        metric_count=metric_count,
        structure_count=structure_count,
    )
    return {
        "selected": write_new_jsonl(selected_output, selected),
        "remaining": write_new_jsonl(remaining_output, remaining),
    }


def _load_formal_checkpoint(path: str | Path) -> Mapping[str, Any]:
    import torch

    source = Path(path).expanduser().resolve(strict=True)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch versions
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise EvaluationError(f"Pista checkpoint is not a mapping: {source}")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise EvaluationError(f"Pista checkpoint has no state_dict: {source}")
    return payload


def _load_pista_model(spec: PistaTrainingSpec, checkpoint: str | Path) -> Any:
    # The training builder owns the exact Stage-B wrapper construction.  The
    # adapter only loads a completed candidate state into that immutable shape.
    from retro_guided_generation.training.pista import _build_pista_stage

    model, _initialization = _build_pista_stage(spec)
    payload = _load_formal_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise EvaluationError(
            "Pista checkpoint state differs from its training configuration: "
            f"missing={list(missing)[:8]} unexpected={list(unexpected)[:8]}"
        )
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise EvaluationError("Pista inference model parameters did not freeze")
    model.eval()
    return model


@dataclass(frozen=True, slots=True)
class _PistaEvaluationContext:
    """Read-only text state needed by rooted-product Pista inference.

    A Pista frozen manifest already records the resolved canonical and
    shuffled text relation for every group.  Reconstructing
    ``GuidanceRuntimeIndex`` here would reparse the complete semantic-group
    corpus once per CUDA process without changing a single generated sample.
    The context therefore owns only the lightweight text pools/token cache;
    frozen rows are validated and rebuilt below.
    """

    split: str
    text_tokens: PistaTextTokenMapping
    max_groups_per_batch: int
    max_candidates_per_batch: int
    guidance_context_mode: str = "frozen_manifest_relations_v1"


@dataclass(frozen=True, slots=True)
class _FrozenPistaText:
    """Minimal resolved text reference required by the sampling adapter."""

    text_id: str


@dataclass(frozen=True, slots=True)
class _FrozenPistaShuffle:
    """Minimal shuffled-donor reference required by the sampling adapter."""

    donor_text_id: str


@dataclass(frozen=True, slots=True)
class _FrozenPistaGroup:
    """Frozen group identity and bounded candidate count for batching only."""

    semantic_group_uid: str
    case_id: str
    signed_candidate_count: int


@dataclass(frozen=True, slots=True)
class _FrozenPistaCondition:
    """Self-contained counterpart of the runtime condition used at inference.

    It deliberately has only the fields consumed by Pista generation.  The
    group count is never a model input; it solely keeps a generation chunk
    within the explicit candidate-memory contract.
    """

    group: _FrozenPistaGroup
    true_text: _FrozenPistaText
    true_relation: str
    shuffled: _FrozenPistaShuffle


def _evaluation_context(
    spec: PistaTrainingSpec,
    *,
    split: str,
    branches: Sequence[str],
) -> _PistaEvaluationContext:
    token_memory = PistaTokenMemory.from_paths(
        token_cache_root=spec.token_cache_root,
        guidance_text_table=spec.guidance_text_table,
        token_extension_root=spec.token_extension_root,
        expected_hidden_dim=spec.guidance_text_dim,
    )
    return _PistaEvaluationContext(
        split=split,
        text_tokens=token_memory.bind_text_pools(
            load_corpus_text_pools(spec.corpus_root, branches=branches, splits=(split,))
        ),
        max_groups_per_batch=spec.max_groups_per_batch,
        max_candidates_per_batch=spec.max_candidates_per_batch,
    )


def _rebuild_condition(
    context: _PistaEvaluationContext,
    row: Mapping[str, Any],
    *,
    fallback_signed_candidate_upper_bound: int,
) -> _FrozenPistaCondition:
    """Validate and rebuild a resolved condition directly from a frozen row.

    The manifest is generated by the canonical runtime before any GPU work and
    records both arms' exact text IDs and the full shuffled provenance.  This
    avoids a redundant full-corpus reload while retaining fail-closed checks
    over every relation Pista actually consumes.
    """

    source_split = _text(row.get("source_split"), "source_split")
    if source_split != context.split:
        raise EvaluationError("Pista frozen row split differs from inference split")
    _ = require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
    group_uid = _text(row.get("group_uid"), "group_uid")
    case_id = _text(row.get("case_id"), "case_id")
    true_text_id = _text(row.get("true_guidance_text_id"), "true_guidance_text_id")
    _text(row.get("true_guidance_text"), "true_guidance_text")
    true_relation = _text(row.get("true_guidance_relation"), "true_guidance_relation")
    if true_relation != "canonical_true":
        raise EvaluationError("Pista frozen row lacks canonical true guidance")
    mapping = row.get("shuffled_mapping")
    if not isinstance(mapping, Mapping):
        raise EvaluationError("Pista frozen row lacks a shuffled mapping")
    if _text(mapping.get("recipient_group_uid"), "shuffled_mapping.recipient_group_uid") != group_uid:
        raise EvaluationError("Pista frozen shuffled recipient differs from group identity")
    if _text(mapping.get("relation"), "shuffled_mapping.relation") != "different_axis":
        raise EvaluationError("Pista frozen shuffled mapping is not a different-axis control")
    donor_axis_id = _text(mapping.get("donor_axis_id"), "shuffled_mapping.donor_axis_id")
    if donor_axis_id == _text(row.get("axis_id"), "axis_id"):
        raise EvaluationError("Pista frozen shuffled donor shares the recipient axis")
    donor_text_id = _text(mapping.get("donor_text_id"), "shuffled_mapping.donor_text_id")
    if donor_text_id != _text(row.get("shuffled_guidance_text_id"), "shuffled_guidance_text_id"):
        raise EvaluationError("Pista frozen shuffled text ID differs from mapping")
    candidate_count_value = row.get("signed_candidate_count")
    if candidate_count_value is None:
        candidate_count = require_int(
            fallback_signed_candidate_upper_bound,
            "frozen_manifest_candidate_upper_bound",
            minimum=1,
        )
    else:
        candidate_count = require_int(
            candidate_count_value,
            "signed_candidate_count",
            minimum=1,
        )
    return _FrozenPistaCondition(
        group=_FrozenPistaGroup(
            semantic_group_uid=group_uid,
            case_id=case_id,
            signed_candidate_count=candidate_count,
        ),
        true_text=_FrozenPistaText(text_id=true_text_id),
        true_relation=true_relation,
        shuffled=_FrozenPistaShuffle(donor_text_id=donor_text_id),
    )


def _signed_candidate_count(condition: Any) -> int:
    frozen_count = getattr(condition.group, "signed_candidate_count", None)
    if frozen_count is not None:
        return require_int(frozen_count, "frozen signed candidate count", minimum=1)
    return sum(
        int(preference.eligible and preference.label in {-1, 1})
        for preference in condition.group.candidate_preferences
    )


def _condition_chunks(
    context: _PistaEvaluationContext,
    pairs: Sequence[tuple[Mapping[str, Any], Any]],
    *,
    max_groups: int,
) -> Iterable[Sequence[tuple[Mapping[str, Any], Any]]]:
    group_limit = require_int(max_groups, "max_groups_per_generation_batch", minimum=1)
    start = 0
    while start < len(pairs):
        total_candidates = 0
        stop = start
        while stop < len(pairs):
            if stop - start >= group_limit:
                break
            candidates = _signed_candidate_count(pairs[stop][1])
            if candidates < 1:
                raise EvaluationError("Pista formal group has no signed candidates")
            if stop == start and candidates > context.max_candidates_per_batch:
                raise EvaluationError("one Pista group exceeds max_candidates_per_batch")
            if stop > start and total_candidates + candidates > context.max_candidates_per_batch:
                break
            total_candidates += candidates
            stop += 1
        yield pairs[start:stop]
        start = stop


def _move_cpu_tensor(value: Any, *, device: Any) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise EvaluationError("Pista evaluation staging value must be a tensor")
    if value.device == device:
        return value
    if device.type == "cuda" and value.device.type == "cpu" and value.numel() > 0:
        if not value.is_pinned():
            value = value.pin_memory()
        return value.to(device=device, non_blocking=True)
    return value.to(device=device)


def _sampling_model_for_arm(model: Any, *, arm: str) -> Any:
    """Select the exact alpha-zero model path for one inference arm."""

    if arm in {"reference", "null"}:
        # The student is h + alpha*h'.  At alpha=0 the exact model is h, so
        # Null must use the reference path and must not require text tokens.
        return model.reference
    if arm in {"true", "shuffled"}:
        return model.student
    raise EvaluationError(f"unsupported Pista arm: {arm!r}")


def _guidance_token_batch(
    context: _PistaEvaluationContext,
    conditions: Sequence[Any],
    *,
    arm: str,
    device: Any,
) -> tuple[Any, Any] | None:
    """Load only frozen text memories; never materialize endpoint graphs."""

    if arm == "reference":
        return None
    if arm not in {"null", "true", "shuffled"}:
        raise EvaluationError("Pista token batch requires null, true, or shuffled guidance")
    import torch

    text_ids: list[str] = []
    for condition in conditions:
        if arm in {"null", "true"}:
            text_ids.append(condition.true_text.text_id)
        else:
            if condition.shuffled is None:
                raise EvaluationError("Pista shuffled arm lacks a frozen donor")
            text_ids.append(condition.shuffled.donor_text_id)
    tokens: list[Any] = []
    for text_id in text_ids:
        try:
            value = context.text_tokens[text_id]
        except KeyError as error:
            raise EvaluationError(f"Pista token cache lacks frozen text {text_id!r}") from error
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.ndim != 2
            or value.shape[0] < 1
            or value.shape[1] < 1
            or not value.is_floating_point()
            or value.requires_grad
            or not bool(torch.isfinite(value.float()).all())
        ):
            raise EvaluationError("Pista frozen token memory has an invalid tensor contract")
        tokens.append(value)
    if not tokens:
        raise EvaluationError("Pista token batch cannot be empty")
    width = int(tokens[0].shape[1])
    dtype = tokens[0].dtype
    if any(int(value.shape[1]) != width or value.dtype != dtype for value in tokens):
        raise EvaluationError("Pista frozen token memories disagree on width or dtype")
    max_tokens = max(int(value.shape[0]) for value in tokens)
    batch = torch.zeros((len(tokens), max_tokens, width), dtype=dtype)
    mask = torch.zeros((len(tokens), max_tokens), dtype=torch.bool)
    for index, value in enumerate(tokens):
        length = int(value.shape[0])
        batch[index, :length] = value
        mask[index, :length] = True
    return _move_cpu_tensor(batch, device=device), _move_cpu_tensor(mask, device=device)


def _rooted_product_batch(
    builder: RootedProductGraphBuilder,
    rows: Sequence[Mapping[str, Any]],
    *,
    trajectory_start: int,
    count: int,
    device: Any,
) -> Any:
    """Construct group-major rooted inputs for one bounded trajectory block."""

    start = require_int(trajectory_start, "trajectory_start", minimum=0)
    draw_count = require_int(count, "trajectory count", minimum=1)
    products: list[str] = []
    permutations: list[Sequence[int]] = []
    for row in rows:
        product = _text(row.get("canonical_product_smiles"), "canonical_product_smiles")
        candidates = row.get("rc_candidates")
        choices = row.get("root_choice_indices")
        if not isinstance(candidates, list) or not isinstance(choices, list):
            raise EvaluationError("Pista frozen row lacks a complete RC schedule")
        if start + draw_count > len(choices):
            raise EvaluationError("Pista rooted block exceeds frozen root schedule")
        for root_slot in choices[start : start + draw_count]:
            if isinstance(root_slot, bool) or not isinstance(root_slot, int) or not 0 <= root_slot < len(candidates):
                raise EvaluationError("Pista frozen root slot is outside RC candidates")
            candidate = candidates[root_slot]
            if not isinstance(candidate, Mapping) or not isinstance(candidate.get("permutation"), list):
                raise EvaluationError("Pista frozen RC candidate lacks a permutation")
            products.append(product)
            permutations.append(candidate["permutation"])
    return builder.build(products, permutations=permutations).to_device(device)


def _autocast(device: Any) -> Any:
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def _new_partial_path(output: Path) -> Path:
    if output.exists() or output.is_symlink():
        raise EvaluationError(f"refusing to overwrite Pista arm output: {output}")
    ensure_output_directory(output.parent)
    return output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")


def _publish_partial(partial: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise EvaluationError(f"Pista arm output appeared during generation: {output}")
    os.replace(partial, output)


def _selected_range(total: int, *, shard_index: int, shard_count: int) -> tuple[int, int]:
    index = require_int(shard_index, "shard_index", minimum=0)
    count = require_int(shard_count, "shard_count", minimum=1)
    if index >= count:
        raise EvaluationError("shard_index must be smaller than shard_count")
    return total * index // count, total * (index + 1) // count


@dataclass(slots=True)
class PistaInferenceSession:
    """Reusable, read-only Pista inference state for one CUDA process.

    A session deliberately owns only immutable scientific inputs: the loaded
    checkpoint, frozen manifest, vocabulary, and guidance runtime.  It can
    emit one or several arms sequentially without reparsing the corpus/token
    cache or reloading model weights.  Independent CUDA ranks still use
    disjoint frozen-manifest shards; inference never needs DDP all-reduce.
    """

    spec: PistaTrainingSpec
    rows: list[dict[str, Any]]
    context: _PistaEvaluationContext
    vocabulary: PistaGraphVocabulary
    model: Any
    target_device: Any

    @classmethod
    def open(
        cls,
        *,
        training_config: str | Path,
        checkpoint: str | Path,
        frozen_manifest: str | Path,
        split: str = "test",
        expected_draw_count: int = 100,
        root_draw_seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
        trajectory_seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
        device: str = "cuda",
        vocabulary_path: str | Path = DEFAULT_PISTA_VOCAB,
        vocabulary_id: str = DEFAULT_PISTA_VOCAB_ID,
    ) -> "PistaInferenceSession":
        """Load one reusable CUDA-local session without changing any output."""

        import torch

        target_device = torch.device(device)
        if target_device.type != "cuda" or not torch.cuda.is_available():
            raise EvaluationError("Pista formal inference requires a CUDA device")
        if split not in _SPLIT_NAMES:
            raise EvaluationError(f"unsupported Pista inference split {split!r}")
        rows = read_rows(frozen_manifest)
        if not rows:
            raise EvaluationError("Pista frozen manifest is empty")
        validate_pista_frozen_rows(
            rows,
            expected_draw_count=expected_draw_count,
            root_draw_seed_base=root_draw_seed_base,
            trajectory_seed_base=trajectory_seed_base,
        )
        spec = load_pista_training_spec(training_config)
        if spec.stage != "stage-b":
            raise EvaluationError("Pista generic inference adapter currently requires a stage-b config")
        frozen_branches = tuple(sorted({_text(row.get("branch"), "branch") for row in rows}))
        if any(branch not in {"metric", "structure"} for branch in frozen_branches):
            raise EvaluationError(f"Pista frozen manifest contains unsupported branches: {frozen_branches}")
        return cls(
            spec=spec,
            rows=rows,
            context=_evaluation_context(spec, split=split, branches=frozen_branches),
            vocabulary=PistaGraphVocabulary.from_json(vocabulary_path, vocabulary_id=vocabulary_id),
            model=_load_pista_model(spec, checkpoint).to(target_device),
            target_device=target_device,
        )

    def generate_arm(
        self,
        *,
        arm: str,
        output_jsonl: str | Path,
        sampling_steps: int = 50,
        trajectory_batch: int | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
        max_open_graph_shards: int | None = None,
        execution_profile: str | Path | None = None,
        progress_every_groups: int | None = None,
        progress_every_trajectory_batches: int | None = None,
        guidance_alpha: float = 1.0,
        terminal_observer: PistaTerminalBatchObserver | None = None,
    ) -> dict[str, Any]:
        """Generate one immutable raw arm shard from this loaded session."""

        if arm not in {"reference", "true", "shuffled", "null"}:
            raise EvaluationError(
                "Pista arm must be reference, true, shuffled, or null"
            )
        if terminal_observer is not None and not callable(
            getattr(terminal_observer, "observe_terminal_batch", None)
        ):
            raise EvaluationError(
                "terminal_observer must implement observe_terminal_batch"
            )
        alpha = _guidance_alpha(guidance_alpha)
        steps = require_int(sampling_steps, "sampling_steps", minimum=1)
        if progress_every_groups is not None:
            progress_every_groups = require_int(
                progress_every_groups,
                "progress_every_groups",
                minimum=1,
            )
        if progress_every_trajectory_batches is not None:
            progress_every_trajectory_batches = require_int(
                progress_every_trajectory_batches,
                "progress_every_trajectory_batches",
                minimum=1,
            )
        profile = resolve_inference_execution_profile(
            profile_path=execution_profile,
            trajectory_batch=trajectory_batch,
        )
        batch_size = profile.trajectory_batch
        pista_options = profile.options_for("pista")
        if max_open_graph_shards is not None or "max_open_graph_shards" in pista_options:
            raise EvaluationError(
                "rooted Pista inference does not materialize endpoint graph shards; "
                "remove max_open_graph_shards from the command or execution profile"
            )
        configured_group_batch = pista_options.get("max_groups_per_generation_batch")
        if configured_group_batch is None:
            # Training's physical batch is not an inference memory contract.  A
            # conservative four-group default prevents a low-candidate cohort
            # from accidentally becoming a many-thousand-trajectory dense batch.
            group_batch = min(self.context.max_groups_per_batch, 4)
        else:
            group_batch = require_int(
                configured_group_batch,
                "execution profile pista.max_groups_per_generation_batch",
                minimum=1,
            )
        configured_candidate_batch = pista_options.get(
            "max_candidates_per_generation_batch",
            self.context.max_candidates_per_batch,
        )
        candidate_batch = require_int(
            configured_candidate_batch,
            "execution profile pista.max_candidates_per_generation_batch",
            minimum=1,
        )
        # Older v2 frozen manifests predate the exact signed-candidate count.
        # Their batching bound is therefore supplied explicitly by the
        # execution profile.  Without that declaration, fall back to one
        # complete candidate-memory budget per group, which is conservative
        # but never underestimates the model-independent chunking cost.
        fallback_signed_candidate_upper_bound = require_int(
            pista_options.get("frozen_manifest_candidate_upper_bound", candidate_batch),
            "execution profile pista.frozen_manifest_candidate_upper_bound",
            minimum=1,
        )
        if fallback_signed_candidate_upper_bound > candidate_batch:
            raise EvaluationError(
                "frozen_manifest_candidate_upper_bound cannot exceed "
                "max_candidates_per_generation_batch"
            )
        per_sample_transition_kernel = _resolve_pista_transition_kernel(pista_options)
        sampling_rng_mode = (
            "paired_stateless_counter_v1"
            if per_sample_transition_kernel == "stateless_counter_v1"
            else "paired_per_trajectory_generators_v1"
        )
        context = (
            self.context
            if candidate_batch == self.context.max_candidates_per_batch
            else replace(self.context, max_candidates_per_batch=candidate_batch)
        )
        template_cache_capacity = require_int(
            pista_options.get("root_template_cache_capacity", 4_096),
            "execution profile pista.root_template_cache_capacity",
            minimum=1,
        )
        start, stop = _selected_range(
            len(self.rows),
            shard_index=shard_index,
            shard_count=shard_count,
        )
        selected = self.rows[start:stop]
        if not selected:
            raise EvaluationError("selected Pista arm shard is empty")
        rooted_builder = RootedProductGraphBuilder(
            vocabulary=self.vocabulary,
            graphdit=self.spec.graphdit,
            template_cache_capacity=template_cache_capacity,
        )
        model_for_sampling = _sampling_model_for_arm(self.model, arm=arm)
        model_for_sampling.eval()
        pairs = [
            (
                row,
                _rebuild_condition(
                    context,
                    row,
                    fallback_signed_candidate_upper_bound=fallback_signed_candidate_upper_bound,
                ),
            )
            for row in selected
        ]
        output = Path(output_jsonl).expanduser()
        partial = _new_partial_path(output)
        written = 0
        started_at = time.monotonic()
        next_progress = progress_every_groups
        trajectory_batches_completed = 0
        next_trajectory_progress = progress_every_trajectory_batches
        try:
            import torch

            with JsonlStreamWriter(partial) as writer:
                for chunk in _condition_chunks(context, pairs, max_groups=group_batch):
                    chunk_rows = [item[0] for item in chunk]
                    chunk_conditions = [item[1] for item in chunk]
                    draws = [row.get("trajectory_seeds") for row in chunk_rows]
                    if not all(isinstance(item, list) and item for item in draws):
                        raise EvaluationError("Pista frozen manifest has an invalid trajectory seed schedule")
                    draw_count = len(draws[0])
                    if any(len(item) != draw_count for item in draws if isinstance(item, list)):
                        raise EvaluationError("Pista frozen rows disagree on draw count within a batch")
                    generated: list[list[str | None]] = [[] for _ in chunk]
                    # The guided student validates its token/mask/group-index
                    # contract even when the residual scale is zero.  Null is
                    # therefore token-conditioned with the same frozen True
                    # tokens, but carries an exact scale of 0.0; Reference is
                    # the only arm that omits guidance inputs entirely.
                    guided = arm in {"null", "true", "shuffled"}
                    effective_guidance_alpha = (
                        alpha
                        if arm in {"true", "shuffled"}
                        else (0.0 if arm == "null" else None)
                    )
                    guidance = _guidance_token_batch(
                        context,
                        chunk_conditions,
                        arm=arm,
                        device=self.target_device,
                    )
                    for trajectory_start in range(0, draw_count, batch_size):
                        count = min(batch_size, draw_count - trajectory_start)
                        rooted = _rooted_product_batch(
                            rooted_builder,
                            chunk_rows,
                            trajectory_start=trajectory_start,
                            count=count,
                            device=self.target_device,
                        )
                        # Every replicated row receives the exact frozen trajectory
                        # seed assigned to its (group, draw) slot.  Legacy kernels
                        # instantiate one generator per row; the stateless kernel
                        # consumes the same frozen row seeds in a counter keyed by
                        # diffusion step and categorical sampling slot.
                        frozen_trajectory_seeds = [
                            int(group_draws[trajectory_start + trajectory_index])
                            for group_draws in draws
                            for trajectory_index in range(count)
                        ]
                        if per_sample_transition_kernel == "stateless_counter_v1":
                            sampler_rng: Any = torch.tensor(
                                frozen_trajectory_seeds,
                                dtype=torch.int64,
                                device=self.target_device,
                            )
                        else:
                            sampler_rng = [
                                torch.Generator(device=self.target_device).manual_seed(seed)
                                for seed in frozen_trajectory_seeds
                            ]
                        kwargs: dict[str, Any] = {}
                        if guided:
                            if guidance is None:  # Defensive: guided arms require a token batch above.
                                raise EvaluationError("Pista guided arm has no frozen token batch")
                            kwargs = {
                                "guidance_tokens_group": guidance[0],
                                "guidance_token_mask_group": guidance[1],
                                "guidance_group_index": torch.arange(
                                    len(chunk),
                                    device=self.target_device,
                                    dtype=torch.long,
                                ).repeat_interleave(count),
                                "guidance_residual_scale": effective_guidance_alpha,
                            }
                        from retro_guided_generation.models.dfm.sampling import (
                            PredefinedSamplingConfig,
                            sample_predefined_graph,
                        )

                        with torch.inference_mode(), _autocast(self.target_device):
                            sampled = sample_predefined_graph(
                                model_for_sampling,
                                p_X=rooted.p_X,
                                p_E=rooted.p_E,
                                node_mask=rooted.node_mask,
                                p_mask=rooted.p_mask,
                                cond=rooted.cond,
                                generator=sampler_rng,
                                config=PredefinedSamplingConfig(
                                    steps=steps,
                                    per_sample_transition_kernel=per_sample_transition_kernel,
                                ),
                                model_kwargs=kwargs,
                            )
                        if terminal_observer is not None:
                            terminal_observer.observe_terminal_batch(
                                chunk_rows=chunk_rows,
                                trajectory_start=trajectory_start,
                                trajectory_count=count,
                                rooted_product=rooted,
                                terminal_state_X=sampled.terminal_state_X,
                                terminal_state_E=sampled.terminal_state_E,
                            )
                        for local_index in range(len(chunk)):
                            for trajectory_index in range(count):
                                output_index = local_index * count + trajectory_index
                                decoded = _decode_pista_sampled_terminal_state(
                                    self.vocabulary,
                                    terminal_state_X=sampled.terminal_state_X[output_index],
                                    terminal_state_E=sampled.terminal_state_E[output_index],
                                    node_mask=rooted.node_mask[output_index],
                                )
                                generated[local_index].append(canonical_reaction_smiles(decoded))
                        trajectory_batches_completed += 1
                        if (
                            next_trajectory_progress is not None
                            and trajectory_batches_completed >= next_trajectory_progress
                        ):
                            print(
                                json.dumps(
                                    {
                                        "schema_version": "rgg_pista_arm_trajectory_progress.v1",
                                        "status": "running",
                                        "arm": arm,
                                        "written_groups": written,
                                        "selected_group_count": len(selected),
                                        "chunk_group_count": len(chunk_rows),
                                        "chunk_first_group_index": chunk_rows[0]["evaluation_group_index"],
                                        "trajectory_start": trajectory_start,
                                        "trajectory_count": count,
                                        "trajectory_batches_completed": trajectory_batches_completed,
                                        "elapsed_seconds": round(time.monotonic() - started_at, 3),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                            next_trajectory_progress += progress_every_trajectory_batches
                    for row, values in zip(chunk_rows, generated, strict=True):
                        score = score_generated_group(
                            values,
                            row.get("positive_target_smiles"),
                            row.get("negative_target_smiles"),
                            expected_draw_count=draw_count,
                            require_canonical_known_sets=True,
                            require_canonical_generated=True,
                        )
                        writer.write(
                            {
                                "schema_version": PISTA_ARM_SCHEMA,
                                "arm": arm,
                                "group_uid": row["group_uid"],
                                "case_id": row["case_id"],
                                "evaluation_group_index": row["evaluation_group_index"],
                                "legacy_group_idx": row["legacy_group_idx"],
                                "root_draw_seeds": row["root_draw_seeds"],
                                "root_choice_indices": row["root_choice_indices"],
                                "trajectory_seeds": row["trajectory_seeds"],
                                "rc_permutation_keys": [
                                    row["rc_candidates"][root_slot]["rc_permutation_key"]
                                    for root_slot in row["root_choice_indices"]
                                ],
                                "generated_smiles": values,
                                "categories": score["categories"],
                                "category_counts": score["category_counts"],
                                "sampling_steps": steps,
                                "per_sample_transition_kernel": per_sample_transition_kernel,
                                "sampling_rng_mode": sampling_rng_mode,
                                "sampling_decode_mode": "hard_sampled_terminal_state_v1",
                                "trajectory_batch": batch_size,
                                "execution_profile_id": profile.profile_id,
                                "guidance_residual_scale": effective_guidance_alpha,
                            }
                        )
                        written += 1
                    if next_progress is not None and written >= next_progress:
                        print(
                            json.dumps(
                                {
                                    "schema_version": "rgg_pista_arm_generation_progress.v1",
                                    "status": "running",
                                    "arm": arm,
                                    "written_groups": written,
                                    "selected_group_count": len(selected),
                                    "elapsed_seconds": round(time.monotonic() - started_at, 3),
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        next_progress += progress_every_groups
            _publish_partial(partial, output)
        except BaseException:
            # Preserve the uniquely named partial artifact for diagnosis and do
            # not replace a completed shard with a failed run.
            raise
        return {
            "schema_version": "rgg_pista_arm_generation.v1",
            "status": "complete",
            "arm": arm,
            "output": str(output.resolve()),
            "group_count": written,
            "range": {"start": start, "stop": stop, "total": len(self.rows)},
            "sampling_steps": steps,
            "trajectory_batch": batch_size,
            "max_groups_per_generation_batch": group_batch,
            "max_candidates_per_generation_batch": candidate_batch,
            "frozen_manifest_candidate_upper_bound": fallback_signed_candidate_upper_bound,
            "guidance_context_mode": context.guidance_context_mode,
            "guidance_residual_scale": (
                alpha
                if arm in {"true", "shuffled"}
                else (0.0 if arm == "null" else None)
            ),
            "per_sample_transition_kernel": per_sample_transition_kernel,
            "sampling_rng_mode": sampling_rng_mode,
            "sampling_decode_mode": "hard_sampled_terminal_state_v1",
            "root_input_mode": "frozen_rc_permutation_product_graph.v1",
            "root_template_cache": rooted_builder.template_cache_statistics,
            "execution_profile": profile.to_dict(),
            "progress_every_groups": progress_every_groups,
            "progress_every_trajectory_batches": progress_every_trajectory_batches,
            "integrity_policy": INTEGRITY_POLICY,
        }


def generate_pista_arm(
    *,
    training_config: str | Path,
    checkpoint: str | Path,
    frozen_manifest: str | Path,
    split: str = "test",
    expected_draw_count: int = 100,
    root_draw_seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
    trajectory_seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
    arm: str,
    output_jsonl: str | Path,
    sampling_steps: int = 50,
    trajectory_batch: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cuda",
    max_open_graph_shards: int | None = None,
    execution_profile: str | Path | None = None,
    vocabulary_path: str | Path = DEFAULT_PISTA_VOCAB,
    vocabulary_id: str = DEFAULT_PISTA_VOCAB_ID,
    progress_every_groups: int | None = None,
    progress_every_trajectory_batches: int | None = None,
    guidance_alpha: float = 1.0,
) -> dict[str, Any]:
    """Generate one immutable raw arm shard suitable for generic scoring."""

    session = PistaInferenceSession.open(
        training_config=training_config,
        checkpoint=checkpoint,
        frozen_manifest=frozen_manifest,
        split=split,
        expected_draw_count=expected_draw_count,
        root_draw_seed_base=root_draw_seed_base,
        trajectory_seed_base=trajectory_seed_base,
        device=device,
        vocabulary_path=vocabulary_path,
        vocabulary_id=vocabulary_id,
    )
    return session.generate_arm(
        arm=arm,
        output_jsonl=output_jsonl,
        sampling_steps=sampling_steps,
        trajectory_batch=trajectory_batch,
        shard_index=shard_index,
        shard_count=shard_count,
        max_open_graph_shards=max_open_graph_shards,
        execution_profile=execution_profile,
        progress_every_groups=progress_every_groups,
        progress_every_trajectory_batches=progress_every_trajectory_batches,
        guidance_alpha=guidance_alpha,
    )


def generate_pista_arms(
    *,
    training_config: str | Path,
    checkpoint: str | Path,
    frozen_manifest: str | Path,
    split: str = "test",
    expected_draw_count: int = 100,
    root_draw_seed_base: int = DEFAULT_ROOT_DRAW_SEED_BASE,
    trajectory_seed_base: int = DEFAULT_TRAJECTORY_SEED_BASE,
    arm_outputs: Mapping[str, str | Path],
    sampling_steps: int = 50,
    trajectory_batch: int | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cuda",
    max_open_graph_shards: int | None = None,
    execution_profile: str | Path | None = None,
    vocabulary_path: str | Path = DEFAULT_PISTA_VOCAB,
    vocabulary_id: str = DEFAULT_PISTA_VOCAB_ID,
    progress_every_groups: int | None = None,
    progress_every_trajectory_batches: int | None = None,
    guidance_alpha: float = 1.0,
) -> dict[str, dict[str, Any]]:
    """Generate several sequential arms while reusing one loaded CUDA session.

    This is useful for bounded execution-profile benchmarks or one-GPU runs.
    For a formal multi-GPU experiment, launch independent disjoint shards for
    each arm so all allocated GPUs remain busy.
    """

    if not isinstance(arm_outputs, Mapping) or not arm_outputs:
        raise EvaluationError("Pista arm session needs at least one arm output")
    normalized: list[tuple[str, str | Path]] = []
    output_paths: set[Path] = set()
    for arm, output in arm_outputs.items():
        if arm not in {"reference", "true", "shuffled", "null"}:
            raise EvaluationError(
                "Pista arm must be reference, true, shuffled, or null"
            )
        if not isinstance(output, (str, Path)):
            raise EvaluationError(f"Pista arm output for {arm!r} must be a path")
        output_path = Path(output).expanduser().absolute()
        if output_path in output_paths:
            raise EvaluationError("Pista arm session outputs must be distinct")
        output_paths.add(output_path)
        normalized.append((arm, output))
    session = PistaInferenceSession.open(
        training_config=training_config,
        checkpoint=checkpoint,
        frozen_manifest=frozen_manifest,
        split=split,
        expected_draw_count=expected_draw_count,
        root_draw_seed_base=root_draw_seed_base,
        trajectory_seed_base=trajectory_seed_base,
        device=device,
        vocabulary_path=vocabulary_path,
        vocabulary_id=vocabulary_id,
    )
    results: dict[str, dict[str, Any]] = {}
    for arm, output in normalized:
        results[arm] = session.generate_arm(
            arm=arm,
            output_jsonl=output,
            sampling_steps=sampling_steps,
            trajectory_batch=trajectory_batch,
            shard_index=shard_index,
            shard_count=shard_count,
            max_open_graph_shards=max_open_graph_shards,
            execution_profile=execution_profile,
            progress_every_groups=progress_every_groups,
            progress_every_trajectory_batches=progress_every_trajectory_batches,
            guidance_alpha=guidance_alpha,
        )
    return results


def merge_pista_arm_shards(
    *,
    arm: str,
    input_jsonl: Sequence[str | Path],
    output_jsonl: str | Path,
    expected_group_count: int | None = None,
    frozen_manifest: str | Path | None = None,
) -> Path:
    """Merge non-overlapping shard files into canonical frozen-group order."""

    if arm not in {"reference", "true", "shuffled", "null"}:
        raise EvaluationError("Pista arm must be reference, true, shuffled, or null")
    if not input_jsonl:
        raise EvaluationError("Pista shard merge needs at least one input")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for source in input_jsonl:
        for row in read_rows(source):
            if row.get("arm") != arm:
                raise EvaluationError("Pista shard arm disagrees with merge arm")
            index = require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
            if index in seen:
                raise EvaluationError(f"Pista shard merge has duplicate group index {index}")
            seen.add(index)
            rows.append(row)
    rows.sort(key=lambda row: int(row["evaluation_group_index"]))
    if frozen_manifest is not None:
        frozen_rows = read_rows(frozen_manifest)
        expected_by_index: dict[int, str] = {}
        for frozen in frozen_rows:
            index = require_int(frozen.get("evaluation_group_index"), "evaluation_group_index", minimum=0)
            group_uid = _text(frozen.get("group_uid"), "frozen group_uid")
            if index in expected_by_index:
                raise EvaluationError("frozen manifest has duplicate evaluation_group_index values")
            expected_by_index[index] = group_uid
        actual_by_index = {
            require_int(row.get("evaluation_group_index"), "evaluation_group_index", minimum=0): _text(
                row.get("group_uid"), "arm group_uid"
            )
            for row in rows
        }
        if len(actual_by_index) != len(rows):  # Defensive duplicate check above remains authoritative.
            raise EvaluationError("Pista shard merge has duplicate group index values")
        if actual_by_index != expected_by_index:
            raise EvaluationError("Pista shard merge does not exactly cover the frozen manifest")
    if expected_group_count is not None:
        expected = require_int(expected_group_count, "expected_group_count", minimum=1)
        if len(rows) != expected:
            raise EvaluationError("Pista shard merge group count differs from expected_group_count")
        if frozen_manifest is None and [int(row["evaluation_group_index"]) for row in rows] != list(range(expected)):
            raise EvaluationError("Pista shard merge does not cover the complete expected group range")
    return write_new_jsonl(output_jsonl, rows)


def write_pista_alpha_zero_parity_audit(
    *,
    reference_jsonl: str | Path,
    null_jsonl: str | Path,
    true_jsonl: str | Path,
    shuffled_jsonl: str | Path,
    output_json: str | Path,
) -> Path:
    """Require exact per-trajectory Reference parity for every alpha-zero arm."""

    sources = {
        "reference": Path(reference_jsonl).expanduser().resolve(strict=True),
        "null": Path(null_jsonl).expanduser().resolve(strict=True),
        "true": Path(true_jsonl).expanduser().resolve(strict=True),
        "shuffled": Path(shuffled_jsonl).expanduser().resolve(strict=True),
    }
    rows = {arm: read_rows(path) for arm, path in sources.items()}
    group_counts = {arm: len(values) for arm, values in rows.items()}
    if not group_counts["reference"] or len(set(group_counts.values())) != 1:
        raise EvaluationError(
            f"alpha-zero parity arms have unequal or empty group counts: {group_counts}"
        )
    identity_fields = (
        "group_uid",
        "case_id",
        "evaluation_group_index",
        "legacy_group_idx",
        "root_draw_seeds",
        "root_choice_indices",
        "trajectory_seeds",
        "rc_permutation_keys",
        "sampling_steps",
        "per_sample_transition_kernel",
        "sampling_rng_mode",
        "sampling_decode_mode",
        "trajectory_batch",
        "execution_profile_id",
    )
    generated_fields = (
        "generated_smiles",
        "categories",
        "category_counts",
    )
    expected_scales: dict[str, float | None] = {
        "reference": None,
        "null": 0.0,
        "true": 0.0,
        "shuffled": 0.0,
    }
    trajectory_count = 0
    for row_index, reference in enumerate(rows["reference"]):
        if reference.get("arm") != "reference":
            raise EvaluationError("reference parity file contains a non-reference row")
        generated = reference.get("generated_smiles")
        if not isinstance(generated, list) or not generated:
            raise EvaluationError("reference parity row lacks generated trajectories")
        trajectory_count += len(generated)
        for arm in ("null", "true", "shuffled"):
            candidate = rows[arm][row_index]
            if candidate.get("arm") != arm:
                raise EvaluationError(f"{arm} parity file contains a different arm")
            for field in identity_fields:
                if candidate.get(field) != reference.get(field):
                    raise EvaluationError(
                        f"alpha-zero {arm} identity differs at row {row_index}, field {field}"
                    )
            for field in generated_fields:
                if candidate.get(field) != reference.get(field):
                    raise EvaluationError(
                        f"alpha-zero {arm} output differs at row {row_index}, field {field}"
                    )
        for arm, expected in expected_scales.items():
            observed = rows[arm][row_index].get("guidance_residual_scale")
            if observed != expected:
                raise EvaluationError(
                    f"alpha-zero {arm} row {row_index} reports scale {observed!r}, "
                    f"expected {expected!r}"
                )

    return write_new_json(
        output_json,
        {
            "schema_version": "rgg_pista_alpha_zero_parity_audit.v1",
            "status": "passed",
            "exact_trajectory_parity": True,
            "arms": {arm: str(path) for arm, path in sources.items()},
            "group_count": group_counts["reference"],
            "trajectory_count_per_arm": trajectory_count,
            "identity_fields_checked": list(identity_fields),
            "generated_fields_checked": list(generated_fields),
            "expected_guidance_residual_scales": expected_scales,
            "test_data_opened": False,
            "content_hashes_computed": False,
        },
    )


__all__ = [
    "DEFAULT_PISTA_VOCAB",
    "DEFAULT_PISTA_VOCAB_ID",
    "deterministic_pista_sample_indices",
    "normalize_pista_selected_group_indices",
    "PISTA_ARM_SCHEMA",
    "PISTA_FROZEN_SCHEMA",
    "build_pista_frozen_rows",
    "generate_pista_arm",
    "merge_pista_arm_shards",
    "partition_pista_frozen_rows",
    "root_choice_indices_for_group",
    "root_draw_seeds_for_group",
    "select_pista_smoke_rows",
    "trajectory_seeds_for_group",
    "validate_pista_frozen_rows",
    "write_pista_frozen_manifest",
    "write_pista_frozen_partition",
    "write_pista_alpha_zero_parity_audit",
    "write_pista_smoke_manifest",
]
