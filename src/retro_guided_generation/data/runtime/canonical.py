"""Runtime resolution of canonical, opposite, and shuffled guidance.

This module is deliberately independent of graph tensorization and text
encoding.  It gives every downstream trainer the same semantic view of a
canonical corpus and is the only place that resolves per-traversal wording or
different-axis shuffled controls.  Graph and text caches may consume its
stable IDs later without changing these relation rules.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from retro_guided_generation.guidance.errors import QualityError, SchemaError
from retro_guided_generation.guidance.io import iter_models, read_json
from retro_guided_generation.guidance.relations import RelationResolver
from retro_guided_generation.guidance.sampling import (
    RuntimeBranchCycleSampler,
    RuntimeTrueGuidanceDraw,
    RuntimeTrueGuidanceSampler,
)
from retro_guided_generation.guidance.schema import (
    SemanticGroup,
    ShuffleDonorPool,
    TextPool,
    TextVariant,
)
from retro_guided_generation.guidance.shuffle import (
    SPLIT_ORDINALS,
    RuntimeShuffleDraw,
    RuntimeShuffleSampler,
)


CANONICAL_RUNTIME_SCHEMA = "retro_guided_generation.canonical_runtime.v1"
_GUIDANCE_DATASET_SCHEMA = "guidance_v3.dataset_manifest.v2"


class GuidanceRuntimeError(RuntimeError):
    """Raised when a canonical corpus cannot support runtime resolution."""


@dataclass(frozen=True, slots=True)
class RuntimeGuidanceCondition:
    """One candidate group with resolved text controls for a traversal."""

    split: str
    group_ordinal: int
    group: SemanticGroup
    true_text: TextVariant
    true_relation: str
    opposite_text: TextVariant
    shuffled: RuntimeShuffleDraw | None

    @property
    def shuffled_text_id(self) -> str | None:
        return None if self.shuffled is None else self.shuffled.donor_text_id

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": CANONICAL_RUNTIME_SCHEMA,
            "split": self.split,
            "group_ordinal": self.group_ordinal,
            "semantic_group_uid": self.group.semantic_group_uid,
            "branch": self.group.branch,
            "axis_id": self.group.axis_id,
            "polarity": self.group.polarity,
            "true_text_id": self.true_text.text_id,
            "true_relation": self.true_relation,
            "opposite_text_id": self.opposite_text.text_id,
            "shuffled": None if self.shuffled is None else self.shuffled.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _SplitRuntime:
    groups: tuple[SemanticGroup, ...]
    pools: tuple[TextPool, ...]
    resolver: RelationResolver
    shuffle_sampler: RuntimeShuffleSampler


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if value >= 2**64:
        raise ValueError(f"{name} must be below 2**64")
    return value


def _numeric_seed(
    *,
    base_seed: int,
    split: str,
    group_ordinal: int,
    epoch: int,
    draw_index: int,
) -> int:
    try:
        split_ordinal = SPLIT_ORDINALS[split]
    except KeyError as error:
        raise GuidanceRuntimeError(f"unsupported split {split!r}") from error
    group_value = _require_nonnegative_int(group_ordinal, "group_ordinal")
    epoch_value = _require_nonnegative_int(epoch, "epoch")
    draw_value = _require_nonnegative_int(draw_index, "draw_index")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise ValueError("base_seed must be an integer")
    nonnegative_seed = 2 * base_seed if base_seed >= 0 else -2 * base_seed - 1
    return (
        (nonnegative_seed << 194)
        | (split_ordinal << 192)
        | (group_value << 128)
        | (epoch_value << 64)
        | draw_value
    )


def _artifact_path(root: Path, artifacts: Mapping[str, object], key: str) -> Path:
    raw = artifacts.get(key)
    if not isinstance(raw, str) or not raw:
        raise GuidanceRuntimeError(f"canonical corpus is missing artifact {key!r}")
    try:
        path = (root / raw).resolve(strict=True)
        path.relative_to(root)
    except (OSError, ValueError) as error:
        raise GuidanceRuntimeError(
            f"canonical corpus artifact {key!r} is unavailable under {root}"
        ) from error
    return path


class GuidanceRuntimeIndex:
    """Load and resolve the canonical guidance protocol for all corpus splits."""

    def __init__(
        self,
        *,
        root: Path,
        manifest: Mapping[str, object],
        split_runtime: Mapping[str, _SplitRuntime],
        training_seed: int,
        evaluation_seed: int,
        branch_weights: Mapping[str, float],
        sampling_mode: str,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.training_seed = training_seed
        self.evaluation_seed = evaluation_seed
        if sampling_mode not in {"hierarchical", "branch_cycle"}:
            raise ValueError("sampling_mode must be 'hierarchical' or 'branch_cycle'")
        self.sampling_mode = sampling_mode
        self.branch_weights = {
            branch: float(branch_weights.get(branch, 0.0))
            for branch in ("metric", "structure")
        }
        self._split_runtime = dict(split_runtime)
        self._group_by_uid: dict[str, tuple[str, int, SemanticGroup]] = {}
        all_groups: list[SemanticGroup] = []
        all_pools_by_id: dict[str, TextPool] = {}
        for split, runtime in self._split_runtime.items():
            for ordinal, group in enumerate(runtime.groups):
                if group.split != split:
                    raise GuidanceRuntimeError(
                        f"semantic group {group.semantic_group_uid!r} has split "
                        f"{group.split!r}, expected {split!r}"
                    )
                if group.semantic_group_uid in self._group_by_uid:
                    raise GuidanceRuntimeError(
                        f"duplicate semantic group {group.semantic_group_uid!r}"
                    )
                self._group_by_uid[group.semantic_group_uid] = (split, ordinal, group)
                all_groups.append(group)
            for pool in runtime.pools:
                previous = all_pools_by_id.setdefault(pool.text_pool_id, pool)
                if previous != pool:
                    raise GuidanceRuntimeError(
                        f"text pool {pool.text_pool_id!r} differs across corpus splits"
                    )
        sampler_class = (
            RuntimeTrueGuidanceSampler
            if sampling_mode == "hierarchical"
            else RuntimeBranchCycleSampler
        )
        self._true_sampler = sampler_class(
            groups=tuple(all_groups),
            text_pools=tuple(all_pools_by_id.values()),
            seed=training_seed,
            evaluation_seed=evaluation_seed,
            branch_weights=self.branch_weights,
        )

    @classmethod
    def from_corpus(
        cls,
        root: str | Path,
        *,
        training_seed: int,
        evaluation_seed: int,
        branch_weights: Mapping[str, float],
        sampling_mode: str = "hierarchical",
    ) -> "GuidanceRuntimeIndex":
        """Load an already validated canonical corpus without cache rebuilding."""

        corpus_root = Path(root).expanduser().resolve(strict=True)
        manifest = read_json(corpus_root / "manifest.json")
        if manifest.get("schema_version") != _GUIDANCE_DATASET_SCHEMA:
            raise GuidanceRuntimeError("unsupported canonical guidance manifest")
        artifacts = manifest.get("artifacts")
        raw_splits = manifest.get("splits")
        if not isinstance(artifacts, Mapping):
            raise GuidanceRuntimeError("canonical guidance manifest artifacts are invalid")
        if not isinstance(raw_splits, list) or not raw_splits:
            raise GuidanceRuntimeError("canonical guidance manifest has no split list")
        splits = tuple(raw_splits)
        if any(not isinstance(split, str) or split not in SPLIT_ORDINALS for split in splits):
            raise GuidanceRuntimeError("canonical guidance manifest has unsupported splits")
        if len(splits) != len(set(splits)):
            raise GuidanceRuntimeError("canonical guidance manifest has duplicate splits")

        metric_pools = tuple(
            iter_models(_artifact_path(corpus_root, artifacts, "metric_text_pools"), TextPool)
        )
        split_runtime: dict[str, _SplitRuntime] = {}
        for split in splits:
            groups = tuple(
                iter_models(
                    _artifact_path(corpus_root, artifacts, f"semantic_groups.{split}"),
                    SemanticGroup,
                )
            )
            if not groups:
                raise GuidanceRuntimeError(
                    f"canonical guidance corpus has no semantic groups for {split!r}"
                )
            structure_pools = tuple(
                iter_models(
                    _artifact_path(
                        corpus_root,
                        artifacts,
                        f"structure_text_pools.{split}",
                    ),
                    TextPool,
                )
            )
            pools = (*metric_pools, *structure_pools)
            if not pools:
                raise GuidanceRuntimeError("canonical guidance corpus has no text pools")
            donors = tuple(
                iter_models(
                    _artifact_path(
                        corpus_root,
                        artifacts,
                        f"shuffle_donor_pools.{split}",
                    ),
                    ShuffleDonorPool,
                )
            )
            if not donors:
                raise GuidanceRuntimeError(
                    f"canonical guidance corpus has no shuffle donors for {split!r}"
                )
            try:
                resolver = RelationResolver(groups=groups, text_pools=pools)
                shuffle_sampler = RuntimeShuffleSampler(
                    donor_pools=donors,
                    seed=training_seed,
                    evaluation_seed=evaluation_seed,
                    metric_source_weights=cls._metric_source_weights(manifest),
                )
            except (QualityError, SchemaError, ValueError) as error:
                raise GuidanceRuntimeError(
                    f"canonical runtime relation setup failed for split {split!r}: {error}"
                ) from error
            split_runtime[split] = _SplitRuntime(
                groups=groups,
                pools=pools,
                resolver=resolver,
                shuffle_sampler=shuffle_sampler,
            )
        return cls(
            root=corpus_root,
            manifest=manifest,
            split_runtime=split_runtime,
            training_seed=training_seed,
            evaluation_seed=evaluation_seed,
            branch_weights=branch_weights,
            sampling_mode=sampling_mode,
        )

    @staticmethod
    def _metric_source_weights(manifest: Mapping[str, object]) -> Mapping[str, float]:
        policy = manifest.get("shuffle_policy")
        if not isinstance(policy, Mapping):
            raise GuidanceRuntimeError("canonical guidance manifest has no shuffle policy")
        weights = policy.get("metric_source_weights")
        if not isinstance(weights, Mapping):
            raise GuidanceRuntimeError(
                "canonical guidance shuffle policy has no metric source weights"
            )
        if set(weights) - {"metric", "structure"}:
            raise GuidanceRuntimeError(
                "canonical guidance shuffle policy has unknown metric donor source"
            )
        normalized = {name: float(weights.get(name, 0.0)) for name in ("metric", "structure")}
        if any(value < 0.0 for value in normalized.values()) or sum(normalized.values()) <= 0.0:
            raise GuidanceRuntimeError(
                "canonical guidance shuffle source weights must be non-negative and non-empty"
            )
        return normalized

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self._split_runtime)

    def group_count(self, split: str) -> int:
        return len(self._runtime(split).groups)

    def epoch_group_count(self, split: str) -> int:
        """Return the sampler-defined deterministic train exposure count."""

        self._runtime(split)
        count = self._true_sampler.epoch_group_count(split)
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise GuidanceRuntimeError("runtime sampler exposed an invalid epoch group count")
        return count

    def group(self, split: str, group_ordinal: int) -> SemanticGroup:
        ordinal = _require_nonnegative_int(group_ordinal, "group_ordinal")
        groups = self._runtime(split).groups
        try:
            return groups[ordinal]
        except IndexError as error:
            raise IndexError(
                f"group ordinal {ordinal} is outside split {split!r}"
            ) from error

    def group_location(self, group_uid: str) -> tuple[str, int]:
        try:
            split, ordinal, _group = self._group_by_uid[group_uid]
        except KeyError as error:
            raise KeyError(f"unknown semantic group {group_uid!r}") from error
        return split, ordinal

    def sample_true(
        self,
        *,
        split: str,
        epoch: int = 0,
        draw_index: int = 0,
        evaluation: bool = False,
    ) -> RuntimeTrueGuidanceDraw:
        """Select a branch-balanced true control without materialized draws."""

        self._runtime(split)
        return self._true_sampler.sample(
            split=split,
            epoch=epoch,
            draw_index=draw_index,
            evaluation=evaluation,
        )

    def branch_cycle_state(
        self,
        *,
        split: str,
        epoch: int = 0,
        draw_index: int = 0,
    ) -> dict[str, int | str]:
        """Return formal cycle coordinates when branch-cycle sampling is active."""

        if self.sampling_mode != "branch_cycle":
            raise RuntimeError("branch_cycle_state requires sampling_mode='branch_cycle'")
        sampler = self._true_sampler
        if not isinstance(sampler, RuntimeBranchCycleSampler):  # pragma: no cover
            raise AssertionError("branch-cycle runtime has the wrong sampler type")
        return sampler.cycle_state(split=split, epoch=epoch, draw_index=draw_index)

    def condition_for_true_draw(
        self,
        draw: RuntimeTrueGuidanceDraw,
        *,
        epoch: int = 0,
        draw_index: int = 0,
        include_shuffled: bool = True,
        evaluation: bool = False,
    ) -> RuntimeGuidanceCondition:
        """Resolve a selected true draw and its optional counterfactual."""

        split, ordinal = self.group_location(draw.group.semantic_group_uid)
        return self.condition(
            split=split,
            group_ordinal=ordinal,
            epoch=epoch,
            draw_index=draw_index,
            include_shuffled=include_shuffled,
            evaluation=evaluation,
            true_text_id=draw.text_id,
        )

    def condition(
        self,
        *,
        split: str,
        group_ordinal: int,
        epoch: int = 0,
        draw_index: int = 0,
        include_shuffled: bool = True,
        evaluation: bool = False,
        true_text_id: str | None = None,
    ) -> RuntimeGuidanceCondition:
        """Resolve all text relations for one fixed semantic group."""

        runtime = self._runtime(split)
        group = self.group(split, group_ordinal)
        true_text = self._resolve_true_text(
            runtime=runtime,
            group=group,
            split=split,
            group_ordinal=group_ordinal,
            epoch=epoch,
            draw_index=draw_index,
            evaluation=evaluation,
            requested_text_id=true_text_id,
        )
        true_relation = runtime.resolver.relation(group.semantic_group_uid, true_text.text_id)
        if true_relation not in {"canonical_true", "synonym_true"}:
            raise GuidanceRuntimeError(
                f"resolved true text has invalid relation {true_relation!r}"
            )
        opposite_text = runtime.resolver.pool(group.opposite_text_pool_id).canonical
        if (
            runtime.resolver.relation(group.semantic_group_uid, opposite_text.text_id)
            != "same_axis_antonym"
        ):
            raise GuidanceRuntimeError("canonical opposite text is not same-axis antonym")
        shuffled = None
        if include_shuffled:
            try:
                shuffled = runtime.shuffle_sampler.sample(
                    group,
                    group_ordinal=group_ordinal,
                    epoch=epoch,
                    draw_index=draw_index,
                    evaluation=evaluation,
                )
                runtime.resolver.assert_different_axis(
                    group.semantic_group_uid,
                    shuffled.donor_text_id,
                )
            except (QualityError, SchemaError, ValueError) as error:
                raise GuidanceRuntimeError(
                    f"could not resolve different-axis shuffle for "
                    f"{group.semantic_group_uid!r}: {error}"
                ) from error
        return RuntimeGuidanceCondition(
            split=split,
            group_ordinal=group_ordinal,
            group=group,
            true_text=true_text,
            true_relation=true_relation,
            opposite_text=opposite_text,
            shuffled=shuffled,
        )

    def _runtime(self, split: str) -> _SplitRuntime:
        try:
            return self._split_runtime[split]
        except KeyError as error:
            raise KeyError(f"canonical runtime has no split {split!r}") from error

    def _resolve_true_text(
        self,
        *,
        runtime: _SplitRuntime,
        group: SemanticGroup,
        split: str,
        group_ordinal: int,
        epoch: int,
        draw_index: int,
        evaluation: bool,
        requested_text_id: str | None,
    ) -> TextVariant:
        pool = runtime.resolver.pool(group.text_pool_id)
        if requested_text_id is not None:
            ownership = runtime.resolver.ownership(requested_text_id)
            if ownership.pool.text_pool_id != pool.text_pool_id:
                raise GuidanceRuntimeError(
                    "requested true text does not belong to the group's true text pool"
                )
            if evaluation and ownership.text.relation != "canonical":
                raise GuidanceRuntimeError(
                    "evaluation true text must use the canonical wording"
                )
            if not evaluation and ownership.text.usage == "heldout":
                raise GuidanceRuntimeError("training true text cannot be heldout-only")
            return ownership.text
        if evaluation:
            return pool.canonical
        generator = random.Random(
            _numeric_seed(
                base_seed=self.training_seed,
                split=split,
                group_ordinal=group_ordinal,
                epoch=epoch,
                draw_index=draw_index,
            )
        )
        return generator.choice(pool.training_variants)


__all__ = [
    "CANONICAL_RUNTIME_SCHEMA",
    "GuidanceRuntimeError",
    "GuidanceRuntimeIndex",
    "RuntimeGuidanceCondition",
]
