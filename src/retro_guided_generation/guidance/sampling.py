"""Runtime sampling of true guidance without materializing training draws."""

from __future__ import annotations

import math
import random
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retro_guided_generation.guidance.config import SamplingConfig
from retro_guided_generation.guidance.errors import QualityError, SchemaError
from retro_guided_generation.guidance.schema import SemanticGroup, TextPool
from retro_guided_generation.guidance.shuffle import SPLIT_ORDINALS
from retro_guided_generation.guidance.text_pools import TextPoolRegistry


@dataclass(frozen=True, slots=True)
class RuntimeTrueGuidanceDraw:
    """One true semantic group and wording selected for a training traversal."""

    group: SemanticGroup
    text_id: str
    text_pool_id: str
    control_id: str
    wording_usage: str
    relation: str = "true_guidance"

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic_group_uid": self.group.semantic_group_uid,
            "semantic_family_uid": self.group.semantic_family_uid,
            "case_id": self.group.case_id,
            "candidate_pool_id": self.group.candidate_pool_id,
            "split": self.group.split,
            "branch": self.group.branch,
            "axis_id": self.group.axis_id,
            "polarity": self.group.polarity,
            "text_id": self.text_id,
            "text_pool_id": self.text_pool_id,
            "control_id": self.control_id,
            "wording_usage": self.wording_usage,
            "relation": self.relation,
        }


@dataclass(frozen=True, slots=True)
class _SemanticFamily:
    """The two signed groups belonging to one case-axis semantic family."""

    split: str
    branch: str
    axis_id: str
    semantic_family_uid: str
    groups_by_polarity: Mapping[int, SemanticGroup]


class RuntimeTrueGuidanceSampler:
    """Hierarchically sample true guidance with wording-independent mass.

    The sequence is branch, axis, semantic family, polarity, then wording.
    Each choice before wording is uniform after the preceding choice, so adding
    synonyms cannot change branch, axis, family, or polarity probability mass.
    The caller supplies groups in canonical corpus order; that order is used
    as the explicit family ordering rather than any content-derived hash.
    """

    def __init__(
        self,
        *,
        groups: Sequence[SemanticGroup],
        text_pools: Sequence[TextPool],
        seed: int,
        evaluation_seed: int,
        branch_weights: Mapping[str, float],
        training_wording: str = "all_train",
        evaluation_wording: str = "canonical",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SchemaError("runtime true-guidance seed must be an integer")
        if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
            raise SchemaError(
                "runtime true-guidance evaluation seed must be an integer"
            )
        if training_wording != "all_train":
            raise SchemaError(
                "runtime true-guidance training_wording must be 'all_train'"
            )
        if evaluation_wording != "canonical":
            raise SchemaError(
                "runtime true-guidance evaluation_wording must be 'canonical'"
            )
        if set(branch_weights) - {"metric", "structure"}:
            raise SchemaError("runtime true-guidance weights have unknown keys")
        weights = {
            branch: float(branch_weights.get(branch, 0.0))
            for branch in ("metric", "structure")
        }
        if any(value < 0.0 for value in weights.values()):
            raise SchemaError("runtime true-guidance weights must be non-negative")
        if sum(weights.values()) <= 0.0:
            raise SchemaError("runtime true-guidance weights must sum above zero")
        if not groups:
            raise SchemaError("runtime true-guidance sampler requires semantic groups")

        self.seed = seed
        self.evaluation_seed = evaluation_seed
        self.branch_weights = weights
        self.training_wording = training_wording
        self.evaluation_wording = evaluation_wording
        self._text_registry = TextPoolRegistry(text_pools)

        groups_by_family: dict[str, dict[int, SemanticGroup]] = {}
        family_order: list[str] = []
        group_ids: set[str] = set()
        for group in groups:
            if group.semantic_group_uid in group_ids:
                raise SchemaError(
                    "runtime true-guidance index contains duplicate semantic groups"
                )
            group_ids.add(group.semantic_group_uid)
            try:
                pool = self._text_registry.pool(group.text_pool_id)
            except KeyError as error:
                raise SchemaError(
                    f"semantic group {group.semantic_group_uid!r} references an "
                    "unknown true text pool"
                ) from error
            if (
                pool.branch != group.branch
                or pool.axis_id != group.axis_id
                or pool.polarity != group.polarity
                or pool.control_id != group.control_id
            ):
                raise SchemaError(
                    f"semantic group {group.semantic_group_uid!r} and its true "
                    "text pool disagree"
                )
            family = groups_by_family.get(group.semantic_family_uid)
            if family is None:
                family = {}
                groups_by_family[group.semantic_family_uid] = family
                family_order.append(group.semantic_family_uid)
            if group.polarity in family:
                raise SchemaError(
                    f"semantic family {group.semantic_family_uid!r} has duplicate "
                    f"polarity {group.polarity:+d}"
                )
            family[group.polarity] = group

        by_split_branch_axis: dict[
            str, dict[str, dict[str, list[_SemanticFamily]]]
        ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for family_uid in family_order:
            signed_groups = groups_by_family[family_uid]
            if set(signed_groups) != {-1, 1}:
                raise SchemaError(
                    f"semantic family {family_uid!r} must contain exactly two "
                    "opposite polarities"
                )
            negative = signed_groups[-1]
            positive = signed_groups[1]
            if (
                negative.split,
                negative.branch,
                negative.axis_id,
                negative.case_id,
                negative.candidate_pool_id,
            ) != (
                positive.split,
                positive.branch,
                positive.axis_id,
                positive.case_id,
                positive.candidate_pool_id,
            ):
                raise SchemaError(
                    f"semantic family {family_uid!r} has inconsistent identity"
                )
            family = _SemanticFamily(
                split=negative.split,
                branch=negative.branch,
                axis_id=negative.axis_id,
                semantic_family_uid=family_uid,
                groups_by_polarity=signed_groups,
            )
            by_split_branch_axis[family.split][family.branch][family.axis_id].append(
                family
            )

        self._families_by_split_branch_axis = {
            split: {
                branch: {
                    axis_id: tuple(families)
                    for axis_id, families in axes.items()
                }
                for branch, axes in branches.items()
            }
            for split, branches in by_split_branch_axis.items()
        }

    def epoch_group_count(self, split: str) -> int:
        """Return one deterministic exposure epoch for replacement sampling.

        Hierarchical sampling is intentionally with replacement, so an epoch is
        the number of active semantic groups rather than a no-replacement
        coverage guarantee.  The explicit epoch seed still makes the resulting
        exposure sequence exactly reproducible.
        """

        try:
            branches = self._families_by_split_branch_axis[split]
        except KeyError as error:
            raise SchemaError(
                f"runtime true-guidance split {split!r} has no semantic groups"
            ) from error
        count = sum(
            2 * len(families)
            for branch, axes in branches.items()
            if self.branch_weights.get(branch, 0.0) > 0.0
            for families in axes.values()
        )
        if count < 1:  # pragma: no cover - constructor enforces an active branch
            raise QualityError(f"no active semantic groups are available for {split!r}")
        return count

    @staticmethod
    def _numeric_seed(
        *,
        base_seed: int,
        split_ordinal: int,
        epoch: int,
        draw_index: int,
    ) -> int:
        for name, value in (("epoch", epoch), ("draw_index", draw_index)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if value >= 2**64:
                raise ValueError(f"{name} must be below 2**64")
        if split_ordinal not in range(4):
            raise ValueError("split_ordinal must be between zero and three")

        def nonnegative(value: int) -> int:
            return 2 * value if value >= 0 else -2 * value - 1

        return (
            (nonnegative(base_seed) << 194)
            | (split_ordinal << 192)
            | (epoch << 64)
            | draw_index
        )

    @staticmethod
    def _weighted_choice(
        generator: random.Random,
        weighted: Sequence[tuple[str, float]],
    ) -> str:
        total = sum(weight for _name, weight in weighted)
        threshold = generator.random() * total
        cumulative = 0.0
        for name, weight in weighted:
            cumulative += weight
            if threshold < cumulative:
                return name
        return weighted[-1][0]

    def sample(
        self,
        *,
        split: str,
        epoch: int = 0,
        draw_index: int = 0,
        evaluation: bool = False,
    ) -> RuntimeTrueGuidanceDraw:
        """Draw one true guidance control for the requested split."""

        try:
            split_ordinal = SPLIT_ORDINALS[split]
            branches = self._families_by_split_branch_axis[split]
        except KeyError as error:
            raise SchemaError(
                f"runtime true-guidance split {split!r} has no semantic groups"
            ) from error
        generator = random.Random(
            self._numeric_seed(
                base_seed=self.evaluation_seed if evaluation else self.seed,
                split_ordinal=split_ordinal,
                epoch=epoch,
                draw_index=draw_index,
            )
        )
        weighted_branches = [
            (branch, self.branch_weights[branch])
            for branch in ("metric", "structure")
            if self.branch_weights[branch] > 0.0 and branches.get(branch)
        ]
        if not weighted_branches:
            raise QualityError(
                f"no positive-weight true-guidance branch is available for {split!r}"
            )
        branch = self._weighted_choice(generator, weighted_branches)
        axes = branches[branch]
        axis_id = generator.choice(tuple(axes))
        family = generator.choice(axes[axis_id])
        polarity = generator.choice((-1, 1))
        group = family.groups_by_polarity[polarity]
        pool = self._text_registry.pool(group.text_pool_id)
        variant = (
            pool.canonical
            if evaluation
            else generator.choice(pool.training_variants)
        )
        return RuntimeTrueGuidanceDraw(
            group=group,
            text_id=variant.text_id,
            text_pool_id=pool.text_pool_id,
            control_id=pool.control_id,
            wording_usage=variant.usage,
        )


class RuntimeBranchCycleSampler:
    """Draw equal-weight branches through deterministic no-replacement cycles.

    Formal hybrid runs need a 1:1 branch schedule that does not turn the
    lower-volume branch into a rare event.  Every active branch receives one
    draw per alternating pair.  Within one branch, every semantic group is
    emitted exactly once before a newly seeded cycle starts.  The shorter
    branch can therefore recycle while the metric branch continues its own
    macro-cycle.

    Ordering uses only explicit split, branch, epoch, cycle, and numeric draw
    coordinates; it never depends on process state, Python string hashes, or
    corpus-content-derived IDs.
    """

    _BRANCH_ORDINALS = {"metric": 0, "structure": 1}
    _CYCLE_ORDER_CACHE_CAPACITY = 8

    def __init__(
        self,
        *,
        groups: Sequence[SemanticGroup],
        text_pools: Sequence[TextPool],
        seed: int,
        evaluation_seed: int,
        branch_weights: Mapping[str, float],
        training_wording: str = "all_train",
        evaluation_wording: str = "canonical",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SchemaError("runtime branch-cycle seed must be an integer")
        if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
            raise SchemaError("runtime branch-cycle evaluation seed must be an integer")
        if training_wording != "all_train":
            raise SchemaError("runtime branch-cycle training_wording must be 'all_train'")
        if evaluation_wording != "canonical":
            raise SchemaError("runtime branch-cycle evaluation_wording must be 'canonical'")
        if set(branch_weights) - set(self._BRANCH_ORDINALS):
            raise SchemaError("runtime branch-cycle weights have unknown keys")
        weights = {
            branch: float(branch_weights.get(branch, 0.0))
            for branch in self._BRANCH_ORDINALS
        }
        if any(value < 0.0 for value in weights.values()):
            raise SchemaError("runtime branch-cycle weights must be non-negative")
        if not groups:
            raise SchemaError("runtime branch-cycle sampler requires semantic groups")

        self.seed = seed
        self.evaluation_seed = evaluation_seed
        self.branch_weights = weights
        self.training_wording = training_wording
        self.evaluation_wording = evaluation_wording
        self._text_registry = TextPoolRegistry(text_pools)
        groups_by_split_branch: dict[str, dict[str, list[SemanticGroup]]] = defaultdict(
            lambda: defaultdict(list)
        )
        seen_group_ids: set[str] = set()
        for group in groups:
            if group.semantic_group_uid in seen_group_ids:
                raise SchemaError("runtime branch-cycle index contains duplicate semantic groups")
            seen_group_ids.add(group.semantic_group_uid)
            try:
                pool = self._text_registry.pool(group.text_pool_id)
            except KeyError as error:
                raise SchemaError(
                    f"semantic group {group.semantic_group_uid!r} references an unknown true text pool"
                ) from error
            if (
                pool.branch != group.branch
                or pool.axis_id != group.axis_id
                or pool.polarity != group.polarity
                or pool.control_id != group.control_id
            ):
                raise SchemaError(
                    f"semantic group {group.semantic_group_uid!r} and its true text pool disagree"
                )
            groups_by_split_branch[group.split][group.branch].append(group)

        schedules: dict[str, tuple[str, ...]] = {}
        frozen_groups: dict[str, dict[str, tuple[SemanticGroup, ...]]] = {}
        for split, branches in groups_by_split_branch.items():
            active = tuple(
                branch
                for branch in self._BRANCH_ORDINALS
                if weights[branch] > 0.0 and branches.get(branch)
            )
            if not active:
                raise QualityError(f"no positive-weight branch is available for {split!r}")
            if len(active) == 2 and not math.isclose(
                weights[active[0]], weights[active[1]], rel_tol=0.0, abs_tol=1e-12
            ):
                raise SchemaError(
                    "branch-cycle sampling requires equal positive metric and structure weights"
                )
            schedules[split] = active
            frozen_groups[split] = {
                branch: tuple(branches[branch]) for branch in active
            }
        self._groups_by_split_branch = frozen_groups
        self._schedules = schedules
        # A formal 96-group batch resolves many consecutive offsets in the
        # same branch cycle.  Constructing the exact deterministic ordering
        # once and reusing it is observationally identical to shuffling a new
        # ``range(len(groups))`` for every draw, but avoids repeatedly doing
        # O(group_count) Python RNG work on the CPU producer.
        self._cycle_order_cache: OrderedDict[
            tuple[str, str, int, int, bool], tuple[int, ...]
        ] = OrderedDict()

    def epoch_group_count(self, split: str) -> int:
        """Return one complete branch-balanced macro-cycle.

        Every active branch is visited equally often.  The longest branch is
        therefore covered once and shorter branches deterministically recycle,
        which is the smallest exposure count that both preserves 1:1 branch
        balance and visits every active group.
        """

        try:
            schedule = self._schedules[split]
            groups = self._groups_by_split_branch[split]
        except KeyError as error:
            raise SchemaError(
                f"runtime branch-cycle split {split!r} has no semantic groups"
            ) from error
        return len(schedule) * max(len(groups[branch]) for branch in schedule)

    @staticmethod
    def _cycle_seed(
        *,
        base_seed: int,
        split_ordinal: int,
        branch_ordinal: int,
        epoch: int,
        cycle_index: int,
    ) -> int:
        for name, value in (("epoch", epoch), ("cycle_index", cycle_index)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if value >= 2**64:
                raise ValueError(f"{name} must be below 2**64")
        if split_ordinal not in range(4):
            raise ValueError("split_ordinal must be between zero and three")
        if branch_ordinal not in range(2):
            raise ValueError("branch_ordinal must be zero or one")
        nonnegative_seed = 2 * base_seed if base_seed >= 0 else -2 * base_seed - 1
        return (
            (nonnegative_seed << 194)
            | (split_ordinal << 192)
            | (branch_ordinal << 190)
            | (epoch << 64)
            | cycle_index
        )

    def cycle_state(
        self,
        *,
        split: str,
        epoch: int = 0,
        draw_index: int = 0,
    ) -> dict[str, int | str]:
        """Return explicit branch/cycle coordinates for a global draw."""

        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")
        if isinstance(draw_index, bool) or not isinstance(draw_index, int) or draw_index < 0:
            raise ValueError("draw_index must be a non-negative integer")
        try:
            schedule = self._schedules[split]
        except KeyError as error:
            raise SchemaError(
                f"runtime branch-cycle split {split!r} has no semantic groups"
            ) from error
        branch = schedule[draw_index % len(schedule)]
        branch_draw_index = draw_index // len(schedule)
        groups = self._groups_by_split_branch[split][branch]
        return {
            "split": split,
            "branch": branch,
            "draw_index": draw_index,
            "branch_draw_index": branch_draw_index,
            "cycle_index": branch_draw_index // len(groups),
            "cycle_offset": branch_draw_index % len(groups),
            "cycle_group_count": len(groups),
        }

    def _cycle_ordering(
        self,
        *,
        split: str,
        branch: str,
        epoch: int,
        cycle_index: int,
        evaluation: bool,
    ) -> tuple[int, ...]:
        """Return the legacy Fisher-Yates ordering for one immutable cycle.

        The cache key contains every coordinate that affects the old shuffle
        seed.  Values are immutable tuples so concurrent readers cannot alter
        a completed cycle order; bounded LRU eviction only causes the exact
        same deterministic permutation to be rebuilt later.
        """

        key = (split, branch, epoch, cycle_index, evaluation)
        cached = self._cycle_order_cache.pop(key, None)
        if cached is not None:
            self._cycle_order_cache[key] = cached
            return cached
        groups = self._groups_by_split_branch[split][branch]
        ordering = list(range(len(groups)))
        random.Random(
            self._cycle_seed(
                base_seed=self.evaluation_seed if evaluation else self.seed,
                split_ordinal=SPLIT_ORDINALS[split],
                branch_ordinal=self._BRANCH_ORDINALS[branch],
                epoch=epoch,
                cycle_index=cycle_index,
            )
        ).shuffle(ordering)
        result = tuple(ordering)
        self._cycle_order_cache[key] = result
        while len(self._cycle_order_cache) > self._CYCLE_ORDER_CACHE_CAPACITY:
            self._cycle_order_cache.popitem(last=False)
        return result

    def sample(
        self,
        *,
        split: str,
        epoch: int = 0,
        draw_index: int = 0,
        evaluation: bool = False,
    ) -> RuntimeTrueGuidanceDraw:
        """Select one canonical group and wording from the formal branch cycle."""

        state = self.cycle_state(split=split, epoch=epoch, draw_index=draw_index)
        branch = str(state["branch"])
        groups = self._groups_by_split_branch[split][branch]
        split_ordinal = SPLIT_ORDINALS[split]
        ordering = self._cycle_ordering(
            split=split,
            branch=branch,
            epoch=epoch,
            cycle_index=int(state["cycle_index"]),
            evaluation=evaluation,
        )
        group = groups[ordering[int(state["cycle_offset"])]]
        pool = self._text_registry.pool(group.text_pool_id)
        wording_generator = random.Random(
            RuntimeTrueGuidanceSampler._numeric_seed(
                base_seed=self.evaluation_seed if evaluation else self.seed,
                split_ordinal=split_ordinal,
                epoch=epoch,
                draw_index=draw_index,
            )
        )
        variant = pool.canonical if evaluation else wording_generator.choice(pool.training_variants)
        return RuntimeTrueGuidanceDraw(
            group=group,
            text_id=variant.text_id,
            text_pool_id=pool.text_pool_id,
            control_id=pool.control_id,
            wording_usage=variant.usage,
        )


def true_guidance_sampling_policy(config: SamplingConfig) -> dict[str, object]:
    """Return serializable configuration for downstream runtime consumers."""

    return {
        "schema_version": "guidance_v3.true_guidance_sampling_policy.v1",
        "mode": "runtime_true_guidance",
        "sampling_order": [
            "branch",
            "axis",
            "semantic_family",
            "polarity",
            "wording",
        ],
        "branch_weights": dict(config.branch_weights),
        "axis_sampling": config.axis_sampling,
        "semantic_family_sampling": config.semantic_family_sampling,
        "polarity_sampling": config.polarity_sampling,
        "training_wording": config.training_wording,
        "evaluation_wording": config.evaluation_wording,
        "seed": config.seed,
        "evaluation_seed": config.evaluation_seed,
        "draws_materialized": False,
        "identity_basis": "explicit_split_and_numeric_epoch_draw_ordinals",
        "content_hashes_computed": False,
    }
