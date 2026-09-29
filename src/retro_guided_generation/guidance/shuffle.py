"""Split-local runtime sampling for different-axis shuffled guidance."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from retro_guided_generation.guidance.errors import QualityError, SchemaError
from retro_guided_generation.guidance.schema import (
    SemanticGroup,
    ShuffleDonorPool,
    TextPool,
)


SPLIT_ORDINALS = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True, slots=True)
class RuntimeShuffleDraw:
    """One sampled donor with enough provenance for relation validation."""

    recipient_group_uid: str
    donor_text_id: str
    donor_text_pool_id: str
    donor_control_id: str
    donor_branch: str
    donor_axis_id: str
    donor_polarity: int
    donor_case_id: str | None
    source_kind: str
    relation: str = "different_axis"

    def to_dict(self) -> dict[str, object]:
        return {
            "recipient_group_uid": self.recipient_group_uid,
            "donor_text_id": self.donor_text_id,
            "donor_text_pool_id": self.donor_text_pool_id,
            "donor_control_id": self.donor_control_id,
            "donor_branch": self.donor_branch,
            "donor_axis_id": self.donor_axis_id,
            "donor_polarity": self.donor_polarity,
            "donor_case_id": self.donor_case_id,
            "source_kind": self.source_kind,
            "relation": self.relation,
        }


def build_shuffle_donor_pools(
    *,
    groups: Sequence[SemanticGroup],
    text_pools: Sequence[TextPool],
) -> tuple[ShuffleDonorPool, ...]:
    """Build one compact donor entry per signed text pool.

    Metric pools are global and therefore have no case owner. Structure pools
    are split-local and must be referenced by groups from exactly one case.
    No recipient-specific donor lists or sampled text IDs are materialized.
    """

    structure_owners: dict[str, set[str]] = defaultdict(set)
    for group in groups:
        if group.branch == "structure":
            structure_owners[group.text_pool_id].add(group.case_id)

    donors: list[ShuffleDonorPool] = []
    seen_pool_ids: set[str] = set()
    for pool in text_pools:
        if pool.text_pool_id in seen_pool_ids:
            raise SchemaError(
                f"duplicate shuffle donor text pool {pool.text_pool_id!r}"
            )
        seen_pool_ids.add(pool.text_pool_id)
        case_id: str | None = None
        if pool.branch == "structure":
            owners = structure_owners.get(pool.text_pool_id, set())
            if len(owners) != 1:
                raise QualityError(
                    f"structure text pool {pool.text_pool_id!r} must have exactly "
                    f"one split-local case owner; found {sorted(owners)!r}"
                )
            case_id = next(iter(owners))
        donors.append(
            ShuffleDonorPool(
                text_pool_id=pool.text_pool_id,
                branch=pool.branch,
                axis_id=pool.axis_id,
                polarity=pool.polarity,
                control_id=pool.control_id,
                case_id=case_id,
                canonical_text_id=pool.canonical.text_id,
                training_text_ids=tuple(
                    variant.text_id for variant in pool.training_variants
                ),
            )
        )
    return tuple(
        sorted(
            donors,
            key=lambda item: (
                item.branch,
                item.axis_id,
                item.polarity,
                item.case_id or "",
                item.text_pool_id,
            ),
        )
    )


class RuntimeShuffleSampler:
    """Sample a legal different-axis control without frozen mappings.

    Sampling is hierarchical: source, axis, polarity, pool, then wording. This
    prevents an axis with many cases or paraphrases from receiving accidental
    extra probability mass. Randomness is derived only from explicit numeric
    ordinals and seeds; content digests and process-dependent string hashing
    are not used.
    """

    def __init__(
        self,
        *,
        donor_pools: Sequence[ShuffleDonorPool],
        seed: int,
        evaluation_seed: int,
        metric_source_weights: Mapping[str, float],
        training_wording: str = "all_train",
        evaluation_wording: str = "canonical",
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SchemaError("runtime shuffle seed must be an integer")
        if isinstance(evaluation_seed, bool) or not isinstance(evaluation_seed, int):
            raise SchemaError("runtime shuffle evaluation seed must be an integer")
        if training_wording != "all_train":
            raise SchemaError("runtime shuffle training_wording must be 'all_train'")
        if evaluation_wording != "canonical":
            raise SchemaError(
                "runtime shuffle evaluation_wording must be 'canonical'"
            )
        if set(metric_source_weights) - {"metric", "structure"}:
            raise SchemaError("runtime shuffle source weights have unknown keys")
        self.seed = seed
        self.evaluation_seed = evaluation_seed
        self.metric_source_weights = {
            key: float(metric_source_weights.get(key, 0.0))
            for key in ("metric", "structure")
        }
        if any(value < 0.0 for value in self.metric_source_weights.values()):
            raise SchemaError("runtime shuffle source weights must be non-negative")
        if sum(self.metric_source_weights.values()) <= 0.0:
            raise SchemaError("runtime shuffle source weights must sum above zero")
        self.training_wording = training_wording
        self.evaluation_wording = evaluation_wording

        self._donors = tuple(donor_pools)
        donor_ids = tuple(item.text_pool_id for item in self._donors)
        if len(donor_ids) != len(set(donor_ids)):
            raise SchemaError("runtime shuffle donor index contains duplicate pools")

        by_branch_axis_polarity: dict[
            tuple[str, str, int], list[ShuffleDonorPool]
        ] = defaultdict(list)
        axis_counts: Counter[tuple[str, str]] = Counter()
        axis_case_counts: Counter[tuple[str, str, str]] = Counter()
        for donor in self._donors:
            by_branch_axis_polarity[
                (donor.branch, donor.axis_id, donor.polarity)
            ].append(donor)
            axis_counts[(donor.branch, donor.axis_id)] += 1
            if donor.case_id is not None:
                axis_case_counts[(donor.branch, donor.axis_id, donor.case_id)] += 1

        self._by_branch_axis_polarity = {
            key: tuple(sorted(value, key=lambda item: item.text_pool_id))
            for key, value in by_branch_axis_polarity.items()
        }
        self._axis_counts = axis_counts
        self._axis_case_counts = axis_case_counts
        self._axes = {
            branch: tuple(
                sorted(
                    {
                        donor.axis_id
                        for donor in self._donors
                        if donor.branch == branch
                    }
                )
            )
            for branch in ("metric", "structure")
        }

    @staticmethod
    def _numeric_seed(
        *,
        base_seed: int,
        split_ordinal: int,
        group_ordinal: int,
        epoch: int,
        draw_index: int,
    ) -> int:
        for name, value in (
            ("split_ordinal", split_ordinal),
            ("group_ordinal", group_ordinal),
            ("epoch", epoch),
            ("draw_index", draw_index),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            if value >= 2**64:
                raise ValueError(f"{name} must be below 2**64")
        if split_ordinal >= 4:
            raise ValueError("split_ordinal must be below 4")
        def nonnegative(value: int) -> int:
            return 2 * value if value >= 0 else -2 * value - 1

        # Fixed-width packing composes explicit numeric coordinates
        # injectively. It is independent of record contents, process hash
        # state, and worker count.
        return (
            (nonnegative(base_seed) << 194)
            | (split_ordinal << 192)
            | (group_ordinal << 128)
            | (epoch << 64)
            | draw_index
        )

    def _axis_has_legal_pool(
        self,
        *,
        branch: str,
        axis_id: str,
        recipient: SemanticGroup,
    ) -> bool:
        if axis_id == recipient.axis_id:
            return False
        total = self._axis_counts[(branch, axis_id)]
        if branch == "structure":
            total -= self._axis_case_counts[(branch, axis_id, recipient.case_id)]
        return total > 0

    def _source_available(self, source: str, recipient: SemanticGroup) -> bool:
        branch = "metric" if source == "metric" else "structure"
        return any(
            self._axis_has_legal_pool(
                branch=branch,
                axis_id=axis_id,
                recipient=recipient,
            )
            for axis_id in self._axes[branch]
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

    def _sample_axis(
        self,
        *,
        branch: str,
        recipient: SemanticGroup,
        generator: random.Random,
    ) -> str:
        axes = self._axes[branch]
        if not axes:
            raise QualityError(f"no {branch} axes exist in the shuffle donor index")
        for _ in range(128):
            axis_id = generator.choice(axes)
            if self._axis_has_legal_pool(
                branch=branch,
                axis_id=axis_id,
                recipient=recipient,
            ):
                return axis_id
        legal = tuple(
            axis_id
            for axis_id in axes
            if self._axis_has_legal_pool(
                branch=branch,
                axis_id=axis_id,
                recipient=recipient,
            )
        )
        if not legal:
            raise QualityError(
                f"no legal {branch} donor axis exists for "
                f"{recipient.semantic_group_uid!r}"
            )
        return generator.choice(legal)

    def _sample_pool(
        self,
        *,
        branch: str,
        axis_id: str,
        recipient: SemanticGroup,
        generator: random.Random,
    ) -> ShuffleDonorPool:
        legal_polarities = []
        for donor_polarity in (-1, 1):
            donors = self._by_branch_axis_polarity.get(
                (branch, axis_id, donor_polarity), ()
            )
            if any(
                donor.case_id is None or donor.case_id != recipient.case_id
                for donor in donors
            ):
                legal_polarities.append(donor_polarity)
        if not legal_polarities:
            raise QualityError(
                f"no legal donor polarity exists for axis {axis_id!r}"
            )
        donor_polarity = generator.choice(legal_polarities)
        donors = self._by_branch_axis_polarity[
            (branch, axis_id, donor_polarity)
        ]
        for _ in range(128):
            donor = generator.choice(donors)
            if donor.case_id is None or donor.case_id != recipient.case_id:
                return donor
        legal = tuple(
            donor
            for donor in donors
            if donor.case_id is None or donor.case_id != recipient.case_id
        )
        if not legal:
            raise QualityError(
                f"no cross-case donor pool exists for axis {axis_id!r}"
            )
        return generator.choice(legal)

    def sample(
        self,
        recipient: SemanticGroup,
        *,
        group_ordinal: int,
        epoch: int = 0,
        draw_index: int = 0,
        evaluation: bool = False,
    ) -> RuntimeShuffleDraw:
        """Draw one legal donor for a recipient group."""

        try:
            split_ordinal = SPLIT_ORDINALS[recipient.split]
        except KeyError as error:
            raise SchemaError(
                f"unsupported shuffle split {recipient.split!r}"
            ) from error
        base_seed = self.evaluation_seed if evaluation else self.seed
        generator = random.Random(
            self._numeric_seed(
                base_seed=base_seed,
                split_ordinal=split_ordinal,
                group_ordinal=group_ordinal,
                epoch=epoch,
                draw_index=draw_index,
            )
        )

        if recipient.branch == "structure":
            source = "structure"
        elif recipient.branch == "metric":
            weighted = [
                (source, self.metric_source_weights[source])
                for source in ("metric", "structure")
                if self.metric_source_weights[source] > 0.0
                and self._source_available(source, recipient)
            ]
            if not weighted:
                raise QualityError(
                    f"no legal different-axis donor source exists for "
                    f"{recipient.semantic_group_uid!r}"
                )
            source = self._weighted_choice(generator, weighted)
        else:
            raise SchemaError(
                f"unsupported shuffle recipient branch {recipient.branch!r}"
            )

        branch = "metric" if source == "metric" else "structure"
        axis_id = self._sample_axis(
            branch=branch,
            recipient=recipient,
            generator=generator,
        )
        donor = self._sample_pool(
            branch=branch,
            axis_id=axis_id,
            recipient=recipient,
            generator=generator,
        )
        wording_id = (
            donor.canonical_text_id
            if evaluation
            else generator.choice(donor.training_text_ids)
        )
        if donor.axis_id == recipient.axis_id:
            raise AssertionError("runtime shuffle selected the recipient axis")
        if donor.branch == "structure" and donor.case_id == recipient.case_id:
            raise AssertionError("runtime shuffle selected a same-case structure pool")
        return RuntimeShuffleDraw(
            recipient_group_uid=recipient.semantic_group_uid,
            donor_text_id=wording_id,
            donor_text_pool_id=donor.text_pool_id,
            donor_control_id=donor.control_id,
            donor_branch=donor.branch,
            donor_axis_id=donor.axis_id,
            donor_polarity=donor.polarity,
            donor_case_id=donor.case_id,
            source_kind=(
                "other_metric_axis"
                if donor.branch == "metric"
                else "cross_case_structure"
            ),
        )
