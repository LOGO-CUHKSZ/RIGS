"""Canonical semantic relations between groups, controls, and surface text."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from retro_guided_generation.guidance.errors import SchemaError
from retro_guided_generation.guidance.schema import SemanticGroup, TextPool, TextVariant


@dataclass(frozen=True, slots=True)
class TextOwnership:
    text: TextVariant
    pool: TextPool


class RelationResolver:
    """Resolve true, synonym, opposite, and different-axis relations.

    This is the only semantic relation implementation used by training,
    validation, test, shuffle construction, and evaluation export.
    """

    def __init__(
        self,
        *,
        groups: Sequence[SemanticGroup],
        text_pools: Sequence[TextPool],
    ) -> None:
        self.groups = tuple(groups)
        self.text_pools = tuple(text_pools)
        group_ids = tuple(group.semantic_group_uid for group in self.groups)
        pool_ids = tuple(pool.text_pool_id for pool in self.text_pools)
        if len(group_ids) != len(set(group_ids)):
            raise SchemaError("relation resolver received duplicate semantic groups")
        if len(pool_ids) != len(set(pool_ids)):
            raise SchemaError("relation resolver received duplicate text pools")
        self._groups = {group.semantic_group_uid: group for group in self.groups}
        self._pools = {pool.text_pool_id: pool for pool in self.text_pools}
        self._texts: dict[str, TextOwnership] = {}
        for pool in self.text_pools:
            for variant in pool.variants:
                if variant.text_id in self._texts:
                    raise SchemaError(f"duplicate text ID {variant.text_id!r}")
                self._texts[variant.text_id] = TextOwnership(variant, pool)
        for group in self.groups:
            pool = self.pool(group.text_pool_id)
            opposite = self.pool(group.opposite_text_pool_id)
            if (
                pool.branch != group.branch
                or pool.axis_id != group.axis_id
                or pool.polarity != group.polarity
                or pool.control_id != group.control_id
            ):
                raise SchemaError(
                    f"group {group.semantic_group_uid!r} differs from its true text pool"
                )
            if (
                opposite.axis_id != group.axis_id
                or opposite.polarity != -group.polarity
                or opposite.control_id != group.opposite_control_id
            ):
                raise SchemaError(
                    f"group {group.semantic_group_uid!r} differs from its opposite pool"
                )

    def group(self, group_uid: str) -> SemanticGroup:
        try:
            return self._groups[group_uid]
        except KeyError as error:
            raise KeyError(f"unknown semantic group {group_uid!r}") from error

    def pool(self, pool_id: str) -> TextPool:
        try:
            return self._pools[pool_id]
        except KeyError as error:
            raise KeyError(f"unknown text pool {pool_id!r}") from error

    def ownership(self, text_id: str) -> TextOwnership:
        try:
            return self._texts[text_id]
        except KeyError as error:
            raise KeyError(f"unknown text ID {text_id!r}") from error

    def relation(self, group_uid: str, text_id: str) -> str:
        group = self.group(group_uid)
        ownership = self.ownership(text_id)
        pool = ownership.pool
        if pool.text_pool_id == group.text_pool_id:
            return "canonical_true" if ownership.text.relation == "canonical" else "synonym_true"
        if pool.text_pool_id == group.opposite_text_pool_id:
            return "same_axis_antonym"
        if pool.axis_id == group.axis_id:
            return "same_axis_other"
        return "different_axis"

    def true_variants(
        self, group_uid: str, *, include_heldout: bool
    ) -> tuple[TextVariant, ...]:
        pool = self.pool(self.group(group_uid).text_pool_id)
        return tuple(
            variant
            for variant in pool.variants
            if include_heldout or variant.usage != "heldout"
        )

    def opposite_variants(self, group_uid: str) -> tuple[TextVariant, ...]:
        pool = self.pool(self.group(group_uid).opposite_text_pool_id)
        return pool.variants

    def assert_different_axis(self, group_uid: str, text_id: str) -> None:
        relation = self.relation(group_uid, text_id)
        if relation != "different_axis":
            raise SchemaError(
                f"text {text_id!r} has relation {relation!r} to group "
                f"{group_uid!r}, not 'different_axis'"
            )

