"""Text-pool indexing and semantic relation validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from retro_guided_generation.guidance.errors import SchemaError
from retro_guided_generation.guidance.registry import MetricAxisRegistry, signed_control_id
from retro_guided_generation.guidance.schema import TextPool, TextVariant, ensure_unique_text_ids


_REVIEWED_METRIC_TRUE_FIELDS = (
    "accepted",
    "natural_english",
    "transferable",
    "axis_specific",
    "negative_direction_correct",
    "positive_direction_correct",
    "negative_synonyms_equivalent",
    "positive_synonyms_equivalent",
    "opposite_direction_only",
)
_METRIC_TEXT_REVIEW_SCHEMA = "guidance_v3.metric_text_review.v1"


class TextPoolRegistry:
    def __init__(self, pools: Sequence[TextPool]) -> None:
        self._pools = tuple(pools)
        if not self._pools:
            raise SchemaError("text pool registry must not be empty")
        ensure_unique_text_ids(self._pools)
        pool_ids = tuple(pool.text_pool_id for pool in self._pools)
        control_ids = tuple(pool.control_id for pool in self._pools)
        if len(pool_ids) != len(set(pool_ids)):
            raise SchemaError("duplicate text_pool_id values")
        if len(control_ids) != len(set(control_ids)):
            raise SchemaError("duplicate control_id values")
        self._by_pool = {pool.text_pool_id: pool for pool in self._pools}
        self._by_control = {pool.control_id: pool for pool in self._pools}
        self._text = {
            variant.text_id: variant
            for pool in self._pools
            for variant in pool.variants
        }
        # A source corpus can legitimately reuse a generic sentence across
        # case-owned structure families.  In that setting +/- describes the
        # two members of one local counterfactual pair, not a global direction.
        # Global pools, by contrast, do use polarity as a reusable semantic
        # control.  Keep the stronger polarity check for them, while checking
        # case-owned wording only within its own semantic family.
        global_surface_polarities: dict[str, int] = {}
        family_surface_polarities: dict[tuple[str, str], int] = {}
        for pool in self._pools:
            pool_surfaces: set[str] = set()
            for variant in pool.variants:
                normalized_surface = " ".join(variant.text.casefold().split())
                if normalized_surface in pool_surfaces:
                    raise SchemaError(
                        f"pool {pool.text_pool_id!r} contains duplicate surface text"
                    )
                pool_surfaces.add(normalized_surface)
                if pool.semantic_family_uid is None:
                    polarity_index = global_surface_polarities
                    polarity_key: str | tuple[str, str] = normalized_surface
                    scope = "global text pools"
                else:
                    polarity_index = family_surface_polarities
                    polarity_key = (pool.semantic_family_uid, normalized_surface)
                    scope = f"semantic family {pool.semantic_family_uid!r}"
                existing_polarity = polarity_index.get(polarity_key)
                if existing_polarity is not None and existing_polarity != pool.polarity:
                    raise SchemaError(
                        "the same guidance surface text is assigned to opposite "
                        f"polarities within {scope}: {existing_polarity:+d} and "
                        f"{pool.polarity:+d}"
                    )
                polarity_index[polarity_key] = pool.polarity
            try:
                opposite = self._by_control[pool.opposite_control_id]
            except KeyError as error:
                raise SchemaError(
                    f"pool {pool.text_pool_id!r} references missing opposite control "
                    f"{pool.opposite_control_id!r}"
                ) from error
            if opposite.opposite_control_id != pool.control_id:
                raise SchemaError(
                    f"opposite relation is not symmetric for {pool.control_id!r}"
                )
            if opposite.axis_id != pool.axis_id:
                raise SchemaError("opposite controls must share axis_id")
            if opposite.polarity != -pool.polarity:
                raise SchemaError("opposite controls must have reverse polarity")
            if opposite.branch != pool.branch:
                raise SchemaError("opposite controls must share branch")
            if opposite.semantic_family_uid != pool.semantic_family_uid:
                raise SchemaError("opposite controls must share semantic family")

    def __iter__(self):
        return iter(self._pools)

    def __len__(self) -> int:
        return len(self._pools)

    def pool(self, text_pool_id: str) -> TextPool:
        try:
            return self._by_pool[text_pool_id]
        except KeyError as error:
            raise KeyError(f"unknown text pool {text_pool_id!r}") from error

    def control(self, control_id: str) -> TextPool:
        try:
            return self._by_control[control_id]
        except KeyError as error:
            raise KeyError(f"unknown guidance control {control_id!r}") from error

    def text(self, text_id: str) -> TextVariant:
        try:
            return self._text[text_id]
        except KeyError as error:
            raise KeyError(f"unknown guidance text {text_id!r}") from error

    def extend(self, pools: Sequence[TextPool]) -> TextPoolRegistry:
        return TextPoolRegistry((*self._pools, *pools))

    def to_rows(self) -> list[dict]:
        return [pool.to_dict() for pool in self._pools]


def validate_metric_text_pools(
    axes: MetricAxisRegistry,
    pools: Sequence[TextPool],
    *,
    require_synonyms: bool,
    require_heldout_synonym: bool,
    require_review_provenance: bool = False,
) -> TextPoolRegistry:
    registry = TextPoolRegistry(pools)
    if any(pool.branch != "metric" for pool in registry):
        raise SchemaError("metric text pool file contains a non-metric pool")
    expected = {
        signed_control_id(axis.axis_id, direction)
        for axis in axes
        for direction in axis.allowed_polarities
    }
    actual = {pool.control_id for pool in registry}
    if expected != actual:
        raise SchemaError(
            "metric text controls differ from axis registry: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    for pool in registry:
        if pool.text_pool_id != pool.control_id:
            raise SchemaError("global metric text_pool_id must equal control_id")
        synonyms = [item for item in pool.variants if item.relation == "synonym"]
        if require_synonyms and not synonyms:
            raise SchemaError(f"metric pool {pool.text_pool_id!r} has no synonyms")
        if require_heldout_synonym and not pool.heldout_variants:
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} has no heldout synonym"
            )
        if require_review_provenance:
            _validate_reviewed_metric_text_pool(pool)
    return registry


def _validate_reviewed_metric_text_pool(pool: TextPool) -> None:
    """Require the metric-text reviewer verdict retained by the LLM protocol."""

    expected_review: dict[str, object] | None = None
    for variant in pool.variants:
        raw_review = variant.review.get("review")
        if not isinstance(raw_review, Mapping):
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} variant {variant.text_id!r} "
                "is missing reviewed provenance"
            )
        review = dict(raw_review)
        if review.get("schema_version") != _METRIC_TEXT_REVIEW_SCHEMA:
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} has an invalid review schema"
            )
        if review.get("axis_id") != pool.axis_id:
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} review axis differs from pool"
            )
        if any(review.get(field) is not True for field in _REVIEWED_METRIC_TRUE_FIELDS):
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} has a rejected review verdict"
            )
        if review.get("identifier_or_case_leakage") is not False:
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} review does not clear leakage"
            )
        if expected_review is None:
            expected_review = review
        elif review != expected_review:
            raise SchemaError(
                f"metric pool {pool.text_pool_id!r} has inconsistent review provenance"
            )


def validate_structure_text_pools(
    pools: Sequence[TextPool],
    *,
    require_synonyms: bool,
    require_heldout_synonym: bool,
) -> TextPoolRegistry:
    registry = TextPoolRegistry(pools)
    if any(pool.branch != "structure" for pool in registry):
        raise SchemaError("structure text pools contain a non-structure pool")
    for pool in registry:
        synonyms = [item for item in pool.variants if item.relation == "synonym"]
        if require_synonyms and not synonyms:
            raise SchemaError(f"structure pool {pool.text_pool_id!r} has no synonyms")
        if require_heldout_synonym and not pool.heldout_variants:
            raise SchemaError(
                f"structure pool {pool.text_pool_id!r} has no heldout synonym"
            )
    return registry
