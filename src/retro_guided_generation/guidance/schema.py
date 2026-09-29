"""Strict, versioned, hash-free contracts for Guidance Dataset V3."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

from retro_guided_generation.guidance._validation import (
    boolean,
    clone_json,
    id_tuple,
    identifier,
    json_object,
    number,
    optional_number,
    optional_text,
    polarity,
    schema,
    split_name,
    strict_object,
    string_tuple,
    text,
)
from retro_guided_generation.guidance.errors import SchemaError


T = TypeVar("T", bound="JsonModel")


class JsonModel:
    SCHEMA_VERSION: ClassVar[str]

    @classmethod
    def from_dict(cls: type[T], value: Mapping[str, Any]) -> T:
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class CandidateRoute(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.candidate_route.v1"

    route_id: str
    precursor_smiles: str
    base_probability: float | None = None
    metric_values: dict[str, float] = field(default_factory=dict)
    structure_evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.route_id, "CandidateRoute.route_id")
        text(self.precursor_smiles, "CandidateRoute.precursor_smiles")
        optional_number(
            self.base_probability,
            "CandidateRoute.base_probability",
            nonnegative=True,
        )
        for key, value in self.metric_values.items():
            identifier(key, f"CandidateRoute.metric_values[{key!r}].key")
            number(value, f"CandidateRoute.metric_values[{key!r}]")
        json_object(self.structure_evidence, "CandidateRoute.structure_evidence")
        json_object(self.metadata, "CandidateRoute.metadata")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidateRoute:
        row = strict_object(
            value,
            model="CandidateRoute",
            allowed={
                "schema_version",
                "route_id",
                "precursor_smiles",
                "base_probability",
                "metric_values",
                "structure_evidence",
                "metadata",
            },
            required={"route_id", "precursor_smiles"},
        )
        if "schema_version" in row:
            schema(row, cls.SCHEMA_VERSION, "CandidateRoute")
        raw_metrics = row.get("metric_values", {})
        if not isinstance(raw_metrics, Mapping):
            raise SchemaError("CandidateRoute.metric_values must be an object")
        metrics = {
            identifier(key, "CandidateRoute.metric_values.key"): number(
                item, f"CandidateRoute.metric_values[{key!r}]"
            )
            for key, item in raw_metrics.items()
        }
        return cls(
            route_id=identifier(row["route_id"], "CandidateRoute.route_id"),
            precursor_smiles=text(
                row["precursor_smiles"], "CandidateRoute.precursor_smiles"
            ),
            base_probability=optional_number(
                row.get("base_probability"),
                "CandidateRoute.base_probability",
                nonnegative=True,
            ),
            metric_values=metrics,
            structure_evidence=json_object(
                row.get("structure_evidence", {}),
                "CandidateRoute.structure_evidence",
            ),
            metadata=json_object(row.get("metadata", {}), "CandidateRoute.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "route_id": self.route_id,
            "precursor_smiles": self.precursor_smiles,
            "base_probability": self.base_probability,
            "metric_values": dict(self.metric_values),
            "structure_evidence": clone_json(self.structure_evidence),
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class NullMultiRouteSet(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.null_multi_route_set.v1"

    case_id: str
    split: str
    product_smiles: str
    candidate_pool_id: str
    routes: tuple[CandidateRoute, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.case_id, "NullMultiRouteSet.case_id")
        split_name(self.split, "NullMultiRouteSet.split")
        text(self.product_smiles, "NullMultiRouteSet.product_smiles")
        identifier(self.candidate_pool_id, "NullMultiRouteSet.candidate_pool_id")
        if len(self.routes) < 2:
            raise SchemaError("NullMultiRouteSet.routes must contain at least two routes")
        route_ids = tuple(route.route_id for route in self.routes)
        if len(route_ids) != len(set(route_ids)):
            raise SchemaError("NullMultiRouteSet.routes contains duplicate route_id values")
        json_object(self.metadata, "NullMultiRouteSet.metadata")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NullMultiRouteSet:
        row = strict_object(
            value,
            model="NullMultiRouteSet",
            allowed={
                "schema_version",
                "case_id",
                "split",
                "product_smiles",
                "candidate_pool_id",
                "routes",
                "metadata",
            },
            required={
                "schema_version",
                "case_id",
                "split",
                "product_smiles",
                "candidate_pool_id",
                "routes",
            },
        )
        schema(row, cls.SCHEMA_VERSION, "NullMultiRouteSet")
        routes = row["routes"]
        if not isinstance(routes, list):
            raise SchemaError("NullMultiRouteSet.routes must be an array")
        return cls(
            case_id=identifier(row["case_id"], "NullMultiRouteSet.case_id"),
            split=split_name(row["split"], "NullMultiRouteSet.split"),
            product_smiles=text(
                row["product_smiles"], "NullMultiRouteSet.product_smiles"
            ),
            candidate_pool_id=identifier(
                row["candidate_pool_id"], "NullMultiRouteSet.candidate_pool_id"
            ),
            routes=tuple(CandidateRoute.from_dict(item) for item in routes),
            metadata=json_object(row.get("metadata", {}), "NullMultiRouteSet.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "case_id": self.case_id,
            "split": self.split,
            "product_smiles": self.product_smiles,
            "candidate_pool_id": self.candidate_pool_id,
            "routes": [route.to_dict() for route in self.routes],
            "metadata": clone_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RouteEvidence(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.route_evidence.v1"

    case_id: str
    candidate_pool_id: str
    route_id: str
    structure_evidence: dict[str, Any]
    metric_values: dict[str, float]
    availability: dict[str, bool]
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.case_id, "RouteEvidence.case_id")
        identifier(self.candidate_pool_id, "RouteEvidence.candidate_pool_id")
        identifier(self.route_id, "RouteEvidence.route_id")
        json_object(self.structure_evidence, "RouteEvidence.structure_evidence")
        for key, value in self.metric_values.items():
            identifier(key, "RouteEvidence.metric_values.key")
            number(value, f"RouteEvidence.metric_values[{key!r}]")
        for key, value in self.availability.items():
            identifier(key, "RouteEvidence.availability.key")
            boolean(value, f"RouteEvidence.availability[{key!r}]")
        json_object(self.provenance, "RouteEvidence.provenance")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RouteEvidence:
        row = strict_object(
            value,
            model="RouteEvidence",
            allowed={
                "schema_version",
                "case_id",
                "candidate_pool_id",
                "route_id",
                "structure_evidence",
                "metric_values",
                "availability",
                "provenance",
            },
            required={
                "schema_version",
                "case_id",
                "candidate_pool_id",
                "route_id",
                "structure_evidence",
                "metric_values",
                "availability",
            },
        )
        schema(row, cls.SCHEMA_VERSION, "RouteEvidence")
        raw_metrics = row["metric_values"]
        raw_availability = row["availability"]
        if not isinstance(raw_metrics, Mapping) or not isinstance(
            raw_availability, Mapping
        ):
            raise SchemaError("RouteEvidence metric_values/availability must be objects")
        return cls(
            case_id=identifier(row["case_id"], "RouteEvidence.case_id"),
            candidate_pool_id=identifier(
                row["candidate_pool_id"], "RouteEvidence.candidate_pool_id"
            ),
            route_id=identifier(row["route_id"], "RouteEvidence.route_id"),
            structure_evidence=json_object(
                row["structure_evidence"], "RouteEvidence.structure_evidence"
            ),
            metric_values={
                identifier(key, "RouteEvidence.metric_values.key"): number(
                    item, f"RouteEvidence.metric_values[{key!r}]"
                )
                for key, item in raw_metrics.items()
            },
            availability={
                identifier(key, "RouteEvidence.availability.key"): boolean(
                    item, f"RouteEvidence.availability[{key!r}]"
                )
                for key, item in raw_availability.items()
            },
            provenance=json_object(
                row.get("provenance", {}), "RouteEvidence.provenance"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "case_id": self.case_id,
            "candidate_pool_id": self.candidate_pool_id,
            "route_id": self.route_id,
            "structure_evidence": clone_json(self.structure_evidence),
            "metric_values": dict(self.metric_values),
            "availability": dict(self.availability),
            "provenance": clone_json(self.provenance),
        }


@dataclass(frozen=True, slots=True)
class AnalysisBasis(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.analysis_basis.v1"
    BASIS_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "metric_anchored",
            "evidence_discovered",
            "llm_route_structure",
            "metric_registry",
        }
    )

    basis_type: str
    evidence_refs: tuple[str, ...]
    supporting_fields: tuple[str, ...]
    candidate_observations: dict[str, str]
    axis_selection_rationale: str
    labels_supported: bool
    review: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.basis_type not in self.BASIS_TYPES:
            raise SchemaError(
                f"AnalysisBasis.basis_type must be one of {sorted(self.BASIS_TYPES)}"
            )
        if not self.evidence_refs or not self.supporting_fields:
            raise SchemaError(
                "AnalysisBasis evidence_refs and supporting_fields must not be empty"
            )
        if not self.candidate_observations:
            raise SchemaError("AnalysisBasis.candidate_observations must not be empty")
        for route_id, observation in self.candidate_observations.items():
            identifier(route_id, "AnalysisBasis.candidate_observations.key")
            text(observation, f"AnalysisBasis.candidate_observations[{route_id!r}]")
        text(self.axis_selection_rationale, "AnalysisBasis.axis_selection_rationale")
        boolean(self.labels_supported, "AnalysisBasis.labels_supported")
        json_object(self.review, "AnalysisBasis.review")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AnalysisBasis:
        row = strict_object(
            value,
            model="AnalysisBasis",
            allowed={
                "schema_version",
                "basis_type",
                "evidence_refs",
                "supporting_fields",
                "candidate_observations",
                "axis_selection_rationale",
                "labels_supported",
                "review",
            },
            required={
                "schema_version",
                "basis_type",
                "evidence_refs",
                "supporting_fields",
                "candidate_observations",
                "axis_selection_rationale",
                "labels_supported",
            },
        )
        schema(row, cls.SCHEMA_VERSION, "AnalysisBasis")
        raw_observations = row["candidate_observations"]
        if not isinstance(raw_observations, Mapping):
            raise SchemaError("AnalysisBasis.candidate_observations must be an object")
        return cls(
            basis_type=text(row["basis_type"], "AnalysisBasis.basis_type"),
            evidence_refs=string_tuple(
                row["evidence_refs"], "AnalysisBasis.evidence_refs"
            ),
            supporting_fields=string_tuple(
                row["supporting_fields"], "AnalysisBasis.supporting_fields"
            ),
            candidate_observations={
                identifier(key, "AnalysisBasis.candidate_observations.key"): text(
                    item, f"AnalysisBasis.candidate_observations[{key!r}]"
                )
                for key, item in raw_observations.items()
            },
            axis_selection_rationale=text(
                row["axis_selection_rationale"],
                "AnalysisBasis.axis_selection_rationale",
            ),
            labels_supported=boolean(
                row["labels_supported"], "AnalysisBasis.labels_supported"
            ),
            review=json_object(row.get("review", {}), "AnalysisBasis.review"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "basis_type": self.basis_type,
            "evidence_refs": list(self.evidence_refs),
            "supporting_fields": list(self.supporting_fields),
            "candidate_observations": dict(self.candidate_observations),
            "axis_selection_rationale": self.axis_selection_rationale,
            "labels_supported": self.labels_supported,
            "review": clone_json(self.review),
        }


@dataclass(frozen=True, slots=True)
class CandidatePreference(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.candidate_preference.v1"

    route_id: str
    eligible: bool
    label: int
    raw_value: float | None
    normalized_value: float | None
    reward: float | None
    teacher_probability: float

    def __post_init__(self) -> None:
        identifier(self.route_id, "CandidatePreference.route_id")
        boolean(self.eligible, "CandidatePreference.eligible")
        if self.label not in {-1, 0, 1}:
            raise SchemaError("CandidatePreference.label must be -1, 0, or +1")
        optional_number(self.raw_value, "CandidatePreference.raw_value")
        optional_number(self.normalized_value, "CandidatePreference.normalized_value")
        optional_number(self.reward, "CandidatePreference.reward")
        probability = number(
            self.teacher_probability,
            "CandidatePreference.teacher_probability",
            nonnegative=True,
        )
        if probability > 1.0 + 1.0e-9:
            raise SchemaError("CandidatePreference.teacher_probability must be <= 1")
        if not self.eligible and (
            self.label != 0
            or self.raw_value is not None
            or self.normalized_value is not None
            or self.reward is not None
            or self.teacher_probability != 0.0
        ):
            raise SchemaError(
                "ineligible CandidatePreference must have neutral/empty supervision"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CandidatePreference:
        row = strict_object(
            value,
            model="CandidatePreference",
            allowed={
                "schema_version",
                "route_id",
                "eligible",
                "label",
                "raw_value",
                "normalized_value",
                "reward",
                "teacher_probability",
            },
            required={
                "schema_version",
                "route_id",
                "eligible",
                "label",
                "raw_value",
                "normalized_value",
                "reward",
                "teacher_probability",
            },
        )
        schema(row, cls.SCHEMA_VERSION, "CandidatePreference")
        label = row["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise SchemaError("CandidatePreference.label must be an integer")
        return cls(
            route_id=identifier(row["route_id"], "CandidatePreference.route_id"),
            eligible=boolean(row["eligible"], "CandidatePreference.eligible"),
            label=label,
            raw_value=optional_number(
                row["raw_value"], "CandidatePreference.raw_value"
            ),
            normalized_value=optional_number(
                row["normalized_value"], "CandidatePreference.normalized_value"
            ),
            reward=optional_number(row["reward"], "CandidatePreference.reward"),
            teacher_probability=number(
                row["teacher_probability"],
                "CandidatePreference.teacher_probability",
                nonnegative=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "route_id": self.route_id,
            "eligible": self.eligible,
            "label": self.label,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "reward": self.reward,
            "teacher_probability": self.teacher_probability,
        }


@dataclass(frozen=True, slots=True)
class TextVariant(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.text_variant.v1"
    RELATIONS: ClassVar[frozenset[str]] = frozenset({"canonical", "synonym"})
    USAGES: ClassVar[frozenset[str]] = frozenset({"all", "train", "heldout"})

    text_id: str
    text: str
    relation: str
    usage: str = "all"
    review: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier(self.text_id, "TextVariant.text_id")
        text(self.text, "TextVariant.text")
        if self.relation not in self.RELATIONS:
            raise SchemaError(
                f"TextVariant.relation must be one of {sorted(self.RELATIONS)}"
            )
        if self.usage not in self.USAGES:
            raise SchemaError(f"TextVariant.usage must be one of {sorted(self.USAGES)}")
        if self.relation == "canonical" and self.usage == "heldout":
            raise SchemaError("canonical text cannot be heldout-only")
        json_object(self.review, "TextVariant.review")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TextVariant:
        row = strict_object(
            value,
            model="TextVariant",
            allowed={"schema_version", "text_id", "text", "relation", "usage", "review"},
            required={"text_id", "text", "relation"},
        )
        if "schema_version" in row:
            schema(row, cls.SCHEMA_VERSION, "TextVariant")
        return cls(
            text_id=identifier(row["text_id"], "TextVariant.text_id"),
            text=text(row["text"], "TextVariant.text"),
            relation=text(row["relation"], "TextVariant.relation"),
            usage=text(row.get("usage", "all"), "TextVariant.usage"),
            review=json_object(row.get("review", {}), "TextVariant.review"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "text_id": self.text_id,
            "text": self.text,
            "relation": self.relation,
            "usage": self.usage,
            "review": clone_json(self.review),
        }


@dataclass(frozen=True, slots=True)
class TextPool(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.text_pool.v1"

    text_pool_id: str
    branch: str
    axis_id: str
    polarity: int
    control_id: str
    opposite_control_id: str
    semantic_family_uid: str | None
    variants: tuple[TextVariant, ...]

    def __post_init__(self) -> None:
        identifier(self.text_pool_id, "TextPool.text_pool_id")
        if self.branch not in {"structure", "metric"}:
            raise SchemaError("TextPool.branch must be 'structure' or 'metric'")
        identifier(self.axis_id, "TextPool.axis_id")
        polarity(self.polarity, "TextPool.polarity")
        identifier(self.control_id, "TextPool.control_id")
        identifier(self.opposite_control_id, "TextPool.opposite_control_id")
        if self.control_id == self.opposite_control_id:
            raise SchemaError("TextPool opposite control must differ from control_id")
        if self.semantic_family_uid is not None:
            identifier(self.semantic_family_uid, "TextPool.semantic_family_uid")
        if not self.variants:
            raise SchemaError("TextPool.variants must not be empty")
        ids = tuple(item.text_id for item in self.variants)
        if len(ids) != len(set(ids)):
            raise SchemaError("TextPool.variants contains duplicate text_id values")
        canonical = [item for item in self.variants if item.relation == "canonical"]
        if len(canonical) != 1:
            raise SchemaError("TextPool must contain exactly one canonical text")
        if self.branch == "metric" and self.semantic_family_uid is not None:
            raise SchemaError("global metric TextPool cannot have semantic_family_uid")
        if self.branch == "structure" and self.semantic_family_uid is None:
            raise SchemaError("structure TextPool requires semantic_family_uid")

    @property
    def canonical(self) -> TextVariant:
        return next(item for item in self.variants if item.relation == "canonical")

    @property
    def training_variants(self) -> tuple[TextVariant, ...]:
        return tuple(item for item in self.variants if item.usage in {"all", "train"})

    @property
    def heldout_variants(self) -> tuple[TextVariant, ...]:
        return tuple(item for item in self.variants if item.usage == "heldout")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TextPool:
        row = strict_object(
            value,
            model="TextPool",
            allowed={
                "schema_version",
                "text_pool_id",
                "branch",
                "axis_id",
                "polarity",
                "control_id",
                "opposite_control_id",
                "semantic_family_uid",
                "variants",
            },
            required={
                "schema_version",
                "text_pool_id",
                "branch",
                "axis_id",
                "polarity",
                "control_id",
                "opposite_control_id",
                "variants",
            },
        )
        schema(row, cls.SCHEMA_VERSION, "TextPool")
        raw_variants = row["variants"]
        if not isinstance(raw_variants, list):
            raise SchemaError("TextPool.variants must be an array")
        return cls(
            text_pool_id=identifier(row["text_pool_id"], "TextPool.text_pool_id"),
            branch=text(row["branch"], "TextPool.branch"),
            axis_id=identifier(row["axis_id"], "TextPool.axis_id"),
            polarity=polarity(row["polarity"], "TextPool.polarity"),
            control_id=identifier(row["control_id"], "TextPool.control_id"),
            opposite_control_id=identifier(
                row["opposite_control_id"], "TextPool.opposite_control_id"
            ),
            semantic_family_uid=optional_text(
                row.get("semantic_family_uid"), "TextPool.semantic_family_uid"
            ),
            variants=tuple(TextVariant.from_dict(item) for item in raw_variants),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "text_pool_id": self.text_pool_id,
            "branch": self.branch,
            "axis_id": self.axis_id,
            "polarity": self.polarity,
            "control_id": self.control_id,
            "opposite_control_id": self.opposite_control_id,
            "semantic_family_uid": self.semantic_family_uid,
            "variants": [item.to_dict() for item in self.variants],
        }


@dataclass(frozen=True, slots=True)
class SemanticGroup(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.semantic_group.v1"

    semantic_group_uid: str
    semantic_family_uid: str
    case_id: str
    candidate_pool_id: str
    split: str
    branch: str
    axis_id: str
    polarity: int
    control_id: str
    text_pool_id: str
    opposite_control_id: str
    opposite_text_pool_id: str
    analysis_basis: AnalysisBasis
    candidate_preferences: tuple[CandidatePreference, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("semantic_group_uid", self.semantic_group_uid),
            ("semantic_family_uid", self.semantic_family_uid),
            ("case_id", self.case_id),
            ("candidate_pool_id", self.candidate_pool_id),
            ("axis_id", self.axis_id),
            ("control_id", self.control_id),
            ("text_pool_id", self.text_pool_id),
            ("opposite_control_id", self.opposite_control_id),
            ("opposite_text_pool_id", self.opposite_text_pool_id),
        ):
            identifier(value, f"SemanticGroup.{name}")
        split_name(self.split, "SemanticGroup.split")
        if self.branch not in {"structure", "metric"}:
            raise SchemaError("SemanticGroup.branch must be 'structure' or 'metric'")
        polarity(self.polarity, "SemanticGroup.polarity")
        if self.control_id == self.opposite_control_id:
            raise SchemaError("SemanticGroup control and opposite control must differ")
        if len(self.candidate_preferences) < 2:
            raise SchemaError("SemanticGroup requires at least two candidate preferences")
        route_ids = tuple(item.route_id for item in self.candidate_preferences)
        if len(route_ids) != len(set(route_ids)):
            raise SchemaError("SemanticGroup contains duplicate route preferences")
        eligible = [item for item in self.candidate_preferences if item.eligible]
        if not any(item.label == 1 for item in eligible) or not any(
            item.label == -1 for item in eligible
        ):
            raise SchemaError(
                "SemanticGroup must contain eligible positive and negative candidates"
            )
        total = sum(item.teacher_probability for item in eligible)
        if abs(total - 1.0) > 1.0e-6:
            raise SchemaError(
                "SemanticGroup eligible teacher probabilities must sum to one"
            )
        if not self.analysis_basis.labels_supported:
            raise SchemaError("SemanticGroup analysis basis must support its labels")
        if self.branch == "structure" and self.analysis_basis.basis_type not in {
            "metric_anchored",
            "evidence_discovered",
            "llm_route_structure",
        }:
            raise SchemaError("structure group has an invalid analysis basis type")
        if self.branch == "metric" and self.analysis_basis.basis_type != "metric_registry":
            raise SchemaError("metric group requires metric_registry analysis basis")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SemanticGroup:
        required = {
            "schema_version",
            "semantic_group_uid",
            "semantic_family_uid",
            "case_id",
            "candidate_pool_id",
            "split",
            "branch",
            "axis_id",
            "polarity",
            "control_id",
            "text_pool_id",
            "opposite_control_id",
            "opposite_text_pool_id",
            "analysis_basis",
            "candidate_preferences",
        }
        row = strict_object(
            value,
            model="SemanticGroup",
            allowed=required,
            required=required,
        )
        schema(row, cls.SCHEMA_VERSION, "SemanticGroup")
        raw_preferences = row["candidate_preferences"]
        if not isinstance(raw_preferences, list):
            raise SchemaError("SemanticGroup.candidate_preferences must be an array")
        return cls(
            semantic_group_uid=identifier(
                row["semantic_group_uid"], "SemanticGroup.semantic_group_uid"
            ),
            semantic_family_uid=identifier(
                row["semantic_family_uid"], "SemanticGroup.semantic_family_uid"
            ),
            case_id=identifier(row["case_id"], "SemanticGroup.case_id"),
            candidate_pool_id=identifier(
                row["candidate_pool_id"], "SemanticGroup.candidate_pool_id"
            ),
            split=split_name(row["split"], "SemanticGroup.split"),
            branch=text(row["branch"], "SemanticGroup.branch"),
            axis_id=identifier(row["axis_id"], "SemanticGroup.axis_id"),
            polarity=polarity(row["polarity"], "SemanticGroup.polarity"),
            control_id=identifier(row["control_id"], "SemanticGroup.control_id"),
            text_pool_id=identifier(
                row["text_pool_id"], "SemanticGroup.text_pool_id"
            ),
            opposite_control_id=identifier(
                row["opposite_control_id"], "SemanticGroup.opposite_control_id"
            ),
            opposite_text_pool_id=identifier(
                row["opposite_text_pool_id"],
                "SemanticGroup.opposite_text_pool_id",
            ),
            analysis_basis=AnalysisBasis.from_dict(row["analysis_basis"]),
            candidate_preferences=tuple(
                CandidatePreference.from_dict(item) for item in raw_preferences
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "semantic_group_uid": self.semantic_group_uid,
            "semantic_family_uid": self.semantic_family_uid,
            "case_id": self.case_id,
            "candidate_pool_id": self.candidate_pool_id,
            "split": self.split,
            "branch": self.branch,
            "axis_id": self.axis_id,
            "polarity": self.polarity,
            "control_id": self.control_id,
            "text_pool_id": self.text_pool_id,
            "opposite_control_id": self.opposite_control_id,
            "opposite_text_pool_id": self.opposite_text_pool_id,
            "analysis_basis": self.analysis_basis.to_dict(),
            "candidate_preferences": [
                item.to_dict() for item in self.candidate_preferences
            ],
        }


@dataclass(frozen=True, slots=True)
class ShuffleDonorPool(JsonModel):
    """Compact split-local index entry used by runtime shuffle sampling."""

    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.shuffle_donor_pool.v1"

    text_pool_id: str
    branch: str
    axis_id: str
    polarity: int
    control_id: str
    case_id: str | None
    canonical_text_id: str
    training_text_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        identifier(self.text_pool_id, "ShuffleDonorPool.text_pool_id")
        if self.branch not in {"structure", "metric"}:
            raise SchemaError("ShuffleDonorPool.branch is invalid")
        identifier(self.axis_id, "ShuffleDonorPool.axis_id")
        polarity(self.polarity, "ShuffleDonorPool.polarity")
        identifier(self.control_id, "ShuffleDonorPool.control_id")
        identifier(self.canonical_text_id, "ShuffleDonorPool.canonical_text_id")
        if not self.training_text_ids:
            raise SchemaError("ShuffleDonorPool.training_text_ids must not be empty")
        for text_id in self.training_text_ids:
            identifier(text_id, "ShuffleDonorPool.training_text_ids.item")
        if len(self.training_text_ids) != len(set(self.training_text_ids)):
            raise SchemaError("ShuffleDonorPool.training_text_ids contains duplicates")
        if self.canonical_text_id not in self.training_text_ids:
            raise SchemaError("ShuffleDonorPool canonical text must be trainable")
        if self.branch == "metric" and self.case_id is not None:
            raise SchemaError("global metric donor pools cannot have a case_id")
        if self.branch == "structure" and self.case_id is None:
            raise SchemaError("structure donor pools require a case_id")
        if self.case_id is not None:
            identifier(self.case_id, "ShuffleDonorPool.case_id")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ShuffleDonorPool:
        required = {
            "schema_version",
            "text_pool_id",
            "branch",
            "axis_id",
            "polarity",
            "control_id",
            "case_id",
            "canonical_text_id",
            "training_text_ids",
        }
        row = strict_object(
            value,
            model="ShuffleDonorPool",
            allowed=required,
            required=required,
        )
        schema(row, cls.SCHEMA_VERSION, "ShuffleDonorPool")
        raw_training_ids = row["training_text_ids"]
        if not isinstance(raw_training_ids, list):
            raise SchemaError("ShuffleDonorPool.training_text_ids must be an array")
        return cls(
            text_pool_id=identifier(
                row["text_pool_id"], "ShuffleDonorPool.text_pool_id"
            ),
            branch=text(row["branch"], "ShuffleDonorPool.branch"),
            axis_id=identifier(row["axis_id"], "ShuffleDonorPool.axis_id"),
            polarity=polarity(row["polarity"], "ShuffleDonorPool.polarity"),
            control_id=identifier(
                row["control_id"], "ShuffleDonorPool.control_id"
            ),
            case_id=optional_text(row["case_id"], "ShuffleDonorPool.case_id"),
            canonical_text_id=identifier(
                row["canonical_text_id"], "ShuffleDonorPool.canonical_text_id"
            ),
            training_text_ids=id_tuple(
                raw_training_ids, "ShuffleDonorPool.training_text_ids"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "text_pool_id": self.text_pool_id,
            "branch": self.branch,
            "axis_id": self.axis_id,
            "polarity": self.polarity,
            "control_id": self.control_id,
            "case_id": self.case_id,
            "canonical_text_id": self.canonical_text_id,
            "training_text_ids": list(self.training_text_ids),
        }


@dataclass(frozen=True, slots=True)
class EvaluationMapping(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.evaluation_mapping.v2"

    semantic_group_uid: str
    semantic_group_ordinal: int
    canonical_true_text_id: str
    seen_synonym_true_text_id: str | None
    heldout_synonym_true_text_id: str | None
    different_axis_shuffle_draw_index: int
    same_axis_antonym_text_id: str
    include_null_guidance: bool = True

    def __post_init__(self) -> None:
        identifier(self.semantic_group_uid, "EvaluationMapping.semantic_group_uid")
        if (
            isinstance(self.semantic_group_ordinal, bool)
            or not isinstance(self.semantic_group_ordinal, int)
            or self.semantic_group_ordinal < 0
        ):
            raise SchemaError(
                "EvaluationMapping.semantic_group_ordinal must be non-negative"
            )
        identifier(
            self.canonical_true_text_id,
            "EvaluationMapping.canonical_true_text_id",
        )
        if self.seen_synonym_true_text_id is not None:
            identifier(
                self.seen_synonym_true_text_id,
                "EvaluationMapping.seen_synonym_true_text_id",
            )
        if self.heldout_synonym_true_text_id is not None:
            identifier(
                self.heldout_synonym_true_text_id,
                "EvaluationMapping.heldout_synonym_true_text_id",
            )
        if (
            isinstance(self.different_axis_shuffle_draw_index, bool)
            or not isinstance(self.different_axis_shuffle_draw_index, int)
            or self.different_axis_shuffle_draw_index < 0
        ):
            raise SchemaError(
                "EvaluationMapping.different_axis_shuffle_draw_index must be "
                "non-negative"
            )
        identifier(
            self.same_axis_antonym_text_id,
            "EvaluationMapping.same_axis_antonym_text_id",
        )
        boolean(self.include_null_guidance, "EvaluationMapping.include_null_guidance")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> EvaluationMapping:
        required = {
            "schema_version",
            "semantic_group_uid",
            "semantic_group_ordinal",
            "canonical_true_text_id",
            "seen_synonym_true_text_id",
            "heldout_synonym_true_text_id",
            "different_axis_shuffle_draw_index",
            "same_axis_antonym_text_id",
            "include_null_guidance",
        }
        row = strict_object(value, model="EvaluationMapping", allowed=required, required=required)
        schema(row, cls.SCHEMA_VERSION, "EvaluationMapping")
        return cls(
            semantic_group_uid=identifier(
                row["semantic_group_uid"], "EvaluationMapping.semantic_group_uid"
            ),
            semantic_group_ordinal=row["semantic_group_ordinal"],
            canonical_true_text_id=identifier(
                row["canonical_true_text_id"],
                "EvaluationMapping.canonical_true_text_id",
            ),
            seen_synonym_true_text_id=optional_text(
                row["seen_synonym_true_text_id"],
                "EvaluationMapping.seen_synonym_true_text_id",
            ),
            heldout_synonym_true_text_id=optional_text(
                row["heldout_synonym_true_text_id"],
                "EvaluationMapping.heldout_synonym_true_text_id",
            ),
            different_axis_shuffle_draw_index=row[
                "different_axis_shuffle_draw_index"
            ],
            same_axis_antonym_text_id=identifier(
                row["same_axis_antonym_text_id"],
                "EvaluationMapping.same_axis_antonym_text_id",
            ),
            include_null_guidance=boolean(
                row["include_null_guidance"],
                "EvaluationMapping.include_null_guidance",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "semantic_group_uid": self.semantic_group_uid,
            "semantic_group_ordinal": self.semantic_group_ordinal,
            "canonical_true_text_id": self.canonical_true_text_id,
            "seen_synonym_true_text_id": self.seen_synonym_true_text_id,
            "heldout_synonym_true_text_id": self.heldout_synonym_true_text_id,
            "different_axis_shuffle_draw_index": (
                self.different_axis_shuffle_draw_index
            ),
            "same_axis_antonym_text_id": self.same_axis_antonym_text_id,
            "include_null_guidance": self.include_null_guidance,
        }


@dataclass(frozen=True, slots=True)
class GuidedMultiRouteSet(JsonModel):
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.guided_multi_route_set.v2"

    case: NullMultiRouteSet
    evidence: tuple[RouteEvidence, ...]
    semantic_groups: tuple[SemanticGroup, ...]

    def __post_init__(self) -> None:
        evidence_ids = tuple(item.route_id for item in self.evidence)
        route_ids = tuple(item.route_id for item in self.case.routes)
        if set(evidence_ids) != set(route_ids) or len(evidence_ids) != len(route_ids):
            raise SchemaError("GuidedMultiRouteSet evidence must cover every route once")
        if any(item.case_id != self.case.case_id for item in self.evidence):
            raise SchemaError("GuidedMultiRouteSet evidence case_id differs")
        if any(item.case_id != self.case.case_id for item in self.semantic_groups):
            raise SchemaError("GuidedMultiRouteSet group case_id differs")
        group_ids = tuple(item.semantic_group_uid for item in self.semantic_groups)
        if len(group_ids) != len(set(group_ids)):
            raise SchemaError("GuidedMultiRouteSet contains duplicate semantic groups")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GuidedMultiRouteSet:
        required = {
            "schema_version",
            "case",
            "evidence",
            "semantic_groups",
        }
        row = strict_object(value, model="GuidedMultiRouteSet", allowed=required, required=required)
        schema(row, cls.SCHEMA_VERSION, "GuidedMultiRouteSet")
        for name in ("evidence", "semantic_groups"):
            if not isinstance(row[name], list):
                raise SchemaError(f"GuidedMultiRouteSet.{name} must be an array")
        return cls(
            case=NullMultiRouteSet.from_dict(row["case"]),
            evidence=tuple(RouteEvidence.from_dict(item) for item in row["evidence"]),
            semantic_groups=tuple(
                SemanticGroup.from_dict(item) for item in row["semantic_groups"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "case": self.case.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "semantic_groups": [item.to_dict() for item in self.semantic_groups],
        }


def ensure_unique_text_ids(pools: Sequence[TextPool]) -> None:
    owners: dict[str, str] = {}
    for pool in pools:
        for variant in pool.variants:
            previous = owners.setdefault(variant.text_id, pool.text_pool_id)
            if previous != pool.text_pool_id:
                raise SchemaError(
                    f"text_id {variant.text_id!r} belongs to multiple pools: "
                    f"{previous!r} and {pool.text_pool_id!r}"
                )
