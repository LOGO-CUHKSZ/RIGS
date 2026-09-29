"""Global metric-axis and signed-control registry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from retro_guided_generation.guidance._validation import (
    identifier,
    number,
    polarity,
    schema,
    strict_object,
    text,
)
from retro_guided_generation.guidance.errors import SchemaError


def signed_control_id(axis_id: str, direction: int) -> str:
    axis = identifier(axis_id, "axis_id")
    sign = polarity(direction)
    return f"{axis}:{sign:+d}"


@dataclass(frozen=True, slots=True)
class MetricAxisSpec:
    SCHEMA_VERSION: ClassVar[str] = "guidance_v3.metric_axis.v1"
    NORMALIZATIONS: ClassVar[frozenset[str]] = frozenset(
        {"pool_minmax", "pool_robust_zscore", "pool_zscore"}
    )

    axis_id: str
    value_key: str
    display_name: str
    value_type: str
    default_polarity: int
    allowed_polarities: tuple[int, ...]
    normalization: str
    teacher_temperature: float
    minimum_valid_candidates: int
    minimum_effect_size: float

    def __post_init__(self) -> None:
        identifier(self.axis_id, "MetricAxisSpec.axis_id")
        identifier(self.value_key, "MetricAxisSpec.value_key")
        text(self.display_name, "MetricAxisSpec.display_name")
        if self.value_type != "continuous":
            raise SchemaError("MetricAxisSpec.value_type currently must be 'continuous'")
        polarity(self.default_polarity, "MetricAxisSpec.default_polarity")
        if set(self.allowed_polarities) != {-1, 1}:
            raise SchemaError(
                "Guidance V3 requires both -1 and +1 metric polarities"
            )
        if len(self.allowed_polarities) != len(set(self.allowed_polarities)):
            raise SchemaError("MetricAxisSpec.allowed_polarities contains duplicates")
        if self.default_polarity not in self.allowed_polarities:
            raise SchemaError("default polarity must be allowed")
        if self.normalization not in self.NORMALIZATIONS:
            raise SchemaError(
                f"MetricAxisSpec.normalization must be one of {sorted(self.NORMALIZATIONS)}"
            )
        if self.teacher_temperature <= 0:
            raise SchemaError("MetricAxisSpec.teacher_temperature must be positive")
        if self.minimum_valid_candidates < 2:
            raise SchemaError("minimum_valid_candidates must be at least two")
        if self.minimum_effect_size < 0:
            raise SchemaError("minimum_effect_size must be non-negative")

    @property
    def metric_key(self) -> str:
        prefix = "metric_values."
        return self.value_key[len(prefix) :] if self.value_key.startswith(prefix) else self.value_key

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MetricAxisSpec:
        required = {
            "schema_version",
            "axis_id",
            "value_key",
            "display_name",
            "value_type",
            "default_polarity",
            "allowed_polarities",
            "normalization",
            "teacher_temperature",
            "minimum_valid_candidates",
            "minimum_effect_size",
        }
        row = strict_object(value, model="MetricAxisSpec", allowed=required, required=required)
        schema(row, cls.SCHEMA_VERSION, "MetricAxisSpec")
        raw_polarities = row["allowed_polarities"]
        if not isinstance(raw_polarities, list):
            raise SchemaError("MetricAxisSpec.allowed_polarities must be an array")
        minimum_valid = row["minimum_valid_candidates"]
        if isinstance(minimum_valid, bool) or not isinstance(minimum_valid, int):
            raise SchemaError("minimum_valid_candidates must be an integer")
        return cls(
            axis_id=identifier(row["axis_id"], "MetricAxisSpec.axis_id"),
            value_key=identifier(row["value_key"], "MetricAxisSpec.value_key"),
            display_name=text(row["display_name"], "MetricAxisSpec.display_name"),
            value_type=text(row["value_type"], "MetricAxisSpec.value_type"),
            default_polarity=polarity(
                row["default_polarity"], "MetricAxisSpec.default_polarity"
            ),
            allowed_polarities=tuple(
                polarity(item, f"MetricAxisSpec.allowed_polarities[{index}]")
                for index, item in enumerate(raw_polarities)
            ),
            normalization=text(row["normalization"], "MetricAxisSpec.normalization"),
            teacher_temperature=number(
                row["teacher_temperature"],
                "MetricAxisSpec.teacher_temperature",
                nonnegative=True,
            ),
            minimum_valid_candidates=minimum_valid,
            minimum_effect_size=number(
                row["minimum_effect_size"],
                "MetricAxisSpec.minimum_effect_size",
                nonnegative=True,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "axis_id": self.axis_id,
            "value_key": self.value_key,
            "display_name": self.display_name,
            "value_type": self.value_type,
            "default_polarity": self.default_polarity,
            "allowed_polarities": list(self.allowed_polarities),
            "normalization": self.normalization,
            "teacher_temperature": self.teacher_temperature,
            "minimum_valid_candidates": self.minimum_valid_candidates,
            "minimum_effect_size": self.minimum_effect_size,
        }


class MetricAxisRegistry:
    def __init__(self, axes: Sequence[MetricAxisSpec]) -> None:
        self._axes = tuple(axes)
        if not self._axes:
            raise SchemaError("metric axis registry must not be empty")
        axis_ids = tuple(axis.axis_id for axis in self._axes)
        value_keys = tuple(axis.metric_key for axis in self._axes)
        if len(axis_ids) != len(set(axis_ids)):
            raise SchemaError("metric axis registry contains duplicate axis_id values")
        if len(value_keys) != len(set(value_keys)):
            raise SchemaError("metric axis registry contains duplicate value keys")
        self._by_id = {axis.axis_id: axis for axis in self._axes}

    def __iter__(self):
        return iter(self._axes)

    def __len__(self) -> int:
        return len(self._axes)

    def axis(self, axis_id: str) -> MetricAxisSpec:
        try:
            return self._by_id[axis_id]
        except KeyError as error:
            raise KeyError(f"unknown metric axis {axis_id!r}") from error

    def to_rows(self) -> list[dict[str, Any]]:
        return [axis.to_dict() for axis in self._axes]
