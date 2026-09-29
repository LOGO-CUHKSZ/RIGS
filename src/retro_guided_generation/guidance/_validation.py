"""Small strict-validation helpers with no third-party model dependency."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from retro_guided_generation.guidance.errors import SchemaError


ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+/-]*$")
SPLITS = frozenset({"train", "validation", "test"})


def strict_object(
    value: Any,
    *,
    model: str,
    allowed: set[str],
    required: set[str],
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{model} must be a JSON object")
    row = dict(value)
    unknown = set(row) - allowed
    missing = required - set(row)
    if unknown:
        raise SchemaError(
            f"{model} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise SchemaError(
            f"{model} is missing fields: {', '.join(sorted(missing))}"
        )
    return row


def schema(row: Mapping[str, Any], expected: str, model: str) -> None:
    actual = row.get("schema_version")
    if actual != expected:
        raise SchemaError(
            f"{model}.schema_version must be {expected!r}, got {actual!r}"
        )


def text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{path} must be a non-empty string")
    return value.strip()


def optional_text(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return text(value, path)


def identifier(value: Any, path: str) -> str:
    result = text(value, path)
    if not ID_PATTERN.fullmatch(result):
        raise SchemaError(
            f"{path} must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', ':', '+', '/', or '-'"
        )
    return result


def split_name(value: Any, path: str = "split") -> str:
    result = text(value, path).lower()
    if result == "val":
        result = "validation"
    if result not in SPLITS:
        raise SchemaError(f"{path} must be one of {sorted(SPLITS)}")
    return result


def number(value: Any, path: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SchemaError(f"{path} must be a finite number")
    if nonnegative and result < 0:
        raise SchemaError(f"{path} must be non-negative")
    return result


def optional_number(
    value: Any, path: str, *, nonnegative: bool = False
) -> float | None:
    if value is None:
        return None
    return number(value, path, nonnegative=nonnegative)


def integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaError(f"{path} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaError(f"{path} must be at least {minimum}")
    return value


def boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{path} must be a boolean")
    return value


def json_object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be a JSON object")
    try:
        return json.loads(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise SchemaError(f"{path} must contain only JSON-compatible values") from error


def string_tuple(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{path} must be an array of non-empty strings")
    result = tuple(text(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise SchemaError(f"{path} must not contain duplicates")
    return result


def id_tuple(value: Any, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{path} must be an array of identifiers")
    result = tuple(
        identifier(item, f"{path}[{index}]") for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise SchemaError(f"{path} must not contain duplicates")
    return result


def polarity(value: Any, path: str = "polarity") -> int:
    result = integer(value, path)
    if result not in {-1, 1}:
        raise SchemaError(f"{path} must be -1 or +1")
    return result


def clone_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise SchemaError("value must be JSON-compatible") from error

