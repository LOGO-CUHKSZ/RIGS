"""Small, dependency-light IO and contract helpers for evaluation."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from retro_guided_generation.guidance.io import JsonlStreamWriter, ensure_output_directory, iter_jsonl


class EvaluationError(ValueError):
    """Raised when an evaluation artifact or contract violates its protocol."""


INTEGRITY_POLICY = "no_hashes_calculated_or_validated"


def now_utc() -> str:
    """Return an explicit UTC timestamp for a run-owned artifact."""

    return dt.datetime.now(dt.timezone.utc).isoformat()


def require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{name} must be a non-empty string")
    return value.strip()


def require_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EvaluationError(f"{name} must be an integer >= {minimum}")
    return value


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def resolve_path(value: str | Path, *, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_document(path: str | Path) -> dict[str, Any]:
    """Read a JSON or YAML contract without writing any derived artifact."""

    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        if resolved.suffix.lower() in {".yaml", ".yml"}:
            import yaml

            value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        else:
            value = json.loads(resolved.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise EvaluationError(f"cannot read evaluation contract {resolved}: {error}") from error
    if not isinstance(value, Mapping):
        raise EvaluationError(f"evaluation contract {resolved} must be an object")
    return dict(value)


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    return [dict(row) for row in iter_jsonl(path)]


def write_new_text(path: str | Path, text: str) -> Path:
    """Create an output once; never silently replace an evaluation artifact."""

    output = Path(path).expanduser()
    ensure_output_directory(output.parent)
    try:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as error:
        raise EvaluationError(f"refusing to overwrite existing output: {output}") from error
    return output.resolve()


def write_new_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return write_new_text(
        path,
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
    )


def write_new_jsonl(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    output = Path(path).expanduser()
    try:
        with JsonlStreamWriter(output) as writer:
            for row in rows:
                writer.write(dict(row))
    except FileExistsError as error:
        raise EvaluationError(f"refusing to overwrite existing output: {output}") from error
    return output.resolve()


def require_complete(path: str | Path, *, description: str) -> dict[str, Any]:
    value = load_document(path)
    if value.get("status") not in {"complete", "passed"} and value.get("complete") is not True:
        raise EvaluationError(f"required {description} is not complete: {path}")
    return value
