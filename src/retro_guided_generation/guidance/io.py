"""JSON/JSONL IO helpers with atomic file replacement."""

from __future__ import annotations

import json
import os
import tempfile
import gzip
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, IO, Protocol, TypeVar, runtime_checkable

from retro_guided_generation.guidance.errors import SchemaError
from retro_guided_generation.guidance.schema import JsonModel


T = TypeVar("T", bound=JsonModel)
OUTPUT_FILE_MODE = 0o644
OUTPUT_DIRECTORY_MODE = 0o775
SHARED_WORK_POOL_DIRECTORY_MODE = 0o777


def _directory_mode(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0o777:
        raise ValueError("directory mode must be an integer between 0 and 0o777")
    return value


def ensure_output_directory(
    path: str | Path,
    *,
    mode: int = OUTPUT_DIRECTORY_MODE,
    force_mode: bool = False,
) -> Path:
    """Create an output directory and, when requested, repair its mode.

    New ancestors are always chmod'd after creation so that the caller's umask
    cannot silently remove required write access. Existing directories retain
    their mode unless ``force_mode`` is explicitly requested; this prevents a
    normal JSON write from demoting an intentionally multi-UID shared pool.
    """

    directory_mode = _directory_mode(mode)
    destination = Path(path).expanduser()
    missing: list[Path] = []
    cursor = destination
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    destination.mkdir(
        mode=directory_mode,
        parents=True,
        exist_ok=True,
    )
    for directory in reversed(missing):
        try:
            directory.chmod(directory_mode)
        except PermissionError:
            if not os.access(directory, os.W_OK):
                raise
    if force_mode:
        try:
            destination.chmod(directory_mode)
        except PermissionError:
            if not os.access(destination, os.W_OK):
                raise
    return destination


@runtime_checkable
class JsonSerializable(Protocol):
    def to_dict(self) -> dict[str, Any]: ...


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve(strict=True)
    opener = gzip.open if resolved.suffix.lower() == ".gz" else open
    with opener(resolved, "rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise SchemaError(
                    f"{resolved}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, Mapping):
                raise SchemaError(f"{resolved}:{line_number}: row must be an object")
            yield dict(value)


def load_models(path: str | Path, model: type[T]) -> tuple[T, ...]:
    rows = []
    for line_number, row in enumerate(iter_jsonl(path), start=1):
        try:
            rows.append(model.from_dict(row))
        except (TypeError, ValueError) as error:
            raise SchemaError(f"{path}:{line_number}: {error}") from error
    return tuple(rows)


def iter_models(path: str | Path, model: type[T]) -> Iterator[T]:
    """Parse models lazily from JSONL while preserving source line context."""

    for line_number, row in enumerate(iter_jsonl(path), start=1):
        try:
            yield model.from_dict(row)
        except (TypeError, ValueError) as error:
            raise SchemaError(f"{path}:{line_number}: {error}") from error


class JsonlStreamWriter:
    """Write one staging JSONL artifact incrementally.

    The caller publishes the containing staging directory atomically, so this
    writer intentionally avoids a second full-size temporary copy.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._binary: IO[bytes] | None = None
        self._text: IO[str] | None = None
        self.row_count = 0

    def __enter__(self) -> JsonlStreamWriter:
        import io

        ensure_output_directory(self.path.parent)
        descriptor = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            OUTPUT_FILE_MODE,
        )
        self._binary = os.fdopen(descriptor, "wb")
        if self.path.suffix.lower() == ".gz":
            compressed = gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=self._binary,
                compresslevel=1,
                mtime=0,
            )
            self._text = io.TextIOWrapper(compressed, encoding="utf-8")
        else:
            self._text = io.TextIOWrapper(self._binary, encoding="utf-8")
        return self

    def write(self, row: JsonSerializable | Mapping[str, Any]) -> None:
        if self._text is None:
            raise RuntimeError("JSONL stream writer is not open")
        value = row.to_dict() if isinstance(row, JsonSerializable) else dict(row)
        self._text.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
        self._text.write("\n")
        self.row_count += 1

    def close(self) -> None:
        if self._text is None:
            return
        self._text.flush()
        self._text.close()
        self._text = None
        self._binary = None

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _atomic_text(path: Path, payload: str) -> None:
    ensure_output_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        os.fchmod(descriptor, OUTPUT_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_jsonl(
    path: str | Path,
    rows: Iterable[JsonSerializable | Mapping[str, Any]],
) -> None:
    output = Path(path)
    ensure_output_directory(output.parent)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        os.fchmod(descriptor, OUTPUT_FILE_MODE)
        raw_handle = os.fdopen(descriptor, "wb")
        if output.suffix.lower() == ".gz":
            binary_handle = gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                compresslevel=1,
                mtime=0,
            )
            import io

            handle = io.TextIOWrapper(binary_handle, encoding="utf-8")
        else:
            import io

            handle = io.TextIOWrapper(raw_handle, encoding="utf-8")
        with handle:
            for row in rows:
                value = row.to_dict() if isinstance(row, JsonSerializable) else dict(row)
                handle.write(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
                handle.write("\n")
            handle.flush()
        with open(temporary, "rb") as completed:
            os.fsync(completed.fileno())
        os.replace(temporary, output)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    _atomic_text(Path(path), payload)


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SchemaError(f"invalid JSON at {resolved}: {error}") from error
    if not isinstance(value, Mapping):
        raise SchemaError(f"JSON at {resolved} must contain an object")
    return dict(value)
