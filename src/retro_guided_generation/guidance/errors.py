"""Errors raised by the canonical guidance subsystem."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class GuidanceError(RuntimeError):
    """Base class for guidance build failures."""


class SchemaError(GuidanceError, ValueError):
    """A JSON-shaped record violates its declared schema."""


class ConfigurationError(GuidanceError, ValueError):
    """A guidance configuration is incomplete or inconsistent."""


class QualityError(GuidanceError):
    """A dataset failed a required semantic quality gate."""


class ReviewSemanticRejection(QualityError):
    """An independent review rejected an otherwise schema-valid draft.

    ``feedback`` is intentionally structured and JSON-shaped so the shared
    work-pool runtime can create a bounded replacement draft task without
    treating a reviewer rejection as a transport or schema retry.
    """

    def __init__(self, message: str, *, feedback: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.feedback = dict(feedback)
