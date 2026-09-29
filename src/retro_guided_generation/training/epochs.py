"""Exact epoch budgets shared by bounded and recoverable training runners.

An epoch is defined in global *group exposures*.  Dataset adapters publish the
number of deterministic training groups in one sampler traversal, and the
runner converts that count to optimizer updates only after the effective global
batch is known.  Fractions such as ``"1/3"`` remain exact all the way to the
checkpoint boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias


EpochValue: TypeAlias = Fraction | int | float | str


def parse_epoch_value(
    value: EpochValue,
    name: str,
    *,
    allow_zero: bool = False,
) -> Fraction:
    """Normalize one user-facing epoch value without binary-float drift."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an epoch number or exact fraction")
    try:
        if isinstance(value, Fraction):
            result = value
        elif isinstance(value, int):
            result = Fraction(value, 1)
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError
            result = Fraction(str(value))
        elif isinstance(value, str) and value.strip():
            result = Fraction(value.strip())
        else:
            raise ValueError
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"{name} must be an epoch number or exact fraction such as '1/3'"
        ) from error
    minimum = Fraction(0 if allow_zero else 1, 1)
    if (result < minimum) if allow_zero else (result <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def epoch_value_json(value: Fraction) -> int | str:
    """Return an exact, JSON/YAML-friendly epoch representation."""

    if not isinstance(value, Fraction):
        raise TypeError("value must be fractions.Fraction")
    return value.numerator if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def epoch_value_slug(value: Fraction) -> str:
    """Return a stable exact filename component for an epoch coordinate."""

    if not isinstance(value, Fraction) or value <= 0:
        raise ValueError("epoch checkpoint coordinate must be a positive Fraction")
    if value.denominator == 1:
        return f"{value.numerator:06d}"
    return f"{value.numerator:06d}of{value.denominator:06d}"


@dataclass(frozen=True, slots=True)
class EpochBudget:
    """A finite user-facing training and candidate-checkpoint budget."""

    max_epochs: EpochValue
    candidate_epochs: tuple[EpochValue, ...]
    first_epoch: int = 0

    def __post_init__(self) -> None:
        maximum = parse_epoch_value(self.max_epochs, "max_epochs")
        if not isinstance(self.candidate_epochs, tuple) or not self.candidate_epochs:
            raise ValueError("candidate_epochs must be a non-empty tuple")
        candidates = tuple(
            parse_epoch_value(value, f"candidate_epochs[{index}]")
            for index, value in enumerate(self.candidate_epochs)
        )
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("candidate_epochs must be sorted and unique")
        if candidates[-1] != maximum:
            raise ValueError("candidate_epochs must include the terminal max_epochs")
        if candidates[0] <= 0 or candidates[-1] > maximum:
            raise ValueError("candidate_epochs must be inside (0, max_epochs]")
        if (
            isinstance(self.first_epoch, bool)
            or not isinstance(self.first_epoch, int)
            or self.first_epoch < 0
        ):
            raise ValueError("first_epoch must be a non-negative integer")
        object.__setattr__(self, "max_epochs", maximum)
        object.__setattr__(self, "candidate_epochs", candidates)

    def as_dict(self) -> dict[str, object]:
        return {
            "max_epochs": epoch_value_json(self.max_epochs),
            "candidate_epochs": [epoch_value_json(value) for value in self.candidate_epochs],
            "first_epoch": self.first_epoch,
        }


@dataclass(frozen=True, slots=True)
class EpochTrainingPlan:
    """Dataset- and batch-resolved optimizer boundaries for an epoch budget."""

    budget: EpochBudget
    epoch_size_groups: int
    effective_global_batch_groups: int
    optimizer_updates_per_epoch: int
    max_optimizer_updates: int
    candidate_optimizer_updates: tuple[int, ...]
    recovery_interval_updates: int
    heartbeat_interval_updates: int
    padded_group_exposures_per_epoch: int

    @classmethod
    def resolve(
        cls,
        *,
        budget: EpochBudget,
        epoch_size_groups: int,
        effective_global_batch_groups: int,
        recovery_interval_epochs: EpochValue,
        heartbeat_interval_epochs: EpochValue,
    ) -> "EpochTrainingPlan":
        if not isinstance(budget, EpochBudget):
            raise TypeError("budget must be EpochBudget")
        for name, value in (
            ("epoch_size_groups", epoch_size_groups),
            ("effective_global_batch_groups", effective_global_batch_groups),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        recovery = parse_epoch_value(recovery_interval_epochs, "recovery_interval_epochs")
        heartbeat = parse_epoch_value(heartbeat_interval_epochs, "heartbeat_interval_epochs")
        updates_per_epoch = math.ceil(epoch_size_groups / effective_global_batch_groups)

        def update_boundary(value: Fraction) -> int:
            return math.ceil(value * updates_per_epoch)

        candidates = tuple(update_boundary(value) for value in budget.candidate_epochs)
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError(
                "candidate_epochs collapse onto the same optimizer update for this "
                "dataset/batch; use wider epoch spacing"
            )
        maximum = update_boundary(budget.max_epochs)
        if candidates[-1] != maximum:
            raise AssertionError("terminal epoch candidate did not resolve to max updates")
        return cls(
            budget=budget,
            epoch_size_groups=epoch_size_groups,
            effective_global_batch_groups=effective_global_batch_groups,
            optimizer_updates_per_epoch=updates_per_epoch,
            max_optimizer_updates=maximum,
            candidate_optimizer_updates=candidates,
            recovery_interval_updates=max(1, math.ceil(recovery * updates_per_epoch)),
            heartbeat_interval_updates=max(1, math.ceil(heartbeat * updates_per_epoch)),
            padded_group_exposures_per_epoch=(
                updates_per_epoch * effective_global_batch_groups - epoch_size_groups
            ),
        )

    def completed_epochs(self, completed_optimizer_updates: int) -> Fraction:
        if (
            isinstance(completed_optimizer_updates, bool)
            or not isinstance(completed_optimizer_updates, int)
            or completed_optimizer_updates < 0
        ):
            raise ValueError("completed_optimizer_updates must be a non-negative integer")
        return Fraction(completed_optimizer_updates, self.optimizer_updates_per_epoch)

    def sampler_epoch(self, completed_optimizer_updates: int) -> int:
        return self.budget.first_epoch + (
            completed_optimizer_updates // self.optimizer_updates_per_epoch
        )

    def is_epoch_boundary(self, completed_optimizer_updates: int) -> bool:
        return (
            completed_optimizer_updates > 0
            and completed_optimizer_updates % self.optimizer_updates_per_epoch == 0
        )

    def candidate_epoch(self, completed_optimizer_updates: int) -> Fraction | None:
        try:
            index = self.candidate_optimizer_updates.index(completed_optimizer_updates)
        except ValueError:
            return None
        return self.budget.candidate_epochs[index]

    def as_dict(self) -> dict[str, object]:
        return {
            **self.budget.as_dict(),
            "epoch_size_groups": self.epoch_size_groups,
            "effective_global_batch_groups": self.effective_global_batch_groups,
            "optimizer_updates_per_epoch": self.optimizer_updates_per_epoch,
            "max_optimizer_updates": self.max_optimizer_updates,
            "candidate_optimizer_updates": list(self.candidate_optimizer_updates),
            "recovery_interval_updates": self.recovery_interval_updates,
            "heartbeat_interval_updates": self.heartbeat_interval_updates,
            "group_exposures_per_epoch": (
                self.optimizer_updates_per_epoch * self.effective_global_batch_groups
            ),
            "padded_group_exposures_per_epoch": self.padded_group_exposures_per_epoch,
        }


__all__ = [
    "EpochBudget",
    "EpochTrainingPlan",
    "EpochValue",
    "epoch_value_json",
    "epoch_value_slug",
    "parse_epoch_value",
]
