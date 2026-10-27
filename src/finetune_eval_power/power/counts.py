"""The two by two table that a paired eval actually produces.

Written out as a type rather than passed around as four integers because the single
most common mistake in this area is reporting the accuracy difference while doing
the arithmetic as though the two measurements were independent. Holding b and c
separately makes the discordance visible at every call site.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finetune_eval_power.errors import ToolError

POINTS = 100.0


@dataclass(frozen=True)
class PairedCounts:
    """Counts over one case set, scored by the same suite before and after.

    - `both_right`: correct before and after
    - `only_before`: correct before, wrong after. These are the regressions.
    - `only_after`: wrong before, correct after
    - `both_wrong`: wrong before and after
    """

    both_right: int
    only_before: int
    only_after: int
    both_wrong: int

    def __post_init__(self) -> None:
        for name in ("both_right", "only_before", "only_after", "both_wrong"):
            if getattr(self, name) < 0:
                raise ToolError(f"{name} cannot be negative")
        if self.total == 0:
            raise ToolError("a paired table over zero cases carries no information")

    @property
    def total(self) -> int:
        return self.both_right + self.only_before + self.only_after + self.both_wrong

    @property
    def discordant(self) -> int:
        """Cases that changed. Only these carry information about the difference."""
        return self.only_before + self.only_after

    @property
    def accuracy_before(self) -> float:
        return (self.both_right + self.only_before) / self.total

    @property
    def accuracy_after(self) -> float:
        return (self.both_right + self.only_after) / self.total

    @property
    def drop(self) -> float:
        """Accuracy lost, as a proportion. Positive means the model got worse."""
        return (self.only_before - self.only_after) / self.total

    @property
    def drop_points(self) -> float:
        """The same quantity in percentage points, which is how it gets published."""
        return self.drop * POINTS

    @property
    def discordance(self) -> float:
        """Proportion of cases that changed verdict, written psi in the power formulas."""
        return self.discordant / self.total

    @property
    def resolution_floor_points(self) -> float:
        """The smallest non-zero drop this case set can represent, in points.

        Receipts rule 4 in one property: a measured drop of zero over n cases is not
        zero, it is below one in n.
        """
        return POINTS / self.total


def paired_counts(before: np.ndarray, after: np.ndarray) -> PairedCounts:
    """Build the table from two boolean correctness vectors over the same cases."""
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        raise ToolError(f"paired vectors must align: {before.shape} against {after.shape}")
    if before.ndim != 1:
        raise ToolError("paired vectors must be one dimensional")
    if before.dtype != bool or after.dtype != bool:
        raise ToolError("paired vectors must be boolean correctness, not scores or labels")
    return PairedCounts(
        both_right=int(np.sum(before & after)),
        only_before=int(np.sum(before & ~after)),
        only_after=int(np.sum(~before & after)),
        both_wrong=int(np.sum(~before & ~after)),
    )
