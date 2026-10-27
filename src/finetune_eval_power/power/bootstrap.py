"""Interval for a measured per-slice drop, resampled over cases rather than assumed.

A per-slice drop is a difference of two proportions measured on the same cases, and
at the slice sizes here the normal interval misbehaves near zero, which is precisely
where most slices sit. So the interval is a paired percentile bootstrap: resample
cases with replacement, keeping each case's before and after verdict together,
because resampling them independently would destroy the pairing that carries the
information.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from finetune_eval_power.errors import ToolError
from finetune_eval_power.power.counts import POINTS

DEFAULT_RESAMPLES = 2000


@dataclass(frozen=True)
class DropInterval:
    """A measured drop in points, with a percentile interval and its own floor."""

    drop_points: float
    low_points: float
    high_points: float
    cases: int
    resamples: int

    @property
    def resolution_floor_points(self) -> float:
        return POINTS / self.cases

    @property
    def excludes_zero(self) -> bool:
        return self.low_points > 0.0 or self.high_points < 0.0


def bootstrap_seed_interval(
    per_seed_points: Sequence[float],
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
) -> DropInterval:
    """Percentile bootstrap over retrainings rather than over cases.

    This is the interval the repository publishes. Resampling cases answers "how
    much would this number move if I had sampled a different eval set", which is the
    question a power calculation already answers. Resampling seeds answers "how much
    would it move if I ran the fine-tune again", and that is the larger of the two by
    a wide margin on this data. Reporting only the first is how a measurement comes
    to look far more precise than it is.
    """
    values = np.asarray([float(value) for value in per_seed_points], dtype=float)
    if values.size < 2:
        raise ToolError("a seed-level interval needs at least two retrainings")
    if resamples < 1:
        raise ToolError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ToolError(f"confidence must be in (0, 1), got {confidence}")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(resamples, values.size))
    draws = values[picks].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return DropInterval(
        drop_points=float(values.mean()),
        low_points=float(low),
        high_points=float(high),
        cases=int(values.size),
        resamples=resamples,
    )


def bootstrap_drop_interval(
    before: np.ndarray,
    after: np.ndarray,
    seed: int,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
) -> DropInterval:
    """Paired percentile bootstrap of the accuracy drop over one case set."""
    before = np.asarray(before)
    after = np.asarray(after)
    if before.shape != after.shape:
        raise ToolError(f"paired vectors must align: {before.shape} against {after.shape}")
    if before.dtype != bool or after.dtype != bool:
        raise ToolError("paired vectors must be boolean correctness")
    cases = before.size
    if cases == 0:
        raise ToolError("cannot bootstrap an empty slice")
    if resamples < 1:
        raise ToolError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ToolError(f"confidence must be in (0, 1), got {confidence}")

    # Each case contributes exactly one of three values to the drop: +1 when it was
    # correct and is now wrong, -1 when the reverse, and 0 otherwise. Resampling
    # cases with replacement is therefore a multinomial draw over those three
    # categories, which is the same distribution as indexing the difference vector a
    # million times and is several orders of magnitude cheaper to compute.
    regressed = int(np.sum(before & ~after))
    improved = int(np.sum(~before & after))
    unchanged = cases - regressed - improved
    observed = (regressed - improved) / cases * POINTS
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        cases, [regressed / cases, improved / cases, unchanged / cases], size=resamples
    )
    draws = (counts[:, 0] - counts[:, 1]) / cases * POINTS
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return DropInterval(
        drop_points=observed,
        low_points=float(low),
        high_points=float(high),
        cases=cases,
        resamples=resamples,
    )
