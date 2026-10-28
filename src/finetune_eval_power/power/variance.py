"""The part of the noise that no suite size removes.

A power calculation treats the eval suite as the only source of error, so its answer
to "this effect is too small to see" is always "use a bigger suite". There is a
second source, and it does not shrink: retrain the same configuration with a
different seed and the measured regression moves. Suite noise falls as one over the
square root of the suite size, seed noise is flat in it.

So when the effect a team wants to detect is smaller than the interval retraining
alone produces, the honest answer is not a larger suite. It is that the question is
unanswerable as asked, which is what exit code 2 says.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from finetune_eval_power.errors import ToolError

MINIMUM_SEEDS = 3


@dataclass(frozen=True)
class SeedVariance:
    """Spread of the measured regression across retrainings of one configuration."""

    seeds: int
    mean_points: float
    standard_deviation_points: float
    minimum_points: float
    maximum_points: float
    multiplier: float

    @property
    def spread_points(self) -> float:
        """Observed range, which is what a reader intuits even when sigma is the rule."""
        return self.maximum_points - self.minimum_points

    @property
    def noise_floor_points(self) -> float:
        """Largest claim that retraining noise alone swallows.

        Inclusive at the boundary on purpose: a claim exactly at the floor is
        refused, because the boundary is not evidence. A test pins that, and the
        documented wording is "at least as wide as", never "wider than".
        """
        return self.multiplier * self.standard_deviation_points

    def swallows(self, claim_points: float) -> bool:
        """True when a claim this small cannot be resolved by any suite size."""
        if claim_points <= 0.0:
            raise ToolError(f"a claimed effect must be positive, got {claim_points}")
        return claim_points <= self.noise_floor_points

    def seeds_needed_for(self, claim_points: float) -> int | None:
        """Seeds needed before the mean regression is resolved to this claim.

        Averaging over seeds shrinks the noise on the *mean* as one over the square
        root of the seed count, so a claim below the single-run floor is still
        reachable by retraining more times. This is the honest remedy to offer with
        exit code 2, and it prices it: the answer is usually a number of retrainings
        nobody is going to pay for. Returns None when the spread is zero.
        """
        if claim_points <= 0.0:
            raise ToolError(f"a claimed effect must be positive, got {claim_points}")
        if self.standard_deviation_points <= 0.0:
            return None
        ratio = self.multiplier * self.standard_deviation_points / claim_points
        return max(1, int(ratio**2) + 1)


def seed_variance(
    regressions_points: Sequence[float],
    multiplier: float,
) -> SeedVariance:
    """Summarise the per-seed measured regressions of one configuration."""
    values = [float(value) for value in regressions_points]
    if len(values) < MINIMUM_SEEDS:
        raise ToolError(
            f"a variance estimate needs at least {MINIMUM_SEEDS} seeds, got {len(values)}"
        )
    if multiplier <= 0.0:
        raise ToolError(f"the noise multiplier must be positive, got {multiplier}")
    return SeedVariance(
        seeds=len(values),
        mean_points=statistics.fmean(values),
        standard_deviation_points=statistics.stdev(values),
        minimum_points=min(values),
        maximum_points=max(values),
        multiplier=multiplier,
    )
