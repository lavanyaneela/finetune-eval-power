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
#: How close a running estimate has to sit to the final one to count as settled.
STABILITY_BAND = 0.10


def running_spread(readings: Sequence[float]) -> list[tuple[int, float]]:
    """Standard deviation of the first k readings, for every k from two upward.

    The seed budget is the one configuration number in this repository that is a
    judgement call, and the ADR that picked it argued qualitatively. This makes the
    argument checkable: a variance estimate from a handful of retrainings is itself
    noisy, and the refusal threshold is computed from it, so an unstable estimate
    sets the bar in the wrong place and nothing downstream notices.
    """
    if len(readings) < 2:
        raise ToolError("a running spread needs at least two readings")
    return [
        (count, float(statistics.stdev(readings[:count]))) for count in range(2, len(readings) + 1)
    ]


def seeds_to_settle(readings: Sequence[float], band: float = STABILITY_BAND) -> int:
    """Fewest retrainings after which the estimate stays inside `band` of the final.

    Reported rather than asserted. If it ever came out at or above the seed budget
    the honest reading would be that the budget is too small, which is why the
    number is published next to the budget instead of being hidden in a constant.
    """
    history = running_spread(readings)
    final = history[-1][1]
    if final <= 0.0:
        return history[0][0]
    # The last entry always satisfies the test against itself, so the loop always
    # returns and a trailing fallback would be unreachable. There was one, and
    # coverage pointed at it rather than at a missing test.
    return next(
        count
        for position, (count, _value) in enumerate(history)
        if all(abs(value - final) / final <= band for _count, value in history[position:])
    )


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
