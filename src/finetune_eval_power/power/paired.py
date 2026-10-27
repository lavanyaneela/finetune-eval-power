"""Power for the test that matches the design: McNemar on paired verdicts.

The suite scores the same cases before and after fine-tuning, so the two accuracies
are not independent samples. All the information about the difference lives in the
discordant cases, the ones whose verdict changed, and the discordant count is
usually a small fraction of the suite. That has a consequence worth stating plainly:
suite size matters less than discordance. Doubling a suite whose cases all agree
with themselves buys nothing.

Two power functions are provided because they disagree, and the disagreement is the
subject of ADR-003:

- `mcnemar_exact_power` enumerates the conditional binomial exactly. It is the one
  whose number is published.
- `mcnemar_normal_power` is the closed form in the textbooks. It is what the suite
  size curve is drawn with, and it is optimistic at the small discordant counts this
  kind of suite actually produces.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from finetune_eval_power.errors import ToolError

POINTS = 100.0
#: Binomial mass outside the enumerated range of discordant counts. Small enough
#: that it cannot move a power value in the third decimal place.
_TAIL_TOLERANCE = 1e-12
_BISECTION_STEPS = 48
#: Bisection stops once the bracket is this narrow, which is one ten-thousandth of a
#: percentage point. Running the full step budget regardless costs several times more
#: and buys digits well past anything that gets published.
_BISECTION_TOLERANCE = 1e-6


def _validate(discordance: float, drop: float, alpha: float) -> None:
    if not 0.0 < discordance <= 1.0:
        raise ToolError(f"discordance must be in (0, 1], got {discordance}")
    if drop < 0.0:
        raise ToolError(f"drop must be non-negative, got {drop}")
    if drop > discordance:
        raise ToolError(
            f"a drop of {drop} is impossible at discordance {discordance}: "
            "the regressed cases cannot outnumber the cases that changed"
        )
    if not 0.0 < alpha < 1.0:
        raise ToolError(f"alpha must be in (0, 1), got {alpha}")


def rejection_boundary(discordant: int, alpha: float, two_sided: bool = True) -> int:
    """Largest regressed count in the lower rejection tail of the exact binomial test.

    Returns -1 when no outcome rejects, which happens whenever the discordant count
    is too small for the exact test to reach alpha at all. That case is not an
    inconvenience to be smoothed over: it is the regime where a suite is incapable
    of a significant result no matter what the model did.
    """
    if discordant <= 0:
        return -1
    tail = alpha / 2.0 if two_sided else alpha
    return int(_boundaries(np.array([discordant]), tail)[0])


def _boundaries(discordant: np.ndarray, tail: float) -> np.ndarray:
    """Vectorised rejection boundary for many discordant counts at once.

    Written against `binom.ppf` rather than by walking candidates upward. The first
    version of this function did walk, one scipy call per candidate, which made a
    single power evaluation quadratic in the suite size and a bisection over it
    unusable above a few hundred cases.
    """
    candidate = stats.binom.ppf(tail, discordant, 0.5)
    candidate = np.nan_to_num(candidate, nan=-1.0, posinf=-1.0, neginf=-1.0)
    candidate = np.minimum(candidate, discordant // 2)
    too_large = stats.binom.cdf(candidate, discordant, 0.5) > tail
    candidate = np.where(too_large, candidate - 1.0, candidate)
    candidate = np.where(discordant <= 0, -1.0, candidate)
    return np.maximum(candidate, -1.0).astype(int)


def mcnemar_exact_power(
    suite_size: int,
    discordance: float,
    drop: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Exact power, summing over how many cases could change verdict.

    The discordant count is itself random, binomial in the suite size, so the power
    is an expectation over it rather than a value at its mean. Taking the value at
    the mean overstates power whenever the mean is small, which is exactly when the
    answer matters.

    With `drop` of zero this returns the true size of the test rather than alpha.
    The exact binomial test is conservative at small discordant counts, and a suite
    whose real false alarm rate is a third of its nominal one is also a suite with
    less power than its nominal alpha suggests.
    """
    if suite_size <= 0:
        raise ToolError(f"suite size must be positive, got {suite_size}")
    _validate(discordance, drop, alpha)

    share_regressed = (discordance + drop) / 2.0 / discordance
    low = max(int(stats.binom.ppf(_TAIL_TOLERANCE, suite_size, discordance)), 0)
    high = min(int(stats.binom.isf(_TAIL_TOLERANCE, suite_size, discordance)), suite_size)
    counts = np.arange(low, high + 1)
    if counts.size == 0:  # pragma: no cover - only reachable if scipy inverts the tails
        return 0.0
    weights = stats.binom.pmf(counts, suite_size, discordance)
    tail = alpha / 2.0 if two_sided else alpha
    boundaries = _boundaries(counts, tail)
    usable = boundaries >= 0
    if not np.any(usable):
        return 0.0
    counts = counts[usable]
    weights = weights[usable]
    boundaries = boundaries[usable]
    # The alternative of interest is a regression, so the upper tail of the regressed
    # count is the one that matters. The lower tail joins it only when the test is
    # two sided.
    rejection = stats.binom.sf(counts - boundaries - 1, counts, share_regressed)
    if two_sided:
        rejection = rejection + stats.binom.cdf(boundaries, counts, share_regressed)
    return float(np.sum(weights * rejection))


def mcnemar_normal_power(
    suite_size: int,
    discordance: float,
    drop: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Closed-form power, the Connor normal approximation.

    Kept because it is what a reader will reach for, and because the suite size
    curve needs thousands of evaluations. Its optimism against the exact value is
    measured in E1 rather than asserted here.
    """
    if suite_size <= 0:
        raise ToolError(f"suite size must be positive, got {suite_size}")
    _validate(discordance, drop, alpha)
    if drop == 0.0:
        return alpha
    quantile = stats.norm.isf(alpha / 2.0 if two_sided else alpha)
    residual = discordance - drop**2
    if residual <= 0.0:
        return 1.0
    statistic = (drop * math.sqrt(suite_size) - quantile * math.sqrt(discordance)) / math.sqrt(
        residual
    )
    return float(stats.norm.cdf(statistic))


def _power_function(method: str):
    if method == "exact":
        return mcnemar_exact_power
    if method == "normal":
        return mcnemar_normal_power
    raise ToolError(f"unknown power method {method!r}, expected 'exact' or 'normal'")


def minimum_detectable_effect(
    suite_size: int,
    discordance: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    method: str = "exact",
    two_sided: bool = True,
) -> float | None:
    """Smallest accuracy drop this suite reaches `target_power` on, as a proportion.

    Returns None when even the largest drop the discordance allows falls short, which
    is a real answer and not an error: at that discordance the suite cannot reach the
    requested power at any effect size.
    """
    power_at = _power_function(method)
    if power_at(suite_size, discordance, discordance, alpha, two_sided) < target_power:
        return None
    low, high = 0.0, discordance
    for _ in range(_BISECTION_STEPS):
        if high - low < _BISECTION_TOLERANCE:
            break
        middle = (low + high) / 2.0
        if power_at(suite_size, discordance, middle, alpha, two_sided) >= target_power:
            high = middle
        else:
            low = middle
    return high


def minimum_detectable_points(
    suite_size: int,
    discordance: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    method: str = "exact",
    two_sided: bool = True,
) -> float | None:
    """`minimum_detectable_effect` in percentage points, which is how it is published."""
    effect = minimum_detectable_effect(
        suite_size, discordance, alpha, target_power, method, two_sided
    )
    return None if effect is None else effect * POINTS


def required_suite_size(
    drop: float,
    discordance: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    method: str = "exact",
    two_sided: bool = True,
    ceiling: int = 2_000_000,
) -> int | None:
    """Smallest suite size that reaches `target_power` against this drop.

    Returns None when the ceiling is reached, so a caller can say "more than N" rather
    than printing a number that is really a loop bound.
    """
    _validate(discordance, drop, alpha)
    if drop <= 0.0:
        return None
    power_at = _power_function(method)
    low, high = 1, 16
    while high <= ceiling:
        if power_at(high, discordance, drop, alpha, two_sided) >= target_power:
            break
        low = high
        high *= 2
    else:
        return None
    while low + 1 < high:
        middle = (low + high) // 2
        if power_at(middle, discordance, drop, alpha, two_sided) >= target_power:
            high = middle
        else:
            low = middle
    return high


def mcnemar_p_value(only_before: int, only_after: int, two_sided: bool = True) -> float:
    """Exact McNemar p-value for one measured table.

    Delegates the test itself to statsmodels. The power arithmetic above is this
    repository's own work; the hypothesis test is a library call and is labelled as
    one so that the two are not confused.
    """
    from statsmodels.stats.contingency_tables import mcnemar as statsmodels_mcnemar

    if only_before < 0 or only_after < 0:
        raise ToolError("discordant counts cannot be negative")
    if only_before + only_after == 0:
        return 1.0
    table = [[0, only_before], [only_after, 0]]
    result = statsmodels_mcnemar(table, exact=True, correction=False)
    p_value = float(result.pvalue)
    return p_value if two_sided else p_value / 2.0
