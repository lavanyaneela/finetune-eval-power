"""The calculation this repository argues against, implemented so the gap is measured.

A team that reports "accuracy went from 91.2 to 89.4" and then asks whether the
suite was big enough almost always reaches for a two independent proportions
calculation. The two accuracies came from the same cases, so that model is wrong.
It is kept here, under its own name, so E1 can publish how far wrong rather than
asserting that it is.
"""

from __future__ import annotations

import math

from scipy import stats

from finetune_eval_power.errors import ToolError

POINTS = 100.0
_BISECTION_STEPS = 48
_BISECTION_TOLERANCE = 1e-6


def unpaired_power(
    per_arm_size: int,
    baseline_accuracy: float,
    drop: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Power of a two independent proportions z test, unpooled variance."""
    if per_arm_size <= 0:
        raise ToolError(f"per-arm size must be positive, got {per_arm_size}")
    if not 0.0 < baseline_accuracy <= 1.0:
        raise ToolError(f"baseline accuracy must be in (0, 1], got {baseline_accuracy}")
    if drop < 0.0 or drop > baseline_accuracy:
        raise ToolError(f"a drop of {drop} is impossible from {baseline_accuracy}")
    if drop == 0.0:
        return alpha
    after = baseline_accuracy - drop
    variance = (
        baseline_accuracy * (1.0 - baseline_accuracy) + after * (1.0 - after)
    ) / per_arm_size
    if variance <= 0.0:
        return 1.0
    quantile = stats.norm.isf(alpha / 2.0 if two_sided else alpha)
    return float(stats.norm.sf(quantile - drop / math.sqrt(variance)))


def unpaired_minimum_detectable_effect(
    per_arm_size: int,
    baseline_accuracy: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    two_sided: bool = True,
) -> float | None:
    """Smallest drop the unpaired calculation claims to see, as a proportion."""
    if unpaired_power(per_arm_size, baseline_accuracy, baseline_accuracy, alpha, two_sided) < (
        target_power
    ):
        return None
    low, high = 0.0, baseline_accuracy
    for _ in range(_BISECTION_STEPS):
        if high - low < _BISECTION_TOLERANCE:
            break
        middle = (low + high) / 2.0
        if unpaired_power(per_arm_size, baseline_accuracy, middle, alpha, two_sided) >= (
            target_power
        ):
            high = middle
        else:
            low = middle
    return high


def unpaired_minimum_detectable_points(
    per_arm_size: int,
    baseline_accuracy: float,
    alpha: float = 0.05,
    target_power: float = 0.80,
    two_sided: bool = True,
) -> float | None:
    effect = unpaired_minimum_detectable_effect(
        per_arm_size, baseline_accuracy, alpha, target_power, two_sided
    )
    return None if effect is None else effect * POINTS
