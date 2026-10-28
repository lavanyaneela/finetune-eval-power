"""One place that decides how a figure is written down.

A minimum must round downward, never to nearest. An earlier build printed a floor
with ordinary rounding and put it under the words "never below", where it was
false by a fraction. The same helper is used by the report and by the metrics
collector, because the report is screenshotted into the README and a report that
rounds differently from the published figure puts two numbers for one quantity in
front of a reviewer.
"""

from __future__ import annotations

import math

DEFAULT_DIGITS = 2


def floor_to(value: float, digits: int = DEFAULT_DIGITS) -> float:
    """Round down, so a published minimum is never larger than the real one.

    This is `math.floor`, which rounds toward negative infinity rather than
    toward zero: for a negative value it rounds away from zero. No published
    minimum is negative today, and a negative one would want the other direction,
    so the caller that introduces one has to think about it rather than inherit a
    docstring that says what this does not do.
    """
    scale = 10**digits
    return math.floor(value * scale) / scale


def format_floor(value: float | None, digits: int = DEFAULT_DIGITS) -> str:
    """Render a minimum, or say plainly that there is not one."""
    if value is None:
        return "no floor"
    return f"{floor_to(value, digits):.{digits}f}"
