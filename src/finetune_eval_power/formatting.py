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
    """Round toward zero's side of the value, so a floor stays a floor."""
    scale = 10**digits
    return math.floor(value * scale) / scale


def format_floor(value: float | None, digits: int = DEFAULT_DIGITS) -> str:
    """Render a minimum, or say plainly that there is not one."""
    if value is None:
        return "unreachable"
    return f"{floor_to(value, digits):.{digits}f}"
