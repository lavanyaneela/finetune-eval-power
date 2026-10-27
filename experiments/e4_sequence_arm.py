#!/usr/bin/env python3
"""E4: does the arithmetic hold away from convolutions?

The sequence arm repeated against the vision arm, side by side. What transfers is
not the size of the effect, which is a fact about two particular models, but the
shape: a suite drawn from the region the fine-tune narrowed onto reports nothing,
its detection floor is set by discordance rather than by case count, and the slices
it never samples are invisible to it at any effect size.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_arm, print_table, write_result  # noqa: E402

FIELDS = (
    ("suite cases", lambda analysis: analysis.suite_size),
    ("probe cases", lambda analysis: analysis.probe_size),
    ("freeze depth shipped", lambda analysis: analysis.operating_depth),
    ("accuracy before", lambda analysis: analysis.suite_accuracy_before),
    ("suite drop, points", lambda analysis: analysis.operating.suite_drop_points),
    ("discordance", lambda analysis: analysis.operating.discordance),
    ("detection floor, points", lambda analysis: analysis.operating.mde_points),
    ("named slices", lambda analysis: analysis.slices_total),
    ("never sampled", lambda analysis: analysis.held_out_slices),
    ("no usable power", lambda analysis: analysis.slices_without_suite_power),
    ("missed slices", lambda analysis: len(analysis.undetected)),
    (
        "worst missed, points",
        lambda analysis: (
            analysis.worst_undetected.drop_points if analysis.worst_undetected else None
        ),
    ),
    (
        "retraining spread, points",
        lambda analysis: analysis.variance.standard_deviation_points,
    ),
)


def main() -> int:
    analyses = {name: load_arm(name)[1] for name in ("vision", "sequence")}
    rows = [
        (label, extract(analyses["vision"]), extract(analyses["sequence"]))
        for label, extract in FIELDS
    ]
    print("\nthe same measurement on both arms")
    print_table(("quantity", "vision", "sequence"), rows)

    payload = {
        "arms": {
            name: {label: extract(analysis) for label, extract in FIELDS}
            for name, analysis in analyses.items()
        },
        "both_arms_missed_slices": all(
            len(analysis.undetected) > 0 for analysis in analyses.values()
        ),
        "both_arms_have_slices_without_power": all(
            analysis.slices_without_suite_power > 0 for analysis in analyses.values()
        ),
    }
    destination = write_result("e4_sequence_arm", payload)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
