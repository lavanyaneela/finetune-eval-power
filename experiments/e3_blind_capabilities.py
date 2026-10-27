#!/usr/bin/env python3
"""E3: what regressed while the suite said nothing?

The headline experiment. For every named capability slice it reports the measured
drop at the freeze depth a team would ship, the interval across retrainings, how
many cases the suite holds for that slice, and whether the suite could have reached
the target power on it at all.

The materiality bar is not a number anyone chose. A slice counts as missed when it
moved further than the suite's own minimum detectable effect while the suite raised
no flag, so the comparison is against the thing that was supposed to be watching.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_arm, print_table, write_result  # noqa: E402


def main() -> int:
    payload: dict = {"arms": {}}
    for name in ("vision", "sequence"):
        _arm, analysis = load_arm(name)
        operating = analysis.operating
        missed = analysis.undetected
        ordered = sorted(analysis.findings, key=lambda item: item.drop_points, reverse=True)
        rows = [
            (
                item.name,
                "yes" if item.covered else "no",
                item.suite_cases,
                item.probe_cases,
                item.drop_points,
                f"[{item.interval.low_points:+.2f}, {item.interval.high_points:+.2f}]",
                item.suite_mde_on_slice_points,
                "MISSED" if item in missed else "",
            )
            for item in ordered
        ]
        print(
            f"\n{name}: freeze depth {analysis.operating_depth}, suite reported "
            f"{operating.suite_drop_points:+.2f} points and flagged a regression in "
            f"{operating.seeds_flagging_regression} of {operating.seeds} retrainings"
        )
        print_table(
            ("slice", "covered", "suite", "probe", "drop", "interval", "slice floor", ""),
            rows,
        )
        worst = analysis.worst_undetected
        payload["arms"][name] = {
            "freeze_depth": analysis.operating_depth,
            "suite_drop_points": operating.suite_drop_points,
            "suite_flagged_in_seeds": operating.seeds_flagging_regression,
            "seeds": operating.seeds,
            "suite_mde_points": operating.mde_points,
            "slices_total": analysis.slices_total,
            "held_out_slices": analysis.held_out_slices,
            "slices_without_suite_power": analysis.slices_without_suite_power,
            "undetected_slices": len(missed),
            "undetected_covered_slices": sum(1 for item in missed if item.covered),
            "worst_undetected_name": worst.name if worst else None,
            "worst_undetected_points": worst.drop_points if worst else None,
            "worst_undetected_suite_cases": worst.suite_cases if worst else None,
            "resolution_floor_points": min(
                item.resolution_floor_points for item in analysis.findings
            ),
            "slices": [
                {
                    "name": item.name,
                    "covered": item.covered,
                    "suite_cases": item.suite_cases,
                    "probe_cases": item.probe_cases,
                    "drop_points": item.drop_points,
                    "interval_low_points": item.interval.low_points,
                    "interval_high_points": item.interval.high_points,
                    "slice_floor_points": item.suite_mde_on_slice_points,
                    "missed": item in missed,
                }
                for item in ordered
            ],
        }
    destination = write_result("e3_blind_capabilities", payload)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
