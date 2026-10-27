#!/usr/bin/env python3
"""E1: how small a regression could this suite have seen?

Produces the minimum detectable effect against suite size, by the exact conditional
calculation and by the normal approximation, and alongside them the unpaired
calculation a team would reach for if it treated the two accuracies as independent
samples. The three disagree, and the size of the disagreement is the reason ADR-003
exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.common import load_arm, print_table, write_result  # noqa: E402

from finetune_eval_power.analysis import (  # noqa: E402
    smallest_suite_with_power,
    suite_size_curve,
    suite_size_for,
)
from finetune_eval_power.config import DecisionRule  # noqa: E402
from finetune_eval_power.power.unpaired import (  # noqa: E402
    unpaired_minimum_detectable_points,
)

SIZES = (120, 240, 480, 960, 1920, 3840)


def main() -> int:
    rule = DecisionRule()
    payload: dict = {"target_power": rule.target_power, "alpha": rule.alpha, "arms": {}}
    for name in ("vision", "sequence"):
        arm, analysis = load_arm(name)
        operating = analysis.operating
        discordance = operating.discordance
        baseline = analysis.suite_accuracy_before
        rows = []
        for row in suite_size_curve(discordance, SIZES, rule):
            size = int(row["suite_size"])
            rows.append(
                (
                    size,
                    row["exact_points"],
                    row["normal_points"],
                    unpaired_minimum_detectable_points(
                        size, baseline, rule.alpha, rule.target_power
                    ),
                )
            )
        print(f"\n{name}: discordance {discordance:.4f}, accuracy before {baseline:.4f}")
        print_table(("suite cases", "exact", "normal", "unpaired"), rows)

        at_shipped = operating.mde_points
        payload["arms"][name] = {
            "discordance": discordance,
            "suite_accuracy_before": baseline,
            "suite_size": analysis.suite_size,
            "mde_points": at_shipped,
            "mde_points_normal": suite_size_curve(discordance, (analysis.suite_size,), rule)[0][
                "normal_points"
            ],
            "mde_points_unpaired": unpaired_minimum_detectable_points(
                analysis.suite_size, baseline, rule.alpha, rule.target_power
            ),
            "smallest_size_with_any_power": smallest_suite_with_power(discordance, rule),
            "cases_needed_for_one_point": suite_size_for(1.0, discordance, rule),
            "curve": [
                {
                    "suite_size": size,
                    "exact_points": exact,
                    "normal_points": normal,
                    "unpaired_points": unpaired,
                }
                for size, exact, normal, unpaired in rows
            ],
        }
    destination = write_result("e1_detection_floor", payload)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
