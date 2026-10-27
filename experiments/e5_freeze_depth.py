#!/usr/bin/env python3
"""E5: what does freezing buy, and what does it cost?

Freeze depth is the one knob a team actually turns. Deeper freezing protects the
capabilities the model already had and gives up adaptation to the new data. This
reports both sides of that trade at every depth, together with the eval suite's
verdict, so the table shows which depth the suite would wave through and what that
depth costs in capabilities it cannot see.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.common import ROOT, print_table, write_result  # noqa: E402

from finetune_eval_power.analysis import analyse  # noqa: E402
from finetune_eval_power.config import ARMS, RUNS_DIRECTORY, DecisionRule  # noqa: E402
from finetune_eval_power.power.counts import POINTS, paired_counts  # noqa: E402
from finetune_eval_power.runs import load_runs  # noqa: E402
from finetune_eval_power.slices.suite import build_probe  # noqa: E402


def main() -> int:
    rule = DecisionRule()
    payload: dict = {"arms": {}}
    for name in ("vision", "sequence"):
        arm = ARMS[name]
        runs = load_runs(ROOT / RUNS_DIRECTORY / name)
        analysis = analyse(arm, runs, rule)
        probe = build_probe(arm.space, arm.probe_repeats)
        region = probe.region_mask(arm.finetune_profile)
        outside = ~region

        rows = []
        depths = []
        for summary in analysis.depths:
            position = runs.depth_index(summary.freeze_depth)
            gains, losses = [], []
            for seed_position in range(len(runs.seeds)):
                before = runs.probe_before[seed_position, position]
                after = runs.probe_after[seed_position, position]
                gains.append(float(np.mean(after[region]) - np.mean(before[region])) * POINTS)
                losses.append(paired_counts(before[outside], after[outside]).drop_points)
            gain = float(np.mean(gains))
            loss = float(np.mean(losses))
            gain_spread = float(np.std(gains, ddof=1))
            loss_spread = float(np.std(losses, ddof=1))
            rows.append(
                (
                    summary.freeze_depth,
                    gain,
                    loss,
                    summary.suite_drop_points,
                    f"{summary.seeds_flagging_regression} of {summary.seeds}",
                    "shipped" if summary.freeze_depth == analysis.operating_depth else "",
                )
            )
            depths.append(
                {
                    "freeze_depth": summary.freeze_depth,
                    "in_region_gain_points": gain,
                    "in_region_spread_points": gain_spread,
                    "outside_region_loss_points": loss,
                    "outside_region_spread_points": loss_spread,
                    "suite_drop_points": summary.suite_drop_points,
                    "seeds_flagging": summary.seeds_flagging_regression,
                    "mde_points": summary.mde_points,
                }
            )
        print(f"\n{name}: freeze depth against what it buys and what it costs")
        print_table(
            ("depth", "in-region gain", "outside loss", "suite drop", "flagged", ""),
            rows,
        )

        shipped = next(item for item in depths if item["freeze_depth"] == analysis.operating_depth)
        safest = min(depths, key=lambda item: item["outside_region_loss_points"])
        payload["arms"][name] = {
            "shipped_depth": analysis.operating_depth,
            "shipped_in_region_gain_points": shipped["in_region_gain_points"],
            "shipped_in_region_spread_points": shipped["in_region_spread_points"],
            "shipped_outside_loss_points": shipped["outside_region_loss_points"],
            "in_region_cases": int(region.sum()),
            "outside_region_cases": int(outside.sum()),
            "safest_depth": safest["freeze_depth"],
            "safest_outside_loss_points": safest["outside_region_loss_points"],
            "safest_outside_spread_points": safest["outside_region_spread_points"],
            "safest_in_region_gain_points": safest["in_region_gain_points"],
            "depths": depths,
        }
    destination = write_result("e5_freeze_depth", payload)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
