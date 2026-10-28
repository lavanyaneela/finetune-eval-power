#!/usr/bin/env python3
"""E2: how much does the fine-tune forget, and when during the fine-tune?

Tracks eval suite accuracy at every fine-tune epoch, with the spread across
retrainings, at every freeze depth. The point of the curve is not the endpoint. It
is that the suite's reading wanders inside a band that is wide compared with the
effects teams argue about, so the epoch a team happened to stop at moves the number
they report.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from experiments.common import ROOT, print_table, write_result  # noqa: E402

from finetune_eval_power.analysis import analyse  # noqa: E402
from finetune_eval_power.config import ARMS, RUNS_DIRECTORY  # noqa: E402
from finetune_eval_power.power.counts import POINTS  # noqa: E402
from finetune_eval_power.runs import load_runs  # noqa: E402


def main() -> int:
    payload: dict = {"arms": {}}
    for name in ("vision", "sequence"):
        arm = ARMS[name]
        runs = load_runs(ROOT / RUNS_DIRECTORY / name)
        analysis = analyse(arm, runs)
        depth_position = runs.depth_index(analysis.operating_depth)
        curves = runs.suite_curve[:, depth_position, :] * POINTS
        start = curves[:, 0]
        rows = []
        for epoch in range(curves.shape[1]):
            column = curves[:, epoch] - start
            rows.append(
                (
                    epoch,
                    float(np.mean(column)),
                    float(np.min(column)),
                    float(np.max(column)),
                )
            )
        print(
            f"\n{name}: suite accuracy change by fine-tune epoch, freeze depth "
            f"{analysis.operating_depth}, {len(runs.seeds)} retrainings"
        )
        # The column is a change in accuracy, so the smallest value is the worst
        # retraining and the largest is the best. Labelling them the other way round
        # is a one word mistake that inverts the reader's conclusion.
        print_table(("epoch", "mean change", "worst run", "best run"), rows)

        widest = max(rows, key=lambda row: row[3] - row[2])
        final = rows[-1]
        all_depths = {}
        for freeze_depth in runs.freeze_depths:
            position = runs.depth_index(freeze_depth)
            series = runs.suite_curve[:, position, :] * POINTS
            all_depths[str(freeze_depth)] = {
                "final_change_points": float(np.mean(series[:, -1] - series[:, 0])),
                "worst_epoch_change_points": float(np.min(np.mean(series - series[:, :1], axis=0))),
            }
        payload["arms"][name] = {
            "freeze_depth": analysis.operating_depth,
            "epochs": curves.shape[1] - 1,
            "final_change_points": final[1],
            "final_band_points": final[3] - final[2],
            # The span of the readings themselves, as distinct from the span of the
            # change since epoch zero. Both are published because a sentence that
            # says "reading" and quotes the second one is false.
            "final_accuracy_span_points": float(np.max(curves[:, -1]) - np.min(curves[:, -1])),
            "widest_epoch": widest[0],
            "widest_band_points": widest[3] - widest[2],
            "per_depth": all_depths,
            "curve": [
                {"epoch": epoch, "mean_points": mean, "worst_points": low, "best_points": high}
                for epoch, mean, low, high in rows
            ],
        }
    destination = write_result("e2_forgetting_curve", payload)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
