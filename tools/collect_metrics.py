#!/usr/bin/env python3
"""Re-measures every published figure and writes docs/metrics.json.

Nothing in this file types a number. It runs the test suite, runs all five
experiments, reads their machine readable output and derives the rest. The test
count and the coverage come from `--junitxml` and `--cov-report=json` rather than
from a parsed progress line, because a parsed progress line quietly becomes whatever
the last run happened to print.

Each metric carries three things: the value, the exact string a document must use,
and the anchor phrases that must surround it. Rendering lives here rather than in
each document so that "2.7" and "2.70" cannot both be considered correct.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPERIMENTS = (
    "e1_detection_floor",
    "e2_forgetting_curve",
    "e3_blind_capabilities",
    "e4_sequence_arm",
    "e5_freeze_depth",
)
REPORTS = ROOT / "reports"
METRICS_PATH = ROOT / "docs" / "metrics.json"
CHECKED_DOCUMENTS = (
    "README.md",
    "docs/defense-guide.md",
    "docs/adr/ADR-001-procedural-generation.md",
    "docs/adr/ADR-002-keras-on-purpose.md",
    "docs/adr/ADR-003-paired-power.md",
    "docs/adr/ADR-004-seed-budget-and-pinned-training.md",
    "docs/adr/ADR-005-boundary.md",
)


def run_tests() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        # The same selection the Makefile and CI use. Without it this path collects
        # the training tests when the extra happens to be installed and writes a
        # count the documents contradict.
        "-m",
        "not slow",
        "--junitxml",
        str(REPORTS / "junit.xml"),
        "--cov=finetune_eval_power",
        "--cov-report",
        f"json:{REPORTS / 'coverage.json'}",
        "-q",
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit("the test suite failed, so no metrics were written")


def read_test_count() -> int:
    """Collected tests, not passed tests.

    Subtracting skips makes the published count differ between a machine with the
    training extra installed and one without, which is a number that moves for
    reasons that have nothing to do with the code.
    """
    tree = ElementTree.parse(REPORTS / "junit.xml")
    root = tree.getroot()
    suites = root.findall("testsuite") or [root]
    return sum(int(suite.get("tests", 0)) for suite in suites)


def read_coverage() -> float:
    payload = json.loads((REPORTS / "coverage.json").read_text(encoding="utf-8"))
    return float(payload["totals"]["percent_covered"])


def run_experiments() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for name in EXPERIMENTS:
        script = ROOT / "experiments" / f"{name}.py"
        outcome = subprocess.run(
            [sys.executable, str(script)], cwd=ROOT, check=False, capture_output=True, text=True
        )
        if outcome.returncode != 0:
            sys.stderr.write(outcome.stdout)
            sys.stderr.write(outcome.stderr)
            raise SystemExit(f"experiment {name} failed")
        results[name] = json.loads(
            (ROOT / "docs" / "experiments" / f"{name}.json").read_text(encoding="utf-8")
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="read existing junit and coverage reports instead of running pytest",
    )
    arguments = parser.parse_args()

    if not arguments.skip_tests:
        run_tests()
    experiments = run_experiments()

    e1 = experiments["e1_detection_floor"]["arms"]
    e2 = experiments["e2_forgetting_curve"]["arms"]
    e3 = experiments["e3_blind_capabilities"]["arms"]
    e5 = experiments["e5_freeze_depth"]["arms"]

    from finetune_eval_power.config import SWEEP_SEEDS, VISION_ARM
    from finetune_eval_power.formatting import floor_to
    from finetune_eval_power.runs import load_runs

    runs = load_runs(ROOT / "fixtures" / "runs" / "vision")
    vision_variance_seeds = len(runs.seeds)

    vision_e3 = e3["vision"]
    sequence_e3 = e3["sequence"]
    vision_e1 = e1["vision"]
    vision_e5 = e5["vision"]

    from finetune_eval_power.analysis import analyse

    analysis = analyse(VISION_ARM, runs)

    metrics: dict[str, object] = {
        "suite_cases": vision_e1["suite_size"],
        "probe_cases": analysis.probe_size,
        "retrainings": vision_variance_seeds,
        "seeds_declared": len(SWEEP_SEEDS),
        "mde_points": floor_to(vision_e1["mde_points"], 2),
        "mde_points_unpaired": floor_to(vision_e1["mde_points_unpaired"], 2),
        "smallest_suite_with_any_power": vision_e1["smallest_size_with_any_power"],
        "sequence_smallest_suite_with_power": e1["sequence"]["smallest_size_with_any_power"],
        "cases_for_one_point": vision_e1["cases_needed_for_one_point"],
        "suite_drop_points": round(vision_e3["suite_drop_points"], 2),
        "suite_flagged_in_seeds": vision_e3["suite_flagged_in_seeds"],
        "slices_total": vision_e3["slices_total"],
        "held_out_slices": vision_e3["held_out_slices"],
        "slices_without_power": vision_e3["slices_without_suite_power"],
        "undetected_slices": vision_e3["undetected_slices"],
        "undetected_covered_slices": vision_e3["undetected_covered_slices"],
        "worst_undetected_name": vision_e3["worst_undetected_name"],
        "worst_undetected_points": round(vision_e3["worst_undetected_points"], 2),
        "worst_undetected_suite_cases": vision_e3["worst_undetected_suite_cases"],
        "probe_resolution_floor_points": floor_to(100.0 / analysis.probe_size, 3),
        "thinnest_slice_floor_points": floor_to(
            max(item.resolution_floor_points for item in analysis.findings), 2
        ),
        "probe_change_points": round(analysis.probe_change_points, 2),
        "suite_worst_class_points": round(analysis.worst_class_on_suite.drop_points, 2),
        "worst_class_probe_cases": next(
            item.probe_cases
            for item in analysis.findings
            if item.name == analysis.worst_class_on_suite.name
        ),
        "suite_worst_class_floor_points": floor_to(analysis.worst_class_on_suite.floor_points, 2),
        "suite_worst_class_interval_low_points": round(
            analysis.worst_class_on_suite.interval_low_points, 2
        ),
        "label_classes": analysis.label_classes,
        "improving_classes": analysis.improving_classes,
        "improving_slices": analysis.improving_slices,
        "significant_verdict_seeds": analysis.significant_verdict_seeds,
        "maximal_undetected_slices": len(analysis.maximal_undetected),
        "improving_regions": analysis.improving_regions,
        "worst_undetected_slice_floor_points": floor_to(
            analysis.worst_undetected.suite_mde_on_slice_points, 2
        )
        if analysis.worst_undetected is not None
        and analysis.worst_undetected.suite_mde_on_slice_points is not None
        else None,
        "cheapest_underpowered_slice_cases": min(
            needed for _name, _cases, needed in analysis.slices_short_of_power if needed
        ),
        "dearest_underpowered_slice_cases": max(
            needed for _name, _cases, needed in analysis.slices_short_of_power if needed
        ),
        # Published so that "all three name a size" is a checked relation rather
        # than a sentence that happens to be true of this measurement. A slice
        # whose cases never change verdict names no size at all.
        "underpowered_slices_with_a_size": sum(
            1 for _name, _cases, needed in analysis.slices_short_of_power if needed
        ),
        "finest_slice_floor_points": floor_to(
            min(item.resolution_floor_points for item in analysis.findings), 3
        ),
        "shipped_outside_spread_points": round(vision_e5["shipped_outside_spread_points"], 2),
        "shipped_outside_noise_floor_points": floor_to(
            vision_e5["shipped_outside_noise_floor_points"], 2
        ),
        "final_epoch_accuracy_span_points": round(e2["vision"]["final_accuracy_span_points"], 2),
        "improving_class_points": round(analysis.improving_class_total_points, 2),
        "underpowered_covered_slices": analysis.underpowered_covered_slices,
        "seed_spread_points": round(analysis.variance.standard_deviation_points, 3),
        "seeds_to_settle": analysis.seeds_to_settle,
        "short_budget_noise_floor_points": floor_to(analysis.short_budget_noise_floor_points, 2),
        "seed_noise_floor_points": round(analysis.variance.noise_floor_points, 2),
        "shipped_depth": vision_e5["shipped_depth"],
        "in_region_gain_points": round(vision_e5["shipped_in_region_gain_points"], 2),
        "in_region_cases": vision_e5["in_region_cases"],
        "in_region_floor_points": floor_to(100.0 / vision_e5["in_region_cases"], 2),
        "in_region_spread_points": round(vision_e5["shipped_in_region_spread_points"], 2),
        "outside_loss_points": round(vision_e5["shipped_outside_loss_points"], 2),
        "safest_depth": vision_e5["safest_depth"],
        "safest_outside_change_points": round(vision_e5["safest_outside_loss_points"], 2),
        "safest_outside_spread_points": round(vision_e5["safest_outside_spread_points"], 2),
        "case_interval_points": round(analysis.suite_case_interval_width_points, 2),
        "retraining_interval_points": round(analysis.retraining_interval_width_points, 2),
        "epoch_band_points": round(e2["vision"]["final_band_points"], 2),
        "widest_band_points": round(e2["vision"]["widest_band_points"], 2),
        "widest_band_epoch": e2["vision"]["widest_epoch"],
        "sequence_suite_cases": e1["sequence"]["suite_size"],
        "sequence_mde_points": floor_to(e1["sequence"]["mde_points"], 2)
        if e1["sequence"]["mde_points"] is not None
        else None,
        "sequence_undetected_slices": sequence_e3["undetected_slices"],
        "sequence_slices_total": sequence_e3["slices_total"],
        "tests_total": read_test_count(),
        "coverage_pct": round(read_coverage(), 1),
    }

    rendered = {
        name: (value if isinstance(value, str) else _render(name, value))
        for name, value in metrics.items()
        if value is not None
    }

    payload = {
        "metrics": {name: value for name, value in metrics.items() if value is not None},
        "rendered": rendered,
        "anchors": ANCHORS,
        "checked_documents": list(CHECKED_DOCUMENTS),
        "note": (
            "Every value here is produced by running the test suite and the five "
            "experiments against the committed measurement. Nothing is typed by hand."
        ),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {METRICS_PATH} with {len(payload['metrics'])} metrics")
    return 0


TWO_DECIMALS = {
    "mde_points",
    "mde_points_unpaired",
    "suite_drop_points",
    "probe_change_points",
    "suite_worst_class_points",
    "suite_worst_class_floor_points",
    "suite_worst_class_interval_low_points",
    "improving_class_points",
    "thinnest_slice_floor_points",
    "in_region_floor_points",
    "in_region_spread_points",
    "safest_outside_change_points",
    "safest_outside_spread_points",
    "worst_undetected_points",
    "worst_undetected_slice_floor_points",
    "shipped_outside_spread_points",
    "shipped_outside_noise_floor_points",
    "final_epoch_accuracy_span_points",
    "seed_noise_floor_points",
    "short_budget_noise_floor_points",
    "in_region_gain_points",
    "outside_loss_points",
    "case_interval_points",
    "retraining_interval_points",
    "epoch_band_points",
    "widest_band_points",
    "sequence_mde_points",
}
THREE_DECIMALS = {
    "seed_spread_points",
    "probe_resolution_floor_points",
    "finest_slice_floor_points",
}
ONE_DECIMAL = {"coverage_pct"}


def _render(name: str, value: object) -> str:
    if name in TWO_DECIMALS:
        return f"{float(value):.2f}"
    if name in THREE_DECIMALS:
        return f"{float(value):.3f}"
    if name in ONE_DECIMAL:
        return f"{float(value):.1f}"
    return str(value)


ANCHORS: dict[str, list[str]] = {
    # Every anchor is anchored on exactly one side of the placeholder, with a word
    # next to it that no other anchor uses in that position. That is not a style
    # preference: two anchors whose contexts can absorb one another will both match
    # the same sentence, and one metric's guard will report the other metric's value
    # as stale. `check_numbers.anchors_can_collide` enforces it.
    "suite_cases": ["{} eval cases"],
    "probe_cases": ["{} probe cases"],
    "retrainings": ["{} retrainings"],
    "seeds_declared": ["{} seeds"],
    "mde_points": ["detection floor of {}"],
    "mde_points_unpaired": ["unpaired says {}"],
    "smallest_suite_with_any_power": ["power below {}"],
    "cases_for_one_point": ["would need {}"],
    "suite_drop_points": ["measured drop of {}"],
    "suite_flagged_in_seeds": ["{} of the retrainings"],
    "slices_total": ["{} named capability"],
    "held_out_slices": ["{} it never samples"],
    "slices_without_power": ["no usable power on {}"],
    "undetected_slices": ["{} slices regressed"],
    "undetected_covered_slices": ["{} sit inside"],
    "worst_undetected_name": ["worst is {}"],
    "worst_undetected_points": ["worst lost {}"],
    "seed_spread_points": ["deviation of {}"],
    "seeds_to_settle": ["settles after {}"],
    "short_budget_noise_floor_points": ["would have set {}"],
    "worst_undetected_suite_cases": ["suite holds {}"],
    "seed_noise_floor_points": ["noise floor reaches {}"],
    "shipped_depth": ["shipped depth {}"],
    "safest_depth": ["safest is depth {}"],
    "in_region_gain_points": ["gains {}"],
    "in_region_cases": ["{} cases in the region"],
    "in_region_floor_points": ["one of them is {}"],
    "in_region_spread_points": ["gain spread of {}"],
    "safest_outside_change_points": ["depth four moves {}"],
    "safest_outside_spread_points": ["that spread is {}"],
    "outside_loss_points": ["costs {}"],
    "case_interval_points": ["resampling cases gives {}"],
    "retraining_interval_points": ["retraining gives {}"],
    "epoch_band_points": ["spans {}"],
    "widest_band_points": ["widest band is {}"],
    "widest_band_epoch": ["at epoch {}"],
    "sequence_suite_cases": ["sequence ships {}"],
    "sequence_mde_points": ["sequence arm reaches {}"],
    "sequence_undetected_slices": ["sequence missed {}"],
    "sequence_slices_total": ["{} sequence slices named"],
    "sequence_smallest_suite_with_power": ["sequence needs {}"],
    "tests_total": ["{} tests"],
    "coverage_pct": ["{}% statement and branch"],
    "probe_change_points": ["probe says {}"],
    "suite_worst_class_points": ["suite shows {}"],
    "worst_class_probe_cases": ["{} referee cases"],
    "suite_worst_class_floor_points": ["its own floor is {}"],
    "suite_worst_class_interval_low_points": ["interval starts at {}"],
    "label_classes": ["across {} classes"],
    "improving_classes": ["{} improve and"],
    "improving_slices": ["{} slices moved up"],
    "significant_verdict_seeds": ["significant verdict in {}"],
    "maximal_undetected_slices": ["{} regions once nesting"],
    "improving_regions": ["{} regions the other"],
    "worst_undetected_slice_floor_points": ["own slice floor is {}"],
    "cheapest_underpowered_slice_cases": ["cheapest needs {}"],
    "dearest_underpowered_slice_cases": ["dearest needs {}"],
    "underpowered_slices_with_a_size": ["all {} name a size"],
    "finest_slice_floor_points": ["finest slice resolves {}"],
    "shipped_outside_spread_points": ["outside spread of {}"],
    "shipped_outside_noise_floor_points": ["noise bar of {}"],
    "final_epoch_accuracy_span_points": ["readings span {}"],
    "improving_class_points": ["together gain {}"],
    "underpowered_covered_slices": ["{} sampled too thinly"],
    "probe_resolution_floor_points": ["probe resolves {}"],
    "thinnest_slice_floor_points": ["thinnest slice resolves {}"],
}


if __name__ == "__main__":
    raise SystemExit(main())
