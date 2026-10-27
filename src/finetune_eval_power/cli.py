"""The command line surface, and the exit codes that carry the actual answer.

The distinction the codes exist to make is between "the tool worked and the answer
is no" and "the tool cannot answer this". Most eval tooling collapses those two, and
a team reading a green pipeline cannot tell which one it got.

    0  the tool answered, and the answer is the reassuring one: `power` means the
       suite can see a regression of the size claimed, `blind` means nothing fell
       past its floor unnoticed.
    1  the tool answered, and the answer is the unwelcome one: `power` means the
       suite cannot see it and prints the suite size that could, `blind` means it
       found capabilities that regressed unflagged and lists them.
    2  refused. Retraining noise alone is larger than the claimed effect, so no
       suite size fixes it. The number of retrainings that would is printed.
    3  the tool itself failed

`suite`, `curve`, `report` and `metrics` describe the measurement rather than
judging it, so they return 0 unless the tool fails.

Code 2 is the one worth reading twice. A power calculation always has a remedy to
offer, because its model of the world contains exactly one noise source. When the
real obstacle is that the same configuration trained twice disagrees by more than
the effect under discussion, the remedy it offers is wrong.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from finetune_eval_power import __version__
from finetune_eval_power.analysis import ArmAnalysis, analyse, suite_size_curve, suite_size_for
from finetune_eval_power.config import ARMS, DEFAULT_CLAIM_POINTS, RUNS_DIRECTORY, DecisionRule
from finetune_eval_power.errors import ToolError, Unanswerable
from finetune_eval_power.formatting import format_floor
from finetune_eval_power.runs import load_runs

EXIT_DETECTABLE = 0
EXIT_NOT_DETECTABLE = 1
EXIT_REFUSED = 2
EXIT_TOOL_ERROR = 3

DEBUG_VARIABLE = "FEP_DEBUG"
CURVE_SIZES = (120, 240, 480, 960, 1920, 3840)


def _runs_root(arguments: argparse.Namespace) -> Path:
    return Path(arguments.runs or RUNS_DIRECTORY)


def _load(arguments: argparse.Namespace) -> tuple[ArmAnalysis, DecisionRule]:
    if arguments.arm not in ARMS:
        raise ToolError(f"unknown arm {arguments.arm!r}, expected one of {sorted(ARMS)}")
    arm = ARMS[arguments.arm]
    runs = load_runs(_runs_root(arguments) / arm.name)
    rule = DecisionRule(claim_points=getattr(arguments, "claim", DEFAULT_CLAIM_POINTS))
    return analyse(arm, runs, rule), rule


def command_power(arguments: argparse.Namespace) -> int:
    analysis, rule = _load(arguments)
    operating = analysis.operating
    claim = rule.claim_points
    variance = analysis.variance

    print(f"arm                  {analysis.arm}")
    print(f"suite cases          {analysis.suite_size}")
    print(f"retrainings          {len(analysis.seeds)}")
    print(f"freeze depth         {analysis.operating_depth} (shallowest the suite allows)")
    print(f"suite discordance    {operating.discordance:.4f}")
    print(f"claimed effect       {claim:.2f} points")
    print(
        f"retraining spread    {variance.standard_deviation_points:.3f} points standard "
        f"deviation over {variance.seeds} retrainings"
    )

    if variance.swallows(claim):
        needed = variance.seeds_needed_for(claim)
        remedy = (
            f"average over about {needed} retrainings"
            if needed is not None
            else "reduce training nondeterminism"
        )
        print(
            f"\nREFUSED: retraining noise alone reaches "
            f"{format_floor(variance.noise_floor_points)} points either side of the "
            f"measurement, which is wider than the {claim:.2f} point effect being claimed."
        )
        print(
            "No suite size resolves this. The eval suite is not the binding "
            f"constraint here. To answer the question as asked, {remedy}."
        )
        return EXIT_REFUSED

    floor = operating.mde_points
    if floor is None:
        print(
            f"\nNO: at a discordance of {operating.discordance:.4f} this suite cannot "
            f"reach {rule.target_power:.0%} power at any effect size. Its cases almost "
            "never change verdict, so there is nothing for the test to work with."
        )
        return EXIT_NOT_DETECTABLE

    # Rounded down, not to nearest, and through the same helper the report and the
    # metrics collector use. A floor printed one way here and another way in the
    # README puts two numbers for one quantity in front of a reviewer, and this
    # command is the one the defense guide tells the candidate to run live.
    print(f"minimum detectable   {format_floor(floor)} points at {rule.target_power:.0%} power")
    if claim >= floor:
        print(f"\nYES: a {claim:.2f} point regression is above this suite's floor.")
        return EXIT_DETECTABLE

    needed = suite_size_for(claim, operating.discordance, rule)
    required = f"{needed} cases" if needed is not None else "more cases than this tool will search"
    print(
        f"\nNO: a {claim:.2f} point regression is below this suite's floor of "
        f"{floor:.2f} points. Detecting it at the measured discordance needs {required}."
    )
    return EXIT_NOT_DETECTABLE


def command_blind(arguments: argparse.Namespace) -> int:
    analysis, _ = _load(arguments)
    operating = analysis.operating
    floor = operating.mde_points
    print(f"arm {analysis.arm}, freeze depth {analysis.operating_depth}")
    # "Drop" is positive when the model got worse, so a negative drop is an
    # improvement. Said out loud every time it is printed, because a signed number
    # under an unsigned noun is how a reader ends up with the opposite conclusion.
    print(
        f"suite measured drop: {operating.suite_drop_points:+.2f} points "
        f"({'accuracy rose' if operating.suite_drop_points < 0 else 'accuracy fell'})"
    )
    print(
        f"  regression flags: {operating.seeds_flagging_regression} of {operating.seeds} "
        f"retrainings; significant either way: {analysis.significant_verdict_seeds}"
    )
    print(f"probe measured drop: {analysis.probe_change_points:+.2f} points over the whole space")
    print(
        "suite detection floor: "
        + ("unreachable at any effect size" if floor is None else f"{format_floor(floor)} points")
    )
    print(
        f"slices: {analysis.slices_total} named, {analysis.slices_without_suite_power} with "
        f"no usable suite power, of which {analysis.held_out_slices} it never samples and "
        f"{analysis.underpowered_covered_slices} whose cases change verdict too rarely"
    )
    print(f"slices that moved the other way, interval clear of zero: {analysis.improving_slices}")
    print("per class, measured on the suite's own cases:")
    print(f"  {'class':<24} {'cases':>6} {'drop':>8} {'its own floor':>14} {'interval':>18}")
    for item in analysis.suite_class_findings:
        floor = "none" if item.floor_points is None else f"{format_floor(item.floor_points)}p"
        interval = f"[{item.interval_low_points:+.2f}, {item.interval_high_points:+.2f}]"
        print(
            f"  {item.name:<24} {item.suite_cases:>6} {item.drop_points:>7.2f}p "
            f"{floor:>14} {interval:>18}"
        )

    missed = analysis.undetected
    if not missed:
        print("\nNo capability slice regressed beyond what the suite could resolve.")
        return EXIT_DETECTABLE
    print(
        f"\n{len(missed)} slices regressed past the suite's own floor without being "
        f"flagged, covering {len(analysis.maximal_undetected)} regions once nesting is "
        "removed."
    )
    print("Drops below are measured on the probe, not on the suite.")
    print(f"{'slice':<40} {'drop':>8} {'95% interval':>18} {'suite':>7} {'probe':>7}")
    for item in sorted(missed, key=lambda row: row.drop_points, reverse=True):
        interval = f"[{item.interval.low_points:+.1f}, {item.interval.high_points:+.1f}]"
        print(
            f"{item.name:<40} {item.drop_points:>7.2f}p {interval:>18} "
            f"{item.suite_cases:>7} {item.probe_cases:>7}"
        )
    return EXIT_NOT_DETECTABLE


def command_suite(arguments: argparse.Namespace) -> int:
    analysis, _ = _load(arguments)
    print(f"arm                   {analysis.arm}")
    print(f"suite cases           {analysis.suite_size}")
    print(f"probe cases           {analysis.probe_size}")
    print(f"named slices          {analysis.slices_total}")
    print(f"never sampled         {analysis.held_out_slices}")
    print(f"no usable power       {analysis.slices_without_suite_power}")
    print(f"accuracy before       {analysis.suite_accuracy_before:.4f}")
    print()
    print(f"{'depth':>5} {'suite drop':>11} {'discordance':>12} {'floor':>9} {'in region':>10}")
    for summary in analysis.depths:
        floor = "none" if summary.mde_points is None else f"{format_floor(summary.mde_points)}p"
        print(
            f"{summary.freeze_depth:>5} {summary.suite_drop_points:>10.2f}p "
            f"{summary.discordance:>12.4f} {floor:>9} "
            f"{summary.in_region_gain_points:>9.2f}p"
        )
    return EXIT_DETECTABLE


def command_curve(arguments: argparse.Namespace) -> int:
    analysis, rule = _load(arguments)
    discordance = analysis.operating.discordance
    print(f"arm {analysis.arm}, discordance {discordance:.4f}, power {rule.target_power:.0%}")
    print(f"{'cases':>7} {'exact':>12} {'normal':>12}")
    for row in suite_size_curve(discordance, CURVE_SIZES, rule):
        exact = row["exact_points"]
        normal = row["normal_points"]
        print(
            f"{row['suite_size']:>7} "
            f"{format_floor(exact, 3) + 'p' if exact is not None else 'unreachable':>12} "
            f"{format_floor(normal, 3) + 'p' if normal is not None else 'unreachable':>12}"
        )
    return EXIT_DETECTABLE


def command_report(arguments: argparse.Namespace) -> int:
    from finetune_eval_power.report import write_report

    analysis, rule = _load(arguments)
    destination = Path(arguments.out)
    write_report(analysis, rule, destination)
    print(f"wrote {destination}")
    return EXIT_DETECTABLE


def command_metrics(arguments: argparse.Namespace) -> int:
    analysis, _ = _load(arguments)
    operating = analysis.operating
    payload = {
        "arm": analysis.arm,
        "suite_size": analysis.suite_size,
        "probe_size": analysis.probe_size,
        "operating_depth": analysis.operating_depth,
        "suite_drop_points": round(operating.suite_drop_points, 4),
        "discordance": round(operating.discordance, 6),
        "mde_points": operating.mde_points,
        "undetected_slices": len(analysis.undetected),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return EXIT_DETECTABLE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="finetune-eval-power",
        description="Computes what a model eval suite could have detected, and what it missed.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--arm", default="vision", choices=sorted(ARMS))
        target.add_argument("--runs", default=None, help="directory holding the measurement")

    power = subparsers.add_parser("power", help="can this suite see a regression of this size")
    add_common(power)
    power.add_argument("--claim", type=float, default=DEFAULT_CLAIM_POINTS)
    power.set_defaults(handler=command_power)

    blind = subparsers.add_parser("blind", help="which capabilities regressed unnoticed")
    add_common(blind)
    blind.set_defaults(handler=command_blind)

    suite = subparsers.add_parser("suite", help="describe the suite, the probe and the depths")
    add_common(suite)
    suite.set_defaults(handler=command_suite)

    curve = subparsers.add_parser("curve", help="detectable effect against suite size")
    add_common(curve)
    curve.set_defaults(handler=command_curve)

    report = subparsers.add_parser("report", help="write the HTML report")
    add_common(report)
    report.add_argument("--out", default="reports/report.html")
    report.set_defaults(handler=command_report)

    metrics = subparsers.add_parser("metrics", help="print the headline figures as JSON")
    add_common(metrics)
    metrics.set_defaults(handler=command_metrics)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except Unanswerable as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        print(refusal.remedy, file=sys.stderr)
        return EXIT_REFUSED
    except ToolError as failure:
        print(f"error: {failure}", file=sys.stderr)
        return EXIT_TOOL_ERROR
    except Exception as unexpected:
        # A broad catch is right here: without it a crash exits 1, which this tool
        # has defined to mean "the answer is no". The escape hatch exists because
        # with it, a real bug looks like a user error during development.
        if os.environ.get(DEBUG_VARIABLE):
            raise
        print(f"error: {type(unexpected).__name__}: {unexpected}", file=sys.stderr)
        print(f"set {DEBUG_VARIABLE}=1 to re-raise with a traceback", file=sys.stderr)
        return EXIT_TOOL_ERROR


if __name__ == "__main__":  # pragma: no cover - exercised by the console script
    raise SystemExit(main())
