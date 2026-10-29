"""The command line contract: what each exit code means and when it is returned."""

from __future__ import annotations

import pytest
from tests.conftest import build_synthetic_runs

from finetune_eval_power.cli import (
    EXIT_DETECTABLE,
    EXIT_NOT_DETECTABLE,
    EXIT_REFUSED,
    EXIT_TOOL_ERROR,
    main,
)


@pytest.fixture
def runs_root(tmp_path, vision_arm):
    """A measurement whose retrainings disagree, which is the ordinary situation."""
    build_synthetic_runs(vision_arm, suite_drop_jitter=(4, 14, 26, 9, 19)).write(
        tmp_path / "vision"
    )
    return tmp_path


@pytest.fixture
def stable_runs_root(tmp_path, vision_arm):
    """A measurement where every retraining agrees, so nothing is refused for noise."""
    build_synthetic_runs(vision_arm, suite_drop_cases=24, rng_seed=1).write(tmp_path / "vision")
    return tmp_path


def run(arguments: list[str], root) -> int:
    return main([*arguments, "--runs", str(root)])


def test_a_large_claim_is_reported_as_detectable(stable_runs_root, capsys):
    code = run(["power", "--claim", "40"], stable_runs_root)
    assert code == EXIT_DETECTABLE
    assert "YES" in capsys.readouterr().out


def test_a_claim_below_the_floor_is_reported_as_not_detectable(stable_runs_root, capsys):
    code = run(["power", "--claim", "1.0"], stable_runs_root)
    captured = capsys.readouterr().out
    assert code == EXIT_NOT_DETECTABLE
    assert "NO" in captured


def test_a_not_detectable_answer_prices_the_suite_that_would_work(stable_runs_root, capsys):
    """Asserted `"cases" in out`, which the header line `suite cases 480` satisfies
    on every invocation including the YES path. This is the only test guarding the
    contract that code 1 names a size, so it has to read the size."""
    import re

    code = run(["power", "--claim", "1.0"], stable_runs_root)
    captured = capsys.readouterr().out
    assert code == EXIT_NOT_DETECTABLE
    priced = re.search(r"needs (\d+) cases", captured)
    assert priced, f"no suite size was printed under code 1:\n{captured}"
    assert int(priced.group(1)) > 0


def test_a_claim_smaller_than_the_retraining_noise_is_refused(runs_root, capsys):
    code = run(["power", "--claim", "0.05"], runs_root)
    captured = capsys.readouterr().out
    assert code == EXIT_REFUSED
    assert "REFUSED" in captured


def test_the_refusal_says_no_suite_size_fixes_it(runs_root, capsys):
    run(["power", "--claim", "0.05"], runs_root)
    assert "No suite size resolves this" in capsys.readouterr().out


def test_the_refusal_offers_the_remedy_that_would_actually_work(runs_root, capsys):
    """Asserted `"retrainings" in out`, which the header line `retrainings 5`
    satisfies on every invocation. The remedy is a count of retrainings to average,
    and the point of code 2 is that it offers one instead of a suite size."""
    import re

    code = run(["power", "--claim", "0.05"], runs_root)
    captured = capsys.readouterr().out
    assert code == EXIT_REFUSED
    offered = re.search(r"average over about (\d+) retrainings", captured)
    assert offered, f"the refusal named no remedy:\n{captured}"
    assert int(offered.group(1)) > 1
    assert "cases" not in captured.split("REFUSED")[1]


def test_refusal_is_checked_before_power_so_a_noise_problem_is_never_read_as_a_size_problem(
    runs_root, capsys
):
    """These two answers send a team in opposite directions. One says buy more eval
    cases, the other says the eval set is not the problem. Getting the order wrong
    would hand out the expensive wrong advice."""
    code = run(["power", "--claim", "0.05"], runs_root)
    assert code == EXIT_REFUSED
    assert "minimum detectable" not in capsys.readouterr().out


def test_the_blind_command_lists_what_the_suite_missed(tmp_path, vision_arm, capsys):
    """Asserted two strings the command prints whether or not anything was missed,
    so it passed on a measurement where nothing was. It has to name the slices."""
    from finetune_eval_power.analysis import analyse
    from finetune_eval_power.config import VISION_ARM
    from finetune_eval_power.runs import load_runs

    root = tmp_path
    build_synthetic_runs(
        vision_arm,
        suite_drop_cases=18,
        suite_gain_cases=20,
        held_out_drop_fraction=0.06,
        rng_seed=1,
    ).write(root / "vision")
    code = run(["blind"], root)
    captured = capsys.readouterr().out
    missed = analyse(VISION_ARM, load_runs(root / "vision")).undetected
    assert missed, "this measurement is built to be quiet and to have missed slices"
    assert code == EXIT_NOT_DETECTABLE
    for item in missed:
        assert item.name in captured, f"{item.name} was missed and is not listed"


def test_the_blind_command_reports_success_when_nothing_was_missed(tmp_path, vision_arm, capsys):
    build_synthetic_runs(vision_arm, held_out_drop_fraction=0.0).write(tmp_path / "vision")
    code = main(["blind", "--runs", str(tmp_path)])
    assert code == EXIT_DETECTABLE
    assert "No capability slice regressed" in capsys.readouterr().out


def test_the_suite_command_describes_every_freeze_depth(runs_root, capsys, vision_arm):
    run(["suite"], runs_root)
    captured = capsys.readouterr().out
    for depth in vision_arm.freeze_depths:
        assert f"\n{depth:>5}" in captured


def test_the_curve_command_prints_both_methods(runs_root, capsys):
    run(["curve"], runs_root)
    captured = capsys.readouterr().out
    assert "exact" in captured
    assert "normal" in captured


def test_the_metrics_command_emits_parseable_json(runs_root, capsys):
    import json

    run(["metrics"], runs_root)
    payload = json.loads(capsys.readouterr().out)
    assert payload["arm"] == "vision"
    assert "undetected_slices" in payload


def test_the_report_command_writes_the_file_it_names(tmp_path, runs_root, capsys):
    destination = tmp_path / "out" / "report.html"
    code = run(["report", "--out", str(destination)], runs_root)
    assert code == EXIT_DETECTABLE
    assert destination.exists()
    assert str(destination) in capsys.readouterr().out


def test_a_missing_measurement_is_a_tool_error_not_an_answer(tmp_path, capsys):
    code = main(["power", "--runs", str(tmp_path / "absent")])
    assert code == EXIT_TOOL_ERROR
    assert "error:" in capsys.readouterr().err


def test_an_unknown_arm_is_rejected_by_the_parser(runs_root):
    with pytest.raises(SystemExit):
        run(["power", "--arm", "olfactory"], runs_root)


@pytest.mark.parametrize(
    "arguments",
    [
        ["power", "--arm", "olfactory"],
        ["notacommand"],
        [],
        ["power", "--claim", "notanumber"],
    ],
)
def test_a_usage_error_exits_three_and_never_collides_with_the_refusal(arguments, runs_root):
    """The bug this pins: argparse exits 2 on `parser.error`, and 2 is this tool's
    "refused" code, the single result the whole repository is about.

    A mistyped arm name therefore produced the headline exit code, and the defense
    guide tells the candidate to demonstrate that code with `echo $?` in front of an
    interviewer. The Makefile carries a comment warning about exactly this collision
    for `make`, while the console script had it. Nothing caught it because the test
    above asserts only that SystemExit is raised, never which code it carries.
    """
    with pytest.raises(SystemExit) as raised:
        run(arguments, runs_root)
    assert raised.value.code == EXIT_TOOL_ERROR


def test_that_usage_errors_are_not_merely_nonzero(runs_root):
    """The companion to the test above: it would pass against the bug if it only
    asserted a non-zero code, because the bug returned 2, which is non-zero."""
    with pytest.raises(SystemExit) as raised:
        run(["power", "--arm", "olfactory"], runs_root)
    assert raised.value.code != EXIT_REFUSED


def test_a_suite_whose_cases_never_change_verdict_is_refused_not_answered(
    tmp_path, vision_arm, capsys
):
    """Code 1 promises the suite size that would work. At a discordance of zero
    there is none, so the branch that prints no size now returns the refusal code
    instead of the answer code. It is the second route into "no suite size fixes
    this", the first being retraining noise."""
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=0, held_out_drop_fraction=0.0)
    runs.write(tmp_path / "vision")
    code = run(["power", "--claim", "2"], tmp_path)
    captured = capsys.readouterr().out
    assert code == EXIT_REFUSED
    assert "REFUSED" in captured
    assert "no suite size to recommend" in captured


def test_blind_does_not_claim_nothing_regressed_when_the_suite_was_loud(
    tmp_path, vision_arm, capsys
):
    """`undetected` is empty for two different reasons and they printed the same
    sentence. On a measurement where the suite flags every retraining, nothing is
    hidden from it, but "no capability slice regressed" is false: classes can be
    several times past their own floor. The question the command answers only
    arises while the suite is quiet, and it now says so."""
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=90, held_out_drop_fraction=0.6)
    runs.write(tmp_path / "vision")
    code = run(["blind"], tmp_path)
    captured = capsys.readouterr().out
    assert code == EXIT_DETECTABLE
    assert "No capability slice regressed" not in captured
    assert "nothing here is hidden from it" in captured


def test_a_drop_of_exactly_zero_is_not_described_as_a_fall(tmp_path, vision_arm, capsys):
    """An unsigned noun over a signed number is how a reader inverts a conclusion,
    which the repository is careful about everywhere except at zero, where the
    branch read `accuracy rose if drop < 0 else accuracy fell`."""
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=0, held_out_drop_fraction=0.0)
    runs.write(tmp_path / "vision")
    run(["blind"], tmp_path)
    captured = capsys.readouterr().out
    assert "accuracy unchanged" in captured
    assert "accuracy fell" not in captured


def test_an_unexpected_failure_becomes_a_tool_error_rather_than_a_no(
    monkeypatch, runs_root, capsys
):
    """Without the broad catch a crash exits 1, which this tool has defined to mean
    the answer is no. That is the worst possible confusion for this particular tool."""
    import finetune_eval_power.cli as cli

    def explode(_arguments):
        raise RuntimeError("something unforeseen")

    monkeypatch.setattr(cli, "command_power", explode)
    code = main(["power", "--runs", str(runs_root)])
    assert code == EXIT_TOOL_ERROR
    assert "something unforeseen" in capsys.readouterr().err


def test_the_debug_variable_re_raises_so_a_real_bug_is_not_disguised(monkeypatch, runs_root):
    import finetune_eval_power.cli as cli

    def explode(_arguments):
        raise RuntimeError("something unforeseen")

    monkeypatch.setattr(cli, "command_power", explode)
    monkeypatch.setenv(cli.DEBUG_VARIABLE, "1")
    with pytest.raises(RuntimeError, match="something unforeseen"):
        main(["power", "--runs", str(runs_root)])


def test_an_unanswerable_raised_deeper_down_still_exits_as_refused(monkeypatch, runs_root, capsys):
    import finetune_eval_power.cli as cli
    from finetune_eval_power.errors import Unanswerable

    def refuse(_arguments):
        raise Unanswerable("cannot be answered", remedy="collect more retrainings")

    monkeypatch.setattr(cli, "command_power", refuse)
    assert main(["power", "--runs", str(runs_root)]) == EXIT_REFUSED
    assert "collect more retrainings" in capsys.readouterr().err


def test_the_four_exit_codes_are_distinct_values():
    codes = {EXIT_DETECTABLE, EXIT_NOT_DETECTABLE, EXIT_REFUSED, EXIT_TOOL_ERROR}
    assert codes == {0, 1, 2, 3}


def test_the_version_flag_prints_the_package_version(capsys):
    from finetune_eval_power import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize("claim", ["nan", "inf", "-inf", "0", "-5", "abc"])
def test_a_claim_that_is_not_a_finite_positive_number_is_a_usage_error(claim, runs_root):
    """`float("nan")` parses, and every comparison downstream is False for it, so
    `--claim nan` walked past the positivity guard, past the refusal rule and past
    the power calculation and printed "a nan point regression is below this suite's
    floor" with exit code 1. Code 1 promises the suite size that would work."""
    with pytest.raises(SystemExit) as raised:
        run(["power", "--claim", claim], runs_root)
    assert raised.value.code == EXIT_TOOL_ERROR


def test_closing_the_pipe_early_does_not_produce_a_documented_exit_code():
    """`blind | head` used to exit 0, which is pinned as meaning nothing fell past
    the suite's floor unnoticed, on a measurement with eight missed slices. The
    codes are the product here, so none of them may be produced by something that
    is not an answer.

    The reader's end of the pipe is closed before the command starts, rather than
    by racing a `head -3` against the command's final flush. That race is what the
    first version of this test did, and it is a race the test loses on a fast
    machine: the whole output is well under the pipe buffer, so if `head` has not
    exited yet the write simply succeeds and the command exits 1, honestly. It
    passed here and failed on every CI runner, which is the worst way round.
    Closing the read end up front makes the first write fail whatever the timing.
    """
    import os
    import signal
    import subprocess
    import sys

    read_end, write_end = os.pipe()
    process = subprocess.Popen(
        [sys.executable, "-m", "finetune_eval_power.cli", "blind", "--arm", "vision"],
        stdout=write_end,
        stderr=subprocess.DEVNULL,
    )
    os.close(write_end)
    os.close(read_end)
    process.wait()
    assert process.returncode not in (0, 1, 2, 3), (
        f"a closed pipe returned {process.returncode}, which is one of the four "
        "documented codes and therefore says something false"
    )
    # Negative when Python reports the signal directly, 128 + n through a shell.
    assert process.returncode in (-signal.SIGPIPE, 128 + signal.SIGPIPE)


def test_a_claim_needing_more_cases_than_the_search_covers_is_refused(tmp_path, vision_arm, capsys):
    """The third route into code 2, and the only line in the CLI with no test.

    A suite whose cases change verdict often has a floor, and a claim far below
    that floor needs a suite larger than the search ceiling. There is then no size
    to print, which is what separates a refusal from code 1, and the branch
    returned 1 until an audit reached it. The pinned exit-code row named two routes
    while the code had three.
    """
    runs = build_synthetic_runs(
        vision_arm, suite_drop_cases=60, suite_gain_cases=60, held_out_drop_fraction=0.0
    )
    runs.write(tmp_path / "vision")
    code = run(["power", "--claim", "0.001"], tmp_path)
    captured = capsys.readouterr().out
    assert code == EXIT_REFUSED, captured
    assert "no suite size this tool will search" in captured
    assert "needs" not in captured.split("REFUSED")[1]


def test_a_claim_too_small_to_print_is_still_printed_exactly(tmp_path, vision_arm, capsys):
    """`{claim:.2f}` rendered a claim of 0.001 as "a 0.00 point regression", which
    is a figure the user did not type and cannot act on."""
    build_synthetic_runs(vision_arm, suite_drop_cases=24, rng_seed=1).write(tmp_path / "vision")
    run(["power", "--claim", "0.001"], tmp_path)
    captured = capsys.readouterr().out
    assert "0.00 point" not in captured
    assert "0.001" in captured
