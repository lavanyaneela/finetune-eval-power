"""Tests for the receipts mechanism itself.

The receipts mechanism is the reason anything else in this repository can be
believed, so it gets the same treatment as the measurement: each rule is pinned by a
test, and the two rules that were found to be broken in practice are pinned by tests
that fail against the broken form.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checker = _load("check_numbers")


def test_an_anchor_matches_its_figure_with_the_separator_intact():
    """Pins the bug that made the checker useless. The pattern builder dropped the
    whitespace between the placeholder and the following word, so every numeric
    anchor compiled to a pattern demanding the digits and the word with nothing
    between them. Nothing matched, every document read clean, and it stayed that way
    until a figure was broken on purpose and the build passed."""
    pattern = checker.anchor_pattern("{} retrainings", numeric=True)
    assert [m.group(1) for m in pattern.finditer("over 10 retrainings")] == ["10"]


def test_the_broken_form_of_the_pattern_builder_would_match_nothing():
    """The second half of the same lesson: a regression test has to reconstruct the
    break and assert the old behaviour fails, or it passes against the bug it was
    written for."""
    import re

    def broken(fragment: str) -> str:
        return r"\s+".join(re.escape(word) for word in fragment.split())

    before, _, after = "{} retrainings".partition("{}")
    pattern = re.compile(rf"{broken(before)}(\d+){broken(after)}")
    assert pattern.search("over 10 retrainings") is None


def test_an_anchor_survives_a_line_break_in_wrapped_markdown():
    pattern = checker.anchor_pattern("detection floor of {}", numeric=True)
    assert pattern.search("the detection floor of\n3.41 points") is not None


def test_an_anchor_survives_bold_markers_around_its_figure():
    text = checker.EMPHASIS.sub("", "the suite **lost 10.58 points** here")
    assert checker.anchor_pattern("lost {} points", numeric=True).search(text)


def test_an_anchor_matches_when_the_sentence_capitalises_its_first_word():
    pattern = checker.anchor_pattern("below {} cases", numeric=True)
    assert pattern.search("Below 118 cases nothing helps") is not None


def test_a_figure_spelled_as_a_word_does_not_satisfy_a_numeric_anchor():
    """Numbers spelled out are a real blind spot: the guard reads the digits while
    the sentence beside it says something else entirely."""
    pattern = checker.anchor_pattern("{} retrainings", numeric=True)
    match = pattern.search("over ten retrainings")
    assert match is not None
    assert match.group(1) != "10"


def test_a_numeric_anchor_does_not_match_an_arbitrary_preceding_word():
    """An earlier placeholder matched any word, which turned an ordinary sentence
    into a stale reading of the suite size."""
    pattern = checker.anchor_pattern("{} eval cases", numeric=True)
    assert pattern.search("it errs toward buying eval cases") is None


def test_a_string_anchor_matches_a_name_rather_than_a_number():
    pattern = checker.anchor_pattern("worst is {}", numeric=False)
    match = pattern.search("the worst is shape=square today")
    assert match is not None and match.group(1) == "shape=square"


def test_an_anchor_with_no_placeholder_is_rejected():
    problems = checker.check_anchor_hygiene({"a": ["no placeholder here"]})
    assert any("exactly one" in problem for problem in problems)


def test_an_anchor_with_too_many_words_is_rejected():
    problems = checker.check_anchor_hygiene({"a": ["{} words in a very long phrase"]})
    assert any("straddle a line break" in problem for problem in problems)


def test_a_metric_with_no_anchor_at_all_is_rejected():
    assert any("no anchor" in problem for problem in checker.check_anchor_hygiene({"a": []}))


def test_two_metrics_sharing_an_anchor_string_are_rejected():
    problems = checker.check_anchor_hygiene({"a": ["{} tests"], "b": ["{} tests"]})
    assert any("overlaps" in problem for problem in problems)


def test_two_anchors_that_overlap_only_as_patterns_are_rejected():
    """Literal containment is not enough. These two share no substring and the
    second matches a sentence written for the first, which is how one metric's
    guard came to report another metric's value as stale."""
    payload = {
        "rendered": {"a": "3.41", "b": "3.92"},
        "anchors": {"a": ["floor of {} points"], "b": ["noise floor of {}"]},
    }
    assert checker.check_anchor_overlap(payload)


def test_anchors_that_do_not_overlap_are_accepted():
    payload = {
        "rendered": {"a": "3.41", "b": "3.92"},
        "anchors": {"a": ["detection floor of {}"], "b": ["noise floor of {}"]},
    }
    assert checker.check_anchor_overlap(payload) == []


def test_the_shipped_anchor_set_passes_its_own_hygiene_rules():
    payload = checker.load_metrics()
    assert checker.check_anchor_hygiene(payload["anchors"]) == []
    assert checker.check_anchor_overlap(payload) == []


def test_every_checked_document_exists():
    payload = checker.load_metrics()
    for relative in payload["checked_documents"]:
        assert (ROOT / relative).exists(), relative


def test_every_metric_is_quoted_by_at_least_one_checked_document():
    payload = checker.load_metrics()
    _problems, occurrences = checker.check_documents(payload)
    unquoted = [name for name, count in occurrences.items() if count == 0]
    assert unquoted == []


def test_the_documents_as_shipped_are_clean():
    payload = checker.load_metrics()
    problems, _ = checker.check_documents(payload)
    assert problems == []


def test_no_exemption_hides_a_measured_value_and_none_is_unused():
    payload = checker.load_metrics()
    assert checker.check_reverse(payload) == []


def test_a_corrupted_figure_is_caught(tmp_path, monkeypatch):
    """The whole point, as a test: change a published figure and the check fails.

    The corrupted string is derived from the anchor set rather than written out, so
    this test cannot quietly stop testing anything when a sentence is reworded.
    """
    payload = checker.load_metrics()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    corrupted = None
    for metric, phrases in payload["anchors"].items():
        value = payload["rendered"][metric]
        if not checker.NUMBER_PATTERN.fullmatch(value):
            continue
        for phrase in phrases:
            original = phrase.replace("{}", value)
            if original in readme:
                corrupted = readme.replace(original, phrase.replace("{}", "99999"))
                break
        if corrupted:
            break
    assert corrupted is not None, "no anchored figure was found in the README"

    (tmp_path / "README.md").write_text(corrupted, encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    problems, _ = checker.check_documents(dict(payload, checked_documents=["README.md"]))
    assert any("stale" in problem for problem in problems)


def test_the_duration_sweep_covers_untracked_files_too():
    """A new file is invisible to a tracked-only sweep until it is committed, and
    then it fails the next clone instead of this one."""
    sweep = _load("check_durations")
    names = [path.name for path in sweep.tracked_files()]
    assert "README.md" in names
    assert len(names) > 20


def test_the_repository_publishes_no_absolute_duration():
    sweep = _load("check_durations")
    assert sweep.main() == 0


@pytest.mark.parametrize("marker", ("1. ", "2) "))
def test_an_ordered_list_marker_is_not_read_as_a_figure(marker):
    assert checker.LIST_MARKER.sub(" ", f"{marker}a point") == " a point"


def test_the_relations_the_documents_assert_hold_against_the_measurement():
    """Anchors guard digits and are blind to the relation joining two of them. Two
    adversarial audits of this repository each found a false comparison sitting
    between two correctly guarded figures, which is what this table is for."""
    assert checker.check_relations(checker.load_metrics()) == []


def test_a_relation_naming_an_unknown_metric_is_reported_rather_than_skipped(monkeypatch):
    monkeypatch.setattr(checker, "RELATIONS", (("not_a_metric > 1", "reason"),))
    problems = checker.check_relations({"metrics": {"a": 1}})
    assert any("could not be evaluated" in problem for problem in problems)


def test_a_relation_that_is_false_is_reported(monkeypatch):
    monkeypatch.setattr(checker, "RELATIONS", (("a > b", "a must exceed b"),))
    problems = checker.check_relations({"metrics": {"a": 1, "b": 2}})
    assert problems and "a must exceed b" in problems[0]


def test_the_relation_table_covers_the_comparisons_the_headline_makes():
    expressions = " ".join(expression for expression, _reason in checker.RELATIONS)
    for metric in (
        "seed_noise_floor_points",
        "mde_points",
        "suite_worst_class_floor_points",
        "held_out_slices",
        "retraining_interval_points",
    ):
        assert metric in expressions, metric


def test_every_markdown_document_in_the_repository_is_a_checked_document():
    """`CHECKED_DOCUMENTS` is a hand-written tuple, so a new document joins the
    repository guarded by nothing. An audit added `docs/results-summary.md` full of
    invented figures and the build stayed green."""
    import subprocess

    listed = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=checker.ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    tracked = {name for name in listed if not name.startswith(("node_modules/", "reports/"))}
    checked = set(checker.load_metrics()["checked_documents"])
    assert tracked - checked == set(), (
        "these documents are tracked, publish prose, and nothing checks them: "
        f"{sorted(tracked - checked)}"
    )


def test_the_exit_code_table_is_pinned_as_text_and_matches_what_the_cli_returns():
    """The reverse sweep is a value allow-list: it asks whether a digit is one of
    the measured values, not whether it is the right one here. Every exit code but
    1 equals some measured value, so the published contract could be rewritten with
    the build green. It is pinned as literal text, and the literal text is checked
    against the codes the CLI actually uses, so the two cannot drift apart.
    """
    from finetune_eval_power import cli

    assert checker.check_literals() == []
    readme = (checker.ROOT / "README.md").read_text(encoding="utf-8")
    for code in (
        cli.EXIT_DETECTABLE,
        cli.EXIT_NOT_DETECTABLE,
        cli.EXIT_REFUSED,
        cli.EXIT_TOOL_ERROR,
    ):
        assert f"| `{code}` |" in readme


def test_a_pinned_line_that_is_edited_fails_the_build(tmp_path, monkeypatch):
    """The guard is only worth having if it fails when the contract is edited."""
    import shutil

    clone = tmp_path / "repo"
    clone.mkdir()
    shutil.copy(checker.ROOT / "README.md", clone / "README.md")
    text = (clone / "README.md").read_text(encoding="utf-8")
    (clone / "README.md").write_text(
        text.replace("| `2` | refused:", "| `7` | refused:"), encoding="utf-8"
    )
    monkeypatch.setattr(checker, "ROOT", clone)
    assert checker.check_literals() != []


def test_an_exemption_is_refused_in_a_document_it_is_not_allowed_in():
    """`80`, `95` and `1.96` were accepted in any sentence of any document, which
    let "the suite resolves 80 distinct capabilities" pass. An exemption now names
    where it is allowed."""
    for value, (_reason, allowed) in checker.REVERSE_EXEMPTIONS.items():
        assert allowed, f"exemption {value} allows no document, so it guards nothing"
        for name in allowed:
            assert name in checker.load_metrics()["checked_documents"]
