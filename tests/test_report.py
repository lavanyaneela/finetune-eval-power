"""Properties of the HTML report, including the one that is really a CSS test."""

from __future__ import annotations

import html
import re

from tests.conftest import build_synthetic_runs

from finetune_eval_power.analysis import analyse
from finetune_eval_power.config import DecisionRule
from finetune_eval_power.report import STYLESHEET, render_report, write_report

HEADING_IDS = (
    "overview",
    "detection-floor",
    "missed-capabilities",
    "freeze-depth",
    "noise-sources",
)


def report_html(vision_arm, **kwargs) -> str:
    runs = build_synthetic_runs(vision_arm, **kwargs)
    return render_report(analyse(vision_arm, runs), DecisionRule())


def test_the_report_is_a_complete_html_document(vision_arm):
    html = report_html(vision_arm)
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")


def test_every_css_class_used_in_the_markup_has_a_rule_in_the_stylesheet(vision_arm):
    """A real failure twice over on earlier projects: a badge class and a table cell
    class collided at equal specificity, the later rule won, and the headline verdict
    rendered green on green. The Python was right, the HTML was right, and the tests
    passed. This is the test that would have caught it."""
    html = report_html(vision_arm)
    body = html.split("</style>", 1)[1]
    used: set[str] = set()
    for attribute in re.findall(r'class="([^"]+)"', body):
        used.update(attribute.split())
    declared = set(re.findall(r"\.([a-zA-Z][\w-]*)", STYLESHEET))
    assert used <= declared, sorted(used - declared)


def test_every_section_the_screenshot_tool_frames_by_is_present(vision_arm):
    html = report_html(vision_arm)
    for heading_id in HEADING_IDS:
        assert f'id="{heading_id}"' in html, heading_id


def test_the_verdict_badge_is_present_and_carries_a_state_class(vision_arm):
    html = report_html(vision_arm)
    assert 'class="badge badge-quiet"' in html or 'class="badge badge-loud"' in html


def test_a_quiet_suite_and_a_loud_suite_get_different_badges(vision_arm):
    quiet = report_html(vision_arm, suite_drop_cases=2)
    loud = report_html(vision_arm, suite_drop_cases=160)
    assert "badge-quiet" in quiet
    assert "badge-loud" in loud


def test_the_badge_colours_are_not_the_same_as_the_surface_behind_them():
    """Both badge states must differ from the page and the table backgrounds, which
    is the collision that produced an invisible verdict on an earlier build."""
    page_colours = {"#ffffff", "#f2f4f7", "#fdf2f2"}
    for state in ("badge-quiet", "badge-loud"):
        rule = STYLESHEET.split(f".{state}", 1)[1].split("}", 1)[0]
        background = re.search(r"background:\s*(#[0-9a-fA-F]{6})", rule).group(1)
        assert background.lower() not in page_colours


def test_every_named_slice_appears_as_a_row(vision_arm):
    runs = build_synthetic_runs(vision_arm)
    analysis = analyse(vision_arm, runs)
    rendered = render_report(analysis, DecisionRule())
    for finding in analysis.findings:
        assert html.escape(finding.name) in rendered, finding.name


def test_missed_slices_are_shaded_and_unmissed_ones_are_not(vision_arm):
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=2, held_out_drop_fraction=0.3)
    analysis = analyse(vision_arm, runs)
    html = render_report(analysis, DecisionRule())
    assert html.count('<tr class="missed">') == len(analysis.undetected)


def test_a_slice_the_suite_never_samples_shows_no_floor(vision_arm):
    html = report_html(vision_arm)
    assert "unreachable" in html


def test_slice_names_are_escaped_so_markup_cannot_be_injected(vision_arm):
    runs = build_synthetic_runs(vision_arm)
    analysis = analyse(vision_arm, runs)
    hostile = analysis.findings[0].__class__(
        **{**analysis.findings[0].__dict__, "name": "<script>alert(1)</script>"}
    )
    patched = analysis.__class__(
        **{**analysis.__dict__, "findings": (hostile, *analysis.findings[1:])}
    )
    html = render_report(patched, DecisionRule())
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_writing_the_report_creates_the_parent_directory(tmp_path, vision_arm):
    runs = build_synthetic_runs(vision_arm)
    destination = tmp_path / "nested" / "report.html"
    write_report(analyse(vision_arm, runs), DecisionRule(), destination)
    assert destination.exists()
    assert destination.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_the_report_compares_the_two_noise_sources_on_a_single_measurement(vision_arm):
    """Both widths must describe one measurement. Comparing a bootstrap over cases
    against a bootstrap over the mean of several retrainings is not like for like:
    the second shrinks with the number of retrainings, which makes retraining noise
    look smaller the more of it you measure. An earlier version of this section did
    exactly that and stated the comparison backwards."""
    rendered = report_html(vision_arm)
    assert "Both figures below describe one measurement" in rendered
    assert "Retraining the same configuration, with the eval cases held fixed" in rendered
