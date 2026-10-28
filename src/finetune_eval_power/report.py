"""The HTML report, which exists so the evidence in the README is a photograph of
output rather than a drawing of it.

Two constraints shape it. Section headings carry stable ids, because the screenshot
tool frames by heading and a hard coded pixel crop silently drifts the first time a
paragraph gets longer. And every CSS class used in the markup must have a rule in
the stylesheet, which a test asserts: on two earlier projects a badge class and a
table cell class collided, the later rule won, and the headline verdict rendered
green on green while every test passed.
"""

from __future__ import annotations

import html
from pathlib import Path

from finetune_eval_power.analysis import ArmAnalysis
from finetune_eval_power.config import DecisionRule
from finetune_eval_power.formatting import format_floor

STYLESHEET = """
:root { color-scheme: light; }
body { font-family: ui-sans-serif, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 56rem; padding: 2rem 1.5rem 4rem; color: #16181d;
       background: #ffffff; line-height: 1.5; }
h1 { font-size: 1.6rem; margin: 0 0 0.25rem; }
h2 { font-size: 1.15rem; margin: 2.25rem 0 0.75rem; padding-bottom: 0.3rem;
     border-bottom: 1px solid #d8dbe0; }
p { margin: 0.6rem 0; }
.subtitle { color: #5a6069; margin: 0 0 1.25rem; }
.badge { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 0.3rem;
         font-weight: 700; font-size: 0.85rem; letter-spacing: 0.02em;
         color: #ffffff; background: #8a1c1c; }
.badge-quiet { background: #8a1c1c; color: #ffffff; }
.badge-loud { background: #1d4d2b; color: #ffffff; }
.figures { display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0 0.5rem; }
.figure { border: 1px solid #d8dbe0; border-radius: 0.4rem; padding: 0.75rem 1rem;
          min-width: 11rem; }
.figure-value { font-size: 1.5rem; font-weight: 700; display: block; }
.figure-label { color: #5a6069; font-size: 0.8rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; margin: 0.75rem 0; }
th, td { border-bottom: 1px solid #e3e5e9; padding: 0.38rem 0.5rem; text-align: left; }
th { background: #f2f4f7; font-weight: 600; }
td.number { text-align: right; font-variant-numeric: tabular-nums; }
tr.missed td { background: #fdf2f2; }
.note { color: #5a6069; font-size: 0.85rem; margin-top: 0.5rem; }
.footer { margin-top: 3rem; color: #5a6069; font-size: 0.8rem; }
"""


def _escape(value: object) -> str:
    return html.escape(str(value))


def _points(value: float | None, digits: int = 2) -> str:
    """Every quantity rendered through this helper is a minimum, so it rounds down.

    Shared with the metrics collector. A report that rounded differently from the
    published figure would put two numbers for one quantity in front of a reader,
    and the screenshot of this report goes in the README.
    """
    return format_floor(value, digits)


def _figure(value: str, label: str) -> str:
    return (
        f'<div class="figure"><span class="figure-value">{_escape(value)}</span>'
        f'<span class="figure-label">{_escape(label)}</span></div>'
    )


def render_report(analysis: ArmAnalysis, rule: DecisionRule) -> str:
    """Build the whole report as one self contained HTML document."""
    operating = analysis.operating
    missed = analysis.undetected
    worst = analysis.worst_undetected
    quiet = operating.suite_is_quiet
    badge_class = "badge badge-quiet" if quiet else "badge badge-loud"
    badge_text = "SUITE RAISED NOTHING" if quiet else "SUITE FLAGGED A REGRESSION"

    # The size each underpowered covered slice would need, so the note under the
    # table can promise one and the table can keep it. "unreachable" on its own
    # read as "nothing would help", which is the misreading this column exists to
    # correct, and it was still doing it after the note was rewritten.
    needed_cases = {name: needed for name, _cases, needed in analysis.slices_short_of_power}

    rows = []
    for item in sorted(analysis.findings, key=lambda row: row.drop_points, reverse=True):
        classes = ' class="missed"' if item in missed else ""
        needed = needed_cases.get(item.name)
        floor_cell = (
            _points(item.suite_mde_on_slice_points)
            if item.suite_has_power
            else (f"needs {needed} cases" if needed else "no suite size")
        )
        rows.append(
            f"<tr{classes}><td>{_escape(item.name)}</td>"
            f'<td class="number">{item.suite_cases}</td>'
            f'<td class="number">{item.probe_cases}</td>'
            f'<td class="number">{item.drop_points:+.2f}</td>'
            f'<td class="number">{item.interval.low_points:+.2f} to '
            f"{item.interval.high_points:+.2f}</td>"
            f'<td class="number">{floor_cell}</td></tr>'
        )

    class_rows = "".join(
        f"<tr><td>{_escape(item.name)}</td>"
        f'<td class="number">{item.suite_cases}</td>'
        f'<td class="number">{item.drop_points:+.2f}</td>'
        f'<td class="number">{_points(item.floor_points)}</td>'
        f'<td class="number">{item.interval_low_points:+.2f} to '
        f"{item.interval_high_points:+.2f}</td></tr>"
        for item in analysis.suite_class_findings
    )

    depth_rows = "".join(
        f'<tr><td class="number">{summary.freeze_depth}</td>'
        f'<td class="number">{summary.suite_drop_points:+.2f}</td>'
        f'<td class="number">{summary.discordance:.4f}</td>'
        f'<td class="number">{_points(summary.mde_points)}</td>'
        f'<td class="number">{summary.in_region_gain_points:+.2f}</td>'
        f'<td class="number">{summary.seeds_flagging_regression} of {summary.seeds}</td></tr>'
        for summary in analysis.depths
    )

    if worst is None:
        worst_sentence = "No capability slice regressed past the suite's own floor unflagged."
    else:
        worst_sentence = (
            f"The worst unflagged regression is {_escape(worst.name)} at "
            f"{worst.drop_points:.2f} points."
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eval suite power report: {_escape(analysis.arm)}</title>
<style>{STYLESHEET}</style></head>
<body>
<h1>Eval suite power report</h1>
<p class="subtitle">Arm {_escape(analysis.arm)}, {len(analysis.seeds)} retrainings,
freeze depth {analysis.operating_depth}, the shallowest the suite does not object to.</p>
<p><span class="{badge_class}">{_escape(badge_text)}</span></p>

<h2 id="overview">What the suite reported</h2>
<div class="figures">
{
        _figure(
            f"{operating.suite_drop_points:+.2f}",
            "suite accuracy drop, points. Negative means accuracy rose",
        )
    }
{
        _figure(
            f"{operating.seeds_flagging_regression} of {operating.seeds}",
            "retrainings where the suite flagged a regression",
        )
    }
{
        _figure(
            _points(operating.mde_points),
            "smallest drop the suite could see, points",
        )
    }
{_figure(f"{analysis.probe_change_points:+.2f}", "the same drop, measured by the referee")}
{_figure(str(len(missed)), "slices that moved further than the floor, unflagged")}
</div>
<p>The suite holds {analysis.suite_size} cases and reports accuracy
{analysis.suite_accuracy_before:.4f} before the fine-tune. The referee probe holds
{analysis.probe_size} cases and enumerates the attribute space, so every slice below has an
exact denominator rather than a sampled one.</p>

<h2 id="detection-floor">What the suite could have seen</h2>
<p>At a measured discordance of {operating.discordance:.4f}, the smallest regression this
suite reaches {rule.target_power:.0%} power against is {_points(operating.mde_points)} points.
Of {analysis.slices_total} named capability slices the suite has no usable power on
{analysis.slices_without_suite_power}: {analysis.held_out_slices} it never samples at all, and
{analysis.underpowered_covered_slices} it samples too thinly for any effect size to reach that
power at the current suite size. The second group would be helped by a larger suite. The first
would not, because no suite size gives a test cases it does not have.</p>
<p>The suite measures a drop of {operating.suite_drop_points:+.2f} points over its own cases,
and a drop is positive when the model gets worse, so that is an improvement. The probe,
enumerating the whole space, measures {analysis.probe_change_points:+.2f}. Measured on the
suite's own cases the classes move against each other, which is why the aggregate is quiet.
The last two columns are the point: each class is read against the floor that applies at its
own case count, and the worst one does not clear it, so reporting per class would not have
rescued this suite either.</p>
<table><thead><tr><th>label class</th><th>suite cases</th><th>drop, points</th>
<th>its own floor</th><th>95% interval over retrainings</th></tr></thead>
<tbody>{class_rows}</tbody></table>

<h2 id="missed-capabilities">What it missed</h2>
<p>{worst_sentence}
Rows are shaded where the slice moved further than the suite could resolve while the suite
raised nothing.</p>
<table><thead><tr><th>capability slice</th><th>suite cases</th><th>probe cases</th>
<th>drop, points</th><th>95% interval over retrainings</th>
<th>suite floor on this slice</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="note">The last column reads three ways. A number is the smallest regression the
suite could resolve on that slice, at its own case count. "needs N cases" means the suite
samples the slice but too thinly to reach that power at its current size, and N is the size
that would: those are fixable by a larger suite, and the size is computed from the slice's
rate of changed verdicts on the suite's own cases. "no suite size" means the suite holds no
cases of the slice, or none that ever change verdict, and no larger suite drawn the same way
would help. The three look alike in a table and have different remedies.</p>

<h2 id="freeze-depth">What freezing buys and what it costs</h2>
<table><thead><tr><th>freeze depth</th><th>suite drop, points</th><th>discordance</th>
<th>suite floor, points</th><th>in-region gain, points</th>
<th>retrainings flagging</th></tr></thead>
<tbody>{depth_rows}</tbody></table>

<h2 id="noise-sources">Which uncertainty actually dominates</h2>
<p>Both figures below describe one measurement, which is the only way they can be compared.
Resampling the eval cases of a single retraining puts a 95% interval
{analysis.suite_case_interval_width_points:.2f} points wide around the suite's reading.
Retraining the same configuration, with the eval cases held fixed, puts one
{analysis.retraining_interval_width_points:.2f} points wide around it. A larger eval suite
shrinks the first and leaves the second alone, which is why this tool checks the second before
it recommends anything about the first.</p>
<p>Retraining moves the measured suite regression with a standard deviation of
{analysis.variance.standard_deviation_points:.3f} points over
{analysis.variance.seeds} retrainings. The refusal threshold is the half width of that
interval, {analysis.variance.noise_floor_points:.2f} points, so any claimed effect below it is
refused rather than answered.</p>

<p class="footer">Generated by the finetune-eval-power report command from the committed
measurement. Every figure on this page is re-derived from that measurement on each run.</p>
</body></html>
"""


def write_report(analysis: ArmAnalysis, rule: DecisionRule, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(analysis, rule), encoding="utf-8")
    return destination
