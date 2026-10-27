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

    rows = []
    for item in sorted(analysis.findings, key=lambda row: row.drop_points, reverse=True):
        classes = ' class="missed"' if item in missed else ""
        rows.append(
            f"<tr{classes}><td>{_escape(item.name)}</td>"
            f'<td class="number">{item.suite_cases}</td>'
            f'<td class="number">{item.probe_cases}</td>'
            f'<td class="number">{item.drop_points:+.2f}</td>'
            f'<td class="number">{item.interval.low_points:+.2f} to '
            f"{item.interval.high_points:+.2f}</td>"
            f'<td class="number">{_points(item.suite_mde_on_slice_points)}</td></tr>'
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
{_figure(f"{operating.suite_drop_points:+.2f}", "suite accuracy change, points")}
{
        _figure(
            f"{operating.seeds_flagging_regression} of {operating.seeds}",
            "retrainings where the suite flagged a regression",
        )
    }
{_figure(_points(operating.mde_points), "smallest drop the suite could see, points")}
{_figure(str(len(missed)), "capabilities that fell further than that, unflagged")}
</div>
<p>The suite holds {analysis.suite_size} cases and reports accuracy
{analysis.suite_accuracy_before:.4f} before the fine-tune. The referee probe holds
{analysis.probe_size} cases and enumerates the attribute space, so every slice below has an
exact denominator rather than a sampled one.</p>

<h2 id="detection-floor">What the suite could have seen</h2>
<p>At a measured discordance of {operating.discordance:.4f}, the smallest regression this
suite reaches {rule.target_power:.0%} power against is {_points(operating.mde_points)} points.
Of {analysis.slices_total} named capability slices, {analysis.held_out_slices} are never
sampled by the suite at all and {analysis.slices_without_suite_power} have no suite size at
which the suite reaches that power.</p>

<h2 id="missed-capabilities">What it missed</h2>
<p>{worst_sentence}
Rows are shaded where the slice moved further than the suite could resolve while the suite
raised nothing.</p>
<table><thead><tr><th>capability slice</th><th>suite cases</th><th>probe cases</th>
<th>drop, points</th><th>95% interval over retrainings</th>
<th>suite floor on this slice</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table>
<p class="note">A slice with zero suite cases has no suite floor, because no suite size gives
a test any cases to work with.</p>

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
{analysis.variance.seeds} retrainings, so any claimed effect below
{analysis.variance.noise_floor_points:.2f} points is refused rather than answered.</p>

<p class="footer">Generated by the finetune-eval-power report command from the committed
measurement. Every figure on this page is re-derived from that measurement on each run.</p>
</body></html>
"""


def write_report(analysis: ArmAnalysis, rule: DecisionRule, destination: Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_report(analysis, rule), encoding="utf-8")
    return destination
