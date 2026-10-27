#!/usr/bin/env python3
"""Photographs the real report, framed by heading rather than by pixel offset.

Three rules, each from a failure that shipped.

**Frame by heading.** Scroll to a named `h2`, read its document offset, and clip from
there to the next named heading. A hard coded crop drifts silently the first time a
paragraph gets longer, and then the README shows half a table. The run fails when a
named heading is missing, because that means the report changed shape and the shot
would be meaningless.

**`full_page` stays on even with a clip.** Without it Chromium clamps the clip to the
viewport and every shot comes out one viewport tall with its bottom missing.

**Read back the badge's computed style and fail if it is invisible.** On two earlier
projects a badge class and a table cell class collided at equal specificity, the later
rule won, and the headline verdict rendered green on green. The Python was right, the
HTML was right, the tests passed. A screenshot tool is the only thing in the pipeline
that can see pixels, so this is where that check belongs.

The report is rebuilt before it is photographed, so a stale report cannot be shot.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUTPUT = ROOT / "docs" / "evidence"
REPORTS = ROOT / "reports"
WIDTH = 1180
SHOTS = (
    ("overview", "detection-floor", "suite-verdict.png"),
    ("missed-capabilities", "freeze-depth", "missed-capabilities.png"),
    ("noise-sources", None, "noise-sources.png"),
)


def build_report(arm: str) -> Path:
    from finetune_eval_power.analysis import analyse
    from finetune_eval_power.config import ARMS, RUNS_DIRECTORY, DecisionRule
    from finetune_eval_power.report import write_report
    from finetune_eval_power.runs import load_runs

    configuration = ARMS[arm]
    runs = load_runs(ROOT / RUNS_DIRECTORY / arm)
    REPORTS.mkdir(parents=True, exist_ok=True)
    destination = REPORTS / f"{arm}.html"
    write_report(analyse(configuration, runs), DecisionRule(), destination)
    return destination


def main() -> int:
    from playwright.sync_api import sync_playwright

    report = build_report("vision")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"report": str(report.relative_to(ROOT)), "shots": []}

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": 900})
        page.route(
            "**/*",
            lambda route: (
                route.abort() if route.request.url.startswith("http") else route.continue_()
            ),
        )
        page.goto(f"file://{report.resolve()}")

        badge = page.evaluate(
            """() => {
                const element = document.querySelector('.badge');
                if (!element) return null;
                const style = getComputedStyle(element);
                return {
                    text: element.textContent.trim(),
                    colour: style.color,
                    background: style.backgroundColor,
                    visible: element.offsetWidth > 0 && element.offsetHeight > 0,
                };
            }"""
        )
        if badge is None or not badge["visible"]:
            raise SystemExit("the verdict badge is missing or has no box; refusing to publish")
        if badge["colour"] == badge["background"]:
            raise SystemExit(
                f"the verdict badge renders {badge['colour']} on {badge['background']}, "
                "so the headline is invisible; refusing to publish"
            )
        manifest["badge"] = badge

        for start_id, end_id, filename in SHOTS:
            geometry = page.evaluate(
                """([startId, endId]) => {
                    const start = document.getElementById(startId);
                    if (!start) return null;
                    const end = endId ? document.getElementById(endId) : null;
                    if (endId && !end) return null;
                    const top = start.getBoundingClientRect().top + window.scrollY - 24;
                    const bottom = end
                        ? end.getBoundingClientRect().top + window.scrollY - 16
                        : document.body.scrollHeight;
                    return { top, height: bottom - top };
                }""",
                [start_id, end_id],
            )
            if geometry is None:
                raise SystemExit(
                    f"heading {start_id!r} or {end_id!r} is gone, so the report has changed "
                    "shape and this screenshot would be meaningless"
                )
            destination = OUTPUT / filename
            page.screenshot(
                path=str(destination),
                full_page=True,
                clip={
                    "x": 0,
                    "y": geometry["top"],
                    "width": WIDTH,
                    "height": min(geometry["height"], 2400),
                },
            )
            manifest["shots"].append(
                {
                    "file": filename,
                    "framed_from": start_id,
                    "framed_to": end_id,
                    "bytes": destination.stat().st_size,
                }
            )
            print(f"wrote {destination} ({destination.stat().st_size} bytes)")
        browser.close()

    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
