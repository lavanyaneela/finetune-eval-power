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

import hashlib
import json
import re
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


#: The WCAG AA ratio for large text. The guard used to be string equality of the
#: two computed colours, which is unreachable for any stylesheet this report can
#: emit, so a low-contrast class collision would have published silently.
MINIMUM_CONTRAST = 3.0


def _channel(value: float) -> float:
    value /= 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _luminance(colour: str) -> float:
    numbers = [float(part) for part in re.findall(r"[\d.]+", colour)[:3]]
    if len(numbers) != 3:
        return 0.0
    red, green, blue = (_channel(part) for part in numbers)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _contrast(foreground: str, background: str) -> float:
    first, second = _luminance(foreground), _luminance(background)
    high, low = max(first, second), min(first, second)
    return (high + 0.05) / (low + 0.05)


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
        if _contrast(badge["colour"], badge["background"]) < MINIMUM_CONTRAST:
            raise SystemExit(
                f"the verdict badge renders {badge['colour']} on {badge['background']}, "
                "which is too little contrast to read; refusing to publish"
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
                    # A byte count can be reconciled by hand after swapping one
                    # shot for another; a digest cannot, without rerunning this.
                    "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
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
