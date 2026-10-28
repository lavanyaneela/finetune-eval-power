#!/usr/bin/env python3
"""Renders the README's diagram source to a committed SVG.

GitHub does not reliably render a mermaid fence: it has been observed failing from
inside its own lazily loaded renderer, where there is nothing to fix from the
repository. So the diagram is rendered here, committed, and embedded as an image.

Four details, each from a rendering that shipped broken:

- `htmlLabels` is off. With it on, mermaid puts HTML inside a `foreignObject`,
  including unclosed break tags, so the file is not well formed XML. It still
  displays when injected into a live page and fails silently as an `img` source.
- The root element gets explicit width and height from the viewBox. Mermaid emits a
  percentage width, which gives a browser no aspect ratio to work with.
- A light background rectangle is painted. Transparent sounds theme neutral and is
  not: subgraph titles and edge labels have no fill behind them and come out dark
  grey on near black in a dark theme.
- The library is attached as a script tag from disk rather than inlined, because
  inlining several megabytes into a scratch page times out unpredictably.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "evidence"
FENCE = re.compile(r"```mermaid-source\n(.*?)```", re.S)
LIBRARY_CANDIDATES = (
    ROOT / "node_modules" / "mermaid" / "dist" / "mermaid.min.js",
    Path("/opt/npm-tools/node_modules/mermaid/dist/mermaid.min.js"),
)
BACKGROUND = "#ffffff"


def find_library() -> Path:
    for candidate in LIBRARY_CANDIDATES:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "mermaid was not found. Install it with `npm install mermaid` in this "
        "directory, or point LIBRARY_CANDIDATES at an existing copy. The committed "
        "SVG is unchanged; nothing was overwritten."
    )


def read_source() -> str:
    match = FENCE.search((ROOT / "README.md").read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit("no fence labelled mermaid-source in README.md")
    return match.group(1).strip()


def give_intrinsic_size(svg: str) -> str:
    view_box = re.search(r'viewBox="([\d.\s-]+)"', svg)
    if view_box is None:
        raise SystemExit("the rendered SVG has no viewBox, so it cannot be sized")
    _, _, width, height = (float(value) for value in view_box.group(1).split())
    svg = re.sub(r'\swidth="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sheight="[^"]*"', "", svg, count=1)
    svg = re.sub(r'\sstyle="[^"]*"', "", svg, count=1)
    svg = svg.replace("<svg ", f'<svg width="{width:.0f}" height="{height:.0f}" ', 1)
    rectangle = f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="{BACKGROUND}"/>'
    return svg.replace(">", ">" + rectangle, 1)


def main() -> int:
    from playwright.sync_api import sync_playwright

    library = find_library()
    source = read_source()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        # Nothing may leave the machine while rendering a committed artefact.
        page.route(
            "**/*",
            lambda route: (
                route.abort() if route.request.url.startswith("http") else route.continue_()
            ),
        )
        page.set_content('<div id="target"></div>')
        page.add_script_tag(path=str(library))
        svg = page.evaluate(
            """async (source) => {
                mermaid.initialize({
                    startOnLoad: false,
                    htmlLabels: false,
                    flowchart: { htmlLabels: false, useMaxWidth: false },
                    theme: 'neutral',
                });
                const { svg } = await mermaid.render('diagram', source);
                return svg;
            }""",
            source,
        )
        browser.close()

    svg = give_intrinsic_size(svg)
    if "<foreignObject" in svg:
        raise SystemExit("the SVG contains a foreignObject, so it will not load as an image")
    destination = OUTPUT / "architecture.svg"
    destination.write_text(svg, encoding="utf-8")
    (OUTPUT / "diagram-manifest.json").write_text(
        json.dumps(
            {
                "source": "README.md fence labelled mermaid-source",
                # The fence itself, hashed. Comparing node labels catches a renamed
                # box and misses a rewired edge: an audit redirected the referee
                # probe into the training stage, which is test-set leakage drawn as
                # an arrow, and every label test stayed green.
                "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "renderer": library.name,
                "output": destination.name,
                "html_labels": False,
                "background": BACKGROUND,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {destination} ({destination.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
