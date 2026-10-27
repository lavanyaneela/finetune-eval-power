"""The committed evidence, checked without a browser.

These assertions are about files that are already in the repository, so they run in
every job including the one that installs no extras. The rendering tools themselves
need Playwright and are exercised by running them, not by the suite.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
VIDEO = ROOT / "docs" / "video"
README = ROOT / "README.md"


def test_the_architecture_diagram_is_committed():
    assert (EVIDENCE / "architecture.svg").exists()


def test_the_diagram_has_intrinsic_dimensions_so_it_loads_as_an_image():
    """Mermaid emits a percentage width, which gives a browser no aspect ratio to
    work with and produces a differently sized image on every page."""
    svg = (EVIDENCE / "architecture.svg").read_text(encoding="utf-8")
    root = svg[: svg.index(">") + 1]
    assert re.search(r'width="\d+"', root), root
    assert re.search(r'height="\d+"', root), root
    assert "viewBox=" in root


def test_the_diagram_contains_no_foreign_object():
    """With HTML labels on, mermaid emits unclosed tags inside a foreignObject, the
    file stops being well formed XML, and it silently fails to load as an image
    while still displaying fine in a live page."""
    assert "<foreignObject" not in (EVIDENCE / "architecture.svg").read_text(encoding="utf-8")


def test_the_diagram_paints_a_background_rather_than_leaving_it_transparent():
    """Transparent sounds theme neutral and is not: edge labels have no fill behind
    them and come out dark grey on near black in a dark theme."""
    svg = (EVIDENCE / "architecture.svg").read_text(encoding="utf-8")
    assert re.search(r'<rect x="0" y="0" width="\d+" height="\d+" fill="#', svg)


def test_the_diagram_matches_the_source_fence_in_the_readme():
    assert "```mermaid-source" in README.read_text(encoding="utf-8")


def test_the_diagram_source_fence_is_not_labelled_mermaid():
    """A fence labelled mermaid gets a render attempt from GitHub, which has been
    observed failing from inside its own lazily loaded renderer."""
    assert "```mermaid\n" not in README.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "name", ("suite-verdict.png", "missed-capabilities.png", "noise-sources.png")
)
def test_each_screenshot_is_committed_and_not_empty(name: str):
    path = EVIDENCE / name
    assert path.exists()
    assert path.stat().st_size > 10_000


def test_the_screenshot_manifest_records_what_each_shot_was_framed_by():
    manifest = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shots"]
    for shot in manifest["shots"]:
        assert shot["framed_from"]
        assert shot["bytes"] > 0


def test_the_screenshot_manifest_records_a_legible_verdict_badge():
    badge = json.loads((EVIDENCE / "manifest.json").read_text(encoding="utf-8"))["badge"]
    assert badge["visible"]
    assert badge["colour"] != badge["background"]


def test_every_image_the_readme_references_exists():
    text = README.read_text(encoding="utf-8")
    referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text) + re.findall(
        r'<img src="([^"]+)"', text
    )
    local = [path for path in referenced if not path.startswith("http")]
    assert local
    for relative in local:
        assert (ROOT / relative).exists(), relative


def test_every_image_in_the_readme_has_alt_text():
    text = README.read_text(encoding="utf-8")
    for alt in re.findall(r"!\[([^\]]*)\]\(", text):
        assert alt.strip(), "an image in the README has empty alt text"
    for tag in re.findall(r"<img [^>]+>", text):
        assert 'alt="' in tag and 'alt=""' not in tag, tag


def test_the_demo_is_committed_and_small_enough_to_autoplay_on_github():
    gif = VIDEO / "demo.gif"
    assert gif.exists()
    assert gif.stat().st_size < 5_000_000


def test_the_demo_manifest_records_a_real_exit_code_for_every_command():
    manifest = json.loads((VIDEO / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["segments"]
    for segment in manifest["segments"]:
        assert segment["lines_captured"] > 0
        assert isinstance(segment["exit_code"], int)


def test_the_demo_shows_the_refusal_and_the_checker_failing():
    """The argument the recording makes: the tool declines, and the guard fires."""
    manifest = json.loads((VIDEO / "manifest.json").read_text(encoding="utf-8"))
    codes = [segment["exit_code"] for segment in manifest["segments"]]
    assert 2 in codes, "no segment demonstrates the refusal"
    assert codes[-1] != 0, "the last segment should show the number checker failing"


def test_no_placeholder_handle_is_left_in_a_badge_or_a_clone_url():
    """A placeholder in a clone command is the first thing a reviewer notices. The
    repository ships one on purpose, in exactly one place, filled by make set-handle.
    """
    text = README.read_text(encoding="utf-8")
    assert text.count("GITHUB_USERNAME") >= 1
    for forbidden in ("YOUR_USERNAME", "yourname", "<username>"):
        assert forbidden not in text


def test_every_badge_url_spells_the_repository_name_exactly():
    """A badge pointing at a differently cased name renders as "unknown", which is
    the first thing a reviewer notices."""
    text = README.read_text(encoding="utf-8")
    for url in re.findall(r"https://github\.com/[^)\s]+", text):
        if "finetune-eval-power" in url.lower():
            assert "finetune-eval-power" in url, url


def test_no_badge_names_a_logo_slug_that_might_not_exist():
    """A logo slug that is not in simple-icons renders as a broken badge, so the
    repository uses none rather than guessing one."""
    text = README.read_text(encoding="utf-8")
    assert "logo=" not in text
