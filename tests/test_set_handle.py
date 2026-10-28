"""The target a reviewer runs first, tested by running it.

`make set-handle` shipped broken: the Makefile ran a `python3 - <<'PY'` heredoc
across several recipe lines without `.ONESHELL:`, so each line went to its own
shell, the heredoc reached no interpreter, and `import pathlib, sys` was executed
by the shell as ImageMagick's `import`. The target failed, the CI badge was never
added, and the README promised both.

The reason it survived two audits is the interesting part: the only test that
mentioned it asserted the README *mentions* `make set-handle`. Nothing executed it.
These tests execute it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "set_handle.py"


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """The documents only, which is everything the script touches."""
    destination = tmp_path / "repo"
    (destination / "docs" / "adr").mkdir(parents=True)
    shutil.copy(ROOT / "README.md", destination / "README.md")
    for source in (ROOT / "docs").rglob("*.md"):
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, target)
    return destination


def set_handle(clone: Path, handle: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), handle, "--root", str(clone)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_placeholder_is_gone_from_every_document_afterwards(clone):
    result = set_handle(clone, "someone")
    assert result.returncode == 0, result.stderr
    remaining = [path for path in clone.rglob("*.md") if "GITHUB_USERNAME" in path.read_text()]
    assert remaining == []


def test_the_ci_badge_is_added_because_the_readme_promises_it(clone):
    set_handle(clone, "someone")
    readme = (clone / "README.md").read_text(encoding="utf-8")
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert "https://github.com/someone/finetune-eval-power/actions" in readme


def test_the_badge_sits_above_the_other_badges_rather_than_anywhere(clone):
    set_handle(clone, "someone")
    lines = (clone / "README.md").read_text(encoding="utf-8").splitlines()
    badge = next(i for i, line in enumerate(lines) if "badge.svg" in line)
    python = next(i for i, line in enumerate(lines) if "img.shields.io/badge/python" in line)
    assert badge == python - 1


def test_running_it_twice_does_not_add_the_badge_twice(clone):
    set_handle(clone, "someone")
    first = (clone / "README.md").read_text(encoding="utf-8")
    set_handle(clone, "someone")
    second = (clone / "README.md").read_text(encoding="utf-8")
    assert first == second
    assert second.count("actions/workflows/ci.yml/badge.svg") == 1


@pytest.mark.parametrize("handle", ["", "   ", "owner/repo", "GITHUB_USERNAME"])
def test_a_handle_that_is_not_a_handle_is_refused_before_anything_is_written(clone, handle):
    original = (clone / "README.md").read_text(encoding="utf-8")
    result = set_handle(clone, handle)
    assert result.returncode == 1
    assert (clone / "README.md").read_text(encoding="utf-8") == original
