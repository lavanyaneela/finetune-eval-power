#!/usr/bin/env python3
"""Fills the GITHUB_USERNAME placeholder and adds the CI badge.

This lived in the Makefile as a `python3 - <<'PY'` heredoc spread over several
recipe lines. GNU make runs each recipe line in its own shell unless `.ONESHELL:`
is set, so the heredoc was never fed to python: the shell executed the Python body
as shell commands, `import pathlib, sys` resolved to ImageMagick's `import`, and
the target died with "unable to open X server". The badge was never added and the
first command a reviewer runs after cloning exited non-zero.

The lesson is not about heredocs. It is that nothing executed this target, so a
target that could not work shipped green. It is now a file with a test that runs
it against a copy of the repository and asserts the badge appears.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

PLACEHOLDER = "GITHUB_USERNAME"
REPOSITORY = "finetune-eval-power"
BADGE_MARKER = "actions/workflows/ci.yml/badge.svg"
ANCHOR = "![python](https://img.shields.io"


def badge_for(handle: str) -> str:
    return (
        f"[![ci](https://github.com/{handle}/{REPOSITORY}/actions/workflows/"
        f"ci.yml/badge.svg)](https://github.com/{handle}/{REPOSITORY}/actions/"
        "workflows/ci.yml)\n"
    )


def documents(root: pathlib.Path) -> list[pathlib.Path]:
    found = [root / "README.md"]
    found.extend(sorted((root / "docs").rglob("*.md")))
    return [path for path in found if path.exists()]


def apply(root: pathlib.Path, handle: str) -> tuple[int, bool]:
    """Returns the number of documents changed and whether the badge was added."""
    changed = 0
    for path in documents(root):
        text = path.read_text(encoding="utf-8")
        if PLACEHOLDER not in text:
            continue
        path.write_text(text.replace(PLACEHOLDER, handle), encoding="utf-8")
        changed += 1

    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    added = False
    if BADGE_MARKER not in text and ANCHOR in text:
        readme.write_text(text.replace(ANCHOR, badge_for(handle) + ANCHOR, 1), encoding="utf-8")
        added = True
    return changed, added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="fill the GitHub handle placeholder")
    parser.add_argument("handle")
    parser.add_argument("--root", default=".")
    arguments = parser.parse_args(argv)
    handle = arguments.handle.strip()
    if not handle or "/" in handle or handle == PLACEHOLDER:
        print(f"error: {arguments.handle!r} is not a GitHub handle", file=sys.stderr)
        return 1
    changed, added = apply(pathlib.Path(arguments.root), handle)
    print(
        f"handle set to {handle} in {changed} documents; "
        + ("CI badge added" if added else "CI badge already present")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
