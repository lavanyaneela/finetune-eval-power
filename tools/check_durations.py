#!/usr/bin/env python3
"""Fails the build when any tracked file publishes an absolute duration.

A millisecond is a fact about the machine that measured it, so this repository
publishes counts and ratios instead. The sweep genuinely takes minutes and that
number belongs in prose about what to expect, not in a claim, so the pattern list
targets the shapes a claim takes.

It sweeps every tracked and untracked file rather than only the checked documents.
A promise about "anywhere in this repository" needs a check over everything, and an
earlier build found that a new file is invisible to a git-tracked-only sweep until
it is committed, and then fails the next clone.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATTERNS = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|microseconds?|nanoseconds?)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?)\s+(?:faster|slower|per)\b", re.I),
    re.compile(r"\btakes\s+\d+(?:\.\d+)?\s*(?:s|seconds?|minutes?|hours?)\b", re.I),
)
SKIP_SUFFIXES = {".npy", ".png", ".gif", ".mp4", ".webm", ".svg", ".json", ".xml"}
SKIP_PARTS = {".git", "reports", "mlruns", "node_modules", "__pycache__", "docs/evidence"}
#: This file necessarily contains the patterns it looks for.
SKIP_FILES = {"tools/check_durations.py"}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / line for line in result.stdout.splitlines() if line]


def main() -> int:
    problems: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        if relative in SKIP_FILES:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if any(part in relative for part in SKIP_PARTS):
            continue
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PATTERNS:
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                problems.append(f"{relative}:{line}: published duration {match.group(0)!r}")
    if problems:
        print(f"{len(problems)} published durations:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"no published durations across {len(tracked_files())} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
