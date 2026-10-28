#!/usr/bin/env python3
"""Fails the build when a tracked text file publishes an absolute duration.

A millisecond is a fact about the machine that measured it, so this repository
publishes counts and ratios instead. The pattern list targets the shapes a claim
takes, in digits and in words.

It does not sweep machine readable artifacts: the demo manifest records how long
each captured command took because the replay is paced by those numbers. So the
rule is "no duration reaches a reader as a claim", not "no duration exists in this
repository", and the documents were corrected to say the narrower thing after an
audit pointed out they claimed the wider one.

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

#: Spelled-out quantities, because the sweep is about what a reader is told, not
#: about which characters it is told in. Two durations written as words sat in the
#: checked documents through three audits: "twelve minutes into a capture" and "the
#: better part of an hour". A rule stated over "anything" has to catch those.
#: Quantity words, without the indefinite article. "a second" is almost always the
#: ordinal in this repository ("a second cost", "a second after it was written"),
#: and an article-matching pattern reported six of those for every real duration.
#: A rule nobody can keep green stops being a rule.
_COUNT = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|ninety|half|several|a\s+few|a\s+couple\s+of|"
    r"the\s+better\s+part\s+of\s+an?)"
)
PATTERNS = (
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|microseconds?|nanoseconds?)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*(?:seconds?|secs?)\s+(?:faster|slower|per)\b", re.I),
    re.compile(r"\btakes\s+\d+(?:\.\d+)?\s*(?:s|seconds?|minutes?|hours?)\b", re.I),
    # Minutes and hours have no ordinal sense in prose, so the article is included
    # for them and excluded for seconds.
    re.compile(rf"\b(?:an?|{_COUNT})\s+(?:minutes?|hours?)\b", re.I),
    re.compile(rf"\b{_COUNT}\s+seconds?\b", re.I),
    # Plural as well as singular. The pattern was `(?:second|minute|hour)` with no
    # `s?`, so "40 seconds", "14 minutes" and "3 hours" all passed, which is the
    # most natural way anyone writes a duration.
    re.compile(r"\b\d+(?:\.\d+)?[- ](?:seconds?|minutes?|hours?)\b", re.I),
)
#: Machine readable artifacts, where a duration is data rather than a claim. The
#: demo manifest records how long each captured command took, and it has to: the
#: replay is paced by those numbers and a reviewer checking the replay is honest
#: needs them. The rule this file enforces is that no duration reaches a reader as
#: a claim, which is narrower than "no duration appears anywhere", and the
#: documents now say the narrower thing. The suffixes are listed in the output so
#: the exception is visible rather than buried here.
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
    skipped = ", ".join(sorted(SKIP_SUFFIXES))
    print(
        f"no published durations across {len(tracked_files())} files "
        f"(machine readable artifacts not swept: {skipped})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
