#!/usr/bin/env python3
"""Fails the build when a document quotes a figure the code no longer produces.

The checker matches anchor phrases rather than bare digits, because a checker that
searches for digits passes happily while the sentence around them has gone stale.
Each metric carries the exact wording a document must use with the value
substituted in, and any occurrence of that wording carrying a different value is a
failure.

Every rule below was added after something got through. In order:

- anchors are matched everywhere, including inside fenced blocks, code spans and
  image alt text, because guarding only the prose closes mutation and leaves
  insertion wide open;
- a metric that no document quotes is reported, because an unquoted metric is a
  figure nothing re-measures;
- two metrics may not share an anchor, including by containment, or a stale figure
  can satisfy the wrong guard;
- an anchor longer than four words straddles a line break in wrapped markdown and
  silently matches nothing;
- every exemption in the reverse check must be used, and none may hide a measured
  value.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "metrics.json"
MAX_ANCHOR_WORDS = 4
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")

#: Numbers allowed to appear in a checked document without matching a metric.
#: Each one must actually occur, and none may collide with a measured value.
REVERSE_EXEMPTIONS = {
    "0": "exit code zero, and the zero in several ratios",
    "1": "exit code one, ADR numbering, and 'one in N' phrasing",
    "2": "exit code two, ADR numbering",
    "3": "exit code three, ADR numbering",
    "4": "ADR numbering, the four label classes",
    "5": "ADR numbering, the five experiments",
    "0.05": "the significance level",
    "80": "the target power as a percentage",
    "95": "the confidence level of the published intervals",
    "1.96": "the normal quantile behind the noise floor",
    "24": "the generator grid size and the sequence length",
    "3.11": "the lowest supported Python version",
    "3.13": "the highest supported Python version",
    "3.12": "the middle Python version in the matrix",
    "0.1": "the version of this package",
}


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit(f"no metrics at {METRICS_PATH}; run tools/collect_metrics.py first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def anchor_pattern(anchor: str) -> re.Pattern[str]:
    """Match the anchor with any number in the placeholder, tolerating line breaks."""
    before, _, after = anchor.partition("{}")
    return re.compile(
        rf"{_flexible(before)}({NUMBER_PATTERN.pattern}|[A-Za-z][\w=&> -]*?){_flexible(after)}"
    )


def _flexible(fragment: str) -> str:
    """Whitespace in an anchor matches a line break, because markdown wraps."""
    return r"\s+".join(re.escape(word) for word in fragment.split())


def check_anchor_hygiene(anchors: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, str] = {}
    for metric, phrases in anchors.items():
        if not phrases:
            problems.append(f"metric {metric} declares no anchor phrase")
        for phrase in phrases:
            words = [word for word in phrase.split() if word != "{}"]
            if len(words) > MAX_ANCHOR_WORDS:
                problems.append(
                    f"anchor {phrase!r} for {metric} has {len(words)} words, "
                    f"more than {MAX_ANCHOR_WORDS}, and will straddle a line break"
                )
            for other_phrase, other_metric in seen.items():
                if other_metric == metric:
                    continue
                if phrase == other_phrase or phrase in other_phrase or other_phrase in phrase:
                    problems.append(
                        f"anchor {phrase!r} for {metric} overlaps {other_phrase!r} "
                        f"for {other_metric}, so one can satisfy the other's guard"
                    )
            seen[phrase] = metric
    return problems


def check_documents(payload: dict) -> tuple[list[str], dict[str, int]]:
    problems: list[str] = []
    occurrences = dict.fromkeys(payload["anchors"], 0)
    for relative in payload["checked_documents"]:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"checked document {relative} does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        for metric, phrases in payload["anchors"].items():
            expected = payload["rendered"].get(metric)
            if expected is None:
                continue
            for phrase in phrases:
                for match in anchor_pattern(phrase).finditer(text):
                    found = match.group(1).replace(",", "")
                    if found == expected:
                        occurrences[metric] += 1
                    else:
                        problems.append(
                            f"{relative}: {phrase.replace('{}', found)!r} is stale, "
                            f"the measurement says {expected}"
                        )
    return problems, occurrences


def check_reverse(payload: dict) -> list[str]:
    """Report numbers in the README that no metric accounts for."""
    problems: list[str] = []
    values = set(payload["rendered"].values())
    used: set[str] = set()
    path = ROOT / "README.md"
    if not path.exists():
        return ["README.md does not exist"]
    text = path.read_text(encoding="utf-8")
    stripped = re.sub(r"```.*?```", " ", text, flags=re.S)
    stripped = re.sub(r"`[^`]*`", " ", stripped)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = re.sub(r"\]\([^)]*\)", " ", stripped)
    for match in NUMBER_PATTERN.finditer(stripped):
        token = match.group(0)
        bare = token.replace(",", "")
        if bare in values or token in values:
            continue
        if bare in REVERSE_EXEMPTIONS:
            used.add(bare)
            continue
        line = stripped[: match.start()].count("\n") + 1
        problems.append(f"README.md line {line}: {token} matches no metric and no exemption")
    for exemption in REVERSE_EXEMPTIONS:
        if exemption not in used:
            problems.append(f"exemption {exemption!r} is never used, so it is guarding nothing")
        if exemption in values:
            problems.append(f"exemption {exemption!r} hides a measured value and must be removed")
    return problems


def main() -> int:
    payload = load_metrics()
    problems = check_anchor_hygiene(payload["anchors"])
    document_problems, occurrences = check_documents(payload)
    problems.extend(document_problems)
    for metric, count in occurrences.items():
        if payload["rendered"].get(metric) is None:
            continue
        if count == 0:
            problems.append(
                f"metric {metric} is quoted by no checked document, so nothing "
                "re-measures it; quote it or stop collecting it"
            )
    problems.extend(check_reverse(payload))

    if problems:
        print(f"{len(problems)} receipts problems:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(
        f"receipts clean: {len(payload['metrics'])} metrics, "
        f"{len(payload['checked_documents'])} documents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
