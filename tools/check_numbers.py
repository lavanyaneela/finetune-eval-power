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
  value;
- markdown emphasis is removed before matching, because a figure wrapped in bold
  sits between the anchor's words and stops it matching while the prose is perfectly
  correct, which is the same failure as a line break and just as invisible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "metrics.json"
MAX_ANCHOR_WORDS = 4
NUMBER_PATTERN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?![\w])")
#: Numbers spelled as words are a real blind spot: the guard reads the digits and a
#: sentence beside it says "forty" while the measurement says something else. These
#: are matched in the placeholder so that spelling a guarded figure out fails the
#: build rather than slipping past it.
NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand"
)

#: Numbers allowed to appear in a checked document without matching a metric.
#: Each one must actually occur, and none may collide with a measured value.
REVERSE_EXEMPTIONS = {
    "80": "the target power, written as a percentage",
    "95": "the confidence level of the published intervals",
}


#: Bold, italic and code markers are removed before an anchor is matched. A figure
#: written as **3.41** is still the figure the sentence is about.
EMPHASIS = re.compile(r"[*_`]+")
#: An ordered list marker is markdown, not a figure.
LIST_MARKER = re.compile(r"(?m)^\s{0,3}\d+[.)]\s")


def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        raise SystemExit(f"no metrics at {METRICS_PATH}; run tools/collect_metrics.py first")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def anchor_pattern(anchor: str, numeric: bool) -> re.Pattern[str]:
    """Match the anchor with any value in the placeholder, tolerating line breaks.

    The placeholder is type aware. For a numeric metric it matches a number or a
    number spelled as a word, and nothing else: an earlier version matched any word
    at all, which turned "errs toward buying eval cases" into a stale reading of the
    suite size. For a string metric it matches one token.
    """
    before, _, after = anchor.partition("{}")
    placeholder = (
        rf"{NUMBER_PATTERN.pattern}|(?i:\b(?:{NUMBER_WORDS})\b)"
        if numeric
        else r"[A-Za-z][\w=.&>-]*"
    )
    # Case insensitive, because a sentence that starts with the anchor capitalises
    # its first word and that is not a stale figure.
    return re.compile(rf"{_flexible(before)}({placeholder}){_flexible(after)}", re.IGNORECASE)


def _flexible(fragment: str) -> str:
    """Whitespace in an anchor matches a line break, because markdown wraps.

    Leading and trailing whitespace is preserved as a separator. An earlier version
    dropped it, so the anchor "{} retrainings" compiled to a pattern demanding the
    number and the word with nothing between them. Every numeric anchor silently
    matched nothing, the checker reported clean on every document, and it stayed
    that way until a figure was deliberately broken and the build did not fail.
    """
    if not fragment:
        return ""
    pattern = r"\s+".join(re.escape(word) for word in fragment.split())
    if fragment[:1].isspace():
        pattern = r"\s+" + pattern
    if fragment[-1:].isspace() and fragment.strip():
        pattern = pattern + r"\s+"
    return pattern


def _context(anchor: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    before, _, after = anchor.partition("{}")
    return tuple(before.split()), tuple(after.split())


def _is_suffix(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    return len(shorter) <= len(longer) and longer[len(longer) - len(shorter) :] == shorter


def _is_prefix(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    return len(shorter) <= len(longer) and longer[: len(shorter)] == shorter


def anchors_can_collide(first: str, second: str) -> bool:
    """True when one anchor can match a sentence written for the other.

    Literal containment is not the right test. "floor of {} points" and
    "noise floor of {}" share no substring, and in a document that says
    "a noise floor of 3.92 points" both patterns match the same number, so one
    metric's guard reports the other metric's value as stale. That happened.

    The condition is about what surrounding prose can supply. Two anchors collide
    when the words before the placeholder are a suffix of one another, so arbitrary
    earlier text could complete the longer one, and the words after are a prefix of
    one another, so arbitrary later text could complete it on the right.

    One more condition is needed, or the rule fires on every pair of one sided
    anchors. An anchor with nothing before the placeholder matches whatever precedes
    it, and one with nothing after matches whatever follows, so two anchors open on
    complementary sides are compatible in the abstract and cannot meet in a real
    sentence. A collision therefore also requires at least one side where both
    anchors say something and one absorbs the other.
    """
    left_first, right_first = _context(first)
    left_second, right_second = _context(second)
    left_compatible = _is_suffix(left_first, left_second) or _is_suffix(left_second, left_first)
    right_compatible = _is_prefix(right_first, right_second) or _is_prefix(
        right_second, right_first
    )
    shares_real_context = (left_first and left_second) or (right_first and right_second)
    return bool(left_compatible and right_compatible and shares_real_context)


def check_anchor_overlap(payload: dict) -> list[str]:
    """Report every pair of anchors from different metrics that can collide."""
    problems: list[str] = []
    anchors = payload["anchors"]
    names = sorted(anchors)
    for position, metric in enumerate(names):
        for other in names[position + 1 :]:
            for phrase in anchors[metric]:
                for other_phrase in anchors[other]:
                    if phrase.count("{}") != 1 or other_phrase.count("{}") != 1:
                        continue
                    if anchors_can_collide(phrase, other_phrase):
                        problems.append(
                            f"anchor {phrase!r} for {metric} can collide with "
                            f"{other_phrase!r} for {other} in ordinary prose"
                        )
    return problems


def check_anchor_hygiene(anchors: dict[str, list[str]]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, str] = {}
    for metric, phrases in anchors.items():
        if not phrases:
            problems.append(f"metric {metric} declares no anchor phrase")
        for phrase in phrases:
            if phrase.count("{}") != 1:
                problems.append(
                    f"anchor {phrase!r} for {metric} must contain exactly one "
                    "placeholder, otherwise it guards a phrase rather than a figure"
                )
                continue
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
        text = EMPHASIS.sub("", path.read_text(encoding="utf-8"))
        for metric, phrases in payload["anchors"].items():
            expected = payload["rendered"].get(metric)
            if expected is None:
                continue
            numeric = bool(NUMBER_PATTERN.fullmatch(expected))
            for phrase in phrases:
                for match in anchor_pattern(phrase, numeric).finditer(text):
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
    stripped = LIST_MARKER.sub(" ", text)
    stripped = re.sub(r"```.*?```", " ", stripped, flags=re.S)
    stripped = re.sub(r"`[^`]*`", " ", stripped)
    stripped = re.sub(r"<[^>]+>", " ", stripped)
    stripped = re.sub(r"\]\([^)]*\)", " ", stripped)
    # ADR-004 is a document name, not a figure. Without this the leading hyphen is
    # read as a sign and every reference becomes an unexplained number.
    stripped = re.sub(r"ADR-\d+", " ", stripped)
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
    problems.extend(check_anchor_overlap(payload))
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
