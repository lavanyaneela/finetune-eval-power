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
#: Numbers that appear legitimately and are not measured. Scoped to the documents
#: they are allowed in, because a global free pass is not a guard: an audit pointed
#: out that `80`, `95` and `1.96` were accepted in any sentence of any document, so
#: "the suite resolves 80 distinct capabilities" passed the build. An exemption now
#: has to name where it is allowed, and is rejected everywhere else.
REVERSE_EXEMPTIONS: dict[str, tuple[str, tuple[str, ...]]] = {
    "80": ("the target power, written as a percentage", ("README.md",)),
    "95": ("the confidence level of the published intervals", ("README.md",)),
    "1.96": (
        "the normal quantile that turns a standard deviation into the noise floor",
        (
            "README.md",
            "docs/defense-guide.md",
            "docs/adr/ADR-004-seed-budget-and-pinned-training.md",
        ),
    ),
    "3.11": ("the minimum supported Python version", ("README.md",)),
    "1": (
        "an exit code, and the only one of the four that is not also a measured value",
        ("README.md",),
    ),
}

#: Lines that must appear in a document character for character. The reverse sweep
#: is a value allow-list: it asks whether a digit is one of the measured values, not
#: whether it is the right one for this sentence. Every exit code but `1` happens to
#: equal some measured value, so the published exit-code contract could be rewritten
#: freely while the build stayed green. These rows are the contract, so they are
#: pinned as text, and a test asserts each one still matches what the CLI returns.
# These are verbatim copies of published lines and must not be wrapped or
# reflowed, so the long-line rule is waived for them specifically.
LITERAL_GUARDS: tuple[tuple[str, str], ...] = (
    (
        "README.md",
        "| `0` | the tool answered and the answer is the reassuring one: `power` means the suite can see a regression of the size claimed, `blind` means nothing fell past its floor unnoticed |",  # noqa: E501
    ),
    (
        "README.md",
        "| `1` | the tool answered and the answer is the unwelcome one: `power` prints the suite size that would work, `blind` lists the capabilities it found |",  # noqa: E501
    ),
    (
        "README.md",
        "| `2` | refused: no suite size fixes it. Either retraining noise alone is at least as wide as the claimed effect, or the suite's cases change verdict so rarely that no size reaches the target power against any effect |",  # noqa: E501
    ),
    (
        "README.md",
        "| `3` | the tool itself failed, or was used wrongly. A mistyped subcommand or `--arm` lands here, not on `2`, so a typo cannot be mistaken for the refusal |",  # noqa: E501
    ),
    (
        "README.md",
        "Requires Python `3.11` or newer. No API keys, no cloud credentials, no network at runtime.",  # noqa: E501
    ),
    ("README.md", "python -m finetune_eval_power.cli blind --arm vision"),
    ("README.md", "python -m finetune_eval_power.cli power --arm vision --claim 2"),
)


def check_literals() -> list[str]:
    """Every pinned line is present as a WHOLE line, character for character.

    This tested `line not in text`, which pins a prefix and nothing else. An audit
    appended "Treat it as advisory: a suite of 480 eval cases clears it and teams
    routinely ship on a `2`" to the refusal row and the build stayed green, which
    is the whole published contract rewritten under an intact guard. A pinned line
    now has to be the entire line.
    """
    problems: list[str] = []
    for relative, line in LITERAL_GUARDS:
        path = ROOT / relative
        if not path.exists():
            problems.append(f"{relative} does not exist")
            continue
        lines = [item.strip() for item in path.read_text(encoding="utf-8").splitlines()]
        if line not in lines:
            near = [item for item in lines if item.startswith(line[:28])]
            detail = f" The closest line is {near[0]!r}." if near else ""
            problems.append(
                f"{relative}: the pinned line {line!r} is not present as a whole "
                f"line. It is part of the published contract and is guarded as "
                f"text, not as a figure.{detail}"
            )
    return problems


#: Bold, italic and code markers are removed before an anchor is matched. A figure
#: written as **3.41** is still the figure the sentence is about.
EMPHASIS = re.compile(r"[*_`]+")
#: An ordered list marker is markdown, not a figure.
LIST_MARKER = re.compile(r"(?m)^\s{0,3}\d+[.)]\s")


#: Relations between metrics that the documents assert in words. Anchors guard the
#: digits in a sentence and are structurally blind to the relation joining them, so
#: two correctly guarded figures can sit either side of a false comparison and the
#: build stays green. Both adversarial audits of this repository found exactly that,
#: and this table is the answer: every relation a document states is also evaluated
#: against the measurement.
RELATIONS: tuple[tuple[str, str], ...] = (
    (
        "seed_noise_floor_points > mde_points",
        "the refusal threshold must be wider than the suite's floor, or the "
        "repository's central claim is false",
    ),
    (
        "seeds_to_settle < seeds_declared",
        "the seed budget is above the point at which the spread estimate settles. "
        "If this ever reversed, the budget would be too small and the refusal "
        "threshold would rest on an estimate that is still moving",
    ),
    (
        "short_budget_noise_floor_points < seed_noise_floor_points",
        "a three retraining budget sets the refusal bar LOWER than the settled "
        "estimate does, so it would accept claims the measurement cannot support. "
        "This is the measured form of ADR-004's argument for ten",
    ),
    (
        "seed_spread_points < mde_points",
        "the raw standard deviation is NOT the quantity wider than the floor; a "
        "document that says it is has dropped the 1.96 step",
    ),
    (
        "retraining_interval_points > case_interval_points",
        "retraining must be the wider of the two noise sources, or the refusal rule "
        "has nothing to stand on",
    ),
    (
        "abs(retraining_interval_points - 2 * seed_noise_floor_points) <= 0.02",
        "the published interval is twice the half-width refusal threshold, allowing "
        "for the two being rounded independently",
    ),
    (
        "slices_without_power > slices_without_power_at_probe_rate",
        "the suite's own rate of changed verdicts reports MORE blind slices than "
        "the referee's does. If this ever reversed, the sentence calling the "
        "referee's count the flattering one would be backwards",
    ),
    (
        "held_out_slices + underpowered_covered_slices == slices_without_power",
        "the never-sampled slices are a subset of the slices with no usable power, "
        "not a separate group to be added to them",
    ),
    (
        "undetected_covered_slices <= undetected_slices",
        "the covered missed slices are a subset of the missed slices",
    ),
    (
        "maximal_undetected_slices <= undetected_slices",
        "slices nest, so the region count cannot exceed the slice count",
    ),
    (
        "suite_cases > smallest_suite_with_any_power",
        "the suite that ships is above the size at which nothing has power, so its "
        "problem is not its size",
    ),
    (
        "sequence_suite_cases > sequence_smallest_suite_with_power",
        "the same, for the second arm",
    ),
    (
        "mde_points_unpaired > mde_points",
        "the unpaired calculation must be the more pessimistic one, or the argument "
        "for taking the pairing seriously runs backwards",
    ),
    (
        "suite_worst_class_points < suite_worst_class_floor_points",
        "the worst class on the suite's own cases sits BELOW its own per-class "
        "floor; any sentence claiming the suite could have resolved it is false",
    ),
    (
        "in_region_gain_points < in_region_spread_points",
        "the in-region gain is smaller than its own retraining spread, so it must "
        "not be described as a gain",
    ),
    (
        "improving_classes + 1 <= label_classes",
        "at least one class got worse while the named number improved, which is what "
        "the cancellation sentence asserts; a class may also be exactly flat",
    ),
    (
        "thinnest_slice_floor_points > probe_resolution_floor_points",
        "a thin slice resolves less finely than the whole probe",
    ),
    (
        "finest_slice_floor_points > probe_resolution_floor_points",
        "no single slice resolves more finely than the whole probe does, so the "
        "finest slice floor is the lower end of the slice range and not of all of it",
    ),
    (
        "thinnest_slice_floor_points > 4 * finest_slice_floor_points",
        "the factor the prose spells as a word rather than a digit, pinned here so "
        "that a number written out is still machine checked",
    ),
    (
        "suite_worst_class_points < worst_undetected_slice_floor_points",
        "the drop the suite's own cases show for that class is below the floor at "
        "the probe's rate as well as below the floor at its own, so the sentence "
        "does not depend on which rate a reader prefers",
    ),
    (
        "worst_undetected_points < worst_undetected_slice_floor_points",
        "the worst missed slice sits BELOW the floor the suite would have had on it, "
        "which is the reason it was missed; a document that puts the drop above that "
        "floor has quoted the floor computed from the suite's own discordance in a "
        "sentence about the probe's",
    ),
    (
        "improving_regions <= improving_slices",
        "slices nest on the improving side too, so its region count cannot exceed "
        "its slice count either",
    ),
    (
        "dearest_underpowered_slice_cases > cheapest_underpowered_slice_cases",
        "the two ends of the underpowered range are named the right way round",
    ),
    (
        "underpowered_slices_with_a_size == underpowered_covered_slices",
        "every underpowered slice names a suite size that would fix it; if one "
        "stopped doing so, a document saying a larger suite closes all of them "
        "would be false and nothing else here would catch it",
    ),
    (
        "outside_loss_points > shipped_outside_spread_points",
        "the outside-region loss is wider than one standard deviation of its own "
        "retraining spread. True, and NOT sufficient to call it a measured cost: "
        "see the relation below, which is the comparison ADR-004 requires",
    ),
    (
        "outside_loss_points < shipped_outside_noise_floor_points",
        "the outside-region loss does NOT clear 1.96 standard deviations of its own "
        "retraining spread, so it is not resolved above retraining noise either. A "
        "paragraph here once compared it against the raw standard deviation and "
        "called it the one asymmetry in the table, which is the exact error ADR-004 "
        "was written about",
    ),
    (
        "outside_loss_points > shipped_outside_mean_bar_points",
        "the outside loss IS resolved as an average over the captured retrainings, "
        "even though a single retraining cannot resolve it, so a sentence saying "
        "the measurement resolves nothing here is false",
    ),
    (
        "in_region_gain_points > shipped_in_region_mean_bar_points",
        "so is the in-region gain, on the same basis",
    ),
    (
        "shipped_outside_mean_bar_points < shipped_outside_noise_floor_points",
        "averaging over retrainings shrinks the bar, which is what the seed budget "
        "buys and why the single-run and averaged answers differ",
    ),
    (
        "in_region_gain_points < shipped_outside_noise_floor_points",
        "neither side of the freeze-depth trade clears retraining noise, which is "
        "what makes the suite's confident improvement verdict the result",
    ),
    (
        "final_epoch_accuracy_span_points < epoch_band_points",
        "the span of the final readings is not the span of the change since epoch "
        "zero; a sentence that says reading and quotes the second one is false",
    ),
)


def check_relations(payload: dict) -> list[str]:
    """Evaluate every relation the documents assert, against the measurement."""
    metrics = payload["metrics"]
    problems: list[str] = []
    for expression, reason in RELATIONS:
        try:
            scope = {"__builtins__": {}, "abs": abs, "min": min, "max": max}
            holds = bool(eval(expression, scope, dict(metrics)))  # noqa: S307
        except (NameError, TypeError, ZeroDivisionError) as failure:
            problems.append(f"relation {expression!r} could not be evaluated: {failure}")
            continue
        if not holds:
            problems.append(f"relation {expression!r} is false. {reason}")
    return problems


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
        # No full stop in the class: it swallows the sentence-ending period and the
        # captured value then never equals the published one.
        else r"[A-Za-z][\w=&>-]*"
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


def check_orphans(payload: dict) -> list[str]:
    """An anchor for a metric that no longer exists guards nothing.

    It also makes the suite's own "every metric is quoted" test fail in a way that
    looks like a documentation problem and is really a leftover.
    """
    return [
        f"anchor set declares {name!r}, which is not a metric any more"
        for name in payload["anchors"]
        if name not in payload["metrics"]
    ] + [
        f"metric {name!r} has no anchor, so nothing guards it"
        for name in payload["metrics"]
        if name not in payload["anchors"]
    ]


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
    """Each metric is quoted, and no two metrics are quoted by the same digits.

    The second half exists because two metrics that render the same string can be
    satisfied by one occurrence. `suite_cases` anchors "{} eval cases" and
    `sequence_suite_cases` anchors "sequence ships {}", both render 480, and the
    single phrase "the sequence ships 480 eval cases" satisfied both guards. Each
    document's genuine statement could then be deleted and replaced by one false
    sentence while every metric still counted as quoted. Claiming a figure is
    re-measured because some other figure's sentence happened to contain it is the
    same failure this whole file exists to prevent, so a match now records where it
    was found and two metrics may not claim the same digits.
    """
    problems: list[str] = []
    occurrences = dict.fromkeys(payload["anchors"], 0)
    claimed: dict[tuple[str, int, int], str] = {}
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
                    if found != expected:
                        problems.append(
                            f"{relative}: {phrase.replace('{}', found)!r} is stale, "
                            f"the measurement says {expected}"
                        )
                        continue
                    occurrences[metric] += 1
                    where = (relative, match.start(1), match.end(1))
                    other = claimed.setdefault(where, metric)
                    if other != metric:
                        problems.append(
                            f"{relative}: the digits {expected!r} at offset "
                            f"{where[1]} are claimed by both {other} and {metric}. "
                            "One occurrence cannot re-measure two figures, so each "
                            "needs a sentence of its own"
                        )
    return problems, occurrences


def check_reverse(payload: dict) -> list[str]:
    """Report numbers that no metric accounts for, in every checked document.

    This ran over the README alone until an audit pointed out that the ADRs are
    listed as checked documents, publish figures, and were guarded by nothing. The
    build stayed green while three of them could go stale, which is the repository's
    own headline bug reappearing somewhere nobody looked.
    """
    problems: list[str] = []
    values = set(payload["rendered"].values())
    used: set[str] = set()
    for relative in payload["checked_documents"]:
        problems.extend(_reverse_one(ROOT / relative, relative, values, used))
    for exemption in REVERSE_EXEMPTIONS:
        if exemption not in used:
            problems.append(f"exemption {exemption!r} is never used, so it is guarding nothing")
        if exemption in values:
            problems.append(f"exemption {exemption!r} hides a measured value and must be removed")
    return problems


def _blank(match: re.Match[str]) -> str:
    """Replace a span with blanks, keeping its line breaks.

    Collapsing the span instead shifts every line number after it, and a checker
    that reports the wrong line is a checker people stop reading.
    """
    return "\n" * match.group(0).count("\n") or " "


def _reverse_one(path: Path, relative: str, values: set[str], used: set[str]) -> list[str]:
    problems: list[str] = []
    if not path.exists():
        return [f"{relative} does not exist"]
    text = path.read_text(encoding="utf-8")
    stripped = LIST_MARKER.sub(" ", text)
    # Fenced blocks, inline code spans, HTML attributes and link targets used to be
    # blanked here. Three of those four render as text a reader believes, and the
    # exit-code table lives entirely in code spans, so the sweep now reads them. An
    # audit changed every cell of that table with the build green.
    # Alt text and title attributes are read by a screen reader and shown when an
    # image fails, so they are prose and are swept. They are pulled out before the
    # tag strip below, which would otherwise blank them with the markup.
    attributes = " ".join(
        match.group(1) for match in re.finditer(r'(?:alt|title)="([^"]*)"', stripped)
    )
    # Both of these used to be `[^>]` and `[^)]`, which match newlines, so a single
    # stray "<" or "](" blanked everything up to the next ">" or ")" anywhere later
    # in the file. An audit hid a fabricated section of invented figures that way.
    # An HTML tag starts with a letter or a slash. Without that, "use the tool when
    # n < 480 cases ... a rate of 0.00042 > 0" looks like a tag and everything
    # between the comparison operators is blanked, which an audit used to hide a
    # section of invented figures on a single line.
    stripped = re.sub(r"</?[A-Za-z][^>\n]*>", _blank, stripped)
    # The target of a link or image is a URL, not a sentence, and a badge URL
    # carries version numbers that are not figures. A link TITLE is a sentence and
    # renders as a tooltip, so it is kept and swept; only the URL itself goes.
    titles = " ".join(match.group(1) for match in re.finditer(r'\]\([^)\n]*"([^"]*)"\)', stripped))
    stripped = re.sub(r"\]\([^)\n]*\)", _blank, stripped)
    stripped = f"{stripped}\n{attributes}\n{titles}"
    # ADR-004 is a document name, not a figure. Without this the leading hyphen is
    # read as a sign and every reference becomes an unexplained number.
    stripped = re.sub(r"ADR-\d+", " ", stripped)
    for match in NUMBER_PATTERN.finditer(stripped):
        token = match.group(0)
        bare = token.replace(",", "")
        if bare in values or token in values:
            continue
        line = stripped[: match.start()].count("\n") + 1
        if bare in REVERSE_EXEMPTIONS:
            _reason, allowed = REVERSE_EXEMPTIONS[bare]
            if relative in allowed:
                used.add(bare)
                continue
            problems.append(
                f"{relative} line {line}: {token} is exempt in {list(allowed)} "
                "but not in this document"
            )
            continue
        problems.append(f"{relative} line {line}: {token} matches no metric and no exemption")
    return problems


def main() -> int:
    payload = load_metrics()
    problems = check_orphans(payload)
    problems.extend(check_literals())
    problems.extend(check_relations(payload))
    problems.extend(check_anchor_hygiene(payload["anchors"]))
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
