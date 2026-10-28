"""Turns the committed correctness vectors into every figure the documents quote.

One rule governs this module: no magic thresholds. The question "is this slice
regression large enough to care about" is not answered with a number somebody liked.
It is answered with the suite's own minimum detectable effect, so the materiality bar
is whatever the suite could have seen. That makes the blindness result self
contained: a slice counts as missed only when it moved further than the thing that
was supposed to be watching it could resolve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finetune_eval_power.config import ArmConfig, DecisionRule
from finetune_eval_power.errors import ToolError
from finetune_eval_power.power.bootstrap import (
    DropInterval,
    bootstrap_drop_interval,
    bootstrap_seed_interval,
)
from finetune_eval_power.power.counts import POINTS, PairedCounts, paired_counts
from finetune_eval_power.power.paired import (
    mcnemar_p_value,
    minimum_detectable_points,
    required_suite_size,
)
from finetune_eval_power.power.variance import (
    MINIMUM_SEEDS,
    SeedVariance,
    running_spread,
    seed_variance,
    seeds_to_settle,
)
from finetune_eval_power.runs import SweepRuns
from finetune_eval_power.slices.spec import CapabilitySlice, enumerate_slices
from finetune_eval_power.slices.suite import ProbeSet, build_probe

REGRESSION = "regression"
IMPROVEMENT = "improvement"
FLAT = "flat"
#: Seed used by every bootstrap in this module, so an interval is reproducible.
BOOTSTRAP_SEED = 909
#: Resamples for the case-level interval, which is only ever used to show how much
#: narrower it is than the retraining-level one. A illustrative comparison does not
#: need the resolution the published interval gets.
CASE_RESAMPLES = 1000


@dataclass(frozen=True)
class SuiteVerdict:
    """What the eval suite reports at one freeze depth, for one retraining.

    Carries the alpha it was judged at. It used to build a fresh default
    `DecisionRule` inside `significant`, so a caller passing a non-default alpha
    got a detection floor at their alpha and every flag, operating depth and
    missed-slice list at 0.05. The config calls alpha "the significance level for
    every hypothesis test the tool runs", which was then not true of any of them.
    """

    seed: int
    freeze_depth: int
    counts: PairedCounts
    p_value: float
    alpha: float = DecisionRule().alpha

    @property
    def significant(self) -> bool:
        return self.p_value <= self.alpha

    @property
    def direction(self) -> str:
        if not self.significant:
            return FLAT
        return REGRESSION if self.counts.drop_points > 0.0 else IMPROVEMENT

    @property
    def flags_regression(self) -> bool:
        """The only thing a team acts on: a significant move in the wrong direction."""
        return self.direction == REGRESSION


@dataclass(frozen=True)
class DepthSummary:
    """One freeze depth, summarised across every seed."""

    freeze_depth: int
    verdicts: tuple[SuiteVerdict, ...]
    suite_drop_points: float
    suite_drop_interval: DropInterval
    discordance: float
    mde_points: float | None
    in_region_gain_points: float
    seeds_flagging_regression: int

    @property
    def seeds(self) -> int:
        return len(self.verdicts)

    @property
    def suite_is_quiet(self) -> bool:
        """True when at most half the retrainings produce a regression flag.

        Not "a majority are quiet": at five flags out of ten this is still true,
        and half is not a majority. The documents say "at most half" for the same
        reason.
        """
        return self.seeds_flagging_regression * 2 <= self.seeds


@dataclass(frozen=True)
class SliceFinding:
    """One capability slice at one freeze depth, measured on the probe."""

    name: str
    covered: bool
    suite_cases: int
    probe_cases: int
    drop_points: float
    interval: DropInterval
    case_interval_width_points: float
    discordance: float
    #: The slice's rate of changed verdicts on the PROBE's cases. Every drop and
    #: interval on this finding is measured there, so this is the discordance that
    #: belongs beside them.
    suite_discordance: float
    #: The same rate measured on the suite's own cases for this slice. It is a
    #: different number, usually smaller, and the two are not interchangeable: a
    #: prescription about how large the suite should be has to come from the
    #: suite's behaviour, not from the referee's.
    suite_mde_on_slice_points: float | None

    @property
    def interval_width_points(self) -> float:
        return self.interval.high_points - self.interval.low_points

    @property
    def materially_regressed(self) -> bool:
        """Dropped, with a seed-level interval that stays on the wrong side of zero."""
        return self.drop_points > 0.0 and self.interval.low_points > 0.0

    @property
    def suite_has_power(self) -> bool:
        """Whether the suite could reach the target power on this slice at all."""
        return self.suite_mde_on_slice_points is not None

    @property
    def resolution_floor_points(self) -> float:
        return POINTS / self.probe_cases


@dataclass(frozen=True)
class ClassFinding:
    """One label class, measured on the eval suite's own cases.

    Carries its own case count, discordance, floor and interval rather than only a
    drop, because the drop on its own invites the comparison that caused the worst
    error in this repository's documents: putting a per-class drop next to the
    aggregate's detection floor, which was computed for a different case count.
    """

    name: str
    suite_cases: int
    drop_points: float
    discordance: float
    floor_points: float | None
    interval_low_points: float
    interval_high_points: float

    @property
    def interval_excludes_zero(self) -> bool:
        return self.interval_low_points > 0.0 or self.interval_high_points < 0.0

    @property
    def resolvable_by_the_suite(self) -> bool:
        """Whether the suite's reading of this class clears its own per-class floor."""
        if self.floor_points is None:
            return False
        return self.drop_points > self.floor_points and self.interval_excludes_zero


@dataclass(frozen=True)
class ArmAnalysis:
    """Everything a document can quote about one arm."""

    arm: str
    seeds: tuple[int, ...]
    suite_size: int
    probe_size: int
    slices_total: int
    held_out_slices: int
    depths: tuple[DepthSummary, ...]
    operating_depth: int
    findings: tuple[SliceFinding, ...]
    variance: SeedVariance
    suite_accuracy_before: float
    suite_case_interval_width_points: float
    probe_change_points: float
    suite_class_findings: tuple[ClassFinding, ...]

    def depth(self, freeze_depth: int) -> DepthSummary:
        for summary in self.depths:
            if summary.freeze_depth == freeze_depth:
                return summary
        raise ToolError(f"freeze depth {freeze_depth} was not analysed")

    @property
    def operating(self) -> DepthSummary:
        return self.depth(self.operating_depth)

    @property
    def undetected(self) -> tuple[SliceFinding, ...]:
        """Slices that moved further than the suite could resolve, unflagged.

        Guarded by `suite_is_quiet`: if the suite did flag a regression at this
        depth then nothing here was missed, and the tuple is empty by construction
        rather than by a filter that could be forgotten.
        """
        if not self.operating.suite_is_quiet:
            return ()
        floor = self.operating.mde_points
        if floor is None:
            # The suite cannot reach the target power at any effect size, so every
            # material regression is beyond it.
            return tuple(item for item in self.findings if item.materially_regressed)
        return tuple(
            item for item in self.findings if item.materially_regressed and item.drop_points > floor
        )

    @property
    def worst_undetected(self) -> SliceFinding | None:
        missed = self.undetected
        return max(missed, key=lambda item: item.drop_points) if missed else None

    @property
    def slices_without_suite_power(self) -> int:
        return sum(1 for item in self.findings if not item.suite_has_power)

    @property
    def underpowered_covered_slices(self) -> int:
        """Slices the suite does sample and still has no usable power on.

        Kept separate from the never-sampled count because the two have different
        remedies. A slice with no suite cases needs cases; one with too few needs
        more of them, and saying that no suite size would help either is false of
        the second group.
        """
        return sum(1 for item in self.findings if item.covered and not item.suite_has_power)

    @property
    def improving_class_total_points(self) -> float:
        """How much the classes that improved gained, measured on the suite.

        This is the other half of the headline. The aggregate stays flat because
        some classes moved the other way, and the arithmetic has to be done on the
        suite's own cases, since the aggregate being explained is the suite's.
        """
        return -sum(
            item.drop_points for item in self.suite_class_findings if item.drop_points < 0.0
        )

    @property
    def label_classes(self) -> int:
        return len(self.suite_class_findings)

    @property
    def improving_classes(self) -> int:
        return sum(1 for item in self.suite_class_findings if item.drop_points < 0.0)

    @property
    def worst_class_on_suite(self) -> ClassFinding:
        return max(self.suite_class_findings, key=lambda item: item.drop_points)

    @property
    def seed_readings_points(self) -> tuple[float, ...]:
        """The measured drop from each retraining at the operating depth."""
        return tuple(verdict.counts.drop_points for verdict in self.operating.verdicts)

    @property
    def seeds_to_settle(self) -> int:
        """Retrainings after which the spread estimate stops moving materially."""
        return seeds_to_settle(self.seed_readings_points)

    @property
    def short_budget_noise_floor_points(self) -> float:
        """The refusal threshold a three retraining budget would have produced.

        The threshold is computed from an estimate of the spread, so an estimate
        taken from too few retrainings sets the bar in the wrong place and nothing
        downstream can tell. Published because the alternative is asserting that
        ten was chosen carefully and offering no number for it.
        """
        history = running_spread(self.seed_readings_points)
        at_three = next(value for count, value in history if count == MINIMUM_SEEDS)
        return self.variance.multiplier * at_three

    @property
    def improving_regions(self) -> int:
        """Improving slices with nesting removed, so it compares with the missed count.

        Applying the nesting correction to the unflattering count and not to the
        flattering one would be selecting the evidence, so both are published in the
        same form.
        """
        return len(_maximal(self.improving_findings))

    @property
    def improving_findings(self) -> tuple[SliceFinding, ...]:
        return tuple(
            item
            for item in self.findings
            if item.drop_points < 0.0 and item.interval.high_points < 0.0
        )

    @property
    def improving_slices(self) -> int:
        """Slices that moved the other way, with an interval clear of zero.

        Published next to the missed count because a reader is entitled to ask how
        many went the other way, and because it is the first thing a statistician
        asks about a list of eight selected from forty.
        """
        return len(self.improving_findings)

    @property
    def significant_verdict_seeds(self) -> int:
        """Retrainings where the suite's test rejected, in either direction."""
        return sum(1 for verdict in self.operating.verdicts if verdict.significant)

    @property
    def maximal_undetected(self) -> tuple[SliceFinding, ...]:
        """Missed slices that are not contained in another missed slice.

        Slices nest: a single-attribute slice and the pairs that partition it are all
        named separately, so a raw count of eight is not a count of eight independent
        regions. This reports the regions.
        """
        return _maximal(self.undetected)

    @property
    def slices_short_of_power(self) -> tuple[tuple[str, int, int | None], ...]:
        """The covered slices with no usable power, and the size each would need.

        Published as the size rather than as an adjective. Calling them "sampled too
        thinly" invited the reading that nothing would help, and measuring it shows
        that a larger suite fixes every one of them, two of them easily.

        A slice whose cases never change verdict has a discordance of zero, and no
        suite size follows from it, so its size is None rather than a number. That
        is not a corner case here: it is the extreme of exactly the condition that
        puts a slice in this list, and asking the power routine for a size at a
        discordance of zero used to raise.

        The size comes from `suite_discordance`, the slice's rate of changed
        verdicts on the suite's own cases, and not from `discordance`, which is the
        rate on the probe. An audit caught the published figure being computed from
        the probe while the sentence around it prescribed a suite size: the cheapest
        slice read 140 cases from the probe's rate and 592 from the suite's. A
        prescription about the suite has to be derived from the suite.
        """
        return tuple(
            (
                item.name,
                item.suite_cases,
                smallest_suite_with_power(item.suite_discordance)
                if item.suite_discordance > 0.0
                else None,
            )
            for item in self.findings
            if item.covered and not item.suite_has_power
        )

    @property
    def retraining_interval_width_points(self) -> float:
        """Width of the interval retraining alone puts around a single measurement.

        Stated this way so it can be compared with the case-level interval, which is
        also a statement about one measurement. The obvious comparison, between a
        bootstrap over cases and a bootstrap over the mean of several retrainings, is
        not like for like: the second shrinks with the number of retrainings and the
        first does not, so it makes retraining noise look smaller the more of it you
        measure.
        """
        return 2.0 * self.variance.noise_floor_points


def _slice_masks(
    arm: ArmConfig, probe: ProbeSet
) -> tuple[list[CapabilitySlice], dict[str, np.ndarray]]:
    slices = enumerate_slices(arm.space, arm.pair_attributes)
    return slices, {item.name: probe.membership(item) for item in slices}


def analyse(arm: ArmConfig, runs: SweepRuns, rule: DecisionRule | None = None) -> ArmAnalysis:
    """Compute every published quantity for one arm from its committed measurement."""
    rule = rule or DecisionRule()
    if runs.arm != arm.name:
        raise ToolError(f"measurement is for arm {runs.arm!r}, not {arm.name!r}")
    probe = build_probe(arm.space, arm.probe_repeats)
    slices, masks = _slice_masks(arm, probe)
    region_mask = probe.region_mask(arm.finetune_profile)
    suite_counts = runs.manifest.get("suite_slice_case_counts", {})
    held_out = set(runs.manifest.get("held_out_slices", []))

    depth_summaries: list[DepthSummary] = []
    for freeze_depth in runs.freeze_depths:
        depth_position = runs.depth_index(freeze_depth)
        verdicts: list[SuiteVerdict] = []
        drops: list[float] = []
        discordances: list[float] = []
        gains: list[float] = []
        for seed_position, seed in enumerate(runs.seeds):
            before = runs.suite_before[seed_position, depth_position]
            after = runs.suite_after[seed_position, depth_position]
            counts = paired_counts(before, after)
            verdicts.append(
                SuiteVerdict(
                    seed=seed,
                    freeze_depth=freeze_depth,
                    counts=counts,
                    p_value=mcnemar_p_value(counts.only_before, counts.only_after),
                    alpha=rule.alpha,
                )
            )
            drops.append(counts.drop_points)
            discordances.append(counts.discordance)
            probe_before = runs.probe_before[seed_position, depth_position]
            probe_after = runs.probe_after[seed_position, depth_position]
            gains.append(
                float(np.mean(probe_after[region_mask]) - np.mean(probe_before[region_mask]))
                * POINTS
            )
        discordance = float(np.mean(discordances))
        depth_summaries.append(
            DepthSummary(
                freeze_depth=freeze_depth,
                verdicts=tuple(verdicts),
                suite_drop_points=float(np.mean(drops)),
                suite_drop_interval=bootstrap_seed_interval(drops, BOOTSTRAP_SEED),
                discordance=discordance,
                mde_points=(
                    minimum_detectable_points(
                        runs.suite_size, discordance, rule.alpha, rule.target_power
                    )
                    if discordance > 0.0
                    else None
                ),
                in_region_gain_points=float(np.mean(gains)),
                seeds_flagging_regression=sum(1 for item in verdicts if item.flags_regression),
            )
        )

    operating_depth = _choose_operating_depth(depth_summaries)
    operating_position = runs.depth_index(operating_depth)
    operating = next(
        summary for summary in depth_summaries if summary.freeze_depth == operating_depth
    )

    suite_cells = runs.manifest.get("suite_cells", [])
    findings: list[SliceFinding] = []
    for item in slices:
        mask = masks[item.name]
        # No emptiness guard: the probe enumerates every cell of the space and a
        # slice constrains attributes to values the space has, so a slice always
        # has cases. A test asserts that invariant directly instead.
        probe_cases = int(mask.sum())
        per_seed: list[float] = []
        per_seed_discordance: list[float] = []
        for seed_position in range(len(runs.seeds)):
            before = runs.probe_before[seed_position, operating_position][mask]
            after = runs.probe_after[seed_position, operating_position][mask]
            counts = paired_counts(before, after)
            per_seed.append(counts.drop_points)
            per_seed_discordance.append(counts.discordance)
        # The case-level interval is computed on the first retraining only. It exists
        # to be compared against the retraining-level interval, and averaging it over
        # retrainings would cost ten times as much without changing that comparison.
        case_interval = bootstrap_drop_interval(
            runs.probe_before[0, operating_position][mask],
            runs.probe_after[0, operating_position][mask],
            BOOTSTRAP_SEED,
            resamples=CASE_RESAMPLES,
        )
        slice_discordance = float(np.mean(per_seed_discordance))
        suite_cases = int(suite_counts.get(item.name, 0))
        suite_mask = np.array([item.matches(cell) for cell in suite_cells], dtype=bool)
        suite_discordance = 0.0
        if suite_mask.any():
            suite_discordance = float(
                np.mean(
                    [
                        paired_counts(
                            runs.suite_before[position, operating_position][suite_mask],
                            runs.suite_after[position, operating_position][suite_mask],
                        ).discordance
                        for position in range(len(runs.seeds))
                    ]
                )
            )
        findings.append(
            SliceFinding(
                name=item.name,
                covered=item.name not in held_out,
                suite_cases=suite_cases,
                probe_cases=probe_cases,
                drop_points=float(np.mean(per_seed)),
                interval=bootstrap_seed_interval(per_seed, BOOTSTRAP_SEED),
                case_interval_width_points=float(
                    case_interval.high_points - case_interval.low_points
                ),
                discordance=slice_discordance,
                suite_discordance=suite_discordance,
                suite_mde_on_slice_points=(
                    minimum_detectable_points(
                        suite_cases, slice_discordance, rule.alpha, rule.target_power
                    )
                    if suite_cases > 0 and slice_discordance > 0.0
                    else None
                ),
            )
        )

    return ArmAnalysis(
        arm=arm.name,
        seeds=runs.seeds,
        suite_size=runs.suite_size,
        probe_size=runs.probe_size,
        slices_total=len(slices),
        held_out_slices=len(held_out),
        depths=tuple(depth_summaries),
        operating_depth=operating_depth,
        findings=tuple(findings),
        variance=seed_variance(
            [verdict.counts.drop_points for verdict in operating.verdicts],
            rule.seed_noise_multiplier,
        ),
        suite_accuracy_before=float(np.mean(runs.suite_before[:, operating_position])),
        probe_change_points=float(
            np.mean(
                [
                    paired_counts(
                        runs.probe_before[position, operating_position],
                        runs.probe_after[position, operating_position],
                    ).drop_points
                    for position in range(len(runs.seeds))
                ]
            )
        ),
        suite_class_findings=_suite_class_findings(arm, runs, operating_position, rule),
        suite_case_interval_width_points=_interval_width(
            bootstrap_drop_interval(
                runs.suite_before[0, operating_position],
                runs.suite_after[0, operating_position],
                BOOTSTRAP_SEED,
            )
        ),
    )


def _maximal(items: tuple[SliceFinding, ...]) -> tuple[SliceFinding, ...]:
    """Drop every slice that is contained in another slice of the same set."""
    constraints = {item.name: set(item.name.split(" & ")) for item in items}
    return tuple(
        item
        for item in items
        if not any(
            other.name != item.name and constraints[other.name] < constraints[item.name]
            for other in items
        )
    )


def _interval_width(interval: DropInterval) -> float:
    return interval.high_points - interval.low_points


def _suite_class_findings(
    arm: ArmConfig, runs: SweepRuns, depth_position: int, rule: DecisionRule
) -> tuple[ClassFinding, ...]:
    """Per label class, measured on the eval suite's own cases.

    Deliberately not measured on the probe. The quantity these numbers explain is
    the suite's aggregate, so they have to come from the cases the suite holds. An
    earlier version of the README explained a suite figure with probe figures, which
    produced a story that did not add up and would not have survived a reviewer with
    a calculator.
    """
    cells = runs.manifest.get("suite_cells", [])
    findings: list[ClassFinding] = []
    for class_name in arm.space.label_classes():
        item = CapabilitySlice.of(**{arm.space.label_attribute: class_name})
        mask = np.array([item.matches(cell) for cell in cells], dtype=bool)
        # No emptiness guard. `build_suite` refuses a coverage profile that selects
        # no cells for a label class, so every class has suite cases by
        # construction. A test asserts that invariant directly, which is better
        # than a branch here that nothing can reach and coverage keeps pointing at.
        per_seed: list[float] = []
        per_seed_discordance: list[float] = []
        for position in range(len(runs.seeds)):
            counts = paired_counts(
                runs.suite_before[position, depth_position][mask],
                runs.suite_after[position, depth_position][mask],
            )
            per_seed.append(counts.drop_points)
            per_seed_discordance.append(counts.discordance)
        cases = int(mask.sum())
        discordance = float(np.mean(per_seed_discordance))
        interval = bootstrap_seed_interval(per_seed, BOOTSTRAP_SEED)
        findings.append(
            ClassFinding(
                name=item.name,
                suite_cases=cases,
                drop_points=float(np.mean(per_seed)),
                discordance=discordance,
                # The floor uses the suite's own discordance on the suite's own
                # cases, so every input to it comes from the thing being judged.
                floor_points=(
                    minimum_detectable_points(cases, discordance, rule.alpha, rule.target_power)
                    if discordance > 0.0
                    else None
                ),
                interval_low_points=interval.low_points,
                interval_high_points=interval.high_points,
            )
        )
    return tuple(findings)


def _choose_operating_depth(summaries: list[DepthSummary]) -> int:
    """The shallowest freeze the eval suite does not object to.

    This models the decision a team actually makes. Freezing more protects the old
    capabilities and gives up adaptation, so nobody freezes more than they have to,
    and what stops them freezing less is the eval suite complaining. The depth the
    suite waves through is therefore the depth that ships.
    """
    for summary in summaries:
        if summary.suite_is_quiet:
            return summary.freeze_depth
    return max(summary.freeze_depth for summary in summaries)


def suite_size_curve(
    discordance: float,
    sizes: tuple[int, ...],
    rule: DecisionRule | None = None,
) -> list[dict[str, float | int | None]]:
    """Minimum detectable effect against suite size, by both methods."""
    rule = rule or DecisionRule()
    rows: list[dict[str, float | int | None]] = []
    for size in sizes:
        rows.append(
            {
                "suite_size": size,
                "exact_points": minimum_detectable_points(
                    size, discordance, rule.alpha, rule.target_power, method="exact"
                ),
                "normal_points": minimum_detectable_points(
                    size, discordance, rule.alpha, rule.target_power, method="normal"
                ),
            }
        )
    return rows


def smallest_suite_with_power(
    discordance: float, rule: DecisionRule | None = None, ceiling: int = 100_000
) -> int | None:
    """Fewest cases at which this suite reaches the target power against any effect.

    Below this size no effect reaches the target power, which is not the same as
    "no outcome can be significant": at the shipped discordance a suite of a
    hundred cases still reaches about two thirds power against the largest effect
    it can represent. An earlier version of this docstring claimed the stronger
    thing, which would have been the sentence a reader quoted.

    It is a different quantity from the detection floor and a more alarming one:
    the floor says how large an effect must be, this says whether the target is
    reachable at all.
    """
    rule = rule or DecisionRule()

    def reachable(size: int) -> bool:
        return (
            minimum_detectable_points(
                size, discordance, rule.alpha, rule.target_power, method="exact"
            )
            is not None
        )

    low, high = 1, 2
    while high <= ceiling:
        if reachable(high):
            break
        low = high
        high *= 2
    else:
        return None
    while low + 1 < high:
        middle = (low + high) // 2
        if reachable(middle):
            high = middle
        else:
            low = middle
    return high


def suite_size_for(drop_points: float, discordance: float, rule: DecisionRule | None = None):
    """How many cases a suite needs to see a drop of this size."""
    rule = rule or DecisionRule()
    return required_suite_size(
        drop_points / POINTS, discordance, rule.alpha, rule.target_power, method="exact"
    )
