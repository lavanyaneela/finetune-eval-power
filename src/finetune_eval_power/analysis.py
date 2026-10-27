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
from finetune_eval_power.power.variance import SeedVariance, seed_variance
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
    """What the eval suite reports at one freeze depth, for one retraining."""

    seed: int
    freeze_depth: int
    counts: PairedCounts
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value <= DecisionRule().alpha

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
        """True when a majority of retrainings produce no regression flag."""
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

    findings: list[SliceFinding] = []
    for item in slices:
        mask = masks[item.name]
        probe_cases = int(mask.sum())
        if probe_cases == 0:
            raise ToolError(f"slice {item.name!r} has no probe cases, which cannot happen")
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
    )


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

    Below this size the exact test has too few discordant cases for any outcome to
    be significant, so the suite cannot produce a positive result whatever the model
    did. It is a different quantity from the detection floor and a more alarming
    one: the floor says how large an effect must be, this says whether the question
    is answerable at all.
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
