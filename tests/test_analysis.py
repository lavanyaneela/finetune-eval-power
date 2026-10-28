"""Properties of the analysis layer, tested against a measurement with a known answer.

The synthetic measurement in conftest plants the result: the suite loses a fixed
number of cases per retraining and held-out probe cases lose a fixed fraction. Every
assertion here is therefore about arithmetic that can be checked by hand, not about
whatever a trained model happened to do.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from tests.conftest import build_synthetic_runs

from finetune_eval_power.analysis import (
    FLAT,
    IMPROVEMENT,
    REGRESSION,
    SuiteVerdict,
    analyse,
    smallest_suite_with_power,
    suite_size_curve,
    suite_size_for,
)
from finetune_eval_power.config import DecisionRule
from finetune_eval_power.errors import ToolError
from finetune_eval_power.power.counts import PairedCounts

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def analysis(vision_arm, synthetic_runs):
    return analyse(vision_arm, synthetic_runs)


def test_analysing_a_measurement_from_another_arm_is_refused(vision_arm, synthetic_runs):
    other = dataclasses.replace(vision_arm, name="sequence")
    with pytest.raises(ToolError, match="measurement is for arm"):
        analyse(other, synthetic_runs)


def test_every_freeze_depth_in_the_measurement_is_summarised(analysis, synthetic_runs):
    assert len(analysis.depths) == len(synthetic_runs.freeze_depths)


def test_a_depth_that_was_not_measured_cannot_be_asked_for(analysis):
    with pytest.raises(ToolError, match="was not analysed"):
        analysis.depth(99)


def test_the_suite_drop_matches_the_cases_the_fixture_planted(analysis, vision_arm):
    expected = 4 / vision_arm.suite_size * 100.0
    assert analysis.operating.suite_drop_points == pytest.approx(expected)


def test_every_named_slice_gets_a_finding(analysis):
    assert len(analysis.findings) == analysis.slices_total
    names = [item.name for item in analysis.findings]
    # Was `set(names) == set(names)`, which cannot fail. The property meant was
    # that no slice is measured twice under one name.
    assert len(set(names)) == len(names)


def test_held_out_slices_are_marked_as_not_covered(analysis):
    held = [item for item in analysis.findings if not item.covered]
    assert held
    assert all(item.suite_cases == 0 for item in held)


def test_covered_slices_have_suite_cases(analysis):
    covered = [item for item in analysis.findings if item.covered]
    assert covered
    assert all(item.suite_cases > 0 for item in covered)


def test_a_slice_the_suite_never_samples_has_no_detection_floor(analysis):
    for item in analysis.findings:
        if item.suite_cases == 0:
            assert item.suite_mde_on_slice_points is None
            assert not item.suite_has_power


def test_every_slice_has_probe_cases_so_every_denominator_is_known(analysis):
    assert all(item.probe_cases > 0 for item in analysis.findings)


def test_a_slice_resolution_floor_is_one_probe_case(analysis):
    item = analysis.findings[0]
    assert item.resolution_floor_points == pytest.approx(100.0 / item.probe_cases)


def test_the_operating_depth_is_the_shallowest_the_suite_does_not_object_to(analysis):
    quiet = [summary for summary in analysis.depths if summary.suite_is_quiet]
    assert analysis.operating_depth == min(summary.freeze_depth for summary in quiet)


def test_the_operating_depth_falls_back_to_the_deepest_when_the_suite_objects_everywhere(
    vision_arm,
):
    loud = build_synthetic_runs(vision_arm, suite_drop_cases=120)
    analysis = analyse(vision_arm, loud)
    assert analysis.operating_depth == max(vision_arm.freeze_depths)


def test_a_verdict_is_flat_when_the_test_does_not_reject():
    verdict = SuiteVerdict(
        seed=1,
        freeze_depth=0,
        counts=PairedCounts(both_right=470, only_before=5, only_after=5, both_wrong=0),
        p_value=0.9,
    )
    assert verdict.direction == FLAT
    assert not verdict.flags_regression


def test_a_significant_move_downward_is_a_regression():
    verdict = SuiteVerdict(
        seed=1,
        freeze_depth=0,
        counts=PairedCounts(both_right=400, only_before=60, only_after=5, both_wrong=15),
        p_value=0.001,
    )
    assert verdict.direction == REGRESSION
    assert verdict.flags_regression


def test_a_significant_move_upward_is_an_improvement_and_is_not_flagged():
    """The case that makes the headline possible: the suite can be significant and
    still raise no alarm, because teams act on regressions and not on improvements."""
    verdict = SuiteVerdict(
        seed=1,
        freeze_depth=0,
        counts=PairedCounts(both_right=400, only_before=5, only_after=60, both_wrong=15),
        p_value=0.001,
    )
    assert verdict.direction == IMPROVEMENT
    assert not verdict.flags_regression


def test_nothing_counts_as_missed_when_the_suite_did_flag_a_regression(vision_arm):
    loud = build_synthetic_runs(vision_arm, suite_drop_cases=120)
    assert analyse(vision_arm, loud).undetected == ()


def test_a_slice_counts_as_missed_only_when_it_moved_past_the_suites_own_floor(
    vision_arm, realistic_runs
):
    """On a measurement that has a floor. Guarded with "if floor is not None" and
    run against a fixture whose floor is always None, this asserted nothing."""
    analysis = analyse(vision_arm, realistic_runs)
    floor = analysis.operating.mde_points
    assert floor is not None
    assert analysis.undetected
    for item in analysis.undetected:
        assert item.materially_regressed
        assert item.drop_points > floor


def test_the_worst_missed_slice_is_the_one_with_the_largest_drop(analysis):
    missed = analysis.undetected
    if missed:
        assert analysis.worst_undetected.drop_points == max(item.drop_points for item in missed)


def test_there_is_no_worst_missed_slice_when_nothing_was_missed(vision_arm):
    loud = build_synthetic_runs(vision_arm, suite_drop_cases=120)
    assert analyse(vision_arm, loud).worst_undetected is None


def test_the_planted_held_out_regression_is_found(vision_arm):
    """The fixture drops a quarter of held-out probe cases, so the analysis must
    report held-out slices regressing by roughly twenty five points."""
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=2, held_out_drop_fraction=0.25)
    analysis = analyse(vision_arm, runs)
    held = [item for item in analysis.findings if not item.covered]
    assert max(item.drop_points for item in held) > 15.0


def test_a_measurement_with_no_held_out_damage_reports_nothing_missed(vision_arm):
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=2, held_out_drop_fraction=0.0)
    assert analyse(vision_arm, runs).undetected == ()


def test_the_seed_interval_is_computed_over_retrainings_not_over_cases(analysis):
    assert analysis.operating.suite_drop_interval.cases == len(analysis.seeds)


def test_the_variance_summary_covers_every_retraining(analysis):
    assert analysis.variance.seeds == len(analysis.seeds)


def test_a_custom_decision_rule_changes_the_floor(vision_arm, realistic_runs):
    """The guard here used to be "if both are not None", on a fixture where both
    are always None, so the body never ran and target power threading was
    untested."""
    relaxed = analyse(vision_arm, realistic_runs, DecisionRule(target_power=0.5))
    strict = analyse(vision_arm, realistic_runs, DecisionRule(target_power=0.95))
    assert relaxed.operating.mde_points is not None
    assert strict.operating.mde_points is not None
    assert relaxed.operating.mde_points < strict.operating.mde_points


def test_the_suite_size_curve_covers_every_size_requested():
    rows = suite_size_curve(0.1, (240, 480, 960))
    assert [row["suite_size"] for row in rows] == [240, 480, 960]


def test_the_suite_size_curve_falls_as_the_suite_grows():
    rows = suite_size_curve(0.1, (240, 480, 960, 1920))
    values = [row["exact_points"] for row in rows]
    assert all(value is not None for value in values)
    assert values == sorted(values, reverse=True)


def test_the_required_suite_size_grows_as_the_claim_shrinks():
    assert suite_size_for(1.0, 0.1) > suite_size_for(4.0, 0.1)


def test_the_smallest_suite_with_power_is_the_point_where_a_floor_first_exists():
    """A different quantity from the detection floor, and a more alarming one. The
    floor says how large an effect must be; this says whether the question is
    answerable at all."""
    from finetune_eval_power.analysis import smallest_suite_with_power
    from finetune_eval_power.power.paired import minimum_detectable_points

    size = smallest_suite_with_power(0.02)
    assert size is not None
    assert minimum_detectable_points(size, 0.02) is not None
    assert minimum_detectable_points(size - 1, 0.02) is None


def test_a_more_stable_model_needs_a_larger_suite_before_it_has_any_power():
    from finetune_eval_power.analysis import smallest_suite_with_power

    assert smallest_suite_with_power(0.01) > smallest_suite_with_power(0.10)


def test_no_suite_size_inside_the_ceiling_helps_at_a_vanishing_discordance():
    from finetune_eval_power.analysis import smallest_suite_with_power

    assert smallest_suite_with_power(1e-6, ceiling=512) is None


def test_the_retraining_interval_describes_one_run_rather_than_the_mean(analysis):
    """The comparison the report makes only works if both widths are about a single
    measurement. This one is two noise floors wide by construction, which is the
    spread of one run, not of the average of ten."""
    assert analysis.retraining_interval_width_points == pytest.approx(
        2.0 * analysis.variance.noise_floor_points
    )


def test_retrainings_that_agree_exactly_leave_no_retraining_interval(analysis):
    """The synthetic measurement loses the same cases every time, so the only
    uncertainty left is case sampling. Useful as a control: it shows the two widths
    are measuring different things rather than two views of one thing."""
    assert analysis.suite_case_interval_width_points > 0.0
    assert analysis.retraining_interval_width_points == 0.0


def test_disagreeing_retrainings_produce_the_wider_interval_of_the_two(vision_arm):
    """The finding the report states, as a property. On a measurement whose
    retrainings disagree, retraining alone moves a single reading further than
    resampling that reading's eval cases does."""
    runs = build_synthetic_runs(vision_arm, suite_drop_jitter=(4, 14, 26, 9, 19))
    result = analyse(vision_arm, runs)
    assert result.retraining_interval_width_points > result.suite_case_interval_width_points


def test_the_case_interval_comes_from_one_retraining_not_an_average(vision_arm, synthetic_runs):
    """A fixture where every retraining is identical must still produce a case-level
    interval, because case resampling is a property of one run."""
    result = analyse(vision_arm, synthetic_runs)
    assert result.suite_case_interval_width_points > 0.0


def test_every_label_class_is_measured_on_the_suites_own_cases(analysis, vision_arm):
    """The aggregate being explained is the suite's, so the per-class figures that
    explain it have to come from the suite's cases. An earlier version explained a
    suite number with probe numbers and the arithmetic did not add up."""
    assert len(analysis.suite_class_findings) == len(vision_arm.space.label_classes())
    assert all(item.suite_cases > 0 for item in analysis.suite_class_findings)


def test_the_class_drops_average_to_the_published_aggregate(analysis):
    mean = sum(item.drop_points for item in analysis.suite_class_findings) / len(
        analysis.suite_class_findings
    )
    assert mean == pytest.approx(analysis.operating.suite_drop_points, abs=1e-6)


def test_each_class_carries_the_floor_that_applies_at_its_own_case_count(
    vision_arm, realistic_runs
):
    """Not the aggregate's floor. Putting a per-class drop beside the aggregate's
    floor compares two numbers computed for different denominators, which is the
    mistake this field exists to prevent.

    This is the test for the claim ADR-005 and the README build an argument on, and
    on the default fixture every class floor is None, so it asserted nothing.
    """
    from finetune_eval_power.power.paired import minimum_detectable_points

    analysis = analyse(vision_arm, realistic_runs)

    checked = 0
    for item in analysis.suite_class_findings:
        assert item.floor_points is not None, "this fixture is chosen to give every class a floor"
        expected = minimum_detectable_points(item.suite_cases, item.discordance)
        assert item.floor_points == pytest.approx(expected)
        assert item.suite_cases < analysis.suite_size
        checked += 1
    assert checked == analysis.label_classes


def test_a_class_below_its_own_floor_is_not_reported_as_resolvable(vision_arm, realistic_runs):
    """Every class on this fixture sits below its own floor, which is the situation
    the README's worst-class paragraph is about. On the default fixture no class
    has a floor at all, so this loop ran zero assertions."""
    analysis = analyse(vision_arm, realistic_runs)
    below = 0
    for item in analysis.suite_class_findings:
        assert item.floor_points is not None
        if item.drop_points <= item.floor_points:
            assert not item.resolvable_by_the_suite
            below += 1
    assert below, "no class is below its floor here, so this proves nothing"


def test_the_improving_total_counts_only_the_classes_that_improved(analysis):
    expected = -sum(
        item.drop_points for item in analysis.suite_class_findings if item.drop_points < 0.0
    )
    assert analysis.improving_class_total_points == pytest.approx(expected)
    assert analysis.improving_classes + 1 <= analysis.label_classes


def test_slices_nest_so_the_region_count_never_exceeds_the_slice_count(analysis):
    assert len(analysis.maximal_undetected) <= len(analysis.undetected)
    names = {item.name for item in analysis.maximal_undetected}
    for item in analysis.undetected:
        if item.name in names:
            continue
        parts = set(item.name.split(" & "))
        assert any(set(other.split(" & ")) < parts for other in names)


def test_slices_that_moved_the_other_way_are_counted_too(vision_arm, synthetic_runs):
    """A list of eight selected from forty invites the question, so the count of
    slices that moved up is published beside it.

    Asserted `improving_slices >= 0` on a sum of booleans, which cannot fail, and
    then looped over a fixture where nothing improves. Built here by reversing the
    measurement so the held-out slices move up instead of down.
    """
    improved = analyse(
        vision_arm,
        dataclasses.replace(
            synthetic_runs,
            probe_before=synthetic_runs.probe_after,
            probe_after=synthetic_runs.probe_before,
        ),
    )
    assert improved.improving_slices > 0
    for item in improved.findings:
        if item.drop_points < 0.0 and item.interval.high_points < 0.0:
            assert item not in improved.undetected


def test_a_significant_verdict_is_counted_even_when_it_is_an_improvement(analysis):
    """`suite_flagged_in_seeds` counts regressions only, so the number of
    significant verdicts is published separately rather than letting a document say
    the suite found nothing."""
    assert analysis.significant_verdict_seeds >= analysis.operating.seeds_flagging_regression


def test_the_improving_side_collapses_nested_slices_the_same_way(vision_arm, synthetic_runs):
    """Built by reversing the fixture, so the held-out slices improve instead of
    regressing and nest exactly as the missed ones do.

    The missed count is corrected for nesting before it is published. Correcting
    that side and not this one would be choosing which number gets the treatment
    that makes it smaller.
    """
    improved = analyse(
        vision_arm,
        dataclasses.replace(
            synthetic_runs,
            probe_before=synthetic_runs.probe_after,
            probe_after=synthetic_runs.probe_before,
        ),
    )
    assert improved.improving_slices > improved.improving_regions
    names = {item.name for item in improved.improving_findings}
    expected = sum(
        1
        for name in names
        if not any(
            other != name and set(other.split(" & ")) < set(name.split(" & ")) for other in names
        )
    )
    assert improved.improving_regions == expected


def test_an_underpowered_slice_is_published_as_the_size_it_would_need(vision_arm):
    """ "Sampled too thinly" reads as "nothing would help", which is not what was
    measured. Each one is published with the suite size that would give it power."""
    runs = build_synthetic_runs(vision_arm, held_out_drop_fraction=0.05)
    analysis = analyse(vision_arm, runs)
    rows = analysis.slices_short_of_power
    assert len(rows) == analysis.underpowered_covered_slices
    assert rows, "the fixture is meant to produce underpowered covered slices"
    for name, cases, needed in rows:
        finding = next(item for item in analysis.findings if item.name == name)
        assert finding.covered and not finding.suite_has_power
        assert cases == finding.suite_cases
        assert needed is not None and needed > cases


def test_a_slice_whose_cases_never_change_verdict_names_no_size_instead_of_raising(
    vision_arm,
):
    """The bug this pins: the size was asked for at a discordance of zero.

    `smallest_suite_with_power` rejects a discordance of zero, correctly, because no
    suite size follows from a sample that carries no information. The caller asked
    anyway, for every slice in the list, and a slice whose cases never change
    verdict is the extreme of the very condition that puts it in that list. It
    raised a ToolError out of a property that documents read, so the measurement
    could not be collected at all on such a measurement.
    """
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=0, held_out_drop_fraction=0.0)
    analysis = analyse(vision_arm, runs)
    rows = analysis.slices_short_of_power
    assert rows, "with nothing moving, every covered slice is short of power"
    assert all(needed is None for _name, _cases, needed in rows)


def test_that_zero_discordance_test_would_fail_against_the_unguarded_version(vision_arm):
    """The guard above is only worth having if it fails when removed.

    A regression test that passes against the bug it was written for is the failure
    mode this portfolio has already shipped once, so the broken call is reconstructed
    here and asserted to raise.
    """
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=0, held_out_drop_fraction=0.0)
    analysis = analyse(vision_arm, runs)
    unguarded = [item for item in analysis.findings if item.covered and not item.suite_has_power]
    assert unguarded
    with pytest.raises(ToolError, match="discordance must be in"):
        smallest_suite_with_power(unguarded[0].suite_discordance)


def test_the_alpha_the_caller_passes_reaches_the_verdicts_and_not_only_the_floor(
    vision_arm, synthetic_runs
):
    """The bug this pins: `SuiteVerdict.significant` built a fresh default
    `DecisionRule` instead of using the one `analyse` was given.

    The alpha threaded correctly into `minimum_detectable_points`, so the detection
    floor moved with it while every flag, operating depth and missed-slice list
    stayed pinned at 0.05. A caller asking for a stricter test silently got two
    different significance levels in one answer, which is worse than either. The
    config calls alpha "the significance level for every hypothesis test the tool
    runs", which was then true of none of them.
    """
    strict = analyse(vision_arm, synthetic_runs, DecisionRule(alpha=1e-6))
    loose = analyse(vision_arm, synthetic_runs, DecisionRule(alpha=0.5))
    assert loose.significant_verdict_seeds > strict.significant_verdict_seeds
    assert all(verdict.alpha == 0.5 for verdict in loose.operating.verdicts)


def test_that_alpha_test_would_fail_against_a_hard_wired_default(vision_arm, synthetic_runs):
    """The companion. A verdict judged at the default alpha is a different verdict
    from one judged at the alpha asked for, so comparing against the default is
    what distinguishes the fix from the bug."""
    loose = analyse(vision_arm, synthetic_runs, DecisionRule(alpha=0.5))
    default = analyse(vision_arm, synthetic_runs)
    assert loose.significant_verdict_seeds != default.significant_verdict_seeds


def test_a_slice_prescription_comes_from_the_suites_own_rate_not_the_probes(vision_arm):
    """The bug this pins: the published "cases it would need" was computed from the
    slice's rate of changed verdicts on the PROBE while the sentence around it
    prescribed a suite size.

    The two rates differ by a factor of four on the shipped measurement, so the
    README advertised 140 cases where the suite's own behaviour says 592. A
    prescription about an instrument has to be derived from that instrument.
    """
    runs = build_synthetic_runs(vision_arm, held_out_drop_fraction=0.05)
    analysis = analyse(vision_arm, runs)
    rows = analysis.slices_short_of_power
    assert rows
    for name, _cases, needed in rows:
        finding = next(item for item in analysis.findings if item.name == name)
        from finetune_eval_power.analysis import smallest_suite_with_power

        assert needed == smallest_suite_with_power(finding.suite_discordance)


def test_the_two_rates_a_slice_carries_are_not_the_same_number(vision_arm, synthetic_runs):
    """The companion. If the suite and probe rates were equal on this fixture the
    test above would pass against the bug it was written for."""
    analysis = analyse(vision_arm, synthetic_runs)
    assert any(
        abs(item.suite_discordance - item.discordance) > 1e-9
        for item in analysis.findings
        if item.suite_cases > 0
    )


def test_the_seed_budget_sits_above_the_point_the_spread_estimate_settles(
    vision_arm, synthetic_runs
):
    """`config.py` told the reader that ADR-004 recorded a variance stability
    measurement. ADR-004 argued it in words and recorded none.

    This asserted `seeds_to_settle <= len(readings)`, which the function's own
    construction guarantees, and `<=` permits equality, which is the opposite of
    "sits above". The property worth asserting is on the shipped measurement.
    """
    from pathlib import Path as _Path

    from finetune_eval_power.runs import load_runs

    shipped = analyse(vision_arm, load_runs(_Path(ROOT / "fixtures" / "runs" / "vision")))
    assert shipped.seeds_to_settle_typical < len(shipped.seed_readings_points)
    local = analyse(vision_arm, synthetic_runs)
    assert 2 <= local.seeds_to_settle <= len(local.seed_readings_points)


def test_a_short_seed_budget_sets_the_refusal_bar_somewhere_else(vision_arm):
    """The point of measuring the budget at all: the refusal threshold is computed
    from the spread estimate, so an estimate taken from too few retrainings moves
    the bar and leaves no trace that it did."""
    runs = build_synthetic_runs(
        vision_arm,
        seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
        suite_drop_jitter=(2, 2, 2, 30, 2, 2, 2, 2, 2, 2),
    )
    analysis = analyse(vision_arm, runs)
    assert analysis.short_budget_noise_floor_points != pytest.approx(
        analysis.variance.noise_floor_points
    )


def test_a_slice_that_moved_but_not_past_the_floor_is_not_reported_as_missed(vision_arm):
    """The materiality bar, on a measurement that actually has one.

    Every other test in this suite runs against a fixture whose suite drops four
    cases in four hundred and eighty, which is too little for the suite to have a
    floor at all, so they all exercise the branch where there is no bar to clear.
    The published headline uses the other branch: a slice counts as missed only if
    it moved further than the suite's own floor. Coverage was what noticed that the
    branch carrying the repository's main claim had no test.
    """
    runs = build_synthetic_runs(
        vision_arm,
        suite_drop_cases=18,
        suite_gain_cases=20,
        held_out_drop_fraction=0.06,
        rng_seed=1,
    )
    analysis = analyse(vision_arm, runs)
    floor = analysis.operating.mde_points
    assert floor is not None, "this fixture is chosen to have a floor"
    assert analysis.operating.suite_is_quiet

    missed = set(analysis.undetected)
    material = {item for item in analysis.findings if item.materially_regressed}
    assert missed < material, "some slice must fall below the floor or this proves nothing"
    for item in material - missed:
        assert item.drop_points <= floor
    for item in missed:
        assert item.drop_points > floor


def test_the_referees_rate_reports_fewer_blind_slices_than_the_suites_own(vision_arm):
    """Why the published count uses the suite's rate and not the probe's.

    Both are computed from the slice's suite case count; they differ only in whose
    rate of changed verdicts goes in. The probe's rate is steadier, which makes the
    suite look more powered than it is, and the published blindness count was
    smaller than the measurement supports because of it. Publishing both makes the
    choice visible rather than buried, and this asserts the direction the prose
    claims.
    """
    runs = build_synthetic_runs(
        vision_arm, suite_drop_cases=18, suite_gain_cases=20, held_out_drop_fraction=0.2
    )
    analysis = analyse(vision_arm, runs)
    at_probe_rate = analysis.slices_without_power_at_probe_rate
    at_suite_rate = sum(1 for item in analysis.findings if not item.suite_has_power)
    assert at_suite_rate >= at_probe_rate
    assert any(
        item.suite_mde_on_slice_points != item.probe_rate_mde_on_slice_points
        for item in analysis.findings
        if item.suite_cases > 0
    ), "the two rates must differ somewhere or this proves nothing"


def test_a_prescribed_suite_size_is_priced_at_the_rule_it_was_selected_under(
    vision_arm, synthetic_runs
):
    """The bug this pins: slices were SELECTED into the underpowered list at the
    caller's rule and then PRICED at the default 0.05 and 80%.

    It is the same defect as the hard-wired alpha in `SuiteVerdict`, one layer out,
    and it produced prescriptions smaller than the case count the slice already
    has: "this slice has 120 cases and no power at your rule; it needs 68 cases".
    """
    strict = analyse(vision_arm, synthetic_runs, DecisionRule(alpha=1e-4, target_power=0.99))
    default = analyse(vision_arm, synthetic_runs)
    for _name, cases, needed in strict.slices_short_of_power:
        if needed is not None:
            assert needed > cases, "a prescription below the current size is not a prescription"
    strict_sizes = {name: needed for name, _c, needed in strict.slices_short_of_power}
    default_sizes = {name: needed for name, _c, needed in default.slices_short_of_power}
    shared = set(strict_sizes) & set(default_sizes)
    assert shared, "the two rules must share a slice or this compares nothing"
    assert any(strict_sizes[name] != default_sizes[name] for name in shared)


def test_the_settling_point_is_published_as_a_distribution_not_as_one_ordering(
    vision_arm, synthetic_runs
):
    """ADR-004 used to quote a single ordering's settling point as a measurement of
    the estimator. It is computed over prefixes, so it is partly a measurement of
    which retraining happened to be listed fifth.

    This asserted a count is inside its own range and a percentage is between zero
    and a hundred, on a fixture whose five readings are identical so both are
    degenerate. The property is that orderings disagree and the published figure is
    the middle of them, which needs readings that differ.
    """
    from finetune_eval_power.power.variance import settling_distribution

    runs = build_synthetic_runs(
        vision_arm, seeds=tuple(range(1, 11)), suite_drop_jitter=(2, 9, 3, 8, 14, 4)
    )
    analysis = analyse(vision_arm, runs)
    spread = settling_distribution(analysis.seed_readings_points)
    assert len(set(spread)) > 1, "orderings must disagree or there is nothing to report"
    assert min(spread) <= analysis.seeds_to_settle_typical <= max(spread)
    assert analysis.never_settles_share_pct == pytest.approx(
        100.0
        * sum(1 for value in spread if value >= len(analysis.seed_readings_points))
        / len(spread)
    )


def test_a_measurement_whose_retrainings_disagree_late_often_fails_to_settle(vision_arm):
    """The uncomfortable half: on readings that keep moving, the share of orderings
    that have not settled by the last retraining is large, which says the budget is
    too small rather than that the estimate is fine."""
    runs = build_synthetic_runs(
        vision_arm,
        seeds=tuple(range(1, 11)),
        suite_drop_jitter=(2, 2, 2, 2, 2, 2, 2, 2, 2, 60),
    )
    analysis = analyse(vision_arm, runs)
    assert analysis.never_settles_share_pct > 0.0
