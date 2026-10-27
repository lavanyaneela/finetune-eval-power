"""Properties of the analysis layer, tested against a measurement with a known answer.

The synthetic measurement in conftest plants the result: the suite loses a fixed
number of cases per retraining and held-out probe cases lose a fixed fraction. Every
assertion here is therefore about arithmetic that can be checked by hand, not about
whatever a trained model happened to do.
"""

from __future__ import annotations

import dataclasses

import pytest
from tests.conftest import build_synthetic_runs

from finetune_eval_power.analysis import (
    FLAT,
    IMPROVEMENT,
    REGRESSION,
    SuiteVerdict,
    analyse,
    suite_size_curve,
    suite_size_for,
)
from finetune_eval_power.config import DecisionRule
from finetune_eval_power.errors import ToolError
from finetune_eval_power.power.counts import PairedCounts


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
    assert {item.name for item in analysis.findings} == {item.name for item in analysis.findings}


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


def test_a_slice_counts_as_missed_only_when_it_moved_past_the_suites_own_floor(analysis):
    floor = analysis.operating.mde_points
    for item in analysis.undetected:
        assert item.materially_regressed
        if floor is not None:
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


def test_a_custom_decision_rule_changes_the_floor(vision_arm, synthetic_runs):
    relaxed = analyse(vision_arm, synthetic_runs, DecisionRule(target_power=0.5))
    strict = analyse(vision_arm, synthetic_runs, DecisionRule(target_power=0.95))
    if relaxed.operating.mde_points and strict.operating.mde_points:
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
