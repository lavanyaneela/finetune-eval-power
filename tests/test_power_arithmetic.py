"""The statistics, tested against closed forms, against a library, and against itself.

The exact power function is this repository's own work, so it is checked three ways:
its own invariants, agreement with the normal approximation in the regime where the
approximation is good, and agreement with statsmodels on the underlying test.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from finetune_eval_power.errors import ToolError
from finetune_eval_power.power.counts import PairedCounts, paired_counts
from finetune_eval_power.power.paired import (
    mcnemar_exact_power,
    mcnemar_normal_power,
    mcnemar_p_value,
    minimum_detectable_effect,
    minimum_detectable_points,
    rejection_boundary,
    required_suite_size,
)
from finetune_eval_power.power.unpaired import (
    unpaired_minimum_detectable_points,
    unpaired_power,
)


def test_paired_counts_recover_the_two_accuracies():
    before = np.array([True, True, False, False])
    after = np.array([True, False, True, False])
    counts = paired_counts(before, after)
    assert counts.accuracy_before == 0.5
    assert counts.accuracy_after == 0.5
    assert counts.drop == 0.0


def test_the_drop_is_positive_when_the_model_got_worse():
    counts = PairedCounts(both_right=80, only_before=15, only_after=5, both_wrong=0)
    assert counts.drop_points == pytest.approx(10.0)


def test_the_drop_is_negative_when_the_model_got_better():
    counts = PairedCounts(both_right=80, only_before=5, only_after=15, both_wrong=0)
    assert counts.drop_points == pytest.approx(-10.0)


def test_only_the_discordant_cases_contribute_to_the_difference():
    agree_more = PairedCounts(both_right=980, only_before=15, only_after=5, both_wrong=0)
    agree_less = PairedCounts(both_right=80, only_before=15, only_after=5, both_wrong=900)
    assert agree_more.discordant == agree_less.discordant == 20


def test_discordance_is_the_share_of_cases_that_changed_verdict():
    counts = PairedCounts(both_right=70, only_before=20, only_after=10, both_wrong=0)
    assert counts.discordance == pytest.approx(0.3)


def test_the_resolution_floor_is_one_case_expressed_in_points():
    counts = PairedCounts(both_right=400, only_before=0, only_after=0, both_wrong=0)
    assert counts.resolution_floor_points == pytest.approx(0.25)


def test_a_table_over_zero_cases_is_refused():
    with pytest.raises(ToolError, match="zero cases"):
        PairedCounts(both_right=0, only_before=0, only_after=0, both_wrong=0)


def test_a_negative_count_is_refused():
    with pytest.raises(ToolError, match="cannot be negative"):
        PairedCounts(both_right=-1, only_before=0, only_after=0, both_wrong=1)


def test_misaligned_correctness_vectors_are_refused():
    with pytest.raises(ToolError, match="must align"):
        paired_counts(np.array([True]), np.array([True, False]))


def test_non_boolean_vectors_are_refused_because_scores_are_not_verdicts():
    with pytest.raises(ToolError, match="boolean correctness"):
        paired_counts(np.array([1, 0]), np.array([1, 1]))


def test_two_dimensional_vectors_are_refused():
    with pytest.raises(ToolError, match="one dimensional"):
        paired_counts(np.ones((2, 2), dtype=bool), np.ones((2, 2), dtype=bool))


def test_exact_power_rises_with_the_size_of_the_effect():
    values = [mcnemar_exact_power(480, 0.1, drop) for drop in (0.01, 0.02, 0.04, 0.08)]
    assert values == sorted(values)


def test_exact_power_rises_with_the_suite_size():
    values = [mcnemar_exact_power(size, 0.1, 0.02) for size in (120, 240, 480, 960)]
    assert values == sorted(values)


def test_exact_power_at_a_zero_effect_is_the_true_size_of_the_test():
    """And it is below alpha, because the exact binomial test is conservative."""
    size = mcnemar_exact_power(480, 0.05, 0.0, alpha=0.05)
    assert 0.0 < size < 0.05


def test_a_one_sided_test_has_more_power_than_a_two_sided_one():
    one = mcnemar_exact_power(480, 0.08, 0.02, two_sided=False)
    two = mcnemar_exact_power(480, 0.08, 0.02, two_sided=True)
    assert one > two


def test_the_normal_approximation_tracks_the_exact_value_in_its_good_regime():
    exact = mcnemar_exact_power(4000, 0.2, 0.03)
    normal = mcnemar_normal_power(4000, 0.2, 0.03)
    assert abs(exact - normal) < 0.02


def test_the_normal_approximation_is_optimistic_at_small_discordant_counts():
    """The gap is the reason the published floor comes from the exact function."""
    exact = minimum_detectable_points(240, 0.10)
    normal = minimum_detectable_points(240, 0.10, method="normal")
    assert exact is not None and normal is not None
    assert exact > normal


def test_a_drop_larger_than_the_discordance_is_impossible_and_refused():
    with pytest.raises(ToolError, match="cannot outnumber"):
        mcnemar_exact_power(480, 0.05, 0.06)


def test_a_discordance_outside_the_unit_interval_is_refused():
    with pytest.raises(ToolError, match="discordance must be"):
        mcnemar_exact_power(480, 0.0, 0.0)


def test_a_negative_drop_is_refused_because_direction_is_handled_elsewhere():
    with pytest.raises(ToolError, match="non-negative"):
        mcnemar_exact_power(480, 0.1, -0.01)


def test_a_non_positive_suite_size_is_refused():
    with pytest.raises(ToolError, match="suite size must be positive"):
        mcnemar_exact_power(0, 0.1, 0.01)


def test_an_alpha_outside_the_unit_interval_is_refused():
    with pytest.raises(ToolError, match="alpha must be"):
        mcnemar_exact_power(480, 0.1, 0.01, alpha=1.5)


def test_the_rejection_boundary_matches_the_exact_binomial_tail_it_claims():
    for discordant in (6, 10, 25, 40):
        boundary = rejection_boundary(discordant, 0.05)
        assert stats.binom.cdf(boundary, discordant, 0.5) <= 0.025
        assert stats.binom.cdf(boundary + 1, discordant, 0.5) > 0.025


def test_there_is_no_rejection_region_when_too_few_cases_changed_verdict():
    assert rejection_boundary(4, 0.05) == -1
    assert rejection_boundary(0, 0.05) == -1


def test_a_suite_whose_cases_never_change_cannot_reach_the_target_power():
    """Not an edge case to be smoothed over. Below roughly 180 cases at a five
    percent discordance there is no effect size this suite reaches eighty percent
    power against, because the exact test needs a minimum number of discordant
    cases before any outcome is significant at all."""
    assert minimum_detectable_effect(120, 0.02) is None
    assert minimum_detectable_effect(120, 0.05) is None
    assert minimum_detectable_effect(240, 0.05) is not None


def test_the_detectable_effect_shrinks_as_the_suite_grows():
    values = [minimum_detectable_points(size, 0.1) for size in (240, 480, 960, 1920)]
    assert all(value is not None for value in values)
    assert values == sorted(values, reverse=True)


def test_the_detectable_effect_is_reported_in_points_as_a_hundred_times_the_proportion():
    proportion = minimum_detectable_effect(960, 0.1)
    points = minimum_detectable_points(960, 0.1)
    assert points == pytest.approx(proportion * 100.0)


def test_the_suite_size_returned_actually_reaches_the_target_power():
    needed = required_suite_size(0.02, 0.05)
    assert needed is not None
    assert mcnemar_exact_power(needed, 0.05, 0.02) >= 0.80
    assert mcnemar_exact_power(needed - 1, 0.05, 0.02) < 0.80


def test_a_zero_drop_needs_no_suite_size_because_it_is_not_an_effect():
    assert required_suite_size(0.0, 0.05) is None


def test_an_unreachable_effect_returns_none_rather_than_the_loop_bound():
    assert required_suite_size(0.000001, 0.5, ceiling=1000) is None


def test_the_p_value_agrees_with_statsmodels_on_a_known_table():
    from statsmodels.stats.contingency_tables import mcnemar

    expected = float(mcnemar([[0, 24], [9, 0]], exact=True, correction=False).pvalue)
    assert mcnemar_p_value(24, 9) == pytest.approx(expected)


def test_a_table_with_no_discordant_cases_has_a_p_value_of_one():
    assert mcnemar_p_value(0, 0) == 1.0


def test_a_negative_discordant_count_is_refused():
    with pytest.raises(ToolError, match="cannot be negative"):
        mcnemar_p_value(-1, 2)


def test_the_one_sided_p_value_is_half_the_two_sided_one_in_the_tested_direction():
    """Half is the right answer ONLY when the observation lies in the direction
    being tested, which is why the name carries that clause.

    Named without it, this test asserted the general halving rule as a property of
    the code, and it passes against the bug two tests below exist to exclude. Two
    tests in one file then said opposite things about the same function.
    """
    assert mcnemar_p_value(20, 5, two_sided=False) == pytest.approx(mcnemar_p_value(20, 5) / 2.0)
    # And the other direction, where halving is wrong, is the complement.
    assert mcnemar_p_value(5, 20, two_sided=False) == pytest.approx(
        1.0 - mcnemar_p_value(20, 5, two_sided=False) + stats.binom.pmf(20, 25, 0.5)
    )


def test_the_paired_test_sees_a_smaller_effect_than_the_unpaired_one_at_low_discordance():
    """The headline consequence of taking the design seriously."""
    paired = minimum_detectable_points(480, 0.05)
    unpaired = unpaired_minimum_detectable_points(480, 0.90)
    assert paired is not None and unpaired is not None
    assert paired < unpaired


def test_the_two_calculations_converge_as_the_model_becomes_unstable():
    """Pairing buys power in proportion to how much the cases agree with themselves."""
    unpaired = unpaired_minimum_detectable_points(480, 0.90)
    low = minimum_detectable_points(480, 0.05)
    high = minimum_detectable_points(480, 0.25)
    assert unpaired is not None and low is not None and high is not None
    assert abs(high - unpaired) < abs(low - unpaired)


def test_unpaired_power_does_not_depend_on_discordance_because_it_cannot_see_it():
    """The point of the whole paired/unpaired argument, asserted rather than named.

    This called the same function with the same arguments twice and compared the
    results, which asserts that a pure function is deterministic. The property the
    name claims is that the unpaired calculation has no discordance parameter at
    all, so two measurements that differ only in how often cases change verdict
    get the same answer from it, and different answers from the paired one.
    """
    import inspect

    assert "discordance" not in inspect.signature(unpaired_power).parameters
    paired = [minimum_detectable_points(480, psi) for psi in (0.02, 0.20)]
    assert paired[0] != paired[1], "the paired floor must move with discordance"
    # And the consequence: at a discordance where the paired test has ample power,
    # the unpaired answer is the same one it gives everywhere, because it was never
    # told. The two calculations part company precisely here.
    unpaired = unpaired_power(480, 0.9, 0.05)
    assert all(minimum_detectable_points(480, psi) is not None for psi in (0.02, 0.20))
    assert isinstance(unpaired, float)


def test_unpaired_power_rises_with_the_arm_size():
    values = [unpaired_power(size, 0.9, 0.03) for size in (120, 480, 1920)]
    assert values == sorted(values)


def test_unpaired_power_agrees_with_statsmodels_on_the_same_comparison():
    from statsmodels.stats.power import NormalIndPower
    from statsmodels.stats.proportion import proportion_effectsize

    effect = proportion_effectsize(0.90, 0.85)
    expected = NormalIndPower().power(effect_size=effect, nobs1=480, alpha=0.05, ratio=1.0)
    ours = unpaired_power(480, 0.90, 0.05)
    # Tightened from 0.03, which was about twenty times the real agreement, so the
    # implementation could have picked up a wrong variance term and stayed green.
    # ADR-003 and the defense guide both lean on this being a pin.
    assert ours == pytest.approx(expected, abs=0.005)


def test_an_impossible_unpaired_drop_is_refused():
    with pytest.raises(ToolError, match="impossible"):
        unpaired_power(480, 0.5, 0.9)


def test_an_unpaired_baseline_outside_the_unit_interval_is_refused():
    with pytest.raises(ToolError, match="baseline accuracy"):
        unpaired_power(480, 1.5, 0.1)


def test_a_non_positive_unpaired_arm_size_is_refused():
    with pytest.raises(ToolError, match="per-arm size"):
        unpaired_power(0, 0.9, 0.1)


def test_an_unknown_power_method_is_refused_by_name():
    with pytest.raises(ToolError, match="unknown power method"):
        minimum_detectable_effect(480, 0.1, method="telepathy")


def test_the_vectorised_boundary_agrees_with_a_naive_walk_over_candidates():
    """Pins a real bug: the first boundary function made one scipy call per
    candidate, which made a single power evaluation quadratic in the suite size and
    a bisection over it hang. The replacement uses the inverse CDF. This test is the
    naive version, kept as a referee so the fast one cannot drift from it."""
    for discordant in range(0, 60):
        tail = 0.025
        naive = -1
        for candidate in range(discordant // 2 + 1):
            if stats.binom.cdf(candidate, discordant, 0.5) <= tail:
                naive = candidate
            else:
                break
        assert rejection_boundary(discordant, 0.05) == naive, discordant


def test_a_power_evaluation_on_a_large_suite_completes_promptly():
    """The property is that the implementation is not quadratic in the suite size.

    Asserted as a ratio against a small-suite call measured in the same process,
    never as an absolute duration, because an absolute bound is a bound on the
    runner rather than on the code.
    """
    import time

    start = time.perf_counter()
    mcnemar_exact_power(200, 0.2, 0.03)
    small = max(time.perf_counter() - start, 1e-6)
    start = time.perf_counter()
    mcnemar_exact_power(8000, 0.2, 0.03)
    large = time.perf_counter() - start
    assert large < max(0.5, 40.0 * small)


def test_the_one_sided_test_knows_which_direction_it_is_testing():
    """The bug this pins: the one sided p was the two sided p halved.

    Halving is correct only when the observation lies in the direction being
    tested. The alternative here is a regression, so twelve regressions and zero
    improvements must be significant, and twelve improvements against zero
    regressions must not be. Both returned 0.000244 before the fix, which would let
    a one sided regression test reject on an unambiguous improvement. ADR-003 says
    the one sided variant is available and is a parameter rather than a rewrite,
    so it has to be correct even though the analysis only ever calls it two sided.
    """
    regression = mcnemar_p_value(12, 0, two_sided=False)
    improvement = mcnemar_p_value(0, 12, two_sided=False)
    assert regression == pytest.approx(0.5**12)
    assert improvement == pytest.approx(1.0)
    assert regression < improvement


def test_the_one_sided_test_is_not_simply_half_the_two_sided_one():
    """The companion that fails against the bug: on a symmetric table the halving
    rule and the correct tail give different answers, so a test written only
    against the regression direction would have passed against the old code."""
    assert mcnemar_p_value(6, 6, two_sided=False) != pytest.approx(mcnemar_p_value(6, 6) / 2.0)


def test_the_one_sided_tail_matches_the_binomial_it_claims_to_be():
    """Checked against the distribution directly rather than against the library,
    because the library call is the thing being replaced on this path."""
    for only_before, only_after in ((9, 3), (3, 9), (7, 7), (1, 0), (0, 0)):
        discordant = only_before + only_after
        expected = (
            1.0 if discordant == 0 else float(stats.binom.sf(only_before - 1, discordant, 0.5))
        )
        assert mcnemar_p_value(only_before, only_after, two_sided=False) == pytest.approx(expected)
