"""The two noise sources, and the refusal that only the second one justifies."""

from __future__ import annotations

import statistics

import numpy as np
import pytest

from finetune_eval_power.errors import ToolError
from finetune_eval_power.power.bootstrap import (
    bootstrap_drop_interval,
    bootstrap_seed_interval,
)
from finetune_eval_power.power.variance import (
    MINIMUM_SEEDS,
    SETTLE_PERMUTATIONS,
    running_spread,
    seed_variance,
    seeds_to_settle,
    settling_distribution,
)

SPREAD = [3.1, 2.4, 4.0, 2.9, 3.6]


def test_seed_variance_summarises_the_retrainings_it_was_given():
    variance = seed_variance(SPREAD, 1.96)
    assert variance.seeds == 5
    assert variance.minimum_points == 2.4
    assert variance.maximum_points == 4.0
    assert variance.mean_points == pytest.approx(sum(SPREAD) / 5)


def test_the_observed_spread_is_the_range_of_the_retrainings():
    assert seed_variance(SPREAD, 1.96).spread_points == pytest.approx(1.6)


def test_the_noise_floor_is_the_multiplier_times_the_standard_deviation():
    variance = seed_variance(SPREAD, 1.96)
    assert variance.noise_floor_points == pytest.approx(1.96 * variance.standard_deviation_points)


def test_a_claim_below_the_noise_floor_is_swallowed():
    assert seed_variance(SPREAD, 1.96).swallows(0.5)


def test_a_claim_well_above_the_noise_floor_is_not_swallowed():
    assert not seed_variance(SPREAD, 1.96).swallows(10.0)


def test_a_claim_exactly_at_the_floor_is_swallowed_so_the_boundary_is_not_a_pass():
    variance = seed_variance(SPREAD, 1.96)
    assert variance.swallows(variance.noise_floor_points)


def test_identical_retrainings_leave_no_noise_floor_at_all():
    variance = seed_variance([2.0, 2.0, 2.0], 1.96)
    assert variance.standard_deviation_points == 0.0
    assert not variance.swallows(0.001)


def test_the_remedy_for_a_swallowed_claim_is_more_retrainings_and_it_is_priced():
    variance = seed_variance(SPREAD, 1.96)
    needed = variance.seeds_needed_for(0.5)
    assert needed is not None and needed > variance.seeds


def test_a_smaller_claim_needs_quadratically_more_retrainings():
    variance = seed_variance(SPREAD, 1.96)
    assert variance.seeds_needed_for(0.25) > 3 * variance.seeds_needed_for(0.5)


def test_no_retraining_count_is_quoted_when_the_retrainings_agreed_exactly():
    assert seed_variance([1.0, 1.0, 1.0], 1.96).seeds_needed_for(0.5) is None


def test_too_few_retrainings_to_estimate_a_spread_is_refused():
    with pytest.raises(ToolError, match=f"at least {MINIMUM_SEEDS} seeds"):
        seed_variance([1.0, 2.0], 1.96)


def test_a_non_positive_multiplier_is_refused():
    with pytest.raises(ToolError, match="multiplier must be positive"):
        seed_variance(SPREAD, 0.0)


def test_a_non_positive_claim_is_refused_rather_than_silently_accepted():
    variance = seed_variance(SPREAD, 1.96)
    with pytest.raises(ToolError, match="must be positive"):
        variance.swallows(0.0)
    with pytest.raises(ToolError, match="must be positive"):
        variance.seeds_needed_for(-1.0)


def test_the_case_bootstrap_recovers_the_observed_drop_as_its_centre():
    before = np.array([True] * 80 + [False] * 20)
    after = np.array([True] * 60 + [False] * 40)
    interval = bootstrap_drop_interval(before, after, seed=1)
    assert interval.drop_points == pytest.approx(20.0)


def test_the_case_bootstrap_interval_brackets_the_observed_drop():
    before = np.array([True] * 80 + [False] * 20)
    after = np.array([True] * 60 + [False] * 40)
    interval = bootstrap_drop_interval(before, after, seed=1)
    assert interval.low_points < interval.drop_points < interval.high_points


def test_the_case_bootstrap_is_reproducible_for_a_given_seed():
    before = np.array([True] * 50 + [False] * 50)
    after = np.array([True] * 40 + [False] * 60)
    first = bootstrap_drop_interval(before, after, seed=4)
    second = bootstrap_drop_interval(before, after, seed=4)
    assert (first.low_points, first.high_points) == (second.low_points, second.high_points)


def test_a_wider_confidence_level_gives_a_wider_interval():
    before = np.array([True] * 70 + [False] * 30)
    after = np.array([True] * 55 + [False] * 45)
    narrow = bootstrap_drop_interval(before, after, seed=2, confidence=0.80)
    wide = bootstrap_drop_interval(before, after, seed=2, confidence=0.99)
    assert (wide.high_points - wide.low_points) > (narrow.high_points - narrow.low_points)


def test_an_interval_that_stays_above_zero_is_reported_as_excluding_zero():
    before = np.array([True] * 100)
    after = np.array([True] * 60 + [False] * 40)
    assert bootstrap_drop_interval(before, after, seed=3).excludes_zero


def test_an_interval_straddling_zero_is_not_reported_as_excluding_it():
    before = np.array([True, False] * 50)
    after = np.array([True, False] * 50)
    assert not bootstrap_drop_interval(before, after, seed=3).excludes_zero


def test_the_case_bootstrap_resolution_floor_is_one_case():
    before = np.ones(200, dtype=bool)
    after = np.ones(200, dtype=bool)
    assert bootstrap_drop_interval(before, after, seed=3).resolution_floor_points == 0.5


def test_bootstrapping_an_empty_slice_is_refused():
    with pytest.raises(ToolError, match="empty slice"):
        bootstrap_drop_interval(np.array([], dtype=bool), np.array([], dtype=bool), seed=1)


def test_bootstrapping_misaligned_vectors_is_refused():
    with pytest.raises(ToolError, match="must align"):
        bootstrap_drop_interval(np.ones(3, dtype=bool), np.ones(4, dtype=bool), seed=1)


def test_bootstrapping_non_boolean_vectors_is_refused():
    with pytest.raises(ToolError, match="boolean correctness"):
        bootstrap_drop_interval(np.ones(3), np.ones(3), seed=1)


def test_a_non_positive_resample_count_is_refused():
    before = np.ones(10, dtype=bool)
    with pytest.raises(ToolError, match="resamples must be positive"):
        bootstrap_drop_interval(before, before, seed=1, resamples=0)


def test_a_confidence_outside_the_unit_interval_is_refused():
    before = np.ones(10, dtype=bool)
    with pytest.raises(ToolError, match="confidence must be"):
        bootstrap_drop_interval(before, before, seed=1, confidence=1.0)


def test_the_seed_bootstrap_centres_on_the_mean_over_retrainings():
    interval = bootstrap_seed_interval([1.0, 3.0, 5.0, 7.0], seed=1)
    assert interval.drop_points == pytest.approx(4.0)


def test_the_seed_bootstrap_reports_the_retraining_count_as_its_sample_size():
    assert bootstrap_seed_interval([1.0, 2.0, 3.0], seed=1).cases == 3


def test_a_seed_bootstrap_needs_more_than_one_retraining():
    with pytest.raises(ToolError, match="at least two retrainings"):
        bootstrap_seed_interval([1.0], seed=1)


def test_a_seed_bootstrap_with_a_bad_confidence_is_refused():
    with pytest.raises(ToolError, match="confidence must be"):
        bootstrap_seed_interval([1.0, 2.0], seed=1, confidence=0.0)


def test_a_seed_bootstrap_with_no_resamples_is_refused():
    with pytest.raises(ToolError, match="resamples must be positive"):
        bootstrap_seed_interval([1.0, 2.0], seed=1, resamples=0)


def test_retraining_spread_dominates_case_spread_on_an_unstable_measurement():
    """The claim the whole repository rests on, as a property rather than a number.

    Cases that all agree give a narrow case-level interval, while retrainings that
    disagree give a wide one. Reporting only the first is how a measurement comes to
    look far more precise than it is.
    """
    before = np.ones(400, dtype=bool)
    after = np.array([False] * 8 + [True] * 392)
    case_interval = bootstrap_drop_interval(before, after, seed=5)
    seed_interval = bootstrap_seed_interval([0.5, 4.0, 1.0, 6.5, 2.0], seed=5)
    case_width = case_interval.high_points - case_interval.low_points
    seed_width = seed_interval.high_points - seed_interval.low_points
    assert seed_width > case_width


def test_the_multinomial_bootstrap_matches_resampling_the_difference_vector():
    """The fast path is a reformulation, not an approximation, so it is checked
    against the obvious implementation. Resampling cases with replacement draws a
    multinomial over the three values a case can contribute, which is why one
    multinomial call replaces a million index lookups."""
    rng = np.random.default_rng(0)
    before = rng.random(300) < 0.85
    after = before.copy()
    after[rng.choice(300, 40, replace=False)] = False
    after[rng.choice(300, 15, replace=False)] = True

    difference = before.astype(np.int8) - after.astype(np.int8)
    naive_rng = np.random.default_rng(99)
    picks = naive_rng.integers(0, 300, size=(20000, 300))
    naive = np.quantile(difference[picks].mean(axis=1) * 100.0, [0.025, 0.975])

    fast = bootstrap_drop_interval(before, after, seed=99, resamples=20000)
    assert fast.low_points == pytest.approx(naive[0], abs=0.4)
    assert fast.high_points == pytest.approx(naive[1], abs=0.4)


def test_the_bootstrap_centre_is_exactly_the_observed_drop_not_a_resampled_mean():
    before = np.array([True] * 90 + [False] * 10)
    after = np.array([True] * 75 + [False] * 25)
    assert bootstrap_drop_interval(before, after, seed=1).drop_points == pytest.approx(15.0)


def test_a_running_spread_needs_at_least_two_readings():
    """One reading has no spread, and `statistics.stdev` would raise something a
    caller cannot act on."""
    with pytest.raises(ToolError, match="at least two readings"):
        running_spread([1.0])


def test_the_running_spread_reports_one_estimate_per_prefix():
    history = running_spread([1.0, 3.0, 5.0, 7.0])
    assert [count for count, _value in history] == [2, 3, 4]
    assert history[-1][1] == pytest.approx(statistics.stdev([1.0, 3.0, 5.0, 7.0]))


def test_retrainings_that_all_agree_settle_at_the_first_estimate():
    """With a final spread of zero there is no band to be inside, and dividing by it
    is how this returns a nonsense answer instead of the obvious one."""
    assert seeds_to_settle([2.0, 2.0, 2.0, 2.0]) == 2


def test_an_estimate_still_moving_at_the_end_settles_late_rather_than_early():
    """The number is reported, not asserted: if it ever came out at the seed budget
    the honest reading is that the budget is too small."""
    readings = [1.0, 1.0, 1.0, 1.0, 40.0]
    assert seeds_to_settle(readings) == len(readings)


def test_a_wider_band_settles_no_later_than_a_narrower_one():
    readings = [1.0, 1.2, 0.9, 4.0, 1.1, 1.0, 1.05, 0.98]
    assert seeds_to_settle(readings, band=0.5) <= seeds_to_settle(readings, band=0.05)


def test_the_settling_point_is_reported_over_orderings_and_not_one_of_them():
    """The settling point is computed over prefixes, so it is a property of the
    order the retrainings are listed in as much as of the estimator.

    An audit shuffled the same ten readings and got answers from 3 to 10. A single
    ordering presented as a measurement of the estimator is not one, so the
    distribution is what the documents quote.
    """
    readings = [0.6, -0.6, 0.8, -1.2, -5.4, -2.1, -3.7, -1.0, 0.4, -0.4]
    results = settling_distribution(readings)
    assert len(results) == SETTLE_PERMUTATIONS
    assert results[0] == seeds_to_settle(readings), "the listed order is reported first"
    assert len(set(results)) > 1, "if every ordering agreed there would be nothing to report"
    assert all(2 <= value <= len(readings) for value in results)


def test_the_settling_distribution_is_reproducible():
    """It is published, so it has to be the same number on every machine."""
    readings = [0.6, -0.6, 0.8, -1.2, -5.4, -2.1, -3.7, -1.0, 0.4, -0.4]
    assert settling_distribution(readings) == settling_distribution(readings)


def test_a_stability_band_outside_zero_to_one_is_refused():
    """A negative band made the search fall off the end of its own history and
    raise StopIteration, which surfaces through the CLI as a tool error with no
    explanation."""
    with pytest.raises(ToolError, match="fraction in"):
        seeds_to_settle([1.0, 2.0, 3.0], band=-0.1)
    with pytest.raises(ToolError, match="fraction in"):
        seeds_to_settle([1.0, 2.0, 3.0], band=1.5)
