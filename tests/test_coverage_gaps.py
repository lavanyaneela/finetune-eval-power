"""Tests for the branches the main files leave untouched.

Kept in their own file rather than scattered, because each one exists to pin a path
that only runs in an unusual situation: a degenerate input, an arm nothing else
exercises, or a printed table nothing else reaches.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from tests.conftest import build_synthetic_runs

from finetune_eval_power.analysis import analyse
from finetune_eval_power.cli import EXIT_NOT_DETECTABLE, main
from finetune_eval_power.config import SEQUENCE_ARM, DecisionRule
from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator import sequence as sequence_generator
from finetune_eval_power.power.paired import mcnemar_exact_power, mcnemar_normal_power
from finetune_eval_power.power.unpaired import (
    unpaired_minimum_detectable_effect,
    unpaired_power,
)
from finetune_eval_power.slices.spec import CoverageProfile
from finetune_eval_power.slices.suite import build_suite


def test_the_sequence_renderer_produces_one_row_per_item():
    from finetune_eval_power.slices.suite import build_probe

    probe = build_probe(SEQUENCE_ARM.space, 1)
    batch = sequence_generator.render_batch(
        list(probe.items[:6]), np.random.default_rng(0), SEQUENCE_ARM.space
    )
    assert batch.shape == (6, sequence_generator.MAX_LENGTH)


def test_the_training_renderer_dispatches_to_the_sequence_arm():
    from finetune_eval_power.models.training import render_items
    from finetune_eval_power.slices.suite import build_probe

    probe = build_probe(SEQUENCE_ARM.space, 1)
    batch = render_items(SEQUENCE_ARM, list(probe.items[:4]), np.random.default_rng(0))
    assert batch.shape == (4, sequence_generator.MAX_LENGTH)


def test_the_sequence_arm_has_its_own_named_blocks():
    from finetune_eval_power.models.architectures import SEQUENCE_BLOCKS, block_names

    assert block_names("sequence") == SEQUENCE_BLOCKS


def test_a_normal_power_call_on_a_zero_effect_returns_the_nominal_alpha():
    assert mcnemar_normal_power(480, 0.1, 0.0, alpha=0.05) == 0.05


def test_a_normal_power_call_saturates_when_every_case_changes_and_all_regress():
    assert mcnemar_normal_power(480, 1.0, 1.0) == 1.0


def test_a_non_positive_suite_size_is_refused_by_the_normal_approximation():
    with pytest.raises(ToolError, match="suite size must be positive"):
        mcnemar_normal_power(0, 0.1, 0.01)


def test_exact_power_is_zero_when_no_discordant_count_can_ever_reject():
    """A tiny suite at a tiny discordance has no rejection region at all, so its
    power is exactly zero rather than something small."""
    assert mcnemar_exact_power(4, 0.01, 0.01) == 0.0


def test_unpaired_power_on_a_zero_effect_returns_alpha():
    assert unpaired_power(480, 0.9, 0.0) == 0.05


def test_unpaired_power_saturates_when_both_arms_are_degenerate():
    assert unpaired_power(480, 1.0, 1.0) == 1.0


def test_an_unreachable_unpaired_effect_returns_none():
    assert unpaired_minimum_detectable_effect(2, 0.5) is None


def test_a_profile_that_excludes_a_whole_label_class_is_refused():
    """The reachable emptiness failure. A profile cannot select nothing at all,
    because the space is a full product, but it can exclude every cell of one
    class, and a suite missing a class is not a suite."""
    profile = CoverageProfile(allowed={"shape": ("disc",)})
    from finetune_eval_power.generator.attributes import VISION_SPACE

    with pytest.raises(ToolError, match="no cells for class"):
        build_suite(VISION_SPACE, profile, 4, np.random.default_rng(0))


def test_every_material_regression_counts_as_missed_when_the_suite_has_no_floor(
    vision_arm,
):
    """When the suite cannot reach the target power at any effect size there is no
    floor to compare against, so nothing it missed can be excused as too small."""
    runs = build_synthetic_runs(vision_arm, suite_drop_cases=1, held_out_drop_fraction=0.3)
    analysis = analyse(vision_arm, runs, DecisionRule())
    assert analysis.operating.mde_points is None
    assert len(analysis.undetected) > 0


def test_the_blind_command_prints_the_table_when_something_was_missed(tmp_path, vision_arm, capsys):
    build_synthetic_runs(vision_arm, suite_drop_cases=2, held_out_drop_fraction=0.3).write(
        tmp_path / "vision"
    )
    code = main(["blind", "--runs", str(tmp_path)])
    captured = capsys.readouterr().out
    assert code == EXIT_NOT_DETECTABLE
    assert "regressed past the suite's own floor" in captured
    assert "probe" in captured


def test_a_claim_far_above_both_bars_is_answered_yes_even_on_a_noisy_measurement(
    tmp_path, vision_arm, capsys
):
    """Named for a suite with no floor, on a fixture whose floor is 2.11, and
    asserted `code in (EXIT_NOT_DETECTABLE, 0)`, which accepts the answer and its
    opposite, plus `"arm" in captured`, which is the first word the command always
    prints. Neither half could fail and the name was wrong as well.

    What this measurement actually exercises is a claim large enough to clear both
    bars at once: well above the suite's floor and well above the retraining noise
    that would otherwise refuse it.
    """
    build_synthetic_runs(
        vision_arm, suite_drop_jitter=(1, 1, 1, 1, 60), held_out_drop_fraction=0.0
    ).write(tmp_path / "vision")
    code = main(["power", "--claim", "90", "--runs", str(tmp_path)])
    captured = capsys.readouterr().out
    assert code == 0
    assert "YES" in captured


def test_loading_an_arm_the_registry_does_not_know_is_a_tool_error(tmp_path):
    import argparse

    from finetune_eval_power.cli import _load

    arguments = argparse.Namespace(arm="olfactory", runs=str(tmp_path), claim=2.0)
    with pytest.raises(ToolError, match="unknown arm"):
        _load(arguments)


def test_every_named_slice_has_probe_cases_in_both_arms():
    """The invariant that replaced a guard: the probe enumerates every cell, and a
    slice constrains attributes to values the space has, so no slice can be empty."""
    from finetune_eval_power.config import ARMS
    from finetune_eval_power.slices.spec import enumerate_slices
    from finetune_eval_power.slices.suite import build_probe

    for arm in ARMS.values():
        probe = build_probe(arm.space, 1)
        for item in enumerate_slices(arm.space, arm.pair_attributes):
            assert int(probe.membership(item).sum()) > 0, (arm.name, item.name)


def test_the_probe_reports_its_labels_in_order():
    from finetune_eval_power.config import VISION_ARM
    from finetune_eval_power.slices.suite import build_probe

    probe = build_probe(VISION_ARM.space, 1)
    labels = probe.labels()
    classes = VISION_ARM.space.label_classes()
    assert len(labels) == probe.size
    assert all(
        classes[label] == item.cell[VISION_ARM.space.label_attribute]
        for label, item in zip(labels, probe.items, strict=True)
    )


def test_a_slice_finding_reports_the_width_of_its_retraining_interval(vision_arm):
    from finetune_eval_power.analysis import analyse

    result = analyse(vision_arm, build_synthetic_runs(vision_arm, suite_drop_jitter=(2, 9, 18)))
    finding = result.findings[0]
    assert finding.interval_width_points == pytest.approx(
        finding.interval.high_points - finding.interval.low_points
    )


def test_the_power_command_says_so_when_no_effect_size_would_reach_the_target(
    tmp_path, vision_arm, capsys
):
    """The branch where the suite has no floor at its current size.

    This is an answer, not a refusal, and the distinction was wrong in both
    directions at different times. Its cases change verdict rarely but not never,
    so a larger suite does reach the target and that size is a real remedy. Code 1
    promises to print the suite size that would work, so it has to print one here.
    The sibling case, a discordance of exactly zero, has no such size and is the
    refusal; `test_cli.py` holds that one.
    """
    from finetune_eval_power.cli import EXIT_NOT_DETECTABLE, main

    build_synthetic_runs(
        vision_arm, suite_drop_jitter=(1, 1, 1, 2, 1), held_out_drop_fraction=0.0
    ).write(tmp_path / "vision")
    code = main(["power", "--claim", "40", "--runs", str(tmp_path)])
    captured = capsys.readouterr().out
    assert code == EXIT_NOT_DETECTABLE
    assert "cannot reach" in captured
    assert "needs 3161 cases" in captured


def test_the_word_beside_a_signed_drop_covers_all_three_directions():
    """Positive is worse, negative is better, and zero is neither. The third case
    printed "accuracy fell" until an audit ran the command on a measurement where
    nothing moved."""
    from finetune_eval_power.cli import _direction

    assert _direction(-1.0) == "accuracy rose"
    assert _direction(1.0) == "accuracy fell"
    assert _direction(0.0) == "accuracy unchanged"


def test_a_class_whose_interval_straddles_zero_is_not_called_resolvable(vision_arm, synthetic_runs):
    """`resolvable_by_the_suite` is the property that decides whether a per-class
    reading means anything, and it needs both halves: the drop must clear the
    class's own floor and the interval across retrainings must stay off zero."""
    from finetune_eval_power.analysis import ClassFinding, analyse

    analysis = analyse(vision_arm, synthetic_runs)
    straddling = ClassFinding(
        name="shape=disc",
        suite_cases=120,
        drop_points=9.0,
        discordance=0.2,
        floor_points=4.0,
        interval_low_points=-1.0,
        interval_high_points=12.0,
    )
    assert not straddling.interval_excludes_zero
    assert not straddling.resolvable_by_the_suite

    clear = dataclasses.replace(straddling, interval_low_points=5.0)
    assert clear.interval_excludes_zero
    assert clear.resolvable_by_the_suite

    below_its_floor = dataclasses.replace(clear, drop_points=3.0)
    assert not below_its_floor.resolvable_by_the_suite

    no_floor = dataclasses.replace(clear, floor_points=None)
    assert not no_floor.resolvable_by_the_suite
    assert analysis.worst_class_on_suite in analysis.suite_class_findings


def test_with_no_floor_at_all_every_material_regression_counts_as_missed(tmp_path, vision_arm):
    """When the suite has no floor there is no bar to clear, so the materiality
    test falls back to the interval alone. Without this branch the comparison would
    be against None."""
    from finetune_eval_power.analysis import analyse

    runs = build_synthetic_runs(
        vision_arm, suite_drop_jitter=(1, 1, 1, 2, 1), held_out_drop_fraction=0.3
    )
    analysis = analyse(vision_arm, runs)
    assert analysis.operating.mde_points is None
    assert set(analysis.undetected) == {
        item for item in analysis.findings if item.materially_regressed
    }


def test_the_suite_cannot_be_built_without_cases_for_every_label_class(vision_arm):
    """The invariant that lets the per-class loop carry no emptiness guard.

    Per-class findings are measured on the suite's own cases, so a class with no
    suite cases would mean a paired table over zero cases, which `PairedCounts`
    refuses outright. Rather than a branch in the analysis that nothing can reach,
    the suite builder refuses the profile and this pins that.
    """

    from finetune_eval_power.slices.spec import CoverageProfile
    from finetune_eval_power.slices.suite import build_suite

    label = vision_arm.space.label_attribute
    classes = vision_arm.space.label_classes()
    assert len(classes) > 2
    profile = CoverageProfile(allowed={label: tuple(classes[:2])})
    with pytest.raises(ToolError, match="selects no cells for class"):
        build_suite(vision_arm.space, profile, vision_arm.suite_size, np.random.default_rng(1))
