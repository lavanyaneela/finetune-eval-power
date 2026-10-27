"""Tests for the branches the main files leave untouched.

Kept in their own file rather than scattered, because each one exists to pin a path
that only runs in an unusual situation: a degenerate input, an arm nothing else
exercises, or a printed table nothing else reaches.
"""

from __future__ import annotations

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


def test_the_power_command_reports_a_suite_with_no_floor_as_not_detectable(
    tmp_path, vision_arm, capsys
):
    build_synthetic_runs(
        vision_arm, suite_drop_jitter=(1, 1, 1, 1, 60), held_out_drop_fraction=0.0
    ).write(tmp_path / "vision")
    code = main(["power", "--claim", "90", "--runs", str(tmp_path)])
    captured = capsys.readouterr().out
    assert code in (EXIT_NOT_DETECTABLE, 0)
    assert "arm" in captured


def test_loading_an_arm_the_registry_does_not_know_is_a_tool_error(tmp_path):
    import argparse

    from finetune_eval_power.cli import _load

    arguments = argparse.Namespace(arm="olfactory", runs=str(tmp_path), claim=2.0)
    with pytest.raises(ToolError, match="unknown arm"):
        _load(arguments)
