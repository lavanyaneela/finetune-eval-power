"""The few properties that genuinely require training a model.

Everything here is marked slow and scaled down to a toy configuration. The suite is
not a place to re-measure the sweep: that is what the committed measurement is for.
What these tests pin are the invariants the measurement would be meaningless
without.
"""

from __future__ import annotations

import numpy as np
import pytest

keras = pytest.importorskip("keras", reason="needs the training extra")

from finetune_eval_power.models.architectures import (  # noqa: E402
    LABEL_HEAD,
    SEQUENCE_BLOCKS,
    VISION_BLOCKS,
    apply_freeze,
    block_names,
    build_model,
)
from finetune_eval_power.errors import ToolError  # noqa: E402

pytestmark = pytest.mark.slow


def test_the_vision_model_has_one_head_with_one_output_per_label_class(vision_arm):
    model = build_model("vision", vision_arm.space)
    assert model.get_layer(LABEL_HEAD).units == len(vision_arm.space.label_classes())


def test_every_named_vision_block_exists_in_the_model(vision_arm):
    model = build_model("vision", vision_arm.space)
    for name in VISION_BLOCKS:
        assert model.get_layer(name) is not None


def test_every_named_sequence_block_exists_in_the_model():
    from finetune_eval_power.config import SEQUENCE_ARM

    model = build_model("sequence", SEQUENCE_ARM.space)
    for name in SEQUENCE_BLOCKS:
        assert model.get_layer(name) is not None


def test_an_unknown_arm_has_no_blocks_and_no_model():
    with pytest.raises(ToolError, match="unknown arm"):
        block_names("olfactory")
    with pytest.raises(ToolError, match="unknown arm"):
        build_model("olfactory", None)


def test_freezing_to_depth_zero_leaves_every_block_trainable(vision_arm):
    model = build_model("vision", vision_arm.space)
    apply_freeze(model, "vision", 0)
    assert all(model.get_layer(name).trainable for name in VISION_BLOCKS)


def test_freezing_to_the_maximum_depth_freezes_every_block(vision_arm):
    model = build_model("vision", vision_arm.space)
    apply_freeze(model, "vision", len(VISION_BLOCKS))
    assert not any(model.get_layer(name).trainable for name in VISION_BLOCKS)


def test_the_head_stays_trainable_at_every_depth_because_the_task_did_not_change(vision_arm):
    model = build_model("vision", vision_arm.space)
    for depth in range(len(VISION_BLOCKS) + 1):
        apply_freeze(model, "vision", depth)
        assert model.get_layer(LABEL_HEAD).trainable


def test_freezing_is_a_prefix_so_depth_two_freezes_the_first_two_blocks(vision_arm):
    model = build_model("vision", vision_arm.space)
    apply_freeze(model, "vision", 2)
    states = [model.get_layer(name).trainable for name in VISION_BLOCKS]
    assert states == [False, False, True, True]


def test_applying_a_shallower_freeze_afterwards_releases_the_earlier_blocks(vision_arm):
    model = build_model("vision", vision_arm.space)
    apply_freeze(model, "vision", 3)
    apply_freeze(model, "vision", 1)
    states = [model.get_layer(name).trainable for name in VISION_BLOCKS]
    assert states == [False, True, True, True]


def test_a_freeze_depth_beyond_the_block_count_is_refused_rather_than_clamped(vision_arm):
    model = build_model("vision", vision_arm.space)
    with pytest.raises(ToolError, match="outside 0 to"):
        apply_freeze(model, "vision", len(VISION_BLOCKS) + 1)


def test_a_negative_freeze_depth_is_refused(vision_arm):
    model = build_model("vision", vision_arm.space)
    with pytest.raises(ToolError, match="outside 0 to"):
        apply_freeze(model, "vision", -1)


def test_the_model_contains_no_normalisation_layer_that_would_leak_through_a_freeze(
    vision_arm,
):
    """A normalisation layer keeps updating its running statistics while frozen, so
    a fully frozen backbone would still move and the freeze control would be a lie."""
    model = build_model("vision", vision_arm.space)
    names = [type(layer).__name__ for layer in model.layers]
    assert not any("Normalization" in name or "BatchNorm" in name for name in names)


def test_training_the_same_seed_twice_gives_identical_predictions(small_arm):
    """Determinism is not a nicety here. Seed-to-seed spread is the denominator of
    the refusal rule, so a run that is not reproducible within a seed would make
    that spread a measurement of the framework rather than of the training."""
    from finetune_eval_power.models.training import build_fixed_cases, train_seed

    cases = build_fixed_cases(small_arm)
    first = train_seed(small_arm, 5, cases)
    second = train_seed(small_arm, 5, cases)
    assert np.array_equal(first.suite_before, second.suite_before)
    assert np.array_equal(first.depths[0].suite_after, second.depths[0].suite_after)


def test_a_fully_frozen_backbone_still_moves_because_the_head_is_trainable(small_arm):
    from finetune_eval_power.models.training import build_fixed_cases, train_seed

    cases = build_fixed_cases(small_arm)
    result = train_seed(small_arm, 5, cases)
    deepest = result.depths[-1]
    assert deepest.freeze_depth == max(small_arm.freeze_depths)
    assert deepest.suite_after.shape == result.suite_before.shape


def test_the_suite_and_probe_are_rendered_once_and_shared_by_every_seed(small_arm):
    """If these moved with the training seed, the seed spread would mix training
    noise with generator noise and the refusal rule would be measuring the wrong
    thing."""
    from finetune_eval_power.models.training import build_fixed_cases

    first = build_fixed_cases(small_arm)
    second = build_fixed_cases(small_arm)
    assert np.array_equal(first.suite_x, second.suite_x)
    assert np.array_equal(first.probe_x, second.probe_x)


def test_the_curve_records_one_point_per_epoch_plus_the_starting_value(small_arm):
    from finetune_eval_power.models.training import build_fixed_cases, train_seed

    cases = build_fixed_cases(small_arm)
    result = train_seed(small_arm, 5, cases)
    assert result.depths[0].suite_curve.shape == (small_arm.finetune_epochs + 1,)


def test_rendering_refuses_an_arm_it_does_not_know(small_arm):
    import dataclasses

    from finetune_eval_power.models.training import render_items

    with pytest.raises(ToolError, match="unknown arm"):
        render_items(dataclasses.replace(small_arm, name="olfactory"), [], None)
