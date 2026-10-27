"""Properties of the two renderers.

The point of these is that every attribute must actually change the data. A slice is
a claim about a capability, and if an attribute is decorative then a regression
measured on that slice means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from finetune_eval_power.generator import sequence as sequence_generator
from finetune_eval_power.generator import vision as vision_generator
from finetune_eval_power.generator.attributes import (
    SEQUENCE_SPACE,
    VISION_SPACE,
    Item,
    enumerate_cells,
)


def vision_cell(**overrides: str) -> dict[str, str]:
    cell = {
        "shape": "disc",
        "texture": "solid",
        "scale": "large",
        "rotation": "r000",
        "contrast": "high",
        "occlusion": "none",
    }
    cell.update(overrides)
    return cell


def sequence_cell(**overrides: str) -> dict[str, str]:
    cell = {
        "motif": "ascend",
        "length": "long",
        "amplitude": "strong",
        "noise": "clean",
        "ordering": "forward",
    }
    cell.update(overrides)
    return cell


def test_a_rendered_image_has_the_declared_shape_and_dtype():
    image = vision_generator.render(vision_cell(), np.random.default_rng(0))
    assert image.shape == (vision_generator.GRID, vision_generator.GRID, 1)
    assert image.dtype == np.float32


def test_rendering_is_reproducible_for_the_same_generator_state():
    first = vision_generator.render(vision_cell(), np.random.default_rng(3))
    second = vision_generator.render(vision_cell(), np.random.default_rng(3))
    assert np.array_equal(first, second)


def test_two_draws_from_one_generator_differ_so_the_fixture_is_not_degenerate():
    rng = np.random.default_rng(3)
    first = vision_generator.render(vision_cell(), rng)
    second = vision_generator.render(vision_cell(), rng)
    assert not np.array_equal(first, second)


@pytest.mark.parametrize("shape", VISION_SPACE.attributes["shape"])
def test_every_shape_puts_some_ink_on_the_canvas(shape: str):
    image = vision_generator.render(vision_cell(shape=shape), np.random.default_rng(1))
    assert (image > 0.5).sum() > 20


def test_a_larger_scale_puts_more_ink_on_the_canvas_than_a_smaller_one():
    small = vision_generator.render(vision_cell(scale="small"), np.random.default_rng(5))
    large = vision_generator.render(vision_cell(scale="large"), np.random.default_rng(5))
    assert (large > 0.5).sum() > (small > 0.5).sum()


def test_low_contrast_lowers_the_peak_intensity():
    high = vision_generator.render(vision_cell(contrast="high"), np.random.default_rng(5))
    low = vision_generator.render(vision_cell(contrast="low"), np.random.default_rng(5))
    assert low.max() < high.max()


def test_the_bar_occlusion_clears_the_middle_rows():
    image = vision_generator.render(vision_cell(occlusion="bar"), np.random.default_rng(11))
    middle = vision_generator.GRID // 2
    band = image[middle - 2 : middle + 2, :, 0]
    assert np.abs(band).max() < 0.4


def test_the_corner_occlusion_clears_the_top_left_block():
    image = vision_generator.render(vision_cell(occlusion="corner"), np.random.default_rng(11))
    assert np.abs(image[0:9, 0:9, 0]).max() < 0.4


def test_occlusion_removes_ink_rather_than_dimming_it():
    clear = vision_generator.render(vision_cell(), np.random.default_rng(2))
    occluded = vision_generator.render(vision_cell(occlusion="corner"), np.random.default_rng(2))
    assert (occluded > 0.5).sum() < (clear > 0.5).sum()


@pytest.mark.parametrize("texture", ("striped", "dotted"))
def test_a_patterned_texture_covers_less_than_a_solid_one(texture: str):
    solid = vision_generator.render(vision_cell(texture="solid"), np.random.default_rng(4))
    patterned = vision_generator.render(vision_cell(texture=texture), np.random.default_rng(4))
    assert (patterned > 0.5).sum() < (solid > 0.5).sum()


def test_rotation_changes_a_square_because_a_square_is_not_rotation_invariant():
    upright = vision_generator.render(
        vision_cell(shape="square", rotation="r000"), np.random.default_rng(6)
    )
    turned = vision_generator.render(
        vision_cell(shape="square", rotation="r030"), np.random.default_rng(6)
    )
    assert not np.array_equal(upright > 0.5, turned > 0.5)


def test_an_unknown_shape_is_refused_rather_than_rendered_blank():
    with pytest.raises(ValueError, match="unknown shape"):
        vision_generator._shape_mask("hexagon", np.zeros((2, 2)), np.zeros((2, 2)), 1.0)


def test_an_unknown_texture_is_refused():
    with pytest.raises(ValueError, match="unknown texture"):
        vision_generator._texture_mask("plaid", np.zeros((2, 2)), np.zeros((2, 2)))


def test_an_unknown_occlusion_is_refused():
    with pytest.raises(ValueError, match="unknown occlusion"):
        vision_generator._apply_occlusion(np.zeros((4, 4), dtype=np.float32), "shadow")


def test_render_batch_returns_one_row_per_item_in_order():
    cells = enumerate_cells(VISION_SPACE)[:5]
    items = [Item(cell=cell, label=0, index=position) for position, cell in enumerate(cells)]
    batch = vision_generator.render_batch(items, np.random.default_rng(0))
    assert batch.shape == (5, vision_generator.GRID, vision_generator.GRID, 1)


def test_render_batch_refuses_a_space_it_does_not_render():
    with pytest.raises(ValueError, match="vision space"):
        vision_generator.render_batch([], np.random.default_rng(0), SEQUENCE_SPACE)


def test_a_rendered_sequence_is_padded_to_the_declared_length():
    tokens = sequence_generator.render(sequence_cell(), np.random.default_rng(0))
    assert tokens.shape == (sequence_generator.MAX_LENGTH,)
    assert tokens.dtype == np.int32


def test_a_short_sequence_carries_fewer_real_tokens_than_a_long_one():
    short = sequence_generator.render(sequence_cell(length="short"), np.random.default_rng(0))
    long = sequence_generator.render(sequence_cell(length="long"), np.random.default_rng(0))
    assert int((short != sequence_generator.PAD).sum()) < int(
        (long != sequence_generator.PAD).sum()
    )


def test_every_token_stays_inside_the_vocabulary():
    for cell in enumerate_cells(SEQUENCE_SPACE):
        tokens = sequence_generator.render(cell, np.random.default_rng(2))
        assert tokens.min() >= 0
        assert tokens.max() < sequence_generator.VOCAB


def test_a_strong_amplitude_spans_more_levels_than_a_weak_one():
    weak = sequence_generator._trajectory("ascend", 24, sequence_generator.AMPLITUDE["weak"])
    strong = sequence_generator._trajectory("ascend", 24, sequence_generator.AMPLITUDE["strong"])
    assert np.ptp(strong) > np.ptp(weak)


def test_noisy_sequences_differ_from_clean_ones_at_the_same_generator_state():
    clean = sequence_generator.render(sequence_cell(noise="clean"), np.random.default_rng(9))
    noisy = sequence_generator.render(sequence_cell(noise="noisy"), np.random.default_rng(9))
    assert not np.array_equal(clean, noisy)


def test_shuffled_ordering_permutes_blocks_rather_than_changing_the_multiset():
    forward = sequence_generator.render(
        sequence_cell(ordering="forward", noise="clean"), np.random.default_rng(13)
    )
    shuffled = sequence_generator.render(
        sequence_cell(ordering="shuffled", noise="clean"), np.random.default_rng(13)
    )
    assert sorted(forward.tolist()) == sorted(shuffled.tolist())


def test_an_unknown_motif_is_refused():
    with pytest.raises(ValueError, match="unknown motif"):
        sequence_generator._trajectory("wobble", 10, 0.1)


def test_sequence_render_batch_refuses_a_space_it_does_not_render():
    with pytest.raises(ValueError, match="sequence space"):
        sequence_generator.render_batch([], np.random.default_rng(0), VISION_SPACE)
