"""Properties of the three case sets, and the ways they must not be confused."""

from __future__ import annotations

import numpy as np
import pytest

from finetune_eval_power.config import VISION_ARM
from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator.attributes import VISION_SPACE, enumerate_cells
from finetune_eval_power.slices.spec import CapabilitySlice, CoverageProfile
from finetune_eval_power.slices.suite import (
    build_finetune_items,
    build_probe,
    build_suite,
)

PROFILE = CoverageProfile(allowed={"scale": ("medium", "large"), "contrast": ("high",)})


def test_the_suite_has_exactly_the_requested_number_of_cases():
    suite = build_suite(VISION_SPACE, PROFILE, 480, np.random.default_rng(0))
    assert suite.size == 480


def test_the_suite_is_balanced_across_label_classes():
    suite = build_suite(VISION_SPACE, PROFILE, 480, np.random.default_rng(0))
    counts = np.bincount(suite.labels(), minlength=len(VISION_SPACE.label_classes()))
    assert set(counts.tolist()) == {480 // len(VISION_SPACE.label_classes())}


def test_every_suite_case_lies_inside_the_coverage_profile():
    suite = build_suite(VISION_SPACE, PROFILE, 240, np.random.default_rng(1))
    assert all(PROFILE.contains(item.cell) for item in suite.items)


def test_a_suite_size_not_divisible_by_the_class_count_is_refused():
    with pytest.raises(ToolError, match="divisible"):
        build_suite(VISION_SPACE, PROFILE, 99, np.random.default_rng(0))


def test_a_profile_selecting_no_cells_is_refused_rather_than_returning_empty():
    profile = CoverageProfile(allowed={"scale": ("small",), "shape": ("disc",)})
    with pytest.raises(ToolError, match="no cells for class"):
        build_suite(VISION_SPACE, profile, 4, np.random.default_rng(0))


def test_the_same_generator_seed_produces_the_same_suite():
    first = build_suite(VISION_SPACE, PROFILE, 120, np.random.default_rng(5))
    second = build_suite(VISION_SPACE, PROFILE, 120, np.random.default_rng(5))
    assert first.cells() == second.cells()


def test_a_different_generator_seed_produces_a_different_suite():
    first = build_suite(VISION_SPACE, PROFILE, 120, np.random.default_rng(5))
    second = build_suite(VISION_SPACE, PROFILE, 120, np.random.default_rng(6))
    assert first.cells() != second.cells()


def test_the_probe_enumerates_every_cell_the_requested_number_of_times():
    probe = build_probe(VISION_SPACE, 3)
    assert probe.size == VISION_SPACE.cell_count * 3


def test_the_probe_is_an_exact_enumeration_so_every_cell_appears_equally_often():
    probe = build_probe(VISION_SPACE, 2)
    seen: dict[tuple, int] = {}
    for item in probe.items:
        key = tuple(sorted(item.cell.items()))
        seen[key] = seen.get(key, 0) + 1
    assert set(seen.values()) == {2}


def test_a_probe_with_no_repeats_is_refused():
    with pytest.raises(ToolError, match="at least one repeat"):
        build_probe(VISION_SPACE, 0)


def test_slice_membership_counts_are_exact_multiples_of_the_repeat_count():
    probe = build_probe(VISION_SPACE, 2)
    item = CapabilitySlice.of(scale="small")
    expected = len(item.cells(VISION_SPACE)) * 2
    assert int(probe.membership(item).sum()) == expected


def test_a_region_mask_selects_exactly_the_cells_inside_the_profile():
    probe = build_probe(VISION_SPACE, 1)
    mask = probe.region_mask(PROFILE)
    expected = sum(1 for cell in enumerate_cells(VISION_SPACE) if PROFILE.contains(cell))
    assert int(mask.sum()) == expected


def test_the_probe_contains_cases_the_suite_region_excludes():
    """Without this the referee could not see anything the suite cannot."""
    probe = build_probe(VISION_SPACE, 1)
    assert int((~probe.region_mask(VISION_ARM.suite_profile)).sum()) > 0


def test_finetune_items_are_balanced_across_label_classes():
    items = build_finetune_items(
        VISION_SPACE, VISION_ARM.finetune_profile, 400, np.random.default_rng(0)
    )
    counts = np.bincount(
        [item.label for item in items], minlength=len(VISION_SPACE.label_classes())
    )
    assert set(counts.tolist()) == {100}


def test_finetune_items_lie_inside_the_finetune_region():
    items = build_finetune_items(
        VISION_SPACE, VISION_ARM.finetune_profile, 40, np.random.default_rng(0)
    )
    assert all(VISION_ARM.finetune_profile.contains(item.cell) for item in items)


def test_every_item_carries_the_label_its_cell_implies():
    suite = build_suite(VISION_SPACE, PROFILE, 80, np.random.default_rng(2))
    classes = VISION_SPACE.label_classes()
    assert all(
        classes[item.label] == item.cell[VISION_SPACE.label_attribute] for item in suite.items
    )


def test_suite_items_are_indexed_in_order_so_pairing_is_positional():
    suite = build_suite(VISION_SPACE, PROFILE, 40, np.random.default_rng(2))
    assert [item.index for item in suite.items] == list(range(40))
