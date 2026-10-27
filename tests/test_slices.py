"""Properties of capability slices and the coverage profile that hides them."""

from __future__ import annotations

import pytest

from finetune_eval_power.config import SEQUENCE_ARM, VISION_ARM
from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator.attributes import VISION_SPACE, enumerate_cells
from finetune_eval_power.slices.spec import (
    MAX_SLICE_DIMENSIONS,
    CapabilitySlice,
    CoverageProfile,
    enumerate_slices,
)


def test_a_slice_name_is_built_from_its_constraints_in_a_stable_order():
    first = CapabilitySlice.of(scale="small", contrast="low")
    second = CapabilitySlice.of(contrast="low", scale="small")
    assert first.name == second.name == "contrast=low & scale=small"


def test_a_slice_matches_exactly_the_cells_that_satisfy_every_constraint():
    item = CapabilitySlice.of(scale="small", contrast="low")
    cells = item.cells(VISION_SPACE)
    assert cells
    assert all(cell["scale"] == "small" and cell["contrast"] == "low" for cell in cells)


def test_a_slice_with_no_constraints_is_refused():
    with pytest.raises(ToolError, match="at least one"):
        CapabilitySlice.of()


def test_a_slice_over_three_attributes_is_refused_because_it_would_be_too_thin():
    with pytest.raises(ToolError, match=f"at most {MAX_SLICE_DIMENSIONS}"):
        CapabilitySlice.of(scale="small", contrast="low", occlusion="bar")


def test_slice_dimensions_counts_the_pinned_attributes():
    assert CapabilitySlice.of(scale="small").dimensions == 1
    assert CapabilitySlice.of(scale="small", contrast="low").dimensions == 2


def test_a_profile_contains_a_cell_only_when_every_restricted_attribute_agrees():
    profile = CoverageProfile(allowed={"scale": ("large",), "contrast": ("high",)})
    inside = next(
        cell
        for cell in enumerate_cells(VISION_SPACE)
        if cell["scale"] == "large" and cell["contrast"] == "high"
    )
    outside = dict(inside, scale="small")
    assert profile.contains(inside)
    assert not profile.contains(outside)


def test_an_attribute_absent_from_the_profile_is_unrestricted():
    profile = CoverageProfile(allowed={"scale": ("large",)})
    cells = [cell for cell in enumerate_cells(VISION_SPACE) if profile.contains(cell)]
    assert {cell["occlusion"] for cell in cells} == set(VISION_SPACE.attributes["occlusion"])


def test_an_empty_profile_contains_the_whole_space():
    profile = CoverageProfile(allowed={})
    assert all(profile.contains(cell) for cell in enumerate_cells(VISION_SPACE))


def test_a_profile_that_allows_no_values_for_an_attribute_is_refused():
    with pytest.raises(ToolError, match="allows no values"):
        CoverageProfile(allowed={"scale": ()})


def test_a_profile_naming_an_unknown_attribute_is_refused():
    with pytest.raises(ToolError, match="not an attribute"):
        CoverageProfile(allowed={"colour": ("red",)}).validate_against(VISION_SPACE)


def test_a_profile_naming_an_unknown_value_is_refused():
    with pytest.raises(ToolError, match="no values"):
        CoverageProfile(allowed={"scale": ("enormous",)}).validate_against(VISION_SPACE)


def test_partial_overlap_counts_as_covered_rather_than_held_out():
    profile = CoverageProfile(allowed={"scale": ("medium", "large")})
    item = CapabilitySlice.of(occlusion="bar")
    assert profile.covers(item, VISION_SPACE)


def test_a_slice_outside_the_profile_on_any_attribute_is_held_out():
    profile = CoverageProfile(allowed={"scale": ("medium", "large")})
    assert not profile.covers(CapabilitySlice.of(scale="small"), VISION_SPACE)


def test_every_single_attribute_value_gets_its_own_slice():
    slices = enumerate_slices(VISION_SPACE, ())
    expected = sum(len(values) for values in VISION_SPACE.attributes.values())
    assert len(slices) == expected


def test_pair_slices_are_added_only_for_the_named_difficulty_attributes():
    singles = enumerate_slices(VISION_SPACE, ())
    with_pairs = enumerate_slices(VISION_SPACE, ("scale", "contrast"))
    added = len(with_pairs) - len(singles)
    assert added == len(VISION_SPACE.attributes["scale"]) * len(VISION_SPACE.attributes["contrast"])


def test_enumerated_slice_names_are_unique():
    slices = enumerate_slices(VISION_SPACE, VISION_ARM.pair_attributes)
    assert len({item.name for item in slices}) == len(slices)


def test_enumerating_slices_over_an_unknown_pair_attribute_is_refused():
    with pytest.raises(ToolError, match="not an attribute"):
        enumerate_slices(VISION_SPACE, ("colour",))


def test_the_vision_arm_holds_out_some_slices_and_covers_others():
    slices = enumerate_slices(VISION_SPACE, VISION_ARM.pair_attributes)
    covered = [s for s in slices if VISION_ARM.suite_profile.covers(s, VISION_SPACE)]
    held = [s for s in slices if not VISION_ARM.suite_profile.covers(s, VISION_SPACE)]
    assert covered and held


def test_the_sequence_arm_holds_out_some_slices_and_covers_others():
    space = SEQUENCE_ARM.space
    slices = enumerate_slices(space, SEQUENCE_ARM.pair_attributes)
    covered = [s for s in slices if SEQUENCE_ARM.suite_profile.covers(s, space)]
    held = [s for s in slices if not SEQUENCE_ARM.suite_profile.covers(s, space)]
    assert covered and held


def test_the_finetune_region_sits_inside_the_suite_region_for_every_arm():
    """The whole setup rests on this: the suite is broader than the fine-tune data.

    If the fine-tune region ever escaped the suite region, the suite would be
    sampling cases the model never specialised on and the blindness being measured
    would be a different phenomenon from the one the documents describe.
    """
    for arm in (VISION_ARM, SEQUENCE_ARM):
        inside = [
            cell for cell in enumerate_cells(arm.space) if arm.finetune_profile.contains(cell)
        ]
        assert inside
        assert all(arm.suite_profile.contains(cell) for cell in inside), arm.name
