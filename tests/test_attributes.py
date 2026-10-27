"""Properties of the attribute space, which every other module depends on."""

from __future__ import annotations

import pytest

from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator.attributes import (
    SEQUENCE_SPACE,
    VISION_SPACE,
    AttributeSpace,
    enumerate_cells,
)


def test_the_vision_space_enumerates_exactly_its_cell_count():
    assert len(enumerate_cells(VISION_SPACE)) == VISION_SPACE.cell_count


def test_the_sequence_space_enumerates_exactly_its_cell_count():
    assert len(enumerate_cells(SEQUENCE_SPACE)) == SEQUENCE_SPACE.cell_count


def test_enumeration_order_does_not_depend_on_attribute_insertion_order():
    reordered = AttributeSpace(
        name="reordered",
        attributes=dict(reversed(list(VISION_SPACE.attributes.items()))),
        label_attribute=VISION_SPACE.label_attribute,
    )
    assert enumerate_cells(reordered) == enumerate_cells(VISION_SPACE)


def test_every_enumerated_cell_is_fully_specified():
    for cell in enumerate_cells(SEQUENCE_SPACE):
        SEQUENCE_SPACE.validate_cell(cell)


def test_enumerated_cells_are_all_distinct():
    cells = enumerate_cells(VISION_SPACE)
    assert len({tuple(sorted(cell.items())) for cell in cells}) == len(cells)


def test_a_space_whose_label_attribute_is_absent_is_refused():
    with pytest.raises(ToolError, match="not an attribute"):
        AttributeSpace(name="broken", attributes={"a": ("x",)}, label_attribute="b")


def test_a_space_with_a_duplicated_attribute_value_is_refused():
    with pytest.raises(ToolError, match="duplicate"):
        AttributeSpace(name="broken", attributes={"a": ("x", "x")}, label_attribute="a")


def test_a_space_with_an_empty_attribute_is_refused():
    with pytest.raises(ToolError, match="no values"):
        AttributeSpace(name="broken", attributes={"a": ()}, label_attribute="a")


def test_validating_a_cell_with_a_missing_attribute_names_what_is_missing():
    with pytest.raises(ToolError, match="missing="):
        VISION_SPACE.validate_cell({"shape": "disc"})


def test_validating_a_cell_with_an_unknown_value_names_the_attribute():
    cell = dict(enumerate_cells(VISION_SPACE)[0])
    cell["shape"] = "hexagon"
    with pytest.raises(ToolError, match="shape"):
        VISION_SPACE.validate_cell(cell)


def test_validating_a_cell_with_an_extra_attribute_is_refused():
    cell = dict(enumerate_cells(VISION_SPACE)[0])
    cell["unexpected"] = "value"
    with pytest.raises(ToolError, match="extra="):
        VISION_SPACE.validate_cell(cell)


def test_the_label_classes_are_the_values_of_the_label_attribute():
    assert VISION_SPACE.label_classes() == VISION_SPACE.attributes["shape"]
    assert SEQUENCE_SPACE.label_classes() == SEQUENCE_SPACE.attributes["motif"]


def test_both_spaces_label_four_classes_so_the_two_arms_are_comparable():
    assert len(VISION_SPACE.label_classes()) == len(SEQUENCE_SPACE.label_classes()) == 4
