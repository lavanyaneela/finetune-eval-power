"""The attribute space, defined once and shared by the generator, the suite and the
slice machinery.

Everything downstream is a region of this space: an eval suite is a set of cells it
samples from, a capability slice is a constraint on one or two attributes, and a
held-out slice is a region the suite never samples. Defining the space in one place
is what keeps "the suite covers this" and "the generator can produce this" talking
about the same coordinates.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field

from finetune_eval_power.errors import ToolError


@dataclass(frozen=True)
class AttributeSpace:
    """A named set of categorical attributes and their allowed values.

    Categorical rather than continuous on purpose: a slice has to be nameable in a
    README table, and "rotation between 41 and 86 degrees" is not a thing a reviewer
    can hold in their head while reading the rest of the document.
    """

    name: str
    attributes: Mapping[str, tuple[str, ...]]
    label_attribute: str

    def __post_init__(self) -> None:
        if self.label_attribute not in self.attributes:
            raise ToolError(f"{self.label_attribute!r} is not an attribute of space {self.name!r}")
        for attribute, values in self.attributes.items():
            if len(values) != len(set(values)):
                raise ToolError(f"attribute {attribute!r} has duplicate values")
            if not values:
                raise ToolError(f"attribute {attribute!r} has no values")

    @property
    def cell_count(self) -> int:
        """How many fully specified attribute combinations exist."""
        total = 1
        for values in self.attributes.values():
            total *= len(values)
        return total

    def label_classes(self) -> tuple[str, ...]:
        return self.attributes[self.label_attribute]

    def validate_cell(self, cell: Mapping[str, str]) -> None:
        if set(cell) != set(self.attributes):
            missing = sorted(set(self.attributes) - set(cell))
            extra = sorted(set(cell) - set(self.attributes))
            raise ToolError(f"cell is not fully specified (missing={missing}, extra={extra})")
        for attribute, value in cell.items():
            if value not in self.attributes[attribute]:
                raise ToolError(f"{value!r} is not a value of attribute {attribute!r}")


@dataclass(frozen=True)
class Item:
    """One generated example: its coordinates, and the label those coordinates imply.

    The item carries its whole cell rather than only its label, because attributing a
    regression to a region is the entire point and a bare (x, y) pair throws the
    coordinates away.
    """

    cell: Mapping[str, str]
    label: int
    index: int = field(default=-1)


VISION_SPACE = AttributeSpace(
    name="vision",
    attributes={
        "shape": ("disc", "square", "triangle", "ring"),
        "texture": ("solid", "striped", "dotted"),
        "scale": ("small", "medium", "large"),
        "rotation": ("r000", "r030", "r060", "r090"),
        "contrast": ("low", "high"),
        "occlusion": ("none", "corner", "bar"),
    },
    label_attribute="shape",
)

SEQUENCE_SPACE = AttributeSpace(
    name="sequence",
    attributes={
        "motif": ("ascend", "descend", "plateau", "spike"),
        "length": ("short", "medium", "long"),
        "amplitude": ("weak", "strong"),
        "noise": ("clean", "noisy"),
        "ordering": ("forward", "shuffled"),
    },
    label_attribute="motif",
)


def enumerate_cells(space: AttributeSpace) -> list[dict[str, str]]:
    """Every fully specified cell, in a fixed order.

    Sorted by attribute name so the enumeration does not depend on dict insertion
    order, which would make a committed fixture depend on how this file was edited.
    """
    names = sorted(space.attributes)
    value_lists = [space.attributes[name] for name in names]
    return [dict(zip(names, combo, strict=True)) for combo in itertools.product(*value_lists)]
