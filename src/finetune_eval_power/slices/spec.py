"""A capability slice and the coverage profile that decides whether a suite can see it.

A slice is a constraint on one or two attributes, never more. The limit is
deliberate: a three-attribute slice in this space can contain so few cells that its
measured accuracy is noise, and publishing a ranking of noise is exactly the failure
this repository is about.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass

from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator.attributes import AttributeSpace, enumerate_cells

MAX_SLICE_DIMENSIONS = 2


@dataclass(frozen=True)
class CapabilitySlice:
    """A named region of attribute space, defined by pinning one or two attributes."""

    name: str
    constraints: tuple[tuple[str, str], ...]

    @classmethod
    def of(cls, **constraints: str) -> CapabilitySlice:
        if not constraints:
            raise ToolError("a slice must constrain at least one attribute")
        if len(constraints) > MAX_SLICE_DIMENSIONS:
            raise ToolError(
                f"a slice constrains at most {MAX_SLICE_DIMENSIONS} attributes, "
                f"got {len(constraints)}"
            )
        ordered = tuple(sorted(constraints.items()))
        name = " & ".join(f"{key}={value}" for key, value in ordered)
        return cls(name=name, constraints=ordered)

    @property
    def dimensions(self) -> int:
        return len(self.constraints)

    def matches(self, cell: Mapping[str, str]) -> bool:
        return all(cell.get(key) == value for key, value in self.constraints)

    def cells(self, space: AttributeSpace) -> list[dict[str, str]]:
        return [cell for cell in enumerate_cells(space) if self.matches(cell)]


@dataclass(frozen=True)
class CoverageProfile:
    """The region an eval suite draws its cases from.

    An attribute absent from `allowed` is unrestricted. Expressing coverage as a
    region rather than as a list of cells is what lets the suite be resampled at a
    different size without changing what it claims to cover.
    """

    allowed: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        for attribute, values in self.allowed.items():
            if not values:
                raise ToolError(f"coverage profile allows no values for {attribute!r}")

    def contains(self, cell: Mapping[str, str]) -> bool:
        return all(cell[attribute] in values for attribute, values in self.allowed.items())

    def covers(self, capability_slice: CapabilitySlice, space: AttributeSpace) -> bool:
        """True when at least one cell of the slice is inside the suite's region.

        Partial coverage counts as covered on purpose. A slice the suite samples one
        corner of is a slice the suite has some power over, and calling it held out
        would overstate the blindness this repository measures.
        """
        return any(self.contains(cell) for cell in capability_slice.cells(space))

    def validate_against(self, space: AttributeSpace) -> None:
        for attribute, values in self.allowed.items():
            if attribute not in space.attributes:
                raise ToolError(f"{attribute!r} is not an attribute of space {space.name!r}")
            unknown = sorted(set(values) - set(space.attributes[attribute]))
            if unknown:
                raise ToolError(f"{attribute!r} has no values {unknown} in space {space.name!r}")


def enumerate_slices(
    space: AttributeSpace, pair_attributes: tuple[str, ...]
) -> list[CapabilitySlice]:
    """Every single-attribute slice, plus every pair drawn from `pair_attributes`.

    Pairs are restricted to the attributes that describe difficulty rather than
    identity, because a `shape=disc & texture=solid` slice is a label combination and
    not a capability.
    """
    for attribute in pair_attributes:
        if attribute not in space.attributes:
            raise ToolError(f"{attribute!r} is not an attribute of space {space.name!r}")
    slices = [
        CapabilitySlice.of(**{attribute: value})
        for attribute in sorted(space.attributes)
        for value in space.attributes[attribute]
    ]
    for left, right in itertools.combinations(sorted(pair_attributes), 2):
        for left_value in space.attributes[left]:
            for right_value in space.attributes[right]:
                slices.append(CapabilitySlice.of(**{left: left_value, right: right_value}))
    return slices
