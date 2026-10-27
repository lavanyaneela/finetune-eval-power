"""The three case sets this repository keeps apart, and why they are different objects.

- The **eval suite** is what a team would actually ship: a sample drawn only from the
  region the team thought to cover, stratified over the label classes.
- The **probe** is the referee. It enumerates every cell of the attribute space a
  fixed number of times, so a per-slice accuracy is an exact proportion over a known
  denominator rather than a sample whose size wobbles slice to slice.
- The **fine-tune items** are the continued-training distribution, narrowed on
  purpose. The narrowing is what makes the model specialise; which capabilities it
  gives up in exchange is measured, not chosen.

Collapsing any two of these would destroy the measurement: a suite that is also the
referee cannot be blind to anything, and a probe that is sampled rather than
enumerated gives every slice a different resolution floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator.attributes import AttributeSpace, Item, enumerate_cells
from finetune_eval_power.slices.spec import CapabilitySlice, CoverageProfile


def _as_items(space: AttributeSpace, cells: list[dict[str, str]]) -> list[Item]:
    classes = space.label_classes()
    return [
        Item(
            cell=dict(cell),
            label=classes.index(cell[space.label_attribute]),
            index=position,
        )
        for position, cell in enumerate(cells)
    ]


def _stratified_sample(
    space: AttributeSpace,
    profile: CoverageProfile,
    size: int,
    rng: np.random.Generator,
    what: str,
) -> list[dict[str, str]]:
    """Draw `size` cells from the profile's region, balanced across label classes.

    Balanced because an unbalanced case set changes the accuracy a regression has to
    move, which would make every published detection floor a fact about the sampling
    rather than about the case count.
    """
    profile.validate_against(space)
    classes = space.label_classes()
    if size % len(classes) != 0:
        raise ToolError(f"{what} size {size} is not divisible by {len(classes)} label classes")
    # No emptiness guard on the region as a whole: the space is a full product and
    # validate_against has already rejected any value the space does not have, so a
    # non-empty profile always selects something. The per-class check below is the
    # one that can actually fire, when a profile excludes an entire label class.
    region = [cell for cell in enumerate_cells(space) if profile.contains(cell)]
    per_class = size // len(classes)
    chosen: list[dict[str, str]] = []
    for class_name in classes:
        pool = [cell for cell in region if cell[space.label_attribute] == class_name]
        if not pool:
            raise ToolError(f"the {what} profile selects no cells for class {class_name!r}")
        picks = rng.integers(0, len(pool), size=per_class)
        chosen.extend(dict(pool[int(pick)]) for pick in picks)
    return chosen


@dataclass(frozen=True)
class EvalSuite:
    """A fixed set of cases, and the region it was drawn from."""

    space: AttributeSpace
    profile: CoverageProfile
    items: tuple[Item, ...]

    @property
    def size(self) -> int:
        return len(self.items)

    def labels(self) -> np.ndarray:
        return np.array([item.label for item in self.items], dtype=np.int32)

    def cells(self) -> list[dict[str, str]]:
        return [dict(item.cell) for item in self.items]


@dataclass(frozen=True)
class ProbeSet:
    """Every cell of the space, repeated, so each slice has an exact denominator."""

    space: AttributeSpace
    items: tuple[Item, ...]
    repeats: int

    @property
    def size(self) -> int:
        return len(self.items)

    def labels(self) -> np.ndarray:
        return np.array([item.label for item in self.items], dtype=np.int32)

    def membership(self, capability_slice: CapabilitySlice) -> np.ndarray:
        """Boolean mask of the probe cases that fall inside this slice."""
        return np.array([capability_slice.matches(item.cell) for item in self.items], dtype=bool)

    def region_mask(self, profile: CoverageProfile) -> np.ndarray:
        """Boolean mask of the probe cases inside a coverage region."""
        return np.array([profile.contains(item.cell) for item in self.items], dtype=bool)


def build_suite(
    space: AttributeSpace,
    profile: CoverageProfile,
    size: int,
    rng: np.random.Generator,
) -> EvalSuite:
    """Sample `size` cases from the covered region, stratified over label classes."""
    chosen = _stratified_sample(space, profile, size, rng, "suite")
    return EvalSuite(space=space, profile=profile, items=tuple(_as_items(space, chosen)))


def build_probe(space: AttributeSpace, repeats: int) -> ProbeSet:
    """Enumerate the whole space `repeats` times, in a fixed order."""
    if repeats < 1:
        raise ToolError("the probe needs at least one repeat of each cell")
    cells = enumerate_cells(space)
    expanded = [dict(cell) for _ in range(repeats) for cell in cells]
    return ProbeSet(space=space, items=tuple(_as_items(space, expanded)), repeats=repeats)


def build_finetune_items(
    space: AttributeSpace,
    profile: CoverageProfile,
    size: int,
    rng: np.random.Generator,
) -> list[Item]:
    """Continued-training cases, drawn from a region narrower than the suite's.

    That narrowing is the engineering mistake this repository reproduces: a model
    specialised on whatever data arrived most recently, evaluated by a suite that was
    built from roughly the same place.
    """
    return _as_items(space, _stratified_sample(space, profile, size, rng, "fine-tune"))
