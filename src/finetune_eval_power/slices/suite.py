"""The three case sets this repository keeps apart, and why they are different objects.

- The **eval suite** is what a team would actually ship: a sample drawn only from the
  region the team thought to cover, stratified over the task A classes.
- The **probe** is the referee. It enumerates every cell of the attribute space a
  fixed number of times, so a per-slice accuracy is an exact proportion over a known
  denominator rather than a sample whose size wobbles slice to slice.
- The **fine-tune cells** are the task B training distribution, narrowed on purpose.
  The narrowing is what makes the backbone drift; which slices it damages is measured,
  not chosen.

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
    a_values = space.task_a_classes()
    b_values = space.task_b_classes()
    return [
        Item(
            cell=dict(cell),
            task_a_label=a_values.index(cell[space.task_a_attribute]),
            task_b_label=b_values.index(cell[space.task_b_attribute]),
            index=position,
        )
        for position, cell in enumerate(cells)
    ]


@dataclass(frozen=True)
class EvalSuite:
    """A fixed set of task A cases, and the region it was drawn from."""

    space: AttributeSpace
    profile: CoverageProfile
    items: tuple[Item, ...]

    @property
    def size(self) -> int:
        return len(self.items)

    def task_a_labels(self) -> np.ndarray:
        return np.array([item.task_a_label for item in self.items], dtype=np.int32)


@dataclass(frozen=True)
class ProbeSet:
    """Every cell of the space, repeated, so each slice has an exact denominator."""

    space: AttributeSpace
    items: tuple[Item, ...]
    repeats: int

    @property
    def size(self) -> int:
        return len(self.items)

    def task_a_labels(self) -> np.ndarray:
        return np.array([item.task_a_label for item in self.items], dtype=np.int32)

    def membership(self, capability_slice: CapabilitySlice) -> np.ndarray:
        """Boolean mask of the probe cases that fall inside this slice."""
        return np.array([capability_slice.matches(item.cell) for item in self.items], dtype=bool)


def build_suite(
    space: AttributeSpace,
    profile: CoverageProfile,
    size: int,
    rng: np.random.Generator,
) -> EvalSuite:
    """Sample `size` cases from the covered region, stratified over task A classes.

    Stratified because an unbalanced suite changes the accuracy a regression has to
    move, which would make the published minimum detectable effect a fact about the
    sampling rather than about the suite size.
    """
    profile.validate_against(space)
    classes = space.task_a_classes()
    if size % len(classes) != 0:
        raise ToolError(f"suite size {size} is not divisible by {len(classes)} task A classes")
    covered = [cell for cell in enumerate_cells(space) if profile.contains(cell)]
    if not covered:
        raise ToolError("the coverage profile selects no cells")
    per_class = size // len(classes)
    chosen: list[dict[str, str]] = []
    for class_name in classes:
        pool = [cell for cell in covered if cell[space.task_a_attribute] == class_name]
        if not pool:
            raise ToolError(f"the coverage profile selects no cells for class {class_name!r}")
        picks = rng.integers(0, len(pool), size=per_class)
        chosen.extend(dict(pool[int(pick)]) for pick in picks)
    return EvalSuite(space=space, profile=profile, items=tuple(_as_items(space, chosen)))


def build_probe(space: AttributeSpace, repeats: int) -> ProbeSet:
    """Enumerate the whole space `repeats` times, in a fixed order."""
    if repeats < 1:
        raise ToolError("the probe needs at least one repeat of each cell")
    cells = enumerate_cells(space)
    expanded = [dict(cell) for _ in range(repeats) for cell in cells]
    return ProbeSet(space=space, items=tuple(_as_items(space, expanded)), repeats=repeats)


def build_finetune_cells(
    space: AttributeSpace,
    profile: CoverageProfile,
    size: int,
    rng: np.random.Generator,
) -> list[Item]:
    """Task B training cases, stratified over the task B classes.

    The profile passed here is narrower than the suite's. That narrowing is the
    engineering mistake this repository reproduces: a fine-tune set collected from
    whatever was convenient, evaluated by a suite collected from somewhere else.
    """
    profile.validate_against(space)
    classes = space.task_b_classes()
    if size % len(classes) != 0:
        raise ToolError(f"fine-tune size {size} is not divisible by {len(classes)} task B classes")
    region = [cell for cell in enumerate_cells(space) if profile.contains(cell)]
    if not region:
        raise ToolError("the fine-tune profile selects no cells")
    per_class = size // len(classes)
    chosen: list[dict[str, str]] = []
    for class_name in classes:
        pool = [cell for cell in region if cell[space.task_b_attribute] == class_name]
        if not pool:
            raise ToolError(f"the fine-tune profile selects no cells for class {class_name!r}")
        picks = rng.integers(0, len(pool), size=per_class)
        chosen.extend(dict(pool[int(pick)]) for pick in picks)
    return _as_items(space, chosen)
