"""Procedural data generation, which is the only reason this repository can exist.

A capability slice is a region of attribute space. Measuring what a suite misses
requires holding out a region deliberately and then producing arbitrarily many items
from it, and no fixed public dataset exposes the attributes needed to do that. See
ADR-001.
"""

from finetune_eval_power.generator.attributes import (
    SEQUENCE_SPACE,
    VISION_SPACE,
    AttributeSpace,
    Item,
    enumerate_cells,
)

__all__ = ["SEQUENCE_SPACE", "VISION_SPACE", "AttributeSpace", "Item", "enumerate_cells"]
