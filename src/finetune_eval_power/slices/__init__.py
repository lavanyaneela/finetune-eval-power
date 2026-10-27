"""Capability slices: named regions of attribute space, and which of them an eval
suite can actually see.

The split between a covered slice and a held-out one is the whole subject of the
repository, so it lives in its own package rather than inside the generator.
"""

from finetune_eval_power.slices.spec import CapabilitySlice, CoverageProfile, enumerate_slices
from finetune_eval_power.slices.suite import (
    EvalSuite,
    ProbeSet,
    build_finetune_items,
    build_probe,
    build_suite,
)

__all__ = [
    "CapabilitySlice",
    "CoverageProfile",
    "EvalSuite",
    "ProbeSet",
    "build_finetune_items",
    "build_probe",
    "build_suite",
    "enumerate_slices",
]
