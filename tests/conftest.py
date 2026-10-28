"""Shared fixtures.

The important one is `synthetic_runs`: a hand-built measurement with a known answer.
Almost every test in this suite runs against it rather than against trained models,
which is what keeps the suite fast and what lets a test assert an exact figure. The
training code has its own small, slow file.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import numpy as np
import pytest

from finetune_eval_power.config import VISION_ARM, ArmConfig
from finetune_eval_power.runs import SweepRuns
from finetune_eval_power.slices.spec import CoverageProfile, enumerate_slices
from finetune_eval_power.slices.suite import build_probe, build_suite

#: Test modules that cannot be imported without the training extra.
TRAINING_ONLY_MODULES = frozenset({"test_training.py", "test_sweep_and_tracking.py"})


def pytest_ignore_collect(collection_path, config):
    """Skip the training-only modules at collection rather than at import.

    A module-level `importorskip` is reported as a collected, skipped entry, so the
    junit test count came out two higher on a machine without the training extra
    than on one with it. The published count then depended on what happened to be
    installed, and `make verify` failed its own receipts on a fresh clone. Ignoring
    the module during collection keeps the count identical either way.
    """
    if collection_path.name not in TRAINING_ONLY_MODULES:
        return False
    return importlib.util.find_spec("keras") is None


@pytest.fixture
def vision_arm() -> ArmConfig:
    return VISION_ARM


@pytest.fixture
def small_arm() -> ArmConfig:
    """A vision arm scaled down so a test that trains finishes in seconds."""
    return dataclasses.replace(
        VISION_ARM,
        suite_size=48,
        probe_repeats=1,
        pretrain_items=256,
        finetune_items=128,
        pretrain_epochs=1,
        finetune_epochs=1,
        freeze_depths=(0, 4),
    )


def build_synthetic_runs(
    arm: ArmConfig,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    suite_drop_cases: int = 4,
    held_out_drop_fraction: float = 0.25,
    rng_seed: int = 7,
    suite_drop_jitter: tuple[int, ...] = (),
    suite_gain_cases: int = 0,
) -> SweepRuns:
    """A measurement with the answer planted in it.

    The suite loses `suite_drop_cases` cases per retraining and nothing else moves
    there, while probe cases inside held-out slices lose a fixed fraction. A test can
    therefore assert both what the suite reports and what the slices actually did.

    `suite_drop_jitter` gives a per-retraining case count instead, which is how a
    test builds a measurement whose retrainings disagree. With it absent every
    retraining agrees exactly and the seed spread is zero, which is useful for
    isolating the power path from the refusal path.

    `suite_gain_cases` moves cases the other way, wrong before and right after.
    Without it the only way to give the suite enough changed verdicts to have a
    detection floor is to make it lose cases, which also makes it raise a flag, so
    every test ran against a suite that either had no floor or was not quiet. The
    real measurement is quiet AND has a floor, because its cases move in both
    directions and nearly cancel. That is the shape the repository is about, and no
    test could build it.
    """
    rng = np.random.default_rng(rng_seed)
    probe = build_probe(arm.space, arm.probe_repeats)
    suite = build_suite(arm.space, arm.suite_profile, arm.suite_size, np.random.default_rng(1))
    slices = enumerate_slices(arm.space, arm.pair_attributes)
    held_out = [item for item in slices if not arm.suite_profile.covers(item, arm.space)]
    held_mask = np.zeros(probe.size, dtype=bool)
    for item in held_out:
        held_mask |= probe.membership(item)

    depths = arm.freeze_depths
    shape_suite = (len(seeds), len(depths), arm.suite_size)
    shape_probe = (len(seeds), len(depths), probe.size)
    suite_before = np.ones(shape_suite, dtype=bool)
    suite_after = np.ones(shape_suite, dtype=bool)
    probe_before = np.ones(shape_probe, dtype=bool)
    probe_after = np.ones(shape_probe, dtype=bool)

    for seed_position in range(len(seeds)):
        per_seed_drop = (
            suite_drop_jitter[seed_position % len(suite_drop_jitter)]
            if suite_drop_jitter
            else suite_drop_cases
        )
        for depth_position in range(len(depths)):
            moved = rng.choice(arm.suite_size, size=per_seed_drop + suite_gain_cases, replace=False)
            lost = moved[:per_seed_drop]
            gained = moved[per_seed_drop:]
            suite_after[seed_position, depth_position, lost] = False
            # Wrong before and right after. The before vector has to say so too, or
            # the case was right in both and nothing changed.
            suite_before[seed_position, depth_position, gained] = False
            held_indices = np.flatnonzero(held_mask)
            count = int(len(held_indices) * held_out_drop_fraction)
            chosen = rng.choice(held_indices, size=count, replace=False)
            probe_after[seed_position, depth_position, chosen] = False

    curve = np.ones((len(seeds), len(depths), arm.finetune_epochs + 1), dtype=np.float32)
    return SweepRuns(
        arm=arm.name,
        seeds=seeds,
        freeze_depths=depths,
        suite_before=suite_before,
        suite_after=suite_after,
        probe_before=probe_before,
        probe_after=probe_after,
        suite_curve=curve,
        manifest={
            "config": arm.describe(),
            "suite_cells": suite.cells(),
            "suite_slice_case_counts": {
                item.name: sum(1 for cell in suite.cells() if item.matches(cell)) for item in slices
            },
            "held_out_slices": [item.name for item in held_out],
            "slice_names": [item.name for item in slices],
            "probe_size": probe.size,
        },
    )


@pytest.fixture
def synthetic_runs(vision_arm: ArmConfig) -> SweepRuns:
    return build_synthetic_runs(vision_arm)


@pytest.fixture
def quiet_profile() -> CoverageProfile:
    return CoverageProfile(allowed={"scale": ("large",)})
