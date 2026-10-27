"""Runs every seed and freeze depth for one arm and writes the committed measurement.

This is the only module in the repository that trains anything, and nothing that
computes a published figure imports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from finetune_eval_power import __version__
from finetune_eval_power.config import ArmConfig
from finetune_eval_power.runs import SweepRuns
from finetune_eval_power.slices.spec import enumerate_slices
from finetune_eval_power.tracking import open_tracker


def _build_manifest(arm: ArmConfig, cases) -> dict:
    """Everything the measurement needs to be interpretable, with no training done.

    Kept separate so it can be built, and serialised, before the sweep starts.
    """
    slices = enumerate_slices(arm.space, arm.pair_attributes)
    suite_cells = cases.suite.cells()
    return {
        "package_version": __version__,
        "config": arm.describe(),
        "suite_cells": suite_cells,
        "suite_slice_case_counts": {
            item.name: sum(1 for cell in suite_cells if item.matches(cell)) for item in slices
        },
        "probe_size": cases.probe.size,
        "slice_names": [item.name for item in slices],
        "held_out_slices": [
            item.name for item in slices if not arm.suite_profile.covers(item, arm.space)
        ],
    }


def run_sweep(
    arm: ArmConfig,
    seeds: tuple[int, ...],
    tracking_directory: str | Path | None = None,
    progress: bool = False,
) -> SweepRuns:
    """Train every seed at every freeze depth and collect the correctness vectors."""
    from finetune_eval_power.models.training import build_fixed_cases, train_seed

    cases = build_fixed_cases(arm)
    # Build and serialise the manifest before a single epoch runs. This ordering is
    # not stylistic: the first version of this function built the manifest after the
    # sweep, a refactor renamed an attribute the manifest reads, and the failure
    # surfaced twelve minutes of training later with nothing written. Anything that
    # can fail cheaply should fail before anything expensive starts.
    manifest = _build_manifest(arm, cases)
    json.dumps(manifest)
    tracker = open_tracker(tracking_directory) if tracking_directory else open_tracker()

    suite_before, suite_after = [], []
    probe_before, probe_after, curves = [], [], []
    for seed in seeds:
        if progress:
            print(f"  {arm.name} seed {seed}", flush=True)
        result = train_seed(arm, seed, cases)
        suite_before.append([result.suite_before] * len(arm.freeze_depths))
        probe_before.append([result.probe_before] * len(arm.freeze_depths))
        suite_after.append([depth.suite_after for depth in result.depths])
        probe_after.append([depth.probe_after for depth in result.depths])
        curves.append([depth.suite_curve for depth in result.depths])
        for depth in result.depths:
            with tracker.run(
                f"{arm.name}-seed{seed}-freeze{depth.freeze_depth}",
                {
                    "arm": arm.name,
                    "seed": seed,
                    "freeze_depth": depth.freeze_depth,
                    "suite_size": arm.suite_size,
                    "pretrain_epochs": arm.pretrain_epochs,
                    "finetune_epochs": arm.finetune_epochs,
                    "finetune_learning_rate": arm.finetune_learning_rate,
                },
            ) as run:
                run.log_metrics(
                    {
                        "suite_accuracy_before": float(np.mean(result.suite_before)),
                        "suite_accuracy_after": float(np.mean(depth.suite_after)),
                        "probe_accuracy_before": float(np.mean(result.probe_before)),
                        "probe_accuracy_after": float(np.mean(depth.probe_after)),
                    }
                )

    manifest["tracking_enabled"] = tracker.enabled
    return SweepRuns(
        arm=arm.name,
        seeds=tuple(seeds),
        freeze_depths=tuple(arm.freeze_depths),
        suite_before=np.array(suite_before, dtype=bool),
        suite_after=np.array(suite_after, dtype=bool),
        probe_before=np.array(probe_before, dtype=bool),
        probe_after=np.array(probe_after, dtype=bool),
        suite_curve=np.array(curves, dtype=np.float32),
        manifest=manifest,
    )
