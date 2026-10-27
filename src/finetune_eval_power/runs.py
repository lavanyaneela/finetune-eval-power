"""Storage for the committed measurement, and the reason it is committed at all.

Training is pinned rather than repeated on every verify. Two reasons, and the second
is the honest one:

1. Speed. `make verify` is the single command the README promises a reviewer, and a
   command that trains eighty models is a command nobody runs twice.
2. Floating point arithmetic on a CPU is not bitwise portable. The same seed, the
   same library versions and a different microarchitecture produce slightly
   different weights. A repository that published a figure to three decimal places
   and re-derived it on an unknown runner would have a red badge and no way to fix
   it. So the measurement is captured once, hashed, and committed, every published
   figure is computed from it by code that re-runs on every verify, and a separate
   CI job retrains from scratch and publishes the drift instead of pretending there
   is none. ADR-004 records this.

Arrays are stored as individual `.npy` files rather than one `.npz`, because the zip
container in `.npz` embeds a modification time and the fixture would stop being byte
reproducible a second after it was written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from finetune_eval_power.errors import ToolError

MANIFEST_NAME = "manifest.json"
ARRAY_NAMES = (
    "suite_before",
    "suite_after",
    "probe_before",
    "probe_after",
    "suite_curve",
)
BOOLEAN_ARRAYS = ("suite_before", "suite_after", "probe_before", "probe_after")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SweepRuns:
    """One arm's measurement: correctness before and after, per seed and freeze depth.

    `suite_*` and `probe_*` are boolean correctness on task A, indexed
    [seed, freeze_depth, case]. The pairing is positional and load-bearing: case i of
    `suite_before` and case i of `suite_after` are the same image scored by the same
    head, which is what makes a McNemar table meaningful.
    """

    arm: str
    seeds: tuple[int, ...]
    freeze_depths: tuple[int, ...]
    suite_before: np.ndarray
    suite_after: np.ndarray
    probe_before: np.ndarray
    probe_after: np.ndarray
    suite_curve: np.ndarray
    manifest: dict[str, Any]

    def __post_init__(self) -> None:
        shape = (len(self.seeds), len(self.freeze_depths))
        for name in BOOLEAN_ARRAYS:
            array = getattr(self, name)
            if array.dtype != np.bool_:
                raise ToolError(f"{name} must be boolean correctness, got {array.dtype}")
            if array.shape[:2] != shape:
                raise ToolError(f"{name} has shape {array.shape}, expected {shape} plus cases")
        if self.suite_curve.shape[:2] != shape:
            raise ToolError(f"suite_curve has shape {self.suite_curve.shape}")
        if self.suite_before.shape != self.suite_after.shape:
            raise ToolError("suite before and after must align case for case")
        if self.probe_before.shape != self.probe_after.shape:
            raise ToolError("probe before and after must align case for case")

    @property
    def suite_size(self) -> int:
        return int(self.suite_before.shape[2])

    @property
    def probe_size(self) -> int:
        return int(self.probe_before.shape[2])

    def depth_index(self, freeze_depth: int) -> int:
        if freeze_depth not in self.freeze_depths:
            raise ToolError(f"freeze depth {freeze_depth} is not in this sweep")
        return self.freeze_depths.index(freeze_depth)

    def seed_index(self, seed: int) -> int:
        if seed not in self.seeds:
            raise ToolError(f"seed {seed} is not in this sweep")
        return self.seeds.index(seed)

    def write(self, directory: Path) -> dict[str, str]:
        """Write the arrays and a manifest that hashes each of them."""
        directory.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for name in ARRAY_NAMES:
            path = directory / f"{name}.npy"
            np.save(path, getattr(self, name), allow_pickle=False)
            hashes[f"{name}.npy"] = sha256_of(path)
        payload = dict(self.manifest)
        payload["arm"] = self.arm
        payload["seeds"] = list(self.seeds)
        payload["freeze_depths"] = list(self.freeze_depths)
        payload["array_sha256"] = hashes
        payload["shapes"] = {name: list(getattr(self, name).shape) for name in ARRAY_NAMES}
        (directory / MANIFEST_NAME).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return hashes


def load_runs(directory: Path, verify_hashes: bool = True) -> SweepRuns:
    """Load one arm's measurement, checking the arrays are the ones the manifest names.

    Hash checking is on by default. A fixture that silently drifts from its manifest
    is how a repository ends up publishing a number from an array nobody can
    reproduce, and the check costs a few milliseconds.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise ToolError(f"no measurement manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        path = directory / f"{name}.npy"
        if not path.exists():
            raise ToolError(f"measurement array {path} is missing")
        if verify_hashes:
            expected = manifest.get("array_sha256", {}).get(f"{name}.npy")
            if expected is None:
                raise ToolError(f"the manifest does not hash {name}.npy")
            actual = sha256_of(path)
            if actual != expected:
                raise ToolError(
                    f"{name}.npy does not match its manifest hash "
                    f"(expected {expected[:12]}, found {actual[:12]})"
                )
        arrays[name] = np.load(path, allow_pickle=False)
    return SweepRuns(
        arm=manifest["arm"],
        seeds=tuple(manifest["seeds"]),
        freeze_depths=tuple(manifest["freeze_depths"]),
        manifest=manifest,
        **arrays,
    )
