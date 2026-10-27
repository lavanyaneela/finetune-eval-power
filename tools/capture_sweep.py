#!/usr/bin/env python3
"""Captures the committed measurement by training every seed at every freeze depth.

This is the only script in the repository that needs the training extra, and the
only one that takes minutes rather than seconds. Everything downstream reads what it
writes. Run it when the generator, the model or the seed budget changes, never to
refresh a number that did not move.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finetune_eval_power.config import ARMS, RUNS_DIRECTORY, SWEEP_SEEDS  # noqa: E402
from finetune_eval_power.sweep import run_sweep  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), action="append", dest="arms")
    parser.add_argument("--out", default=str(ROOT / RUNS_DIRECTORY))
    parser.add_argument("--seeds", type=int, default=len(SWEEP_SEEDS))
    arguments = parser.parse_args()

    chosen = arguments.arms or sorted(ARMS)
    seeds = SWEEP_SEEDS[: arguments.seeds]
    for name in chosen:
        arm = ARMS[name]
        started = time.perf_counter()
        print(f"capturing {name}: {len(seeds)} seeds, depths {arm.freeze_depths}", flush=True)
        runs = run_sweep(arm, seeds, progress=True)
        directory = Path(arguments.out) / name
        runs.write(directory)
        elapsed = time.perf_counter() - started
        print(f"  wrote {directory} in {elapsed:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
