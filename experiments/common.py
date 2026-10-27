"""Shared plumbing for the five experiments: load, write, print.

Every experiment reads the committed measurement, computes, writes one JSON file to
`docs/experiments/` and prints a table. None of them train anything, which is why
`make verify` runs all five in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from finetune_eval_power.analysis import ArmAnalysis, analyse
from finetune_eval_power.config import ARMS, RUNS_DIRECTORY, ArmConfig
from finetune_eval_power.runs import load_runs

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "docs" / "experiments"


def load_arm(name: str) -> tuple[ArmConfig, ArmAnalysis]:
    arm = ARMS[name]
    runs = load_runs(ROOT / RUNS_DIRECTORY / name)
    return arm, analyse(arm, runs)


def write_result(name: str, payload: dict[str, Any]) -> Path:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_DIRECTORY / f"{name}.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def print_table(headers: tuple[str, ...], rows: list[tuple[Any, ...]]) -> None:
    widths = [len(header) for header in headers]
    rendered = [[format_cell(cell) for cell in row] for row in rows]
    for row in rendered:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))
    line = "  ".join(header.ljust(widths[position]) for position, header in enumerate(headers))
    print(line)
    print("  ".join("-" * width for width in widths))
    for row in rendered:
        print("  ".join(cell.ljust(widths[position]) for position, cell in enumerate(row)))


def format_cell(value: Any) -> str:
    if value is None:
        return "unreachable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)
