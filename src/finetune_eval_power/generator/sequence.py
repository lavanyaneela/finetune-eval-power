"""Renders a sequence item from its attribute cell.

The sequence arm exists to show that the power arithmetic is a property of the
measurement and not of convolutions. Its attributes are chosen to be the sequence
analogues of the vision ones: length stands in for scale, noise for contrast,
ordering for occlusion. The two arms therefore produce comparable slice tables.
"""

from __future__ import annotations

import numpy as np

from finetune_eval_power.generator.attributes import SEQUENCE_SPACE, AttributeSpace, Item

MAX_LENGTH = 24
VOCAB = 16
PAD = 0
MARKER = 15
LEVELS = 13  # tokens 1 to 13 carry the trajectory, so PAD and MARKER stay unambiguous
SPAN = {"short": 10, "medium": 16, "long": 24}
NOISE_RATE = {"clean": 0.0, "noisy": 0.18}
MARKER_COUNT = {"even": (2, 4), "odd": (1, 3)}


def _trajectory(motif: str, span: int) -> np.ndarray:
    """A coarse numeric trajectory, quantised to token ids 1 to LEVELS."""
    t = np.linspace(0.0, 1.0, span)
    if motif == "ascend":
        values = t
    elif motif == "descend":
        values = 1.0 - t
    elif motif == "plateau":
        values = np.full(span, 0.5)
    elif motif == "spike":
        values = np.where(np.abs(t - 0.5) < 0.12, 1.0, 0.15)
    else:
        raise ValueError(f"unknown motif {motif!r}")
    return np.clip((values * (LEVELS - 1)).round().astype(int) + 1, 1, LEVELS)


def render(cell: dict[str, str], rng: np.random.Generator) -> np.ndarray:
    """One padded integer sequence of length MAX_LENGTH for this cell."""
    SEQUENCE_SPACE.validate_cell(cell)
    span = SPAN[cell["length"]]
    tokens = _trajectory(cell["motif"], span)

    low, high = MARKER_COUNT[cell["parity"]]
    count = int(rng.choice([low, high]))
    positions = rng.choice(span, size=min(count, span), replace=False)
    tokens[positions] = MARKER

    rate = NOISE_RATE[cell["noise"]]
    if rate > 0.0:
        corrupt = rng.random(span) < rate
        # Noise never creates or destroys a marker, so parity stays the label it claims
        # to be and the two tasks remain independently controllable.
        corrupt &= tokens != MARKER
        tokens[corrupt] = rng.integers(1, LEVELS + 1, size=int(corrupt.sum()))

    if cell["ordering"] == "shuffled":
        block = max(2, span // 4)
        chunks = [tokens[i : i + block] for i in range(0, span, block)]
        order = rng.permutation(len(chunks))
        tokens = np.concatenate([chunks[i] for i in order])

    padded = np.full(MAX_LENGTH, PAD, dtype=np.int32)
    padded[:span] = tokens[:span]
    return padded


def render_batch(
    items: list[Item], rng: np.random.Generator, space: AttributeSpace = SEQUENCE_SPACE
) -> np.ndarray:
    if space is not SEQUENCE_SPACE:
        raise ValueError("render_batch renders the sequence space only")
    out = np.zeros((len(items), MAX_LENGTH), dtype=np.int32)
    for position, item in enumerate(items):
        out[position] = render(dict(item.cell), rng)
    return out
