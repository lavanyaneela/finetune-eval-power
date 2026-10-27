"""Renders a sequence item from its attribute cell.

The sequence arm exists to show that the power arithmetic is a property of the
measurement and not of convolutions. Its attributes are deliberate analogues of the
vision ones: length stands in for scale, amplitude for contrast, ordering for
occlusion, and noise is noise. The two arms therefore produce comparable slice
tables rather than two unrelated experiments.
"""

from __future__ import annotations

import numpy as np

from finetune_eval_power.generator.attributes import SEQUENCE_SPACE, AttributeSpace, Item

MAX_LENGTH = 24
VOCAB = 16
PAD = 0
LEVELS = 13  # tokens 1 to 13 carry the trajectory, so PAD stays unambiguous
SPAN = {"short": 10, "medium": 16, "long": 24}
#: How far the motif is allowed to depart from the midpoint. The weak setting is
#: the sequence analogue of low contrast: the motif is still there and there is
#: less of it.
AMPLITUDE = {"weak": 0.055, "strong": 0.125}
# Even the clean setting carries some corruption. A noiseless arm solves its task
# perfectly, and a model at a hundred percent has no capability left to lose, which
# would make the whole measurement vacuous.
NOISE_RATE = {"clean": 0.14, "noisy": 0.38}


def _trajectory(motif: str, span: int, amplitude: float) -> np.ndarray:
    """A coarse numeric trajectory, quantised to token ids 1 to LEVELS."""
    # The four motifs are deliberately compressed into the middle of the range and
    # overlap each other. Well separated motifs are classified perfectly at every
    # length and noise level, and a task nothing can fail measures nothing.
    t = np.linspace(0.0, 1.0, span)
    if motif == "ascend":
        values = 0.5 - amplitude + 2.0 * amplitude * t
    elif motif == "descend":
        values = 0.5 + amplitude - 2.0 * amplitude * t
    elif motif == "plateau":
        values = np.full(span, 0.5)
    elif motif == "spike":
        values = np.where(np.abs(t - 0.5) < 0.14, 0.5 + 1.6 * amplitude, 0.5 - 0.3 * amplitude)
    else:
        raise ValueError(f"unknown motif {motif!r}")
    return np.clip((values * (LEVELS - 1)).round().astype(int) + 1, 1, LEVELS)


def render(cell: dict[str, str], rng: np.random.Generator) -> np.ndarray:
    """One padded integer sequence of length MAX_LENGTH for this cell."""
    SEQUENCE_SPACE.validate_cell(cell)
    span = SPAN[cell["length"]]
    tokens = _trajectory(cell["motif"], span, AMPLITUDE[cell["amplitude"]])

    rate = NOISE_RATE[cell["noise"]]
    corrupt = rng.random(span) < rate
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
