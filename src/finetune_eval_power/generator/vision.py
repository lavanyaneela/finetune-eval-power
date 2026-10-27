"""Renders a vision item from its attribute cell.

Every attribute has to actually change the pixels, otherwise a "capability slice" is
a label with nothing behind it and a regression measured on it means nothing. So
rotation rotates the sampling frame rather than the finished bitmap (no resampling
blur), texture is applied in that same rotated frame so it interacts with rotation,
and occlusion removes evidence rather than dimming it.
"""

from __future__ import annotations

import numpy as np

from finetune_eval_power.generator.attributes import VISION_SPACE, AttributeSpace, Item

GRID = 24
RADIUS = {"small": 4.0, "medium": 6.0, "large": 8.5}
ROTATION_DEGREES = {"r000": 0.0, "r030": 30.0, "r060": 60.0, "r090": 90.0}
FOREGROUND = {"low": 0.45, "high": 1.0}
NOISE_SIGMA = 0.08


def _rotated_frame(rotation: str) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates of every pixel in the shape's own rotated frame, centred on zero."""
    centre = (GRID - 1) / 2.0
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    x = xx - centre
    y = yy - centre
    theta = np.deg2rad(ROTATION_DEGREES[rotation])
    cos, sin = np.cos(theta), np.sin(theta)
    return x * cos + y * sin, -x * sin + y * cos


def _shape_mask(shape: str, u: np.ndarray, v: np.ndarray, radius: float) -> np.ndarray:
    distance = np.sqrt(u**2 + v**2)
    if shape == "disc":
        return distance <= radius
    if shape == "ring":
        return (distance <= radius) & (distance >= radius * 0.55)
    if shape == "square":
        return (np.abs(u) <= radius * 0.85) & (np.abs(v) <= radius * 0.85)
    if shape == "triangle":
        # Three half planes of an upward equilateral triangle in the rotated frame.
        height = radius * 1.6
        return (v <= height * 0.5) & (v >= -height * 0.5 + np.abs(u) * 1.5)
    raise ValueError(f"unknown shape {shape!r}")


def _texture_mask(texture: str, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    if texture == "solid":
        return np.ones_like(u, dtype=bool)
    if texture == "striped":
        return (np.floor(v) % 3) < 2
    if texture == "dotted":
        return ((np.floor(u) % 3) < 2) & ((np.floor(v) % 3) < 2)
    raise ValueError(f"unknown texture {texture!r}")


def _apply_occlusion(canvas: np.ndarray, occlusion: str) -> np.ndarray:
    if occlusion == "none":
        return canvas
    out = canvas.copy()
    if occlusion == "corner":
        out[0:9, 0:9] = 0.0
    elif occlusion == "bar":
        middle = GRID // 2
        out[middle - 2 : middle + 2, :] = 0.0
    else:
        raise ValueError(f"unknown occlusion {occlusion!r}")
    return out


def render(cell: dict[str, str], rng: np.random.Generator) -> np.ndarray:
    """One 24 by 24 single channel image for this cell, as float32 in roughly [0, 1]."""
    VISION_SPACE.validate_cell(cell)
    u, v = _rotated_frame(cell["rotation"])
    jitter_x, jitter_y = rng.integers(-2, 3, size=2)
    u = u - jitter_x
    v = v - jitter_y
    mask = _shape_mask(cell["shape"], u, v, RADIUS[cell["scale"]])
    mask = mask & _texture_mask(cell["texture"], u, v)
    canvas = mask.astype(np.float32) * FOREGROUND[cell["contrast"]]
    canvas = _apply_occlusion(canvas, cell["occlusion"])
    canvas = canvas + rng.normal(0.0, NOISE_SIGMA, canvas.shape).astype(np.float32)
    return canvas.astype(np.float32).reshape(GRID, GRID, 1)


def render_batch(
    items: list[Item], rng: np.random.Generator, space: AttributeSpace = VISION_SPACE
) -> np.ndarray:
    """Render many items in their given order, which keeps the fixture reproducible."""
    if space is not VISION_SPACE:
        raise ValueError("render_batch renders the vision space only")
    out = np.zeros((len(items), GRID, GRID, 1), dtype=np.float32)
    for position, item in enumerate(items):
        out[position] = render(dict(item.cell), rng)
    return out
