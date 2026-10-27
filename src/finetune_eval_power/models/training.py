"""The deterministic training loop that produces the committed measurement.

Four choices here are load bearing, and each prevents a specific failure.

**The suite and the probe are rendered once, not per seed.** A real eval suite is a
fixed set of inputs. If the inputs moved with the training seed, the seed-to-seed
spread would mix two different things and the refusal rule in `power.variance` would
be measuring the generator instead of the training.

**The training data does move with the seed.** Seed variance answers "retrain this
configuration, how far does the measured regression move", and a retraining that
reuses the identical batch order understates it.

**One pretrain per seed, reused by every freeze depth.** The depths are variations of
the same fine-tune, so they must start from identical weights. Pretraining per depth
would put initialisation noise inside the comparison the freeze-depth experiment is
trying to make.

**In-distribution accuracy is read off the probe, not off the fine-tune batch.**
Scoring the fine-tune set would report memorisation. The probe already enumerates the
whole space, so the fine-tune region is a mask over cases the model never trained on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from finetune_eval_power.config import ArmConfig
from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator import sequence as sequence_generator
from finetune_eval_power.generator import vision as vision_generator
from finetune_eval_power.generator.attributes import Item
from finetune_eval_power.models.architectures import apply_freeze, build_model
from finetune_eval_power.slices.spec import CoverageProfile
from finetune_eval_power.slices.suite import (
    EvalSuite,
    ProbeSet,
    build_finetune_items,
    build_probe,
    build_suite,
)

# oneDNN reorders floating point accumulations for speed. Within one machine it is
# deterministic either way, but turning it off removes one source of difference
# between the machine that captured the fixture and the machine that re-measures it.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

#: Pretraining sees the whole attribute space, which is the only reason the model
#: ever has the capabilities this repository then watches it lose.
UNRESTRICTED = CoverageProfile(allowed={})


def configure_determinism(seed: int) -> None:
    """Pin every generator Keras consults, and turn on deterministic kernels."""
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def render_items(arm: ArmConfig, items: list[Item], rng: np.random.Generator) -> np.ndarray:
    if arm.name == "vision":
        return vision_generator.render_batch(items, rng, arm.space)
    if arm.name == "sequence":
        return sequence_generator.render_batch(items, rng, arm.space)
    raise ToolError(f"unknown arm {arm.name!r}")


def labels_of(items: list[Item]) -> np.ndarray:
    return np.array([item.label for item in items], dtype=np.int32)


@dataclass(frozen=True)
class FixedCases:
    """The suite and the probe, rendered once and shared by every seed."""

    suite: EvalSuite
    suite_x: np.ndarray
    suite_y: np.ndarray
    probe: ProbeSet
    probe_x: np.ndarray
    probe_y: np.ndarray


def build_fixed_cases(arm: ArmConfig) -> FixedCases:
    """Build and render the suite and the probe from the arm's own fixed seed."""
    suite = build_suite(
        arm.space, arm.suite_profile, arm.suite_size, np.random.default_rng(arm.suite_seed)
    )
    probe = build_probe(arm.space, arm.probe_repeats)
    suite_x = render_items(arm, list(suite.items), np.random.default_rng(arm.suite_seed + 1))
    probe_x = render_items(arm, list(probe.items), np.random.default_rng(arm.suite_seed + 2))
    return FixedCases(
        suite=suite,
        suite_x=suite_x,
        suite_y=suite.labels(),
        probe=probe,
        probe_x=probe_x,
        probe_y=probe.labels(),
    )


def correct_mask(model, x: np.ndarray, y: np.ndarray, batch_size: int) -> np.ndarray:
    """Per-case correctness. The whole repository is built on these boolean vectors."""
    predictions = model.predict(x, batch_size=batch_size, verbose=0)
    return np.argmax(predictions, axis=1).astype(np.int32) == y


@dataclass(frozen=True)
class DepthResult:
    """One fine-tune at one freeze depth, scored on the suite and on the probe."""

    freeze_depth: int
    suite_after: np.ndarray
    probe_after: np.ndarray
    suite_curve: np.ndarray


@dataclass(frozen=True)
class SeedResult:
    seed: int
    suite_before: np.ndarray
    probe_before: np.ndarray
    depths: tuple[DepthResult, ...]


def _compile(model, learning_rate: float) -> None:
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
    )


def train_seed(arm: ArmConfig, seed: int, cases: FixedCases) -> SeedResult:
    """Pretrain on the whole space once, then fine-tune at every freeze depth."""
    configure_determinism(seed)
    rng = np.random.default_rng(seed)

    pretrain_items = list(build_suite(arm.space, UNRESTRICTED, arm.pretrain_items, rng).items)
    pretrain_x = render_items(arm, pretrain_items, rng)
    pretrain_y = labels_of(pretrain_items)

    model = build_model(arm.name, arm.space)
    _compile(model, arm.learning_rate)
    model.fit(
        pretrain_x,
        pretrain_y,
        epochs=arm.pretrain_epochs,
        batch_size=arm.batch_size,
        shuffle=True,
        verbose=0,
    )
    pretrained_weights = model.get_weights()
    suite_before = correct_mask(model, cases.suite_x, cases.suite_y, arm.batch_size)
    probe_before = correct_mask(model, cases.probe_x, cases.probe_y, arm.batch_size)

    finetune_items = build_finetune_items(arm.space, arm.finetune_profile, arm.finetune_items, rng)
    finetune_x = render_items(arm, finetune_items, rng)
    finetune_y = labels_of(finetune_items)

    depth_results: list[DepthResult] = []
    for freeze_depth in arm.freeze_depths:
        model.set_weights(pretrained_weights)
        apply_freeze(model, arm.name, freeze_depth)
        _compile(model, arm.finetune_learning_rate)
        curve = [float(np.mean(suite_before))]
        for _ in range(arm.finetune_epochs):
            model.fit(
                finetune_x,
                finetune_y,
                epochs=1,
                batch_size=arm.batch_size,
                shuffle=True,
                verbose=0,
            )
            curve.append(
                float(np.mean(correct_mask(model, cases.suite_x, cases.suite_y, arm.batch_size)))
            )
        depth_results.append(
            DepthResult(
                freeze_depth=freeze_depth,
                suite_after=correct_mask(model, cases.suite_x, cases.suite_y, arm.batch_size),
                probe_after=correct_mask(model, cases.probe_x, cases.probe_y, arm.batch_size),
                suite_curve=np.array(curve, dtype=np.float32),
            )
        )

    return SeedResult(
        seed=seed,
        suite_before=suite_before,
        probe_before=probe_before,
        depths=tuple(depth_results),
    )
