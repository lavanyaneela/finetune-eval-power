"""Every knob the measurement depends on, in one place and serialisable.

A threshold that lives in a function body cannot appear in a manifest, and a figure
whose inputs are not in the manifest is not reproducible even when the code is. So
the arms, the coverage regions, the seed budget and the decision rule are all data,
and the sweep writes them next to the arrays they produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from finetune_eval_power.generator.attributes import SEQUENCE_SPACE, VISION_SPACE, AttributeSpace
from finetune_eval_power.slices.spec import CoverageProfile

#: Significance level for every hypothesis test the tool runs.
ALPHA = 0.05
#: The power a suite must reach before the tool says it can see an effect.
TARGET_POWER = 0.80
#: Multiplier on the seed-to-seed standard deviation below which no suite size helps.
#: 1.96 because that is the two-sided normal quantile at ALPHA: a claim smaller than
#: the retraining noise interval is not a suite-size problem. See ADR-004.
SEED_NOISE_MULTIPLIER = 1.96


@dataclass(frozen=True)
class ArmConfig:
    """One experimental arm: a space, two regions, a model size and a training budget."""

    name: str
    space: AttributeSpace
    suite_profile: CoverageProfile
    finetune_profile: CoverageProfile
    pair_attributes: tuple[str, ...]
    suite_size: int
    probe_repeats: int
    pretrain_items: int
    finetune_items: int
    pretrain_epochs: int
    finetune_epochs: int
    batch_size: int
    freeze_depths: tuple[int, ...]
    learning_rate: float = 0.004
    suite_seed: int = 20260101

    def describe(self) -> dict[str, Any]:
        """A JSON-safe copy for the manifest."""
        payload = asdict(self)
        payload["space"] = {
            "name": self.space.name,
            "attributes": {k: list(v) for k, v in self.space.attributes.items()},
            "task_a_attribute": self.space.task_a_attribute,
            "task_b_attribute": self.space.task_b_attribute,
        }
        payload["suite_profile"] = {k: list(v) for k, v in self.suite_profile.allowed.items()}
        payload["finetune_profile"] = {k: list(v) for k, v in self.finetune_profile.allowed.items()}
        payload["pair_attributes"] = list(self.pair_attributes)
        payload["freeze_depths"] = list(self.freeze_depths)
        return payload


VISION_ARM = ArmConfig(
    name="vision",
    space=VISION_SPACE,
    # What a team thought to cover: well lit, not tiny, not badly occluded.
    suite_profile=CoverageProfile(
        allowed={
            "scale": ("medium", "large"),
            "contrast": ("high",),
            "occlusion": ("none", "corner"),
        }
    ),
    # What the fine-tune data happened to be: narrower still, and narrowed along
    # axes nobody wrote down.
    finetune_profile=CoverageProfile(
        allowed={
            "scale": ("large",),
            "contrast": ("high",),
            "occlusion": ("none",),
            "rotation": ("r000", "r090"),
        }
    ),
    pair_attributes=("scale", "contrast", "occlusion"),
    suite_size=480,
    probe_repeats=2,
    pretrain_items=3072,
    finetune_items=1536,
    pretrain_epochs=9,
    finetune_epochs=6,
    batch_size=64,
    freeze_depths=(0, 1, 2, 3),
)

SEQUENCE_ARM = ArmConfig(
    name="sequence",
    space=SEQUENCE_SPACE,
    suite_profile=CoverageProfile(
        allowed={
            "length": ("medium", "long"),
            "noise": ("clean",),
            "ordering": ("forward",),
        }
    ),
    finetune_profile=CoverageProfile(
        allowed={
            "length": ("long",),
            "noise": ("clean",),
            "ordering": ("forward",),
        }
    ),
    pair_attributes=("length", "noise", "ordering"),
    suite_size=480,
    probe_repeats=8,
    pretrain_items=3072,
    finetune_items=1536,
    pretrain_epochs=9,
    finetune_epochs=6,
    batch_size=64,
    freeze_depths=(0, 1, 2),
)

ARMS: dict[str, ArmConfig] = {arm.name: arm for arm in (VISION_ARM, SEQUENCE_ARM)}

#: Seeds used by the committed sweep. Ten is not a round number chosen for looks:
#: ADR-004 records the variance-stability measurement that picked it.
SWEEP_SEEDS: tuple[int, ...] = (11, 23, 37, 41, 59, 67, 73, 89, 97, 101)

#: Where the committed measurement lives, relative to the repository root.
RUNS_DIRECTORY = "fixtures/runs"

DEFAULT_CLAIM_POINTS: float = 2.0


@dataclass(frozen=True)
class DecisionRule:
    """The thresholds `fep power` branches on, so the exit code is configuration."""

    alpha: float = ALPHA
    target_power: float = TARGET_POWER
    seed_noise_multiplier: float = SEED_NOISE_MULTIPLIER
    claim_points: float = DEFAULT_CLAIM_POINTS
    tags: tuple[str, ...] = field(default_factory=tuple)
