"""The arithmetic that turns a suite size into a detection floor.

Split across four modules because three different questions get confused in
practice: how large an effect this suite could see (`paired`), what the usual
unpaired calculation would have claimed instead (`unpaired`), how uncertain a
measured per-slice drop is (`bootstrap`), and the part of the noise that no suite
size removes (`variance`).
"""

from finetune_eval_power.power.bootstrap import (
    bootstrap_drop_interval,
    bootstrap_seed_interval,
)
from finetune_eval_power.power.counts import PairedCounts, paired_counts
from finetune_eval_power.power.paired import (
    mcnemar_exact_power,
    mcnemar_normal_power,
    mcnemar_p_value,
    minimum_detectable_effect,
    required_suite_size,
)
from finetune_eval_power.power.unpaired import unpaired_minimum_detectable_effect
from finetune_eval_power.power.variance import SeedVariance, seed_variance

__all__ = [
    "PairedCounts",
    "SeedVariance",
    "bootstrap_drop_interval",
    "bootstrap_seed_interval",
    "mcnemar_exact_power",
    "mcnemar_normal_power",
    "mcnemar_p_value",
    "minimum_detectable_effect",
    "paired_counts",
    "required_suite_size",
    "seed_variance",
    "unpaired_minimum_detectable_effect",
]
