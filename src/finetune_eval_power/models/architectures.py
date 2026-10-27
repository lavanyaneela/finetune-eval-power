"""Two backbones with named blocks, because freeze depth has to mean something.

Both arms share one shape: a backbone with named blocks, then a single classifier
head. Pretraining runs the whole thing over the full attribute space. Fine-tuning
continues training the same head, on the same label, over a narrower distribution,
with the first `freeze_depth` blocks held fixed. Freeze depth is therefore a real
engineering knob rather than a demonstration: it is the dial between specialising on
the new distribution and keeping the representation that served the old one.

No batch normalisation anywhere, on purpose. A normalisation layer keeps updating its
running statistics even when its weights are frozen, so a fully frozen backbone would
still shift and the freeze-depth control would leak.
"""

from __future__ import annotations

from finetune_eval_power.errors import ToolError
from finetune_eval_power.generator.attributes import AttributeSpace
from finetune_eval_power.generator.sequence import MAX_LENGTH, VOCAB
from finetune_eval_power.generator.vision import GRID

LABEL_HEAD = "label"
VISION_BLOCKS = (
    "vision_block_1",
    "vision_block_2",
    "vision_block_3",
    "vision_block_4",
)
SEQUENCE_BLOCKS = ("sequence_embedding", "sequence_recurrent")


def block_names(arm_name: str) -> tuple[str, ...]:
    if arm_name == "vision":
        return VISION_BLOCKS
    if arm_name == "sequence":
        return SEQUENCE_BLOCKS
    raise ToolError(f"unknown arm {arm_name!r}")


def build_model(arm_name: str, space: AttributeSpace):
    """A two headed model for this arm. Imports Keras, so call it inside a sweep."""
    import keras

    if arm_name == "vision":
        inputs = keras.Input(shape=(GRID, GRID, 1), name="image")
        x = keras.layers.Conv2D(16, 3, padding="same", activation="relu", name=VISION_BLOCKS[0])(
            inputs
        )
        x = keras.layers.MaxPooling2D(name="pool_1")(x)
        x = keras.layers.Conv2D(24, 3, padding="same", activation="relu", name=VISION_BLOCKS[1])(x)
        x = keras.layers.MaxPooling2D(name="pool_2")(x)
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name=VISION_BLOCKS[2])(x)
        x = keras.layers.Conv2D(40, 3, padding="same", activation="relu", name=VISION_BLOCKS[3])(x)
        features = keras.layers.GlobalAveragePooling2D(name="pooled")(x)
    elif arm_name == "sequence":
        inputs = keras.Input(shape=(MAX_LENGTH,), dtype="int32", name="tokens")
        x = keras.layers.Embedding(VOCAB, 8, mask_zero=True, name=SEQUENCE_BLOCKS[0])(inputs)
        features = keras.layers.LSTM(12, name=SEQUENCE_BLOCKS[1])(x)
    else:
        raise ToolError(f"unknown arm {arm_name!r}")

    head = keras.layers.Dense(len(space.label_classes()), activation="softmax", name=LABEL_HEAD)(
        features
    )
    return keras.Model(inputs=inputs, outputs=head)


def apply_freeze(model, arm_name: str, freeze_depth: int) -> None:
    """Freeze the first `freeze_depth` backbone blocks, leaving the head trainable.

    At the maximum depth every block is frozen and only the classifier moves, which
    is the linear-probe end of the dial. It is included because it is the
    configuration a cautious team reaches for, and because a regression measured
    there cannot be blamed on representation drift.

    Raises rather than clamping when the depth is out of range. A silently clamped
    freeze depth would publish two different configurations under one label.
    """
    blocks = block_names(arm_name)
    if not 0 <= freeze_depth <= len(blocks):
        raise ToolError(
            f"freeze depth {freeze_depth} is outside 0 to {len(blocks)} for arm {arm_name!r}"
        )
    for layer in model.layers:
        layer.trainable = True
    for name in blocks[:freeze_depth]:
        model.get_layer(name).trainable = False
