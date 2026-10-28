# ADR-002: Keras, and a backbone with named blocks

**Status:** accepted

## Context

Freeze depth is the engineering knob this repository turns. For the knob to mean
anything, "freeze the first k blocks" has to be a statement a reader can check
against the model definition at a glance, and it has to be exactly true at
runtime.

## Decision

Keras on the TensorFlow CPU backend. The backbone is a sequence of explicitly named
blocks, and freezing is a prefix operation over those names.

Two reasons, in order of weight.

**The freeze operation is legible.** Setting `layer.trainable` on a named layer and
recompiling is a two line operation whose effect a reviewer can confirm by reading
the model summary. The experiment this repository is built around is a sweep over
that one setting, so the setting should be the most readable thing in the file.

**Working deliberately in the other major framework.** The author's production work
is in PyTorch, which makes PyTorch the default rather than a choice. Building the
freeze-depth machinery in Keras forced the layer-freezing semantics to be understood
rather than copied, and it surfaced the normalisation issue below, which a habitual
implementation would have shipped.

## The decision inside the decision: no normalisation layers

A normalisation layer keeps updating its running statistics during a forward pass
even when its weights are frozen. A fully frozen backbone containing one would still
drift, the maximum freeze depth would not be a true control, and every freeze-depth
figure in the repository would be measuring something other than what it claims. The
backbone therefore contains none, and a test asserts it.

This is the kind of thing that is obvious once stated and silently wrong otherwise.
It is recorded here rather than as a comment because it changes what the published
numbers mean.

## Alternatives considered

**PyTorch.** Equivalent on the merits. Freezing is `requires_grad = False` over a
parameter group, which is no harder, and the author already knows it. Rejected for
the second reason above.

**Keras 3 on the PyTorch backend.** Would give the same layer API over a familiar
runtime. Rejected because it adds a dependency pairing that very few readers run, so
a reviewer reproducing the sweep would be the first person to exercise that
combination.

## What this costs

TensorFlow is a heavy dependency, so it sits behind an optional `train` extra.
Nothing that computes a published figure imports it, and a CI job installs without
it and runs the whole analysis path to keep that boundary honest.
