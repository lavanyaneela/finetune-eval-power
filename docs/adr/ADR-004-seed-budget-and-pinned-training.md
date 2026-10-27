# ADR-004: The seed budget, and a measurement that is captured rather than repeated

**Status:** accepted

## Context

Two separate questions arrive together here, and they have the same root. How many
times should a configuration be retrained before its measured regression is
believed, and should the published figures be recomputed from training on every
verify or read from a captured measurement.

## Decision one: 10 retrainings, and the spread is a published figure

Each arm is trained with 10 seeds at every freeze depth. Training data is resampled
per seed, while the eval suite and the referee probe are rendered once and shared,
so the spread measures retraining and not the generator.

Ten was chosen because the variance estimate has to be stable enough to divide by. A
standard deviation from three runs is itself so noisy that the refusal rule would
fire or not fire depending on which three. Running 10 keeps the estimate usable while
keeping a full capture inside a single sitting, which matters because a capture
nobody will rerun is a capture that silently goes stale.

The spread is not a diagnostic. It is the headline's companion, and the arithmetic
has to be done in full or the claim is false. Retraining this configuration moves the
measured suite regression with a standard **deviation of 2.002** points. That standard
deviation is **not** the quantity to compare against the suite's floor, and an earlier
draft of this ADR made exactly that comparison and reached the right conclusion by a
wrong route. The refusal threshold is 1.96 of those standard deviations, so the
**noise floor reaches 3.92** points, and that is what exceeds the suite's
**detection floor of 3.41**. The tool's refusal checks the retraining spread before it
checks the suite size for that reason, and `tools/check_numbers.py` now evaluates both
inequalities as relations so neither can be restated loosely again.

## Decision two: the measurement is captured, hashed and committed

Every published figure is computed from boolean correctness vectors stored under
`fixtures/runs/`, with a manifest that hashes each array. The analysis re-runs on
every verify. The training does not.

**Why not retrain on every verify.** Floating point arithmetic on a CPU is not
bitwise portable. The same seed, the same library versions and a different
microarchitecture produce slightly different weights and therefore slightly
different accuracies. A repository that published figures to two decimal places and
re-derived them on an unknown runner would have a red badge it could not fix, and
the usual response to that is to loosen the check until it stops failing, at which
point nothing is checked.

**What is kept honest instead.** A separate CI job retrains the whole sweep from
scratch and reports the drift against the committed measurement in the job summary.
It publishes the difference rather than asserting there is none.

That job is manual and non-blocking, and it is worth saying so rather than letting a
reader assume otherwise. It is gated on a workflow dispatch and runs with
`continue-on-error`, for two reasons. It takes the better part of an hour, which is
not something to put on every push. And it exists to measure a difference this ADR
already predicts, so wiring it to fail the build would paint the badge red for
something documented, and a badge that is red for a known reason is a badge everybody
learns to ignore. The result is a number a reader can go and look at, not a gate.

The arrays are stored as individual `.npy` files rather than one `.npz`, because the
zip container inside an `.npz` embeds a modification time and the fixture would stop
matching itself a second after it was written.

## Alternatives considered

**A smaller seed budget, and a wider interval.** Cheaper, and it would have made the refusal
rule depend on which particular seeds were drawn.

**A larger seed budget.** The variance estimate improves as the square root, so doubling the
capture buys a modest tightening. The tool prices this for the user: when it refuses
a claim it prints how many further retrainings would be needed to resolve it, and that
number is usually large enough to be the real answer to the question.

**Retraining in CI and rounding the published figures to one decimal place.** This
only moves the problem. A figure rounded until it is portable is a figure that has
been chosen to survive the check rather than to be informative.
