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

Ten was chosen because the variance estimate has to be stable enough to divide by,
and that is a measurement rather than a feeling. Taking the standard deviation of the
first k retrainings for every k and asking when it stops moving, the estimate
**settles after 8** of them, in the sense that it and every later one sit within a
tenth of the final value.

That figure is computed over prefixes, so it is a property of the order the
retrainings happen to be listed in as much as of the estimator, and an audit was
right to say so. Repeating it over two thousand orderings of the same ten readings
puts the **median ordering at 8**, and in **9.4% of orderings** the estimate first
settles only at the tenth retraining, which is as late as this measurement can
express: the test is against the final estimate, so the last reading satisfies it
against itself by construction and no ordering can be reported as never settling.
Settling only at the last reading is the measurable form of not having settled. So
ten is the smallest defensible budget
rather than a comfortable one, and the honest reading is that a team doing this
properly should expect to need more, not fewer.

The consequence is not academic, because the refusal threshold is computed from this
estimate. A three retraining budget **would have set 1.54** points as the bar where
the settled estimate puts it at the **noise floor reaches 3.92** points. The tool
would have accepted, and confidently answered, claims between those two that the
measurement cannot support, and nothing downstream would have noticed, because a
threshold carries no record of how well it was estimated.

So the budget is the settling point with margin, not a round number. Ten also keeps a
full capture inside one sitting, which matters because a capture nobody will rerun is
a capture that silently goes stale. An earlier version of this ADR argued all of this
in words and recorded no measurement, while `config.py` told the reader that this
document held one.

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
`continue-on-error`, for two reasons. It is far and away the longest thing in the repository, which is
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
a claim it prints the TOTAL number of retrainings whose mean would resolve it, not a
number of further ones, because averaging shrinks the bar as one over the square root
of the count. On this measurement a two point claim needs four, which is fewer than
the ten already captured. That is the useful shape of the answer, not a large number:
it says a team that retrains a handful of times can settle a question the eval suite
cannot, and that the obstacle was never the size of the eval set.

**Retraining in CI and rounding the published figures to one decimal place.** This
only moves the problem. A figure rounded until it is portable is a figure that has
been chosen to survive the check rather than to be informative.
