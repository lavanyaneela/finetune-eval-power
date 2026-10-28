# ADR-005: What this repository cannot establish

**Status:** accepted

Every other ADR records a choice. This one records the edge of what the choices can
support, so that nothing in the README is read as a wider claim than it is.

## The capability space is unbounded, so coverage is unprovable

This repository measures **40 named capability** slices for the vision arm and
**29 sequence slices named** for the other, and reports every one of them. That is not coverage. The set of
capabilities a model might have is not enumerable, and a slice only exists here
because somebody wrote down an attribute. Every figure published about what the
suite missed is a lower bound: it counts the capabilities that were named and
measured, and says nothing about the ones that were not.

Any sentence of the form "the model did not regress" is out of reach here. It would
remain out of reach with a real model, a real dataset and unlimited compute, because
the quantifier is over a set nobody can enumerate. What is in reach, and what this
repository publishes, is a detection floor and a list of named things that fell
below it.

## A count of slices is not a count of capabilities

Slices nest. A single-attribute slice and the pairs that partition it are named and
measured separately, so eight missed slices are not eight independent regions: on the
vision arm they cover **3 regions once nesting** is removed. Every missed count in this
repository is a count of named slices, and the region count is published beside it for
that reason.

The companion number matters too. Eight slices selected from forty invites the
question of how many went the other way, and **5 slices moved up** with retraining
intervals clear of zero. Both counts are published, because publishing only the first
would be selecting the evidence.

## Every floor comes from the suite, and that choice moved a published number

The per-slice floor in the E3 table and in the report is computed from the slice's
suite case count and the slice's rate of changed verdicts **on the suite's own
cases**. So is the per-class floor. Both columns share a convention, and a reader
can compare them.

It was not always so. An earlier version used the probe's rate at the suite's case
count, on the reasoning that many slices hold too few suite cases to estimate a rate
from and the probe's estimate is steadier. That is true and it is the wrong trade,
because the question being asked is what the eval suite could have done, and the
probe's rate is not a property of the eval suite. The referee is steadier in a
particular direction: using it moved five slices out of "no usable power" and
published 19 blind slices where the suite's own behaviour says 24. A convention that
makes the instrument under test look better is not a neutral one.

The probe-rate floor is still computed, under a name that says so, and two published
figures use it. One is the floor for the worst missed class at the probe's rate,
quoted in a sentence that says in words which rate it is. The other is the count of
blind slices the probe's rate would report, published next to the suite's count so
the size of the choice is visible rather than argued about. The README sets it beside
the suite-rate floor for the same class so the disagreement is visible, and the
conclusion there is built to survive either choice.

What remains genuinely uncertain is that these rates are estimated from the suite's
own cases for each slice, which run from about a hundred to the whole suite, and a
rate from the thin end is noisy. The floors inherit that noise. This
repository does not quantify that, and it is the first thing to add.

## The effect sizes are about these models

Both arms are small models trained on generated data. The numbers describe them.
A reader should take the arithmetic, the refusal rule and the shape of the blindness
as the transferable content, and should not expect a production fine-tune to lose a
similar amount.

## The fine-tune distribution was chosen, not observed

The narrowing that causes the damage is a configuration value. In production the
equivalent narrowing happens by accident and nobody records how narrow it was.
Measuring this on real traffic would require logging the attribute distribution of
the fine-tune set against the attribute distribution the model is expected to serve,
which most teams do not have, and the absence of that logging is the production form
of the same blindness.

## The arrival of the regression is not modelled

This repository compares one before state against one after state. It says nothing
about when during deployment a regression becomes visible, which is a different
question with a different shape and belongs to a monitoring tool rather than to an
eval suite.

## Floating point reproducibility stops at the machine boundary

Within one machine the training is deterministic and a test asserts it. Across
machines it is not, which is why the measurement is captured rather than recomputed.
ADR-004 has the detail. The practical consequence for a reader: regenerating the
sweep on different hardware will move the figures slightly, and the CI job that does
exactly that publishes the drift rather than hiding it.

## What would be needed to go further

One claim in this repository genuinely needs something it does not have, and it is
worth naming precisely. To say "an eval suite of this size, on this real model, has
this detection floor" requires the discordance of that real suite on that real
model, measured across several real retrainings of it. Discordance is cheap to measure and
almost nobody measures it, because the paired structure of the comparison is usually
thrown away before anyone looks. That is the single number a team should take from
here and compute on their own suite.
