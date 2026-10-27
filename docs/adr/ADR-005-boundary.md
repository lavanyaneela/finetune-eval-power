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

## A per-slice floor mixes two sources, and the column says so

The per-slice floor in the E3 table is computed from the slice's suite case count and
the slice's discordance **as measured on the probe**, because many slices have too few
suite cases to estimate a discordance from. For the label classes, where the suite has
enough cases, the floor is computed entirely from the suite and those two figures
differ: on the worst class the probe-based estimate is higher than the suite-based one.
Neither is wrong, and a reader comparing the two columns is entitled to know they do
not share an input.

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
