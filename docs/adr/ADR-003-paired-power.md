# ADR-003: The power calculation is paired, and exact

**Status:** accepted

## Context

An eval suite scores the same cases before and after a fine-tune. The two accuracies
that result are not two independent samples. They are two readings of one sample,
and almost every case agrees with itself across the pair.

This changes the arithmetic more than it looks. In a paired design all the
information about the difference lives in the discordant cases, the ones whose
verdict changed. Everything else cancels. So the question "is this suite big enough"
has the wrong shape: what governs detection is how many cases change their mind, and
a suite of cases that never change their mind is not improved by adding more of
them.

## Decision

Three things, each of which moves the published number.

**The test is McNemar, and the power calculation matches it.** The published
detection floor comes from an exact conditional calculation: the discordant count is
binomial in the suite size, and the power is an expectation over that count rather
than its value at the mean. Taking the value at the mean overstates power whenever
the mean is small, which is exactly the regime this kind of suite sits in.

**The normal approximation is kept, and used only for the curve.** It is the closed
form a reader will reach for, and the suite size curve needs thousands of
evaluations. Its optimism against the exact value is published in E1 rather than
asserted here.

**The unpaired calculation is implemented too, under its own name.** A team that
reports "accuracy went from 91 to 89" and asks whether the suite was big enough
almost always reaches for two independent proportions. On this measurement that
calculation would claim this suite needs a regression of 5.70 points to see,
where the paired calculation that matches the design puts the floor at 3.41 points.
The unpaired number is not a small correction, and it errs toward buying eval cases
that were not needed.

## The test is two sided, which is a choice

A regression gate arguably only cares about the downward direction, and a one sided
test would lower the floor. Two sided is kept as the default because it is the
conventional reading of "did anything change", and because the improvement direction
turns out to matter here: on this measurement the suite moves upward, and a one
sided regression test would not have noticed that at all. The one sided variant is
available and the difference is a parameter, not a rewrite.

## Delegation, stated so it is not mistaken for original work

The hypothesis test itself is `statsmodels.stats.contingency_tables.mcnemar`. The
power arithmetic, the detection floor search and the required-suite-size search are
this repository's own, and a test checks the unpaired implementation against
`statsmodels.stats.power` so the one piece that has a library equivalent is pinned
to it.

## What this costs

The exact calculation is slower than the closed form, which mattered more than
expected. The first implementation made one library call per candidate inside the
rejection region search, so a single power evaluation was quadratic in the suite
size and a bisection over it did not terminate in any usable time. It is now driven
by the inverse CDF, and the original walk is kept as a test referee so the fast path
cannot drift from it.
