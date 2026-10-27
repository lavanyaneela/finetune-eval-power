# ADR-001: A procedural generator rather than a public dataset

**Status:** accepted

## Context

This repository measures what an eval suite cannot see. That requires three things
at once: a set of named capabilities, a suite that samples some of them and not
others, and the ability to produce as many examples of any capability as the
measurement needs, including the ones the suite was never given.

A capability here is a region of attribute space, for example small scale at low
contrast. To hold such a region out deliberately and then measure it afterwards,
the attributes have to be addressable. In a fixed image dataset they are not. The
labels are there and the factors of variation are not, so "the suite never saw
small, low contrast examples" is a sentence that cannot be written, let alone
checked.

## Decision

Both arms generate their data from an attribute space defined once in
`generator/attributes.py`. Every item is rendered from a fully specified cell, every
cell carries its label, and a capability slice is a constraint on one or two
attributes of that cell.

The eval suite is a stratified sample from a coverage region. The referee probe
enumerates the entire space a fixed number of times, which is why every slice in the
published tables has an exact denominator rather than a sampled one.

## Alternatives considered

**A public image dataset with annotated factors.** The closest candidates annotate a
handful of factors and fix the rest, and none let a region be excluded from one
sample and enumerated in another. Even where the factors exist, the examples are
finite, so a thin slice has a denominator set by whatever the collectors happened to
photograph.

**Sampling a real production log.** This was rejected for the same reason project
one rejected it: the interesting region is the one nobody collected.

## What this costs, stated plainly

The results are about a synthetic task. Nothing here establishes how much a real
model forgets, and the repository does not claim it. What transfers is the
arithmetic and the shape of the blindness, not the magnitude.

A second cost is more subtle and worth naming because a reviewer will think of it.
The fine-tune distribution was narrowed on purpose, so it is not a surprise that
something suffered. The measurement is not "does narrowing hurt", which is already
known. It is which named capabilities move, by how much, and whether the suite could
have resolved any of it. Those are measured against injected ground truth, and
several of them came out differently from the way the build expected: one slice
improved by nearly eight points while another lost ten, and that cancellation is
what keeps the suite quiet.

## A rejected design that is worth recording

The first version of this repository pretrained on one task and fine-tuned on a
different one, with the original head frozen. In a model this small that produces
total collapse: the eval suite lost more than fifty points at every freeze depth
below a fully frozen backbone, and zero at a fully frozen one. There was no
intermediate regime, so there was nothing to measure.

That is a fact about model capacity rather than about eval suites. Real models
degrade gracefully under a task switch because they are enormously
over-parameterised, and reproducing that cheaply is not possible. So the fine-tune
here is the one teams run far more often anyway: continued training on the same
task over a narrower, more recent slice of data.
