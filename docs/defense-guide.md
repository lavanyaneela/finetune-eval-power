# Defense guide: finetune-eval-power

For the author, before an interview. Every figure here is checked against the
committed measurement by `tools/check_numbers.py`, so if a number in this document
has drifted the build fails.

## The 30 second version

"A passing eval suite gets read as evidence that fine-tuning broke nothing. It is
not. I built a tool that computes how small a regression a suite could actually have
detected, and then measures what fell below that. On my vision arm the suite's
accuracy went up, it never once flagged a regression, and eight named capabilities
had regressed past the suite's own detection floor. The worst one lost more than ten
points while the suite held a hundred and twenty cases of exactly that kind."

Then stop.

The question that usually comes next is "isn't that just because you held those
cases out". Half of it is, and the better half is not: two of the missed slices are
inside the region the suite samples, and the aggregate stays flat because one class
improved by nearly as much as the worst one lost.

## The claims, and how each is proved

### Claim 1: the suite reported nothing while capabilities regressed

**Command:** `python -m finetune_eval_power.cli blind --arm vision`

The suite **changed by -1.27** points, meaning it improved, and it raised a flag in
**0 of the retrainings**. Against that, **8 slices regressed** past the suite's own floor,
and **2 sit inside** the region the suite samples. The **worst is shape=square**, the
**worst lost 10.58** points, and the suite holds **120 cases of its own** for that shape.

**If pushed on why the aggregate stayed flat:** because another shape improved by
nearly eight points over the same fine-tune. The suite is not insensitive, it is
averaging over a dimension it never reports. That is the part of the result I did
not predict, and it is the part worth remembering.

### Claim 2: the detection floor is set by discordance, not by case count

**Command:** `python -m finetune_eval_power.cli curve --arm vision`

The suite scores the same cases before and after, so the comparison is paired and
all the information lives in the cases whose verdict changed. The **detection floor of 3.41**
points follows from that. The unpaired calculation most people reach for is the one the
**unpaired says 5.70** about, and seeing a one point regression **would need 5385** cases.

**If pushed on whether paired versus unpaired really matters:** it is not a small
correction, it is about a factor of one and a half here, and it points the wrong way
for a team's budget. The hypothesis test itself is statsmodels. The power arithmetic
is mine, and the unpaired path is pinned against `statsmodels.stats.power` in a test
so the piece with a library equivalent cannot drift from it.

### Claim 3: the suite is not the binding constraint anyway

**Command:** `python -m finetune_eval_power.cli power --arm vision --claim 2`, which
exits `2`.

Retraining the same configuration moves the measured regression with a standard
**deviation of 2.002** points, so the **noise floor reaches 3.92** points, which is wider
than the suite's own floor.
So for any claim smaller than that the tool refuses rather than recommending a larger
suite, and it prints how many retrainings would resolve it instead.

**If pushed on why this deserves its own exit code:** because the two answers send a
team in opposite directions. "Buy more eval cases" and "your eval set is not the
problem" cost very different amounts, and a power calculation can only ever give the
first, since its model of the world contains one noise source.

### Claim 4: the blindness is structural, not a property of this fixture

**Command:** `python -m finetune_eval_power.cli suite --arm vision`

Of **40 named capability** slices there are **16 it never samples**, and the suite has
**no usable power on 19** of them. With **power below 118** cases it cannot reach 80% power at any effect size at all. The second arm reproduces the shape with a different architecture
and a different data type: the **sequence missed 9** of its **29 sequence slices named**.

## Questions that are meant to be hard

**"Isn't this just a wrapper around statsmodels?"**
No, and the line is worth drawing precisely. statsmodels provides the McNemar test and
the unpaired power function. What this repository adds is the exact conditional power
for the paired design, the detection floor and required suite size searches on top of
it, the per-slice power accounting, and the refusal rule that compares a claim against
retraining spread rather than against suite size. The last one is the part I would
defend hardest, because it is the only part that can tell a team to stop buying eval
data.

**"The data is synthetic. Does any of this transfer?"**
The magnitudes do not and I do not claim them. What transfers is the arithmetic and
one cheap, concrete action: measure the discordance of your own suite across a few
retrainings. Almost nobody does, because the paired structure gets thrown away before
anyone looks at it, and it is the number everything else here follows from.

**"You chose the fine-tune distribution to make this happen."**
Yes, the narrowing is a configuration value, and ADR-001 says so. What was not chosen
is which slices moved, by how much, or that one class would improve enough to hide the
worst loss. Several of those came out differently from the way I expected. The
repository publishes what was measured, including the parts that undercut the story I
set out to tell.

**"What is the weakest part?"**
The operating point. I define the depth that ships as the shallowest freeze the eval
suite does not object to, which models a team keeping as much plasticity as the eval
allows. It is a reasonable model of the decision and it is still a model. On this
measurement the suite objects at no depth at all, so the rule picks the shallowest
one and the headline is attached to that choice. A team with a different convention
would read a different row of the same table. I would rather have measured a real
team's convention than assumed one.

The second weakest part is that the capability space is mine. Forty slices is not
coverage, it is forty things I thought to name, and every missed-capability count in
here is a lower bound over that list.

**"Why is the training not re-run by `make verify`?"**
Because floating point on a CPU is not bitwise portable, so a figure published to two
decimal places and re-derived on an unknown runner would produce a red badge nobody
can fix, and the usual response is to loosen the check until it stops failing. The
measurement is captured, hashed and committed, the analysis re-runs on every verify,
and a separate job retrains from scratch and publishes the drift instead of pretending
there is none. ADR-004.

**"What would you do differently with more time?"**
Two things. I would measure discordance on a real eval suite against a real fine-tune,
which is the one claim here that genuinely needs a model I do not have. And I would
set the per-slice power budget before training rather than discovering it afterwards,
which is what a team would actually need to act on this.

**"Your checker says every number is verified. How do you know the checker works?"**
Because I broke it and watched it not fail. The anchor patterns dropped the whitespace
between the number and the following word, so every numeric anchor matched nothing and
the checker reported every document clean. I found it by deliberately corrupting a
figure and seeing the build pass. The fix is in `_flexible`, and there is a test that
asserts the broken form fails. It is the best argument in the repository for why a
guard nobody has seen fail should not be trusted.

## Things to say

- "The suite's detection floor is set by how often its cases change their verdict, not
  by how many there are."
- "Retraining noise is wider than the suite's floor here, so a bigger eval set would
  not have helped."
- "The aggregate stayed flat because one class improved while another collapsed."
- "Every figure in the README is re-measured by CI and the build fails if the prose
  drifts from it."

## Things not to say

- Do not say "the model did not regress". That claim is unavailable here and
  unavailable anywhere, because the set of capabilities is not enumerable. Say "no
  regression above the detection floor on the slices we named".
- Do not say "zero false alarms". Say the suite flagged nothing in any of the
  retrainings, over the slices measured, at the stated power.
- Do not say the numbers predict what a production fine-tune would lose. They describe
  two small models on generated data.
- Do not say the repository proves that eval suites are useless. It measures what one
  suite could resolve and names what fell below it. That is an argument for computing
  the floor, not for throwing the suite away.
- Do not quote a wall clock time for anything. The repository publishes counts and
  ratios on purpose, and a sweep over all tracked files fails the build if a duration
  appears.

## Live demo, five commands

```bash
make install                                                  # core install, no deep learning framework
python -m finetune_eval_power.cli suite --arm vision          # the shape of the problem
python -m finetune_eval_power.cli blind --arm vision          # the headline: what it missed
python -m finetune_eval_power.cli power --arm vision --claim 2  # the refusal, exit code 2
make verify                                                   # lint, tests, every figure re-measured
```

Run the third and fourth from a terminal where the exit code is visible: `echo $?`
after the power command is the moment the point lands.
