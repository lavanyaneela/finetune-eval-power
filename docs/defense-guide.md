# Defense guide: finetune-eval-power

For the author, before an interview. Every figure here is checked against the
committed measurement by `tools/check_numbers.py`, so if a number in this document
has drifted the build fails.

## The thirty second version

"A passing eval suite gets read as evidence that fine-tuning broke nothing. It is not.
I built a tool that computes how small a regression a suite could actually have
detected, and then measures what fell below that. On my vision arm the suite's
accuracy went up and it never flagged anything, while the referee that enumerates the
whole input space says the same fine-tune made the model worse. They disagree about
the sign."

Then stop.

The question that usually comes next is "isn't that just because you held those cases
out". Partly, and the better part is not: two of the eight missed slices are inside
the region the suite samples, and the suite's own cases on the worst class show a drop
above its own detection floor. It had the signal and averaged it away.

## The claims, and how each is proved

### Claim one: the suite and the referee disagree about the sign

**Command:** `python -m finetune_eval_power.cli blind --arm vision`

The suite **changed by -1.27** points, meaning it improved, and it raised a flag in
**0 of the retrainings**. The probe, enumerating the whole attribute space, says the
same fine-tune cost **probe says 1.75** points.

**If pushed on which one to believe:** the probe, and not because it is bigger. It is
an enumeration rather than a sample, so it has no coverage region to be flattered by.
The suite's region is a choice somebody made, and the fine-tune narrowed onto part of
that same region.

### Claim two: the aggregate hides the signal rather than missing it

**Command:** the per-class table printed by the same command.

Measured on the suite's own cases, one class gets worse by what the **suite shows 4.75**
points while two others **together gain 9.83** points, and the four average to the
published figure. The worst class is **worst is shape=square**. On the probe that same
class **worst lost 10.58** points, and the suite holds **120 cases of its own** for it.

**If pushed on the two different numbers for one class:** they are different
denominators and both are stated. The probe measures the class across the whole space
and sees 10.58 points. The suite measures it on the 120 cases it happens to hold and
sees 4.75, which is still above its own floor. The failure is not that the suite was
blind to the class. It is that the aggregate it reports cannot show a class.

### Claim three: the detection floor is set by discordance, not by case count

**Command:** `python -m finetune_eval_power.cli curve --arm vision`

The suite scores the same cases before and after, so the comparison is paired and all
the information lives in the cases whose verdict changed. The **detection floor of 3.41**
points follows from that. The unpaired calculation most people reach for is the one the
**unpaired says 5.70** about, a factor of about one and two thirds, and seeing a one
point regression **would need 5385** cases.

**If pushed on whether paired versus unpaired really matters:** it is not a small
correction, and it points the wrong way for a team's budget. The hypothesis test itself
is statsmodels. The power arithmetic is mine, and the unpaired path is pinned against
`statsmodels.stats.power` in a test so the piece with a library equivalent cannot drift
from it.

### Claim four: the suite is not the binding constraint anyway

**Command:** `python -m finetune_eval_power.cli power --arm vision --claim 2`, which
exits `2`.

Retraining the same configuration moves the measured regression with a standard
**deviation of 2.002** points, so the **noise floor reaches 3.92** points, which is
wider than the suite's own floor. For any claim smaller than that the tool refuses
rather than recommending a larger suite, and prints how many retrainings would resolve
it instead.

**If pushed on why this deserves its own exit code:** because the two answers send a
team in opposite directions. "Buy more eval cases" and "your eval set is not the
problem" cost very different amounts, and a power calculation can only ever give the
first, since its model of the world contains one noise source.

### Claim five: the blindness is structural, not a property of this fixture

**Command:** `python -m finetune_eval_power.cli suite --arm vision`

Of **40 named capability** slices the suite has **no usable power on 19**: **16 it never samples**
at all, and **3 sampled too thinly** for any effect size to reach the target power. The
second arm reproduces the shape with a different architecture and a different data
type: the **sequence missed 9** of its **29 sequence slices named**.

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
- "The suite and the probe disagree about the sign of what the fine-tune did."
- "The aggregate stayed flat because two classes improved while a third got worse, and
  an aggregate cannot show a class."
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
- Do not say the suite was blind to the worst class. It held 120 cases of it and the
  drop on those cases was above its own floor. The aggregate hid it, which is worse
  and more interesting.
- Do not quote the probe's drop for a class and the suite's case count for that class
  in the same breath without saying they are different denominators. The two numbers
  are both published for exactly that reason.
- Do not say the fine-tune gained anything in region. The gain is barely one flipped
  case out of ninety-six and its spread is larger than itself. Say it bought nothing
  this measurement can resolve.
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
