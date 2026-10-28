# Defense guide: finetune-eval-power

For the author, before an interview. Every figure here is checked against the
committed measurement by `tools/check_numbers.py`, so if a number in this document
has drifted the build fails.

## The thirty second version

"A passing eval suite gets read as evidence that fine-tuning broke nothing. It is not.
I built a tool that computes how small a regression a suite could actually have
detected, and then measures what fell below that. On my vision arm the suite's
accuracy went up and it raised no regression flag, while the referee that enumerates
the whole input space says the same fine-tune made the model worse. They disagree
about the sign."

Then stop.

The question that usually comes next is "isn't that just because you held those cases
out". Partly. Six of the eight missed slices are regions the suite never samples. The
part that is not about holding out is the worst class, which the suite does sample
with a hundred and twenty of its own cases and still cannot resolve, because at that
case count its floor is nearly twice the drop.

## The claims, and how each is proved

### Claim one: the suite and the referee disagree about the sign

**Command:** `python -m finetune_eval_power.cli blind --arm vision`

The suite's **measured drop of -1.27** points is negative, so its accuracy rose. It
raised a regression flag in **0 of the retrainings** and returned a **significant verdict in 2**
of them, both improvements. The probe, enumerating the whole attribute
space, says the same fine-tune cost what the **probe says 1.75** points.

**If pushed on which one to believe:** the probe, and not because it is bigger. It is
an enumeration rather than a sample, so it has no coverage region to be flattered by.
The suite's region is a choice somebody made, and the fine-tune narrowed onto part of
that same region.

**If pushed on the minus sign:** the quantity is a drop, and a drop is positive when
the model gets worse, so a negative drop is an improvement. The tool prints the word
next to the number for exactly this reason.

### Claim two: the aggregate cannot show a class, and neither can the suite

**Command:** the per-class table printed by the same command.

Measured on the suite's own cases, one class gets worse by what the **suite shows 4.75**
points while **2 improve and** the fourth holds, **together gain 9.83** points
**across 4 classes** that average to the published aggregate. So the aggregate hides
the movement.

It is worse than that, and this is the honest version. The class is **worst is shape=square**,
and the **suite holds 120** cases of it. At that case count and that rate of changed
verdicts, **its own floor is 8.74** points, and its retraining interval
**interval starts at -1.09**, which means it includes zero. The suite could not have
resolved this class even if it had reported per class. The referee resolves it because
it holds **432 referee cases** of the class and sees **worst lost 10.58** points there.
The suite's side agrees: at its case count for that class, and at the rate of changed
verdicts the probe measures there, the suite's **own slice floor is 11.60** points, which
is above the drop.

**If pushed on the two different numbers for one class:** they are different
denominators and both are published, with the floor that applies at each. The useful
conclusion is not that the suite averaged away a signal it had. It is that a suite this
size cannot resolve a per-class regression of this size at all, so reporting per class
would not have saved it either. That is an argument about suite design, not about
aggregation.

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
**deviation of 2.002** points. That standard deviation is not itself wider than the
floor, and saying so would be the easy mistake. At 1.96 of them the
**noise floor reaches 3.92** points, and that is the quantity wider than the suite's
own floor. For any claim smaller than it the tool refuses rather than recommending a
larger suite, and prints how many retrainings would resolve it instead.

**If pushed on why this deserves its own exit code:** because the two answers send a
team in opposite directions. "Buy more eval cases" and "your eval set is not the
problem" cost very different amounts, and a power calculation can only ever give the
first, since its model of the world contains one noise source.

### Claim five: the blindness is structural, not a property of this fixture

**Command:** `python -m finetune_eval_power.cli suite --arm vision`

Of **40 named capability** slices the suite has **no usable power on 19**: **16 it never samples**
at all, and **3 sampled too thinly**, meaning their cases almost never change verdict
rather than that there are few of them, and **all 3 name a size**: the
**cheapest needs 140** cases of its own slice and the **dearest needs 459**.
**8 slices regressed** unflagged, covering **3 regions once nesting** is removed, and
**5 slices moved up** with intervals clear of zero, which is **2 regions the other**
way under the same correction. The second arm reproduces the shape with a different architecture and a
different data type: the **sequence missed 9** of its **29 sequence slices named**.

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
- "And a suite this size could not have resolved that class even per class, which is
  the stronger version of the same point."
- "Every figure in the README is re-measured by CI and the build fails if the prose
  drifts from it."

## Things not to say

- Do not say "the model did not regress". That claim is unavailable here and
  unavailable anywhere, because the set of capabilities is not enumerable. Say "no
  regression above the detection floor on the slices we named".
- Do not say the suite flagged nothing in any retraining. It raised no regression
  flag, and it did return a significant verdict twice, in the improvement direction.
- Do not say the numbers predict what a production fine-tune would lose. They describe
  two small models on generated data.
- Do not say the repository proves that eval suites are useless. It measures what one
  suite could resolve and names what fell below it. That is an argument for computing
  the floor, not for throwing the suite away.
- Do not say the suite had the signal and averaged it away. On its own cases the worst
  class is below its own per-class floor with an interval spanning zero. Say the suite
  could not have resolved that class at this case count, which is a sharper criticism.
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
