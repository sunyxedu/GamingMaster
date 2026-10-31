parent: 0013

## Hypothesis

Uniform sampling among equally optimal Mastermind probes is not necessarily the
best way to protect each secret remaining after the first clue. Optimize the
probe probabilities as a finite zero-sum game, where a probe's payoff for each
secret is its expected terminal reward under the parent's unchanged uniform
final-action policy. Restrict support to exactly reward-and-solve-optimal probes.
This should weakly improve the conditional worst-secret expectation without any
loss of the exact uniform-prior mean or solve probability. Strict improvement is
a testable hypothesis, not an assumption.

The leader's `0006/trace_seed13.jsonl` records `[5 3 2 4]` receiving "0 black
peg(s), 3 white peg(s)", then `[5 2 3 6]` receiving "1 black peg(s), 2 white
peg(s)". Its 18-secret final posterior logs expected_score 0.5416666666666666
and realizes 0.5; every `verified_step` has `issue: null`. This is residual
uncertainty, not an execution fault. Parent 0013 already expands the optimal
probe support to 24–120 actions and improves conditional protection by uniform
mixing. Its exhaustive mean remains 1915/2880. Candidate 0007 optimizes terminal
mixtures, but does not optimize the probabilities of these complete contingent
second probes. This experiment targets that earlier decision only.

## Changes from parent

Keep exact public-code enumeration, all legal second guesses, numeric-reward
optimization, solve tie-breaking, uniform final-action mixing, TicTacToe minimax,
one-step execution and evidence-triggered Astra recovery. Replace uniform probe
selection with a certified minimax distribution over the same complete probe
policies. Reuse the archive's bounded standard-library simplex implementation;
verify primal/dual agreement and that every row preserves the exact mean.
Compare its guarantee with the parent uniform distribution before sampling.

Use one uniform private six-digit relabeling for the entire policy, including
simplex row and column ordering. This makes tie-breaking symmetry consistent
and transfers the all-code distribution to every fixed secret over harness
randomness. It uses no evaluator seed, inferred secret generator, or archived
secret. The final action remains a uniform sample from the parent's exact
optimal set, whose belief and reward certificate are recomputed before use.

Every action still checks ctx.progress(), rejected actions, native move count,
feedback persistence, posterior consistency, consumed legal action and score.
A missing branch, invalid certificate or search interruption hands the evidence
to budgeted Astra for a legal one-step repair, then checks the repair's effect.
Timeouts and garbage model replies have bounded legal fallbacks. No extra model
call is added on the verified normal path.

## Confirmation and refutation

Enumerate the canonical opening's ten non-solving public-feedback branches.
Compare the optimized and uniform minimum expected reward over every consistent
secret, and independently verify mean/solve preservation, probability sums,
feasibility and primal/dual bounds. Strictly higher conditional floors support
the hypothesis; identical floors refute the claimed opportunity. Do not claim a
higher Bayes mean, a higher realized reward floor, or general native score gains
from a small dev sample. Analyze the full resulting distribution because better
conditional expectations need not reduce realized tail risk.

Run required preflight, at most two native dev evaluations, and synthetic
fault-injection checks. No full training evaluation or validation/held-out
inspection.

## Results and interpretation

The strict-improvement hypothesis is **refuted**. Uniform probe mixing already
attains the conditional minimax value in all ten non-solving opening branches.
This is established with an exact independent dual certificate: for each branch,
put uniform adversarial mass on the secrets with the lowest payoff under the
parent's uniform strategy. Every optimal probe's expectation against that
distribution equals the parent's minimum, so no reweighting can improve it.
For example, the (0 black, 3 white) branch has minimax value **75/128 =
0.5859375**, exactly the parent's value. The (0,2) value is **19/32 = 0.59375**.

The new solver produces alternative, often sparser optimal mixtures, but they
do not improve the full reward distribution either. Independent enumeration
finds every eligible probe has the same aggregate reward histogram within its
opening-feedback branch when followed by uniform optimal final guesses.
Consequently **any** distribution over those probes has the same all-code
histogram as the parent. Expected counts among 360 codes are 10/3 at 0.25,
191/6 at 0.375, 238/3 at 0.5, 167/2 at 0.625, 109 at 0.75, and 53 at 1.
Mean is **383/576 = 0.6649305556**, below-0.5 probability **211/2160**, and
solve probability **53/360**. The realized floor remains 0.25. Root relabeling
transfers this distribution to every fixed secret over independent harness
randomness; it cannot raise the mean.

Completed validation:

- `check_policy.py` checks all 129,600 feedback/reward pairs against an
  independent Counter-based rule calculation, all ten opening branches, and
  **2,928** terminal action-set certificates against direct payoff summation.
  Exact rational arithmetic certifies the parent uniform strategy's optimality
  and the unchanged histograms. Floating simplex primal/dual gaps stay below
  6e-13 reward eighths in this audit, within the runtime 1e-7 tolerance.
- `check_symmetry.py` compares complete probe probabilities and **5,856**
  terminal branches under a digit swap and six-cycle. Probabilities agree
  exactly after the same permutation is applied to row and column ordering.
  It also checks the actual constructor and opening, not just transformed
  solver fixtures.
- `check_recovery.py` covers all **787** terminal TicTacToe opponent paths with
  no losses and seven injected recovery classes: rejected action, no progress,
  action exception, changed feedback format, model timeout, backend error and
  garbage reply. Every failure invokes the context model interface and finishes
  within four actions. `check_certificates.py` adds **15** combinations of
  damaged belief/action set/reward/solve count/missing branch with legal,
  malformed or timed-out model replies. All complete without invalid actions.
- Required preflight passed with a four-action TicTacToe draw, zero model
  calls and no error. The usual unexercised-model warning is covered by the
  synthetic recovery tests, which use stand-ins rather than model backends.
- Two permitted native dev runs were used. `dev/20261031-133510` scored
  0.5/0.5. Its trace exposed an inherited assignment overwriting the sampled
  digit order, so it did not execute the intended root symmetry. Removing that
  assignment fixed the issue, and the constructor/symmetry regression check
  now catches it. `dev/20261031-134156` scored **0.5/0.75**, mean **0.625**,
  with zero calls/errors/rejections and seven clean effect checks. Mastermind
  took about **0.88 seconds**. Both digests and all step traces were inspected.
  The second run was for the concrete initialization repair, not to select a
  favorable random map. No further native evaluation was run.

Retain this candidate as a controlled negative experiment. Reweighting probes
alone adds computation without improving this parent's guarantees or reward
distribution. A future search for greater conditional robustness would have to
change the continuation policies as well, and would need a new certificate;
the present result does not establish that such an improvement exists. The
favorable final dev score is sampling variation, not evidence against the exact
negative result. Search interruptions and Astra repairs prioritize completion
and lie outside the certified normal-path policy.
