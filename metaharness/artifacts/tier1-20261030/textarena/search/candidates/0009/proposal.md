parent: 0004

## Hypothesis

The exact mean-optimal policy still has avoidable variance. The leader 0006
randomizes digit labels and therefore equalizes expected score across hidden
codes, but this does not reduce its single-episode payoff variance. Parent 0004
instead lexicographically suppresses the lowest reward categories; that can
retain unnecessary spread elsewhere in the distribution. A mean-preserving
minimum-variance contingent policy should produce more consistent scores under
the short horizon, at the same optimal uniform-code expectation. This is an
experiment in the full terminal distribution, not a claim to beat the known
Bayes upper bound.

Concrete evidence: leader 0006's worst Mastermind trace, seed 13, opens
`[5 3 2 4]` and receives `0 black peg(s), 3 white peg(s)`, then probes
`[5 2 3 6]` and receives `1 black peg(s), 2 white peg(s)`. It chooses
`[4 2 3 5]` from an 18-code posterior with expected score 0.5416667 and realizes
0.5. Every `verified_step` has `issue: null`: this is residual uncertainty,
not a broken executor. In parent 0004's seed 11, `[1 2 3 4]` and `[1 3 2 5]`
both receive `0 black peg(s), 3 white peg(s)`. Its terminal `[2 1 4 3]` has
reward counts 5/7/5/3 at 0.375/0.5/0.625/0.75, expected reward 0.5375, and
realizes 0.375. Its entire run reports no errors. Pure lower-tail ordering has
not removed posterior variability. Leader 0006's 0.640625 training suite mean
and parent's 0.578125 are only four Mastermind draws each; their proven uniform
means are equal.

## Change from parent

Retain 0004's exact public model, deterministic canonical opening, complete
second-guess search, terminal histogram certificates, TicTacToe minimax, and
checked one-action execution. Replace its Bellman rank `(sum reward, negative
histogram counts from worst to best)` with `(sum reward, -sum squared reward)`.
Use integer eighths for both moments. Every terminal leaf and every second-guess
probe obeys this rank. Add branch histograms before ranking probes: summing
conditional variances would incorrectly omit variance between branch means.
Solves contribute reward 8 and squared reward 64. This preserves optimal mean
first and minimizes variance only on the exact mean-optimal policy face. It
needs no risk coefficient, seed lookup, random label selection, or model call
on ordinary verified play.

The full objective remains additive and lexicographic, so it admits exact
Bellman optimization over the 360-code public universe. Minimizing second moment
at fixed expectation is equivalent to minimizing variance. It is not equivalent
to maximizing worst-case score or suppressing each low-score bin in order, and
can deliberately accept more low outcomes if it reduces total dispersion.

Every action verifies progress/rejections, native move count, consumed legal
actions, board or clue persistence, nonempty beliefs, and the terminal reward.
The verified branch rechecks its belief and full predicted histogram before
execution. A mismatch sends the public evidence, rules, and legal actions to
Astra for replanning. Every repair action is checked again; timeouts, backend
errors, and garbage replies are caught with a bounded legal fallback. Successful
Mastermind termination is handled explicitly because native solve observations
omit ordinary feedback. No hidden state, game implementation source, training
replay, or held-out results are used.

## Confirmation and refutation

Enumerate all 360 abstract secrets under the public rules. Confirm if the
uniform mean remains 1915/2880 while terminal variance decreases relative to
0004 and 0006, and native preflight/dev and injected recovery finish cleanly.
Refute a variance benefit if the parent already reaches the same minimum;
refute correctness on any mean loss, certificate mismatch, illegal action or
unhandled model response. Record the entire score histogram and solve count:
variance reduction need not improve the minimum score or probability of solving.
A small training improvement alone cannot establish this mechanism. Deterministic
labels do not provide the leader's per-fixed-secret randomized guarantee; the
claim is about the uniform public-code distribution. No guarantee is made for
an arbitrary nonuniform secret distribution.

## Validation results

- Final preflight passed: TicTacToe draw, four actions, zero model calls, no
  errors. The warning that ordinary play does not exercise System-2 is addressed
  by the fault-injection checks below.
- One native dev evaluation, `dev/20261031-125638`, scored 0.5 on 210 and 0.75
  on 211 (mean 0.625), with zero model calls and no errors. The digest and both
  traces were inspected: all seven effect checks passed, the second-guess search
  evaluated all 359 legal probes, and the terminal histogram certificate held.
  No second dev run was used; the post-dev change only strengthens checks of
  historical feedback at a successful solve, exercised separately below.
- `check_search.py` / `search_checks.json` compare all 360 abstract secrets
  against parent 0004, check all 129,600 feedback/payoff pairs independently,
  and check every reached final decision against every legal final alternative.
  Both policies have mean **0.6649305556** and minimum **0.25**. Parent variance
  is **0.0326503424**; this policy's variance is **0.0324767313**, a **0.53%**
  reduction. The leader's archived all-code histogram implies variance
  **0.0342996480**, also larger. These are public-rule calculations, not
  native training or held-out scores.
- The full new reward counts are 0.25: **2**, 0.375: **33**, 0.5: **77**,
  0.625: **82**, 0.75: **117**, and 1.0: **49**. Parent 0004 solves 50 cases
  and has 33 outcomes below 0.5; this policy solves 49 and has 35 below 0.5.
  Thus minimum variance does **not** improve every downside metric. Mean action
  count is unchanged at 2.9944444. Only the opening-feedback (1 black, 1 white)
  branch changes its overall histogram; all ten second probes remain the same.
- `check_bellman.py` / `bellman_checks.json` use an independent direct payoff
  enumeration, without the harness's marginal optimization, to verify all ten
  nonterminal opening-feedback branches and all 359 second probes per branch.
  The optimal total reward is **1915 integer eighths** and, subject to that
  maximum, the minimum total squared reward is **10935** (parent: 10939).
  Early solves contribute 8 and 64 explicitly. This verifies the full Bellman
  objective, rather than just the chosen terminal leaves.
- All **787** terminal TicTacToe opponent continuations across both learner
  seats avoid a loss. Sixteen ordinary fault/reply combinations, fifteen
  corrupted contingent-plan combinations, and six direct mismatch cases pass.
  Native-style success without a new feedback line is accepted, while altered
  earlier clues and a solved code excluded by the prior clues are rejected by
  the verifier. These checks use synthetic observations and stand-in model
  responses through the context interface; they do not call model backends.

The narrow hypothesis is confirmed: an exact mean-optimal policy can have less
variance than the parent. A claim of a material reliability improvement is not
supported: the change is small and the lower-tail count actually worsens. The
favorable single dev result is not evidence of higher uniform expectation.
This candidate retains the minimum-variance objective as a controlled experiment
and documents its limitations instead of changing it to fit that result.
