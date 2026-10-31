parent: 0006

## Hypothesis

The leader already maximizes uniform expected Mastermind reward, but its
solve-oriented tie breaking leaves avoidable probability of scoring below the
public TicTacToe draw benchmark, 0.5. Optimize that probability only among
exactly mean-optimal policies, then symmetrize the entire contingent policy.
This can improve the probability of an adequate score for EVERY fixed secret
(over independent harness randomness) without sacrificing its expected score.
This is a constrained distributional planning experiment, not a claim to beat
the known uniform-mean optimum.

Evidence: in parent trace_seed13.jsonl the opener [5 3 2 4] receives
"0 black peg(s), 3 white peg(s)". Probe [5 2 3 6] receives
"1 black peg(s), 2 white peg(s)". The final [4 2 3 5] has 18 surviving codes,
logged expected_score 0.5416666666666666, and realizes 0.5. Every verified_step
has issue: null. This is uncertainty, not execution failure. The parent's
all-code audit reports 36/360 outcomes below 0.5 at mean 0.6649305556.
Candidate 0008 reduces weak outcomes to 11/360 but lowers mean to 0.6357639;
0004's lowest-bin-first insurance leaves 33/360 weak outcomes. Neither directly
tests the smallest possible below-draw probability at the EXACT optimal mean
with the leader's per-secret symmetrization guarantee.

## Change from parent

Keep the root private random digit permutation, exact public belief filtering,
TicTacToe negamax, every legal second probe (including impossible secrets), and
all legal final guesses. Rank complete payoff distributions lexicographically
by (total integer reward, negative count below 4/8, solve count). Add histograms
across feedback branches before ranking a probe. Reward is the hard primary
objective, with no tolerance, risk coefficient, fitted threshold, or seed prior.
The second objective is additive, so Bellman optimization is exact. Terminal
marginals identify every mean-optimal action; only those need histogram ranking.
Root relabeling applies to opening and ALL later tie orders, making the whole
score distribution identical for every fixed secret over harness randomness.

The final contingent leaf carries its belief, action, integer payoff and full
histogram; recheck all certificates before execution. Verify ctx.progress(),
rejections, native plies, clue persistence, nonempty belief, legal consumption,
terminal score and the actual three-guess horizon after every action. Missing
plan branches explicitly fail. Any contradiction hands public evidence to
budgeted Astra for one legal action, and every repair action is checked again.
Garbage replies and model timeouts use a bounded legal fallback. No background
work, hidden-state access, model backend access or external dependencies.

## Confirmation and refutation

Exhaust all 360 abstract public codes and all possible second probes. Confirm
if total reward remains 1915 integer eighths while fewer than 36 codes score
below 0.5. Check the optimum against direct independent payoff enumeration,
validate histograms and digit equivariance, then test preflight/native dev and
injected recovery. Refute if no safety gain remains under the exact mean
constraint, if mean falls, or if effect checks reject valid native transitions.
Report full score distribution: the probability guarantee does not raise the
minimum realized reward, and lower-tail metrics can conflict. Four training
Mastermind episodes cannot establish the theoretical gain empirically.

No full training evaluation, validation or held-out inspection will be run.
Validation results will be appended after the complete harness is written.

## Validation results

- Required preflight passed on a four-action TicTacToe draw with no exception
  and zero model calls. The normal-path warning about unexercised model replies
  is addressed by the injected recovery cases below.
- One native dev evaluation (`dev/20261031-130953`) scored 0.5 on both 210 and
  211, with zero model calls/errors and seven clean effect checks. Its digest
  and both step traces were inspected. Mastermind enumerated all 359 probes;
  the terminal belief had 12 codes and predicted histogram 2/4/6 at scores
  0.5/0.625/0.75, so the realized 0.5 is consistent. No second dev evaluation
  was used to select a favorable random map.
- `check_policy.py` / `policy_checks.json` independently check all 129,600
  feedback/payoff pairs, enumerate all 360 abstract codes, verify 210 final
  certificates across three label maps, and check equivariance under a digit
  swap and six-cycle. All 720 digit permutations map every fixed secret to
  each abstract code twice. Thus the entire reported distribution applies to
  every fixed secret over private harness randomness, not just a uniform
  distribution of secrets.
- Exact mean remains **1915/2880 = 0.6649305556**. Below-draw probability falls
  from the parent's **36/360 = 10%** to **33/360 = 9.1667%**. Solves remain
  **53/360**, and mean action count remains **2.9944444**. Full reward counts:
  0.25: **4**; 0.375: **29**; 0.5: **84**; 0.625: **80**; 0.75: **110**;
  1.0: **53**. The minimum realized score remains 0.25.
- `check_bellman.py` / `bellman_checks.json` independently enumerate every
  second probe for all ten opening feedback branches, scoring terminal actions
  by direct per-secret payoff sums rather than the harness's marginal formula.
  They certify the global lexicographic optimum **(1915, -33, 53)**. The
  33-code lower tail is therefore the minimum possible at this exact optimal
  mean under the public three-turn model. Random mixtures cannot improve it
  while preserving that mean: every component must itself be mean-optimal.
- `check_recovery.py` covers all **787** TicTacToe terminal opponent paths
  without defeat and seven failure classes (rejection, no progress, action
  exception, changed feedback, timeout, backend error, garbage answer).
  `check_certificates.py` adds **15** corrupted-plan/reply combinations:
  missing branch, wrong belief, illegal leaf, wrong total or histogram, each
  with a legal answer, garbage answer or TimeoutError. Every case invokes
  Astra through the context interface and completes within the bounded action
  loop. Omitted feedback on a successful solve and incorrect legal-action
  consumption are checked explicitly. These use synthetic public observations
  and stand-in model responses, not real-model recovery measurements.

The narrow hypothesis is confirmed, but the practical improvement is small:
only 0.8333 percentage points in Mastermind below-draw probability. Candidate
0004 already reaches the same 33/360 threshold rate with a different tie
objective; this candidate preserves more solves (53 versus 50) and supplies
0006's per-fixed-secret randomized guarantee. It is not lower-tail dominance:
0004 has only two 0.25 outcomes, while this policy has four. The larger safety
improvement sought by 0008 requires sacrificing expected score. No general
mean improvement or higher realized floor is claimed, and the dev sample does
not demonstrate an empirical score gain. Deadline fallbacks and rule-repair
paths prioritize completion and lie outside the exact-search guarantee.
