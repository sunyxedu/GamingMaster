parent: 0011

## Hypothesis

A single worst-case floor hides how often the policy barely meets that floor.
Parent 0011 removes every 0.25 outcome but leaves 34 of the 360 public abstract
codes at 0.375. Its mean-first continuation has no incentive to improve those
weak outcomes if that costs aggregate reward. Test a different architecture:
lexicographically maximize the entire sorted vector of terminal scores. First
eliminate the worst achievable score, then minimize its frequency, then minimize
the frequency of the next score, and continue through the reward distribution.
This is complete distributional maximin (leximin), without a selected risk
threshold or a fitted risk coefficient.

The leader's weak trace is an objective limitation, not an execution error.
In candidate 0006's trace_seed13.jsonl, `[5 3 2 4]` receives `0 black peg(s),
3 white peg(s)`, `[5 2 3 6]` receives `(1,2)`, and final `[4 2 3 5]` realizes
0.5. The final 18-code belief has logged expected_score 0.5416666666666666;
all three verified_step records have issue null. Its proposal establishes a
0.25 tail under exact mean optimization. Candidate 0011's root guarantee fixes
that extreme tail, but its distribution leaves substantial near-floor mass.

## Change from parent

Retain parent 0011's full public-information decision-tree compilation, exact
48-element opening stabilizer reduction to 16 second-probe orbits, public-board
TicTacToe minimax, belief certificates, and one-action verified controller.
Replace the floor-versus-mean frontier with a reward histogram ordered by the
negative count at each score from lowest to highest. Add those histogram vectors
across feedback branches; lexicographic order is translation invariant, so local
Bellman choices yield the exact whole-tree leximin optimum. Terminal search
includes all unused legal guesses, including eliminated secret hypotheses.
Action count breaks identical histogram ties, then canonical code order.
The opening is canonical by public label/position symmetry. No benchmark seed,
engine source, hidden code, sampled label map or learned secret prior is used.

This differs from 0008's single chance constraint and 0011's single floor
followed by mean. It also differs from 0004's lower-tail tie-breaking inside the
mean-optimal set: here lower-tail protection has priority over mean. The resulting
mean may decrease. The experiment asks whether protecting all successive lower
tails yields a useful, general risk profile, not whether it can exceed the
established Bayes-optimal mean of 0.66493056.

After every action the controller checks ctx.progress(), rejection counters,
native move advancement, legal-action consumption, persistent public clues or
board marks, belief consistency, score and terminal horizon. Mismatches abandon
compiled control and send the full evidence to budgeted Astra for a replacement
move. Replacement moves receive the same verification. Model exceptions are
caught; garbage replies fall back to an unrejected legal action. Search checks
the deadline between probes and runs no background workers.

## Confirmation and refutation

Exhaust all 360 abstract codes and compare the exact sorted outcome vector with
0011, 0008 and mean-optimal search. Independently enumerate all 359 second probes
in each opening-feedback branch to verify that symmetry reduction preserves the
leximin optimum. Confirm if the weakest score stays at least 0.375 and the sorted
payoff vector strictly improves over 0011; report the complete histogram and mean
cost. A worse tail, a mismatch in enumeration, or false ordinary-play recovery
refutes the implementation. A worse training mean remains possible even with a
correct certificate: the evaluator averages numeric rewards, not sorted tails.

Run required preflight, one native dev evaluation, exhaustive public-rule
controller replays, and injected faults/garbage/timeout recovery tests. Native
scores and synthetic distribution guarantees will be reported separately. No
full evaluation or held-out results are used.

## Results

The distributional hypothesis is confirmed, while the cost under the evaluator's
mean objective is substantial. The exact leximin outcome histogram over all 360
public abstract codes is:

| Native score | 0011 parent | 0008 threshold policy | 0015 leximin |
|---|---:|---:|---:|
| 0.375 | 34 | 11 | 11 |
| 0.5 | 81 | 147 | 137 |
| 0.625 | 86 | 94 | 100 |
| 0.75 | 108 | 62 | 91 |
| 1.0 | 51 | 46 | 21 |

All three policies eliminate 0.25 outcomes. The new policy's sorted vector
strictly improves on both comparators: it reduces the parent's weakest-score
mass, and ties 0008 at that score while removing ten 0.5 outcomes. This is
lexicographic improvement, not stochastic dominance or per-secret improvement.
The mean is **1795/2880 = 0.6232638889**, versus parent **1912/2880 =
0.6638888889**, threshold policy **1831/2880 = 0.6357638889**, and exact
mean optimum **1915/2880 = 0.6649305556**. Protecting the successive lower tails
costs 117 reward-eighths against the parent and 36 against threshold protection.
Solve count drops to 21/360. This outcome warns against treating increasingly
strict lexicographic safety as a free or mean-improving refinement.

A concrete distinction appears after opening feedback `(0 black, 4 white)`.
The new second probe `[2 3 4 1]` produces a complete continuation distribution
of one 0.625, seven 0.75 and one solve across nine hypotheses. The parent's
`[2 3 4 5]` produces three 0.625, two 0.75 and four solves. The new objective
trades three solves for fewer weak outcomes even in this relatively easy branch.
This follows from the global distributional objective and does not use a
benchmark secret or observed training reward.

Validation:

- Required preflight passed: TicTacToe draw, four actions, zero model calls and
  no error. Its unexercised-model warning is covered by synthetic recovery tests.
- `check_policy.py` independently builds all **129,600** feedback/reward pairs
  and enumerates all **3,590** branch/probe combinations without symmetry pruning.
  It compares sorted outcome lists, rather than the harness's histogram rank,
  and exactly matches every selected probe, histogram and action tie-break.
  Cold compilation took approximately **0.82 seconds** in the concurrent checks.
- All **360 controller replays** finish with the certified reward histogram,
  no false recovery, no rejected actions, and clean effect checks. Synthetic
  solves omit the final ordinary clue, matching the native behavior. One code
  solves on guess one, two on guess two, and 357 use three guesses: mean actions
  **2.9888888889**. Independent TicTacToe traversal covers **787** terminal
  opponent paths across both seats without a loss.
- `check_failures.py` passes **16** ordinary fault/reply combinations, **15**
  corrupted-plan/reply combinations, and **six** effect mismatches plus omitted
  successful-solve feedback. Tests cover stalls, rejected/raised actions, changed
  clue format, missing branches, wrong beliefs/probes, illegal leaves, changed
  histograms, garbage answers, TimeoutError and RuntimeError. These are stand-in
  model checks; no real Astra recovery latency is claimed.
- One native dev evaluation, `dev/20261031-132030`, scored **0.5 on 210** and
  **0.625 on 211**, mean **0.5625**, with zero model calls or errors. Its digest
  and both step traces were inspected: all seven plan_verified events are clean.
  Mastermind followed `[1 2 3 4]`, `[1 2 4 5]`, `[1 2 3 5]`, receiving `(1,1)`,
  `(1,1)`, `(2,1)`. The single dev gain over the parent's 0.5 does not overturn
  the exhaustive evidence of lower mean. No second dev run or full evaluation
  was used, and no validation or held-out result was accessed.

The guarantees apply to the certified public rules and complete search. Recovery
and deadline fallbacks prioritize legal completion and carry no leximin or 0.375
floor claim. This candidate is submitted as a distinct distributional experiment;
future work should retain the negative result about mean cost rather than infer
a broadly stronger policy from the favorable dev episode.
