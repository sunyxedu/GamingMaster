parent: 0002

## Hypothesis

The robust parent's conservative partial-credit hedges suppress exact solves, while
its floor guarantee still permits poor realized outcomes. Test the opposite policy:
maximize the probability of identifying the code within the three actual guesses,
then maximize expected native partial credit among equally solve-capable policies.
This is deliberately different from the leader's expected-reward objective. It is
not a claim that solve rate is the evaluator's objective: a higher solve rate may
cost mean score, and the experiment must quantify that tradeoff.

Concrete archive evidence: candidate 0002 seed 11 uses `[1 2 3 4]` and `[1 3 2 5]`,
each receiving `0 black peg(s), 3 white peg(s)`. Its `local_plan` then reports 20
candidates, floor 0.375, and mean 0.5375, before `[2 1 4 3]` receives `(0,3)` and
scores 0.375. Candidate 0001 has exactly the same first two clues, selects
`[2 4 5 3]`, and scores 0.5, with a reported final solve count of 1. Its second
probe has only 6/88 solve probability despite expected score 0.61079545. All effect
checks succeed; no model error or rejection explains these outcomes. Seed 15 of
the leader finishes at 0.5 despite successful verification; its final search
explicitly reports `solves: 0` for `[2 5 3 4]` among 15 surviving candidates. These observations
motivate testing how many hypotheses the *probe* can make exactly solvable, rather
than interpreting one unlucky final outcome as a reasoning bug.

## Changes from parent

Retain the parent's public-state minimax, exact clue parser and 360-code belief,
legality checking, and action-by-action verification with Astra recovery. Replace
maximin scoring with a full solve-first contingent tree over every legal probe and
feedback branch. Each unresolved branch has one final guess: maximize solve count
first (one feasible code), then its expected numeric reward. Sum these solve counts
across branches and choose the probe with the most solvable codes, then the highest
native-score sum, then earlier solve opportunity and lexical order. The final guess
can still sacrifice partial-credit expectation to remain a possible secret. All
first guesses are equivalent under digit/position symmetry, so keep `[1 2 3 4]`.
No training identities, hidden secrets, RNG reconstruction or tuned digit maps.

The controller commits one action, checks `ctx.progress()`, native move count,
legality consumption, public-board effects, feedback consistency, horizon, and
terminal arithmetic, and gives discrepancies to budgeted Astra for replanning.
It stays under Astra control after a failure and verifies revised actions too.
Model exceptions and garbage answers fall back to a currently legal, unrejected
move; 360-second timeouts exceed the archive's observed maximum. No background work.

## Confirmation and refutation

Exhaustively enumerate the public abstract code space to measure solve rate, mean
score, floor, and score distribution. Confirm the mechanism if solve rate rises
against both 0002 and 0001; refute its benefit for selection if mean native score
loses materially despite more solves. Report native dev separately from synthetic
results. Verify both TicTacToe seats against all opponent continuations and inject
stalls, rejections, parser changes, garbage replies and timeouts into public-only
stand-ins to exercise recovery. Run required preflight and at most two dev runs.
No full training evaluation or validation/held-out inspection.

## Validation

The implementation and hypothesis were written before testing. No digit mapping,
probe, final action, or ranking was selected from dev/training outcomes.

- Required preflight passed: TicTacToe draw, four actions, zero model calls,
  no error. The ordinary path does not exercise model responses; fault injection
  below covers that path with stand-in responses.
- `check_policy.py` exhaustively evaluates all 360 abstract secrets: **78 solves
  (21.667%), mean 0.64340278, minimum 0.25**. Distribution: 17 at 0.25, 43 at
  0.375, 90 at 0.5, 86 at 0.625, 46 at 0.75, 78 at 1. One code solves on the
  opening, four on the second guess, and 355 cases need the third guess.
- The archived parent has 30/360 solves and mean 0.63715278; the leader has
  53/360 and mean 0.66493056. Thus this hypothesis raises solve rate and slightly
  beats the parent in uniform mean, but **loses 0.02152778 Mastermind expected
  reward to the leader**. The stronger claim that solving more improves the
  evaluator's mean is refuted under the uniform public model. This is useful
  objective-tradeoff evidence, not a universal strategy improvement.
- For the leader's weak `(0,3)` opening branch, this policy probes `[2 3 4 5]`
  and can solve 11/88 codes, versus the leader's 6/88. Expected score falls from
  0.61079545 to 0.59943182. More solvable branches are not synonymous with better
  numeric reward. A separate count of nonempty feedback cells establishes the
  solve-count bound without using the terminal-action ranking.
- All 129,600 feedback/reward table entries pass independent counting checks.
  Both TicTacToe seats pass exhaustive opponent-continuation traversal over 684
  reachable states without a learner loss. These are synthetic public-rule tests,
  not benchmark-seed evaluations; results are in `policy_check.json`.
- `check_recovery.py` passes 16 fault/reply combinations: missing feedback,
  rejected action, raised action exception, and stagnation crossed with a legal
  reply, garbage text, TimeoutError, and RuntimeError. Every case calls Astra's
  stand-in interface and completes legally, suppressing rejected actions.
  Reward-model mismatch detection passes. See `recovery_check.json`.
- One permitted native dev run, `dev/20261031-122706`, scores **0.5 and 0.75**,
  mean 0.625, with zero System-2 calls, invalid moves, or errors. Both traces and
  the digest were read. All seven effect checks are clean; the final Mastermind
  move explicitly verifies its contingent branch. This single Mastermind sample
  does not outweigh the exhaustive mean-score loss above.

The policy remains solve-first to preserve this experiment's distinct hypothesis.
No full evaluation was run and no validation or held-out results were inspected.
