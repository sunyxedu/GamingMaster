parent: 0011

## Hypothesis

A single worst-case floor leaves the frequency of near-worst outcomes uncontrolled.
Parent 0011 guarantees 0.375, then maximizes mean, but its public-code audit still
places 34/360 secrets at that floor. Leader 0006 instead maximizes mean under root
randomization; its weak seed-13 trace opens `[5 3 2 4]` with `0 black peg(s), 3 white
peg(s)`, probes `[5 2 3 6]` with `(1,2)`, then ends at 0.5 with `[4 2 3 5]`.
Every `verified_step` has `issue: null`. This is residual decision risk, not a
formatting or execution failure. The leader's all-code distribution also has four
0.25 outcomes. Exact uniform mean is already saturated, so this proposal tests a
different robustness objective rather than claiming a new mean optimum.

Compile the policy that lexicographically maximizes its sorted vector of rewards
across all 360 public abstract secrets: maximize the minimum, then the second
smallest, and so on. Equivalently minimize the reward histogram from low to high.
Unlike a single floor or fixed score threshold, this gives every lower-tail level
priority without a tunable risk coefficient. It may materially reduce floor mass
but sacrifice too much mean; that is the hypothesis to measure, not conceal.

## Changes from parent

Replace 0011's floor-indexed reward frontiers and shared root floor with additive
histogram-valued Bellman search. Each terminal action is ranked by negative counts
at rewards 0/8 through 7/8; each probe adds the selected child histograms and its
immediate solve. Fixed posterior size determines the remaining 1.0 mass. Compare
these vectors lexicographically, then fewer actions, then canonical action order.
Histograms add across disjoint feedback branches and lexicographic ordering is
translation invariant, so independently optimal children yield the globally
optimal distribution. There is no separately chosen floor or mean-first
constraint, and no seed or learned secret enters the search.

Retain the parent's exact 48-element opening stabilizer to reduce 359 second
probes to 16 orbit representatives; all terminal legal guesses remain available,
including guesses excluded as possible secrets. Preserve the canonical opener,
TicTacToe minimax, belief and payoff certificates, and one-action execution.
Check ctx.progress(), rejected actions, consumed legality, native move counts,
clue/board persistence and terminal reward after every step. Discrepancies disable
compiled control and send the evidence and public state to Astra. Repaired actions
receive the same verification; garbage replies and caught model failures fall
back to an unrejected legal action. No normal-path model calls or background work.

## Confirmation and refutation

Independently enumerate every second probe in each opening branch, then replay all
360 public abstract secrets. Confirm the structural claim if this matches the
compiled distribution, preserves the maximal 0.375 floor, and reduces its mass
below parent 0011's 34 outcomes. Record the complete histogram and exact mean to
measure the cost. Refute practical value if mean loss outweighs reliability gains
under the evaluator's mean-score objective; neither a better minimum nor a lucky
dev result establishes a higher expected score. Test all ordinary execution and
recovery paths, run required preflight, and at most two native dev evaluations.
No training evaluation or validation/held-out results will be accessed.

Harness and proposal were written before validation. Results follow when available.

## Results

The mechanism is confirmed but its expected-score justification is refuted.
Complete leximin optimization keeps the **0.375** floor and reduces its frequency
from parent 0011's **34/360 to 11/360**. Its exact reward counts are:

| Score | Parent 0011 | Candidate 0014 |
|---|---:|---:|
| 0.375 | 34 | 11 |
| 0.5 | 81 | 137 |
| 0.625 | 86 | 100 |
| 0.75 | 108 | 91 |
| 1.0 | 51 | 21 |

Mean falls from **1912/2880 = 0.6638888889** to
**1795/2880 = 0.6232638889**, a loss of **0.040625** Mastermind reward.
Relative to the unconstrained Bayes optimum, the loss is exactly **1/24**.
The equally weighted theoretical suite mean is **0.5616319444**. This is an
abstract public-rule calculation, not a measured training or unseen-split score.
It also is not stochastic dominance: mass at 0.5 grows substantially and solves
fall from 51 to 21. The policy takes 1 guess on one code, 2 on two codes and 3
on 357 codes, for 2.9888889 mean guesses; the tiny action saving cannot compensate
for score loss under the evaluator's ordering.

Compared with 0008's archived threshold policy, both leave 11 codes below 0.5.
Full leximin further reduces the count AT 0.5 from 147 to 137, but reduces mean
from 0.6357639 to 0.6232639 and solves from 46 to 21. Giving successive lower-tail
levels absolute priority buys additional safety at a disproportionate upper-tail
cost. This is useful negative evidence against unrestricted leximin for a
mean-score evaluator; it should not be represented as a stronger expected policy.

Validation artifacts:

- `check_policy.py` / `policy_checks.json`: an independent implementation ranks
  sorted individual payoff vectors instead of histograms, enumerates all **3,590**
  opening-branch/second-probe pairs without symmetry pruning, and matches every
  selected probe and payoff distribution. All **129,600** independent feedback
  pairs match. Compilation took about **0.94 seconds** in this local audit.
- All **360** synthetic controller replays finish with correct numeric rewards,
  no recovery calls and every effect check passing, including solves that omit
  ordinary feedback. Exact TicTacToe traversal covers **787** terminal opponent
  paths across both seats without a learner loss.
- `check_failures.py` / `failure_checks.json`: **16** ordinary fault/reply
  combinations, **15** corrupted-plan combinations, **six** direct effect
  mismatches and omitted successful-solve feedback pass. Rejections, stalls,
  action exceptions, changed clue format, corrupted beliefs/probes/leaves and
  payoff certificates trigger Astra handoff. Replies include legal moves,
  garbage, TimeoutError and RuntimeError. These use stand-in models.
- Required preflight passes with a TicTacToe draw, four actions, zero calls and
  no error. Its warning about an unexercised model path is addressed by the
  synthetic recovery checks above.
- One native dev run, `dev/20261031-132037`, scores **0.5 on 210 and 0.625 on
  211**, mean **0.5625**, with zero calls, rejected moves or errors. Digest and
  both step traces were inspected: all seven `plan_verified` events report
  `ok: true`. Cold native Mastermind runtime is **3.19 seconds**. No second dev
  run or full evaluation was performed, and no unseen-split results were read.

The candidate intentionally remains this one tested hypothesis. The favorable
dev comparison with the parent is insufficient to reverse the exact mean loss.
Its guarantees apply only when the public rules match and certified compilation
completes; Astra recovery and deadline fallback are checked for legality and
progress but do not retain the leximin certificate.
