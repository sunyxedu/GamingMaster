parent: 0000-tier1_astra_naive

# Hypothesis

The leader's remaining weakness is Mastermind decision quality under the three-guess
numeric-reward objective, not TicTacToe strategy or model reliability. Exact belief
tracking plus a two-move terminal-reward search should outperform unaided language
reasoning, while eliminating ordinary model latency. Information gain alone is the
wrong objective for the last guess and may also choose the wrong penultimate probe.

Concrete evidence: `trace_seed15.jsonl` records `[1 2 3 4]` receiving "1 black
peg(s), 2 white peg(s)", followed by `[1 3 2 5]` receiving "0 black peg(s), 2 white
peg(s)". The final `[4 2 3 6]` receives "0 black peg(s), 3 white peg(s)" and scores
0.375. The leader has no candidate-set or expected-reward computation. Seed 13 spends
199.6 seconds on its second move and still finishes at 0.5; all four TicTacToe
episodes draw. The eight-episode mean is 0.515625, with a minimum of 0.375.

The arithmetic narrows the diagnosis: **seed 15's final guess was already optimal**
for its 16 surviving codes, at expected reward 0.59375. Its bad realized score is
not evidence of a bad final decision. The actionable weakness is the earlier probe:
after the first `(1 black, 2 white)` clue, the parent's `[1 3 2 5]` has value
0.616319 with optimal final continuation, while `[1 2 4 5]` has value 0.644097.
Similarly, after seed 13's first `(2,1)` clue, the parent's probe has value
0.744792 and the new probe has value 0.765625. These values integrate all codes
consistent with the public clue, not the realized secret or training seed identity.

# Changes

* Dispatch on the public game ID. Solve TicTacToe with full minimax over the public
  board and learner identity, checking returned board transitions after every move.
* Enumerate all 360 legal distinct-digit Mastermind secrets and filter against the
  complete visible feedback transcript. No seed, hidden code, RNG, or game internals
  is accessed. Assume a uniform prior over codes, as the documented sampling implies.
* Use the canonical `[1 2 3 4]` opening: every legal opening is equivalent under
  digit/position permutations. With two turns left, enumerate every legal probe and
  every possible response, then maximize the exact expected terminal reward in each
  response branch. An immediate solve contributes reward 1 and needs no third action.
* At the final turn, optimize `(overlap + black)/8` across ALL legal guesses,
  including guesses excluded as secrets. Marginal digit/position counts give this
  exact expected utility cheaply. Break reward ties by solve probability, then
  earlier solve opportunity, then fixed lexical order; never tune to training codes.
* Validate every action against current legal actions. After each step check
  `ctx.progress()` rejection/stagnation signals, native move count, consumed action,
  board changes or feedback append/consistency, and Mastermind terminal reward.
  Unexpected behavior disables compiled control for that episode and sends the
  full public evidence plus failure reason to Astra for one-action replanning.
  Replanned effects receive the same checks. Garbage replies and model failures
  fall back to an available, unrejected action. Recovery timeout is 360 seconds,
  above the archive's 217-second worst latency, capped by remaining wall budget.

# Confirmation / refutation

Confirm if Mastermind mean improves while TicTacToe stays at 0.5,
ordinary episodes need zero model calls, and no invalid moves or errors appear.
Compare expected reward of the archived weak episode's final move to the solver's
choice using only its two visible clues. Exhaustively evaluate the abstract policy
on all 360 possible codes to check generality rather than selecting tie breaks on
the tiny training split. Check all TicTacToe opponent branches and both seats.
Inject missing feedback and stalled/rejected actions into public-observation test
doubles to verify Astra recovery and garbage-response handling.

Track weak episodes separately; maximizing expected reward does not guarantee an
improved minimum. Refute or narrow the hypothesis if reward search does not improve the uniform
code average over information-first or candidate-only policies, if the assumed
feedback/reward rules disagree with actual observations, or if parser/recovery
failures outweigh search gains. A stronger all-code mean need not improve every
individual seed; training outcomes are a small sample. Dev results are diagnostic,
not the selection score. No validation or held-out results are inspected.

# Validation

* Required pre-flight passed: score 0.5, four actions, zero model calls, no error.
  Its warning that no model response was exercised is covered by the explicit
  recovery doubles below, although these are not a live Astra recovery evaluation.
* One dev evaluation, `dev/20261031-120826`: seed 210 draws in four moves; seed 211
  finishes at 0.5 in three guesses. Mean 0.5, no errors, invalid actions, model calls,
  or detected rule mismatches. Wall times approximately 0.09 s and 1.20 s. Read both
  traces: all seven post-action verification records report success. The one
  Mastermind sample does not establish a score gain; no changes target that secret.
* `python3 candidates/0001/check_search.py` enumerates the abstract rules over all
  360 possible codes, without loading the game implementation or any seed mapping.
  Reward search averages **0.6649305556**, with 53/360 solves and 2.99444 mean actions.
  An ablation minimizing expected surviving candidates for the second probe, using
  the SAME reward-optimal final policy, averages **0.6548611111**. This is a modest
  +0.0100694 Mastermind gain from reward-aware probing specifically. With perfect
  TicTacToe draws, the uniform-code mixture expectation is 0.5824652778; this is
  an analytic policy result, not a training or held-out evaluation result.
* The exhaustive score distribution is 0.25: 4 codes; 0.375: 32; 0.5: 77;
  0.625: 85; 0.75: 109; 1.0: 53. Thus the worst outcome remains below the parent's
  sampled minimum; no universal floor improvement is claimed.
* Independent direct feedback/reward enumeration checks all legal final moves for
  139 posterior subsets from the policy and ablation, agreeing with marginal-count
  optimization. Full opponent-branch TicTacToe traversal finds no losses in either
  seat (152 terminal paths as player 0, 635 as player 1).
* Sixteen fault/response combinations pass: stalled action, rejected action, raised
  action exception, or missing feedback, crossed with garbage response, legal
  replanned action, TimeoutError, or RuntimeError. Each triggers a System-2 request,
  terminates legally within four attempts, and verifies each returned transition.
  Results are retained in `search_checks.json`; these use model stand-ins only.

No full training evaluation was run, and no validation/held-out results were read.
