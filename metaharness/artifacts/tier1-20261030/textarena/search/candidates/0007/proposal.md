parent: 0006

## Hypothesis

The leader's root digit randomization equalizes expected reward before play, but
does not protect individual surviving secrets after feedback. Mixing only among
terminal guesses with exactly maximal posterior mean can improve conditional
worst-secret expected reward without sacrificing the evaluator's mean objective.
This tests posterior hedging on the mean-optimal face, rather than unconstrained
local maximin (which the parent's investigation found sacrificed mean).

Evidence: in `0006/trace_seed13.jsonl`, `[5 3 2 4]` receives `0 black peg(s),
3 white peg(s)`. The next probe is `[5 2 3 6]`. The final `mixed_search` record
reports 18 candidates, expected score 0.5416666666666666 and solve probability
zero for `[4 2 3 5]`; its realized score is 0.5. All effect checks pass. This is
residual uncertainty rather than a parser, legality, or search error. The archive
establishes that the canonical expected-score policy already attains 0.66493056
over all 360 public abstract codes, so a general mean improvement is not claimed.

## Change from parent

Keep exact TicTacToe, root digit symmetrization, the second-probe search and its
tie ordering, public belief tracking, and action-by-action checks with Astra
replanning. At the last guess, enumerate all actions whose exact integer reward
sum equals the optimum. Solve the finite zero-sum game against the surviving
secrets restricted to those actions, using standard-library simplex. Sample one
action from the resulting distribution. Check probability validity, exact mean
preservation, and primal/dual minimax certificates before execution. The local
mix is private randomness independent of the game and evaluator seeds.

Each guess still executes separately and verifies ctx.progress(), the clue
history, legal-action consumption, native move count, and terminal arithmetic.
Unexpected evidence hands control to budgeted Astra; arbitrary text, timeouts,
and failed calls receive a legal fallback and the next effect is checked again.

## Confirmation / refutation

Compare the complete canonical policy against the parent on all 360 abstract
codes using exact reward arithmetic and mixture expectations, not sampled game
seeds. Confirm if posterior worst-secret expected payoffs improve on some reached
branches, never lose to the parent's pure action, and preserve global mean.
Refute a broad benefit if useful branches have no alternative optimal actions,
if conditional protection harms the realized lower tail materially, or if native
effect checks fail. Report changes in solves and realized reward distribution:
an expected floor over random choices is not a guaranteed score on one episode.
The four training Mastermind episodes are too small to establish a mean gain.

Required preflight, synthetic failure injection, and at most two native dev
evaluations will be recorded below. No full training evaluation or held-out
inspection will be run.

## Validation

- `check_policy.py` exhaustively checked all 360 public codes. Mean reward is
  **0.6649305556**, exactly equal to 0006. The terminal mixture has minimum
  expected reward 0.3125 versus 0006's 0.25 and lowers public-rule reward
  variance from 0.03430 to 0.03287. It mixes in 29 of 70 final feedback
  branches and improves the parent's conditional floor in 30 branches. These
  are expectations over the local random draw; realized rewards remain discrete.
- The same audit checked the simplex on small games, probability normalization,
  primal/dual certificates, digit relabeling equivariance, exact native reward
  arithmetic, and preservation of the parent branch mean. It did not inspect a
  secret or evaluator seed.
- `check_recovery.py` covered rejection, no progress, action exceptions,
  changed feedback, timeout, backend error, and garbage model replies. Every
  case called Astra, suppressed rejected actions, and terminated legally.
- Required preflight passed with no exception. One native dev evaluation
  (`dev/20261031-125023`) scored 0.5 on both seeds, with three Mastermind
  actions, zero model calls, zero rejected actions, and no errors. This small
  sample neither confirms nor refutes a training improvement.
