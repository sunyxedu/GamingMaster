parent: none

## Hypothesis

The baseline's weak Mastermind outcomes reflect an unprotected downside in its final guess, not insufficient language reasoning. A policy that explicitly maximizes its worst possible terminal partial-credit score, then its mean score subject to that guarantee, should raise weak outcomes across unseen secrets. This is a fresh robust-planning experiment, distinct from maximizing information gain, solve probability, or unconstrained expected reward. Its risk is deliberate: sacrificing expectation for a floor may hurt the leaderboard's mean objective.

Concrete archive evidence: in baseline seed 15, `[1 2 3 4]` received `1 black peg(s), 2 white peg(s)` and `[1 3 2 5]` received `0 black peg(s), 2 white peg(s)`. Astra then chose `[4 2 3 6]`; the trace records `0 black peg(s), 3 white peg(s)` and terminal score **0.375**. All three calls succeeded; the problem is action quality, not an exception. Seed 13 spent 199.6 seconds choosing its second guess and 129.6 seconds choosing its third, yet scored 0.5. All four TicTacToe episodes already drew, so the remaining score opportunity lies in Mastermind.

## Change

Replace the generic per-turn model loop with a public-state solver and an evidence-driven recovery controller. TicTacToe uses exact negamax. Mastermind enumerates the 360 distinct-digit codes, retains all observed feedback constraints, and searches every legal second guess and every possible feedback branch. The final-guess search includes legal codes inconsistent with earlier clues: partial-credit compromises can be valuable even when they cannot solve.

The second guess maximizes the minimum achievable final reward over all consistent secrets. Subject to that common guarantee, each branch maximizes mean terminal reward under a uniform belief, with solve count breaking ties. The actual third guess executes only the matching, verified feedback branch. The opening is chosen by digit/position symmetry, not by any episode seed. There are no seed-specific rules, secret access, model-backend shortcuts, or persistent cross-episode caches.

Every action checks `ctx.progress()`, rejection counts, observation changes, legal actions, and game-specific effects. Mastermind additionally verifies old feedback persistence, nonempty beliefs, native turn count, and terminal score arithmetic. A contradiction disables the scripted controller and gives Astra the evidence, public observation, rules, and legal actions for a new single-step plan. That action is checked by the same loop. Model timeouts and malformed answers are caught, with a legal fallback; recovery calls allow 480 seconds, above the archive's 217-second maximum.

## Confirm or refute

Confirm if exhaustive public-rule simulation establishes a useful worst-secret floor, the minimum Mastermind score rises above 0.375 without losing TicTacToe draws, and the mean improves with fewer calls. Refute if robust hedging reduces mean reward enough to lose to a risk-neutral policy, if terminal guarantees fail in native observations, or if parsers trigger unnecessary model recovery. Report simulated all-code performance separately from native dev/training results. No held-out seed/result inspection is used.

## Validation

Preflight passed (`ok: true`, native TicTacToe draw, four actions, no model calls). One dev evaluation was run, at `dev/20261031-121014`: seed 210 scored **0.5**, seed 211 scored **0.625**, mean **0.5625**; both ended normally with zero model calls, zero invalid actions, and no verification discrepancies. All native dev feedback and terminal rewards matched the local model. No second dev run was needed.

`check.py` and `checks.json` record independent public-rule checks (not native training or held-out evaluation):

- All 129,600 guess/secret feedback and reward pairs agree with an independent Counter-based implementation.
- Exhaustive enumeration of all 360 possible Mastermind secrets gives mean **0.6392361**, minimum **0.375**, and **38/360 solves**. Scores occur at 0.375/0.5/0.625/0.75/1 with counts 26/105/107/84/38. Every planned guarantee holds; every final branch attains the highest mean available under its common guaranteed floor.
- The universal-floor hypothesis is **partially refuted**: 26 codes still score 0.375. The branch corresponding to baseline seed 15's first feedback does guarantee at least 0.5, and the branch for seed 13's first feedback guarantees 0.625. These are mathematical consequences of public clues, not replays using hidden answers.
- Given the baseline seed 15's exact two public clues, 16 secrets remain. Its final `[4 2 3 6]` has a worst-case score of 0.375 and mean 0.59375. A robust final `[6 2 3 4]` guarantees 0.5. This supports the diagnosed unprotected downside without claiming what the unseen actual secret was.
- Exploring all opponent continuations under the selected TicTacToe policy yields 152 terminal paths as player 0 and 635 as player 1, with no losses.
- Seven injected failure classes exercise actual harness recovery: rejected actions, unchanged observations, action exceptions, unfamiliar feedback format, model timeout, model RuntimeError, and malformed model replies. Every test finishes in at most four actions; every anomaly invokes System-2 and all returned actions remain legal.

The exhaustive code model implies a 0.5696181 equally weighted suite mean under uniformly sampled Mastermind codes and perfect TicTacToe, not a prediction of the small fixed training sample. The observed native dev mean is lower. The risk-neutral-versus-robust comparison remains for the archive's candidate evaluation; do not interpret the 0.6392 simulated mean as a native benchmark score. The implemented search deliberately preserves the floor-first hypothesis rather than tuning it after seeing dev results.
