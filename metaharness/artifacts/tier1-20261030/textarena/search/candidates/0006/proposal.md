parent: 0003

## Hypothesis

Robust Mastermind planning needs a mixed strategy over **whole contingent
policies**, rather than a single safest action in each posterior. Uniformly
mixing all digit relabelings of a Bayes-optimal policy equalizes expected score
across every fixed secret, while retaining the optimal uniform-code mean. This
can protect unseen secrets from the arbitrary lexical asymmetry of a deterministic
policy without the expectation loss of pure maximin. The guarantee is over the
harness's randomness, not the realized score of a single episode.

Archive evidence: leader 0001's seed 11 opens `[1 2 3 4]`, receives "0 black
peg(s), 3 white peg(s)", probes `[1 3 2 5]`, and receives the same feedback. Its
20-code posterior ends with `[2 4 5 3]`, scoring 0.5. Parent 0003 reaches the same
posterior but selects `[2 1 4 3]`, scoring 0.375; its logged guarantee is 0.375.
Neither reports an execution error. Parent 0003's exhaustive public-rule check
also leaves 26/360 codes at 0.375 while lowering the all-code mean to 0.6392361,
below the leader's 0.6649306. More pure worst-case optimization has limited room.

## Change

Retain 0003's public belief tracking, TicTacToe negamax, one-action execution,
contingent-branch verification and Astra recovery. Replace pure floor-first
planning with an episode-level mixed policy. At initialization, uniformly shuffle
all six digit labels using a private `random.SystemRandom` instance. This
permutation defines the opening **and all later tie ordering**. Search still
considers every legal second probe, each possible feedback branch, and every
legal terminal guess, including eliminated secrets. Maximize exact expected native
reward, then solve probability, with the sampled canonical order resolving ties.
Store each branch's belief, chosen action and reward certificate and recheck them
before execution. No arbitrary risk coefficient or seed-fitted permutation.

The group argument is exact: for any fixed secret, a uniform six-digit permutation
maps it to each of the 360 abstract codes exactly twice. A fixed canonical policy
therefore has the same payoff distribution on every fixed secret when conjugated
by this random permutation. Because every opening is symmetry-equivalent and the
remaining two decisions are searched exhaustively, the canonical policy maximizes
uniform-code mean. Its symmetrization attains that same value against every secret
in expectation, which is the optimal mixed maximin value: no policy's minimum
expected payoff can exceed its uniform average. This is a change in robustness
architecture, not a claim to improve the leader's already-optimal Bayes mean.

The randomness is independent of episode seeds and engine randomness; neither is
read. Runtime uses only observations, legal actions, local public-rule arithmetic,
and budgeted ctx calls. Normal play requires no Astra. Every action checks
ctx.progress(), rejection counters, native move count, old board/clue persistence,
feedback consistency, and terminal reward/horizon. A mismatch disables compiled
control and sends public evidence to Astra; the replacement action is verified
again. Garbage replies and caught model failures fall back to an unrejected legal
action. Rejected/raised actions are suppressed. The timeout is 480 seconds capped
by remaining wall time, with bounded search and no background workers.

An initial implementation mixed terminal actions independently via a zero-sum LP.
It raised the archived 20-code posterior's worst-secret expected score from 0.375
to 0.529661, confirming that mixing matters, but its selected second probe after
(0 black, 3 white) had mean only 0.551792. That local maximin constraint unnecessarily
sacrificed mean. Root symmetrization retains the mixed-strategy hypothesis while
removing this cost; it does not cherry-pick a better random draw from dev results.

## Confirmation / refutation

Exhaustively evaluate the canonical contingent policy over all 360 abstract codes;
then verify digit equivariance and the 720-permutation orbit counts. Together
these establish the randomized policy's exact distribution for **any** fixed
secret without repeatedly sampling random seeds. Confirm if its expected
worst-secret value equals its uniform mean and exceeds the parent's pure guarantee.
Check native preflight/dev behavior and injected recovery separately.

Refute implementation if equivariance or certificates fail, if the canonical
all-code mean loses to exact reward search, or if verification triggers falsely.
The practical benefit can fail even with the theorem: the evaluator samples only
four Mastermind episodes, so a randomized candidate can have a worse realized mean.
A single low dev score is neither proof of faulty planning nor a reason to retune
its permutation. This candidate intentionally tests fairness across secret codes,
not an improvement in expected mean over the leader. No training, validation or
held-out evaluation is run during development.

## Validation

- Required final preflight passed: TicTacToe draw, four actions, zero model calls,
  no exception. Its ordinary-path warning about unexercised model responses is
  covered by explicit recovery stand-ins below.
- One native dev evaluation (`dev/20261031-123343`) scored 0.5 on 210 and 0.625 on
  211, mean 0.5625, with zero model calls and no errors or rejected actions.
  Both digest and step traces were inspected: all seven verification records have
  `issue: null`. Mastermind took about two seconds. Its sampled digit map is logged
  for reproducibility of the action trace; that map is not installed as a constant.
  No second dev run was used to search for a favorable realization.
- `check_policy.py` / `policy_checks.json` independently verify all 129,600
  feedback/payoff entries and exhaust the canonical 360-code policy. Mean reward
  is **0.6649305556**, 53 solves, and mean action count 2.9944444. Reward counts are
  0.25: 4; 0.375: 32; 0.5: 77; 0.625: 85; 0.75: 109; 1.0: 53. This equals the
  leader's all-code mean, as predicted; there is no claimed Bayes-mean gain.
- The same audit verifies complete policy equivariance under a digit swap and
  six-cycle (generators of all label permutations), checks 210 terminal payoff
  certificates, and enumerates all 720 group elements for each of the 360 secrets.
  Every secret's orbit contains each abstract code exactly twice. Consequently
  every fixed secret has **expected score 0.6649305556**, solve probability 53/360,
  and the score distribution above. This is an abstract public-rule calculation,
  not a held-out score. The equally weighted expected suite score is 0.5824652778.
- `check_recovery.py` / `recovery_checks.json` explore all 787 terminal TicTacToe
  opponent paths across both seats with no losses. Seven failure classes—rejection,
  no observation progress, action exception, changed feedback format, model
  timeout, backend RuntimeError, and garbage reply—each call the budgeted model
  interface and finish within four actions. These are synthetic stand-ins rather
  than real-model recovery measurements.

The mixed guarantee is supported, while any claim of improving the uniform mean
is explicitly rejected. Conditional on full search completing under the documented
rules, the root mixture is optimal against the worst fixed secret **in expectation**.
It does not improve the minimum realized reward, and a tiny training sample may
favor either policy by chance. Deadline fallbacks and rule-repair paths prioritize
legal completion and are outside that optimality claim.
