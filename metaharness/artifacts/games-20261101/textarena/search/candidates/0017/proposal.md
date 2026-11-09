parent: 0011

## Hypothesis

Combine two distinct guarantees: the parent's realized score floor and equal
expected reward for every fixed secret. Candidate 0011 commits to the strongest
whole-tree floor (0.375), then maximizes mean reward subject to that commitment.
Its deterministic digit ordering still assigns different scores to different
secrets. Uniformly conjugating this entire safe policy by all six-digit
permutations should remove that bias without losing either its mean or its hard
floor. This is a conservative composition of verified mechanisms, with 0011 as
the parent; it does not change the leader's expected-reward objective or tune a
random map against benchmark outcomes.

The leader's weak episode is not an execution failure. In
`0006/trace_seed13.jsonl`, `[5 3 2 4]` gets `(0 black, 3 white)`, then
`[5 2 3 6]` gets `(1,2)`; the final `[4 2 3 5]` scores 0.5. All three
`verified_step` records contain `issue: null`. Its exhaustive public-rule
distribution nevertheless includes four 0.25 outcomes among 360 codes. Parent
0011 removes those outcomes for a mean cost of only 3/2880. But its own
`trace_seed11.jsonl` ends with `[2 1 4 3]`, `score: 0.375`, after two `(0,3)`
clues. The logged 20-code leaf has expected score 0.5375, and every
`plan_verified` record has `ok: true`. That is a real tail outcome, not a reason
to invoke recovery or pretend the model misunderstood the game.

## Change from parent

Keep the complete root-floor compiler, 16-orbit exact probe reduction,
TicTacToe minimax, single-action execution, and Astra recovery. Before any
Mastermind move, sample one uniform permutation using `random.SystemRandom`.
Map every public guess, legal action, posterior, and chosen action through that
same bijection. Compile and resolve all ties in canonical coordinates. The map
is recorded in the trace and checked before each decision. Randomness never
reads the episode seed or the environment RNG.

Native observations remain authoritative: check that their exact posterior is
the image of the canonical posterior, and independently verify the selected
terminal action's entire native reward histogram and floor. The inherited
post-action verifier checks `ctx.progress()`, rejections, consumed legal moves,
native plies, persistent clues/marks, rule consistency, terminal horizon and
reward. Disagreement abandons compiled control and sends the evidence to Astra.
Every replacement action is checked again. Model failures and garbage replies
fall back to an unrejected legal move; calls are capped at 480 seconds and by
the remaining episode budget. No background work or nonstandard imports.

For any fixed native secret, a uniform digit permutation maps it to each of the
360 canonical secrets exactly twice. Therefore its expected score is the
parent's uniform-code mean, 1912/2880 = 0.6638888889, and every realized score
still meets the parent's 0.375 floor. This is expected-score equalization WITH
realized insurance. The leader's unconstrained symmetrization has slightly
higher expectation, 1915/2880, but permits realized 0.25 outcomes. Neither
policy dominates the other under the evaluator's mean objective.

## Confirmation and refutation

Confirm by independently checking the canonical 360-code payoff distribution,
all 720 label permutations, and controller replays in native coordinates.
Require every fixed secret to inherit exactly twice the canonical histogram
over the permutations, no outcome below 0.375, and unchanged canonical mean
and action count. Inject corrupted maps/branches, rejected actions, stalls,
changed feedback, garbage model responses and model exceptions to test the
repair path. Run pre-flight and one native dev evaluation after files exist.

Refute if any conjugate loses the floor, posterior or payoff certificates fail,
or native execution falsely enters recovery. The four-Mastermind training
sample may favor either policy by chance. This proposal predicts no increase
in uniform Bayes mean; it tests whether hard insurance and secret-independent
expectation can coexist at the parent's already-measured small cost. No full
evaluation or held-out results will be accessed. Guarantees apply to the
certified policy under matched rules, not to deadline or Astra repair fallbacks.

## Validation

- Required pre-flight passed: TicTacToe draw, four actions, zero model calls
  and no errors. Its warning that the normal path does not exercise model
  responses is addressed by the explicit fault-injection checks below.
- `check_policy.py` / `policy_checks.json` independently recompute all canonical
  leaf rewards, then enumerate **720 permutations × 360 abstract secrets =
  259,200 pairs**. Every native feedback and terminal reward matches its
  canonical counterpart. Every fixed secret receives exactly twice this
  histogram over the 720 policies: **34 at 0.375, 81 at 0.5, 86 at 0.625,
  108 at 0.75, and 51 at 1.0**. Thus each fixed secret has expected reward
  **0.6638888889**, floor **0.375**, and expected action count **2.9944444**.
  These are public-rule calculations, not training or held-out measurements.
- **1,080 synthetic controller replays** cover all 360 abstract codes under
  identity, a digit swap and a six-cycle. Native-to-canonical belief checks,
  external payoff certificates, omitted solve feedback, and every post-action
  effect check pass without invoking recovery. Compiling the unchanged root
  policy took about **1.12 seconds** in the local audit. TicTacToe still avoids
  defeat on all **787** terminal paths against arbitrary opponent moves.
- **16 new mapping corruption/reply combinations** detect duplicate labels,
  broken inverse maps, broken forward maps and changed labels. Each enters
  Astra recovery and finishes legally. Inherited tests also pass **16**
  rejection/stall/action-exception/parser-failure combinations, **15**
  contingent-plan corruption combinations, and **six** direct effect mismatch
  cases plus successful-solve feedback omission. Responses include legal
  actions, garbage, `TimeoutError` and backend `RuntimeError`. These use
  stand-ins through the context interface, not direct model backend calls.
- One native dev run, `dev/20261031-133544`, scored **0.5 on 210 and 0.75 on
  211**, mean **0.625**, with zero model calls, rejected actions or errors.
  The digest and both traces were inspected; all seven `plan_verified`
  records have `ok: true`. Mastermind completed in **2.76 seconds**. Its
  sampled map is recorded in the trace and is not installed as a policy
  constant. No second dev run was used to seek better randomness.

The composition works: a realized floor and a secret-independent expectation
coexist. The cost versus the unconstrained leader remains **0.0010416667**
expected Mastermind reward (half that in the balanced suite). The favorable dev
sample does not establish a mean improvement. Compared with deterministic
0011, the gain is eliminating permanently disadvantaged secret labels; compared
with 0006, it is removing the 0.25 tail at a known, small expectation cost.
