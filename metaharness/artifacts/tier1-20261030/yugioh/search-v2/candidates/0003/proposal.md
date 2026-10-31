parent: none

## Hypothesis

A public-board tactical controller can handle the many routine native decisions in this small card pool, while sparse Astra arbitration improves difficult exchanges. Exact action transport, subset handling, and locally bounded model waits are prerequisites: neither good reasoning nor a backend timeout alone guarantees that an engine decision will be answered.

This is a fresh code-first design rather than a refinement of the naive model-per-step recipe. It tests whether useful play can continue, and duels can finish, even when Astra is slow or unavailable.

## Archive evidence

In `0000-tier1_astra_naive/trace_seed1101.jsonl`, the first model response is `choose r1p1 1`, but the next step posts `choose` and receives `Expected current prompt token and arguments`. All 192 posted actions across the six baseline traces are bare `choose` and receive that error. Seed 1101 finishes with three forfeits and zero native completions; seed 1106 repeats the failure over 1568 seconds. The leaderboard therefore does not establish that Astra plays poorly: its selected actions never reached the engine intact.

## Implementation

- Preserve complete tokenized actions. Validate model choices against the current prompt, and validate subsets/permutations, including the prompt's tribute weights. Subset prompts may also contain a cancel choice: their `mode`, not the presence of `choices`, determines dispatch.
- Rank routine summons, Sets, positions, optional activations and targets using public board state, current statistics and the observed card catalogue. Handle optional effect confirmations separately from target selection. Retain effect intent through mandatory follow-up prompts.
- Search visible battle assignments, valuing destruction and LP damage, and replan after every native decision. Unrevealed defenders are anonymous estimated 1800-DEF blockers; this is explicitly a heuristic, not knowledge of a hidden card. Guard repeated main/battle choices using public state and semantic actions, since cancellation can produce a new token without changing the board.
- Ask Astra about contested main-phase choices and consequential chains, with compact current state, relevant card texts, recent events and the controller's ranking. Allow at most six calls per duel and eighteen per episode, preserving call-budget headroom. Advice can reserve named optional cards during the current turn's main phases; it expires at the turn boundary. Future action tokens are never replayed.
- Request a 65-second backend timeout, but independently wait at most 70 local seconds on a `ctx.spawn` inference future. On delay, malformed output or failure, act with code immediately. Only one inference call may remain pending. Late answers are discarded; the evaluator owns cancellation of inference at episode close. There is no autonomous background gameplay loop.

## Development evidence and limits

The first dev run reached 78 legal engine steps, then was interrupted after approximately eleven minutes because one synchronous model request had remained pending for over seven minutes despite `timeout_s=95`. It yielded a partial trace, no completed evaluation/digest, and no claimed score. This motivated the independent local wait limit. Its observations also exposed `activate_effect` yes/no choices that both contain the source card; generic target ranking had incorrectly declined Penguin Soldier.

The second and final permitted dev evaluation tested the bounded-wait version on both dev seeds. Both completed without errors or forfeits, and each won its first duel using code while a model request remained pending. However, each later entered an attack/cancel or summon/cancel loop: a subset prompt's supplementary `cancel / select none` choice was mistaken for its entire action space. Each episode reached the harness's 16000-action guard, with one native win, one truncation and one unfinished duel. Mean score was 0.358958. **This is an intermediate version's result, not a measurement of the final harness.** The digest and public traces were inspected before repairing subset dispatch.

After the subset repair and final guards, the final preflight completed the entire episode in 435 actions with score 1.090833 and zero model calls, rather than running to its deadline. Offline regression checks passed on 394 distinct public prompt states spanning eight observed prompt kinds, plus attack/tribute cancellation, weighted tribute validity, visible battle arithmetic, stale tokens, garbage model replies and ctx-owned stand-in model failures. The final preflight result is saved in `preflight.json`; `check_recorded.py` reproduces the offline checks without launching games or inference.

No further dev evaluation was run after the two permitted attempts. Preflight is contract/behavioral validation, not a training or held-out score. No game implementation source, private state, seed-conditioned strategy or other experiment archive was used.

## Confirmation and refutation

Confirmation would be native completion of all three training duels, no invalid-action loops or response forfeits, substantially fewer than 40 model calls, and positive native win rate. Refutation would be continued legal-action loops, systematic effect/target mistakes, poor unknown-defender estimates, or advice that disrupts an otherwise coherent controller. The most useful next analysis is to separate code-only battle/interaction mistakes from Astra overrides and to check whether the sparse calls improve outcomes enough to justify their latency.
