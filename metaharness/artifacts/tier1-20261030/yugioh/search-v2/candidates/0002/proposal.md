parent: none

## Hypothesis
The baseline has not demonstrated weak Yu-Gi-Oh tactics: it never submitted a complete legal response. In `0000-tier1_astra_naive/trace_seed1106.jsonl`, line 3 contains Astra's response `choose r1p1 0`, while line 4 posts `choose` and receives `Expected current prompt token and arguments`. The prompt remains `r1p1`. The aggregate archive records zero native completed duels and 2.67 forfeits per episode. Repairing serialization is necessary but a model call for each engine prompt would still be a poor match for a forty-call three-duel budget.

This fresh design tests a different division of labour: an observation-only tactical controller should complete all three duels, with sparse Astra adjudication at resource-trade and chain decisions. The controller is always capable of acting without the model. Model output is advisory and must pass the same current-prompt validator as scripted actions.

## Implementation
- Preserve exact `choose TOKEN INDEX` strings; validate model responses against the current prompt. Constrained subsets and orderings are encoded separately. Invalid responses are blacklisted for that prompt and retried without another model call.
- Hydrate legal card references from current visible zones and the observation's catalogue. Never treat a slot as a permanent identity, and never assign identities to hidden cards.
- Script summons, setting, recovery/removal targets, position choices, and visible battle arithmetic. Small subsets are enumerated subject to the displayed constraints. Card-specific rules distinguish destroy from negate, ordinary removal from mass removal, and tribute investment from raw ATK.
- Ask Astra at most seven times per duel, at a difficult main-phase or chain decision. Every request is self-contained, includes current effect text, recent events, all legal options, and the controller's ranked suggestion. Store only a short plan; never execute stale planned tokens.
- Cap individual calls at 65 seconds, catch model failures, reserve calls and wall time, and continue scripted play after the discretionary allocation is exhausted.

## Confirmation and refutation
Primary confirmation is three native completions with zero illegal-response loops and no forfeits/errors. Secondary confirmation is positive native win rate with substantially fewer than forty calls. A completion improvement alone does not establish that the tactical rules are strong. If duels finish but systematically lose to mis-targeted effects, wasted combat tricks, poor tribute trades, or avoidable battle losses, the tactical abstraction is refuted and traces should identify the next specific correction. No seed-conditioned policies or private game state are used.

## Validation
Pending preflight and a limited dev evaluation. This initial version is written before any dev feedback; subsequent fixes and results will be appended below.

## Dev probe
A single permitted dev seed produced 104 consecutive legal steps through turn 9 without a `last_action_error`; the trace exercised `select_zone`, `select_cards`, `battle_command`, and `activate_effect` prompts. One Astra call timed out and was caught, after which the scripted fallback continued. The probe was stopped before a terminal result, so it is evidence for contract robustness only, not a score claim.
