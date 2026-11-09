parent: 0017

## Hypothesis

The public race interrupt in 0017 catches an opponent at 8–9 VP, but it can still spend many accepted roll/end-turn/build-road actions before that threshold is visible. The concrete archive failure is a long legal plateau: the agent remains alive and its construction contract remains legal, yet `ctx.progress()` reports many actions since the last score change. This is an opportunity-cost failure rather than an illegal-action failure. I test whether a single evidence-gated conversion interrupt, triggered by sustained score stagnation, improves recovery on boards where a route has become unproductive.

## Change versus 0017

I retain 0017's staged settlement financing, public-threat interrupt, exact integer maritime trades, postcondition checks, bounded Astra calls, and live legal-action matching. The new executor watches `ctx.progress()` after each observation. When 48 accepted actions have passed without a VP change, at least 12 turn numbers have elapsed since the last point, and the same interrupt has not fired for 24 turns, it records the progress evidence, abandons the current construction contract, and asks Astra to re-plan. For the next 16 turns, the local policy prioritizes any currently funded city, settlement, or award (including exact trade steps), then useful tactical cards, and will not choose an otherwise idle end-turn while a conversion exists. The gate is edge-triggered and cools down, so ordinary productive saving is preserved and model calls do not repeat every action.

This is a careful structural test of the contract's failure detector, separate from 0017's opponent-VP threshold and from the earlier partial-trade and investment experiments. Every action still comes from the current legal menu and is checked with the existing postcondition verifier; malformed or timed-out model output keeps the local fallback.

## Confirmation / refutation

Support would be traces showing `conversion_urgency` with nonzero `since_score_change`, a fresh Astra request containing that evidence, fewer long roll/end-turn plateaus, and more point conversions after a stalled route. Refute if the interrupt fires during a productive city build-up, abandons routes that would have won, increases development/road waste, or leaves win rate and weak-seed terminal VP unchanged. The trigger is intentionally public and conservative; it does not infer hidden opponent hands or future dice.
