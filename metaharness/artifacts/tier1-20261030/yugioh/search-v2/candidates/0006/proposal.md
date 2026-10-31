parent: 0003

## Hypothesis

Candidate 0003 solved the catastrophic action transport problem and its public-board controller can already finish duels. Its remaining risk is arbitration cadence: opportunistic Astra calls during chains and later turns add tens of seconds, can return after the prompt token has changed, and provide advice after the controller has already accumulated enough public evidence to re-plan. A stable opening policy selected once from the hand, seat, board and opponent-visible context should preserve the model's strategic value while removing stale mid-duel overrides and most latency exposure.

## Evidence from the parent

The leader's digest reaches a native win in 215 steps on seed 1101, then records a 36.6-second System-2 call cancelled because the game was already over. The parent permits up to 18 episode calls and six per duel, even though routine chain responses and battle prompts are resolved by the same deterministic ranking. Earlier candidates also show that long model calls can consume the episode deadline. These observations motivate moving model use to a single deliberate opening decision per duel rather than using it as a recurring tactical oracle.

## What changed

Starting from 0003's exact-action parser, constrained subset/order validation, visible battle search, effect bookkeeping, bounded background inference and exception-safe fallback, this candidate changes arbitration cadence. It allows at most one Astra call for each of the three duels, and only on the first meaningful `main_command` prompt. The request still returns an exact legal action plus optional target names, a short plan and cards to reserve; that plan remains in memory and the controller re-plans every later native prompt from fresh public state. Chain responses, target subsets, tribute choices and attacks therefore never wait for a new model answer. The episode-wide System-2 ceiling is three calls, leaving ample budget and wall time for all three duels.

This is a structural test of division of labour, not a card or seed-specific rule change. If the opening hand is simple, no call is needed; malformed, stale or timed-out advice falls back to the same legal scripted action.

## Confirmation or refutation

Confirmation would be native completion of all three duels with zero response forfeits, fewer System-2 calls and lower wall time than 0003, while retaining a positive win rate across deck and seat strata. A useful secondary signal is that no model answer is logged after a terminal result and no action follows a stale late answer. Refutation would be a systematic drop in wins caused by the opening plan being too coarse, or cases where a mid-duel intervention is clearly required to stop lethal. Such traces would indicate that arbitration should be reintroduced only for a narrowly identified prompt class, with explicit token freshness checks.
