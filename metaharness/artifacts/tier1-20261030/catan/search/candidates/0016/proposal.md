parent: 0013

## Hypothesis

The remaining failure in the leader is an early capital-allocation error rather than a
legality or parsing error.  In `0013/trace_seed103.jsonl`, after the two setup settlements
the agent repeatedly passed, bought development cards (for example the actions at turns 11,
43, 63, and 71), and spent roads without completing a third settlement.  It finished at
6 true VP while RED reached 10.  The partial-inventory compiler was legal and its recovery
worked, but it still allowed a card project to outrank an available expansion when the
economy had not yet acquired another production source.

## What changed

This candidate copies `0013`'s verified graph bookkeeping, exact maritime compiler, partial
conversion forecast, tactical-card rules, model prompt, progress detector, and bounded
planner recovery.  It adds one narrow expansion-first gate: below 8 VP, while still owning
only the two setup settlements, a ready development-card purchase is skipped when a current
settlement project has a legal route or is fundable.  The gate is released for an explicit
`army` mode, for late-game VP pressure, or when no expansion project exists, so it does not
ban development cards globally.  Every action remains selected from the current legal menu.

At 9 VP it also checks the exact development-card cost as an immediate hidden-VP win route;
if the hand is fundable it executes the current maritime trade (if needed) and then buys the
card rather than ending the turn while waiting for a board build.  Existing postcondition
checks, `ctx.progress()` stagnation evidence, stale-response handling, and garbage-reply
fallbacks are unchanged.

## Evidence that would confirm or refute it

Confirmation would show fewer early development purchases before a third settlement, earlier
productive expansion on weak-production maps, and a higher weak-seed terminal VP or win rate
without illegal actions.  The endgame safeguard would be supported by a trace reaching 9 VP
with an affordable card and winning through that purchase.  Refutation would be an increased
loss rate on boards where an early Knight/hidden-VP or Army race is the correct route, or a
drop in the leader's existing wins because the gate delays a decisive card path.  A legal
execution with no score gain would be evidence against the hypothesis, not a reason to widen
the gate blindly.

