parent: 0002

## Hypothesis

The remaining losses after 0002 are caused by the controller dividing decisions too coarsely: it either applies a fixed local command heuristic or asks Astra to answer every prompt. The fixed heuristic cannot express the current turn's objective, while repeated per-prompt calls spend the episode budget and invite stale answers. A concrete failure is `candidates/0002/trace_seed1101.jsonl`: after only 8 engine steps and 6 System-2 requests, the episode hit its 1800-second deadline with two forfeits. Conversely, the 0003 trace asks Astra for a “SHORT plan” containing “up to SIX FUTURE main-command intentions”; its seed 1108 run still reached 557 actions and ended at -1.07, showing that long intent queues do not guarantee useful progress. A short priority policy from Astra should preserve strategic context without binding future prompt tokens.

## What changed

Candidate 0006 asks Astra once at a turn or on a detected failure for the exact current action plus an ordered list of card/command priorities. The exact action is checked against the current menu. Later command menus can consume only a matching priority, and otherwise use conservative strong-attacker/end-turn automation. Effect, chain, target, and compound prompts remain model-controlled; code validates selections, orders, tokens, and weighted constraints. `ctx.progress()` and prompt-token changes trigger replanning after rejection or a stalled board. Calls use a persistent Astra session and a bounded 150-second timeout, with all model and action failures recovered to legal fallbacks.

This is structurally different from 0003's multi-step future-command plan: priorities describe intent, while every action is selected from the current native menu and no future action is committed.

## Evidence that would confirm or refute it

Confirmation would be fewer repeated Astra calls than 0002, no stale/illegal actions, and improved native win rate on both seats, especially games where the right choice is sequencing removal, a Flip monster, or a strong attacker. Traces should show priorities being rebound to menus after opponent responses rather than blindly replayed. Refutation would be frequent empty or unavailable priorities, repeated end-turn cycles, poor handling of mandatory compound prompts, or no win-rate improvement despite completed duels.
