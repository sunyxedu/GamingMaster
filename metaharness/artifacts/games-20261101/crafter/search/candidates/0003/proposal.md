parent: none

## Hypothesis and evidence

Stable geography and renewable supplies may be a useful alternative to roaming achievement dependency planners. Establishing a persistent camp near water, keeping table and furnace together, planting crops, and returning with resource loads should reduce survival interruptions and repeated cross-map travel. This is a fresh camp-and-expedition design, distinct from the parallel proposals' general objective compilers. It also removes the first model call as a prerequisite for acting.

Baseline `trace_seed1106.jsonl` and `trace_seed1108.jsonl` end with `steps: 0`, `score: 0.0`, one System-2 request and the 1800-second deadline. Seed 1104 ends with seven movement actions, health 9, and zero achievements. Seed 1101 reaches four achievements in only 29 actions. These are latency/interaction failures, not evidence that the model cannot choose useful strategic goals. The successful baseline inventory deltas verify a two-wood table and one-wood wooden pickaxe. Archived dev observations additionally verify one wood for a wooden sword, one wood plus one stone for stone tools, four stone for a furnace, and `plant_growing` as the semantic object name for young crops. These are public observation traces, not simulator-source inspection.

## Final design

Maintain observed terrain separately from current creatures, remembered crops, and a 12-tick memory of threats just outside the moving view. Threat memory affects navigation costs only; attacks require currently observed enemies. Choose a camp using observed water, stone, trees and navigability, reserving a shared crafting position beside a table and furnace. Every action is followed by a fresh observation and survival checks.

Collect four wood before building the table, then craft the wooden sword and pickaxe together. Provision wood and stone for both stone tools, furnace and a placed block in one excursion when health permits. Prioritize stone weapons before farm expansion. Farming receives a bounded work allowance and targets one to three plants depending on Astra's advice. Rare iron is gathered for one tool at a time so the first pickaxe is not delayed while searching for a second iron. Return trips use weighted Dijkstra search over observed walkable and mineable terrain; unexplored cells are never treated as known passable terrain. Exploration values new visibility and modestly penalizes distance from camp.

Placement retains its target while moving into the required position/facing. Successful inventory deltas verify recipes; failed placements, unchanged interactions and failed craft states are bounded. Verified recipe costs cannot be overwritten by model suggestions. Combat targets wounded adjacent enemies and attempts a finishing hit even at low health; otherwise it retreats when outmatched. Drink/food have hysteresis, and sleep is interrupted by threats.

Astra gets a compact camp ledger with resource coordinates, missing achievements, recent deltas and recipes. Its first review is after 120 useful ticks; later reviews require 400 ticks between calls and stagnation, with at most five calls. It selects an exploration heading, farm/mine/hunt emphasis, and optional corrections to unverified recipe hypotheses. Calls have a 150-second timeout and retain executable fallback behavior after malformed JSON, timeout or runtime failure. No background tasks or System-1 calls are used.

## What happened during development

Files were written before testing. Initial pre-flight passed. Dev evaluation 1 (`dev/20261030-094342`) scored 10/22 on both seeds, mean 0.454545, with zero model calls and no errors. Both died before 220 ticks. Traces refuted the naive version of the hypothesis: one-item procurement caused repeated forest/camp trips, and replacing destroyed plants delayed stone weapons. The first revision batched common supplies, prioritized swords, bounded farming, and improved combat target selection.

Dev evaluation 2 (`dev/20261030-094938`) scored 14/22 and 17/22, mean **0.704545**, with one and two Astra calls respectively and no errors. Episodes ended at 257 and 584 ticks. Seed 3302 established table, furnace, crops and stone tools, collected coal and iron, and maintained supplies for hundreds of ticks. It then spent roughly 230 ticks acquiring a second iron instead of immediately crafting its first iron tool, followed by a long dangerous return. Neither seed unlocked `eat_plant`; the farming benefit remains unconfirmed.

Three final repairs followed inspection of that second run: persistent placement targets (a moving target caused a long stone-placement detour), immediate crafting from the first rare iron load, and short threat memory (steps 509–520 alternated left/right as a zombie crossed the observation boundary). Astra's mine focus also reduces the crop target to one. These final changes were **not** evaluated in a third dev run: the two-run budget is exhausted. Therefore 0.704545 is the measured score of the second-run snapshot, not a measured score of the final file.

The final pre-flight passed with `ok: true`, no exceptions, 14/22 achievements and 214 actions. Intermediate pre-flight scores varied with revisions and are not training/dev estimates. Pre-flight answered no System-2 call, so `validate.py` separately checks eight valid/malformed/failing response cases, placement from all four orientations, and replay of the visibility-boundary route loop using archived public observations. All checks pass. The replay changes the loop's repeated reversal into a continuing route; it does not prove long-run survival.

## Confirmation, refutation, and next questions

Confirm the camp hypothesis if independent episodes consistently convert gathered loads into upgraded tools, revisit mature crops, maintain supplies, and raise the minimum achievement count. Refute it if farming work and return trips outweigh their benefits, camps remain too far from iron/diamond, or defensive routing prevents any return to the stations. More actions alone is not success.

Current evidence strongly supports executing deterministic skills before waiting for inference and batching abundant materials. It only partially supports a persistent camp, and does not yet support renewable farming. The clearest unresolved problem is expedition return safety: avoid sacrificing an iron load on a long night-time return, or test whether a second temporary station near deep resources is worth its cost. Any such relocation is a future hypothesis, not part of this submission.

Only standard-library imports are used. All runtime world knowledge comes through observations; no hidden state, seed-conditioned policy, game implementation source or direct model backend call is used.
