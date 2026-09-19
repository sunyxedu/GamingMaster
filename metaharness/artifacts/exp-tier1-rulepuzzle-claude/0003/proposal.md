# Candidate 0003 — "rules are episode constants": cross-level transfer of colour semantics

## Where the parent stands (evidence)

The leaderboard is saturated: `0001` and `0002` both score 2.0 with 0 System-2 calls
on seeds 0/1 (both `teleport`), and 0002 is 2.0 on held-out 100–102 too. The only
measurable axis left is *action efficiency* — the margin the harness keeps if the
evaluator tightens `budget_per_level` or a level needs a long route.

I measured 0002 per level against a full-knowledge oracle (Dijkstra with the true
rules), 300 local seeds, 8×8 / 2 levels / 40 per level, Astra raising:

| family   | L1 used / oracle / over | L2 used / oracle / over |
|----------|------------------------|-------------------------|
| teleport | 9.0 / 5.0 / +3.9       | 4.9 / 4.9 / **+0.0**    |
| mirror   | 12.8 / 10.0 / +2.8     | 10.0 / 10.0 / **+0.0**  |
| key_door | 12.8 / 10.1 / +2.6     | 16.6 / 10.1 / **+6.5** (worst seeds +20) |
| toggle   | 15.6 / 11.0 / +4.6     | 12.3 / 11.0 / **+1.3**  |

Level-1 overhead is discovery cost (which action is which direction, which colour
does what) and is bounded below by information. **Level 2 has no such excuse**: the
rules are fixed for the episode, so level 2 should be pure planning at oracle cost —
and for two families it was not.

Concrete failure (seed 207, 0002, level 2 = 30 actions vs oracle 10): in level 1 the
key was picked up incidentally, so the door colour (2) was *walked through* and is in
`passable`. In level 2 the planner routes straight through the door, bumps it, spends
`ACTION5`+`ACTION6` on it (`experiment: interact colour 2`), walks 7 steps to the key
tile, stands on it without knowing it holds the key (the vanish is only detected on
leaving), spends two more interact tests on a grey wall, logs `stuck`, and only then
walks through the door. Seeds 146/59/287 are the same story.

Toggle (seed 178, level 2 = 14 vs 11): at (6,7), one step below the exit, the agent
steps left, then right, then up. The trace shows the planner chose the bottom-row route
(needs the still-unknown `up` action) over the interact route because a toggle cell was
charged interact cost 2 **plus** the +2 "untested colour" surcharge (toggle colours are
never in `passable`), and because once a toggle colour was cleared once it was popped
from `blocked`, so `can_enter` said walkable and the planner walked *into* solid toggle
tiles.

## Hypothesis

Everything learned in level 1 about *colour semantics* is a hard constraint in level 2,
but 0002's beliefs are stored as unconditional facts (`passable`, `blocked`) that are
not valid across levels because keys are lost between levels. Making passability
**conditional on the keys held** and inferring door/key relations from history should
bring level 2 to oracle cost for every family, without touching the level-1 engine or
adding any Astra calls.

## What changed vs. 0002 (engine, planner, Astra prompt/parse otherwise identical)

1. **CONDITIONAL-PASSABILITY** (`World.enter_keys`, `level_start_inference`): every time a
   colour is entered we record the set of keys held. At the start of a new level a rare
   colour that was *only ever entered while holding key K* is hypothesised as a door for K
   (`unlocks[c]=K`), so the level-2 plan is key → door → exit from the first step.
2. **DOOR-FROM-HISTORY** (`World.door_hypothesis`): a colour that was passable earlier in
   the episode and blocks now, while a consumable colour is known, is a door for that key —
   no interact tests on it. **DOOR-WHEN-KEY-KNOWN**: bumping a rare, non-toggle colour
   while a key colour is known → door hypothesis first (interact test only if refuted).
   Refutations (`door_refuted`): passable without the key, or still blocked while holding it.
3. **KEY-UNDER-FEET** (`World.ingest`): stepping onto a cell whose colour is already known
   consumable adds the key immediately instead of waiting to step off.
4. **Toggle planning fixes**: a cell still showing a toggle colour is always solid (cleared
   cells turn black) → always charged interact+move, never walked into; and toggle cells
   no longer get the extra "untested colour" surcharge (the double penalty that made the
   planner prefer the bottom-row detour).
5. **VOI tie → experiment** (`detour >= exp_cost`): an interact test that ties with the
   detour is worth it because the answer is reused in level 2.

## Local results (Astra raising = pure scripted; 300 seeds)

Per-level overhead vs. oracle, 0003: teleport +3.9 / **+0.0**, mirror +2.8 / **+0.0**,
key_door +2.8 / **+0.1**, toggle +4.1 / **+0.0**. Level 2 is at the oracle everywhere.

| config (8×8, 2 levels) | 0002 | 0003 |
|---|---|---|
| 40/level, mean score / actions per episode: key_door | 2.00 / 29.4 | 2.00 / **23.2** |
| 40/level: toggle | 2.00 / 28.0 | 2.00 / **26.2** |
| 40/level: teleport, mirror | 2.00 / 13.8, 22.8 | unchanged |
| 20/level, all families | 1.917 (key_door 1.63) | **1.993** (key_door 1.97) |
| 14/level, all families | 1.377 (toggle 0.42) | **1.547** (toggle 0.71, key_door 1.77) |

Eval seeds 0/1 and held-out 100/101/102: 2.0 each, 0 System-2 calls (seed 100 toggle
33→29 actions, seed 102 toggle 29→28). Mocked Astra with the engine forced "stuck" twice:
`PLAN:` line parsed from the end of a commentary reply, `RETRY_BLOCKED` honoured, 2 calls,
episode still 2.0 — the unchanged advisor path still works after the edits.

## Predictions / what would refute this

* Confirm: evaluator mean 2.0 with 0 S2 calls (ties 0002 on both ranked metrics); `steps`
  in eval.json ≤ 0002's on any key_door/toggle seed; the trace's `events` contain
  `level start: colour c was only walkable while holding key k -> door hypothesis`
  on key_door seeds and no `experiment: interact` in level 2.
* Refute: a key_door seed where the level-start door hypothesis is wrong (`door hypothesis
  refuted` / `dropped` events followed by a long level 2) — that would mean a rare colour
  was entered only while holding a key by coincidence; the fix would be to require the
  colour to be adjacent to the exit or to have ≥2 entries.
* What remains is level-1 discovery (+2.8 to +4.1 per family). The next candidate should
  attack that with experimental design: e.g. choose the first unknown action so that the
  outcome distinguishes the most hypotheses *and* moves toward the exit, and, for toggle,
  weigh the interact test by the prior that a rare colour ringing the exit is a mechanism.
  Cross-seed memory can only help if the evaluator runs more than 2 seeds.
