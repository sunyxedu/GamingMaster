# 0003 — curiosity routing + a call-thrifty object mode

Parent: `0002` (train mean 2.000, 0 S2 calls, 28 actions/episode; 4/7 levels on the held-out
`arc3`/`vc33` dry run with 15 calls).

## Diagnosis

**Selection.** `Archive.best()` ranks by `mean_score → stale → S2 calls → mean_steps` and `max()`
keeps the *earlier* candidate on ties, so 0003 must finish the train seeds in **< 28 actions** with
0 calls or it is never selected for the held-out game.

**Train game: 0002 routes around information.** Both train seeds are the teleport family. The
optimal solutions (BFS with full knowledge, `/tmp/opt.py`) are 6+6 (seed 0) and 5+6 (seed 1)
actions; 0002 spends 4 (probe) + 14 + 10 = 28 on seed 0. Trace evidence (`candidates/0002/
trace_seed0.jsonl`): after the probe the agent walks `ACTION3 x2, ACTION4 x2, ACTION3 x3,
ACTION4 x2, ACTION1, ACTION4 x2` — a 14-step detour under the wall ring to reach the exit from
the right, never touching the two orange (teleport) cells at (4,3)/(5,6) although (4,3) lies on a
shortest path. The cause is `path_to`'s cost of **3 for unknown values**: the Dijkstra treats
"never touched" as "probably bad", so the teleport is discovered only on level 2 by accident.
In a hidden-rule game the opposite is true: a rare, never-touched value is the cheapest
experiment available (keys, teleports, doors, toggles all live in rare colours).

**Held-out game: the scarce resource is the System-2 call, and 0002 wastes both calls and
clicks.** From `/tmp/h0002-arc3-claude/trace_seed100.jsonl` (real Claude stand-in, 150 actions,
15 calls, 4/7 levels):

* After every level-up the harness re-probes 6 objects although all classes were already known
  (`class_stats = {}` on level change): steps 35-40, 66-71, 96-101 are pure re-probe clicks
  (`ACTION6 1 17, 28 41, 8 54, …`) — 18 clicks, 12% of the action budget, zero information.
* Plans were capped at 8 actions while Astra had already computed longer sequences: level 3,
  call 7: *"E down 7 units, D net up 4, A up 6: … (21 presses total)"* → only 8 executed, then
  two more calls to finish what it had already planned. Level 2 took 5 calls, level 3 took 5.
* Astra's hypotheses were not fed back explicitly; it had to re-derive the "moves by 4" unit
  from colour-transition counts (`3->0 x108, 0->3 x104`) each time, because the effect summary
  is cell-level, not object-level.
* The 15th call came at step 130 with 20 actions and 3 levels left.

## Hypothesis

1. **Curiosity routing.** Treat never-touched *rare* values (≤ max(3, 10 % of cells); not the
   agent, not the goal) as targets: before routing to the goal, walk to the nearest one whenever
   `len(path) + manhattan(cell, goal) ≤ len(goal_path) + 4`; also lower the Dijkstra cost of
   unknown values from 3 to 1.5. This turns "discover the rule by accident on level 2" into
   "discover it on level 1 for ~1 extra action", and it is rule-agnostic (keys get picked up,
   doors/walls get bumped once and become `blocked`, teleports fire).
2. **Object classes keep their meaning across levels** in click games, so re-probing is waste:
   keep episode-level `class_stats`, add per-level counts for the prompt, and probe only classes
   never seen this episode (≤ 4 clicks after level 1).
3. **Longer, self-verifying plans save calls.** `MAX_PLAN_OBJ` 8 → 20 with a repeat syntax
   (`"obj5 x6"`, `{"obj":5,"times":6}`), the existing "two consecutive no-change actions" cut-off
   as the safety net, and the prompt states the remaining calls so Astra commits when confident.
4. **Object motion summaries** (`motion_summary`: same-shape objects matched before/after a
   click → `"4 4x6 object at [x46-49,y44-49] MOVED (+4,+0)"`) plus explicit feedback of the
   previous hypothesis/goal/expectation and its outcome make refutation visible in one line.

## What changed vs. 0002 (diff is ~150 lines; everything else is inherited)

* `World.unknown_rare_values()`, `Harness.curiosity_step()` (called in `movement_loop` before
  goal routing; ≤ 2 attempts per (level, value)); `path_to` unknown cost 3 → 1.5.
* Object mode: `level_stats` instead of resetting `class_stats`; per-level probe budget 6 → 4
  and only for unseen classes; `motion_summary()` appended to every effect note; `last_plan`
  feedback block, budget line and repeat syntax in `object_prompt` / `parse_obj_plan`;
  `MAX_PLAN_OBJ = 20`.

## Local evidence (scripted stub Astra that answers `ACTION1`, i.e. useless)

* Eval config (8x8, 2 levels, 40/level), seeds 0,1: **2.0 / 2.0, 0 calls, 16 / 15 actions**
  (0002: 28 / 28). Seed 0 trace: probe ×4, `ACTION3 ×2, ACTION4 ×4` (teleport at (4,3) fires on
  the 4th right move) → level 1 in 6; level 2 in 6 via the value-level portal rule.
* Held-out seeds 100,101 (same game): 2.0 / 2.0, 0 calls, 38 / 24 actions.
* 40-seed sweep (seeds 2-41, all four families): **2.0 mean, 0 calls, 24.5 actions** vs 0002's
  1.95 / 30 calls / 33.1 actions — the two key_door misses of 0002 are solved because curiosity
  walks onto the key before the door is reached.
* arc3 / vc33 seed 100, stub Astra: no errors; the first prompt already contains
  `colour-9 4x4 object … : 264 cells changed …; 4 4x6 object at [x46-49,y44-49] MOVED (+4,+0);
  b 2x6 object … MOVED (+4,+0)`.
* arc3 / vc33 seed 100 with the real System-2 (Claude stand-in): see the addendum below.

## What would confirm / refute

* Confirm: train 2.0 / 0 calls / < 28 actions (selection); held-out ≥ 4 levels with the level-up
  step numbers earlier than 0002's (16 / 34 / 65 / 95) and fewer calls per level; traces showing
  `x<N>` repeat plans executed in full and no probe clicks after level 1 except on new classes.
* Refute: (a) curiosity increases train actions on some family ⇒ the +4 detour slack is too
  loose — make it proportional to the remaining level budget; (b) held-out plans of 20 actions
  repeatedly `refuted` after a few clicks ⇒ Astra over-commits; lower the cap to 12 or require
  an `expect` check at the midpoint; (c) a level whose classes change meaning (e.g. buttons
  become windows) is not re-probed and Astra stalls ⇒ re-probe classes whose *level* response
  rate is 0 after two failed plans.

## Ideas for the next iteration

* Verify plans against Astra's stated expectation (parse `expect` into "object X moves (dx,dy)"
  and stop the plan on the first contradiction rather than after two no-change actions).
* Scripted counting of "units": when the same click repeatedly moves an object by a constant
  vector, let the harness compute how many presses reach a stated target cell.
* An episode-end "rules summary" call is not affordable at 15 calls; instead store Astra's
  hypothesis strings per object class in `ctx.memory` as discovery priors for later seeds.

## Addendum — arc3 / vc33 seed 100 with the real System-2 (Claude stand-in, `/tmp/h0003-arc3-claude`)

* **4 of 7 levels, 150 actions, 14 calls** — same score as 0002's dry run (n=1) but every level
  came earlier and cheaper: level-ups at steps **12 / 26 / 57 / 88** after **2 / 4 / 6 / 9** calls
  (0002: 16 / 34 / 65 / 95 after 2 / 5 / 9 / 14). No probe clicks after level 1 except on
  never-seen classes; three plans of 20 / 18 / 19 actions executed in full without being cut.
* Level 5 was reached with 62 actions and 6 calls left; Astra identified the window/flush rule
  by call 13 but computed that finishing needed 22 more presses with 13 actions left. The limit
  there is the per-level action cost of Astra's *exploratory* plans, not the harness bookkeeping.
* Bug found while reading the last prompt and fixed: 0002's `del w.history[n:]` leaked the
  World's cell-level notes into the object-mode log once the history hit its 60-line cap;
  0003 restores the snapshot instead (re-verified: train 16 / 15 actions, 42-seed sweep 2.0 /
  0 calls / 23.8 actions; arc3 stub run error-free).
