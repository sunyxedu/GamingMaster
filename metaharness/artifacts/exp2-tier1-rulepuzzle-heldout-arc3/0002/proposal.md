# 0002 — object mode for agent-less games + value-level rules + leaner probing

Parent: `0001` (train mean 2.000, 0 S2 calls, 30 actions/episode).

## Diagnosis

**The train game is saturated; the leaderboard tie-break is now efficiency.** 0001 solves both
train seeds with zero Astra calls. `Archive.best()` orders by `mean_score → S2 calls → fewer
mean_steps`, so any successor must keep 2.0 / 0 calls and use *fewer than 30 actions* — otherwise
it is never selected and never reaches the held-out game, which is where the real gap is.

**0001 scores 0 on the held-out game and the failure is structural.** I dry-ran this run's 0001
on `arc3` (`vc33`, `max_actions=150`, stub Astra). Trace (`/tmp/h0001-arc3`):

* `probe_done: move_map {} agent None n_acts 0` — the only available action is `ACTION6 x y`, which
  0001 never probes (`a in w.param_actions: continue`).
* 15 Astra calls whose knowledge table is entirely "never tried" (prompt shows a 64x64 map and
  nothing else) — every reply was unusable, so the harness fell through to `explore_step`, i.e.
  a blind stride sweep `ACTION6 7 3, ACTION6 14 6, ACTION6 21 9, …`.
* The episode ended at step 65 with `state GAME_OVER`: the game has a hidden per-level click
  budget (a meter in row 0 that shrinks one or two cells per click) and blind clicks exhausted it.

A 10-click probe by hand (one click per connected component) shows that the structure is cheap
to expose: two 4x4 yellow blocks each change 265 cells (the second undoes the first), every other
object only ticks the row-0 meter. 0001 had no way to see, store or tell Astra any of this.

**Efficiency waste in 0001 on the train game** (trace_seed0): probe = 6 actions although the four
directions are known after 4; level 2 takes 12 actions where ~5 suffice, because the teleport is
stored as a *cell pair* (`portals`, cleared on level-up) rather than as a rule about the colour.

## Hypothesis

1. Games without a unique moving cell (click/ARC-style) need an **object-level** interface, not a
   cell-level one: segment the frame into connected components, spend one click per object
   *class*, filter out "indicator" cells that change on almost every action, and let Astra plan
   over object ids. This converts a 0-information prompt into an effect table, which is the
   minimum Astra needs to reason about a hidden rule.
2. Rules are episode constants, so a **value-level** hypothesis ("entering colour P moves you to
   the other P cell") should survive level changes and be refuted on first failure; this cuts
   level-2 route length on teleport seeds without touching correctness.
3. Probing non-moving actions is only worth it when stuck; deferring it saves ≥2 actions per
   episode with no loss (the lazy probe fires before Astra is consulted).

## What changed vs. 0001

* **Mode selection**: after the (possibly empty) movement probe, if no unique moving cell / no
  movement map exists → `object_mode`; otherwise the 0001 movement loop.
* **Object mode** (new): `segment()` (4-connected components, background = most common colour),
  `indicator()` (cells changing in ≥60% of actions **or** rows/columns in which almost every
  action changed a few cells — a sweeping meter changes different cells each time), object probe
  (one click per `(colour,w,h)` class, small objects first, background-like regions skipped when
  ≥3 other classes exist; 10 clicks on level 1, 6 afterwards; classes reset per level), effect
  summaries as colour transitions inside a bounding box with indicator cells removed, and a prompt
  with object table, per-class response counts, indicator composition now vs. level start (so a
  budget meter is legible), scalar observation fields, experiment log and a map downsampled by the
  detected pixel zoom (2x for vc33). Reply = JSON `{hypothesis, goal, actions[≤8], expect, lesson}`;
  `parse_obj_plan` accepts `"ACTION6 x y"`, `"obj5"`, `{"obj":5}`, `{"x":..,"y":..}`, `[x,y]` and
  bare `ACTIONk`, and refuses actions not in `available_actions`. Plans stop after two consecutive
  no-change actions or a level change; identical stuck states are not re-asked more than twice;
  with no budget left, scripted exploration rotates through responsive classes.
* **Movement mode**: probe stops once four distinct unit displacements are known; `lazy_probe`
  tries a still-untried action when no route exists (before Astra); `value_portal` rule with
  refutation; the harness rejects actions outside `available_actions` itself.
* **Memory**: adds `class_stats` (which object classes responded — a discovery prior for probe
  ordering on later seeds); rest as in 0001.

## Local evidence (stub Astra that answers `ACTION1`, i.e. useless)

* Eval config (8x8, 2 levels, 40/level), seeds 0,1: **2.0 / 2.0, 0 S2 calls, 28 actions** each
  (0001: 30). Held-out seeds 100,101: 2.0 / 2.0, 0 calls, 38 / 24 actions.
* 40-seed sweep, same config: 0002 mean **1.95** (0001: 1.875), 30 vs 60 stub calls, 32.1 vs 35.3
  actions; toggle family 2.0 (0001: 1.7) because the lazy probe finds the interact action when
  stuck; the two remaining misses are `key_door` seeds that need real reasoning.
* arc3 / vc33 seed 100 (stub): runs to completion without errors; the first prompt already says
  "row 0 is an indicator (7 x54 now, 7 x64 at level start)", lists 9 objects with "[class clicked
  x1: 1 responded]" on exactly the two yellow buttons and the two blue tabs, and shows a 32x32 map.
* arc3 / vc33 seed 100 with the real System-2 (Claude stand-in, same CLI the evaluator uses;
  `/tmp/h0002-arc3-claude`): **4 of 7 levels in 150 actions with 15 calls** (0001: 0 levels,
  GAME_OVER at step 65). Level-ups at steps 16 / 34 / 65 / 95 after 2 / 5 / 9 / 14 calls; the
  episode ended on the action cap (`state NOT_FINISHED`), not on the click meter. Replies were
  well-formed JSON with evolving hypotheses ("the two colour-9 blocks are left/right buttons" →
  "black is a fluid pumped between chambers; each 9 button moves 4 columns across its wall") and
  mixed `obj5` / `ACTION6 39 40` plans that the parser executed. n=1, but the failure mode of the
  parent (no information, blind sweep) is gone.

## What would confirm / refute

* Confirm: train 2.0 / 0 calls / <30 actions (selection), and on the held-out game ≥1 level with
  ≤15 calls, the trace showing `mode: object`, an indicator row and per-class response counts.
* Refute: (a) held-out 0 levels with the meter exhausted mostly by *probe* clicks ⇒ probe budget
  too high for the game's step limit — make it proportional to the indicator's decay rate;
  (b) Astra plans repeatedly `refuted` after one click ⇒ the "two no-change actions" cut-off is
  too aggressive for games with delayed effects — verify only the final state of a plan;
  (c) train regression on key_door/toggle ⇒ the 4-direction early stop hides an action that is
  needed early — probe all actions once when the level budget is generous.

## Ideas for the next iteration

* Track objects across frames (match by colour+shape) and report *motion* ("obj moved (+4,0)")
  instead of colour transitions; this is what a physics/sliding puzzle needs.
* Give the object mode a scripted "undo detector" (click A then B restores the frame ⇒ A/B are a
  toggle pair) so Astra is not asked to rediscover it.
* On the train game, transfer key/door semantics across levels as typed value rules (as done for
  portals) — the two remaining key_door misses.
