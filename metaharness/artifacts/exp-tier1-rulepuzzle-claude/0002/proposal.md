# Candidate 0002 — value-of-information experiments on top of 0001's belief engine

## Where the parent stands (evidence)

`0001` is already perfect on the evaluation config: local sweep of 300 seeds of
`RulePuzzleGame(size=8, n_levels=2, budget_per_level=40)` with Astra raising →
mean **2.0** on all four families, 0 System-2 calls, heldout seeds 100–102 also
2.0. So the primary metric and the tiebreaker are both saturated; the only
thing left to improve is *action efficiency* — i.e. how much margin the
harness has if the evaluator tightens the budget, enlarges the grid, or a
family needs a long detour. That margin is real: with `budget_per_level=20`
0001 falls to 1.74 (key_door 1.41, toggle 1.54) and at 16 to 1.37.

Concrete waste, from 0001 traces at the eval config:

* toggle seed 15: at (6,4) it bumps the toggle colour (`ACTION4=down bumped
  colour 6`) and then walks a 13-move detour along the bottom row instead of
  spending ≤2 actions on ACTION5/ACTION6 while standing next to the tile.
  Level 1 took 31 actions; the optimistic route was 12.
* key_door seed 38: bumps the door (`bumped colour 3`), never considers the
  key tile, detours around the 3-cell door ring: 23 actions for a level whose
  route through the door is ~12.
* Every level starts with 4 discovery moves that net to zero displacement.

Root cause: 0001 only runs an experiment (interact test, visit a special tile)
when it has **no** path at all. Whenever a detour exists, however long, it
walks it. Information is free only when you are stuck.

## Hypothesis

Treat every experiment as a purchase with a known price and compare it with
the price of ignorance: each step compute

    belief path   = Dijkstra with current beliefs
    optimistic    = same, with soft-blocked colours assumed open (+2 per such cell)
    detour        = len(belief) - len(optimistic)

and if `detour > cost(experiment)` run the experiment instead. Experiments,
cheapest first, are refutable hypotheses:

1. **interact test** next to the first blocked colour on the optimistic route
   (cost = untested interact actions, ≤2; charged +4 if the colour is
   *frequent* on the grid (>4 cells, probably plain wall), +8 if cross-seed
   memory says interact never affected it). Refuted by "no grid change";
   confirmed by a diff → toggleable.
2. **key hypothesis**: walk to an unvisited rare special tile when
   `d(pos,tile) + d_opt(tile,exit) + 2 < len(belief)`. Refuted if the tile
   does not vanish (it becomes visited and is never retried).
3. otherwise follow the belief path; the 0001 stuck-ladder and Astra
   stuck-advisor are unchanged behind that.

Supporting changes that the sweeps forced:

* **Curiosity prior**: an unvisited, never-bumped *rare* colour costs 0.6 to
  enter (instead of 1+2) so the planner picks up keys/teleports it passes
  within a step of, instead of avoiding every unknown colour.
* **Unlock hypothesis at pickup**: when a key-like tile vanishes, every
  *rare* colour bumped this episode is hypothesised to be its door
  (`unlocks[D]=K`) so the next level plans key→door directly. Refuted by
  bumping D while holding K, or passing D without K. (Applying it to all
  bumped colours — including grey walls — cost 0.08 levels at budget 20;
  restricting to rare colours fixed that.)
* **Unknown-direction cost** +0.5 per step whose direction has no confirmed
  action yet, so ties do not go to routes that need discovery moves.
* **Loop guard** (a latent 0001 bug that curiosity exposed): if the chosen
  unknown action bumps an unknown colour, nothing is eliminated and 0001
  repeats the same action forever (seed 55 at 8×8: 38× `ACTION2 no move` at
  (1,3)). Now `(action, pos)` failures are remembered and never repeated; a
  4-step no-move streak resets soft-blocked beliefs and takes a known move.
* Fixed a double penalty (assumed-open cell charged 1+2 and then +2 again)
  that made the optimistic planner walk around through unknown directions
  instead of testing the adjacent toggle tile (toggle seed 91).
* Cross-seed memory now also records `special` (consumable/toggle/teleport/
  door) and `inert` (interact tried twice, no effect) per colour; the wall
  prior never applies to a colour that was ever special, and key candidates
  are excluded only for colours *bumped this episode*, not prior-only ones.

## What changed vs. parent — summary

Same belief model, planner, refutation-by-world-event, Astra prompt/PLAN
parsing and stuck ladder as 0001. New: `scripted_step` runs the VOI
comparison before following the belief path; planner gets `assume_open`,
`start_pos`, curiosity and unknown-direction costs; `World` gets
`bumped`, `failed_at`, `count_colour/is_rare`, `colours_on_path`,
`special_tiles`; unlock hypotheses at pickup; memory `special`/`inert`.

## Local results (Astra raising, i.e. pure scripted; mean levels / actions per episode)

| config | 0001 | 0002 |
|---|---|---|
| 8×8, 2 levels, 40/level, seeds 0–299 | 2.00 / 26.3 | **2.00 / 23.4** |
| heldout 100–102 | 2.00 / 31.3 | 2.00 / 28.7 |
| 8×8, 20/level, 200 seeds | 1.74 | **1.925** (toggle 1.54→2.0, key_door 1.41→1.63) |
| 8×8, 16/level, 200 seeds | 1.365 | **1.675** |
| 12×12, 3 levels, 40/level, 100 seeds | 2.90 | 2.91 |
| 10×10, 3 levels, 60/level, 100 seeds | 3.00 / 47.5 | 3.00 / 42.8 |

Mocked Astra (engine forced "stuck" twice, reply with commentary then a final
`PLAN:` line): plan parsed and executed, 2 S2 calls, episode still 2.0.

## Predictions / what would refute this

* Confirm: mean 2.0 with 0 S2 calls on the evaluator, and `steps` in
  eval.json ≈ 23 per episode instead of 22–31.
* Refute: any eval-config seed below 2.0 — look for `experiment` events
  followed by a level timeout; that means an experiment's price was
  under-estimated (most likely the key hypothesis on a far tile).
* Remaining known weakness: key_door under very tight budgets (≤20/level) —
  the level-1 sequence "discover moves → bump door → 2 interact tries → fetch
  key" is inherently ~6 actions of overhead. The next candidate could use
  cross-seed memory more aggressively (skip interact tests on colour classes
  memory has seen as doors, try interact *first* on colour classes seen as
  toggles) — but only if the evaluator runs enough seeds for memory to matter.
