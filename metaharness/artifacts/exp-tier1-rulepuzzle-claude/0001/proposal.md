# Candidate 0001 — scripted hypothesis engine, Astra only when stuck

## Diagnosis of parent `0000-tier1_astra_naive` (score 0.000, 32 S2 calls)

Evidence from `trace_seed0.jsonl` / `eval.json`:

1. **Reply parsing desynchronised Astra from the game.** The harness took the
   *first* `ACTION[1-6]` token in the reply. Astra answers commentary-first:
   call 2 replied `"ACTION1 moved left. Now at (1,0). Try ACTION2.\n\nACTION2"`
   and the harness executed **ACTION1** (step 2 in the trace); call 3 said
   "ACTION2 did nothing … ACTION3" and the harness executed ACTION2. From the
   second step on, Astra's model of what had been tried was wrong.
2. **Budget starvation.** One action per call × 15 calls = 15 of the 80
   available actions. Both episodes end with `actions_used: 15` after a 16th
   call raised `BudgetExceeded`; discovery of the move map alone ate ~6 calls,
   and the agent never got near the exit.
3. Nothing was stored: no map, no action→direction table, no record of which
   colours had blocked movement; every call re-reasoned from a raw grid.

## Hypothesis

Everything this game asks for — learning which action moves where, which
colours block, and reacting to keys / toggles / teleports — is a small
*experimental-design + search* problem that a scripted System-0 can do at
~1 action per bit of information. Astra's value is not in choosing single
steps but in repairing a wrong belief when the search engine has exhausted
its own strategies. So: **move the control loop into code, make beliefs
explicit and refutable, and spend Astra calls only on "stuck" states.**

## What the harness does

* **Action semantics by elimination.** `ACTION1..4` are move candidates with
  a candidate-direction set each. An unknown action is only tried when the
  planner *wants* a direction (`action_for(d)`), so every exploratory action
  is also a planning step. A displacement confirms the direction (and
  removes it from the other actions' candidate sets); a non-move eliminates
  directions whose neighbour is certainly passable; a non-unit displacement
  is recorded as a teleport with its twin. `ACTION5/6` are interact
  candidates, tested only when standing next to a colour that blocked us.
* **Colour beliefs.** `passable`, `blocked` (soft, with bump counts),
  `consumable` (tile vanished under us → key-like), `unlocks`
  (blocked colour became passable after a consumable was taken),
  `toggleable` + `interact` (grid diff after an interact candidate),
  `tele_colours` + twin pairs.
* **Planner.** Dijkstra over `(pos, keys)` with a surcharge for untested
  colours, key pickup as state, teleport landing, and "interact then move"
  edges for toggleable colours. Executed one step at a time and replanned.
* **Refutation.** Blocked-colour hypotheses are *conditional on the world*:
  any world event (key consumed, toggle, teleport discovered) clears all
  soft-blocked colours (except those bumped ≥3 times) so they get re-tested.
  A confirmed move that fails into an empty cell unlearns that mapping.
* **Stuck ladder.** No path → try untested interact next to the blocking
  colour (≤2 actions) → walk to the nearest unvisited special tile (could be
  key or teleport) → **ask Astra** → fallback (forget blocked colours, then a
  random move).
* **Astra prompt/parse.** Coordinate-indexed ASCII map, legend, learned
  facts, last 12 transitions, discoveries; the reply must end with
  `PLAN: ACTIONx …` (≤10 actions) parsed from the *last* such line, with a
  fallback to the last line that contains action tokens — never the first
  token. Optional `RETRY_BLOCKED` resets blocked beliefs. Session id is reused.
* **Memory.** `colour_stats` across seeds are used only as a *prior* (a
  colour blocked ≥3 times and never passable starts as soft-blocked) — never
  as truth, since key/door/teleport colours are re-drawn per seed.

## Local smoke test (no Astra, `think` raising)

60 seeds of `RulePuzzleGame(size=8, n_levels=2, budget_per_level=40)`:
mean score **2.0** across all four families (teleport 22, toggle 19,
key_door 10, mirror 9 seeds), ~26 actions per episode, 0 System-2 calls.
With a mocked Astra and the engine forced to report "stuck" twice, the
advice path executed the parsed plan and the episode still scored 2.0.

## Predictions / what would refute this

* Confirm: mean score ≈ 2.0 with ≤1 S2 call per episode in the evaluation.
* Refute: seeds where the engine reports `stuck` and Astra's plan does not
  help (check `harness` events with `stuck: true` and `astra_plan` in the
  traces). That would point to a family/geometry the belief model does not
  cover (e.g. exit reachable only via a diagonal-adjacent toggle) and the
  next candidate should extend the planner rather than the prompt.
* If the evaluator's game kwargs differ (bigger grid, more levels), the
  action budget per level is the thing to watch: `actions_used` vs. levels.
