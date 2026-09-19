# 0001 — probe → world model → autopilot; Astra only when stuck

Parent: `0000-tier1_astra_naive` (mean 0.000, 16 S2 calls / episode).

## Diagnosis of the parent (from the traces)

* **Budget death, not rule difficulty.** One Astra call per action: both seeds end at
  `"steps": 15, "system2_calls": 16` — the 16th call raised `BudgetExceeded` with 65 of the
  80 available actions unused. `info` shows both seeds were the *teleport* family with a
  carved corridor to the exit — solvable in ~12 moves once the movement map is known.
* **No learning across calls.** Each prompt is `describe + JSON obs + "reply with one action"`.
  Seed 0 answered `ACTION4` five times in a row and `ACTION3` three times while the grid did
  not change (blocked by a wall) — the model is never told what the previous action did,
  so it cannot refute its own hypothesis. Output was 7 tokens per call (no reasoning).

## Hypothesis

Levels are lost in the *harness*, not in the model: (a) the movement map and passability of
colours can be learned by cheap scripted experiments and diffs, (b) once a goal value and a
movement map are known, a Dijkstra path over "believed passable" cells solves most levels with
**zero** Astra calls, (c) Astra is only needed for the genuinely hidden rules (key→door,
toggle-by-interact) and then needs an ASCII map with coordinates, the learned action table,
colour statistics and recent history — and should return a *multi-action* JSON plan.

## What changed vs. the parent

1. **Scripted probe (0 calls):** try each parameter-less action once; actions with no visible
   effect are retried once from a cell whose neighbours are all passable (position of the
   failure remembered, so "blocked" cannot masquerade as "no effect"). ≈9 actions.
2. **Game-agnostic world model** (`World`): grid = largest list-of-lists (or multi-line
   string) in the observation; legend = any dict of str→str; available actions = any list of
   action names. Agent = the unique value that moves. Per value: walked-on / blocked / jump /
   vanish counters; per action: displacement counter. Refutation rules: expected move but no
   move ⇒ target value blocked (3× ⇒ wall); any non-agent cell change ⇒ passability beliefs
   reset (keys/toggles change the world); a jump records a portal edge for that cell, with the
   symmetric edge as a hypothesis (dropped if entering that cell later does not jump).
3. **Autopilot:** Dijkstra from the agent to any goal cell (portal edges honoured, unknown
   values cost 3, blocked values excluded), executed with per-step verification; any surprise
   ⇒ replan. Goal value comes from (i) memory of earlier seeds (validated on the grid),
   (ii) a legend name found in a "reach/complete/goal" sentence of `game.describe()`,
   (iii) Astra's JSON, (iv) the cell stepped into when the score rose.
4. **Astra only when stuck:** no route / no goal / autopilot loops (state repeated 3×). Prompt =
   describe + action spec + learned action table + colour stats + last 18 events + coordinates
   map + legend + lessons from memory + the concrete reason. Reply = one JSON object
   `{goal_value, actions[≤15], expect, lesson}`; parsed with a JSON-first, regex-fallback
   parser (keeps `ACTION6 x y` arguments). Plans stop early when two consecutive actions change
   nothing or the score changes; invalid action strings are rejected by the harness, not the game.
5. **Memory across seeds:** confirmed goal/agent values, per-action move statistics (probe
   ordering), and Astra's one-line "lesson" strings (≤10). Rules are re-drawn per seed, so only
   priors that are re-validated on the live grid are used.
6. **Budget policy:** the harness never calls Astra when `ctx.budget.remaining()` is 0 and
   never re-asks twice on an identical stuck state; with no budget it explores least-visited
   neighbours so the action budget is always consumed.

## Local evidence (stub Astra returning *random* 8-action plans, size 8 / 2 levels / 40 actions)

* seeds 0–7: mean 1.875 (5 S2 calls total); seeds 8–23: mean 2.0 (8 S2 calls total), all four
  rule families; the only miss (seed 7, key_door) is exactly the case that needs real reasoning.
* A mock held-out game (different keys `frame`/`available_actions`, string cells, 20×20, click
  action `ACTION6 x y`) runs without error and completes both levels.

## What would confirm / refute

* Confirm: mean ≥ 1.5 on the train seeds with ≤ 4 S2 calls per episode; key_door/toggle seeds
  solved after 1–2 Astra plans that mention the key/toggle colour.
* Refute: (a) score stays ~0 with few calls ⇒ the autopilot's passability model is wrong
  (check `blocked`/`hard_wall` notes in the prompts); (b) many calls with `"refuted"` reasons ⇒
  Astra's plans are cut too early (raise the two-consecutive-dud threshold or verify only
  known moves); (c) held-out game scores 0 with errors ⇒ grid/agent discovery too narrow
  (see `find_grid`, `unique_positions`).

## Ideas for the next iteration

* Let Astra name *which colour is the key / door / toggle* and store it as a typed hypothesis
  the autopilot can plan with (e.g. "visit value K before crossing value D"), instead of
  executing its raw action list.
* Ask Astra one "rules summary" question at episode end and feed it as the lesson.
* On the held-out (ARC-like) game, add click-sweep experiments for parameterised actions.
