# Candidate 0002 — predict-ahead, consequence-annotated options

Parent: `0000-tier2_jev_only` (best, mean 0.0). Also read `0001-tier2_astra_only`.

## Diagnosis (from the traces)

Both parents score 0 and die on the wall within 6–41 ticks, for two concrete reasons.

1. **Decisions are applied one tick late.** Jev latency is 0.57–0.63 s (p50 0.57–0.61, max 0.73)
   against a 0.5 s tick, so every answer computed on state *k* is applied at transition *k+1→k+2*.
   `trace_seed1` of 0000, last calls: state tick 39 (`head [2,1] heading left`) → Jev answers
   `down`; state tick 40 (`head [2,0] heading up, danger up=True`) → Jev answers `right` (0.71) —
   but the game log shows `tick 40 action=up`, `tick 41 action=down`; `down` is a reverse at
   heading `up`, is ignored, and the snake runs into the wall. The correct answer (`right`)
   arrived after death. Seed 2 dies the same way (`left` at 0.75 arrives after tick 8).
2. **Jev on the raw observation is a coin flip that ignores the food.** Confidences are 0.41–0.49
   for almost every call; seed 1 at `head [6,5] food_delta [-4,+5]` (food below-left) it picks `up`
   (0.52) vs `left` (0.44) and zig-zags up-left to the top wall. It never uses `danger`.
3. 0001 confirms Astra (2–3 s/call) cannot be in the per-tick loop: 81% stale ticks.
4. A runner detail matters: the wall-mode ticker starts at `ticker.start()` and is *not* aligned to
   absolute multiples of `tick_s`, so `ctx.wait_for_tick()` is off-phase. The reference harness
   relies on it; this harness instead detects boundaries by polling `obs['tick']` (4 ms).

## Hypothesis

Score is limited by (a) latency-induced staleness and (b) asking a zero-shot classifier a question
it cannot answer from a raw grid. If the harness (i) removes every fatal/trapping move itself with
an exact one-step simulator, (ii) asks Jev about the **predicted next state** so its 0.6 s answer
lands in the tick window it is meant for, and (iii) phrases the options as concrete consequences
("food 3 steps away; 96 free cells afterwards; tail reachable") with Astra's rules as context, the
snake should stop dying early and start eating (expected mean score ≫ 0, likely 10–30), with
~0% stale ticks and ≤2 Astra calls per evaluation.

## What changed vs. the parent

* **Harness-side simulator** (`Sim`): exact replica of `SnakeGame.tick`; for each non-fatal heading
  computes time-aware BFS steps-to-food, flood-fill free area, tail reachability. Deterministic
  fallback (`heuristic`) ranks options by (safe-enough, distance, area) — or (safe-enough, area,
  distance) if Astra says `DEFAULT: safest`.
* **Pipelined Jev**: after posting the action for tick *k*, predict state *k+1*, compute its option
  table, and if ≥2 **non-dominated** options remain (distance-vs-space trade-off, within +6 steps of
  best) spawn a Jev call on the *predicted* state. At tick *k+1* the fallback is posted immediately
  (never stale), then the harness waits for the Jev future until `boundary + 0.42 s`; if the
  prediction matched reality and the pick is among the offered options it overwrites the fallback.
  Late or mismatched answers are dropped and counted (`jev_late`, `pred_mismatch` in the trace).
* **Jev state/question**: `planner_rules`, head/heading/length/food_delta, an `options` dict with
  the numbers, the predicted ASCII grid; one `Choice` whose criteria are the per-move consequences.
  Jev is only ever asked when a real trade-off exists → far fewer S1 calls than one per tick.
* **Astra**: one background call at the first episode if `ctx.memory` has no rules (rules + `DEFAULT`
  mode are cached across seeds → later seeds use 0 S2 calls), plus a background re-plan when the
  harness detects a trap (no move keeps the tail reachable), rate-limited to one per 25 ticks and
  ≤4 per episode.

## Evidence that would confirm / refute

* Confirm: mean score well above 0 with death mostly by `timeout`/`self` late in the episode;
  stale fraction ≈ 0; `episode_summary` shows `jev_used` ≫ `jev_late` (pipelining works) and
  `pred_mismatch` ≈ number of food pickups (the only unpredictable transition).
* Refute (latency): `jev_late` ≫ `jev_used` → Jev is too slow even with a full tick of lead;
  next step would be two alternating Jev workers or asking Jev for 2-tick plans.
* Refute (Jev value): compare `jev_pick` events where `pick != fallback` against subsequent score /
  death — if Jev's overrides correlate with earlier deaths, Jev adds noise and the next iteration
  should give it a narrower role (e.g., Score questions on "how cramped is this region") or a
  probability threshold before overriding the heuristic.
* Refute (Astra value): if `mode` never changes from `shortest` and rules do not alter Jev's picks,
  Astra could be dropped to 0 calls (metric tie-breaker) or given a structured role (parameter
  setting for the heuristic) instead of prose.
