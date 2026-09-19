# Candidate 0003 — plan-and-commit: verified multi-tick routes, Jev picks between *plans*

Parent: `0002` (best, mean 24.0, 0% stale, 5 S2 calls).

## Diagnosis (from 0002's traces)

All three episodes of 0002 survive to the 300-tick timeout (`death: timeout`, lengths 20/28/33),
so the score is bounded by **ticks per food**, not by dying. Food-to-food gaps per seed:

* seed 0: `[3,3,16,16,10,2,3,4,9,4,7,11,21,18,10,10,14]` (17 food, mean 17.6 ticks/food)
* seed 1: `[…,10,1,70,19,5,…]` — **70 ticks (152→222) without eating**
* seed 2: `[…,22,16,15,19,…]`

The 70-tick gap: food at the corner `(0,0)`, snake length 20. At tick 164 the option table says
`down: 17 steps, right: 17 steps` although the food is 2 cells above the head; at 188 the snake has
folded itself into a 3-wide box (`free_cells_after: 7`); at 206 both options are `tail CUT OFF`. The
cause is structural: 0002 decides *one step at a time* from a `(steps_to_food, free_cells,
tail_reachable)` table with an `area >= 2*len` filter, so near a corner or behind its own body the
greedy step and the filter fight each other and the snake circles. Nothing in that design ever
verifies that the whole route to the food (and the state after eating) is fine.

Jev's per-tick role added nothing: of 310 `jev_pick` events, 304 chose a move with the same
`steps_to_food` as the fallback, 0 were better and 6 were *worse* — seed 0 ticks 171–175 Jev refused
food 0 steps away three ticks in a row because Astra's rule said "length ≥ 20 → most free cells".
It cost ~150 S1 calls/episode with 12–36 late answers each.

## Hypothesis

A harness that (i) plans a **complete route** to the food with a time-aware BFS, (ii) **simulates
the route exactly** and accepts it only if the tail stays reachable after eating, (iii) **commits**
to it (the world is deterministic until the food is eaten, so no model call is needed mid-route),
and (iv) wanders safely when no verified route exists, will eat at close to the shortest-path rate
(≈ 9–10 ticks/food ⇒ ~30 food in 300 ticks) with zero deaths. The right unit of decision for the
slow/fast split is therefore the *plan*, not the *move*: Jev is asked once per food (pipelined on
the predicted next state as in 0002, so its 0.6 s latency never matters) to choose between
Pareto-distinct verified routes (length / free area after eating / body compactness), and Astra
supplies, once per evaluation, the rules Jev reads plus two harness parameters (`DEFAULT` plan
preference for late answers, `TOLERANCE` = extra steps a roomier route may cost).

Offline sanity run (deterministic game, no models / mock Jev): seeds 0–5 score 28–34 (0002 got
17/25/30 on seeds 0–2), every episode times out alive, 0–3 wander ticks, 1–4 Jev asks per episode;
a *random* Jev changes the score by at most ±2 (TOLERANCE 0 ⇒ Jev only chooses among equal-length
routes by default), so the model bridge cannot destroy the planner's gain.

## What changed vs. the parent

* `Sim.plans`: for every legal first move and two BFS tie-break orders (vertical-first /
  horizontal-first ⇒ routes through different regions), a time-aware shortest route to the food,
  simulated exactly with a replica of `SnakeGame.tick`; each plan carries `length`, `tail_ok`
  (time-aware BFS head→tail after eating), `area` (flood fill after eating) and `perim` (free cells
  touching the body — compactness).
* `_offer`: safe = `tail_ok` (fallback: `area ≥ 2·len`), within `best + TOLERANCE`, Pareto-pruned,
  max 3. `_fallback`: shortest (then area, compactness) or, if Astra says `DEFAULT: safest`, the
  roomiest.
* Commitment: the chosen plan's cells are followed tick by tick; re-planned only if reality
  diverges (never happens except by eating). Every tick posts an action computed by the simulator
  before anything else ⇒ 0 stale ticks.
* Wander mode (no verified route): choose the move after which a verified route exists next tick,
  else the one keeping the tail reachable with the most room, else most room.
* Jev: `Choice` over `plan_i` with criteria = the numbers of each route; asked only when a plan was
  just created from the fallback and the predicted next state still offers ≥ 2 distinct plans.
  Late-decision policy: fallback plan is already posted; Jev's pick replaces it at the next tick if
  it arrives before `boundary + 0.42 s` and the predicted state matched. ⇒ S1 calls drop from ~150
  to a handful per episode.
* Astra: 1 call per evaluation (cached in `ctx.memory`: rules, `DEFAULT`, `TOLERANCE`), plus at
  most one in-episode call if the snake has wandered 40 ticks. Expected S2 total ≈ 1 (vs 5).

## Evidence that would confirm / refute

* Confirm: mean score ≥ 28 with all deaths `timeout`; `episode_summary.wander_ticks` ≈ 0;
  `plans` ≈ score + 1; stale 0%; S2 calls ≈ 1.
* Refute (planner): deaths by `self`, or `wander_ticks` large → the tail-reachability check is
  insufficient for long snakes; next step: require `area ≥ len` in addition, or a longest-path
  tail-follow in wander mode.
* Refute (bridge): `jev_plan` events where `chosen_first != fallback_first` followed by longer
  food gaps than the fallback would have produced → Jev's plan choice is noise; keep Jev but set
  `TOLERANCE` to 0 permanently and use Jev only for equal-length tie-breaks (already the default),
  or drop Jev to 0 calls. If Astra's `TOLERANCE`/`DEFAULT` never matter, Astra can go to 0 calls
  (tie-breaker metric) in the next iteration.
* Ceiling check: if ticks/food ≈ mean Manhattan distance (≈ 8–9) the harness is at the
  shortest-path ceiling and further gains need Astra-level foresight (e.g. preferring routes that
  end near the board centre — note that all routes end at the food, so this needs multi-food
  lookahead, which this harness does not attempt).
