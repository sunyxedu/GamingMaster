# Candidate 0004 — one-step-lookahead value ("expected detour to the next food"); Astra designs the objective, Jev arbitrates close calls

Parent: `0003` (best, mean 32.0, 0 % stale, 1 S2 call, 26 S1 calls).

## Diagnosis (from 0003's traces + an offline replay of its planner)

0003 never dies (all 3 episodes `death: timeout`, lengths 34/34/37, `wander_ticks: 0`), so the score is
`300 / ticks-per-food`. Replaying 0003's planner offline (deterministic `SnakeGame`, no models, fallback
plan only) on seeds 0–7 gives 31.25, and per food the gap decomposes as
`Manhattan(head, food at spawn) + slack`, with **mean slack only 0.4–1.3 ticks/food**:

```
seed 0 slack [0,0,0,0,0,0,0,0,0,0,0,0,4,2,0,0,0,0,0,2,2,0,0,0,14,0,0,0,0,0,0]
seed 2 slack [0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,0,0,2,0,4,0,2,0,2,20,0,4,0,0,6,0]
seed 4 slack [...,0,6,0,16,4,4]      seed 6 slack [...,0,14,0,0,0]
```

i.e. the planner is at the shortest-path ceiling (expected Manhattan between consecutive foods ≈ 8.7
⇒ ceiling ≈ 34.5) and the remaining loss is **one or two huge detours per episode**, always of the
same kind: the food spawns in a region that the snake's own body sealed off a few foods earlier.
Seed 0, tick 189 (length 25): head `(6,11)` heading right, food `(8,11)` two cells away, but the row
above is the body (`......oooooo`) so the only route is a 1-wide dead-end pocket (`tail_ok False,
area 3`); the snake wanders 25 ticks. Tick 214: the body spans row 7 from x=2 to the right wall, food
at `(10,2)` (Manhattan 10) costs 22 steps because the whole top half is sealed. Nothing in 0003's plan
features (`length / area / perim`) distinguishes "the board is split by my body" from an open board —
`area` after eating is ~118 in both cases — so neither the fallback ranking nor Jev's rule-following
(all 26 `jev_plan` events in 0003 chose between plans whose numbers differed by ≤ 2 steps / ≤ 14 cells)
could prevent it. Astra's `TOLERANCE` was also moot: plans for the 3 first moves only ever differ by
0 or 2 steps (offline: `len_spread {0: 25, 2: 18}` over 43 sampled states), so any tolerance ≥ 2 is the
same as ∞.

## Hypothesis

The right one-step value of the state *after eating* is the **expected extra distance to the next
food**: since the next food is uniform over free cells,

    detour(state) = mean over free cells c of [ BFS_time_aware(head → c) − Manhattan(head, c) ],
                    unreachable c counted as PENALTY

is literally the expected number of wasted ticks on the next food. It is exact, one BFS per candidate
plan (≈ 3 ms per tick in total), and it directly measures board-splitting and pockets. Ranking plans by
`cost = steps + LAMBDA·detour` (and, in wander mode, preferring the move with the lowest detour) should
remove part of the big-detour tail without giving up shortest-path speed.

Offline replay (mock, no models), 16 seeds:

| planner | seeds 0–7 | seeds 8–15 |
|---|---:|---:|
| 0003 (length, −area, perim) | 31.25 | 32.00 |
| 0004 value (steps + 1.0·detour, penalty 20) | **32.375** | **32.875** |

Sweeps over LAMBDA ∈ {0.5,1,2}, PENALTY ∈ {20,40}, TOL ∈ {2..8}, 4 BFS orders, a time-expanded
"waiting" BFS and body-hugging Dijkstra routes all produced *identical* trajectories: the value only
ever changes the choice among **equal-length** routes (detour differences are < 1 tick, so a longer
route never wins). So the honest expectation is ≈ +1 food/episode, not a jump; the leaderboard is
near the ceiling for single-food planning and later iterations should not expect more from route
choice alone (the remaining ~2 ticks/food of Manhattan-plus-slack needs multi-food foresight, which is
impossible without predicting the RNG — deliberately *not* exploited here: the seed could be
brute-forced from the first food position, but that is a leak, not a harness design).

## The Astra / Jev bridge under test

* **Astra designs the objective, not the move.** One call per evaluation (cached in `ctx.memory`),
  at most one more if the snake wanders 40 ticks. It returns `LAMBDA`, `PENALTY`, `MARGIN`, `TOL`
  plus ≤ 4 rules for Jev; each number is clamped to a range verified offline (`lambda 0.5–3`,
  `penalty 5–40`, `margin 0–3`, `tol 0–6`) so Astra can tune but cannot break the planner. Expected
  S2 total = 1 (same as 0003).
* **Jev arbitrates only genuine close calls**: it is asked (pipelined on the predicted next state, as
  in 0003) only when the runner-up plan's cost is within `MARGIN` of the best and the routes differ;
  it sees per-plan `steps`, `expected_detour_next_food`, `free_cells_after_eating`, `tail_reachable`
  and the harness `cost`, and Astra's rules say to minimise expected detour. The value-optimal plan
  is always posted first; Jev's pick replaces it at the next tick if it arrives before
  `boundary + 0.42 s`. Mock runs with a *random* Jev: 31.1–31.6 on seeds 0–7 (vs 32.375 without),
  ≈ 20 calls/episode at MARGIN 0.5 — i.e. the bridge can lose at most ~1 food if Jev were noise,
  and 0003's traces show Jev follows the fallback ~75 % of the time, so the expected cost is small.
* Commitment, late-decision policy, wander mode and "post a simulator action before any model work"
  are inherited from 0003 ⇒ 0 stale ticks.

## What changed vs. the parent (code)

* `Sim.dist_map` (time-aware BFS returning all distances, early exit for a target) replaces
  `bfs_path`'s internals; `Sim.expected_detour` is new; `perimeter` is gone.
* `Sim.plans(..., penalty)` annotates every plan with `detour`; `_offer` sorts by
  `steps + LAMBDA·detour` (then steps, then area) and returns the top 3.
* `_close_call` gates Jev; `_wander_move` uses `−detour` as its third key.
* Astra prompt/parse: `LAMBDA / PENALTY / MARGIN / TOL` instead of `DEFAULT / TOLERANCE`; params are
  clamped and persisted in `ctx.memory["params"]`.
* `episode_summary` logs `close_calls`, `chosen_detours` (the detour of every committed plan).

## Evidence that would confirm / refute

* Confirm: mean ≥ 33 with all deaths `timeout`; the per-seed gap lists show no slack ≥ 14; `wander_ticks`
  ≈ 0; stale 0 %; S2 = 1.
* Refute (value): score ≤ 32 and the big detours are still there → the pocket/wall is built more than
  one food earlier than the value can see; next step is a 2-food lookahead (sample ~10 random next-food
  cells, plan to each, average their detours) which costs ~30 ms/tick and is still affordable.
* Refute (bridge): `jev_plan` events with `chosen_first != fallback_first` followed by longer gaps than
  the fallback plan would have produced (compare `chosen_detours`) → set `MARGIN` to 0 permanently
  (Jev off) and keep Astra only for the weights; if Astra's numbers never leave the defaults, S2 can go
  to 0 for the tie-breaker metric.
* Ceiling check: if ticks/food − Manhattan ≈ 0.5 on every seed, the single-food ceiling (~34) is
  reached and further gains need multi-food foresight.
