# 0003 — "harvest the bridge": make 0002's compiled bridge actually land

Parent: `0002` (best so far, mean 0.667). Same architecture (Astra writes `bridge(obs) → options /
utility / state` as code, Jev arbitrates among labelled non-fatal options), fixed so that the bridge
exists when the game is played.

## Diagnosis of the parent (evidence from traces)

**The bridge never ran.** All three `episode_summary` events of 0002 say `had_bridge: false`; there is
no `bridge_adopted` anywhere. Episodes lasted 3.5–6.6 s wall (ticks 6/11/13, all `death: wall`). The
one Astra reply that ever landed inside an episode was the *guidance* call (5.1 s in seed 0, 4.9 s in
seed 2); the *code* call — longer output, launched at the same instant — was still in flight when the
snake died. At episode end the pending future was simply discarded, so `memory["bridge_code"]` stayed
`None`, seed 1 and seed 2 re-issued both Astra calls (`system2_calls: 2` per seed, `latency_s2.n = 1`)
and got nothing again: **6 System-2 calls, 0 harvested**. 0002's score (two lucky food pickups) came
entirely from its fallback path.

**The fallback path is one tick too late by construction.** Seed 1, Jev generic answers tick by tick:
`end:right = 0.04, 0.04, 0.06, 0.09, 0.16, 0.93`, `end:keep = 0.05, 0.07, 0.09, 0.15, 0.44, 0.93`,
`best = keep ×5`, then `up` — the correct `up` was produced by the call that started at tick 6 with the
head already at x = 11, and Jev's ~0.59 s latency posted it after the wall tick. The Noul question was
"would this action end the game *immediately*?", which Jev only answers *yes* when the wall is
adjacent, i.e. exactly when it is too late for a > 0.5 s classifier.

## Hypothesis

0002's design is right but its *plumbing* wastes Astra: (a) Astra output must survive episode
boundaries — the seconds *after* game over are free, so that is where blocking, validation, repair and
post-mortem improvement belong; (b) seed 0 needs a fallback whose questions tolerate a one-tick-late
answer. If both hold, seeds ≥ 1 should play with the bridge (hundreds of ticks, several food, ~0 stale
ticks), seed 0 should die later than tick 6, and the total Astra bill should drop.

## What changed vs. 0002

| | 0002 | 0003 |
|---|---|---|
| Astra futures | dropped at episode end | kept in `ctx.memory["pending"]`; **at game over the harness blocks (≤ 240 s) on the code → validation → repair chain**, then on the post-mortem *improve* round if the episode was not clean; repairs of a failed improvement fall back to the previous module |
| Astra calls at start | 2 concurrent (guidance + code) | 1 lean call; the module carries `RULES = [...]` (Jev's coach rules) — no second call competing for Astra time; prompt asks for a short module |
| Generic phase question | "ends the game immediately?" + "best action" | "taken now **and held for the next 2–3 ticks**, ends the game?" → risk minimiser: Jev's Choice only breaks ties among actions with p(end) ≤ 0.35 |
| Generic phase scheduling | synchronous Jev (loop blocked 0.6 s) | Jev in a background future; every answer is posted the moment it arrives; the standing action is re-affirmed on ticks in between |
| Bridge phase | unchanged: bridge argmax posted before the Jev call; Jev (≤ 2 ticks old, max-normalised, weight 0.5) blended in; runtime errors ×3 → repair | same |

Worst-case Astra per evaluation: 1 code + 3 repairs + 2 improves (+ repairs) ≈ 6–10, well under 20 per
episode, and every one of them is either adopted or answered by a repair.

## Offline verification (wall clock, tick 0.15 s, 40 ticks, scripted Astra sleeping 3 s and returning a
*broken* module first, heuristic Jev)

* snake: seed 0 dies at tick 8 in the generic phase; end window: `bridge_reject(code)` → `repair` →
  `bridge_adopted` (waited 4.8 s); seeds 1–2 run the full 40 ticks with 4 and 6 food, 1 % stale,
  **2 S2 calls total** (0002 used 6 and never got code).
* lanerunner (held-out shape, different keys/actions): same path, seeds 1–2 driven by its own bridge
  (14.8 / 5.2 points vs 0.3 in the generic phase), improve rounds exercised, 4 S2 calls.

## What would confirm / refute

* **Confirm:** seed 0 trace ends with `end_window … have_bridge: true`; seeds 1–2 show `bridge_adopted`
  absent (cached) or from `improve`, `had_bridge: true`, ≥ 100 ticks, several food, stale ≈ 0;
  total S2 ≤ 6.
* **Refute / next steps:**
  - seeds 1–2 still die early *with* the bridge → the module's lookahead is the bottleneck; feed the
    post-mortem (already collected) with more history, or ask Astra for an explicit flood-fill /
    reachable-space term in `utility`;
  - `jev_changed` correlates with deaths → drop `JEV_WEIGHT` toward 0.25 or restrict Jev to exact ties;
  - seed 0 still dies at tick ≈ 6 → the 3-tick Noul phrasing does not move Jev; the remaining lever
    for seed 0 is to shrink Astra's first artifact (e.g. a 10-line `fatal(obs)` filter before the full
    module) so it lands within ~3 s;
  - repeated `bridge_reject` in the end window → prompt too demanding; simplify the contract to
    `options` only.
