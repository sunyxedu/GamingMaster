# 0004 — "every tick is a bridge tick": close the stale-tick hole, break limit cycles

Parent: `0003` (best so far, mean 9.33; compiled Astra bridge harvested across episodes). Same
architecture — Astra writes `bridge(obs) → options / utility / state` as code, the harness posts the
argmax, Jev arbitrates — with the control loop rebuilt so that the bridge runs on **every** tick, plus a
generic stall detector and a module scoreboard.

## Diagnosis of the parent (evidence from traces)

**Both deaths happened on a STALE tick.**

* seed 0 (score 17): `TICK 203 up fresh (score 16→17)` — the food sat against the top wall
  (`situation: head [1,1], heading up, "eats food, open 126, safe"`) — then `TICK 204 up STALE → death: wall`.
* seed 2 (score 7): `DEC t=64 best=down "eats food, open 136, must turn next tick"` → `TICK 74 down fresh
  (score 7)` → `TICK 75 down STALE → death: self`.

0003's loop was *bridge → act → synchronous `ctx.decide` (~0.58 s) → loop*. With a 0.5 s tick that is one
decision per ~1.16 ticks, so ~13 % of ticks (22/204, 42/300, 11/75) re-applied the previous action with
the bridge never having looked at the new state. The bridge only guarantees "non-fatal on the next tick";
the tick after that is exactly where a move that ended facing a wall/body kills — i.e. right after eating
food that sits against an obstacle. Astra's own post-mortem said it: *"the turn did not land in time"*,
and it tried to fix that in the *module* (run-length penalties) when it was a harness plumbing bug.

**seed 1 (300 ticks, score 4) was a limit cycle, and it was never post-mortemed.** Sampling every 12th
Jev call shows the same situations recurring for ~250 ticks: `[9,x] down "away from food"` →
`[7,2]/[8,2] right "closer to food"` → `[7,5] up` → `[7,9]/[8,9] left` → `[9,x] down` … with `food_dist`
oscillating 4–13 and no food eaten after tick ~50. Jev was split (`down 0.41 / left 0.47`) at the turning
points, so its blended vote decided the cycle. Because the episode reached `max_ticks` it counted as
`clean_end`, so no `improve` round ran and the wasted 250 ticks taught Astra nothing.

## Hypothesis

1. The score ceiling of the compiled-bridge design is set by the *harness*, not the module: the bridge is
   already good enough to survive hundreds of ticks (seed 0: 204 ticks, 17 food) and dies only where the
   harness stops running it. A loop that never lets a slow call sit between two ticks should turn both
   stale-tick deaths into survivals (→ 0 % stale, and the seed-0-like episodes run to 300 ticks).
2. Jev is useful only on near-ties; letting a 1-tick-old probability vector shift a clear argmax (0003
   `JEV_WEIGHT = 0.5`) is how the seed-1 cycle got closed. Restricting Jev to ties keeps its value (labels
   distinguish "eats food" from "closer to food") without letting it override lookahead.
3. Stalls are a generic, observable failure ("score flat for N ticks") that the harness can both break
   locally (randomise among near-best options) and escalate to Astra with the *right evidence* (a sampled
   trajectory of the stalled stretch) — mid-episode, because the loop no longer blocks.

## What changed vs. 0003

| | 0003 | 0004 |
|---|---|---|
| Control loop | bridge → act → **blocking Jev (0.58 s)** → loop; `wait_for_tick` only when no tick passed | **tick-synchronous**: `observe → bridge (µs) → act → wait_for_tick`; nothing slow between ticks |
| Jev in the bridge phase | every tick, synchronous; prob (max-normalised) × 0.5 added to utility → could flip a clear argmax | background future, asked only when the top-2 utilities are within `TIE_MARGIN+0.05`; its ≤2-tick-old preference may override only among options within `TIE_MARGIN = 0.15` of the best |
| Stalls | none | score flat for `STALL_TICKS = 45` → `EXPLORE_TICKS = 6` ticks of random choice among options within 0.25 of the best (repeats every 45 ticks) + one **`unstick` Astra call per episode** with 12 sampled `(state, options, action)` records from the stalled stretch; the revised module is validated and adopted mid-episode |
| Post-mortem trigger | episode not `clean_end` | not clean **or** `max_stall ≥ 45`; the improve prompt carries the sampled stall trajectory |
| Module bookkeeping | latest adopted module wins | **scoreboard** in `ctx.memory["modules"]`: each version keeps its episode scores; an untested revision is tried once, a tested one that scores below the best tested module is reverted at the next episode start (`module_revert` event) |
| Validation | compile + one call | + reject modules taking > 100 ms per call (asks for a repair) |
| Code prompt | — | tells Astra the bridge runs every tick, that a missed tick repeats the action, and that survival terms must be large penalties while progress toward reward is the main graded term (anti-orbit) |

Generic phase (before the first module lands) is unchanged from 0003 (background Jev, 3-tick risk
minimiser, re-affirm the standing action) — it survived 46 ticks in 0003 seed 0. S2 worst case per
evaluation: 1 code + ≤3 repairs + ≤4 revisions (unstick/improve) — still ≤ 8, far below 20/episode.

Nothing names a game, an observation key or an action; the held-out game gets the same treatment.

## Offline verification (wall clock, tick 0.1 s, scripted Astra returning a broken module first, then the
real snake module Astra wrote in 0003; heuristic Jev at 0.3 s)

* snake: seeds 1–2 run 150/150 ticks, 15 and 18 food, **0 stale ticks** (0003's loop: 13 %); seed 0 pays
  the generic phase (10 ticks at 0.1 s is unrealistically short) — stale 1 % overall, 2 S2 calls.
* lanerunner shape (different keys/actions, hand-written stand-in module): 0 stale in bridged episodes,
  end-window `improve` fires after deaths.
* stall path (module edited to *flee* food): `stall` at tick 133 → `unstick` → `bridge_adopted
  (source=unstick)` at tick 143 while the game keeps running → snake resumes eating; the episode is
  post-mortemed (`max_stall 63`), scoreboard shows `[["code", []], ["unstick", [10.0]]]`.

## What would confirm / refute

* **Confirm:** `stale_fraction` ≈ 0 in bridged episodes; no episode whose last tick is `STALE`; seeds with a
  module reach ≥ 200 ticks; mean score > 9.33 (target: 15+). `decision … how="jev"` events rare and not
  followed by deaths.
* **Refute / next steps:**
  - deaths still occur with `fresh` last ticks → the module's lookahead is the real bottleneck; ask Astra
    for an explicit 2-ply / reachable-space term, or let the harness ask the bridge for a "safe if repeated"
    flag and prefer those options;
  - `stall` events with `unstick` adopted but score still flat → the sampled trajectory is not enough for
    Astra; include the ASCII renders of the stalled stretch, or let the harness add a generic novelty bonus
    (hash of the bridge `state` dict) to break cycles without Astra;
  - `module_revert` fires often → Astra's revisions are net-negative; freeze the module after the first
    clean 300-tick episode and spend revisions only on deaths;
  - seed 0 still dies in the generic phase → shrink the first Astra artifact (a 10-line `fatal(obs)` filter)
    so it lands within ~3 s, then request the full module.
