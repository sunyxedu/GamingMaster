# Candidate 0019 — effect-tested heading frontier with contact escape

## Parents and hypothesis

This is a crossover of `candidates/0011/` and `candidates/0015/`. From 0011 I retain the mechanism that actually improved the leaderboard: a committed asynchronous motor, local heading-frontier turns, finite collision/no-novelty escapes, visible-label interaction, and Jev as a hint rather than a blocking motor. From 0015 I retain the bounded moving aim lease, but make it conditional on measured motion instead of blindly issuing four backward diagonals.

**Hypothesis:** 0011's high score is mostly a consequence of continually posting deterministic actions and using heading visits to reduce local geometric repetition. Its remaining failures have two coupled causes. First, the route motor knows that it is revisiting a cell, but not whether the preceding heading/escape actually opened space; it can therefore spend many epochs replaying a geometrically bad branch. Second, the 0011/0015 combat priority is evaluated before the collision escape priority. A close target can keep returning `backward_left/right` or turn actions even when that direction is blocked. The 0015 movement lease changes the button but not the outcome: its seed-1003 trace stayed at `(1134.76,16.04)` while `backward_right` was held for roughly tics 700–846 and health fell 40→3. The 0011 seed-1012 trace has the same fatal signature: around `(752,-304)`, alternating turns and near-zero displacement while health fell 18→6, with a Zombieman and then an Imp visible.

A route can only be credited when a move changes observed position/cell or produces a verified interaction/count change. Therefore this candidate tests a **route-epoch ledger**: every frontier turn or escape is a short experiment with a start cell/heading and an outcome. Failed, blocked epochs are penalized at that local state; successful epochs are allowed a longer translation lease. A separate contact escape watchdog can interrupt combat when a movement lease has no effect. The old heading-frontier signal remains the tie-breaker, so this is not a return to long model-controlled actions or a seed route.

The candidate also lets Astra author a bounded macro sequence from the public map/trajectory window, but only code validates and executes it at a quiet macro boundary. Astra may choose `frontier`, `left_trace`, `right_trace`, or `safe` mode and capped movement/use phases; it may not name hidden geometry, emit a raw button stream, or bypass the effect ledger. This gives the slow layer a real opportunity to revise the route mode after a repeated-cell episode instead of changing only `hand/sweep/combat` fields that seldom influence actions.

## Concrete parent evidence

* 0011 led at 10.958 with 0.345% stale ticks, but all 12 episodes had `exit_success=0`. Its best MAP01 seed 1008 reached 281 cells and 15 kills/19 items, then at the deadline was still around `(978,-272)` with only the player visible. Its trace repeatedly returned to local collision pairs despite 250 escapes/turns. Its worst seed 1012 ended after 2,737 steps at health 6; the final 100+ observations were almost stationary turn-left/right actions.
* 0015's lease did reduce some pure-turn streaks, but mean fell to 9.324 and it still had no exits. In the worst seed-1003 trace, `backward_right` (and later `backward_left`) was issued while the position was unchanged against a close LostSoul/Imp/Zombieman group; the lease was a button lease, not an escape. In the high-progress seed-1006 trace, 22 kills/20 items were reached but no exit, showing that more combat progress is not route completion.
* The lessons explicitly confirm that heading memory and moving leases are actuator improvements only, and refute treating more USE, pickups, or local contact following as a sufficient route. 0019 therefore does not reward optional pickups or ask Jev to invent a map route. It uses count/position effects as the only evidence for a route decision and makes immediate contact mobility outrank optional combat.

## What changed

1. **Effect-tested route epochs.** A `RouteEpoch` records the cell, heading, position, health, counts and intended direction when a turn/escape/trace phase starts. At its end, code scores new-cell displacement and verified count/interaction change, subtracts damage and a return-to-same-cell penalty, and stores a bounded result by `(cell, target_heading, mode)`. `frontier_turn_plan` still prefers the least-observed heading, but a known failed heading is delayed and an untried/effectively successful heading is preferred. After repeated failed epochs at one cell, the controller enters a bounded left/right trace phase rather than repeatedly issuing the same 5-back/turn/10-forward escape. Trace phases are abandoned on damage, no displacement, or a newly visible immediate threat.
2. **Measured contact escape.** The 0015 four-tic aim lease is retained only after the controller has selected a legal side. If position does not change for a few engine states, the lease is canceled and a contact escape sequence uses the currently better observed movement family (forward-diagonal, strafe, or backward), rotates through a frontier heading, then translates. The combat branch can no longer suppress this watchdog. At critical health, a close threat produces motion first; firing is secondary.
3. **Exit-oriented goal filter.** Interactions, keys and capability-changing weapons remain high-value visible goals. Routine ammo/armor/health bonuses are pursued only when code-derived need is real; otherwise they cannot interrupt an effect-tested route epoch. Periodic USE remains a one-tic pulse and is repeated at bounded trace/blocked-face checkpoints, not held through combat.
4. **Astra macro plan.** Up to four background Astra calls receive map/IWAD, compact recent trajectory, route-epoch outcomes, counts and visible labels. A parsed plan is capped to at most four phases of 48 tics and is applied only when no immediate threat exists. Unknown JSON, stale plans, bad durations and all model failures leave the deterministic controller intact. This is intentionally a policy sketch, not a map-specific waypoint oracle.

## Confirmation and refutation

Confirmation requires a higher mean than 0011 without an increase in stale ticks or invalid episodes, plus trace evidence of (a) fewer repeated failed heading/position pairs, (b) contact leases that actually displace the player, and (c) longer survival on seeds 1012/1003. The strongest confirmation is a terminal exit, with the trace showing a position/interaction/count transition immediately before it. A survival improvement with no exit would support the contact hypothesis but leave route discovery as the next bottleneck.

The route hypothesis is refuted if the ledger merely records outcomes without changing chosen headings, if trace mode cycles the same cells, or if effect testing spends more time turning and lowers 0011's aggregate score. The contact hypothesis is refuted if blocked leases remain long, if the new escape causes more deaths than 0011/0015, or if motion increases without improving health. More kills/items by themselves are not confirmation; all episodes with `exit_success=0` remain failures for the primary objective.

## Tier-2 Jev interface

Every request is one strict JSON state made only from the latest observation and code-derived facts (no hidden map data):

```json
{
  "task":"reach and activate the exit alive",
  "health":"critical|wounded|healthy",
  "armor":"none|present",
  "damage_recent":true,
  "threats":[{"name":"ShotgunGuy","kind":"hitscan|projectile|melee|routine",
               "range":"contact|near|medium|far","direction":"left|center|right",
               "major":false,"body_blocker":true}],
  "specials":[{"name":"BlueCard","kind":"key|weapon|interaction|health|other",
                "range":"near|medium|far","direction":"left|center|right"}],
  "motion":{"phase":"frontier|trace|escape|combat","blocked":true,
             "novelty":"new|old","cell_revisits":"low|high",
             "epoch":"untried|productive|failed"},
  "route":{"hand":"left|right","failed_recently":true,
           "trace_side":"left|right|none","previous":"forward"}
}
```

Code derives range/direction, threat taxonomy, blocked status, route-epoch outcome and health buckets. Jev never performs arithmetic, chooses a weapon, or receives coordinates. One request batches four independent questions:

* `route`: `frontier`, `trace_left`, `trace_right`, `use`, `retreat`, or `unknown`; it can change the next macro boundary only, and an accepted trace request is ignored while the threat watchdog owns control.
* `dodge`: `left`, `right`, `forward`, `back`, or `none`; it changes the preferred side for the next measured combat lease only.
* `engage`: `fight`, `break_contact`, `bypass`, or `unknown`; `break_contact` can start an escape only if code sees a close threat and does not already have a safe lease.
* `open`: a `Noul` asking whether the named currently visible interaction/blocked-face facts justify one USE pulse.

Choices require a legal value, selected-option probability >= **0.58** for `route`/`engage` and >= **0.54** for `dodge`; confidence below 0.10, missing probabilities that are non-typed, an exception/timeout, or an answer older than 55 Doom tics is rejected. `open` requires >= **0.72`. Answers are tagged with the request tic and are rechecked against the newest observation before use. A Jev hint never overrides a close projectile, critical-health movement, blocked lease, terminal state, weapon/ammo legality, or an active route epoch.

## Fallback and scheduling

The controller posts a fresh legal action on every processed engine state. While Jev or Astra is pending, the priority stack is: terminal check; close projectile strafe; measured contact escape/critical-health movement; active route epoch; immediate visible threat combat; verified interaction/needed capability goal; blocked-face USE; bounded trace/frontier motor. The previous action is never allowed to be the only watchdog.

Jev is spawned at a threat-sensitive cadence (about 22 tics) and a quiet cadence (about 52 tics), with a 1.35-second timeout. Its future is dropped if older than 55 tics. Astra is spawned only with at least 120 seconds remaining, at the initial route observation and after long route epochs, with a 180-second timeout and at most four calls. Astra output is applied only at a safe phase boundary and is capped/validated in code. `EpisodeOver`, `BudgetExceeded`, cancellation, timeout and malformed model text are normal fallbacks; `ModelUnavailable` is allowed to propagate as required by the contract. All background futures are canceled in `finally`, and no background loop survives `ctx.done`.
