# Candidate 0020 — committed escape corridors instead of per-tick route thrashing

## Parent and diagnosis

Parent: `candidates/0016/` (a rewrite of `0009`).  I also inspected `0009`, the
high-scoring `0012` traces, and the failed `0017` crossover.

`0016` is the strongest controller so far: it posts every tick, reaches 1,790–5,010
points, and often repopulates the dot grid.  Its traces also expose the remaining
failure more precisely than the aggregate score.  In seed 0, shortly before the first
loss at frame 7,488, the route leases around frames 7,256–7,280 alternate between
`[148,103] -> [140,103]` and back, with risk 1/2 and margins down to -9.08.  Around
frame 7,320 it even accepts a Jev checkpoint route with risk 3 and margin -14.  The
same pattern occurs before other losses: code repeatedly rebuilds one-edge retreat
and dot packets as the projected ghost moves.  The player is therefore not following
a stable escape corridor; the controller is reacting to its own changing candidate
set.  At the terminal observations, ghosts are adjacent (for example seed 1 has
player `[53,73]` and a ghost at `[51,75]`, seed 5 has `[29,50]` and a ghost at
`[29,54]`).  All six runs still lose all three lives.

This is different from the already-tested questions of pill quota, epoch detection,
or Jev confidence.  `0017` attempted a lock but collapsed at the opening because its
no-progress recovery alternated between `[72,103]` and `[84,103]` every few ticks.
The new change therefore keeps the proven 0016 executor and puts hysteresis around
route selection, rather than making a new controller wait at a stall.

## Hypothesis

The score ceiling is caused in part by **route-selection churn in the short ghost
horizon**.  When a route becomes threatened, replacing it every four native frames
with the currently best one-edge branch can produce reversals, lose the turn window,
and enter a corridor whose risk estimate is already stale.  A controller that commits
to a selected multi-node corridor for a short lease, and commits to an emergency
escape until its first safe junction (or until it is demonstrably invalid), should
preserve dot coverage while reducing lethal oscillation.  The safety rule is
lexicographic: an immediate collision can break a lease, but a merely different
risk score cannot.  Jev will choose a *route goal* only; it cannot switch a current
lease or post a direction.

## What changed

Relative to 0016, this harness adds a **corridor lease / escape lock** state machine:

* Every executable route has a goal, path, route risk, and expiry in native frames.
  The ordinary route is retained through small risk fluctuations.  Code may replace
  it only when the current path is invalid, its goal disappeared, its lease expired,
  or the new path improves the risk class by at least two levels.  Equal-risk and
  one-level changes do not trigger a new lease.  This removes the repeated
  `code_guard` one-edge route churn seen in the best trace.
* A level-2 hazard installs an **escape lease** for a bounded horizon.  Its first
  action is held through the next physical turn window; subsequent actions follow
  the same escape path, revalidated against fresh observations.  It is released at
  a safe junction, after reaching the power-pill goal, or if the current escape
  has become an immediate collision; it is never replaced merely because another
  branch gets a slightly better computed score.  The emergency fallback is still
  code-owned and cannot be overridden by a Jev result.
* Route history records the previous packet's target/action/risk and logs
  `corridor_commit`, `escape_commit`, and `lease_hold`.  This makes the hypothesis
  falsifiable in the trace rather than relying on a high score alone.  The existing
  dot-repopulation detector, power accounting, inferred graph, ghost projection,
  and cardinal pre-turn execution are retained, but the old per-tick `hold_action`
  is replaced by a path-level lease.
* Jev receives the same compact derived route candidates, but a response is accepted
  only if its selected target is still in the current packet set and is no safer than
  the active lease by less than the emergency threshold.  Accepted answers can
  install a normal goal lease, never an immediate escape lease.  Bad, arbitrary,
  stale, low-confidence, or late stand-in answers fall back to the active code lease.
* Astra remains an asynchronous rubric author.  It is not placed on the action path;
  malformed or cancelled answers are ignored.  The last life forces survival mode.

The harness deliberately does not import an arcade map, ghost personality, or a
training-seed route.  All geometry comes from observed pill coordinates and all
hazard arithmetic remains code-owned.

## System-1 interface, questions, thresholds, and fallback

Jev is called asynchronously at branch/hazard intervals, not on every tick.  Its
strict JSON state is:

```text
{
  rules: {objective, power, choice_semantics},
  plan: {epoch, rubric, stage, active_kind, active_risk,
         active_target, lease_phase, last_life},
  options: {route_name: {purpose, risk_label, exit_branch,
                         dot_yield, reverses_motion, continues_lease}},
  recent: {score_progress, guard_recent, escape_locked},
  observation: {frame, lives, deaths, visible_dots, visible_power_pills}
}
```

Code computes legal corridors, graph paths, ghost projection, risk classes, dot yield,
route validity, and lease state.  Jev does not calculate distances, infer walls, or
choose diagonals.  One request batches:

* `route_a`: a `Choice` over the current non-dominated route goals;
* `route_b`: the same choice with criteria reversed to reduce first-option bias;
* `renew`: a `Noul` asking whether recent evidence supports replacing a stalled
  ordinary lease.

A choice is usable only when both choices agree, both selected probabilities are at
least **0.70**, the request is at most 48 native frames old, the epoch/death/rubric
version is unchanged, and the target remains legal.  A route answer is ignored while
an escape lease is locked, during a level-2 hazard, or when its recomputed risk is
worse than the active lease.  The Noul is advisory and can request an Astra audit;
it cannot select a route.  Missing probabilities, unknown values, disagreement,
exceptions, timeout, and all malformed stand-in outputs are treated as no answer.

The deterministic fallback is always a valid cardinal action: continue the active
path lease, follow a code-selected power escape, select a new dot route only at a
lease boundary, or hold the observed cardinal motion during opening calibration.
An immediate hazard can preempt an ordinary lease, but its replacement is held as an
escape lease rather than reconsidered every tick.  `ctx.wait_for_tick()` keeps the
real-time loop paced and there is no synchronous model wait.

## Confirmation and refutation

Confirmation requires zero stale/error episodes, trace evidence of `corridor_commit`
and `lease_hold`, fewer route switches near deaths, and either a higher mean than
0016 or a concrete survival gain: fewer than three deaths, a maze clear, or a bonus
life.  In particular, an escape should remain committed across several observations
instead of alternating two adjacent branches.

Refutation is unchanged adjacent-ghost contact despite a stable escape lease, a large
coverage/score drop because the lease is too sticky, or a trace showing an escape
lease repeatedly invalidated without movement.  If leases reduce score but survival
improves, the next iteration should tune route breadth or resource use rather than
reintroduce per-tick route arbitration.  If no `lease_hold`/`escape_commit` event
appears, the lock is not wired into the live loop and the hypothesis has not been
actually tested.
