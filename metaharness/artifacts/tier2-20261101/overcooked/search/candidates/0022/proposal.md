# Candidate 0022 — route-reserved production

## Parents and hypothesis

Parent: `candidates/0018/`, with the high-scoring production motor from
`candidates/0017/` as the comparison baseline.

The best recent controller is already good at recipe bookkeeping and primitive
execution.  The remaining low tail is a *traffic reservation* failure, not a
failure to recognize pot phases.  In `0017/trace_seed0.jsonl`, the controller
starts a new soup at tick 385, fetches a dish at tick 386, and commits
`wait_dish` at tick 390.  At tick 400 it is still at `[2,2]` holding a dish,
while the partner is at `[3,1]` holding an onion and the pot is only at cooking
tick 15.  The final observation has 9 deliveries and no error.  The central
wait cell is not a static feature approach, but it is on the partner's likely
route `[3,1] -> [3,2] -> [2,2] -> [2,1]` to fill the pot.

The same causal shape is more explicit in `0018/trace_seed3.jsonl`: after the
100-point delivery at tick 210, the controlled chef fetches and fills an onion
at ticks 210--215, then commits `park` at tick 216 with target `[2,1]` and does
not issue another productive task.  The final state has the controlled chef at
`[2,1]`, the partner at `[2,2]` holding an onion, and a two-onion idle pot.
`0018`'s role confirmation and `0019`'s bounded waits do not solve a parking
cell that is safe according to static geometry but lies in the partner's next
transaction corridor.  Existing controllers predict one next cell or exclude
feature approaches; they do not make the waiting cell a dynamic reservation
against the partner's full inferred route to a pot when the partner carries an
ingredient.

**Hypothesis:** a chef that is waiting with a dish (or yielding while a pot is
cooking) should continuously move out of the partner's predicted route tube,
including ingredient-to-pot routes, rather than stay at a fixed pot-adjacent or
central cell.  This should release the one-cell approach in cramped layouts
without sacrificing the production pipeline.  The reservation is soft for an
urgent carried soup: the motor may still take the shortest safe path after a
bounded wait.

## What changed

`harness.py` is a new `RouteReservedPilot`, not a role-locked or event-calendar
executor.

* It infers a partner intent from the partner's held object and live pot facts:
  ingredient -> compatible pot, dish -> ready/cooking pot, soup -> serving.
  It computes a BFS route and retains the next several route cells plus all
  interaction poses as a route reservation.
* Waiting is a **dynamic task** with no permanent floor target.  Every tick,
  `free_parking` scores reachable floor cells outside the current route tube.
  Static feature approaches are only a penalty, so the controller can escape a
  central one-cell gate even when all ordinary safe-floor candidates are near a
  dispenser or serving pose.
* The full route tube is a hard reservation for dynamic parking, but only the
  partner's current/next cell is a hard obstruction for productive movement.
  This prevents the ring from deadlocking on a long shared corridor.  After two
  actual blocked ticks the motor takes a side-step escape; after a longer stall
  the task is re-planned.  A carried soup or ready pickup is never discarded
  for a Jev suggestion.
* Production remains code-owned: compatible ingredient selection, explicit
  full-idle start, dish timing, ready pickup, serving, counter legality, facing,
  and all arithmetic are checked from the current JSON observation.  A partner
  corridor can change the route, but cannot change a legal interaction
  precondition.
* Astra receives one initial protocol snapshot and may label the broad tempo
  and handoff policy.  Jev only selects among current code-generated,
  equal-priority transactions and independently judges whether the derived
  route conflict merits a short yield.  Garbage, stale, late, low-confidence,
  or unknown model output is ignored.

Confirmation would be a reduction in `collision_ticks` and `pot_idle_time`,
with the cramped seeds 0/3/6 adding at least one delivery relative to `0017`,
while the 16--19 delivery asymmetric episodes do not regress.  Trace evidence
would show a `route_reservation`/dynamic parking move before the partner's
ingredient reaches the pot, followed by a successful fill or start rather than
a long static wait.  The hypothesis is refuted if the route tube makes the
controlled chef oscillate or starves its own dish pickup, if the partner's
script does not actually use the inferred route, or if the ring's genuine
choke points cause more deadlock than the old one-step prediction.

## Tier-2 System-1 interface

All Jev requests are strict JSON derived from the current observation.  The
schema is:

```json
{
  "game": "Overcooked-AI symbolic realtime gridworld",
  "tick": 123,
  "layout": "...",
  "self": {"position": [x,y], "facing": [dx,dy], "held": "dish"},
  "partner": {
    "position": [x,y], "held": "onion", "last_action": "east",
    "intent": {"kind": "fill", "target": [x,y], "urgent": false,
               "route": [[x,y], [x,y]]},
    "stationary_ticks": 2
  },
  "reservation": {"cells": [[x,y]], "route_conflict": true},
  "pots": [{"position": [x,y], "phase": "cooking",
             "missing": [], "remaining": 5}],
  "orders": [["onion", "onion", "onion"]],
  "current": {"kind": "wait", "target": null, "age": 3, "blocked": 1},
  "reason": "traffic_conflict",
  "candidates": [{"id": "...", "kind": "fill", "priority": 2,
                  "description": "fill"}]
}
```

One request batches three independent judgments:

1. `task` is a `Choice` over the small list of code-generated task IDs.
2. `task_reversed` repeats that choice with the option order reversed.  A
   selection is usable only when both values agree, the ID is still present,
   and its probability is at least `0.52` (or both stand-in answers omit
   probabilities).  It may not move a task more than the deterministic
   priority band.
3. `yield` is a `Noul`.  A true probability of at least `0.78` sets a five-tick
   yield window only if code independently sees a route conflict and the
   partner intent is non-empty.  It cannot override carried-soup delivery or
   create a route/target.

Requests are launched with `ctx.spawn` at task boundaries, after four blocked
movement ticks, or at a 55-tick heartbeat, with one outstanding request and a
1.0-second timeout.  The epoch tags the answer.  A late answer, old epoch,
unknown ID, exception, timeout, or malformed stand-in answer has no effect.
The always-live fallback is the route-reserved deterministic planner and
motor: it posts one legal primitive before `ctx.wait_for_tick` on every loop.

Astra is launched once off-thread with a 90-second timeout.  Its parsed labels
are advisory and cannot disable route reservation, explicit starts, legality,
or the carried-object invariant.  The realtime loop never waits for Astra or
Jev, and every model call is isolated from primitive action posting, preserving
zero stale ticks as the primary safety target.
