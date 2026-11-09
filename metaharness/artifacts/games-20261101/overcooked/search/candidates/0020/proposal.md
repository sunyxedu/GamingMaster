# Candidate 0020 — event-calendar planning with a joint action lattice

## Parent and diagnosis

Parent: `candidates/0016/`; the current leaderboard reference is `candidates/0017/`.
I retained neither candidate's role lease nor its progress-triggered dish takeover as
the primary mechanism.  The new controller is an event-calendar executor with a
short joint occupancy search.

The best trace is productive but repeatedly loses time at the boundary between
one chef's event and the scripted partner's event.  In cramped seed 0, 0017 commits
`fetch_dish` at tick 65, then `wait_dish` at 67/68 and remains in the same dish
holding pattern until the ready pickup at tick 84.  The next cycle starts at 92,
then the same pattern repeats; at tick 380 it is still filling a new pot and ends
at tick 400 with a dish in hand and a pot only at cooking tick 15.  Seeds 3 and 6
show the same repeated `fetch_dish -> wait_dish -> pickup_ready` baton.  The final
states are not model-latency failures: every episode has 0% stale ticks, but the
cramped episodes still finish at 9 deliveries.

The ring reveals a different instance of the same boundary problem.  In seed 2,
0017 commits an ingredient route at tick 95 after a delivery, then posts
`stay,north,south` oscillations from ticks 99--114 before reaching the next fill;
seed 8 spends ticks 87--102 in `stay` after a dish-stage transition.  The
one-step partner prediction and a persistent job know the next feature, but do not
represent the partner's *intent route* or the future occupancy of the shared lane.
The controller therefore alternates between a shortest path and an immediate
partner conflict.  0010 tested a token/time-expanded traffic idea, and the lessons
showed that its ownership version could deadlock; this proposal deliberately does
not create pot tokens or indefinite yielding.

## Hypothesis

The remaining loss is caused by **myopic event handoff**, not by recipe arithmetic
or Jev latency.  Each chef should be treated as a moving event stream: infer the
partner's current medium-level destination from its held object and the active pot,
forecast only its short route, and choose the controlled chef's next primitive from
a small joint occupancy lattice.  A finite forecast should resolve a crossing
before it becomes a north/south oscillation, while the lack of a lease lets the
controlled chef take the route again as soon as the partner stops making progress.

The controller also keeps an event baton (`fill`, `start`, `get_dish`, `pickup`,
`serve`) across ordinary observations.  Replanning happens on a phase/hand/object
transition or a bounded physical stall, not merely because a model answer arrived.
This is intended to preserve the high-throughput asymmetric traces while reducing
ring crossings and the late, unfinished cramped baton.

## What changed

* **Partner intent inference:** code derives `serve`, `pickup`, `fill`,
  `get_dish`, or `get_ingredient` from the partner's hand, pots, recipes, and
  geometry.  It tracks route distance, still ticks, and route-not-improving ticks.
  A held object is evidence for a destination, not an automatic exclusion.
* **Event calendar:** code owns recipe matching, pot phases, explicit start,
  counter legality, interaction poses, dish timing, and a persistent next-event
  baton.  Hard priority is `serve/pickup > start/fill > ingredient/counter >
  timed dish > park`.
* **Joint action lattice:** instead of following only one BFS next step, the motor
  evaluates five primitive steps over a five-tick beam.  It scores future shared
  cells, swaps, route overlap, distance to the current interaction pose, and
  bounded waiting.  Conflict penalties are finite and depend on partner progress;
  there is no permanent gate claim.  If the controlled agent is index 1 and both
  chefs are already facing the same interaction target, the exact player-order
  rule gives the index-0 interaction one tick of precedence.
* **Astra:** one asynchronous initial call authors only `focus`, `traffic`, and
  `dish` protocol hints.  It cannot choose a target or primitive and is never
  awaited by the tick loop.
* **Jev:** asynchronous batched judgments are only tie-breakers over code-created
  event candidates.  Two reversed-order Choices must agree, and a separate Noul
  may request at most a two-tick yield only when code already sees an urgent,
  overlapping, stalled partner route.  All answers are checked against the
  current epoch and candidate set.
* **Realtime safety:** every loop iteration computes and posts a legal action
  before `wait_for_tick`; model calls run in a background future.  Garbage Astra
  text, arbitrary Jev stand-ins, stale answers, timeouts, exhausted budgets, and
  action-post exceptions fall back to the deterministic event motor.

## Confirmation or refutation

The hypothesis is confirmed if ring traces show fewer repeated stay/oscillation
runs and more deliveries without recreating the long-yield deadlocks from 0008--
0010, while asymmetric seeds remain near 17--19 deliveries.  In cramped seeds,
the useful sign is a stable event sequence with fewer late partial pots or held
soup/dish states; a change from 9 to 10 deliveries would be strong evidence,
though the serial layout may make 9 optimal.

It is refuted if the partner forecast makes the controller wait while the partner
is merely passing through, if route-overlap penalties cause repeated detours, or
if a final observation still has a ready/servable soup while the controlled chef
is parked.  A high collision count with the same delivery times would mean that
the intent forecast is wrong; unchanged collision counts but unchanged scores
would mean the apparent handoff loss is intrinsic cooking/route time rather than
myopic traffic.

## Tier-2 System-1 interface

Jev receives a strict compact JSON object, freshly tagged by `task_epoch`:

```json
{
  "tick": 123,
  "layout": "coordination_ring",
  "controlled_index": 0,
  "self": {"position": [x,y], "facing": [dx,dy], "held": "dish"},
  "partner": {
    "position": [x,y], "held": "", "intent": "fill",
    "target": [x,y], "route_distance": 4,
    "still_ticks": 2, "route_not_improving": 3,
    "last_action": "west"
  },
  "pots": [{"position": [x,y], "phase": "cooking",
            "missing": [], "cook_remaining": 5}],
  "event_baton": {"kind": "get_dish", "target": [x,y],
                   "age": 4, "blocked": 1},
  "protocol": {"focus": "balanced", "traffic": "joint", "dish": "timed"},
  "candidates": [{"id": "...", "kind": "start",
                   "priority": 1, "target": [x,y],
                   "description": "..."}]
}
```

The request contains:

1. `event_a`: a `Choice` over at most ten code-generated event IDs;
2. `event_b`: the same Choice with reversed option order, to reduce first-option
   bias; and
3. `yield`: a `Noul` asking whether an urgent partner route is actually stalled.

A Choice is accepted only if both selected IDs agree, the ID is current, and its
probability is at least `0.55` (or the stand-in exposes no probabilities).  It may
only replace the code choice within one priority band.  A `yield` probability of
`0.75` is still advisory: code additionally requires the partner to hold dish or
soup, the inferred route to overlap the current event, and `route_not_improving`
to be positive.  It produces at most two `stay` ticks and never overrides a
carried soup, ready pickup, start, or fill.

Requests launch at a new event baton, a four-tick route conflict, or a 42-tick
heartbeat, with at most one outstanding request and an 18-tick minimum interval.
Late answers are discarded by epoch.  The deterministic fallback is always:
current held soup to serving, held dish to ready soup, full idle start, compatible
fill, matching counter/dispenser, timed dish, then a safe floor cell.  The joint
lattice and current-state legality checks—not Jev—produce every primitive action.
