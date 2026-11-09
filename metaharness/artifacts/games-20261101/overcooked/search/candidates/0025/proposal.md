# Candidate 0025 — congestion-triggered dish yield

## Parent and targeted hypothesis

Parent: `candidates/0017/`.  I retain its code-owned recipe/pot state machine,
BFS interaction motor, bounded transaction ages and recovery, stalled-dish
backup, and asynchronous Astra/Jev bridge.

The targeted hypothesis is narrower than changing the dish parking geometry:
`0017` treats a `wait_dish` transaction as passive whenever it has reached its
parking cell.  In the one-pot cramped room that cell is usually `[2,2]`, the
only non-interaction floor square.  A partner that is trying to cross that
square can therefore collide with the controlled chef, but `_update_motion`
only counts a blocked **directional self move**; a deliberate `stay` at the
parking cell never triggers recovery.  The result is a legal but traffic-
blocking wait rather than a model-latency failure.

The evidence is clearest in `candidates/0017/trace_seed0.jsonl`: `wait_dish`
commits at ticks 30, 68, 111, 151, 190, 229, 306, and 348, with long
`stay` runs at ticks 32–45, 70–83, 112–126, 153–164, 192–203,
231–242, 308–321, and 350–363.  The episode has 44 collision ticks and
ends at tick 400 with only 9 deliveries, a controlled-held dish at `[2,2]`,
a partner-held onion, and a soup cooking at 15/20.  Seed 3 has the same
repeated central waits (commits at 29, 66, 147, 190, 233, 275, 316, and
358), 34 collision ticks, and 9 deliveries.  The parent has 0% stale ticks,
so reducing model waits is not the explanation.

## One change

When a dish is waiting for a cooking pot, the controller keeps the parent's
central non-gate wait initially.  It now watches an observable physical
condition: the partner's last directional action predicts the partner's next
cell to be the controlled chef's current wait cell.  After three such
consecutive congestion observations, and only in a one-pot layout, the wait
transaction yields to the serving-side interaction pose (the route is
revalidated against the partner and its predicted cell).  The yield is a
short-lived alternate target; once the soup becomes ready, the normal
`pickup_ready` invariant immediately preempts it.  If the pose is unsafe or
unreachable, the original parent parking choice remains in force.

This is deliberately not the sibling's unconditional peripheral parking:
it does not move every dish to a dispenser-side gate, and it does not react to
mere partner idleness.  It tests whether a measured crossing conflict, rather
than the existence of a dish wait itself, causes the central collision tail.
All other task priorities, model thresholds, action validation, and fallback
behavior are unchanged from the parent.

## Confirmation or refutation

Confirmation would show `dish_wait_yield` only after the partner's predicted
move repeatedly enters the occupied wait cell, fewer collision/stuck ticks and
shorter central `stay` intervals on cramped seeds, and at least one additional
soup delivered or a later final `start`.  Ordinary two-pot/asymmetric/ring
episodes should remain near the parent because the new branch is gated on one
pot and a measured conflict.

The hypothesis is refuted if no yield is observed despite the same collision
signature, if yields occur while the partner is merely stationary, if the
serving-side pose creates new collisions, or if the final 9-delivery/cooking
at-15-to-20 tail is unchanged.  A score drop without reduced collision or wait
runs would indicate that the selected yield cell is a poor route reservation,
not that the parent needs more Jev decisions.

## Tier-2 Jev interface and fallback

The parent Jev bridge remains unchanged.  Each request is strict JSON with
code-derived facts only: tick/layout, self and partner positions/facing/held
objects, partner `still` and `route_not_improving` counters, pot phase and
cook remainder, current transaction/age/blocked count, recovery state, Astra
hints, and at most ten legal code-generated transaction candidates.  The
batched `transaction` `Choice` may select only a listed candidate, with a
0.55 selected-probability gate and no more than a one-priority-band demotion.
The `rescue` `Noul` is accepted only at probability 0.78 or higher and can
only select an already-generated backup-dish candidate.

Jev never emits a direction or target.  The new yield is code-owned and is
not a Jev option: its trigger is the exact partner-next-cell predicate plus a
three-observation threshold, and the motor still checks BFS, actor occupancy,
orientation, and interaction preconditions on every tick.  A late, malformed,
stand-in, timed-out, or unavailable model answer is ignored.  While any model
call is in flight, the deterministic current task posts a validated primitive;
`stay` is the fallback when a safe route does not exist, and bounded recovery
still invalidates a physically stalled non-soup task.

Pre-flight cause: the congestion-state patch accidentally omitted initialization of `partner_held`, so the first `_update_motion` call raised `AttributeError`; it is now initialized with the other partner-monitor fields, and malformed model answers remain contained by the existing polling fallbacks.
