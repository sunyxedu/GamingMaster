# Candidate 0007 — event-driven production with a leased right-of-way motor

## Parents and diagnosis

This is the requested crossover of `candidates/0005/` and `candidates/0003/`.

`0005` supplied the important score mechanism: it owned the symbolic kitchen
mechanics in code (recipe-aware pot phases, explicit cooking starts, shortest
paths to interaction poses, and a rolling job list), while Jev selected only
among legal jobs and Astra never blocked the clock.  It reached a mean of
195.56, 9.89 deliveries, and only 6.1% stale ticks.  Its worst training
episode was seed 3 (cramped room, score 40).  The trace is a concrete failure,
not a lack of recipe knowledge: at ticks 126, 138, and 150 the controlled chef
was still at `[1,1]` holding a dish, the partner was at `[2,1]` holding an
onion, and the only approach to the pot at `[2,0]` was occupied.  The motor
then posted an alternating `south,north,south,north,...` escape sequence for
most of the remaining episode.  The final state still contained a ready soup,
a held dish, and a partner-held onion.  Similar lower scores correlate with
collision/stuck counts (seed 6 had 54 collision ticks; seed 2 had 20).

`0003` supplied the other useful mechanism: its observation-only traffic rule
noticed when the partner was facing into the controlled chef's cell and gave
the partner right of way before trusting Astra's answer.  That parent retained
all 400 ticks and had no stale actions, but its Astra-only bridge could not
produce a sustained production policy.  Its score (131.11) shows that traffic
yielding alone is not a kitchen controller.

## Hypothesis

The next bottleneck after adding a deterministic production motor is **traffic
state**, not another high-level plan.  A one-tick escape heuristic is unstable
when the partner owns the sole pot approach: it can move away and immediately
walk back, creating a collision-free-looking oscillation while neither chef
completes the handoff.  If right of way is represented as a short-lived lease,
the controlled chef should (1) evacuate to a stable non-approach parking tile,
(2) stay there while the partner remains the blocker, and (3) resume after the
partner moves (or after the short facing-only lease expires).  Combining this with 0005's exact
production loop should preserve the partner-only idle baseline, improve the
cramped-room deadlock, and reduce ring collisions without asking either model
to emit directions.

Evidence that would confirm the hypothesis is lower collision/stuck and pot
idle time on seeds 2, 3, and 6, with the controlled chef resuming the ready-pot
pickup after the partner deposits/starts the soup.  It is refuted if the chef
still oscillates at one blocked approach, if the parking lease makes the
partner wait indefinitely, or if score falls on unobstructed asymmetric runs;
then the traffic detector is too broad or the partner needs an explicit
resource handoff protocol.

## What changed

* The harness recomputes active recipes, missing ingredients, pot phases,
  dispenser/counter candidates, and interaction poses from every observation.
  It explicitly starts a full idle pot and prioritizes delivery, ready-soup
  pickup, filling, and ingredient collection in code.  It never adds shaped
  rewards to control.
* A fresh background Astra call receives a compact derived snapshot and can
  return only `preferred_kind` and a traffic hint.  The hint is advisory and
  cannot override a ready-soup emergency or create an action.
* Jev is launched in the background after a safe action is posted.  A single
  request batches a Choice over at most eight code-generated jobs and a Noul
  about immediate right of way.  Its result is used only when the exact job
  signature is still current and the Choice probability is at least `0.40`
  (or the stand-in has no distribution).  The deterministic first job is the
  fallback for timeout, malformed text, stale state, unavailable Jev, or no
  System-1 model.
* The traffic layer is independent of both models.  It treats a partner facing
  into the controlled cell, an occupied sole interaction pose, a direct next
  step, or a predicted partner step as a conflict.  It computes a nearby
  parking tile with BFS and holds a lease keyed by partner position and job.
  At the parking tile it posts `stay` while the partner still occupies the
  approach, rather than selecting a fresh escape direction every tick.  The
  latest observation always revalidates paths, facing, and interaction
  preconditions.
* All model work is bounded to one outstanding Astra and one outstanding Jev
  future.  The real-time loop posts a legal primitive before launching Jev,
  catches ordinary model failures, and uses `stay`/the current safe route as a
  fallback.  A future is never allowed to run a control loop after `ctx.done`.

## Tier-2 Jev interface

The strict JSON `state` contains only derived facts needed for a job judgment:

* `observation.tick`, layout, recent events, and compact traffic/stuck facts;
* controlled and partner positions, facings, held-object names, partner last
  action/task;
* for each pot, position, `empty|partial|full_idle|cooking|ready` phase,
  current ingredients, and recipe-derived missing ingredients;
* active `orders`; and
* up to eight candidates with exact IDs, kind, and plain-language descriptions
  (for example, `fill_onion_2_0` or `dish_1_3`).

The `job` Choice asks for one listed medium-level candidate and explicitly says
not to invent directions, arithmetic, IDs, or a raw primitive.  The `yield`
Noul is speculative advice only: the executor independently verifies partner
facing/occupancy and can establish the traffic lease without Jev.  A valid
Choice with selected-option probability below `0.40` is ignored; the Noul is
used only at `>=0.72`, and only as a hint to the geometry-checked lease.  An
answer whose job signature no longer matches the latest pot/held-object/candidate
facts is dropped.  `ctx.decide` uses a 3-second timeout in a background
future; timeout, runtime, type, malformed-answer, and budget/episode errors
never escape the episode loop, and four ordinary failures disable further Jev.

The fallback is deterministic shortest-path execution with explicit facing:
move one tile at a time, turn on a separate directional tick, interact only
when the target is adjacent and facing is correct, and stay during a valid
cooking-stage wait.  Traffic leases have precedence over model preferences.
Astra is not re-queried per tick; its initial output is useful only if it
matches a currently generated job kind.  If later traces show that Astra's
preference increases pot idle time, it can be removed without changing the
motor or Jev contract; if leases reduce collisions but leave handoffs late,
that would motivate a future explicit counter reservation rather than more
model calls.
Pre-flight fix: dispenser iteration unpacked `(symbol, (x, y))` as coordinates; all such loops now discard the symbol explicitly.
