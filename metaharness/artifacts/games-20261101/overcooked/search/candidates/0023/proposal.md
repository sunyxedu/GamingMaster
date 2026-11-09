# Candidate 0023 — peripheral dish waiting in the one-pot choke

## Parent and targeted hypothesis

Parent: `candidates/0017/` (the current best, 255.556 mean).  I keep its
code-owned recipe bookkeeping, BFS interaction poses, progress-bounded jobs,
backup-dish monitor, asynchronous Jev/Astra bridge, and realtime fallback.

The targeted hypothesis is that the parent's repeated **dish waiting pose** is
wrong in the single-pot cramped kitchen.  It correctly avoids the pot
interaction pose, but its only ordinary non-gate floor cell is the central
choke.  A chef carrying a dish therefore parks in `[2,2]` while the scripted
partner must cross the same cell to fetch onions or approach the pot.  This is
not a model-latency problem: the parent has 0% stale ticks, yet the cramped
traces show repeated physical waiting and collisions.

Concrete evidence from `candidates/0017/trace_seed0.jsonl`:

* at tick 30 it commits `wait_dish` at `[2,2]`; the tick records then post
  `stay` from ticks 32–45 before the transaction expires;
* the same `[2,2]` wait repeats at ticks 68–83, 111–126, 151–164, 190–203,
  229–242, 306–321, and 348–363;
* the final observation is only 9 deliveries/180, with the controlled chef
  holding a dish at `[2,2]`, the partner holding an onion, and the pot cooking
  at 15/20.  The episode reports 44 collision ticks.

Seed 3 shows the same signature: `wait_dish [2,2]` at ticks 29, 44, 66,
81, 147, 162, 190, 205, 233, 248, 275, 290, 316, 331, 358, 374; its final
state has 9 deliveries, a dish at `[1,2]`, a three-onion pot cooking at tick 3,
and 34 collision ticks.  Seed 6 similarly repeats central waits and ends at
9 deliveries with 40 collision ticks.  The parent proposal's backup-handoff
logic is never the explanation for these episodes: the trace has no
`jev_rescue` event, and the issue occurs even when the partner is not carrying
a dish.

The test is therefore: in a one-pot kitchen only, keep a newly collected dish
at the dispenser-side interaction pose when that pose is not occupied by the
partner or its predicted next cell.  This leaves the central floor route free
for the ingredient runner.  If the partner enters that pose, or if the kitchen
has multiple pots, fall back exactly to the parent's safe non-gate parking.
This is a single scheduling change, not a new lease or role assumption.

## What changed

`harness.py` is `PeripheralDishPipeline`, a parent-compatible executor with
one new code-owned parking decision:

* `_dish_wait_spot` recognizes the one-pot geometry from the observed terrain;
* immediately after dish pickup it prefers the current `D` interaction pose,
  rather than asking `_safe_parking` to send the chef to the central cell;
* it rejects the peripheral pose when it equals the partner or the partner's
  predicted next cell, and otherwise preserves the parent's route/precondition
  checks;
* two-pot layouts, non-dish waits, and all model behavior retain the parent
  implementation.

The peripheral pose is only a waiting location.  The controller still enters a
pot interaction pose only after `is_ready`, and it still revalidates every
interaction against the latest observation.  A malformed Astra/Jev response,
late future, timeout, or unavailable answer leaves the deterministic motor in
charge.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a reduction in cramped-room collision/stuck ticks and
fewer long central `stay` intervals, with the controlled chef's next route
using the open top corridor, followed by at least one additional team delivery
or an earlier final start.  The two-pot seeds should remain close to parent
scores because they take the unchanged parking branch.

The hypothesis is refuted if the peripheral wait makes the partner collide with
the dish-side cell, delays pickup enough to lose deliveries, or leaves the
same `[2,2]` wait/collision signature.  A score drop on asymmetric or ring
seeds would indicate that the one-pot guard is incomplete; a trace with a
partner approaching the dish pose while it is retained would indicate that the
predicted-cell guard is insufficient.

## Tier-2 Jev interface and fallback

Jev continues to receive a strict JSON state of the form:

```json
{
  "game":"Overcooked-AI symbolic kitchen",
  "observation": {
    "tick": 123,
    "layout":"cramped_room",
    "self":{"position":[1,2],"facing":[1,0],"held":"dish"},
    "partner":{"position":[3,1],"facing":[0,-1],"held":"onion",
                "last_action":"north",
                "progress":{"still":2,"route_not_improving":0}},
    "pots":[{"position":[2,0],"phase":"cooking","missing":[],
              "cook_remaining":7}],
    "current_transaction":{"kind":"wait_dish","target":[1,2],
                            "age":3,"blocked":0},
    "recovery":{"active":false,"reason":"none"},
    "astra":{"mode":null,"lane":"left","backup":"conservative"},
    "candidates":[{"id":"...","kind":"...","priority":4,
                   "description":"code-generated legal transaction"}]
  }
}
```

The batched questions remain:

1. `transaction`: a `Choice` over only the code-generated candidate IDs.  A
   returned ID is usable only if it is still present, its probability is at
   least 0.55 (or a preflight stand-in supplies no distribution), and its
   priority is no more than one band below the deterministic best.
2. `rescue`: a `Noul` asking whether a partner-held dish handoff is stalled at
   an urgent pot.  A true probability must be at least 0.78 before it can
   select the already-generated backup-dish candidate.

The questions never choose a direction, count ticks, or create a target.  The
request is asynchronous, tagged with the transaction epoch, and discarded if
late or stale.  `ctx.act` receives only one of the advertised primitive names.
The deterministic fallback is: ready soup delivery, ready pickup, explicit
start, compatible fill, ingredient/counter work, timed dish wait, then safe
parking; BFS and current partner position reject blocked steps.  After five
blocked movement ticks the parent's three-tick escape runs, and after eight or
an expired task the task is rebuilt.  `ctx.wait_for_tick()` keeps the realtime
clock moving while model calls are in flight.
