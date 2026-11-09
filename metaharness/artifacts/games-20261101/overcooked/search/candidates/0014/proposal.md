# Candidate 0014 — soft event allocation instead of gate ownership

## Parent and hypothesis

Parent: `candidates/0009/`; comparison baseline: `candidates/0006/` (the
leaderboard best before this proposal) and `0011`.

The useful part of 0009 was its code-owned recipe/pose motor, but its failure
is visible in the coordination-ring trace rather than in inference latency.
In `trace_seed2.jsonl`, 0009 commits `yield_partner` at tick 62, then again
at tick 83, commits `wait` at tick 90, and posts `stay` for the remainder of
the episode. It finishes with score **20**, **164 collision ticks**, and
**659 pot-idle ticks**, including two ready soups and a dish transaction in
the end state. Thus the partner claim became a global exclusion and had no
productive release path. This is the same failure mode recorded for 0008 and
0010: a traffic belief can be correct locally while its fallback starves the
pipeline.

The other loss is a different form of gate ownership. 0006's seed-3 trace
reaches a stable production loop, but holds the one pot approach while staging
a dish; its summary is **108 collision ticks**, **105 pot-idle ticks**, and
9 deliveries. At tick 400 it still has a three-onion pot cooking and the
partner holding a dish. 0006 nevertheless proves that its exact production
loop is valuable: its asymmetric episodes reach 16 deliveries with no
collisions and its ring episodes reach 11–12 deliveries. The hypothesis is
therefore not that the recipe arithmetic or shortest paths are wrong. It is
that a *single committed interaction pose* is the wrong unit of coordination
when the scripted partner is also navigating the same narrow route.

**Hypothesis:** a soft, event-level assignment with a receding-horizon motor
will retain the productive 0006 loop while avoiding the 0009 deadlock. The
controlled chef should reserve only the next legal interaction, not a pot gate
through an entire cooking window. A dish holder should park on a nearby
non-interaction floor cell until the ready boundary is close; an ingredient
claim should merely penalize that pot while another generated job exists. If
all alternatives are blocked, the motor must escape/re-route after a bounded
number of observations instead of manufacturing a permanent wait.

## What changed

* `AdaptiveKitchen` builds a complete set of code-generated events from the
  current observation: delivery, ready-soup pickup, explicit start, compatible
  fill, counter retrieval/deposit, ingredient fetch, dish fetch, and a safe
  park. It computes recipe compatibility, soup phases, counter occupancy,
  interaction poses, and all primitive actions locally. It supports onion and
  tomato recipes from `all_orders`; no training layout or fixed recipe value is
  embedded.
* The job is a soft event identity (`kind`, target, pot, ingredient). Its route
  is rebuilt every tick against the partner's current and predicted next cell.
  A blocked route tries another interaction pose or a deterministic escape
  step. After six blocked observations the current low-urgency event is
  temporarily avoided and another legal event is selected. There is no
  `yield_partner` task and no unbounded claim/lease.
* Cooking-window dish work is split into two event horizons. While a pot has
  ample cooking time, a dish holder parks away from all interaction poses; it
  approaches a pot only when the computed route is short enough to meet the
  ready boundary. At readiness, the same pot identity changes to
  `pickup_ready`, with legality rechecked before `interact`. This removes the
  0006 behavior of standing on the gate for the entire cooking interval.
* The partner is used as a dynamic fact, not as an owner token. A carried
  ingredient identifies the nearest compatible pot and adds a penalty to that
  pot's supply events; a carried dish/soup identifies the urgent transaction.
  Delivery, pickup, and start remain hard priorities. If no alternative
  exists, the controller continues to route toward the necessary event rather
  than parking forever.
* Astra is a background protocol director. It chooses only circulation,
  delivery bias, and whether counters are worth using; a second call is
  launched only after a long observed stall and only if the first call has
  completed. Astra never emits directions or a task id that code has not
  generated.
* Jev is a background transition adjudicator rather than a motor. It can
  select an already-safe event within the current urgency band, classify the
  partner's carried-object intent, and recommend a bounded replan. Its output
  cannot downgrade delivery, ready pickup, or explicit start, and it cannot
  create a claim.

This is a structural division change from 0009: code owns a soft event graph
and dynamic execution, Jev judges event boundaries and partner intent, and
Astra authors a layout protocol/recovery policy rather than supplying a weak
mode hint to a permanently committed gate job.

## Confirmation and refutation

Confirmation would be a higher mean than 0006/0011 while preserving the
16-delivery asymmetric traces, with the largest gain on ring seed 2 and the
cramped seeds. The traces should show no long `wait`/`stay` suffix, short dish
parking intervals, a `park -> stage_dish -> pickup_ready` transition near
readiness, and lower collision and pot-idle counts. A useful partial
confirmation is recovery after a partner-held ingredient without a global
lease or stale ticks.

The hypothesis is refuted if the soft alternatives make the controlled chef
contest both pots, if parking misses ready soup, or if the dynamic escape
wanders so much that the stable asymmetric loop loses deliveries. It is also
refuted on mixed layouts if recipe compatibility or counter events are
incorrect. The decisive diagnostics are deliveries and sparse score first,
then collision/stuck ticks, pot idle time, end-state held objects, and the
number/length of park and replan events. A next iteration should then learn a
more explicit counter handoff protocol rather than lengthening a traffic
lease.

## Tier-2 model contract

### Astra

The initial prompt contains the terrain, layout, current players/held objects,
pot phases and missing ingredients, and the number of usable counter sites.
It requests JSON:

```json
{"circulation":"shortest|clockwise|counterclockwise",
 "delivery":"self|partner|balanced",
 "handoff":"counter|direct|none",
 "replan":"event|blocked"}
```

The fallback is `shortest/balanced/counter`. The result is accepted only for
these enumerated fields and is used as a tie-breaker. A second, session-linked
review is requested only when the local stall counter reaches 14 and no Astra
call is outstanding. Both calls run via `ctx.spawn`; the tick loop never waits
for them. Garbage, timeout, or an unavailable answer leaves the fallback
protocol in place.

### Jev state schema

Every request sends a strict, compact JSON object (not the full observation):

* `tick`, `layout`, and the current advisory `protocol`;
* `self`: position, held object, and last posted action;
* `partner`: position, held object, last partner action, code-predicted next
  cell, and a code-derived `intent` (`kind`, `pot_id`, `item`);
* each pot's id, position, phase (`empty`, `partial`, `full_idle`, `cooking`,
  `ready`), ingredients, missing ingredients, and cooking ticks remaining;
* `current_job`: kind, target, pot id, blocked/stall counts; and
* up to 12 code-generated `candidates`, each with id, kind, pot id, and a
  plain description.

The questions are batched in one request:

1. `event: Choice` — choose one listed candidate for the next event. The
   options are generated from legal state facts; Jev never invents movement.
2. `partner: Choice` — classify `ingredient`, `dish`, `delivery`, or `none`
   from the explicit carried-object and intent fields.
3. `release: Noul` — whether a non-urgent current approach should be
   recomputed because the partner owns the gate.

A request is launched on a new job, after two blocked observations, or after a
long stall, with one outstanding request and a one-second timeout. A response
is discarded if its job epoch is stale. An event answer is used only if its id
is still present and its computed urgency is no worse than the current best
band; its selected-option probability must be at least **0.50** (a missing
probability uses the deterministic code fallback). A release answer requires
at least **0.75**, at least three code-observed blocked ticks, and a
non-urgent job. It causes at most one event replan, never a `stay` lease.
Partner classification only changes a diagnostic hint and the code's next
soft ranking.

When Jev is late, malformed, uncertain, or absent, code uses the current job,
rebuilds a route, and posts a legal action. Collision avoidance blocks the
partner's current/predicted cell; after bounded blockage it selects a safe
side step. Thus model latency can reduce advice freshness but cannot produce a
stale tick or stop the realtime clock. All model exceptions are contained in
the background worker, and all primitive actions are validated against the
latest symbolic observation before posting.
