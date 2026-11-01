# Candidate 0010 — physical ownership tokens and time-expanded traffic

## Parent and diagnosis

Parent: `candidates/0008/`; I also used the stronger `0006` trace as the
control.  This is a structural rewrite of the motor/scheduler, not a Jev
threshold change.

The parent’s seed-2 trace is a clear failure of a finite transaction lease:
Jev repeatedly selected `yield_pot_4_2` after about tick 132, while the final
state still had two ready soups, a partner-held dish, only one delivery, and
661 pot-idle ticks.  The lessons correctly say that yielding without an
observable release condition is refuted.  The best candidate, `0006`, removed
that particular long stale-action interval, but still had 49–110 collision
ticks on ring/cramped seeds (seeds 0, 2, 3, 5, 6, 8).  Its motor predicts one
partner step and then either contests the predicted gate or holds a fixed
approach.  For example, `0006` seed 3 ends at tick 400 with the controlled
chef holding a dish, the partner holding a dish, and a three-onion pot only at
cooking tick 16; it delivered 9 soups despite having run the production loop.
The traces show repeated `fetch_ingredient/fill_pot` commits around the same
single approach, so exact recipe bookkeeping is no longer the missing piece.

## Hypothesis

The remaining bottleneck is **route ownership over time**, not recognition of
the next medium-level task.  A current partner position is insufficient in a
one-cell kitchen: both chefs can select individually legal shortest paths that
collide several ticks later.  Conversely, an unconditional time lease can
suppress the only productive chef.  A controller should therefore:

1. infer a concrete partner transaction from its carried object and recent
   approach/interact evidence;
2. give that pot a physical ownership token, released when the observed pot or
   hand state advances, not merely because a timer expired;
3. plan the controlled chef’s next few cells in a time-expanded occupancy graph
   using a short partner-motion forecast; and
4. resolve a bad forecast with a bounded retreat to a non-gate parking cell,
   then re-evaluate and take over if the token has no fresh evidence.

This is deliberately different from `0008`’s 18-tick partner-fill lease:
`0008` leased based on a local claim and made the controlled chef wait.  0010
uses a token only when an object/approach supports a specific pot, clears it on
any pot signature transition, expires only tentative empty-handed approach
claims after four ticks, and lets a stalled low-priority job retreat or take
the next code-generated task.  It is also different from `0006`’s one-step
collision test: the route planner reserves predicted cells for several ticks
and treats stay as an explicit route edge.

## What changed

* The symbolic layer still owns recipe matching, pot phases, explicit cooking
  start, interaction poses, counter legality, and all primitive action names.
* `partner_tokens` map a pot to evidence such as `matching_ingredient`,
  `dish_for_soup`, `recent_pot_interact`, or a short `approaching_pot` history.
  A changed ingredient/phase signature clears the token.  Current carried
  soup/dish/ingredient evidence is stronger than a stale timestamp.
* Tasks are role-aware.  A dish holder does not occupy the sole pot approach
  while the partner is filling that pot; it parks.  A redundant ingredient is
  staged on a free counter.  Ready soup, soup pickup, and explicit pot start
  remain hard priorities.  If every useful pot is partner-owned, the empty chef
  parks away from all pot interaction poses rather than contesting a gate.
* `_time_path` searches `(cell, time)` states with wait, cardinal moves,
  predicted partner occupancy, and swap avoidance.  Routes persist through
  ordinary movement and are recomputed only after a conflict or route drift.
  Five unresolved conflict ticks trigger a safe retreat/detour instead of the
  parent’s north/south oscillation.
* Astra is one asynchronous protocol review.  It may return only
  `partner_first|self_first|balanced` and a lane tie-break; it cannot change a
  legal task or emit a direction.  Its answer is never required for progress.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a reduction in collision/stuck ticks and late partial
pots on `0006`’s weak seeds, especially cramped seed 3 and coordination-ring
seeds 2/5/8, without recreating `0008`’s 46.4% stale fraction.  Traces should
show a token tied to a concrete carried object, a route that waits or parks
before a predicted crossing, then token clearing on a pot transition and a
new fill/start/pickup job.  A useful result would keep the 0006 asymmetric
scores near 320 while increasing cramped/ring deliveries.

The hypothesis is refuted if time-expanded forecasts make the chef yield on
ordinary traffic, if token creation suppresses productive work while the
partner is merely nearby, or if parking leaves ready soups unserved.  The
most diagnostic refutation is a final state with a ready soup and an empty
partner hand while the controlled chef remains parked; that would indicate
that the evidence/release rule is too conservative.  Another refutation is
stale ticks rising: all model work must remain off the tick path.

## Tier-2 Jev interface and fallback

Each asynchronous Jev request receives strict JSON containing:

* `tick`, layout, controlled/partner positions, facing, held objects, recent
  partner positions and last partner action;
* each pot’s position, phase, ingredients, missing recipe items, cooking
  remainder, and whether the code has an active partner token;
* the committed job, epoch, conflict/block counters, and up to eight
  code-generated legal candidates; and
* the current Astra tie-break hint.

One request batches three independent typed judgments:

* `protocol_a` and `protocol_b` are the same closed-world `Choice` with
  reversed option order: `continue_job`, `yield_partner`, `detour`, or
  `take_ready`.  A protocol is accepted only when both answers agree and the
  selected option has probability at least 0.58 (or the stand-in has no
  probabilities).  This directly counters first-option bias and prevents a
  single uncertain answer from taking a gate.
* `partner_claim` is a `Noul` about whether current evidence supports the
  active token.  A probability of at least 0.72 can add a four-tick yield hint,
  but only if code independently sees a live token and the chef is not
  carrying soup.

Requests are spawned after a new ownership job or a persistent conflict, with
one outstanding request and a 1.1-second timeout.  Answers carry the job epoch
and are discarded after a task transition; no answer can emit a direction or
bypass current preconditions.  A late, malformed, timed-out, or cancelled
answer leaves the deterministic fallback in control.

The fallback is: deliver held soup; pick up ready soup; start a full idle pot;
fill a non-owned compatible pot; retrieve/stage a needed ingredient or dish;
otherwise route to a free counter or a non-gate parking cell.  Every tick
posts an advertised primitive before `wait_for_tick`, so Astra/Jev latency
cannot create stale actions.  Contract exceptions and malformed observations
are contained, and arbitrary stand-in model text is ignored.

Pre-flight cause: a freshly committed `park` job reached `_motor` before its local route variable was initialized; it now defaults to the controller’s stored route.
