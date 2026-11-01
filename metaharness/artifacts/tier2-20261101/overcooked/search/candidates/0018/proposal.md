# Candidate 0018 — lock complementary production roles at delivery boundaries

## Parent, evidence, and hypothesis

Parent: `candidates/0015/` (with the current best `0017` used as the trace
reference).  I kept the parent family’s verified exact symbolic bookkeeping,
BFS interaction poses, explicit modern cooking start, asynchronous model work,
and collision-safe one-tick motor.  This is a structural rewrite of the
planner/model boundary, not a timeout or threshold tweak.

The traces point to a specific loss that the previous task voters cannot see:
the controlled chef’s *actual role* is stable within a layout, but the planner
reopens ingredient work whenever a pot is partial.  In `0017/trace_seed0`,
`role_task`-equivalent transitions after deliveries repeatedly go
`take_counter` at ticks 53/92/170/209/248/282/331/371, then `fill`, while the
controlled chef is the delivery-side chef.  After delivery 371 it takes a
counter at 371, fills at 380, starts at 385, fetches a dish at 386, and is still
holding the dish at tick 400 with the pot cooking at tick 15.  The same tail is
clear in seed 3: after delivery 379, it takes/fills supply work at 383/388,
puts another ingredient on a counter at 397, and only picks up a dish at 400;
the pot is cooking at tick 3 and score remains 180.  These episodes have 0%
stale ticks, so the loss is not inference latency.

Conversely, the high-throughput layouts show the opposite stable role.  In
`0017` seeds 1, 4, and 5/7/8 the controlled agent earns little or no delivery
credit while the partner repeatedly delivers; the controlled side should stay
supplier.  A single global “ready > ingredients” task priority is therefore
not enough: it permits role drift and lets the controlled side contest the
partner’s delivery or miss its own late dish transaction.

**Hypothesis:** a persistent complementary role contract, confirmed by the
first sparse delivery credit and revised only on a verified handoff anomaly,
will remove the delivery chef’s speculative ingredient detours in the cramped
room while preserving the supplier behavior that already gives 17–19 deliveries
on the asymmetric/ring layouts.  The runner fetches a dish at the start of a
cooking phase and waits off the gate; the supplier owns compatible ingredients,
full-pot starts, and the next buildable pot.  Exact object ownership still
overrides the contract, so a mistaken role cannot make an illegal action.

## What changed and division of labour

* **Code:** derives recipes, pot phases, cooking remainder, counter contents,
  interaction poses, route lengths, partner progress, collision avoidance, and
  the legal primitive.  It runs a role-specific state machine rather than
  asking Jev to pick among raw task candidates.  A reward attribution
  (`sparse_r_by_agent`) confirms runner versus supplier after a delivery; ready
  soup, carried soup, and held-object preconditions are hard invariants.
* **System-2/Astra:** is asked once, off-thread, for a compact opening contract:
  runner/supplier prior, early/timed dish policy, and the event that permits a
  switch.  It cannot emit directions, change recipe arithmetic, or override a
  role confirmed by sparse reward.  Its output is useful even if it arrives
  late because the deterministic geometric prior is already running.
* **System-1/Jev:** no longer chooses every medium task.  At role boundaries and
  handoff anomalies it receives a filtered role/phase state and answers two
  typed judgments in one request: `role` (`runner|supplier|keep_code_role`) and
  `handoff` (Noul).  Jev can revise an unconfirmed role only with probability
  >=0.68.  A handoff “yield” is accepted at >=0.75 and “rescue” at <=0.25;
  otherwise code keeps its current policy.  Jev never invents a target or
  direction.

This changes the structural control allocation: code plans and executes exact
transactions, Astra authors the macro contract, and Jev recognizes only the
semantic uncertainty that code cannot get from one snapshot (whether a
partner-held dish is a live handoff or a stalled one).

## Confirmation or refutation

Confirmation would show the cramped seeds 0/3/6 switching to `runner` after
self delivery credit, with fewer ingredient/counter tasks after delivery,
earlier dish possession, and at least one additional completed delivery or a
later cooking phase before tick 400.  Seeds 1/4/7 and ring seeds 2/5/8 should
retain their supplier-side delivery throughput rather than losing their clean
0-collision high scores.  A successful trace should end with no runner still
carrying a newly fetched dish while a pot is only beginning to cook.

The hypothesis is refuted if early role locking makes the partner and
controlled chef duplicate ingredients, if supplier role loses the existing
17–19 delivery asymmetric episodes, or if a partner-held dish that is actually
progressing is repeatedly rescued and causes gate collisions.  It is also
refuted if cramped traces still show the same post-delivery ingredient detours
and unfinished cooking phases despite the confirmed runner role; that would
implicate the scripted partner’s supply cadence rather than role drift.

## Tier-2 Jev contract and realtime fallback

Each Jev request is strict JSON under
`{"game":"Overcooked-AI symbolic kitchen","observation":...}` with:

* `tick`, `layout`, active `orders`, and each pot’s code-derived
  `empty|partial|full_idle|cooking|ready` phase plus missing ingredients and
  cook remainder;
* `self`: position, facing, held object, code role and whether delivery credit
  has locked it;
* `partner`: position, held object, last action, stationary-tick count, and
  code-computed route-not-improving count;
* `current_task`: only the macro kind and age; and
* the closed role options `runner`, `supplier`, `keep_code_role`.

The `role` Choice recognizes the complementary job, not a primitive.  It is
ignored when delivery credit has already locked the role, when its value is not
one of the closed options, or when its selected probability is below 0.68.
The `handoff` Noul is about the exact partner-held dish/soup transaction.  A
high true probability gives the code a short yield hint; a low probability
permits a backup dish run only when a ready/nearly-ready pot and the code’s
stalled-progress test agree.  It does not create a job.

Model calls run through `ctx.spawn`, with one outstanding request and a 1-second
Jev timeout.  Late, malformed, cancelled, or arbitrary stand-in answers are
ignored.  Every action comes from the code role planner, is revalidated against
the latest observation, and falls back to `stay` or a three-tick escape when a
partner blocks a route.  `ctx.act` is posted before `ctx.wait_for_tick`, so the
slow Astra call and Jev tail latency cannot create stale ticks.  A role answer
is invalidated by delivery credit, a new held-object phase, or a changed task
epoch.
