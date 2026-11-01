# Candidate 0009 — bounded partner transactions around committed jobs

## Parents and concrete diagnosis

Parents: `candidates/0006/` and `candidates/0008/`.  I retain 0006's
committed, pose-aware production motor and its asynchronous model bridge, and
cross it with 0008's observation-based partner transaction idea.  This is not
a parameter-only change: ownership is now a bounded state layer between task
selection and primitive execution.

0006 is the strongest candidate (244.44 mean, zero stale ticks), but its
cramped-room episodes still finish at 180/200.  Its traces repeatedly show a
complete loop followed by a long approach wait: for example, seed 3 commits
`stage_dish` at tick 138 and does not commit `deliver` until tick 153; its final
cycle starts cooking at tick 384 and therefore cannot deliver by tick 400.  The
same trace has 108 collision ticks; its final observation has both chefs
holding dishes while that last soup is still cooking.  The physical job
commitment is valuable, but it does not know that the greedy partner has
already picked up the next matching onion or dish, so both workers can claim
the one pot approach.

0008 identified this race, but its transaction lease was too blunt and its
synchronous Jev calls blocked the realtime loop.  Its coordination-ring seed 2
ends with only 20 points, two ready soups and a dish in the partner's hand;
`pot_idle_time` is 661.  Across the run it has 46.4% stale ticks.  In other
words, the ownership insight is useful, but a lease must neither become a
permanent wait nor be implemented on the tick path.

## Hypothesis

A scripted partner's carried matching ingredient, combined with a reachable
nearest compatible pot, is a strong short-lived ownership signal.  Preventing a
duplicate ingredient run until that transaction visibly progresses should
remove the one-cell race.  The lease should be **progress keyed and bounded**:
its clock is restarted only by an ingredient/signature transition, it lasts at
most 12 ticks while carrying (with only a four-tick put-down grace), and a
multi-pot kitchen may work on an unclaimed pot instead.
When yielding is necessary, the controlled chef walks to a computed parking
cell away from every interaction pose; it does not stand in the partner's
approach.  Ready soup, pickup, and explicit start always outrank a lease.

This should preserve 0006's throughput on asymmetric/ring layouts while
recovering the missed cramped-room cycle.  A further part of the hypothesis is
that a committed interaction pose should be held only for the current physical
transaction: ordinary movement never invalidates it, but a partner claim or
actual held-object/pot transition may replace it.

## What changed

* `PipelineController` builds an exact task graph from current objects, recipes,
  pot phases, counters and walkable poses.  It commits one interaction job and
  uses a fixed pose through `cooking -> ready`, rather than asking Jev for
  directions.
* `update_claims` infers a claim only for the partner's nearest compatible pot
  while carrying its missing ingredient, or for a recent partner interaction
  at a partial-pot approach.  Claims are keyed by `(phase, ingredients,
  missing)`, get a short grace period after a put-down, expire after 12 ticks,
  and do not hide ready/cooking/start work.
* A live claim suppresses only duplicate ingredient tasks.  On single-pot
  layouts the controller gets a `yield_partner` job with a global parking route;
  on multiple-pot layouts it continues on an unclaimed pot.  If both chefs hold
  a dish, the closer chef owns the ready/cooking-pot transaction and the other
  parks, avoiding the two-dish gate contest seen at the end of weak cramped
  traces.  This directly combines 0008's ownership mechanism with 0006's
  stable job and traffic recovery without copying 0008's global idle failure.
* The motor rejects moves into the partner, predicted same-cell moves, and
  direct swaps.  After a bounded obstruction it computes a real detour instead
  of alternating two directions.  All interaction preconditions are checked
  against the latest observation.
* Astra is a one-shot background role reviewer.  Jev is also background and
  batched: it chooses only among code-generated task IDs and separately judges
  whether an already visible claim deserves a brief yield.  Neither model can
  emit an action or create a lease.

## Confirmation / refutation

Confirmation would be a higher score than 0006 on cramped-room seeds 0/3/6,
with fewer collision ticks and a shorter delay from ready-pot observation to
pickup/delivery, while retaining roughly 0006's 320-point asymmetric results
and 220–240 ring results.  Traces should show a partner claim, a short parking
job, a pot-signature change, then normal `fill -> start -> stage -> pickup`.
Stale ticks should remain near zero because every model request is submitted
after an action decision through `ctx.spawn` and never awaited by the motor.

The hypothesis is refuted if claims repeatedly expire with the partner still
blocked, if parking itself starves a ready soup, or if mixed/tomato recipes are
suppressed by an onion-only assumption.  The relevant evidence is the final
pot phases, claim age/progress, collision/stuck ticks, and deliveries—not
shaped reward.  A next iteration would then learn a partner route from a longer
action history or use explicit counter handoffs, rather than increasing the
lease timeout.

## Tier-2 Jev contract

The JSON state sent on every admitted request is compact and strict:

* `tick`, `layout`, active recipe lists;
* controlled and partner position, facing, held object, partner last action and
  task;
* each pot's position, `empty|partial|full_idle|cooking|ready` phase,
  ingredients, missing ingredients, cook ticks remaining, and boolean
  `partner_claim`;
* the committed job kind/target and blocked count; and
* at most ten code-generated candidates, each with an ID, kind and description.

The batched questions are:

1. `task: Choice` chooses one listed candidate.  The bridge accepts it only if
   the ID is current and it is in the code-selected highest-priority band; it
   can choose between dispenser/counter routes but cannot replace delivery,
   start, or a live lease with speculative work.
2. `yield: Noul` asks whether an existing `partner_claim` should be respected
   briefly.  A value of at least `0.72` records a conservative hint only when
   code already found a live claim; it never creates one and never interrupts
   a held soup/dish transaction.

Responses carry the job epoch and are discarded when the physical job has
changed.  Choice probability below `0.42` is treated as undecided; absent or
malformed probabilities use the deterministic fallback, not an exception.
Requests have one outstanding future and a one-second timeout.  They occur at
job/claim boundaries and verified obstruction, not every tick.  Astra's parsed
`preferred_kind` is likewise only a same-priority tie-breaker.  Timeout,
stand-in garbage, budget closure, or model unavailability leaves the local
fallback running.  The fallback priority is: deliver held soup, pick up ready
soup, start full idle pot, fill a compatible pot, respect a live partner claim,
retrieve/fetch an ingredient, fetch/stage a dish, then wait.
