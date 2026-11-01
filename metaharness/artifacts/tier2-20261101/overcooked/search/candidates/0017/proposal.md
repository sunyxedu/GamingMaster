# Candidate 0017 — progress-bounded jobs with a verified dish-backup transaction

## Parents and diagnosis

Parents: `candidates/0012/` and `candidates/0005/`.

The useful mechanism in 0012 is its code-owned, just-in-time production loop:
recipes and pot phases come from the observation, interactions use BFS poses,
modern cooking is explicitly started, and a dish is fetched late rather than
occupying the pot gate for an entire cook interval.  That change is real: its
mean score is **246.667**, with only **28.11** collision ticks and **197.56**
pot-idle ticks, better than the earlier committed controller.  Its clean
asymmetric episodes reach 18 deliveries/360 points.

The useful mechanism in 0005 is the continuously executable, recomputed motor
with a bounded escape.  It posts actions while Jev/Astra reason, rechecks the
current state, and leaves a blocked route rather than repeatedly pushing the
same direction.  It is less productive overall (195.556 mean), but its ring
trace still reaches 11 deliveries/220 on seed 2 and its failure is not a
model-latency failure.

The best candidate still has a concrete coordination hole.  In
`0012/trace_seed0.jsonl`, the Jev snapshots at ticks 235, 288, 326, and 380
show a `ready` pot while the partner holds a `dish`; the controller chooses
`wait` instead of a backup transaction.  At tick 380 the partner is at `[2,2]`
while the ready pot is `[2,0]`, and the episode ends at tick 400 with only
9 deliveries and a new soup cooking at tick 4.  The corresponding final
observation has a ready soup, no controlled object, and the partner has not
completed that last handoff.  The same pattern appears in seed 3: the final
state is a cooking soup at tick 5 after 9 deliveries, while the controller has
spent many boundaries in `wait`/`park_cook`.  The 0012 trace is not stale—the
episode has 0% stale ticks—so this is not fixed by a shorter Jev timeout.

This also explains why the older 0005-style “partner has a dish, so do not
fetch one” rule is unsafe.  A held dish is only an intention, not evidence
that the partner is progressing to the ready pot.  The lessons explicitly
refute indefinite traffic leases and unconditioned yielding: the ring cases
ended with ready soups and long `stay` runs.  I therefore did not add another
lease or make Jev choose raw directions.

## Hypothesis

The remaining score loss is caused by confusing **object ownership** with
**transaction progress**.  When the scripted partner has a dish, 0012 often
waits correctly near a ready pot, but it has no bounded takeover condition if
the partner is moving slowly, has stopped at a non-interaction cell, or is
blocked by the same gate.  The controlled chef can safely fetch another dish
from the unlimited dispenser, but the harness never does so in that state.

A controller should keep 0012's timed dish staging and its exact production
bookkeeping, while borrowing 0005's progress watchdog and escape.  It should
consider a partner handoff active only if the partner's route distance is
improving, or the partner is already facing the pot.  If a ready/nearly-ready
pot has a dish handoff that is not improving for a bounded number of ticks,
the controlled chef should launch a second, code-legal dish transaction.  All
other jobs should also have a maximum age and a physical no-progress recovery,
so a stale parking or interaction commitment cannot become an episode-long
wait.

## What changed

`harness.py` is a new `AdaptivePipeline`, not a copied task executor.

* It reconstructs active recipes, missing ingredients, pot phases, cook
  remainder, counters, interaction poses, and walkable paths directly from
  each JSON observation.  The only primitive emitted is an advertised action
  after code checks facing and interaction preconditions.
* It retains the JIT dish rule from 0012: while cooking, it performs a legal
  ingredient/counter job when possible and fetches a dish only when a route can
  meet readiness.  A carried dish waits on a non-gate floor cell.
* It adds a **partner-progress monitor**.  For a partner-held dish or soup it
  tracks the partner's route distance, position stagnation, held-object
  transitions, and whether the partner is already facing the target.  A ready
  or nearly-ready pot gets a `fetch_dish` option tagged `backup` only when the
  partner handoff is demonstrably stale; a partner who is approaching is not
  contested.
* It uses a bounded transaction executor rather than an indefinite gate lock.
  Each medium job has an age limit.  Five blocked movement ticks invoke a
  three-tick escape to a non-gate cell; eight blocked ticks or an expired job
  forces a fresh code-generated task.  Recovery is skipped for a carried soup
  except where the normal delivery route itself needs a detour.
* The task selector preserves hard priorities (`deliver`, ready pickup,
  backup/urgent dish, explicit start, compatible fill, supply, timed dish,
  parking).  Jev may only bias a current candidate within one urgency band;
  it cannot demote delivery or invent an action.  Astra supplies only a
  broad tempo, soft lane tie-breaker, and conservative/aggressive backup
  hint.  Neither model can alter legality or route search.
* Every model operation is backgrounded.  A late, malformed, cancelled, or
  garbage stand-in answer leaves the deterministic task and motor running.
  Actions are posted before waiting for the next realtime tick, so model work
  cannot create stale ticks.

## Evidence that would confirm or refute it

Confirmation would show, especially on cramped seeds 0/3/6:

1. `backup` dish transactions occur only after the partner's route stops
   improving, and ready soups no longer survive to tick 400 with a dish still
   in a distant partner hand;
2. the controller retains 0012's clean asymmetric throughput while reducing
   its repeated `monitor_ready`/`wait` tails, collision ticks, and unfinished
   final jobs; and
3. `bounded_recovery` is short and followed by a new productive transaction,
   rather than by a long `stay` run.

The hypothesis is refuted if backup fetches cause two chefs to contest an
approach while the original partner is progressing, if dish duplication
increases collisions without increasing deliveries, or if the watchdog
abandons long but valid routes on the larger layouts.  A trace showing a
ready pot, a partner whose route distance is decreasing, and a controller
still taking a backup dish would refute the handoff classifier.  A trace with
low collision counts but unchanged ready-soup tails would mean the bottleneck
is elsewhere (for example, the scripted partner's fixed recipe schedule),
not stale ownership recognition.

## Tier-2 Jev interface, thresholds, and fallback

Jev receives one strict JSON state per request under
`{"game":"Overcooked-AI symbolic kitchen","observation":...}`.  The
observation schema is:

* `tick`, `layout`, and `orders`;
* `self`: position, facing, and held object name;
* `partner`: position, facing, held object, last action, and derived
  `progress` (`still` ticks and `route_not_improving` ticks);
* `pots`: position, phase (`empty|partial|full_idle|cooking|ready`), missing
  ingredients, and cook ticks remaining;
* `current_transaction`: kind, target, age, and blocked count;
* `recovery`: whether the bounded escape is active;
* Astra's optional broad hints; and
* at most ten code-generated `candidates`, each with an exact ID, kind,
  priority, and legal plain-language description.

The request batches two independent typed judgments:

1. `transaction` is a `Choice` over the listed candidate IDs.  Its
   instructions forbid invented IDs, directions, counting, and arithmetic.
   The answer is accepted only when its ID is current and its selected
   probability is at least **0.55**; stand-in answers without a probability
   are still constrained by the current candidate set.  Even an accepted
   choice is used only when its priority is no more than one band below the
   deterministic best candidate.
2. `rescue` is a `Noul` about the derived partner progress and an urgent pot.
   A true probability of at least **0.78** can select the already-generated
   backup dish candidate.  It never creates a new target or direction.

Requests are asynchronous, have at most one outstanding future, and are
launched at a transaction boundary, a blocked route, or a 40-tick heartbeat,
with a 16-tick minimum interval.  Answers carry the transaction epoch; a late
answer from an old job is discarded.  Timeout, malformed values, absent
probabilities, exhausted budget, or model unavailability leaves code's current
job untouched.

The fallback is entirely deterministic: exact pot/recipe invariants decide
what must happen, BFS chooses a pose and a primitive, partner progress decides
whether a backup option exists, and the bounded escape handles a blocked
route.  `ctx.act` receives only a validated advertised action, and a failed
post gets one safe noop retry.  Thus a bad Jev/Astra response cannot crash the
episode or turn reasoning latency into stale action ticks.
