# Candidate 0006 — commit the physical job, reserve the gate

## Parent and diagnosis

Parent: `candidates/0001-tier2_astra_only/`, with `candidates/0005/` as the
strongest observed crossover.  This is a **rewrite**, not a parameter change.
I retained the useful parts the traces established—symbolic recipes and pot
phases, BFS to an interaction pose, an explicit modern `start_cooking` action,
and a model fallback—but changed who owns the decision and when it is revised.

The best trace is already high scoring (195.556), so the failure is structural,
not a missing rule.  Its worst training episode is seed 3: it scores only 40,
with `collision_ticks=26`, `stuck_ticks=11`, and `pot_idle_time=292`.  The
trace shows the controlled chef successfully builds and delivers two soups,
then at about ticks 126–390 has a dish while the pot is `ready` and the partner
holds an onion at the sole pot approach.  The repeated Jev requests keep
presenting `pickup_ready_2_0`; the motor alternates `north,south` and never
claims the approach.  Seed 0 shows the same failure in milder form: after the
dish is staged, the controlled chef reaches the pot approach around tick 29,
then voluntarily leaves it when the partner enters the gate, and does not pick
up the ready soup until roughly tick 153.  The prior controller's “escape” rule
therefore turns a useful delivery reservation into an oscillating detour.

There is a second, less visible problem: 0005 asks Jev for a new job whenever a
signature changes and then recomputes the route every tick.  That is safe for
malformed answers, but it lets a semantic choice (fetch/start/stage) and a
moving partner repeatedly invalidate the physical approach.  Jev is being used
as a task scheduler where code already has the exact preconditions.

## Hypothesis

Overcooked's bottleneck on these layouts is **commitment at the one-cell
interaction gate**, not recognition of the next recipe step.  A chef that
reserves a gate before a soup is ready, keeps that pose while waiting, and
replans only after an observable hand/object/pot transition should remove the
long ready-pot stalls without sacrificing the partner's independent work.

The new division is:

* **Code** owns all exact facts: recipe matching, soup phase/timer extraction,
  dispenser/counter discovery, interaction preconditions, target poses, BFS,
  stale-job invalidation, collision/swap checks, and the production priority.
* **Astra** receives one compact snapshot and allocates a weak high-level mode
  (`delivery`, `supply`, `handoff`, or `balanced`) plus an optional side/traffic
  preference.  It runs in the background and can never block a tick or override
  a ready-soup invariant.  Its answer is useful if it returns in time, but the
  code controller is complete without it.
* **Jev** is no longer asked to choose an exact job or a direction.  At a new
  job or persistent traffic block it classifies the current phase and whether a
  two-tick yield is appropriate.  Code accepts only a current closed-world
  phase with a confidence gate, and continues to own the exact job and action.

## What changed

`0006` uses a stateful `job` record.  The record contains the medium-level
operation, target feature, selected interaction pose, and a gate-lock bit.  It
survives ordinary movement observations.  A cooking-pot dish job keeps its
chosen pose through the `cooking -> ready` transition, changes only from
`stage_dish` to `pickup_ready`, and stays at the pose while the partner tries
to approach.  When the controlled chef is already at the gate, `stay` is a
reservation; it is not treated as a reason to flee.  If the approach is not yet
claimed, the motor uses the partner as a dynamic obstacle, detects predicted
same-cell/swap moves, and uses a deterministic non-reversing traffic step
instead of the old north/south escape oscillation.  A delivery/start reservation
may contest a predicted entry into its exact gate, relying on the MDP collision
rule to retain the reservation.

The code-generated task list still covers held soup, ready pickup, explicit pot
start, compatible ingredient fill, staged-counter retrieval, dispenser runs,
dish staging, and counter deposit.  All geometry and recipe/value-independent
legality is derived from the current observation, so the same executor can
operate mixed recipes and randomized control indices.

Both model bridges are asynchronous.  The loop posts an action before waiting
for the next tick, and it never calls `ctx.decide` or `ctx.think` synchronously.
A late, malformed, low-confidence, or cancelled response is simply ignored.
This tests a structural alternative to 0005's per-boundary synchronous Jev
scheduler while preserving its useful motor fallback.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a sharp reduction in ready-pot pickup latency and
collision/stuck ticks on seeds 0 and 3, with at least 0005's 195.556 mean score
and fewer stale ticks.  Traces should show a `job_commit` for `stage_dish`, a
stable pose, then `pickup_ready` and delivery rather than an escape loop.  On
seeds where the partner can finish independently, score should not fall below
the idle/0005 baseline because all model delays are off the tick path.

The hypothesis is refuted if the controller reaches a gate but reservation
causes the partner to deadlock for many ticks, or if held-object transitions do
not invalidate jobs and full pots remain idle.  In that case the next revision
should add a partner-specific handoff protocol or relax the gate lock after a
verified failed interaction—not restore raw per-tick model directions.

## Jev interface and fallback policy (tier 2)

Every Jev request carries a strict JSON state with:

* `tick`, `layout`, and `orders`;
* `self.position`, `self.facing`, and `self.held`;
* partner position/facing/held object, last action, and whether it is near the
  current target;
* each pot's position, phase, ingredients, missing ingredients, and derived
  cooking ticks remaining;
* the committed job kind/target/lock and blocked-tick count; and
* `traffic.yield_ticks`, the current Jev mode hint, Astra's accepted role mode,
  and a short list of code-built legal medium options.

The batched questions are:

1. `phase: Choice` over `continue`, `delivery`, `supply`, `handoff`, and
   `yield`.  It recognizes a strategic phase; it is not permitted to invent an
   ID, arithmetic result, or primitive action.  A selected phase is accepted
   only if it is one of the closed-world values and its returned probability is
   at least `0.55` (stand-in answers without a distribution are still bounded
   by the closed-world check), and it expires after 14 ticks or a job epoch.
2. `yield: Noul`, true only for an explicitly urgent partner soup/dish or a
   partner occupying the sole approach.  A true probability of at least `0.72`
   gives an empty-handed chef at most two `stay` ticks.  It never interrupts a
   chef carrying soup, dish, or an ingredient.

Questions are submitted with `ctx.spawn`, one outstanding request at a time,
with a one-second Jev timeout.  Responses are tagged by the committed job
epoch; a late answer for an invalidated job is discarded.  All exceptions,
missing answer fields, bad probabilities, and budget/deadline closure leave
the deterministic code priority in control.  That priority is, in order,
deliver held soup, pick up ready soup, start a full idle pot, fill a compatible
pot, retrieve/stage ingredients, fetch/stage a dish, then wait.  `ctx.act` is
always given an advertised action name, and a final `stay` retry protects the
pre-flight stand-in from arbitrary model text.

Astra's short JSON has only `mode`, `side`, and `traffic`; its answer is parsed
closed-world and is a weak tie-breaker.  It is invoked once at reset, via
`ctx.spawn` with a bounded 90-second timeout, rather than being re-called at
every event: the measured System-2 latency makes repeated replanning too late.
A verified persistent block is sent to the fast Jev traffic interface instead;
the deterministic executor remains the final recovery policy.  Thus System-2
can revise the role allocation when it finishes, while the real-time controller
keeps running if it never does.
