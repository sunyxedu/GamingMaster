# Candidate 0012 proposal — safe cycle re-entry routes selected per food

## Parent and diagnosis

The direct parent is `candidates/0006/`; the strongest comparison is
`candidates/0009/`, with `candidates/0011/` as the failed endgame warning.
0006's model/route portfolio removed stale ticks but still ended five of six
training episodes in self-collision, demonstrating that deadline correctness
alone is not a long-horizon body invariant.  The verified cycle family is
retained because it is the only safety boundary that has survived every
reference episode: 0009 scored 119–129 with zero collisions and zero stale
ticks.  Its trace also shows the remaining failure:
seed 5 reached score 119 at tick 3944 and then spent ticks 3945–4000 on the
food at `[2, 1]`, ending with 22 free cells.  In the same traces the planner
repeatedly logged `not_shorter_than_cycle` near the deadline (for example
seed 0 at ticks 3981, 3983, …, 3999).  The controller had a valid body but
was forced to wait for the active cycle's ordering.

Candidate 0011 tested the tempting opposite—an unrestricted short food BFS—but
its rescue deliberately left the invariant and then used an unproved fallback;
its rescue was followed by self-collisions in four of six episodes.  That does
not test whether an unrestricted approach is safe when its *post-growth state*
is required to rejoin a validated cycle.

## New hypothesis

A food route does not need to preserve the current Hamiltonian order at every
intermediate cell.  It is safe to leave that order for the finite, known-food
segment if all of the following are verified in code:

1. every intermediate transition uses exact wall, reverse, body, tail-vacating,
   and growth semantics;
2. the route is attached to the exact observed body, heading, and food, and
   every later action is dropped if the live state is not its simulated prefix;
3. the post-food body is in strict head-to-tail order on a separately validated
   Hamiltonian cycle.

At the food transition the snake is therefore immediately returned to the
cycle invariant, after which the normal cycle controller is resumed.  This is
structurally different from 0009's route search (which filtered every search
state by one fixed cycle) and from 0011's rescue (which had no re-entry proof).
The controller also switches the active cycle only when a new food appears;
this tests whether food-specific order gives the portfolio benefit of 0010
without its per-tick cycle oscillation.

The hypothesis is confirmed by a material score increase over 124.333 while
retaining zero wall/self collisions, zero stale ticks, and time-limit
completion.  It is refuted if re-entry routes are rare and scores stay at the
0009 plateau, or if any accepted route reaches a body that does not match its
simulated endpoint.  A collision or an unexpected state-prefix mismatch would
refute the safety part even if food count rose.

## What changed

* The cycle builder generates a bounded family of individually checked
  Hamiltonian cycles (bounds, uniqueness, every edge, and closing edge).
* A food boundary selects the best currently compatible cycle once.  It is not
  reselected every tick, so the action stream has a stable invariant while
  traversing a target.
* A background A* worker first gives the active cycle a small, invariant-
  preserving slice (retaining 0009's safe throughput layer), then searches the
  full moving body without a cycle-order filter.  The permissive route is
  still capped below one tick and never blocks `ctx.act`.
* A result is accepted only when it is no longer than the active-cycle
  greedy route (or when no such route exists), and only its exact live-state
  suffix is executed.  Equal-length routes are retained only when they can
  re-enter a different verified cycle, which may improve the next target.  At
  the food step the controller adopts the worker's verified endpoint cycle.  A
  late, malformed, longer, or mismatched result is discarded and the cycle
  action continues.
* No model call is made.  On this 100 ms game clock the recorded Jev latency
  spans more than one tick, and the cycle controller already outperformed all
  model-driven predecessors.  Deterministic search supplies the facts that a
  model would otherwise have to count or prove.

## Tier-2 bridge and fallback

There is intentionally no System-1/System-2 interface in this candidate:

* state schema: none sent to Jev;
* questions: none;
* thresholds: none;
* answer-to-action bridge: none.

The tier-2 bridge is instead a bounded `ctx.spawn` search with a monotonic
0.070 s worker deadline.  The foreground always has an action: first the
currently active cycle's one-step action, then a newly selected compatible
cycle, then an exact one-step reachable-area/food fallback.  It never waits
for Astra, Jev, or the worker and posts one absolute direction before waiting
for the next tick.

The worker's state schema is explicit code data: `(origin_body, origin_heading,
origin_food, rows, cols, first_action, baseline_length, validated_cycle_family)`.
Its search question is the mechanical predicate “does this exact simulated
route reach the visible food no later than the baseline and leave a strict
cyclic head-to-tail ordering on any validated cycle?”  The only accepted answer
is a route plus endpoint cycle; there is no probabilistic confidence threshold.
Freshness is checked twice: the route begins with the action already posted,
and `_route_suffix` must match the current complete body, heading, and food
before every suffix is used.  Any timeout/exception/result parse failure is a
normal discard, never an error path.  If a future remains late, the cycle
fallback continues and the stale result cannot steer a later state.

For a future model-enabled revision, Astra could choose the cycle-family size
or whether to permit the finite re-entry search after observing traces; Jev
could judge only among code-provided route candidates.  Neither is allowed to
emit a cell action or weaken the endpoint proof.
