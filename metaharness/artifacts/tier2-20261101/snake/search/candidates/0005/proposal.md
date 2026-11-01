# Candidate 0005 proposal — verified routes, asynchronous judgment

## Parents and diagnosis

This is a crossover of `0000-tier2_jev_only` and `0002-tier2_idle`.

`0000` did at least post actions informed by Jev (`Jev→act = 100%`), and its
latency was about 0.15 s, but it asked for an unconstrained action while the
clock was already moving.  Its seed-3 trace is a concrete failure: at tick 0
the snake was headed LEFT at `[6,5]`, while the food was `[11,10]`; the first
Jev answer was `NOOP` after 0.1827 s.  The trace records tick 1 as `NOOP` and
then tick 2 as another `NOOP` before a later answer is available.  At tick 6
it is at `[6,0]`, has `safe_directions_now = []`, and dies with
`terminal_reason = wall_collision`.  The final score is zero.  The same
pattern appears in the other seeds: the raw observation plus a generic “which
action is best?” question does not give Jev enough time or enough structure to
turn a full state into a safe multi-tick route.

`0002` is the robust part of the crossover: it makes no model calls, never
fails pre-flight, and lets the adaptor's autonomous clock run without a bad
late action.  It is nevertheless only a passive baseline; every training run
ends in a wall collision and scores zero.  The useful combined hypothesis is
therefore not “ask Jev more often” or “idle longer,” but **make survival and
exact legality deterministic, then use models only as advisory judgments over
already executable alternatives**.

## Hypothesis

The best candidate fails before strategic reasoning matters.  A one-tick
real-time Snake controller cannot afford to ask a model to discover collision
semantics, count a body, or produce a path.  A controller that simulates the
published transition locally, queues a short route to food only after a
post-growth tail-access test, and falls back to the moving tail/free space
should survive the opening and earn food.  Slow model answers can improve a
borderline choice, but a late, malformed, low-confidence, or stale answer must
be equivalent to the idle parent's safe fallback rather than a crash or an
unsafe action.

Evidence that would confirm this hypothesis is a substantial increase in food
count and ticks survived while model answers are often discarded as stale but
there are no wall/self-collision errors.  It would be refuted if the
code-validated controller still dies early despite selecting only the reported
one-step-safe transitions, or if safe food routes systematically trap the
snake; that would motivate a stronger time-expanded/cycle-entry planner rather
than more Jev calls.

## What changed

`harness.py` makes the structural change in the controller, not in a prompt:

* `_transition` implements absolute steering, ignored reverse requests,
  simultaneous tail vacating on non-food moves, and retained tail on growth.
  Every candidate action is checked against bounds and the occupied set before
  it can be posted.
* A bounded A* search uses the complete head-first body and heading as its
  state.  It simulates every move to the current food and rejects a route when
  the resulting grown body cannot statically reach its tail or has no connected
  free space.  This is deliberately a safety filter/ranking heuristic, not a
  claim of an infinite-survival proof.
* If food is not safely reachable, a dynamic-body BFS follows the old tail while
  it moves.  If that is unavailable, a one-step scorer prefers tail access,
  connected area, and then food progress.  A validated two-row-strip
  Hamiltonian cycle is also available on even rectangular boards and is used
  as a late, order-checked fallback; it permits gaps in cycle indices but never
  assumes that an arbitrary shortcut preserved the invariant.
* The controller retains a predicted next state only while the observed body,
  heading, and food agree with it.  Any missed/stale tick invalidates the plan
  and replans from the fresh observation.  Thus the real-time loop posts a
  deterministic action immediately, waits for one game tick, and never blocks
  that loop on Astra or Jev.
* Astra is spawned only after the deterministic controller has survived 20
  ticks.  It is asked for a short risk mode (`conservative`, `balanced`, or
  `aggressive`), not coordinates or actions.  Its text is parsed defensively;
  garbage leaves the balanced default.  This tests whether low-frequency
  System-2 supervision can change the food-route conservatism without
  endangering the opening.
* Jev is spawned only at a genuine junction with at least two exact legal
  candidates.  Its state is a compact JSON summary of the current tick,
  head/tail, direction, food, length/free cells/score, the Astra risk mode, and
  code-derived candidate facts.  It is never asked to search a grid or do
  arithmetic.  The controller accepts an answer only if both reordered Choice
  questions agree, the answer is still legal on the current observation, its
  selected-option probability is at least 0.34, and it is no more than one tick
  old.  Otherwise the deterministic action remains in force.

This keeps the passive parent's robustness as the fallback while retaining the
active parent's use of fresh System-1 judgments where they can actually be
useful.

## Jev state schema and questions

Each request sends strict JSON of the following shape (no raw prose archive or
irrelevant history):

```json
{
  "rule": "Choose only among code-legal candidate actions...",
  "observation": {
    "tick": 12, "head": [6, 4], "tail": [6, 7],
    "direction": "UP", "food": [9, 1], "length": 8,
    "free_cells": 136, "score": 5
  },
  "policy": "balanced",
  "candidates": [
    {"action":"RIGHT", "food_now":false, "tail_access":true,
     "space":"large", "near_food":"near", "keeps_heading":false}
  ],
  "deterministic_preference": "RIGHT"
}
```

The candidate descriptions explicitly say whether the action passed exact
transition checks, whether it eats and passed the post-growth filter, whether
tail access is available, and whether the free component is large or limited.
The actual complete body remains in code, where it is needed for simulation;
Jev sees only the derived facts relevant to its judgment.

One batched request contains:

1. `pick_a`: a `Choice` over the candidate direction names, preferring a
   post-growth-safe food move and otherwise tail access/space;
2. `pick_b`: the identical Choice with options in reverse order, to reduce
   option-order bias;
3. `urgent`: a `Noul` asking whether continuing the current heading needs an
   immediate turn.  This is logged as evidence but is not allowed to override
   the exact legal-action filter by itself.

The answer-to-action bridge is intentionally narrow: only an agreed
`pick_a == pick_b` that names a current candidate can replace the deterministic
choice. A missing `.value`, arbitrary stand-in output, unknown label, absent
probability, probability below 0.34, or stale tick is ignored. The candidate
set itself is recomputed before applying the answer, so an old Jev choice cannot
enter a newly occupied cell. Jev calls are background futures with a 0.6 s
call timeout; a late response is discarded rather than awaited.

## Astra revision trigger and fallback

The initial Astra call is a finite supervisor call, not part of the tick path.
Its interface should be revised only when traces show a repeated class of
failure: for example, food paths pass the static tail test but produce a
small-area trap, or Jev repeatedly agrees on a route that the next observation
invalidates.  In that case Astra should alter the risk-mode thresholds or
candidate facts, not write cell-by-cell actions.  If Astra is unavailable or
returns non-JSON prose, the balanced policy and exact code remain active.

The final fallback ladder is therefore:

1. current validated food route;
2. a newly simulated safe food route;
3. an order-validated Hamiltonian step at high occupancy;
4. a dynamic route toward the moving tail;
5. the legal one-step action with maximum tail access and connected area;
6. current heading/`NOOP` only if the observation is already terminal or
   malformed.

No model call is required for any rung.  This is the principal safety lesson
from the two parents: do not let a slow model answer become the clock.

Pre-flight cause: `_maybe_start_jev` referenced an undefined `heading` local while constructing its JSON state; it now derives the direction defensively from the observation.
