# Candidate 0006 proposal — bounded route portfolio with semantic Jev advice

## Parent and diagnosis

Declared parent: `candidates/0001-tier2_astra_only/`.  I also used the best
available comparison, `0005`, because it is the first candidate that actually
scores food.

The parent demonstrates the fatal division-of-labour mistake: it blocks the
real-time loop on Astra.  Its seed-0 trace begins with an Astra request at tick
0, but the snake receives no action and reaches the wall at tick 8 while that
call is still in flight.  `0005` fixes the broad architecture and reaches a mean
score of 6, but its trace identifies a second bottleneck.  In seed 0 the state at
tick 30 is `[5,11]`, direction `RIGHT`, with safe directions `UP` and `DOWN`;
the next tick is nevertheless logged as `action: RIGHT, fresh: false`, and the
snake dies at `[5,11]` attempting `[5,12]`.  In seed 2, after the food at tick
66, ticks 67–70 repeatedly apply stale `DOWN` before the controller posts again;
the snake then reaches `[11,11]` and dies on tick 75.  Thus the remaining
failure is not primarily Jev quality: `0005` can compute a safe action, but its
body-state A*/tail searches occasionally take longer than one 0.10-second tick.
Its Jev answers are also mostly discarded because a one-cell action judgment
arrives about two ticks after the state it judged.

## Hypothesis

A real-time Snake controller should make **every tick decision from a bounded
route portfolio**, rather than run an unbounded dynamic-body search in the tick
path.  A conservative static BFS to food, followed by exact simulation of the
whole finite route and a post-growth tail/area filter, should preserve most of
`0005`'s food performance while eliminating the stale wall deaths caused by
search latency.  When no such route exists, a one-step tail/space scorer and an
order-validated Hamiltonian step are cheap enough to run before the deadline.

Jev should not select a direction.  Its 0.15–0.23 s response spans multiple
reference ticks, so it should choose between code-produced **route families**
(`FOOD_FAST`, `FOOD_WIDE`, `CYCLE`, `TAIL`, or `SPACE`) and its answer should be
cached only as a short-lived preference.  Code rechecks that family against the
latest body, food, legality, and freshness before using it.  Astra should remain
a rare late supervisor for an actually unusual high-occupancy/no-food-route
state, never an opening action source.

Evidence that would confirm the hypothesis is: (1) stale ticks fall sharply
relative to `0005`, especially no repeated stale action at a wall; (2) food
count remains at least near 6 on the training seeds; and (3) a longer run enters
cycle-compatible safety rather than ending on a wall.  It would be refuted if
fast static routes systematically enter self-traps despite exact simulation, or
if the tail/space fallback survives but eats substantially fewer foods; then the
next change should add an incremental time-expanded planner, not increase model
call frequency.

## What changed structurally

* The tick path is now a **bounded portfolio**, not a body-state A*/BFS search.
  For each legal first turn, a small position/heading BFS finds a conservative
  route to food while treating the current body as blocked (the tail may vacate).
  `_simulate_route` applies the exact food/non-food transition to every route
  step, and only post-growth-safe routes enter the portfolio.  There is no
  potentially thousands-of-expansions search or long tail route in the real-time
  loop; an 8 ms local budget is an additional guard.
* Food routes have two deterministic variants: the shortest validated route and
  the validated route with the largest post-growth free region.  If neither is
  available, `_best_nonfood` evaluates only the next exact transition and ranks
  moving-tail access, flood-fill area, food distance, and straightness.
* A simple even-by-even Hamiltonian cycle is built and independently checked for
  uniqueness and adjacent closure.  It is used only when the complete current
  body is in tail-to-head cycle order; arbitrary shortcuts never activate it.
  Odd dimensions and incompatible entries use the finite safety fallback rather
  than assuming a cycle exists.
* The first action is selected before any model request.  A predicted one-tick
  successor is compared with the next observation; a skipped or stale state
  clears the remaining route.  Every action is checked by `_transition` one more
  time, including the observation's `safe_directions_now` fallback.

This tests a different structural hypothesis from `0005`: model calls are
semantic route-policy hints and deterministic code is a deadline-bounded
controller.  A low-confidence or malformed model answer is harmless.

## Jev state, questions, and bridge

Each Jev request sends strict JSON like:

```json
{
  "task": "Choose a route family, not a cell action. Code rechecks every transition.",
  "rules": {
    "absolute_actions": true,
    "reverse_requests_ignored": true,
    "tail_vacates_on_nonfood": true,
    "tail_stays_on_food": true,
    "only_listed_modes": true
  },
  "observation": {
    "tick": 24,
    "head": [0,10], "tail": [3,9], "direction": "RIGHT",
    "food": [5,11], "occupancy": "low_occupancy",
    "free_space": "large", "score": 2
  },
  "route_facts": {
    "FOOD_FAST": {"food_route": true, "post_growth_tail_access": true,
                  "space": "large", "horizon": "short_route"},
    "FOOD_WIDE": {"food_route": true, "post_growth_tail_access": true,
                  "space": "large", "horizon": "short_route"},
    "TAIL": {"food_route": false, "post_growth_tail_access": true,
             "space": "open", "horizon": "one_step"}
  },
  "available_route_families": ["FOOD_FAST", "FOOD_WIDE", "TAIL"]
}
```

The actual body, route cells, collision arithmetic, and candidate transitions
stay in code.  `route_facts` contains only code-derived named buckets and
predicates.  Candidate descriptions say which exact predicates passed and
whether a route is shortest, wide-space, cycle-validated, or tail-unwinding.
The request batches two independent `Choice` questions:

1. `route_a` chooses the best listed route family;
2. `route_b` asks the same choice with the option order reversed, reducing
   first-option bias.

The bridge accepts an answer only when both `.value` fields agree, the value is
still present in the current code-generated portfolio, its option probability is
at least 0.42, the judged food and score are unchanged, and the response is at
most four ticks old.  It stores only the semantic family for 18 ticks; it never
stores a stale direction.  `FOOD_FAST` is the default.  A `TAIL`/`CYCLE` answer
cannot veto a safe food route at low occupancy, because score is the objective;
at high occupancy it can select the conservative fallback.  If the response is
late, malformed, disagrees, or has low probability, the deterministic ranking
continues.  The request timeout is 0.45 s and is backgrounded, so the game
clock is never awaited on its behalf.

## Astra role and revision trigger

Astra is not called in the opening and is never on the tick path.  The harness
starts at most one background Astra supervisor call only after tick 120 when
there have been at least seven consecutive observations without food, or when
occupancy is high and no validated cycle entry is available.  It receives the
compact portfolio/state and may return only `FOOD_FAST`, `FOOD_WIDE`, `CYCLE`, or
`TAIL`.  Invalid prose is ignored and all transitions remain code-validated.
This is the revision trigger: a persistent no-food/high-occupancy trace is
structural evidence that the route-family interface needs a more conservative
policy.  Ordinary short reference episodes use no Astra call, avoiding the
parent's slow opening failure and preserving the system-2 tie-break advantage.

The fallback ladder is therefore:

1. the already validated food route, with its next edge rechecked;
2. a newly validated shortest/wide food route;
3. an order-checked Hamiltonian step;
4. the exact one-step moving-tail/area choice;
5. a locally rechecked `safe_directions_now` entry or any legal successor.

None of these depends on a model answer arriving before the next tick.
