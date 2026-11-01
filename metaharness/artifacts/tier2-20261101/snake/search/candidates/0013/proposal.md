# Candidate 0013 proposal — one-tick-ahead unrestricted food search

## Parent and diagnosis

Parent: `candidates/0009/`.

I keep 0009's architecture: a validated Hamiltonian cycle is the safety
invariant, a bounded full-body worker searches for shorter food routes, and the
foreground revalidates every action before posting it.  I change only the
worker's origin and first-action constraint.

The parent is safe but leaves throughput on the table.  Its seed-5 trace has
1,504 `food_plan_started` events but only 35 `food_plan_ready` events; the
late-game trace repeatedly shows, for example, `tick: 3980` starting a plan and
`tick: 3981` discarding it with `reason: "not_shorter_than_cycle"`, continuing
through ticks 3999.  At tick 4000 it is alive and collision-free but has only
`score: 119`, `free_cells: 22`, and `food: [2, 1]`.  Seed 1 shows the same
shape (1,488 starts, 29 ready, final score 120 and 21 free cells).  These are
not timing failures: both episodes have zero stale ticks and end at the time
limit.  The parent code also explicitly pins `_food_search`'s `first_action`
to the action already selected by the greedy cycle controller.

## Hypothesis

The worker is spending its first search step re-proving a move that has already
been posted.  Therefore it cannot discover a shorter safe route whose useful
turn begins on the *following* tick unless that route happens to share the
current greedy first action.  If the worker instead starts from the exact
post-action state and leaves its next action free, more cycle-compatible
shortcuts should be found without changing the safety boundary.

The hypothesis predicts more accepted food plans and a mean score above 0009's
124.333, while retaining 0 stale ticks and 0 wall/self collisions.  It is
refuted if accepted-plan frequency rises without reducing food-arrival time,
if score is no better than the parent, or especially if any route causes a
cycle-order violation/collision or leaves a worker running after return.

## Targeted harness change

After selecting and posting a non-food action at observation tick `t`, the
controller applies the exact local transition in code.  That produces the
known state at tick `t+1` (body, heading, and unchanged food), so it starts the
background search from that predicted state with `first_action=None`.  At the
next observation the existing exact-prefix matcher can install the route
immediately; if the worker is late, the state has changed, or food was eaten,
the result is discarded.  Each installed action is still simulated again and
must preserve the complete head-to-tail cycle ordering.

The current action remains the parent's cycle/route action, so this change does
not delay the clock or permit a speculative move.  Food moves do not start a
search because the next food is private RNG output.  The worker remains
bounded and asynchronous, and no seed or food sequence is used.

## System-1 and System-2 interface

This candidate intentionally makes no model calls.  The Jev state schema is
empty, the question set is empty, and there is no typed answer-to-action
bridge, confidence threshold, or stale-answer cache.  On this verified,
obstacle-free rectangular board, exact body simulation and cycle-order checks
are more reliable than asking Jev to choose among already validated moves;
Jev's measured latency would also span a tick.  Freshness is handled by the
observation tick and exact `(body, heading, food)` prefix validation rather
than by a model answer.

The fallback is deterministic: use an installed route only after current-state
transition and cycle-order checks; otherwise use the retained cycle action;
if cycle entry is invalid, use 0009's one-step exact transition/reachable-area
fallback.  Malformed observations and unavailable routes never require model
output parsing.

Astra should revise this interface only if traces show a repeated, structural
failure that code facts do not explain—such as cycle entry failing on a
permitted board shape, an unexpected rule/obstacle, or invariant-compatible
shortcuts systematically stalling.  Ordinary `not_shorter_than_cycle`, late
worker, or state-mismatch events are evidence for improving deterministic
search, not for blocking the real-time loop with Astra.
