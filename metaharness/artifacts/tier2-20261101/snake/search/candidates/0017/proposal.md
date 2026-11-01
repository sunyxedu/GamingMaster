# Candidate 0017 proposal — certificate-at-food routing

## Parents and concrete diagnosis

Parents: `candidates/0009/` and `candidates/0013/`.

Both parents supply the mechanism that made the leaderboard safe: a validated
rectangular Hamiltonian cycle, exact tail-vacating simulation, exact
body/heading/food freshness checks, and a bounded worker so the 100 ms clock is
never blocked.  `0009` added full-body food search inside that invariant and
raised the mean to 124.333.  `0013` kept the invariant but started the worker
from the exact predicted post-action state with its next action unconstrained;
that isolated change reached 125.000 without stale ticks or collisions.

The remaining failure is visible in the worst parent trace, not a timing or
collision failure.  In `0013` seed 5, the controller repeatedly starts a search
for food `[2, 1]` at ticks 3962--3998; each result is discarded as
`not_shorter_than_cycle`, while every tick remains fresh and the snake reaches
the limit alive at score 119 with 22 free cells.  The corresponding parent
pattern is over a thousand route starts but only 35 accepted plans in that
episode.  The final body is still cycle-ordered, so the cycle proof is doing
its job; it is also preventing safe approaches that temporarily leave the
currently selected tour.  `0013`'s post-action origin fixed wasted first-step
search, but it did not change that geometric restriction (and its +4 seed-4
improvement did not generalize).

## Hypothesis

A route can leave the active Hamiltonian cycle safely for a finite, known
approach if every transition is simulated with the real body and the
post-growth endpoint is certified as ordered on a validated Hamiltonian cycle.
The cycle need not be an invariant on every approach state: the route itself is
the safety certificate until food, and the endpoint cycle is the continuation
certificate.  This should expose shorter food paths in the late game while
retaining the parent fallback whenever a worker is late or cannot prove
re-entry.

The new harness therefore combines both parent timing mechanisms (search from
the predicted next state and exact prefix matching) with a structural change
in the safety boundary (cycle membership is required at food, not at every
pre-food node).  It also chooses the active cycle only at a food boundary by
simulating its complete conservative route, rather than selecting a cycle from
only the first action or oscillating a portfolio every tick.  Search results are
still accepted only when strictly shorter than that cycle route, so the change
cannot knowingly slow the current food objective.

Evidence that would confirm the hypothesis is a higher mean food count and
more `food_route_ready` events, especially in the late-game traces, with zero
stale ticks, zero wall/self collisions, and no pending worker after return.
Evidence that would refute it is any collision during an installed route, a
cycle-certificate mismatch after eating, or a score no better than `0013`
while the relaxed search consumes most of its budget.  A lower score with no
collisions would specifically mean that endpoint re-entry is too restrictive or
that route ranking needs revision; it would not justify weakening the exact
transition checks.

## Harness changes

* `_advance` is the only transition primitive.  It models absolute steering,
  ignored reversals, simultaneous tail vacating on non-food moves, and retained
  tail on growth.
* The cycle family is generated and independently checked for bounds,
  uniqueness, and adjacency.  A body is accepted as ordered only when all
  head-to-tail cycle gaps are positive and their total span is less than one
  tour.
* At a new-food boundary, `_choose_cycle` evaluates the full simulated greedy
  food route for each compatible reflected row/column cycle.  The chosen cycle
  is stable until the next food or a verified route endpoint; this is a
  food-specific route-length choice, not a seed table.
* The worker is queued from the exact predicted
  body/heading/unchanged-food state, but is spawned only immediately *after*
  `ctx.act` has posted the current action.  This prevents a CPU-heavy worker
  from taking the GIL before the deadline of that action.  Its A* search is
  bounded by 85 ms, 18,000 expansions, and the current cycle route length when
  one is available.  Intermediate nodes need only pass `_advance`; a goal is
  accepted only if the grown body is ordered on one of the validated cycles.
  Among shortest goals it prefers larger reachable area and more open
  continuations.
* The foreground attaches the result to its origin food and exact route prefix.
  It re-simulates every route action immediately before posting it.  A route
  action that eats adopts its endpoint cycle; any mismatch, stale worker, bad
  result, or non-shorter route is discarded and the checked cycle action is
  used instead.
* The fallback is a one-step legal action ranked by food, tail access,
  reachable area, and heading continuity.  It is only a recovery path for
  malformed observations or profiles without a usable cycle, not a claimed
  infinite-survival proof.
* No private seed, food sequence, or model output is used.  Workers stop on
  `ctx.done` and have a monotonic deadline, so they cannot strand the episode.

## System-1 and System-2 interface

This candidate deliberately makes no Jev or Astra calls.  On the reference
board all fields needed for the decision are exact public geometry: the full
body, food, direction, dimensions, and transition semantics.  Asking Jev to
choose among already simulated actions would add roughly two ticks of latency,
while asking Astra to supervise a 100 ms junction would block neither the
clock nor the safety proof.  The evaluator should therefore report an empty
System-1 schema, zero S1/S2 calls, and all actions as deterministic rather than
model-informed.

If this controller were moved to an irregular board, the intended Jev state
would be a small JSON object containing `tick`, `head`, `tail`, `direction`,
`length`, `food`, `occupancy_bucket`, and a list of code-generated candidate
routes.  Each candidate would contain only named, code-verified facts:
`one_step_safe`, `all_steps_simulated`, `food_reached`, `endpoint_cycle_ordered`,
`route_shorter_than_cycle`, `reachable_area_bucket`, and `tail_access`.  Two
reordered `Choice` questions over those candidate IDs would be used only for a
non-lethal policy preference; agreement and selected-option probability at
least 0.60 would be required.  A timeout, disagreement, probability below
0.60, stale tick, or any failed precondition would select the deterministic
cycle/fallback action.  Jev would never be asked to find coordinates or prove
legality.

Astra should revise that interface only after traces show a repeated structural
failure that exact geometry cannot classify: for example, a permitted obstacle
variant, an odd board with no generated cycle, or many certified routes whose
score trade-off cannot be explained by route length/area.  Ordinary
`not_shorter_than_cycle`, prefix mismatch, and late-worker events are evidence
for deterministic search changes, not reasons to put System-2 in the real-time
loop.
