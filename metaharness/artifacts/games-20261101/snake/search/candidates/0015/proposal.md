# Candidate 0015 proposal — late-only safe cycle re-entry

## Parent and diagnosis

Parent: `candidates/0013/`.  I keep its validated Hamiltonian-cycle
invariant, exact moving-body transition, asynchronous planner, exact-prefix
matching, and constant-time fallback.

The parent’s worst episode is seed 5.  Its trace reaches score 119 at tick
3944 with food `[2, 1]`, then repeatedly starts a worker and discards it as
`not_shorter_than_cycle` at ticks 3945, 3947, …, 3999.  It remains alive and
fresh throughout, but the final observation at tick 4000 still has food
`[2, 1]`, score 119, and 22 free cells.  This is not a timing or collision
failure: the parent has spent the remaining clock following one fixed cyclic
ordering whose best safe route to this food is too slow.

## Hypothesis

The fixed-cycle restriction is the remaining throughput bottleneck only after
the board is substantially occupied.  A finite route may safely leave the
current cycle while pursuing the visible food if its every transition is
simulated from the exact body and its *post-food* body is proven to be in the
order of some independently validated Hamiltonian cycle.  The parent does not
try this: it rejects every route that is not shorter while staying on its
single active cycle.

I therefore add a **late-only** re-entry search.  Before the board is crowded,
0013 remains byte-for-byte in spirit: it uses the current cycle and its
cycle-order-preserving food search.  Once `free_cells` is at most one quarter
of the board, a re-entry search is allowed for a food target after the normal
cycle plan has failed.  It accepts only a route strictly shorter than the
current-cycle baseline, attached to the exact post-action state, with exact
wall/reverse/body/tail/growth simulation at every edge and a validated cycle
at the food endpoint.  The cycle is adopted only after that food transition;
while the finite route is active, every prefix is still checked against the
simulated body, so it has no collision proof gap.

The hypothesis is confirmed by more accepted late food routes and a score
above 125.0 (especially improvement on seed 5), while retaining 0 stale ticks,
0 wall/self collisions, and 4000-tick completion.  It is refuted if the
late search rarely finds a strict improvement, if scores remain at the same
plateau, or if any endpoint/prefix check or cycle adoption causes a collision.
A lower score before the occupancy gate would refute the isolation claim.

## Targeted change

The only policy change is a guarded second planner.  The existing parent
planner remains the first choice and the existing cycle action remains the
fallback.  For the late gate, a bounded A* searches unrestricted intermediate
body states, but never ignores exact Snake transitions.  A goal is accepted
only on the food move when the grown body is ordered on one of the generated
and independently validated cycle variants.  Results are installed only at an
exact `(body, heading, food)` prefix and only when strictly shorter than the
parent cycle route.  During a re-entry suffix, the foreground rechecks each
transition; after food it switches to the proven endpoint cycle.  A mismatch,
late result, malformed result, or non-improving route is discarded and the
parent cycle action is posted immediately.

No seed, food sequence, or training trace is encoded.  The occupancy gate is
computed from observed `rows`, `cols`, and `free_cells`, so it is not a seed
rule.  The worker is bounded and runs through `ctx.spawn`; the action loop
never waits for it.

## Tier-2 interface and fallback

This candidate makes no Astra or Jev calls.  Its System-1 state schema and
question set are empty, with no confidence threshold or model-to-action
bridge.  On this symbolic board, legality, body order, route freshness, and
endpoint re-entry are exact code predicates; adding Jev’s measured
0.14–0.16 s latency would put the choice behind a tick.  Garbage model output
therefore cannot affect pre-flight.

The deterministic bridge is: (1) install an exact, strictly improving
late-reentry suffix if available; (2) otherwise install the parent’s
cycle-order-preserving food suffix; (3) otherwise execute the current
validated cycle action; and (4) if no cycle is compatible, use the parent’s
one-step legal/reachable-area fallback.  A re-entry route is never retained
past a food transition, and its endpoint cycle is adopted only after the
foreground rechecks the food transition.

A future Astra revision would be justified only if traces show that the
late-gated endpoint proof is systematically too restrictive or that a
specific validated cycle family is missing compatible endpoints.  Astra
should select among verified route policies, never emit an unverified cell
sequence.
