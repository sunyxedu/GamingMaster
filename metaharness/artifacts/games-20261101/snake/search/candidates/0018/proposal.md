# Candidate 0018 proposal — food-epoch certificates with durable Jev policy

## Parent and diagnosis

Parent: `candidates/0014/`, with the collision-free baseline and post-action
planning mechanism from `0013`.

The archive establishes two different facts that should not be mixed.  The
fixed cycle controller in `0013` is safe and reaches the limit in all six
training episodes (mean 125.0, zero stale ticks and zero collisions), but its
food planner repeatedly reports `not_shorter_than_cycle` near the end.  In the
seed-5 trace, the snake is alive at score 119 with food `[2, 1]` and 22 free
cells while searches at ticks 3951, 3953, 3957, and onward are discarded.  The
same trace has 1,502 starts but only 35 accepted routes.  This is a throughput
ceiling, not a deadline failure.

`0014` changed the route space and added Jev/Astra arbitration, but its trace
shows the model interface is attached to ephemeral routes: 4,804
`prefix_or_food_mismatch` discards, only 29 of 49 started Jev judgments applied,
and the mean fell to 122.833.  A representative seed-3 call starts at tick 23,
but the answer is rejected at tick 25 for low probability; another answer is
rejected at tick 67 because its route prefix is already stale.  The model is
being asked to choose an object whose useful lifetime is shorter than Jev's
0.16--0.23 s response.

The negative result from `0017` is also important: certifying only the food
endpoint led to 5/6 collision episodes and 19 stale ticks.  This candidate does
not repeat that relaxation.

## Hypothesis

The missing structural layer is a **moving certificate**.  A Hamiltonian cycle
need not be the same cycle throughout a food approach, but every intermediate
body must still be ordered on at least one independently validated cycle.  A
finite route with that property has two protections:

1. every edge is simulated with exact wall, reverse, body, tail-vacating, and
   growth semantics; and
2. every non-food intermediate state and the grown food endpoint retain a
   validated Hamiltonian continuation.

This is less restrictive than `0013`'s single-cycle search, but stronger than
`0017`'s endpoint-only certificate.  It should expose safe shortcuts that move
between compatible cycle certificates without opening the collision gap that
hurt `0017`.

A second part of the hypothesis concerns model division of labour.  Jev should
not arbitrate a route suffix.  Code now chooses a stable cycle and computes a
small whole-food shortlist; Jev gives a low-frequency `SPEED`, `BALANCE`, or
`FLEX` policy tie-break that can remain valid for an entire food epoch.  The
answer changes whether the background search may accept an equal-length route
with extra certified continuations; it never directly posts an action or
revives a stale route.  Astra is deliberately not put in the 100 ms loop: on
this fully specified symbolic board, its only useful role would be to revise
the policy schema after repeated unexplained failures, not to do arithmetic that
the adaptor exposes exactly.

The hypothesis is supported if dynamic-certificate routes become ready in the
late traces, the mean exceeds 125.0, and all episodes retain zero stale ticks
and zero wall/self collisions.  It is refuted by any collision from an
installed route, by a certificate mismatch after a food move, by a large
increase in `prefix_or_certificate_mismatch`, or by a score no better than the
parent despite a materially larger search cost.  A lower score with no
collisions would mean that the certificate bank or policy objective is too
conservative or that the extra cycle flexibility does not improve food timing.

## What changed in `harness.py`

* The code builds a bounded, independently checked bank of rectangular
  Hamiltonian cycles.  It rejects a candidate unless it contains every cell
  exactly once and every closing edge is orthogonally adjacent.
* A food epoch selects one stable certificate using a complete simulated cycle
  route from a small, bounded candidate bank, rather than changing tours on
  every tick.  The current certificate is retained when still compatible;
  `SPEED`/`BALANCE` require a strict shortcut improvement, while `FLEX` also
  permits an equal-length shortcut when post-food certificate/open-area facts
  improve.  The foreground evaluates only a small shortlist so it stays under
  the action deadline; the larger certificate bank is searched in the worker.
* The asynchronous worker starts from the exact predicted state after the
  action that was just posted, leaving its next action unconstrained.  Its
  search may change certificate between steps, but it enqueues only states for
  which `_certificates(body)` is nonempty.  The grown food state must also have
  a certificate.  It accepts a route only if it is shorter than the epoch's
  conservative route, or equal length with more continuation certificates.
* The foreground replays every route prefix with `_advance` immediately before
  posting it and rechecks certificate membership on the latest body.  An exact
  state mismatch, changed food, late worker, malformed result, or missing
  certificate discards the route and returns to the current certificate.  On a
  food move only the checked endpoint certificate is adopted.
* Jev is started at most once per food epoch, after code has produced a small
  JSON shortlist of certified whole-route metrics.  The two reordered Choices
  select a durable policy, not a direction.  A response is accepted only when
  both labels agree, both probabilities for that label are at least 0.55, and
  the food target is still the one judged.  The policy is applied only through
  a fresh code recomputation and worker baseline; no answer is used as a stale
action.

No seed, food sequence, or private game state is encoded.  The only numeric
limits are board-size/search-budget limits derived from the public profile.
Workers have monotonic deadlines and check `ctx.done`, so they do not strand
an episode.

## System-1 state schema, questions, and bridge

Each Jev request carries strict JSON of this form:

```json
{
  "purpose": "choose a durable Snake food-policy tie-break; do not return an action",
  "food_epoch": {"tick": 123, "target_present": true},
  "current_policy": "BALANCE",
  "candidates": [
    {
      "certificate": 7,
      "route_steps": "medium",
      "post_food_certificates": "large",
      "post_food_open_area": "very_large",
      "is_current_certificate": false
    }
  ]
}
```

All coordinate arithmetic, route lengths, certificate checks, and area counts
are computed by code.  Jev receives only semantic buckets and the fact that
all candidate transitions have already been mechanically certified.  The two
questions are independent reordered `Choice`s over:

* `SPEED` — fewest verified food steps, flexibility only as a tie break;
* `BALANCE` — short route with flexibility/open area as a close-call tie break;
* `FLEX` — route within one step of shortest when it leaves materially more
  post-food continuation choices.

The answer bridge is intentionally delayed and durable.  A valid agreeing pair
with minimum selected-option probability 0.55 changes `controller.policy` and
sets a reselect flag.  At the next observation, code recomputes compatible
certificates and routes from the latest body before the policy affects an
action.  If the food changed, the target is stale; if values disagree, are not
one of the three labels, probabilities are low, the call times out, or the
stand-in returns garbage, the current deterministic policy remains in force.
There is no model-generated direction and no model-generated route.

The deterministic action fallback is, in order: an exact route whose current
prefix and next transition are revalidated; the current food-epoch cycle move;
a newly selected compatible cycle; and a one-step safe move ranked by food,
tail reachability, reachable area, and certificate availability.  Thus Jev
latency can reduce advice but cannot create a stale tick.

## System-2 boundary and future evidence

No Astra call is made in this candidate.  The board shape, action semantics,
full body, and safety invariants are public and exact, so System-2 arithmetic
would be slower and less reproducible than code.  Astra should be introduced
only if traces show a repeated structural anomaly that code facts cannot
classify: for example, a permitted non-rectangular/obstacle profile, a valid
board with no certificate bank, or many certified routes whose score trade-off
is not explained by route length and post-food flexibility.  It should revise
policy labels or thresholds, never bypass the per-step transition and
certificate checks.
