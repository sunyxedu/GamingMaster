# Candidate 0009 proposal — shortest food routes inside a verified cycle

## Parents and diagnosis

Parents: `candidates/0008/` and `candidates/0007/`.

The two parents established different halves of the problem. 0008 is the only
candidate with a complete safety result: it followed a validated 12x12
Hamiltonian cycle from tick zero, never had a stale tick or a collision, and
scored 118--126 foods (mean 121.167) at the 4000-tick limit. Its archived
seed-1 trace shows the important property: after selecting a cycle at tick 0,
every tick through tick 4000 has `fresh=true`, and the final state is a live
`time_limit` with 23 free cells. It also shows the headroom: the snake did not
fill the board, so cycle traversal time, not collision avoidance, is now the
limiting objective.

0007 contributed the other useful mechanism: an exact full-body transition
search in a background worker, together with origin-state/prefix validation.
That made route computation nonblocking, but its unrestricted routes and
heuristic fallback eventually broke the body invariant. In seed 1 its trace
reports `route_not_found` around ticks 209--215 and then repeated `DOWN` into a
self-collision at tick 216. The analogous seed-2 failure is repeated `RIGHT`
into the wall after route results were discarded. Thus I do not reuse 0007's
routes as executable merely because they pass a static tail test.

## Hypothesis

A shortest food route can improve on 0008's cycle-distance greedy shortcut
policy without sacrificing its safety if **every state in the route is required
to preserve the validated cycle ordering**. The cycle is the invariant; the
worker is only a speed optimizer inside that invariant. A route that is late,
malformed, or based on a different body is discarded, and the foreground
immediately returns to the same constant-time cycle action that produced
0008's zero-stale, zero-collision result.

The hypothesis predicts more food per 4000 ticks, with the same six/ six
`time_limit` completions and no wall/self collisions. It would be supported by
shorter tick intervals between food events and a score above 121.167 while the
cycle-order check remains true. It would be refuted by any collision while the
route is active, by a frequent route-discard loop that performs worse than
0008, or by a late worker that remains alive after an episode. A low score with
no collisions would mean the route search is not finding useful shortcuts, not
that the invariant is unsafe; in that case the next test should compare route
objective/tie breaking rather than weaken safety.

## What changed in `harness.py`

* It retains 0008's validated rectangular cycle generator, reflections,
  complete head-to-tail order check, exact tail-vacating transition, and
  constant-time cycle fallback. The observed body is revalidated before every
  action; a cycle is never blindly trusted after an unexpected state.
* It adds a bounded A* worker derived from 0007's body-aware planner. The
  worker searches `(body, heading)` states toward the current food, with the
  actual growth rule, and accepts a successor only if the complete post-move
  body remains in strict cycle order. It is capped by both 7,000 expansions and
  45 ms of monotonic time.
* A worker result contains its origin body, heading, food, and action route.
  The foreground accepts it only when the live body/heading/food is an exact
  prefix of that origin route. Each route action is simulated again immediately
  before posting and must still pass the cycle-order test. A food transition
  ends the route and forces replanning for the newly sampled food.
* The worker also reconstructs the current 0008 greedy route under the same
  forced first action. A searched route is installed only when it is strictly
  shorter than that baseline; equal or worse routes are discarded. This makes
  the experiment a local food-speed test rather than a silent regression of
  the already successful cycle policy.
* The worker is started after a decision and runs through `ctx.spawn`, with its
  first search action constrained to equal the action just posted. This makes a
  completed result an exact one-tick prefix instead of the common 0007-style
  stale route. No search, model call, or future wait can consume the 100 ms
  tick deadline. If it is late or fails, the next action is still the verified
  cycle action.
* If no rectangular cycle can be entered (for example an odd-by-odd profile or
  a malformed observation), the fallback tests exact one-step legality,
  simultaneous tail vacating, a short repeated-action horizon, reachable area,
  and growth. This recovery path is deliberately conservative and is not
  claimed to have the cycle's completion guarantee.

The controller does not use a private seed table or assume a particular food
sequence. It uses only dimensions and the current public observation; all
route geometry is recomputed from that observation.

## System-1 / System-2 interface

This experiment intentionally has **no active Jev or Astra call** on the
reference board. The complete body and the rectangular, obstacle-free rules
make cycle membership, collision legality, and the route objective exactly
computable. Sending a 136--158 ms Jev request at a junction would create the
same deadline pressure that 0005 exposed, while it cannot improve a choice
among candidates already proved safe. Likewise, the measured Astra latency
(about 42 s median) cannot supervise a 100 ms route junction.

For clarity, the Jev schema for this candidate is the empty schema: no state is
sent, there are no questions, no confidence threshold, and no answer-to-action
bridge. The deterministic bridge is instead:

1. validate the live body against the retained cycle;
2. use a validated route suffix if present;
3. otherwise use a cycle-order-checked food shortcut / cycle edge; and
4. otherwise use the one-step repeated-horizon fallback.

Thus freshness is stronger than a model-age rule: an action is computed from
the observation immediately preceding its post, and every route action is
rechecked against that same observation. Garbage model output cannot affect
pre-flight because no model output is parsed.

If a future board profile repeatedly fails cycle entry or contains an
obstacle/variant rule, the appropriate Jev state would be a small JSON object
with `tick`, `head`, `tail`, `direction`, `length`, `food`, and a list of
**code-validated** candidates. Each candidate would carry only named facts
such as `one_step_safe`, `repeat_safe`, `food_now`, `tail_access`, and an area
bucket. Two reordered `Choice` questions (`pick_a`, `pick_b`) would choose
among those candidates; the bridge would require agreement, a selected-option
probability at least 0.6 for a lethal decision, age at most one tick, and
current revalidation. A disagreement, timeout, low probability, or stale
answer would leave the deterministic fallback in control. Jev would never be
asked to find coordinates or prove a route.

Astra should revise that interface only after traces show a stable, repeated
failure class that code facts do not explain—for example, an initial body that
cannot enter any generated cycle, an obstacle/wrap rule contrary to the
contract, or an invariant-compatible route policy that stalls at high
occupancy.
Astra's output should then be a high-level policy choice (such as
`cycle-conservative`, `safe-food`, or `tail-unwind`) and thresholds over the
code-produced facts, not per-cell actions. Evidence of only late worker
results or an ordinary route-discard event is not a reason to call Astra:
the intended response is the immediate cycle fallback.
