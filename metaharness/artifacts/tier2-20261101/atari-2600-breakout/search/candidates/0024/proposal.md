# Candidate 0024 — trajectory-fenced capture invariant

## Parent and diagnosis

The direct performance reference is `candidates/0022/`, an impact-latched refinement of
`0019` that scored 763.333 in the formal six-seed evaluation. I also used `0019` as the
structural parent: it supplied the reflected landing interval, one-flight objective
bookkeeping, measured paddle dynamics, and sparse-wall cleanup. This candidate is a
careful final-generation change to the lifetime of the low-level controller, not a
return to model-driven frame actions.

The archive's strongest evidence points to a collision-boundary problem that the impact
lock did not actually isolate. `0022`'s locks are established and then continue to act
through stale or moving geometry. In the formal seed-1 trace, the misses at frames 7221
and 7786 are preceded by locks at 7211 and 7776. The first has an interval around
`[38.9,50.9]`, paddle x=47 and velocity about +0.26 at lock creation; it then reaches
x=27 with velocity -4.69 before the missed contact. The action window is
`LEFT, LEFT, NOOP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT`, so the lock's
adaptive terminal policy still creates a large excursion. The second begins at x=117
for interval `[112,124]`, then reaches x=136 with velocity +4.54 before returning to
x=124 and losing. A stationary interval was not a sufficient certificate because the
paddle was still carrying queued/sticky motion.

The seed-2 trace shows the same failure with different signs: the miss at 6181 begins
with a lock at frame 6170 and ends at x=87 with positive velocity 1.84 against a
contract that had moved from roughly `[81,93]` to `[76,88]`; the miss at 13711 follows
an x=92 lock with velocity -2 and ends x=101 with velocity -4.22. Seed 3's losses at
11888 and 14803 similarly show a lock followed by a moving paddle and a late reversal.
These are all fresh code actions, so stale ticks and Jev latency are not the cause.

There is a second, more structural flaw in 0022: the `impact_lock` is only cleared by
an event seen by the outer loop. A side reflection or brick hit can change the
horizontal trajectory while a terminal lock is already pending. If the event detector
misses that one-frame change (or if a reward event takes precedence over side evidence),
old landing/interval assumptions remain authoritative in the next frame. The public
observation already supplies enough evidence to fence the old plan: a change of `dx`,
a score/brick change, a `dy` sign transition, or a ball disappearance is a trajectory
epoch boundary. The harness should not ask a model to repair that bookkeeping.

## Hypothesis

The remaining lives are lost because the controller combines two incompatible
lifetimes: a receding landing estimate and a terminal actuator reservation that can
survive an unrecognized collision. In the last few frames, the adaptive lock also
counter-steers from a velocity forecast while the paddle is already inside the
reported catch set. With sticky actions and non-instantaneous NOOP, that is exactly the
condition that produces the observed x=47 -> 27 and x=117 -> 136 excursions.

**Hypothesis:** a trajectory-fenced, observed-position capture invariant will preserve
more lives than an adaptive velocity-braking lock. Once a descending return has a
reachable capture interval, code should either move toward an observed interval edge
or release while inside it; it should not reverse merely because a one-frame velocity
forecast predicts that the paddle might leave. Every visible collision boundary must
invalidate the lock and require a short re-acquisition period from the new trajectory.

This is a structural change to plan lifetime and terminal control. It deliberately does
not add a more ambitious Astra plan, a larger Jev action vocabulary, or seed-specific
schedules.

## What changed in `harness.py`

### 1. Explicit trajectory epochs

The controller keeps `trajectory_epoch`, `contact_id`, a filtered velocity history, a
capture interval, and the terminal lock separately. On `serve`, `brick_hit`,
`side_bounce`, `paddle_bounce`, `ball_lost`, `life_loss`, and wall transition, it:

* increments the epoch;
* clears the old interval, terminal lock, and velocity cadence history;
* cancels any short motor lease; and
* waits a bounded one or two frames before admitting a new terminal lock.

The event detector uses only public observations: lives/wall/score changes, ball
presence, descending-to-ascending contact near the paddle, and a horizontal velocity
sign change below the brick band. The old lock records its epoch, so even if a later
path reaches it unexpectedly, an epoch mismatch cannot control an action. A side or
brick collision therefore cannot inherit the previous return's landing prediction.

### 2. Capture-set terminal policy replaces adaptive braking

The reflected projection still computes a paddle-left interval from ball geometry and
public `dx/dy`. Far from contact, a short leased pulse moves toward the landing band.
Within twelve native frames of a reachable return, a lock may be admitted only when the
paddle is within a bounded outer band of the interval. Once locked:

* if the observed paddle is outside the lower edge, post `RIGHT`;
* if it is outside the upper edge, post `LEFT`; and
* if it is inside the fixed interval, post `NOOP` and do not counter-steer from
  `raw_paddle_v` or a forecast.

If a later observed position leaves the interval, the controller corrects from that
observed side. The interval is intentionally fixed through the terminal window; it is
not receded on every quantized RAM sample. A bounce, ball loss, side reflection,
brick-score change, or expiry clears it. This tests whether the decisive failure is
queued actuator motion rather than target arithmetic.

The interval is a conservative interior overlap set, not a claim that the decoded
paddle rectangle is the exact collision mask. It remains bounded by the observed
paddle travel range. At the edge of the legal paddle travel, the code uses the clipped
interval rather than counter-steering into a wall.

### 3. Measured shaping remains subordinate

The two-wall objective is retained, but only as a code-gated one-contact preference.
Astra can propose LEFT/RIGHT/SURVIVE at strategic boundaries. Jev receives two reordered
Choices and can accept a measured one-contact side only when lives, speed, rally
history, edge status, and a wide early capture set make it eligible. Code requires two
stable observed contacts with the requested outgoing direction before using a measured
endpoint offset. Neither model can emit a frame action, alter the capture interval,
or bypass a trajectory fence. If an answer is absent, late, malformed, disagrees, or
has insufficient selected-option probability, the controller preserves the ball.

This keeps the experiment testable: an improvement should come from the terminal
capture invariant and event fencing, not from a larger model authority surface.

## Evidence that would confirm or refute the hypothesis

Confirmation would have all of the following properties in traces:

1. A visible `side_bounce` or `brick_hit` is followed by `impact_fenced` and no action
   from the old epoch; a new lock is created only after the new velocity history warms.
2. In missed-contact windows, the last several observed paddle positions remain inside
   the fixed capture interval or visibly correct toward it, without the 0022 pattern
   of repeated opposite actions and velocity swings.
3. The fraction of locks ending in an observed `paddle_bounce` rises, terminal paddle
   speed falls, and life losses per thousand native frames fall.
4. First-wall completion stays at least as strong as 0022 while more episodes finish
   the second wall; the score must improve without increasing stale ticks or model
   failures.
5. The same controller behaves similarly when Jev/Astra answers are unavailable,
   demonstrating that the critical path is actually deterministic.

Refutation would be a stationary or nearly stationary paddle inside a correctly fenced
interval still missing repeatedly, or a large increase in losses because the fixed
interval is not the cartridge's collision set. It would also be refuted if event
fencing fires on ordinary cadence noise and repeatedly disables a healthy return. In
that case the next iteration should investigate collision masks/plane and RAM event
semantics, not add more model questions. A score increase with unchanged terminal
motion but only different tunnel choices would not confirm this hypothesis; it would
be a model/shaping effect and needs an ablation.

## Jev state schema and questions

Every Jev request is a strict JSON object assembled from derived facts. It contains:

```text
{
  game, rules,
  observation: {
    frame, event, lives, score, wall, bricks_remaining,
    phase, eta_bucket, speed, landing_lane, capture_width,
    paddle_motion, impact_lock, shape_eligible,
    candidate_side, left_edge, right_edge,
    strategy, current_objective,
    recent_contact_support: {left, right},
    last_progress_age, trajectory_epoch, contact_id
  }
}
```

Code computes all arithmetic, reflection, interval widths, event comparisons, brick
counts, action legality, and shaping eligibility. Jev is not asked to compare
coordinates, count cells, infer velocities, or generate an action. The explicit rule
fields tell it that a side/brick/paddle collision fences old plans and that the terminal
capture controller is code-owned.

The request batches two independent `Choice` questions:

* `preference_a` and `preference_b`, with reversed option order, choose one of
  `PRESERVE`, `SHAPE_LEFT`, `SHAPE_RIGHT`, or `NO_CHANGE`.

`PRESERVE` keeps the code-owned catch interval. `SHAPE_LEFT`/`SHAPE_RIGHT` authorize
only the corresponding code-generated, measured one-contact preference when
`shape_eligible` and the candidate-side checks are true. `NO_CHANGE` leaves the current
objective alone. The instructions explicitly say to choose preservation for imminent,
fast, narrow, moving-away, or life-scarce returns, and that neither question can waive
code safety gates.

The bridge accepts a result only if both values agree, each selected-option probability
is at least 0.60, the epoch/contact/plan stamp is current, and the answer is at most
120 native frames old. It rechecks lives, speed, candidate side, and eligibility on
arrival. A missing probability map is uncertainty; `.confidence is None` is not
converted into certainty. A timeout, runtime error, arbitrary stand-in object,
malformed value, disagreement, or stale stamp is logged and ignored. The deterministic
fallback continues without waiting.

Requests are launched asynchronously at serve, paddle-bounce, life, and wall
boundaries, or when a new early shaping opportunity becomes eligible. At most one is
outstanding and there is a 48-frame scheduling gap plus a small call-budget reserve.
The loop posts an action before waiting for the next native tick.

## Astra role and revision trigger

Astra is strategic rather than reflexive. Its prompt includes the same compact derived
state and asks for exactly `LEFT`, `RIGHT`, or `SURVIVE`. It is called once per wall at
startup/boundary and again only after a long scoring stall with a sparse wall. Its
answer is a prior for the next Jev judgment; it cannot change the trajectory epoch,
terminal capture invariant, action mapping, or collision arithmetic. The answer is
strictly parsed and invalidated on a life or wall transition. Astra is asked to revise
when observed progress stalls despite catches, not when a single quantized `dx/dy`
sample changes.

The harness uses a 72-second timeout consistent with the measured Astra tail and does
not wait for the future. If Astra is slow or unavailable, the episode remains eligible
and the same code controller runs. There are no background loops beyond the spawned
inference calls, and all model exceptions are contained as normal unavailable advice.

## Tier-2 fallback and contract safety

The real-time loop calls `game.observe()`, derives one action, calls `ctx.act(action)`,
and waits with `ctx.wait_for_tick()`. It uses only the legal names `NOOP`, `FIRE`,
`RIGHT`, and `LEFT`. `FIRE` is repeated while `serve_required` is true; a missing ball
is never treated as an episode reset. It plays through all five lives and through wall
transitions until the game or context ends.

All model parsing is defensive. Pre-flight stand-ins may return arbitrary text,
`None`, missing answer keys, or answer objects without probabilities; none can escape
as a harness error. `BudgetExceeded` and `EpisodeOver` around model futures and the
real-time loop are normal episode termination paths. No private emulator state,
screenshots, seed-conditioned logic, source-code import, or model backend is used.

The most important fallback invariant is simple: if no valid model answer exists, the
capture-set motor still posts a legal action every native frame; if it cannot certify a
reachable terminal lock, it uses the broad reflected position controller. No slow
answer is ever required to catch a ball.

Pre-flight repair: lock diagnostics referenced a removed `raw_v` local after the queued-input terminal policy replaced velocity braking; the harness now derives that diagnostic value defensively from observed motor state.
