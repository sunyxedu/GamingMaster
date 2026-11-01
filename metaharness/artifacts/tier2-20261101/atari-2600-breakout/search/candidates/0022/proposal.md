# Candidate 0022 — impact-latched braking and model-selected return profiles

## Parent and hypothesis

The direct parent is `candidates/0019/`, the strongest evaluated harness (mean
743.5). I retain its useful code-owned reflected interception, one-flight
contact bookkeeping, measured actuator fit, endpoint evidence, and sparse-wall
ray cleanup. The new experiment is deliberately lower-level: it changes the
last few native frames before a paddle collision rather than giving Astra or
Jev frame-action authority.

**Hypothesis:** the remaining losses are caused by a moving paddle and a
receding target during the collision window, not by a lack of geometric
prediction. The parent repeatedly changes its action after the paddle is
already inside the projected overlap interval. In the seed-2 trace, for
example, the miss at frame 5510 has a target `[122.1,134.1]` and paddle x=124
at eta 6.6, but the final actions are `RIGHT, RIGHT, RIGHT, NOOP, LEFT,
NOOP, RIGHT, RIGHT, NOOP, NOOP` while the measured paddle velocity swings
-2.91, -3.79, -2.20, -0.43, +1.43, +1.67, +0.13, -1.19, -0.27, +1.10.
The frame-6834 miss is similar: x=22 is inside `[16.4,28.4]` at eta 7.8
with velocity +3.58, then the controller alternates LEFT/RIGHT and reaches
x=9. The misses at 12834, 17424, and 24581 also enter the interval but
reverse near eta 5–2. Successful parent contacts instead commonly keep
NOOP and a nearly stationary paddle through y=184–187.

The untested claim is therefore an **impact-latched actuator contract**:
when a descending return is safely inside its catch set, code should stop
receding-horizon replanning, latch the physical overlap interval, and issue a
short committed brake pulse. It then uses observed paddle position and
velocity against that fixed interval, with hysteresis, until the collision is
resolved. It must not recompute a target after the ball has crossed the
nominal contact plane. Sticky actions make a short same-action commitment
safer than alternating requests; adaptive continuation handles the case where
queued input carries the paddle after the first pulse. This is a structural
test of collision order and queued input at impact, not a parameter-only
change.

## What changed

* The parent’s early/ascending reflected controller remains available for
  reaching the projected interval. Its ordinary predictive motor is no
  longer allowed to reverse on every quantized RAM sample once an impact
  lock is established.
* Each descending flight has one `impact_lock`. It is created only near the
  return (a short ETA) when the current paddle is inside an expanded physical
  overlap set, or when a bounded approach has made the set reachable. The lock
  stores the interval, the predicted landing, and the observed paddle-motion
  sign; it is not changed by the 1-pixel dx/dy cadence.
* A lock has a short `BRAKE` commitment and an `ADAPTIVE` phase. The first
  phase repeats one direction long enough to overcome queued/sticky input; the
  second uses the observed paddle position and velocity against the latched
  physical interval, with hysteresis, rather than releasing blindly or
  recomputing the ball landing. The lock remains authoritative through a
  small post-plane grace window. If the paddle is not yet safely in the
  interval the approach controller remains active; a miss, serve, bounce, life
  loss, wall transition, speed change, or a large observed trajectory change
  invalidates the lock.
* Endpoint/tunnel shaping is still an early-flight preference and the
  measured cleanup objective still supplies candidates. Neither can override
  the impact lock or reduce physical overlap. This isolates whether better
  impact execution converts the parent’s already-measured contact objectives
  into wall progress.
* Astra now advises a coarse return profile (`DAMP`, `TRACK`, or `SHAPE`) at
  wall/stall boundaries. It is a prior for the *next* flight, never a frame
  action. Jev validates a profile at paddle-bounce/life/wall boundaries with
  two reordered Choices. The default is the safe impact-latched profile;
  malformed, late, low-probability, conflicting, or unavailable answers leave
  the deterministic controller unchanged.

Evidence would confirm the hypothesis if impact locks show fewer late
LEFT/RIGHT reversals, lower terminal paddle speed, a higher fraction of
contacts that remain within the latched interval, and improved completion of
the second wall without increasing life losses. It would be refuted if locks
are established while outside the true collision mask, if the committed
brake systematically drives the paddle through the interval, or if misses
remain despite a stationary terminal paddle; then the geometry/contact-plane
model, rather than queued-action chatter, is the next suspect.

A short development ablation supports testing this exact change: the first
fixed-duration version reached only 63 points and exhausted all five lives by
about 5,278 frames on seed 2, while the adaptive fixed-interval revision
reached 247 points with one miss at 7,199 frames and no game-over. A separate
five-minute seed-2 try reached 452, cleared the first wall, entered wall 1,
and had four miss windows. These are smoke tests with Astra off, not selection
results; they show that the adaptive phase changes the observed failure mode
rather than merely increasing model calls.

## Jev interface and bridge

Every asynchronous Jev request carries a strict JSON object with only derived
facts:

```text
frame, event, lives, wall_index, bricks_remaining, score,
phase, urgency, speed, eta_bucket, landing_lane, catch_width,
paddle_motion, terminal_risk, impact_lock_state,
shape_ready, candidate_side, left_edge, right_edge,
current_profile, astra_profile, last_contact_outcome,
rally_bounces, return_epoch
```

Code computes all distances, reflection, timing, brick counts, legal actions,
and the candidate side. `terminal_risk` is a named bucket (`settled`,
`moving_toward_set`, `moving_away_from_set`, or `unknown`), so Jev is not
asked to do arithmetic. It is told that `DAMP` means the earlier/adaptive
brake contract, `TRACK` means ordinary early interception with the same
terminal safety lock, and `SHAPE` means an eligible early endpoint
preference; all choices remain subject to code gates.

One request batches two independent `Choice` questions over
`DAMP`, `TRACK`, `SHAPE`, and `NO_CHANGE`, with the options in reversed order
for the second question. Jev is asked for the profile of the **next** return,
not a raw action. Both answers must agree, their selected-option probabilities
must each be at least 0.60, and the request must still belong to the current
life/wall epoch and be no more than 120 native frames old. A directional or
shaping profile is accepted only with at least four lives, normal speed,
wide/known geometry, and a non-open candidate edge; otherwise it is coerced
to `DAMP`. `DAMP` never disables the code-owned catch fallback.

Requests are launched with `ctx.spawn` at serve/paddle-bounce/life/wall
boundaries, with one outstanding request and a 42-frame cooldown. The loop
never waits for Jev: while it is late, the current deterministic profile and
impact lock post actions every native frame. `TimeoutError`, `RuntimeError`,
`BudgetExceeded`, `EpisodeOver`, malformed stand-in answers, missing
probabilities, and arbitrary values are logged and ignored.

## Astra role and revision trigger

Astra receives the same derived evidence plus recent measured contact
outcomes, and must return one token: `DAMP`, `TRACK`, or `SHAPE`. It is called
asynchronously at the beginning of a wall and after a long no-reward stall,
not during an imminent return. A completed token is stored as a prior and
validated by the next Jev boundary; a life loss or wall transition invalidates
old advice. The revision trigger is observed terminal instability or scoring
stall, rather than a periodic frame count. Astra cannot change the impact
lock, action names, or collision arithmetic.

## Fallback and real-time contract

The loop owns the 60-Hz schedule, posts a legal action before any model result,
and waits only for the runner’s next tick. FIRE is posted while
`serve_required` is true; a missing ball is never treated as an episode reset.
If a model is late or garbage, the reflected early controller and the
impact-latched brake/adaptive state continue. If `ctx.act` or the game closes,
only the normal `BudgetExceeded`/`EpisodeOver` termination is caught; no
background loop is created, and spawned model calls are allowed to be
cancelled by the evaluator.
