# Candidate 0007 — adaptive braking and measured contact shaping

## Parents and diagnosis

This is the requested crossover of `0005` (fresh native-frame interception with
advisory Jev/Astra) and `0002-tier2_idle` (a harmless loop that never blocks or
errors).  The useful mechanism from 0005 is that code, rather than a synchronous
model request, posts an action on essentially every tick: its stale fraction was
0.000125 and its mean score was 288.  The useful lesson from idle is a strict
fallback: if advice is unavailable, continue the episode with legal no-op/control
logic rather than waiting for a model.

The score trace also shows the remaining failure.  0005's controller often
reached long rallies, but its fallback used `NOOP` whenever the paddle was close
to the projected target.  The game explicitly warns that NOOP does not promise an
instantaneous brake, and the trace's paddle positions continue to drift between
model-informed corrections.  The fast phase is especially revealing: in seed 4,
for example, the run reaches score 125 at frame 6569 and then loses the last life
before frame 6821; in seed 0 it reaches 382 but still dies with 14 bricks left.
The failures are not model-call errors (`0005` had zero failed S1 calls), and its
Astra calls mostly timed out without informing an action (5 failed S2 calls out of
6).  Thus more prose or more synchronous model authority is unlikely to solve the
bottleneck.

## Hypothesis

The geometric landing projection is useful, but position-only steering is brittle
because the paddle has motion state and because a descending ball can first hit a
remaining brick and reverse before the predicted paddle crossing.  A controller
that (1) estimates paddle velocity from every successive RAM observation and
counter-steers predicted overshoot, (2) marks a return uncertain when the exact
bitmap shows a brick in the descending path, and (3) learns the actual paddle
contact offset -> outgoing horizontal direction in this ROM should preserve more
lives and therefore score more.  Once an outer column is genuinely open, the same
measured contact table can shape a side tunnel, but only while lives and speed
make that optional maneuver safe.

This is a structural change from 0005: models no longer choose a direction bias
that is applied to a position-only controller.  The code owns a stateful
interceptor and uses model answers only to select a semantic tactical mode.
`0002`'s robustness is retained because every malformed, late, missing, or
unavailable answer leaves the same deterministic controller running.

## Harness changes

* `harness.py` runs one native-frame loop and never waits for Jev or Astra before
  posting `ctx.act`.  FIRE is retried whenever `serve_required` is true; a null
  ball without serve permission is not treated as a reset.
* The controller reflects the ball's horizontal path to the paddle height, but
  suppresses a false direct landing while a RAM bitmap brick intersects the
  descending path.  It keeps a broad center anchor during that uncertainty.
* It tracks observed paddle displacement and uses a short forecast.  If the
  forecast crosses the target, it emits a one-frame opposite-direction brake;
  it emits NOOP only when the forecast remains inside the deadband.  This is
  designed for both sticky actions and the documented non-instantaneous NOOP.
* Paddle-bounce observations are used to collect a short, per-episode empirical
  table of contact offsets for outgoing LEFT/RIGHT.  A side plan uses the median
  measured offset (a small conservative bootstrap is used until one is observed)
  rather than assuming a generic bounce formula.
* A tunnel is eligible only after the exact bitmap has a fully clear outer column,
  and is disabled during a post-miss cooldown, at one life, or in the fast regime's
  immediate catch window.  The default side is computed from observed outer-wall
  occupancy, not from a seed or a fixed opening.
* One asynchronous Astra call is made after live play begins.  It returns only an
  optional LEFT/RIGHT/SURVIVE preference, which code verifies against the newest
  bitmap.  Its 24-second timeout and background execution mean it cannot stall
  the episode; code can complete the whole episode without it.

## Jev state and answer bridge

Every request sends a strict JSON object with only derived, relevant facts:

```text
{
  game: mode/difficulty and native cadence,
  observation: {
    frame, lives, score, event, serve_required,
    flight_phase: serve|calibrating|ascending|descending,
    return_certainty: direct|uncertain,
    landing_lane: left|center|right|unknown,
    landing_urgency: early|near|imminent|uncertain,
    eta, paddle_relation, paddle_motion, speed_regime,
    fallback_action, tunnel_side, left_open, right_open,
    left/right_clear_columns, bricks_remaining, wall_index,
    contact_samples_left/right, plan_version, paddle_x
  },
  rules: legal actions, FIRE rule, catchability-first priority, and
         "mode is advisory; code recomputes the newest action"
}
```

A single Jev request batches two independent `Choice` questions, `mode_a` and
`mode_b`, with the same five code-defined options in rotated and reversed order:

* `TRACK`: follow the computed direct target;
* `BRAKE`: prefer a counter-steer when named paddle motion predicts overshoot;
* `CENTER`: use the conservative anchor while a return is uncertain/far;
* `SHAPE_LEFT` / `SHAPE_RIGHT`: use a measured side-contact target only for the
  matching active tunnel plan.

No question asks Jev to count bricks, calculate a reflection, or invent an action.
The bridge accepts an answer only when both choices agree, the selected key is
legal, its weaker returned probability is at least `0.50`, and its request's
`plan_version` still matches.  An accepted answer expires after 42 frames and is
invalidated by a life loss, wall transition, plan change, or Astra preference
change.  Requests are launched at least 24 frames apart, never during an
imminent return, with `timeout_s=1.5`; a late result is simply ignored.

Safety gates are code-owned: Jev cannot shape an uncertain or near/imminent
return, cannot choose FIRE, and cannot override the bitmap/life/speed tunnel
conditions.  Even `BRAKE` is applied only through the velocity-aware action
selector.

## Astra revision trigger and fallback

Astra is not a reflex controller and is not asked for a frame trace.  It sees the
current derived state and side occupancy after play has begun and may revise the
optional side preference once.  A future interface revision would be warranted
if traces show that direct returns are safe but clear outer columns are rarely
exploited, or if measured contact samples repeatedly disagree with the assumed
side target.  Conversely, if misses cluster while `return_certainty=uncertain`,
that path classifier should be tightened before giving either model more control.

The complete fallback requires no model: FIRE for a real serve request; NOOP for a
terminal/null-ball state; otherwise reflect the ball, hold the broad anchor for a
possible brick reversal, steer toward the legal paddle target, and counter-steer
when observed paddle velocity predicts overshoot.  Thus arbitrary stand-in model
text, malformed typed answers, timeouts, and all Astra unavailability are normal
control conditions rather than episode errors.

## Evidence that would confirm or refute the hypothesis

Confirmation is a lower stale fraction than 0005 (already expected from the same
nonblocking loop), fewer life losses at comparable frame counts, and a higher mean
than 288 with no candidate errors.  Useful trace evidence would be brake pulses
following measured drift and more late-game bricks, plus `contact_calibrated`
events that make side shaping directional.  The hypothesis is refuted if the
velocity-aware controller loses earlier than 0005, if `uncertain` returns cause
long center-only stretches with lower score, or if survival rises but scores stay
near 0005's unfinished first-wall total; that would indicate that the ROM's
collision timing needs a more exact state-transition search rather than this
adaptive interception layer.
