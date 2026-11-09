# Candidate 0005: predictive controller with asynchronous supervision

## Parents and diagnosis

Parents: `candidates/0000-tier2_jev_only/` and `candidates/0002-tier2_idle/`.

The Jev parent is the only parent that earns positive points, but it asks for a raw
six-way action choice synchronously on every decision.  Its evaluation is explicit:
mean return `-20.333`, one own point in each of seeds 0 and 1, and `89%` stale ticks
(`1315` Jev calls, `1315` informed actions, but thousands of repeated actions).  The
worst episode, seed 2, ended `-21` with `3381/3784` stale ticks.  In its seed-0 trace,
for example, a request at frame 239 sees the ball approaching right at
`ball_x=140.5, ball_y=129.5` while the right paddle is at `191.5`; the answer is
`RIGHT`, but the answer arrives after several emulator frames and the paddle is still
at `191.5` at a later request.  The model is being asked to react to a trajectory while
its latency is roughly nine frames.  It also has no supplied intercept target or wall
reflection arithmetic.

The idle parent is robust (`ok`, no failed calls), but scores `-21` in every episode:
it never posts an action, so it cannot miss because of a bad model call, but it also
never moves the paddle.  I retain that robustness as the fallback: the game loop is
fully executable without either model and model failures never stop frame control.

## Hypothesis

The dominant failure is structural, not a Jev knowledge deficit: raw action selection
cannot meet a 60 Hz collision deadline.  A verified geometric controller should win
more rallies if it (1) projects the ball to the right paddle, including top/bottom
reflections, (2) moves early toward that projected y rather than chasing the visible
ball, and (3) posts a fresh short-horizon action every emulator frame.  Jev can still
help classify uncertain/occluded frames and gate an optional button feature, but it
must not be on the critical path.  Astra can choose safe versus attacking supervision
settings and diagnose systematic drift, but its slow answer must also be off the
critical path.

This tests the study's specific intercept-vs-chasing claim on the public visible-pixel
coordinates.  It uses the verified court/paddle bounds and action orientation, not
seed-specific behavior.  The controller resets old velocity at `serve_delay` and
`finished`, so a stale ball visible on a point frame cannot be mistaken for a rally.
It uses central coverage while the ball travels away and uses no FIRE to start a
serve.

## What changed

* `harness.py` runs a local controller in the real-time loop.  It calls `ctx.act` and
  then `ctx.wait_for_tick` on every frame, so a pending model request cannot repeat a
  stale action.
* It keeps the last visible ball and paddle samples, uses adaptor velocity when
  present, recovers velocity from consecutive visible positions when safe, and
  projects to x=140.5.  The y projection reflects continuously between 37.5 and
  191.5.  A target deadband prevents oscillation and the visible paddle height is
  retained for small-paddle cases.
* During `serve_delay`, occlusion without a reliable trajectory, or ball travel away
  from the right side, it falls back to a central target.  `RIGHT` is selected for a
  target above the paddle, `LEFT` for a target below, and `NOOP` when aligned.
* One Astra request is launched asynchronously at reset.  Its compact JSON plan may
  set only clamped supervisory fields (`safe/attack`, optional feature, target bias
  -6..6 px, deadband 2..8 px).  After four observed point events, one optional
  asynchronous diagnostic replan may run.  Malformed prose or a late/cancelled call
  leaves the conservative defaults; Astra never supplies a one-frame action.
* Jev requests are also asynchronous and batched.  They are launched near a possible
  contact, after a point, or when perception is uncertain, about every 18 frames at
  most.  A result is tagged by the current state in the request.  Two direction
  Choices with reversed option order must agree before a hint is accepted, and the
  hint expires after five frames.  It can override code only when the ball trajectory
  is currently uncertain; a reliable projection always wins.  Thus a late answer is
  useful as recognition but cannot damage a clear intercept.
* FIRE is disabled by default.  It is possible only when Astra explicitly enables a
  feature, Jev selects `USE` with probability at least `.68`, the variation is a
  Robot Pong variation, the ball is within three projected frames of contact, and the
  movement action is already safe.  It is used at most once per point.  This prevents
  the parent's tendency to let a generic action model press FIRE away from contact.

## Jev state and questions

Every Jev request receives a strict compact JSON object, rebuilt from the current
observation and code-derived facts:

* `scenario`: mode, difficulty, variation, and right paddle size;
* `score`: own/opponent points and last reward;
* `observation`: frame, phase, serve timer, and serve status;
* `ball`: visible/estimated x and y, signed vx/vy, approach (`RIGHT`, `AWAY`, or
  `UNKNOWN`), reliability, and visibility gap;
* `right_paddle`: y and measured y velocity;
* `control`: code-computed reflected `predicted_contact_y`, `target_relation`
  (`ABOVE`, `ALIGNED`, `BELOW`, `UNKNOWN`), `contact_window`, contact-time bucket,
  `code_action`, and `uncertain`;
* `plan`: Astra's safe, clamped supervisory settings; and
* the last four score events as short evidence.

The state contains actual values and meanings, not variable names or hidden context.
Jev is asked independent typed Choices in one request:

1. `approach`: `RIGHT`, `AWAY`, `UNKNOWN`, recognizing the supplied horizontal fact;
2. `alignment`: `ABOVE`, `ALIGNED`, `BELOW`, `UNKNOWN`, recognizing the supplied
   target relation;
3. `contact`: `NOW` (at most three frames), `SOON` (4--12), `LATER`, or `UNKNOWN`;
4. `direction_a`: `UP`, `HOLD`, `DOWN`; and
5. `direction_b`: the same choices in reversed order.  The two must agree and have
   minimum per-answer probability `.58` before a hint is stored;
6. `feature`: `USE` or `NO`; `USE` is accepted only with probability `.68` and all
   code/Astra contact preconditions.

The direction answer maps to the calibrated action names: `UP -> RIGHT`, `DOWN ->
LEFT`, `HOLD -> NOOP`.  It is deliberately not asked to add numbers, integrate
velocity, reflect the ball, or invent an action.  The code handles all of those.
Choice confidence, when present, is also checked; an absent/low-confidence value
falls back to the code controller.  Stand-in answers, invalid values, timeout, and
runtime errors are all ignored safely.

## Astra interface and adaptation trigger

The initial Astra prompt includes the game description and the compact initial
observation.  It is told that the frame controller owns exact timing, arithmetic,
wall reflection, and score bookkeeping, and is asked for a short JSON supervisory
plan.  A second prompt is launched only after four score events and includes the
recent point evidence; this is the point at which a systematic intercept drift is
observable and a high-level diagnosis is worth another slow call.  Neither call is
awaited by the game loop.  Plan fields are range-checked, and no Astra text is
parsed as an action.

## Evidence that would confirm or refute the hypothesis

Confirmation is a lower opponent-point count and/or more own points than `-20.333`
on the training scenarios, with near-zero stale ticks.  Traces should show the paddle
moving before rightward contacts, `target_relation` becoming `ALIGNED` near the
projected contact, and score events occurring after the controller continues to post
frame actions during model calls.  The safe fallback should still finish cleanly if
both model backends time out.

Refutation would be repeated misses despite a reliable `ALIGNED` projection, or a
systematic offset that persists across variations.  The score-event evidence supplied
to the optional Astra replan is specifically intended to reveal that case (wrong
contact x, paddle response lag, or wall bounds), while the next iteration could adjust
those verified controller assumptions.  A model answer arriving after its five-frame
expiry, a malformed answer, or an unavailable backend is not evidence against the
policy: it is intentionally ignored rather than allowed to create stale or invalid
control.
