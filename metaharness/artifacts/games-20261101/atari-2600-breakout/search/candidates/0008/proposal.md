# Candidate 0008 — measured-velocity servo and empirical paddle contact

## Parent and evidence

The starting point is `candidates/0005/`, the asynchronous interception crossover. It
raised the mean from 3.33 to 288 and eliminated stale control, but it still lost all five
balls in every training episode. Its worst episode was seed 4 (143 points); its best was
seed 0 (382). The traces show that the problem is no longer model latency: 0005 posted
9,918 fresh actions in seed 0 and only one tick was stale. Nevertheless, near misses often
occurred after the controller had declared the paddle to be in the landing lane. For
example, seed 4's final pre-loss requests report `descending right`, landing x 130.5 then
136.0, `paddle_relation: left_of_landing`, and fallback `LEFT` at frames 6787 and 6805;
the game ended at frame 6821. Other misses show the complementary pattern: `paddle_relation:
overlap` followed by an immediate ascent/life loss. This is consistent with a controller
that follows one point estimate while the paddle is still moving and with a sticky action
or inertial carry-through after a NOOP.

The parent also resets its projected target to the center whenever the ball is ascending,
and its strategic bias is disabled in the impact window. Thus it is a strong survival
interceptor but never deliberately learns which paddle section produces a desired outgoing
angle. Its Astra call was usually too late to matter: the one completed call in the archive
took 58.4 seconds and returned `SURVIVE`, while five of six calls timed out or were
cancelled. This candidate therefore changes the fast controller first and makes Astra
advisory rather than making the game wait for it.

## Hypothesis

The remaining score ceiling is caused primarily by **unmodelled actuator state and
single-point landing control**, not by insufficient Jev deliberation. A controller that
(1) estimates paddle displacement from consecutive RAM observations, (2) predicts an
interval of possible ball landing positions, (3) uses a short-horizon bang-bang servo with
an observed-velocity brake, and (4) learns the sign of a small paddle-section offset from
actual post-contact `dx` should lose fewer balls. Once the safety servo is stable, cautious
section shaping should convert some of the otherwise safe rallies into a side tunnel and
raise raw brick reward.

This is a structural test: 0005's point target/NOOP policy is replaced, not tuned by
changing its Jev cadence. The code never assumes the manual's qualitative five-section
angle table is numerically correct; it tests the effect of an offset and reverses its
sign if the observed outgoing velocity disagrees.

Evidence that would confirm the hypothesis is a higher score and longer elapsed frame
count, especially an improvement in the low seed-4/seed-2 outcomes, with the same near-zero
stale fraction. A useful trace signature is fewer life losses immediately after an
`overlap` classification and a measured `contact_sign_reversed` event followed by a
consistent outgoing direction. It is refuted if deaths remain at similar frames despite
measured braking, or if side shaping reduces mean score while safe mode alone does not
improve survival; in that case the next revision should remove contact shaping or build a
more exact collision model.

## Harness changes

* Every tick still posts a legal action before waiting, so neither Astra nor Jev can block
the 60-Hz loop. FIRE is posted repeatedly only while `serve_required` is true; a missing
ball is never treated as an episode reset.
* `_update_tracker` compares paddle rectangles frame-to-frame, estimates current and
smoothed paddle velocity, estimates movement step size, and records whether a NOOP really
stops motion in this ROM. It also repairs transient invalid ball velocities from position
differences and resets that history across a serve.
* `_flight` projects the descending ball with side-wall reflection and an uncertainty
envelope over x and horizontal velocity. During ascent it uses only a low-confidence,
partially blended no-brick return estimate; subsequent descending RAM observations replace
that estimate. The state includes a safety landing and interval rather than asking Jev to
calculate geometry.
* `_movement_action` predicts where the paddle will be over a short horizon. It steers
against the predicted error, uses the safety target in the impact window, and only
counter-steers when measured NOOP drift makes braking useful. A side offset is fenced off
when the ball is near, fast, uncertain, or the episode has one life left.
* `LEFT`/`RIGHT` are tactical **contact modes**, not direct actions. When selected, the
code shifts the desired paddle center by four pixels at a safe descending approach. The
left/right sign is calibrated from the next observed descent-to-ascent transition and
flipped if the outgoing ball direction is wrong. The mode is automatically abandoned when
its outer two brick columns are open, after a life/wall transition, or on a fast-speed
change.

## System-1 interface

### State schema

Each Jev request carries one strict JSON-compatible object:

```text
{
  game,
  observation: {
    frame, event, score, lives, lives_are_low, serve_required,
    flight_phase, landing_urgency, landing_lane, paddle_relation,
    paddle_motion, speed_regime, safety_margin,
    left_edge, right_edge, wall_index, walls_cleared, bricks_remaining
  },
  current_code_mode,
  astra_proposal,
  rules: {legal_actions, priority, serve}
}
```

The numerical landing and paddle arithmetic is done in code. Jev sees semantic buckets,
edge status (`blocked`, `partial`, `open`), and the current proposal, not the 6-by-18 grid
or raw values it would have to count.

### Questions and bridge

One request batches two independent `Choice` questions, `mode_a` and `mode_b`. Both choose
among:

* `SAFE`: follow the code-computed safety landing and do not shape contact;
* `LEFT`: if the code says the approach is early and safe, try the empirically calibrated
  left-going paddle section;
* `RIGHT`: the symmetric right-going attempt.

The option orders are rotated and reversed between the two questions to reduce Jev's
first-option bias. Instructions explicitly say that `near`/`imminent`, `fast`, low lives,
or an open requested edge means `SAFE`; the descriptions also say not to calculate
numbers. Code accepts a result only when both values agree, the weaker selected-option
probability is at least `0.50`, and the answer is from the allowed set. A value is dropped
if it is stale, belongs to an old plan version, or the requested edge has become open.
The accepted mode is cached for 60--roughly 240 native frames depending on events, and is
invalidated by life loss, ball loss, wall transition, fast-speed detection, or a changed
edge condition. Jev is requested only after a live velocity is available and at strategic
boundaries, not every frame. A timeout, malformed stand-in answer, disagreement, or
unavailable model leaves the code servo in `SAFE` mode.

### Astra handoff

At most one Astra request is launched after the first reliable live-ball observation. It
receives a compact public summary (lives, score, flight buckets, speed, edge buckets, wall,
and remaining bricks) and is asked for exactly `SAFE`, `LEFT`, or `RIGHT`. The result is
parsed defensively and stored as `astra_proposal`; it never directly posts an action or
overrides the servo. The next Jev state includes that proposal and asks whether current
facts support it. If Astra is still running after a rally or the episode ends, its result
is simply ignored/cancelled. This specifically addresses the archive's 58-second Astra
latency rather than pretending a slow strategic answer is a reflex.

## Fallback and timing policy

The deterministic fallback has priority over all model advice:

1. `FIRE` while `serve_required` is true;
2. in live play, predict the reflected landing interval and servo the paddle toward its
   safety target, with measured velocity/braking;
3. in the near/imminent window, ignore contact shaping and preserve overlap;
4. use `NOOP` only when the predicted paddle remains in the target corridor, or a measured
   no-op stop is safe; counter-steer only after observed drift supports it.

The loop posts one action, then calls `ctx.wait_for_tick()`. Background futures are never
joined by a plain Python loop, and every exception from a late or garbage model answer is
contained. Thus a failed/late model changes only a future advisory mode; it cannot create a
stale run or an episode error.
