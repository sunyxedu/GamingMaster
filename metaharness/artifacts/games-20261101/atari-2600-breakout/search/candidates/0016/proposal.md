# Candidate 0016 — endpoint contracts for safe tunnel shaping

## Parent and diagnosis

The direct parent is `candidates/0008/`.  I used `0014` as the strongest control
reference because it is the first archive candidate to make substantial wall
progress.

The parent established two things that must not be given back: a deterministic
60-Hz controller is essential, and ball/paddle geometry belongs in code.  Its
seed-1 trace nevertheless ended at score 98 with 58 bricks remaining; the
archive records losses at frames 1062, 1732, 2686 and 5338 and fresh
LEFT/RIGHT alternation around the early losses.  More importantly, the selected
0008 traces accepted only `SAFE` Jev modes and emitted no
`contact_calibrated` event.  The supposed contact experiment therefore did not
actually become a stable actuator policy.

`0014` fixed much of the survival problem with an interval motor: its mean was
411.5, and seed 0 cleared the first 432-point wall at frame 12362 before
starting the second wall.  But it still finished that episode at 436 (only
four second-wall bricks), while seeds 1 and 3 finished at 388 with eight
bricks left.  Its six Astra calls all timed out, and its Jev contract was a
home-anchor preference that did not select a paddle contact section.  The
trace's final returns are repeatedly represented by ordinary overlap intervals
(for example, seed 1 has a low/high corridor around 8--10.5 immediately before
its final loss); the controller uses the middle of that set and has no verified
way to turn a safe return into a side opening.

Thus the useful discovery is not that models should steer faster.  It is that
`0014` has a good *survival primitive* but no deliberate scoring primitive.
Repeatedly adding Jev labels to the same point/center motor is the wrong
experiment.

## Hypothesis

A paddle return is a set of legal paddle positions, not a single target.  When
that overlap set is wide, its interior endpoints should produce different
horizontal bounce directions in the actual cartridge.  We can therefore
attempt a side tunnel without sacrificing catchability by steering to an
interior endpoint of the already computed overlap corridor.  The controller
should shape only after a couple of safe bounces, only with four or five lives,
and only while the requested outer edge is occupied.  If the observed
post-paddle `dx` disagrees with the requested side, the code should flip the
endpoint mapping and test it once more; it must never ask a language model to
invent a numeric bounce table.

The test is structural: the low-level motor is an event-locked corridor motor,
while the model division changes from “choose a home anchor” to “validate a
contact objective.”  A model can select `LEFT`, `RIGHT`, or `SURVIVE`, but it
cannot alter the catch set or emit a raw frame action.  If shaping is unsafe,
the exact same motor falls back to the center of the corridor, preserving the
successful 0014 behavior.

Confirmation would be a mean above 411.5 with comparable or longer lives and
at least one additional first-wall transition or meaningful second-wall
progress.  Trace evidence should include `edge_contact_trial` followed by
`contact_shape_confirmed`, decreasing outer-edge occupancy, and no increase in
losses while the paddle is inside the logged corridor.  The hypothesis is
refuted if losses occur while the paddle remains inside the corridor, if the
endpoint mapping flips repeatedly, or if safe center mode matches 0014 but
endpoint mode lowers the mean.  Those outcomes would point to a collision-time
or ROM-physics error and justify removing shaping rather than increasing model
authority.

## Harness structure

### Code-owned facts and motor

* The loop posts one legal action before waiting for a tick.  FIRE is repeated
  only while `serve_required` is true; a null ball is a serve/lost-ball state,
  not an episode reset.
* `_velocity_between` repairs a transient invalid decoder velocity with the
  observed RAM displacement.  Code computes reflected interception using the
  public paddle/ball rectangles and side-wall bounds.  It propagates a small
  uncertainty set and intersects the resulting paddle-left overlap intervals;
  Jev never performs arithmetic.
* A catch corridor is latched for a return.  It is rebuilt after serve, paddle,
  brick/side impact, life/wall changes, speed change, or a measured trajectory
  change larger than the guard.  Small frame-to-frame estimates cannot cause a
  new direction every tick.
* The motor uses short directional leases, measured paddle velocity, a neutral
  release, and at most one counter-pulse in a cooldown.  At the collision
  window it values physical overlap over the contact objective.  This is the
  fallback that runs when both models are late, malformed, or unavailable.
* A `LEFT` objective targets the high end of the safe paddle-left interval by
  default (the ball meets the left section); `RIGHT` targets the low end.  In
  the early part of a wide return the motor actually drives toward that small
  interior endpoint band; near impact it stops shaping and uses the full
  physical overlap set.  The post-bounce sign is observed.  A disagreement
  flips the endpoint mapping for that side and is logged; repeated uncertainty
  disables the trial through the lives/corridor gates.  No generic Breakout
  angle constants are assumed.
* Code reads the exact bitmap only to bucket each outer pair of columns as
  `blocked`, `partial`, or `open`.  An open requested side ends shaping and
  switches to preservation/another side.  Shape attempts require normal speed,
  at least four lives, two prior safe paddle bounces, a corridor at least ten
  pixels wide, an intact requested edge, and a cooldown-free return.  Low lives,
  fast/unknown speed, narrow corridors, imminent returns, and ambiguous sides
  always use the center objective.

### Astra's role and replanning

After the first live ball is visible, one asynchronous Astra request receives
only a compact state: lives, score/wall progress, flight/urgency buckets,
measured speed, outer-edge buckets, and the rally-bounce count.  It is asked for
one token, `LEFT`, `RIGHT`, or `SURVIVE`, as a strategic contact subgoal, never
an action trace.  The answer is parsed defensively and stored as an untrusted
`astra_plan`; it cannot directly change the motor.  The timeout is 72 seconds
because the archive measured useful short Astra answers around a minute, while
waiting synchronously would lose the game.  At a wall transition the old plan
is invalidated and a second asynchronous plan may be started if budget remains;
a life loss invalidates the plan for the current rally but does not waste a
slow call while the code can safely continue.

### System-1 state and questions

Jev is called only at `start`, `serve`, a detected paddle bounce, life loss,
wall transition, or speed change, with a 42-frame request cooldown.  Each
request is one strict JSON object (no grid or raw coordinate arithmetic):

```text
{
  game,
  observation: {
    frame, event, score, lives, serve_required,
    phase, urgency, speed, landing_lane, corridor,
    paddle_motion, eta_bucket, shape_ready,
    candidate_side, current_intent, astra_plan,
    left_edge, right_edge, bricks_remaining, wall_index,
    rally_bounces, return_epoch
  },
  rules: {
    legal_frame_actions,
    FIRE,
    priority,
    bridge
  }
}
```

`candidate_side` is generated by code from the current measured direction and
bitmap; `shape_ready` is a code boolean that already includes the life,
speed, width, and cooldown gates.  The request batches two independent
`Choice` questions, `objective_a` and `objective_b`, with the same options in
rotated/reversed order:

* `SURVIVE`: preserve this return with the corridor midpoint; it is a
  high-level caution for the current boundary, not a permanent veto of later
  code-gated endpoint trials;
* `LEFT`: use the safe corridor's left-going endpoint for the occupied left
  edge, only when `shape_ready` supports it;
* `RIGHT`: the symmetric right-edge objective;
* `NO_CHANGE`: keep the current objective and let code act.

The bridge accepts an answer only if both values agree, both are in the allowed
set, the weaker selected-option probability is at least `0.55`, the answer is
no more than 90 frames old, and its intent epoch still matches the current
life/wall/strategy epoch.  A selected side is rechecked against the latest
lives, speed, edge bucket, and corridor before being installed.  `NO_CHANGE`
or any disagreement retains the current intent; a timeout, malformed
stand-in, stale answer, or exhausted budget is ignored.  `SURVIVE` suppresses
one eligible endpoint choice, after which the deterministic gates may resume
an opportunistic trial; this keeps a conservative Jev prior from disabling the
experiment for the entire episode.  Jev never chooses `FIRE` and never
overrides an imminent corridor catch.

The deterministic fallback when no current contract is accepted is:

1. FIRE while serving;
2. center/home corridor tracking while the trajectory is uncertain;
3. for a wide early descending return with plentiful lives, choose the
   direction of the current ball (or the non-open side) as an opportunistic
   endpoint trial;
4. otherwise preserve the corridor midpoint, with measured braking only when
   paddle motion would leave the physical overlap set.

This fallback makes pre-flight arbitrary answers safe and ensures a late model
cannot create stale ticks or an illegal action.

## What would trigger a future Astra revision

If the endpoint trials confirm catches and clear an outer edge but the ball
still dies with a nearly full inner wall, the next Astra prompt should plan a
verified sequence of alternating side objectives from the observed bitmap.  If
endpoint trials do not survive, Astra should be removed from contact selection
and the next experiment should verify the collision timing/endpoint mapping.
If the first wall clears more often but the second-wall transition resets the
controller poorly, Astra should be called specifically at that transition,
with wall index and the newly reset bitmap, rather than being asked for a
generic episode-long plan.
