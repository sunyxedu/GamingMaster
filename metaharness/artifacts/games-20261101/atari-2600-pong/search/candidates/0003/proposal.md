# Candidate 0003: frame-rate intercept bridge around Jev

## Parent and hypothesis

Parent: `candidates/0000-tier2_jev_only/`.

The parent’s failure is primarily an execution/freshness failure, not evidence
that Jev cannot recognize a useful Pong action.  Its Jev-only loop synchronously
asked for a raw action, then let the runner repeat that action while Jev thought.
The evaluation reports a `0.889599` stale-tick fraction.  The measured Jev p50
was about `0.15 s`, or roughly nine 60-Hz frames.

The worst trace gives a concrete failure mode.  In
`trace_seed1.jsonl`, near the final rally, the requests see
`frame=3901, ball=(129.5,165.5), ball_v=(1,0), right_paddle_y=191.5` and
return `LEFT`; the later request at `frame=3928` still returns `LEFT`, while the
ball is already `(156.5,191.5)`.  A nearby point observation at frame 3649
records `right_paddle_y=191.5, right_paddle_height_px=4` while the ball is at
`y=191.5`; the episode finally ends at `own_points=1` and `opponent_points=21`.
Seed 2 is worse: the final score is `-21` (`0:21`).
Thus a direction selected from a valid observation can become actively harmful
when held for the model latency: the paddle is driven into a clipped wall while
the ball crosses the right contact zone.

**Hypothesis:** if deterministic code posts a predictive, wall-reflected
intercept action on every emulator frame while Jev is pending, Robot Pong will
return more balls and the score will improve; Jev should not be on the
frame-level movement critical path.

## One targeted change

I retain the parent’s Jev-only architecture and do not add Astra.  I change the
answer-to-action bridge: Jev is now an asynchronous, periodic advisory judge
for the optional contact feature, while code owns exact geometry, movement,
wall reflection, action legality, and timing.  The harness posts one legal
movement/NOOP action, waits for exactly one tick, and repeats this every frame.
There is at most one outstanding Jev request, started every 18 ticks; its
latency cannot stop the local controller.

The local controller:

* treats `serve_delay` and stale point sprites as central-coverage states;
* retains the last verified velocity for short ball occlusions;
* when the ball approaches from the left, projects its Y to the right paddle’s
  X (`140`) with reflection between court rows `34..193`;
* moves UP/DOWN only when the predicted target is outside a paddle-size-aware
  deadband, otherwise uses NOOP; and
* never uses FIRE for a serve or away from contact.  A clear Jev feature
  judgment can authorize one FIRE pulse only within four predicted contact
  frames, for Robot Pong variation 1 or 2.

This isolates frame-rate execution/prediction as the change: an absent, late,
malformed, or uncertain Jev answer falls back to the same deterministic
movement controller and no FIRE.

## Jev state and question contract

Each request sends a strict compact JSON state, not the full raw observation:

* `phase` and `variation`;
* `right_paddle.y`, nominal height, and code-computed
  `target_relation` (`ABOVE`, `ALIGNED`, or `BELOW`);
* `ball.visible`, `approach` (`TOWARD_RIGHT` or `AWAY_OR_UNKNOWN`), and a
  code-computed time-to-contact bucket (`IMMINENT` <=4 frames, `NEAR` <=18,
  `FAR`, or `UNKNOWN`);
* `controller.target_y`, `contact_window`, whether FIRE is legal for this
  variation, and whether a feature pulse was already used this rally; and
* reward-derived own and opponent points.

It asks one independent `Choice` question, `feature`:

* `USE_FEATURE`: only if `contact_window` is true, the ball is approaching,
  the variation is Robot Pong 1/2, and alignment is safe;
* `NO_FEATURE`: all serves, non-contact frames, unknown/occluded situations,
  or uncertain alignment.

The response is associated with its snapshot.  Code accepts `USE_FEATURE` only
when its returned probability for that option is at least `0.60`, and then
rechecks the *current* observation’s variation and four-frame contact window.
It consumes at most one pulse per rally.  A late response cannot directly
select a stale movement action.  A missing probability, low confidence, call
failure, timeout, or garbage stand-in answer means no feature and ordinary
local movement.  The Jev call timeout is `1.0 s`; the 18-tick spacing prevents a
request storm and leaves the 60-Hz controller independent of the measured
roughly `0.15 s` Jev latency.

## Confirmation or refutation

Confirmation would be a materially higher mean return than `-20.333`, fewer
stale ticks, and traces showing the paddle following projected contact targets
instead of remaining at a clipped wall.  Own points should increase without
requiring FIRE outside the contact window.  Refutation would be a fresh-action
controller that still loses essentially every rally, especially when its
predicted target is within the paddle’s visible extent; that would indicate
that the public ball-speed/wall model or the action-to-paddle timing needs
calibration rather than that Jev scheduling was the main defect.  An unchanged
score together with low stale fraction would specifically refute the claim
that stale raw Jev actions caused the parent’s losses.
