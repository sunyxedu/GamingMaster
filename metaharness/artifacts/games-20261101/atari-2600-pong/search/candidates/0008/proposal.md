# Candidate 0008: attack the stable draw with contact-timed feature pulses

## Parent and diagnosis

Parent: `candidates/0005/` (which itself combined the predictive controller from
`0003` with the idle controller's model-independent fallback). I also used
`0003` as the comparison because it reached `21-2` on one training scenario.

The parent fixed the first-generation execution failure: it posted a fresh local
action on essentially every frame (`9` stale ticks in each 53,999-tick deadline
episode) and kept model calls off the collision loop. That change is necessary,
but its score pattern identifies a different failure:

* `0005` seed 0 ran for 53,999 ticks with **0:0** and no score event. Its final
  observation was still a live rally (`ball=(68.5,50.5)`, `ball_v=(1,2)`,
  right paddle `y=157.5`). The controller was not dead; it was sustaining a
  draw.
* The same local controller earned only **2:0** on seed 1 and **21:1** on seed
  2. Thus better freshness and defensive interception alone do not reliably
  turn a returned ball into a point.
* Both `0003` and `0005` recorded only `NO_FEATURE` Jev judgments, and all of
  `0005`'s Astra plans chose `safe`, `use_feature=false`, and zero bias. No
  candidate has actually tested the study's variation-specific Speed/WHAMMY
  contact behavior.

The concrete failure is therefore not another stale-action problem. A centered,
repeatable return can be good enough for the computer to return forever. Against
that opponent, survival has positive value only until the frame deadline; it does
not produce the primary score. The missing mechanism is an intentional attacking
return: put the ball on a safe paddle edge to produce a steeper outgoing angle,
and test the Robot Pong button at the collision rather than disabling it forever.

## Hypothesis

**A mode-legal, one-frame FIRE pulse at a deliberately off-center right-paddle
contact will break the stable defensive rally and increase own points, provided
that the predictive frame controller still supplies the paddle position and
code—not a model—decides the exact pulse frame.**

This is a structural change from `0005`: attack is the code's conservative
fallback instead of an optional Astra setting that defaulted to safe, and Jev
now authorizes an upcoming contact pulse rather than deciding whether to press
FIRE on a stale snapshot. Robot Pong 1 is treated as the documented Speed form;
Robot Pong 2 as the documented WHAMMY form. The harness never presses FIRE during
automatic serve delay or on a non-Robot variation.

## What changed

1. **Keep the proven frame-rate bridge.** The loop observes, computes, posts one
   legal action, and waits for exactly one emulator tick. Astra/Jev futures are
   polled but never awaited by the frame controller. A failed, late, malformed,
   or unavailable advisory answer leaves frame control running.
2. **Use an alternating edge target.** When the ball is approaching right, code
   projects its y to `x=140.5` with top/bottom reflection. In attack style it
   offsets that target by `+3.5` or `-3.5` visible-pixel centers, alternating
   after rallies. For a nominal half paddle the offset is capped at `1.75` so
   the ball remains overlapped rather than turning the experiment into random
   misses. Away travel and uncertain/serve states still use the court center.
   This tests edge angle, not seed-specific opening behavior.
3. **Make attack the safe fallback for this experiment.** The initial plan is
   `style=attack`, `use_feature=true`, strength `3.5`, deadband `2.5`. An
   asynchronous Astra plan may change style, strength, deadband, or feature
   policy after parsing a short JSON object; malformed prose cannot change the
   defaults. After two observed point events one diagnostic Astra call may
   revise the policy from recent reward evidence. Astra is never allowed to
   supply a one-frame action.
4. **Use Jev as a contact-intent judge.** A Jev request is launched when a
   rightward contact is within 55 projected frames (or perception is uncertain).
   It returns two reversed-order Choices. Only agreeing `USE` answers with both
   per-answer probabilities at least `0.60`, from the same rally and within the
   request's predicted contact horizon, arm one pulse. At the live contact
   window (at most four frames), code rechecks phase, variation, approach,
   feature policy, and that no pulse has been used in this rally. It maps
   `UP/DOWN/HOLD` to `RIGHTFIRE/LEFTFIRE/FIRE`, respectively. Allowing plain
   `FIRE` when already aligned is deliberate: the button must not be lost merely
   because movement has stopped.
5. **Reset at verified boundaries.** Reward-derived score changes and
   `serve_delay`/`finished` invalidate old ball velocity, feature votes, and the
   one-pulse-per-rally latch. A stale ball sprite on a point frame is therefore
   not treated as a new trajectory.

## System-1 state and questions

Every request sends strict JSON built from the current observation and code-owned
facts. It contains:

* `scenario`: mode, difficulty, variation, right-paddle size, and the documented
  feature kind (`Speed` for Robot Pong 1, `WHAMMY` for Robot Pong 2);
* `score`: reward-derived own/opponent points and last reward;
* `observation`: frame, phase, and verified serve countdown/waiting state;
* `ball`: visible or short-gap estimated x/y, signed vx/vy, approach, reliability,
  and visibility gap;
* `right_paddle`: y, nominal height, and measured y velocity;
* `control`: both safe and attack predicted contact y, attack offset, named
  target relation, code movement action, numeric time-to-contact plus named
  bucket, `contact_window`, `feature_horizon`, `feature_legal`, and whether the
  rally has already consumed its pulse;
* `plan` and the last four reward events.

It asks two independent Choices in one request:

* `feature_a`: whether to authorize one pulse for the **next** right-paddle
  contact; options are `USE` and `NO`;
* `feature_b`: the same judgment with the option order reversed.

The instructions explicitly define authorization as a future intention, not an
immediate FIRE action. Jev is not asked to add numbers, integrate velocity,
reflect walls, choose a raw action, or discover a RAM mapping. Code performs
those operations. A missing value, non-string stand-in, disagreement, probability
below `0.60`, wrong rally generation, or expired snapshot is ignored. The
fallback is ordinary predictive movement with no feature pulse. An accepted vote
expires at the predicted contact plus six frames and is rechecked against the
latest observation before use.

## Astra interface and adaptation

The initial asynchronous prompt includes `game.describe()` and the compact
initial observation. It tells Astra that this candidate is specifically testing
edge returns and the mode-legal button, and asks for only:

```
{"style":"attack|safe", "use_feature":true|false,
 "attack_strength_px":0..6, "deadband_px":1.5..6, "diagnosis":"..."}
```

The code clamps all numeric fields. The fallback remains attack/feature-enabled
if Astra times out, returns arbitrary text, or never runs. Once two point events
are visible, one second prompt supplies recent reward and target evidence and
asks whether edge/feature play is helping or causing misses. This is a policy
revision trigger, not a frame-control dependency.

## Fallback and timing policy

* The local reflected-intercept controller is executable with zero model answers.
* Jev timeout is `1.0 s`, safely above the measured p90 and max-ish warm latency;
  one request is outstanding and launches no more often than every 24 frames.
* Astra is launched through `ctx.spawn` with a 60-second timeout and is never
  joined by the game loop. Its future has no loop of its own, so cancellation at
  game/deadline end is safe.
* `ctx.act` is posted before `ctx.wait_for_tick`; hence a model latency cannot
  create a stale repeated action. Invalid or missing action names fall back to
  the legal NOOP name.
* FIRE is impossible during `serve_delay`, without a live rightward contact,
  outside Robot Pong 1/2, after a pulse in the current rally, or after an
  expired Jev authorization.

## Confirmation or refutation

Confirmation would be a higher mean return than `0005`'s `7.333`, especially
turning the `0005` seed-0 scoreless deadline into own points, while stale ticks
remain near zero. Traces should contain `contact_feature_pulse` events only at
rightward contact windows, show nonzero attack offsets, and show shorter rallies
ending in positive reward or an opponent miss. Robot Pong 1 and held-out Robot
Pong 2 should each show legal, variation-specific use rather than serve FIRE.

Refutation would be a clean frame-rate trace with repeated pulses and edge
contacts but the same scoreless rallies, or a substantial increase in opponent
points while centered no-feature control would have survived. That would mean
that Speed/WHAMMY is ineffective or that the offset/contact model is wrong, not
that asynchronous execution failed. If pulses are absent despite `feature_horizon`
being valid, the Jev authorization threshold/interface—not the game tactic—is
the next thing to revise.
