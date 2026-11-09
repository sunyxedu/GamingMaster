# Candidate 0011 — committed landing lane with measured counter-steering

## Parent and targeted hypothesis

Parent: `candidates/0005/`.  I keep its architecture: code owns the 60-Hz
interceptor, Jev is an asynchronous advisory classifier, and Astra is an
asynchronous side-shaping adviser.  I test one targeted control hypothesis:

> The remaining misses are caused in part by treating `NOOP` as a brake and
> reacting to a one-frame change in the projected landing.  A paddle already
> moving toward the landing needs a short measured counter-steer, and the
> landing lane should be committed before the impact window rather than chased
> frame by frame.

This is deliberately not a new tunnel planner or a larger model prompt.

## Trace evidence

The parent is fresh-action dominated, yet its worst episodes contain repeated
late reversals.  In `candidates/0005/trace_seed0.jsonl`, the loss at frame
2492 (score 22, paddle x=79) is preceded by this posted sequence at frames
2478–2492:

```
RIGHT, RIGHT, RIGHT, NOOP, NOOP, LEFT, LEFT, LEFT,
NOOP, NOOP, RIGHT, RIGHT, RIGHT, RIGHT, NOOP
```

The same pattern appears in `trace_seed4`: the loss at frame 6569 (score 125,
paddle x=70) follows five RIGHT actions, two NOOPs, three LEFT actions, two
NOOPs, then three RIGHT actions.  These are not stale ticks; the parent posted
fresh actions.  The generation lessons independently identify `0007` seed 3
RIGHT/LEFT alternation at frames 3313–3322 and `0008` seed 1 fresh
LEFT/RIGHT alternation around its first losses.  The traces support an
actuator-state/collision-timing change rather than more Jev prose.

## What changed

`harness.py` adds only a fast-controller guard:

1. It measures paddle displacement from consecutive RAM observations and keeps
   a small velocity estimate.
2. During descent it exponentially filters the reflected paddle-intercept
   target.  At about 14 frames from contact it commits that target until the
   next brick/wall/vertical or horizontal reflection.
3. In the near/imminent window, if the paddle is in the capture corridor or
   its measured velocity would carry it past the target, it issues a bounded
   opposite-direction pulse.  The pulse is latched for one or two frames;
   `NOOP` is not assumed to stop the paddle.
4. Jev biases remain small, expire, and are ignored near impact.  Astra still
   supplies at most one long-horizon LEFT/RIGHT/SURVIVE preference.  Neither
   model can override the committed geometric catch.

All arithmetic, reflection, event invalidation, legal action mapping, FIRE
serving, and malformed/late-answer fallbacks remain in code.  The controller
continues to post an action before waiting for the next realtime tick.

## Jev interface and thresholds

Each request contains strict JSON with:

- frame, event, lives, score, serve_required;
- code-derived flight phase, landing lane, landing urgency, ETA, landing
  center, capture-corridor boolean, speed regime, and control mode;
- measured paddle motion (`moving_left`, `moving_right`, or `still`);
- fallback action, Astra strategy, side-opening booleans, brick/wall progress,
  and plan version.

Two independent reordered `Choice` questions select `TRACK`, `CENTER`,
`BIAS_LEFT`, or `BIAS_RIGHT`.  An answer is accepted only if both agree, the
selected probability is at least 0.45 in both answers, it matches the current
plan version, and it is no more than 42 frames old.  Requests are separated by
24 frames and use a 1.4-second timeout; a late result never blocks the loop.
`TRACK` is the effective choice when urgency is near/imminent, although the
code ignores every Jev bias in that window.

The fallback is deterministic: FIRE whenever `serve_required` is true;
otherwise reflect the ball to the paddle band, follow the committed target,
and use measured counter-steering near contact.  Missing, malformed,
low-confidence, disagreeing, timed-out, or stale model results leave this
fallback in control.  `ctx.spawn` futures are the only background work and are
not joined after episode end.

## Astra revision and falsification

Astra is called once after the first live observation and may choose only the
long-lived shaping preference.  Its output is parsed defensively and its
strategy invalidates the Jev cache and target commitment.  A future interface
revision is warranted only if the traces show accepted strategic advice is
causing misses or the fast controller survives but fails to remove bricks.

The hypothesis is confirmed if, relative to `0005`, the action traces show
fewer late direction oscillations, fewer lives lost at comparable score, and
higher mean score without increasing stale ticks or episode errors.  It is
refuted if misses persist despite stable committed lanes and explicit braking,
or if counter-steering systematically moves the paddle away from otherwise
catchable balls; in that case the velocity estimate or the assumed Atari
actuator response must be measured differently rather than delegating control
to Jev.
