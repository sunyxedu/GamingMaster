# Candidate 0006: adaptive return-policy bridge

## Parent and concrete diagnosis

Parent: `candidates/0005/` (which crossed the local controller from 0002 with
Jev/Astra supervision); the best reference is `0003/`.

The parent proved that a local frame loop is necessary: 0005 posted fresh
actions on 99.99% of ticks and still scored `0, 2, 20`, while 0003 scored
`7, 0, 19`. Thus the remaining problem is not model latency. It is two
controller-level failures that the parent did not test:

* 0005's Astra plans were `safe`, zero bias, and no feature in all four plans.
  The seed-0 episode consequently remained `0:0` through 53,999 ticks and a
  deadline. A center-return policy can survive without ever making the robot
  miss. 0004 shows the same symptom in seed 1 (`0:0` at the deadline), and its
  Jev style answers were overwhelmingly `CENTER`; the proposed attack
  mechanism was therefore never exercised.
* 0005 computes and logs `paddle_velocity`, but `_trajectory` makes its action
  from the instantaneous paddle center only.  The startup trace visibly
  oscillates in four-frame blocks (`LEFT` x4, `NOOP` x4, `RIGHT` x4, ...), and
  the failed seed-1 trace repeatedly reports a target near `54.5`, a paddle at
  `45.5`, and `DOWN` while the same trajectory recurs.  This is consistent with
  acting on a filtered/settling paddle as though each posted direction stopped
  immediately.  The public contract explicitly warns that `NOOP` can take a few
  frames to settle.

## Hypothesis

A strong harness needs two separate adaptive loops, not model-selected raw
actions or a permanently safe macro:

1. A **collision loop** should control the paddle by overlap with its visible
   collision band and a short actuator prediction.  It should brake/hold when
   the band already covers the reflected intercept, rather than chase the
   target center and reverse every few frames.
2. A **return-policy loop** should preserve safe center contacts long enough to
   calibrate, then deliberately test small upper/lower edge contacts.  Against
a robot that tracks ordinary returns, a mixed edge schedule (and, only when
   approved at a verified contact, the variation's Speed/WHAMMY button pulse)
   is the missing mechanism for converting an endless survival rally into
   points.  A miss resets briefly to center; a successful point alternates the
   edge sign.

This is a structural rewrite of the parent: code now owns exact actuator-aware
execution and a generic, score-independent attack schedule; Astra authors and
revises that schedule from verified rally evidence; Jev recognizes whether a
named contact/feature request is supported. Neither model is on the 60-Hz
critical path.

## What changed in `harness.py`

* The local controller projects visible ball position and velocity to the right
  contact x with wall reflection, as before, but compares the projected y with
  the **visible paddle band** (`right_paddle_y` plus
  `right_paddle_height_px`) rather than only comparing paddle centers.  It
  predicts the paddle's short settling displacement from consecutive observed
  y values, uses NOOP as a brake when overlap is already safe, and makes a
  reversal only when the target is outside an expanded band.  Missing/serve
  velocity and stale point sprites fall back to center coverage.
* A point boundary resets trajectory and button state.  The controller counts
  completed right-side contacts without using score-specific openings.  It
  begins each point with center coverage, then uses a code-owned mixed policy:
  an occasional small edge bias, alternating upper/lower, and a forced edge
  experiment after many contacts without a point.  Bias magnitude is a
  fraction of the observed nominal paddle size and is range-clamped, so half
  paddles are not given the full-paddle offset.  After an opponent point a
  short center recovery is mandatory; after an own point the edge sign changes.
* Astra is no longer asked merely for a safe/no-feature answer.  It is an
  asynchronous policy author: it may set `CENTER`, `MIXED`, or `EDGE`, the
  edge sign, attack cadence, bias fraction, warmup count, settling horizon,
  and whether a legal contact feature is worth testing.  It is called at
  startup and can replan after either repeated scoreless right contacts or
  several point outcomes.  JSON is optional and every field is clamped; a
  timeout, malformed answer, or cancelled future leaves the executable mixed
  fallback intact.
* Jev is a per-rally/per-contact recognition gate rather than a raw-action
  chooser.  It receives the compact derived state and the code-computed
  target/band facts.  It independently chooses a named `CENTER`, `EDGE_UP`,
  `EDGE_DOWN`, or `HOLD_CENTER` interpretation (with reversed option order for
  bias control) and a `USE`/`NO` feature judgment.  Code requires agreement and
  probability thresholds, tags the result with the rally/contact snapshot,
  and only allows FIRE when the current variation is Robot Pong 1/2, the
  projected contact is within four frames, the selected edge policy is safe,
  and the pulse has not already been used for the point.  Jev can veto an
  uncertain edge and feature, but cannot replace a reliable code trajectory.

## Jev contract

The strict JSON state contains only:

* scenario: variation, difficulty and right paddle size;
* score and phase/serve fields, frame and completed right contacts this point;
* ball approach (`RIGHT`, `AWAY`, `UNKNOWN`), reliability, and an ETA bucket;
* current paddle y, its visible collision band, predicted y after settling, and
  the code direction; and
* the code-requested contact style, target relation, contact window, and the
  explicit feature precondition.

The questions are batched and independent:

* `style_a` and `style_b`: agreement-gated choices among `CENTER`, `EDGE_UP`,
  `EDGE_DOWN`, and `HOLD_CENTER`; Jev recognizes the supplied request and does
  no arithmetic;
* `feature`: `USE` or `NO`, with `USE` explicitly restricted to a verified
  imminent Robot Pong contact and a preapproved Astra edge experiment.

Each answer must have probability at least `.56` for style agreement; feature
needs `.62`. A missing probability, low confidence, malformed stand-in, model
error, or answer tagged to an old rally is ignored. Requests use a `0.9 s`
timeout, run through `ctx.spawn`, and are never awaited by the frame loop.
A code-derived trajectory is always the fallback; unknown/occluded states hold
or cover center. Thus a late Jev result cannot select a stale movement action.

## Astra revision trigger and evidence

The initial Astra call sees the public game description, the initial verified
observation, current policy, and no hidden ROM state. Later calls are started
only after eight scoreless right contacts or four point outcomes, and include
recent reward/contact evidence. This tests the claim that System-2 is more
useful as a policy author/diagnostician than as a frame controller. A maximum
of three asynchronous calls is allowed; a short JSON parse failure is benign.

Confirmation would be a higher return than 0005's `7.333` mean, especially
turning a trace that previously ended `0:0` at the deadline into scored points
or a match win, while retaining near-zero stale ticks. Traces should show
visible-band overlap and fewer direction reversals near contact, followed by
edge contacts after scoreless survival and a replan whose policy changes from
its fallback. A useful feature experiment would show only one-frame FIRE
pulses at safe contacts.

Refutation would be repeated misses with the ball visibly inside the paddle
band, or no scoring improvement despite many clean edge returns and no stale
ticks. That would point to a wrong contact-angle/feature assumption rather
than a model scheduling problem. If the band servo improves survival but never
scores, the next iteration should change return-angle mechanics, not restore
synchronous model action selection.
