# Candidate 0010 — opponent-aware edge choice with a latched actuator loop

## Parent and diagnosis

Parent: `candidates/0006/`; comparison: `candidates/0007/`.

The parent is the strongest confirmed harness (mean return 20.333, with returns
20, 20, and 21 and only 1, 2, and 0 stale ticks).  Its remaining failure is
not missed frame ownership.  In `candidates/0006/trace_seed1.jsonl`, the
controller reached `own_points=20, opponent_points=0` at tick 53,999, with a
live rally still running (`ball=(38.5,36.5)`, `left_paddle_y=38.5`, and
`right_paddle_y=113.5`) when the 900-second deadline arrived.  The code's edge
experiment alternated signs, but never used the visible left-paddle position.
Candidate 0007's more aggressive edge policy won two episodes but fell to 18
on seed 0, so simply making every return an edge return is less safe.

## Hypothesis

A small edge return should be chosen **away from the robot's observed paddle
side**, rather than by blind alternation.  The computer is the opponent that
must miss; if its visible paddle is materially above the current ball, prefer
the lower edge, and if it is below, prefer the upper edge.  This may turn the
parent's long, reliable rallies into points without adopting 0007's always-edge
risk.  The sign is latched for the planned contact, so changing pixels or a
late model answer cannot make the servo oscillate.

This tests a new structural mechanism: opponent-aware return-angle selection,
while preserving 0006's actuator-aware reflected-intercept controller and its
conservative mixed schedule.  Missing or weak opponent evidence falls back to
the parent's alternating sign.  The feature path remains disabled by default;
this isolates edge direction from the unproven FIRE effect.

## What changed

* Copied the proven 0006 frame-rate-owned controller, including visible-band
  braking, score/serve boundaries, and asynchronous model supervision.
* Added a code-owned `active_edge_sign`, reset at each point/contact and latched
  once for an edge attempt.
* When the schedule requests an edge and `left_paddle_y` and `ball_y` differ by
  at least 5 pixels, select the edge on the opposite side of the left paddle.
  Tiny differences, null/occluded paddles, center returns, and half-paddle
  uncertainty retain the tested alternating fallback.
* Jev and Astra cannot issue frame actions.  Jev can only recognize the supplied
  center/edge request; Astra can tune the rally-level mixed schedule.  Their
  malformed, late, or arbitrary stand-in outputs leave the local policy alive.

## System-1 interface, thresholds, and fallback

Each batched Jev request is strict JSON containing:

* scenario: variation, difficulty, and controlled paddle size;
* score and phase, including automatic serve/countdown state;
* ball approach, reliability, and ETA bucket;
* right-paddle visible band, measured velocity, predicted settled position;
* code-derived target relation, requested return style, contact window, and
  whether a legal feature is preapproved; and
* recent reward evidence and contact count.

The questions are independent:

1. `style_a` and `style_b` are reversed-order Choices among `CENTER`,
   `EDGE_UP`, `EDGE_DOWN`, and `HOLD_CENTER`.  Agreement at probability at
   least 0.56 may label the already code-generated edge choice; Jev does no
   arithmetic.
2. `feature` is a Choice between `USE` and `NO`.  It requires a live, safe,
   mode-legal contact, but the code currently requires the Astra plan and
   edge-contact preconditions too; the default plan says no feature.

A response is discarded on timeout, exception, malformed value, low
probability, old rally, or an expired snapshot.  The reflected local controller
always supplies the action.  Unknown ball state or serve delay centers the
paddle and never presses FIRE.  `RIGHT` means screen-y up and `LEFT` means
down; `NOOP` is the brake when the visible collision band covers the target.

## Astra and timing

Astra is launched asynchronously at startup and may replan after eight verified
right contacts without a point or after four point outcomes.  It receives the
public description, compact observation, current mixed policy, and recent
reward/contact evidence.  It may return only a short JSON rally policy
(`CENTER`/`MIXED`/`EDGE`, edge sign, cadence, bias fraction, brake horizon,
feature flag).  Numeric fields are clamped.  The frame loop never waits for it.
Jev uses a 0.9-second background request and Astra uses a 58-second background
request; `ctx.act` is posted before `ctx.wait_for_tick`, so late answers cannot
create stale ticks.  All model and future failures are advisory and caught.

## Confirmation or refutation

Confirmation is a mean above 20.333, especially a 21-point win for the former
seed-1 deadline episode, with stale ticks still near zero.  Traces should show
latched opponent-aware signs on edge contacts and shorter rallies without a
rise in opponent points.  Refutation is a clean, fresh-action run that still
finishes around 20:0 despite visible left-paddle separation, or a substantial
increase in losses relative to 0006.  If the adaptive sign improves survival
but not scoring, the opponent's paddle position is not a useful proxy for the
outgoing-angle outcome and the next experiment should measure contact-offset
physics directly rather than add more model supervision.
