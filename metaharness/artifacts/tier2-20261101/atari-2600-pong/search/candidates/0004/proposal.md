# Candidate 0004 proposal

## Parent and hypothesis

Parent: `candidates/0002-tier2_idle/`.  The relevant comparison is the best
prior candidate, `0000-tier2_jev_only`; the idle parent is useful as the passive
`-21` floor.

My hypothesis is that the failure is structural, not that Jev needs a better
wording for “which action?”  The best candidate asks Jev to choose a raw action
on every request and then waits for that answer before posting it.  The trace
shows a request every roughly 8--9 emulator frames, with a median latency of
0.138--0.151 seconds and 88.96% stale ticks.  More importantly, the answer is
not a useful substitute for prediction.  In `trace_seed0.jsonl`, at frame 63
the ball is moving left (`ball_vx_sign=-1`) and Jev chooses `LEFT`; after that
input is repeated, frame 80 has the right paddle at `y=137.5`, frame 88 has it
at `y=183.5` and Jev changes to `NOOP`, and frame 97 still has the paddle at
`y=191.5` before Jev chooses `RIGHT`.  This is delayed chasing and braking
after the paddle has already travelled too far, rather than projecting the
contact point.

The score-change observations show the consequence.  In the seed-2 trace the
right side reaches `-1, -2, ... -20` at frames 256, 636, 1020, ... while the
phase is `serve_delay` and the verified serve countdown is 63.  It makes no
positive return before the final `-21`.  The Astra parent is an even stronger
negative control: its calls take 33--40 seconds, produce 100% stale ticks, and
it posts at most one action in an episode.  A slow model must not own the
frame-critical loop.

The test in this candidate is therefore: move exact game arithmetic and
execution into a local controller, and reserve both models for asynchronous
rally-level policy choices.  If predictive geometry is the missing ingredient,
this should beat the Jev-only `-20.333` mean while posting an action on almost
every tick.

## What changed

`harness.py` has three layers.

1. **Frame controller (always active).**  It posts a legal action immediately
   and once per tick; it never waits for a model.  During `serve_delay` it
   returns the paddle to the court centre.  During an outbound rally it keeps
   central coverage.  When `ball_approaching_right` is true, code computes the
   time to the right paddle and reflects `ball_y + vy*time` between the observed
   ball-centre extrema 37.5 and 191.5.  It then moves toward that intercept
   with a small deadband and one-frame settling prediction.  Thus the action
   controls the predicted contact, not the current ball location.  Missing or
   occluded velocity falls back to the last short-lived target or centre.

2. **Rally policy.**  The first rally is a central, high-margin calibration
   rally.  Later rallies alternate a small upper/lower edge bias; the bias is
   scaled down after a lost point and up after a won point.  This tests the
   study's claim that the Robot Pong computer is vulnerable to changed return
   angles without making a half-size paddle depend on a large edge gamble.
   FIRE is never used during serve delay.  For Robot Pong 1/2 it is optionally
   pressed for one frame only in the verified right-contact window, using the
   appropriate movement+FIRE action, so Speed/WHAMMY is not held throughout a
   rally.  A score event resets that one-frame latch.

3. **Asynchronous model bridge.**  Jev is asked at the start of a rally about a
   named return style (`CENTER`, `EDGE_UP`, `EDGE_DOWN`) and whether to use the
   variation feature at contact.  It does not select a frame action or perform
   arithmetic.  Astra is only escalated after three opponent points, while the
   local loop continues; it can revise the macro mode and feature gate, not
   paddle timing.  Strict parsing means arbitrary or incomplete model text is
   ignored.

This is deliberately a structural change from both parents: the model is no
longer a synchronous actuator, and the harness owns time, geometry, score
bookkeeping, legal action mapping, and the late-answer fallback.

## System-1 interface and freshness policy

Each Jev request carries a compact strict-JSON state, not the full observation.
The schema is:

```json
{
  "facts": {
    "phase": "rally|rally_ball_occluded|serve_delay|...",
    "approach": "toward_right|away_from_right|unknown",
    "contact_eta": "far|near|contact|unknown",
    "projected_contact_band": "upper|middle|lower|unknown",
    "paddle_relation": "above_target|aligned|below_target|unknown",
    "right_paddle_size": "full|half|unknown",
    "feature_legal": true,
    "score_state": "ahead|behind|tied",
    "local_default": "CENTER|EDGE_UP|EDGE_DOWN"
  },
  "controller_rule": "code computes the trajectory and posts the frame action",
  "candidate_meanings": { "CENTER": "...", "EDGE_UP": "...", "EDGE_DOWN": "..." },
  "rally_id": 3
}
```

The values such as `contact_eta` and `projected_contact_band` are computed in
code before the request.  Jev is not asked to compare coordinates, count
frames, add scores, or invent an action mapping.

The request batches three independent typed questions:

* `style_a`: choose one of the three styles;
* `style_b`: the same judgment with the option order reversed, to reduce the
  documented first-option bias;
* `feature`: `USE_AT_CONTACT` or `NO_FEATURE`.

Both style answers must agree and each must assign at least 0.58 probability to
the selected value.  A feature answer must assign at least 0.72 to
`USE_AT_CONTACT` or `NO_FEATURE`.  Answers are submitted with
`ctx.spawn`, tagged by `rally_id`, and used only if they arrive during that
rally.  A late answer, an invalid value, a timeout, a failed request, or
ambiguous probabilities is discarded.  If Astra has explicitly selected
`CENTER`, Jev cannot override that safety constraint with an edge style.

The fallback is fully executable without Jev: centre on the first rally, then a
small alternating edge bias with the local score-based scale; no FIRE unless
the local contact gate and the accepted policy both permit it.  Consequently a
slow or unavailable Jev answer changes no frame deadline and cannot produce an
invalid action.

## System-2 scheduling

There is at most one Astra request.  It is started only after the third negative
point, with the current verified observation, score, miss streak, and edge
scale.  Astra is instructed to return exactly:

```text
MODE: CENTER|EDGE
EDGE: UP|DOWN
FEATURE: YES|NO
MARGIN: SMALL|MEDIUM
```

Only these explicit fields are parsed.  Astra does not see a requirement to
supply a frame action, and its output is not placed directly in `ctx.act` or in
an arithmetic expression.  If it has not returned before the match ends, the
local controller simply finishes; a cancelled/failed call is harmless.  This
is the “go back to Astra” trigger: repeated lost points are evidence of a
macro-policy problem, while a single late observation is not enough to stall a
real-time controller.

## Confirmation and refutation

Evidence confirming the hypothesis would be:

* training mean return above `-20.333`, with at least some rallies surviving
  the first right-side contact and ideally positive returns from the edge or
  contact-feature policy;
* trace observations showing the right paddle moving toward the reflected
  contact band before `x` reaches the right paddle, rather than oscillating
  around the current ball; and
* near-zero stale ticks despite Jev/Astra latency.

Evidence refuting it would be a clean, zero-stale run that still loses 21--0 or
21--1 because the projected target systematically misses, or a run in which
one-frame FIRE/edge contacts reduce score relative to the central policy.  In
that case the next iteration should retain the per-frame predictive controller
but remove the feature/edge experiment or recalibrate the ball-centre bounds
and paddle settling margin; it should not return to synchronous model actions.
