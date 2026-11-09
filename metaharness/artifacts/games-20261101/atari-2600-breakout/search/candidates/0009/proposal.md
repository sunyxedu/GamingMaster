# Candidate 0009 — reflected interception with hysteretic actuator control

## Parents and concrete diagnosis

This is a crossover of `candidates/0005/` and `candidates/0008/`.

* 0005 supplied the useful mechanism: a code-owned reflected landing predictor
  that posts an action every native frame, with Jev/Astra advisory rather than
  synchronous.  It is still the leaderboard best at 288 points and zero stale
  ticks, so the basic real-time architecture should be preserved.
* 0008 supplied the useful robustness idea: measure paddle displacement and
  ball velocity from consecutive RAM observations instead of treating `NOOP` as
  an instantaneous brake or trusting an unmeasured actuator.  Its intended
  contact calibration was not evidenced in the selected traces, and its score
  fell to 234.

The failure evidence points to a narrower problem than model latency.  In the
0008 seed-1 trace, fresh actions alternate `LEFT`, `RIGHT`, `LEFT`, `RIGHT`
through frames 1045–1061 immediately before the life-loss observation at frame
1062; the same trace alternates around frames 1715–1731 before the next loss at
1732.  0007 shows the same reversal pattern at frames 3313–3317 before its
frame-3318 loss.  These are not stale ticks.  They are a feedback controller
that keeps changing its mind inside the paddle-impact window.  0005 is safer
than 0008, but its point estimate also lets small projection/actuator changes
choose a new direction every frame.  The previous generation therefore did not
need more Jev prose first; it needed an action-commitment policy.

## Hypothesis

The remaining wall-progress ceiling is primarily **direction chatter combined
with unmodelled paddle motion**.  A reflected intercept is useful, but a
controller should not reverse for a small change in the predicted error.  If it
measures the paddle's actual velocity, predicts only a short drift, and uses a
wide overlap corridor with hysteresis, it should keep the paddle catchable
through sticky actions and collision-induced target changes.  The best parent’s
simple projection is retained as the safety target.  0008’s measurement is
retained, but its bang-bang reversal is replaced by short committed pulses and
a cooldown after a brake.  A small, empirically reversible contact offset is
still available as an optional strategic experiment, fenced away from the
lethal window.

Confirmation would be: (1) the stale fraction remains near zero; (2) the trace
contains fewer alternating directional pairs near life loss; (3) elapsed frames
and score increase, especially in low 0005/0008 seeds; and (4) measured contact
samples either establish a stable outgoing-direction sign or the controller
safely falls back to `SAFE`.  Refutation would be equal-or-earlier losses with
long committed pulses, or a score reduction despite fewer reversals.  That
would suggest the remaining error is the ROM’s collision timing/angle table,
not actuator chatter, and the next candidate should use a more explicit
short-horizon search or remove shaping entirely.

## What changed in `harness.py`

1. `_reflect_x` and the descending projection from 0005 remain the only
   geometry used for a safety catch.  The ball’s missing/serve state is kept
   separate and FIRE is posted until the game accepts a serve.
2. Consecutive observations estimate paddle velocity, movement step, ball
   velocity, and whether NOOP drifts.  The action controller predicts at most
   six frames of paddle drift; it never extrapolates a slow model answer.
3. `_movement_action` uses a deadband, a meaningful reverse threshold, a
   two-frame directional pulse, and a two-frame brake-release cooldown.  A
   small opposite error produces `NOOP`, not an immediate mirror action.  The
   near/imminent target is always the unshaped safety target.
4. A descent target is low-pass filtered within one flight and reset after a
   brick/life/wall event.  During ascent, when brick collisions make a future
   return lane uncertain, the controller uses a stable conservative central
   anchor and measured braking rather than chasing an old speculative landing.
5. Jev chooses `SAFE`, `LEFT`, or `RIGHT` as a preference, not a raw action.
   If a side preference is accepted, code applies only a three-pixel contact
   offset during an early, normal-speed, certain descent with at least three
   lives and a blocked requested edge.  A descent-to-ascent transition near the
   paddle checks the outgoing `dx` and reverses the learned offset sign when
   the observed direction disagrees.
6. Astra gets one asynchronous strategic request for `SAFE`, `LEFT`, or
   `RIGHT`.  Its result is stored as `astra_plan` and included in the next Jev
   state; Jev must validate it against current edge/life/speed facts.  Astra
   never posts an action and never blocks the frame loop.

## System-1 interface and thresholds

Each Jev request is a strict JSON-compatible state with:

```text
{
  game,
  observation: {
    frame, event, score, lives, low_lives, serve_required,
    flight_phase, landing_urgency, landing_lane, paddle_relation,
    paddle_motion, speed_regime, return_certainty,
    left_edge, right_edge, wall_index, walls_cleared, bricks_remaining,
    shape_is_code_eligible
  },
  current_code_mode,
  astra_plan,
  rules: {legal_actions, decision, priority, safe_conditions}
}
```

Code computes all arithmetic, the reflected landing, overlap, and legality.
Jev only recognizes whether a strategic contact preference is appropriate.
Two independent `Choice` questions (`preference_a` and `preference_b`) use
rotated/reversed option order to reduce first-option bias.  The bridge accepts
the answer only if both values are identical, legal, and the smaller selected
probability is at least `0.50`.  It rejects answers older than 100 frames or
from an old plan version.  An open requested edge is converted to `SAFE` in
code.  A timeout, malformed answer, disagreement, or unavailable Jev leaves
the deterministic `SAFE` servo running.

Astra is launched once after the first live-ball observations (at most one
System-2 call).  It returns a token proposal; non-token prose is ignored.  The
proposal increments the plan version and triggers a fresh Jev judgment rather
than directly changing the paddle.  The model loop is asynchronous and each
posted action is executable without any model result.

## Fallback and real-time safety

Priority is: `NOOP` on termination, `FIRE` while `serve_required`, then the
measured-velocity reflected interceptor.  The near/imminent phase ignores
contact shaping and aims at the safety target.  The harness catches timeouts,
contract budget closure, malformed stand-in answers, and late futures; it does
not join background model work or run a background stepping loop.  It posts one
legal action before waiting for the next runner tick, so the evaluator’s stale
tick metric measures only actual control failure, not model latency.
