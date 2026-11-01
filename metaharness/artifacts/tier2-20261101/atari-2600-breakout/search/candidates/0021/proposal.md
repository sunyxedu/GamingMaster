# Candidate 0021 — flight reservations with an evidence-gated contact objective

## Parents and crossover diagnosis

The direct parents are `candidates/0019/` and `candidates/0018/`.

* **0019's score mechanism:** a code-owned 60-Hz reflected interceptor, six-sample velocity cadence, an actuator model that included queued/sticky input, and a *one-flight/one-objective* bookkeeping rule.  It was the best selected candidate (743.5 mean) and reached 864 on two training episodes.  Its sparse-wall cleanup ray was the right kind of progress mechanism: code generated candidate sides from observed endpoint directions and the remaining bricks, rather than asking Jev to do geometry.
* **0018's robustness mechanism:** it treated a reported paddle rectangle as an unverified envelope, recorded actual descending crossings and losses, fitted delayed paddle motion from public observations, and let Astra/Jev audit evidence rather than emit frame actions.  Its local six-seed validation reached 816, although its official evaluation was lower and still contained moving-paddle failures.

The failure evidence points below the model layer.  In the 0019 seed-2 trace, the miss at frame 5510 goes through actions `RIGHT, RIGHT, NOOP, LEFT, NOOP, RIGHT, RIGHT, NOOP` while the predicted intervals remain approximately `[123,135]`; the seed-1 miss at 7665 similarly reverses during the final eight frames.  These are not stale ticks: every action was fresh.  In 0018 seed-1, failed contacts at frames 6495 and 7832 had prediction residuals below one pixel but paddle drift of 3–5 pixels/frame, with fresh reversals around the crossing.  Conversely, 0019 correctly showed that repeated `endpoint_not_reached` trials must not flip the learned LEFT/RIGHT mapping: in seed 2, many requested LEFT trials observed outgoing RIGHT at offsets near zero and were not reliable endpoint evidence.

## Hypothesis

The parent controllers replan a *direction* at every native frame even after the paddle has entered the catch envelope.  Quantized ball coordinates and a queued actuator make that a bad control lifetime: a one-frame change in a projected interval can reverse a command while the paddle is still carrying momentum.  The missing mechanism is a **flight-level collision reservation**:

1. recompute the landing envelope from current RAM and observed bounce-plane evidence;
2. use the fitted queued-input model to choose an action while the return is early;
3. when the return is within ten frames, reserve the collision window, hold a direction for a short lease, release as soon as the paddle is safely inside, and permit at most one clearly necessary emergency reversal;
4. never narrow the hard catch interval to pursue an endpoint.  An endpoint/tunnel choice is a soft utility target and is started only when a short rollout proves that it is reachable.

This combines 0019's sticky/queued prediction and one-flight objective with 0018's measured crossing evidence and online contact-plane/landing-bias calibration.  It is a structural change in the action lifetime, not another Jev label or a new hard-coded bounce-angle table.

## What changed in the harness

### Deterministic 60-Hz path

* FIRE is posted whenever `serve_required` is observed; a lost ball increments no episode state beyond the life/rally bookkeeping.
* The controller reads only compact public observations and posts a legal action before waiting for the next tick.  `ctx.wait_for_tick()` is the only synchronization in wall mode.  Inference is never on the reflex path.
* Ball velocity uses a same-sign six-sample window.  Zero displacement is retained as quantization, while sign changes, large speed changes, serves, brick/side contacts and paddle bounces clear the window.
* The landing estimate uses reflected x motion, a contact plane learned from actual successful crossing y values (bounded to the observed 184–188 range), three nearby velocity/plane samples, and an observation-derived landing bias.  No private emulator state or seed-specific schedule is used.

### Reservation motor

The paddle model is `v_next = a*v + b*u + c*u_previous`, initialized from the two parents and refit with ridge-like sufficient statistics from interior public transitions.  It is blended conservatively and bounded.  Each candidate action is evaluated under fresh and sticky first-input branches, with a cost for leaving the catch interval, terminal motion, and only a small optional contact goal.

The new action lifetime is explicit:

* **approach:** early descending frames use the one-frame queued-input rollout; a paddle far outside the interval always gets a directional reach action rather than an absorbing NOOP;
* **reserve:** at ETA <= 10, the first chosen direction is leased for 1–2 frames.  Entering the interval cancels the lease and posts NOOP; a direction is not mirrored merely because a quantized interval edge moved;
* **impact:** a direction may flip only after a three-frame cooldown and only once per flight, when the rollout is clearly outside the interval.  Otherwise the controller keeps the reservation or releases it.  NOOP is simulated as carry-through, not as an instantaneous brake.

This is intentionally not 0017/0020's unconditional continuity lock: it can cancel a direction on actual entry, and it has one observation-driven emergency correction.  It is also not 0018's unconstrained 3^4 sequence choice at every tick.

### Evidence and scoring objective

Every descending contact stores the last public samples, actual paddle drift, projected residual, offset and trial metadata.  A bounce is a catch record; a disappearance is a miss record.  Stable, prediction-supported catches update a conservative offset band, landing bias and observed contact plane.  A miss immediately cancels shaping and restores the broad catch fallback.  Endpoint mappings are learned only from an achieved, stable contact with several post-bounce velocity samples; an endpoint-not-reached trial cannot reverse the mapping.

A tunnel objective is one trial per flight, gated by four lives, two safe catches, normal speed, a wide early interval, an occupied corresponding edge, and a reachable short rollout.  The objective changes only a soft `goal` in the motor cost, while the full physical catch interval remains hard.  The sparse-wall objective reuses 0019's measured outgoing-speed rays.  For a wall with more than eight bricks it requires at least three reliable fast contacts and an observed endpoint mapping; for the final eight bricks it permits one bounded fallback ray scored against both remaining-brick sides, still requiring at least three reliable fast-flight contacts.  It is disabled with one life and for a mismatched speed regime.

## Astra and Jev bridge

### Astra

Astra is called asynchronously at most three times: once per wall/stage when the initial objective is available, and again only for a newly detected sparse-wall cleanup stage.  Its prompt includes wall, lives, score, remaining bricks, side occupancy, speed, stable contact/miss counts, endpoint support and the last trial outcome.  It must return exactly `SURVIVE`, `LEFT` or `RIGHT`; prose is parsed defensively and any other response is ignored.  A returned token is only a prior.  Life, wall, open-side and current-speed gates are reapplied, and Jev/code still validate the next contact.  An old result cannot alter geometry.

The strategic revision trigger is observed progress, not RAM cadence: after reward has stalled with at most twelve bricks, code can enter cleanup; a wall transition resets the prior.  This tests whether Astra is useful for choosing a measured sparse-wall ray while leaving survival to code.

### Jev state schema

Each request is strict JSON and contains:

```text
{
  game, rules,
  observation: {
    frame, event, lives, score, wall, bricks_remaining,
    phase, urgency, speed, catch_interval, safe_for_shaping,
    candidate_side, objective_kind, left_edge, right_edge,
    rally_catches, trials_this_life, current_objective, astra_proposal,
    trial_outcome, last_outgoing, endpoint_support,
    recent_contacts, contact_id, safety_epoch, astra_revision
  }
}
```

All coordinates, reachability, brick ray intersections, counts and comparisons are computed in code.  Jev sees named buckets and a short evidence tail, not a raw 108-cell calculation task.

One `ctx.decide` batches:

1. `objective_a`: Choice among `SURVIVE`, `LEFT`, `RIGHT`, `NO_CHANGE`;
2. `objective_b`: the same Choice with reversed/rotated option order, guarding the provider's first-option bias;
3. `safe_now`: a speculative Noul about the precomputed `safe_for_shaping` fact.

Only the two Choices affect control.  They must agree and the lesser selected-option probability must be at least **0.58**.  The answer must match the safety epoch, wall, contact, Astra revision and be at most 120 frames old.  Current lives, speed, side occupancy and cleanup support are checked again on receipt.  `LEFT`/`RIGHT` selects at most the next contact's soft goal; it is never converted to a raw action.  `SURVIVE` is a one-contact veto.  Missing probability maps, malformed values, disagreement, timeout or a late answer leave the deterministic reservation active.

Requests are launched only at serve/paddle-bounce/life/wall boundaries or when a newly eligible contact is detected, with one outstanding request and a 45-frame scheduling gap.  There is a call-budget reserve, and no polling thread: completed futures are collected in the native-frame loop.  Model errors are logged and ignored, so arbitrary pre-flight stand-in text cannot crash the episode.

## Falsification and expected evidence

Confirmation would be a higher mean than 743.5 with at least the same first-wall completion and fewer late-window direction reversals.  Traces should show `flight_reset`/`contact_evidence`, a stable late reservation, fewer `endpoint_not_reached` trials, and successful endpoint measurements that actually influence soft goals.  A useful ablation is the same reservation motor with `objective=AUTO`; if scores remain similar, the motor deserves the credit rather than the model bridge.

The hypothesis is refuted if stable catches inside the reservation still disappear while the paddle is stationary, if the one-flip rule causes systematic edge misses, or if the learned contact-plane/bias estimate drifts across ordinary flights.  If survival improves but the score stalls near a sparse wall, the remaining problem is measured outgoing-angle coverage/ray selection, not more model calls.  If cleanup causes losses, disable it before relaxing the hard interval; the tunnel prior is subordinate to a live ball.

The module uses only the standard library and `metaharness.*`, has no process/server/screenshot access, and has no cross-episode or seed-conditioned state.
