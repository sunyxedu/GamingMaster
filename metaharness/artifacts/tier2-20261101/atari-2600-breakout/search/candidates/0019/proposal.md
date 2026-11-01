# Candidate 0019 — flight-scoped contact objectives and measured interception

## Parents and central hypothesis

This is a crossover of **0015** (mean 420.167) and **0016** (312.333).

* **0015's useful mechanism:** same-sign temporal RAM velocity filtering,
  receding-horizon reflected paddle-overlap intervals, and a code-owned loop
  which never waits for inference. Its score is a survival-controller result,
  not evidence for successful Astra planning: all six Astra calls timed out.
* **0016's useful mechanism:** interior contact endpoints as a scoring actuator,
  with Jev validating an objective rather than an action, and a realistic
  asynchronous Astra timeout. Its intended contact experiment was not actually
  validated by its evaluation: it recorded no confirmed or flipped endpoint
  mappings in any of the six traces.

**Hypothesis:** a contact objective must live for a flight, while geometry must
be refreshed at native-frame cadence. Mixing those lifetimes makes quantized
motion look like a new experiment. Combining filtered interception with a
one-flight endpoint objective should produce *completed, measured* trials, and
then permit safe trajectory choices when a nearly cleared wall stops scoring.

The first implementation and this proposal were written before live testing.
Refinement retained this architecture but corrected motion/collision assumptions
that the new contact telemetry falsified. There is one submitted harness; the
files in `diagnostics/` are snapshots/tests of its development, not other
candidates.

## Concrete archive diagnosis

### 0016: a trial quota consumed without a contact

In its worst episode, seed 1, `edge_contact_trial` occurs at **284, 286, 288**.
The corresponding `catch_corridor` at 284 is `[21,37]`, landing 37; at 285 it is
`[56,72]`, landing 72, with shape SURVIVE. It then alternates `[20,36]`, `[55,71]`,
`[19,35]`, `[54,70]`. Three trial starts are consumed during one descent, not
three paddle contacts. The RIGHT cluster at **871/873/875** repeats the defect.
There is no contact-confirmation event. Its raw speed/corridor updates can erase
the objective on the alternating RAM samples.

That episode loses lives at **611, 3001, 5360, 5587**, finishing at **99**.
Astra does return RIGHT at frame **2915**, latency **48.437 s**. Thus its slow
plumbing is worth crossing over, but its raw/latching geometry and trial
bookkeeping are not.

### 0015: fresh control still misses

Its worst episode, seed 5, ends at **382**, with eleven bricks left. Losses are
logged at **3470, 6284, 7017, 8699**. Before the first loss the landing estimate
stays around 42 rather than flipping; NOOP persists through ticks 3459–3461
before late LEFT. Before the third loss NOOP persists through 7009. Those old
logs omit paddle positions at the relevant ticks, so they cannot prove that
nominal overlap was physically safe. This harness adds that missing evidence.

This acts on the archive's repeated lesson: stale ticks are already near zero;
more model mode labels or another unverified braking deadband are not a solution.

## What local observations changed

The initial crossover resolved the quota defect: its first seed-5 probe starts
one RIGHT trial at 322 and resolves it at 384. That probe scores 357, while seed 1
scores 434. These are not improvements over both corresponding parent scores.
They establish that the trial can now reach a contact.

The resulting public-observation telemetry exposed three additional assumptions:

1. **Input delay and release motion.** A fit to 2,753 interior paddle transitions
   from those two probes gave approximately
   `v_next = 0.596 * v + 2.465 * previous_requested_direction`, with 0.484-pixel
   residual RMS. Adding the newly posted direction gave coefficient 0.024: the
   new request is not an instantaneous paddle velocity. The old motor also
   counter-steered at a clipped side boundary, despite there being no further
   out-of-bounds paddle motion to cancel.
2. **More than one quantization period.** Fast observations include dy
   `3,3,4`; a two-sample mean still alternates 3 and 3.5. Slow center-section
   flights include dx `0,-1,0,-1`. Treating zero as a sign reversal recreates a
   large landing flip: in the contact-band probe seed 5, frames 527–530 alternate
   `[120,132]`, `[107,119]`, `[119,131]`, `[108,120]`, preceding a miss at 545.
3. **Collision and measurement timing.** Successful last descending samples
   mostly have ball y 186–188, later than the old y=185 leading-edge prediction.
   Some horizontal velocities change again just after the vertical rebound.
   The first post-bounce dx alone is not a reliable endpoint calibration.

Finally, safer centering exposes a scoring limitation: a cadence-aware seed-1
probe reaches **428 with three lives** but remains there until its 480-second
try deadline. A motor-only seed-5 probe also stops at 428 with one brick left
and two lives at 300 seconds. These observations justify a bounded, measured
remaining-brick objective rather than waiting indefinitely or inventing an
unverified generic bounce-angle table.

## Final code architecture

### 1. Frame-level state and interception

* Legal action names are used throughout. FIRE is posted while the observation
  requires a serve, including replacement balls. A missing ball is not a reset.
* The loop posts an action and calls `ctx.wait_for_tick()`. All inference is
  asynchronous; it inspects only completed futures. No private emulator state,
  game source, alternate stepping, seed identifier, or held-out result is used.
* The 0015 filtered/receding geometry is extended to a **six-sample component
  window**, covering the observed periods two and three. Zero displacement can
  be a cadence sample; a genuine nonzero sign reversal or large jump clears that
  component's history immediately. Histories are episode-local and broken on
  missing balls or skipped observations.
* Code projects the descending ball below the brick guard with side reflection.
  It uses a contact plane at paddle-y minus one (188 for the observed geometry),
  and three pixels of interior overlap margin. These are calibrated approximations,
  **not a claim to know the cartridge's exact collision routine**. A catch
  interval is recomputed every descending frame. Uncertain/ascending flight uses
  the original broad home band.

### 2. An observed-input predictor replaces unverified braking

The original interceptor remains explicit in the base class; the final motor
is an override. Code estimates the first-order motion coefficient and input gain
again within each episode using exponentially weighted sufficient statistics.
Defaults 0.60/2.40 come from the public-observation fit above, not a training-seed
lookup or game implementation. Estimates are accepted only in bounded ranges
and away from clipped paddle positions.

For NOOP/LEFT/RIGHT, the motor rolls out a short pulse followed by release,
including both the queued input's sticky alternative and the new request's
sticky alternative. It infers the last consumed direction from actual paddle
motion; the requested-action label is not treated as ground truth for ALE's
internal stickiness. Future release input decays under the known 0.25 repeat
probability. Predictions clip at legal paddle bounds. Leaving the catch interval
has a much larger cost than the weak interior-target preference. This is a small
approximate predictive controller, not exact clone-state search.

### 3. One flight, one objective, one observed resolution

`contact_id` changes on a bounce, serve, loss, or wall transition. Brick hits,
side reflections, cadence samples, and geometric replans do **not** consume new
trials. A selected endpoint remains associated with the same contact until a
bounce or loss resolves it. One safe contact separates trials.

Ordinary tunnel trials require at least four lives, two prior catches, a warm
normal-speed estimate, a wide early descending catch interval, and a quota of
three trials per life. Early positioning uses a small interior band. Near impact
the **full catch interval** remains the hard safety objective, with only a soft
endpoint preference. An accepted SURVIVE objective can veto the trial's actuation.

The harness records the actual pre-bounce offset and then waits for several
ascending samples before estimating outgoing motion. It only learns a mapping
from stable-paddle, interior-offset contacts sufficiently far from a side wall.
A requested endpoint which was never reached cannot confirm its mapping. Two
contrary observed endpoint outcomes are needed before switching the default
mapping. Fast-regime evidence is tracked separately.

### 4. Sparse-wall adaptation

After **420 frames without reward**, with at most **12 bricks** remaining:

* With **three or more lives**, a wide interval and an observed fast regime,
  at most **four endpoint-calibration trials per wall** may gather missing
  fast-regime evidence. These are explicitly unverified experiments, not claims
  that a numerical angle has been learned.
* With **two or more lives** and sufficient measured endpoint evidence, code
  projects the measured outgoing speed in each supported direction toward the
  remaining brick rectangles. A reflected upward ray ranks direct intersections
  and nearby misses using the actual row values. This is a candidate generator,
  not an exact simulation through future brick collisions.
* The current speed must match the observed fast regime. A newly served slow
  ball cannot reuse the fast ray model. At one life, all deliberate shaping is
  disabled; the predictive catch controller continues until the actual end.

No stage or action is selected by seed or by a memorized frame schedule.

## Jev state and questions

Every request is strict JSON with explicit game/rule context and an
`observation` object containing:

```text
frame, event, lives, score, wall, bricks_remaining,
phase, urgency, filtered speed,
shape_ready, next_contact_eligible, objective_kind,
safe_catches_since_serve, candidate_side, cleanup_candidate,
left_edge, right_edge, current_objective, astra_proposal,
trial_outcome, endpoint_evidence, contact_id, plan_revision
```

`objective_kind` distinguishes tunnel, endpoint calibration, and verified
cleanup. Counts, velocity, eligibility, ray comparisons, legality and coordinates
are computed in code. Jev receives named evidence/buckets rather than a raw grid
or a request to do arithmetic.

One request batches **two reordered Choice questions** over:

* **SURVIVE:** keep the ordinary catch objective for the next contact;
* **LEFT / RIGHT:** approve that future interior-contact objective, using the
  code candidate and Astra prior; an open edge/fast ball is allowed only for
  the explicit calibration/cleanup stages;
* **NO_CHANGE:** leave the code's gated choice or preservation fallback in place.

The wording distinguishes eligibility for the **next** contact from whether the
ball can be shaped at this instant. It also distinguishes a historical miss
followed by safe catches from an unresolved current failure.

### Answer bridge and freshness

Both values must agree and each selected-option probability must be at least
**0.60**. Missing probabilities mean uncertainty, not certainty; confidence None
is never coerced to certainty. Answers must match the current life/wall epoch,
contact ID and plan revision, and be no more than **90 frames** old. The current
life, speed and stage gates are rechecked. A cleanup-side override additionally
needs measured support for its endpoint. A side choice changes a preferred
interior target, never emits a raw game action. SURVIVE is a one-contact veto.

Requests are boundary-triggered or triggered by a newly eligible descent, with
one outstanding call, a 42-frame cooldown and a **2-second timeout**. Late,
conflicting, malformed or failed answers leave the executable catch controller
running. The call budget is checked before submission and a small reserve is
kept; no background polling loop survives `ctx.done`.

`jev_contact_accepted` records semantic acceptance. More importantly,
`contact_objective_changed_action` compares the chosen motor action against the
same predictor's unshaped action, recording the objective and its source label. The `astra_prior` label records
that a side prior was present; in cleanup mode the ray ranker can still choose
the target, so this label is **not a counterfactual attribution to Astra**.
The unshaped-action comparison establishes contact-objective influence, not by
itself a causal benefit from either model. Model-call counts are not presented
as proof of useful cooperation.

## Astra's role and revision trigger

Astra is asked for exactly LEFT, RIGHT or SURVIVE, not motor code or numeric
angles. It receives the same explicit derived state and the applicable tunnel,
calibration or cleanup semantics. The **72-second** asynchronous timeout inherits
0016's working plumbing; 0015's 12-second timeout was refuted by the archive.

It is called once at the initial stage of each wall and once at the first
observed sparse-wall stall, with at most **four calls per episode**. Returning
advice is checked against the current wall, strictly parsed, stored as a prior,
and included in the next Jev request. A changed plan increments the version so
an outstanding old Jev judgment cannot authorize it. Code safety gates apply
regardless of which model proposed the objective. Arbitrary prose, ambiguous
multi-token output, timeout and transient failure are ignored.

This is the runtime revision hypothesis: **ask Astra again when observed scoring
progress stalls despite successful catches**, not whenever RAM alternates a
velocity sample. The finite Choice interface is deliberately fixed here; Astra
is not asked to generate new Jev questions, so no unavailable conversation context
or implicit question-generation contract is assumed.

## Falsification and next iteration

Confirmation requires completed achieved contacts, no quota churn, comparable
survival, and improvement over 420.167 with additional wall progress. It is not
enough to increase S1/S2 calls or to log trial starts.

Refuting evidence includes persistent misses within the logged interval, poor
out-of-sample actuator prediction, inconsistent measured endpoint directions,
unreached calibration offsets, or less progress after approved cleanup. Such
results should narrow the collision/input model or remove shaping, not grant a
language model more authority over individual frames. A lingering one-brick
orbit despite verified rays would justify adding ceiling-return paths and
freshness-aware rejudgment, not memorizing a seed's winning actions.

## Validation

Synthetic contract tests in `diagnostics/test_contract.py` pass with arbitrary
answer objects, None instead of an answer mapping, model timeouts, missing
confidence, valid typed choices, and arbitrary Astra prose. They run a complete
synthetic episode and assert legal action names. This is a contract test, not
an ALE score benchmark. Compilation also passes.

Development probes, **not interchangeable evaluations of the final file**:

| Revision / probe | Seed | Score | End / useful observation |
|---|---:|---:|---|
| Initial crossover | 5 | 357 | Natural end; first resolved endpoint trial |
| Initial crossover | 1 | 434 | Natural end |
| Delayed motor | 5 | 428 | 300-second try deadline, two lives left |
| Delayed motor | 1 | 409 | Natural end |
| Delayed motor, Astra live | 2 | 414 | Natural end; Astra returned in 9.44 s |
| Six-sample cadence | 5 | 618 | Natural end; meaningful second-wall progress |
| Six-sample cadence | 3 | 418 | Natural end |
| Six-sample cadence | 1 | 428 | 480-second try deadline, three lives left |
| Zero-tolerant / queued-stickiness revision | 5 | 770 | Natural end |

All completed probes above have `error=null`. Most used Jev live and Astra off;
off-mode Astra attempts may log the helper's normal RuntimeError and do not
establish a backend failure. The Astra-enabled probe explicitly demonstrated a
successful parsed prior. Real-time scheduling/model choices make repeated seeds
non-identical; the scores do not isolate a causal contribution from Astra or Jev.

Final-file training-seed tries used the full 900-second limit; their results
are below. The official selection evaluation is still separate and unknown.

### Frozen final-file results

The final submitted module is frozen at SHA-256
`b9a44ef6b23b21f14caf9965cbb8208fbecbd592c80f5308a6517b2f64590302`.
The complete local six-seed sweep was:

| Seed | Score | End | Astra enabled |
|---:|---:|---|---|
| 0 | 534 | Natural game end | No |
| 1 | 860 | Natural game end | No |
| 2 | 687 | Natural game end | Yes |
| 3 | 828 | Natural game end | No |
| 4 | 774 | Natural game end | No |
| 5 | 431 | 900-second deadline, two lives left | No |

**Local mean: 685.667; errors: zero; stale fraction: 0.00006856
(0.006856%, 11 of 160,445 ticks).** Five of six runs cleared the first wall.
None of this first sweep cleared both walls. Because most of these tries had
Astra off, this is not the official selection mean or a controlled comparison
against the parents' fully configured evaluations. The frozen file's additional
Astra-enabled confirmations scored **846 on seed 0** and **863 on seed 1**, also
without errors. The separate seed-5 Astra confirmation also completed without
error at **864**, with `both_walls_cleared=1`. All additional confirmations used
the same frozen module and both live models.

Machine-readable results, model latencies and exact trace locations are in
`diagnostics/validation.json`, refreshable with `diagnostics/report_validation.py`.
The weak primary seed-5 episode stayed at 431 with one brick left: its eventual
flight was dx=-2, dy=-2, outside the separately learned 3.3-pixel vertical-speed
cleanup regime, and two lives did not meet the unverified-calibration gate.
That is a concrete conservative-gate limitation, not evidence that the last
brick is unbreakable. Extending the per-regime calibration safely is left for
a later experiment rather than silently reusing a mismatched fast-speed table.

These traces provide more specific evidence than score alone:

* Final seed 0 has twelve resolved contact trials, eight confirmed, and 21
  action differences attributed to a Jev-selected objective. Seed 4 has fifteen
  confirmed trials and 26 such Jev-objective action differences. This establishes
  an executable bridge, not that Jev improved score relative to an ablation.
* Seed 2 made three successful Astra calls, all returning PRESERVE, with a
  10.37-second median and 11.31-second maximum. It reached 687, but this does not
  establish a causal scoring benefit from Astra. Its cleanup choices were
  code-generated and safety-gated.
* The 860-point run still missed with one four-point brick remaining. At frame
  25182 its paddle x was 124 against a projected safe interval approximately
  `[102.2,114.2]`; LEFT was already requested but existing motion continued.
  The 828-point run similarly has an unshaped final miss after sustained opposite
  motion. Rare sticky/queued actuator excursions, rather than an absent model
  answer, remain a concrete next target.

Additional unit regressions pass for zero/nonzero velocity cadence, immediate
sign reversal, avoiding a counter-steer away from a clipped wall, rejecting an
unsupported cleanup direction, and invalidating the fast cleanup model on a
slow serve. These supplement, rather than replace, the evaluator's pre-flight.

### Completed both-model confirmations

| Training seed | Score | End | Error |
|---:|---:|---|---|
| 0 | 846 | Natural game end | None |
| 1 | 863 | Natural game end | None |
| 5 | **864** | Natural game end | None |

Together with the primary seed-2 run (687, Astra enabled), these verify the
actual two-model runtime path on four training seeds. They are repeated local
tries, **not a matched ablation or a replacement six-seed selection mean**.
The seed-5 confirmation cleared both walls; its trace contains 19 confirmed
contact trials, and Astra returned PRESERVE, RIGHT, PRESERVE, LEFT over four
stage calls. This shows the planned/advisory pipeline executing through wall
transitions and sparse-wall replanning without a contract error. It does not
prove those tokens caused the 864 rather than timing and controller effects.

The final controller was not edited between these confirmations and the frozen
six-seed sweep. Compilation, import checks, garbage-output tests and all focused
regressions still pass. No other candidate directory was modified.
