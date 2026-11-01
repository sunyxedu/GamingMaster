# Candidate 0026 — native-frame wall prediction and evidence-scoped advice

## Parent, retained evidence, and hypothesis

This rewrites **0010**. Its useful property is the code-owned asynchronous
frame loop, legal serving, and explicit event bookkeeping—not its motor.
Its seed-0 trace posts fresh NOOP at ticks 315–339 despite a live return;
it finishes at 5 points after 1,063 ticks. The archive already refuted the
claim that a latched interval, by itself, preserves a catch.

The strongest evaluated reference, **0022 (763.333)**, supplies the measured
queued/sticky paddle motor, six-sample cadence filtering, impact controller,
and one-contact trial bookkeeping. Those mechanisms remain intact. This is
not another lease-length, braking-gain, or terminal-horizon experiment.

**Hypothesis:** an important residual error is treating a discrete Atari
side-wall transition as continuous reflection. In 0022 seed 1's final miss,
frames 13046–13049 show ball x `148,150,152,150`. Its code reflects at x=150.
The impact interval initially lies around `[124,136]`, then changes toward
`[128,140]` after the observed turn, while its lock leaves the stationary
paddle at x=123. The miss follows at 13057. The corresponding seed-3 trace
contains 92 observed `152 -> 150` turns, 50 `151 -> 149` turns, and 26
`7 -> 9` turns in its contact telemetry. These are native-step overshoots,
not noisy evidence for a continuously reflecting boundary.

Other misses remain actuator failures: seed-1 frame 7221 follows a locked
LEFT pulse and x `47,48,46,42,38,33,27`. Correcting geometry cannot be claimed
to fix that. Retaining the strongest motor isolates the new prediction rather
than repeating 0021/0023's refuted reservation/horizon changes.

## What changes

1. **Native wall projection.** For observed, stable integer horizontal steps,
   code advances native frames and reverses an outward trajectory only after
   observing the threshold has been crossed. It preserves fractional final
   contact time. A trajectory visiting x=152 is not folded prematurely to 148.
   Noninteger, cold, and ambiguous cadence retains the reference predictor.
   Only a changed predicted landing changes the catch interval; the physical
   margin, contact plane and motor are unchanged.
2. **Shadow identification.** At clean descending boundary samples code compares
   native and continuous one-step predictions with the actual next ball x.
   Three consecutive substantial contradictions suspend the new predictor.
   This law is inferred from public observations, not game implementation,
   hidden RAM, clone state, or scenario identifiers.
3. **Different model responsibilities.** Jev no longer selects a brake profile
   on every bounce. Code owns motion and numerical tests. Jev reviews a compact
   evidence verdict and whether a sparse-wall stall warrants the existing
   measured cleanup or bounded calibration. The final code also labels the
   slow diagonal (`|dx|≈|dy|≈2`) regime separately: a `PROBE` answer can permit
   at most four interior calibration contacts with two lives and a long stall;
   fast-regime endpoint evidence is never reused for this orbit. Astra is
   consulted only when that scoring bottleneck exists, not at startup for a
   reflex-control token.
4. **Separate lifetimes.** The frame-level geometric estimate, contact trial,
   life/wall-scoped evidence review and slow strategic prior no longer share a
   single expiration rule. Endpoint experiments still resolve only at actual
   contacts; a requested endpoint that was not reached cannot calibrate a ray.

## Jev schema, questions, and action bridge

Each request supplies explicit task/rule meanings and:

- event, frame, wall, lives, named phase/urgency;
- `progress`: scoring / stalled / long_stall;
- `wall_density`: dense / sparse;
- `geometry_evidence`: insufficient / corroborated / contradicted / mixed;
- native predictor status, `speed_regime`, endpoint evidence (specific to that
  regime), code-computed calibration and cleanup eligibility, recent catch
  count bucket and last trial outcome;
- current policy, Astra prior, candidate side;
- small numerical shadow-error diagnostics, already classified in code.

One request batches three independent Choices:

* Two reordered **policy** choices: MEASURED / PROBE / PRESERVE / NO_CHANGE.
  MEASURED retains supported cleanup; PROBE permits only the already bounded,
  life-gated calibration; PRESERVE defers those experiments for 360 frames.
  The side/ray calculation remains code-owned. No answer can force a frame
  direction, change the motor, or shrink the physical catch interval.
* A **geometry** choice: ACCEPT / REJECT / UNKNOWN. ACCEPT can activate the
  native predictor only when code also classifies the measurements as
  corroborated; REJECT can deactivate it only with observed contradiction.

Both policy answers must agree with selected-option probability >=0.65.
Geometry needs >=0.80 and the independent numerical gate. Missing probability
is uncertainty, never certainty; confidence None is not promoted. Answers
must match the life/wall epoch and strategic revision and be <=180 frames old.
Current eligibility is rechecked. One request is outstanding, with a 180-frame
cooldown and 2-second timeout. New wall evidence or a stalled sparse-wall
contact triggers requests; routine successful bounces do not.

Astra returns one strictly parsed token, MEASURED / PROBE / PRESERVE, at the
first sparse-wall stall and, at most, when missing endpoint evidence becomes
measured, once per stage per wall and at most four calls per episode. Its
90-second asynchronous timeout covers archived latencies. A stale life/wall
answer or arbitrary prose is ignored. It supplies a prior for the next Jev
review, not a raw command. Revising the numerical interface is deliberately
outside runtime Astra's authority in this final-generation experiment.

## Fallback and eligibility

The reference catch controller runs immediately and posts every frame, using
`ctx.wait_for_tick()`. FIRE serves initial/replacement balls; no reset occurs.
A late, malformed, failed or unavailable model leaves deterministic control
running. Spawned work is one bounded inference call, not a polling thread.
Only normal termination/deadline ends the loop. No outside imports, direct
backends, emulator access, seed-dependent policies or held-out information
are used.

## Confirmation and refutation

Confirmation requires lower shadow boundary error, corrected landing intervals
before—not after—the side turn, fewer stationary wrong-lock misses, and
comparable survival/score with better wall completion. Events
`native_wall_check`, `native_landing`, `native_wall_suspended`,
`jev_geometry_review`, and `jev_experiment_policy` supplement the inherited
contact/miss telemetry.

Refutation includes observed boundary predictions worse than continuous
reflection, misses despite accurate stationary overlap, or score regression
from experiment deferral. Model call counts are not evidence of causal
benefit. Live tries and contract tests are supporting diagnostics, not
substitutes for the evaluator's six-seed selection.

## Supporting checks

`diagnostics/replay_geometry.py` replays only archived public
`contact_tick` observations. Across the six 0022 traces it found 910 changed
native/continuous predictions: native-step mean absolute error was lower in
all six seeds (0.0006–0.243 pixels versus 2.13–4.09 pixels for continuous
reflection), with 910/910 changed cases better and none worse. These samples
are correlated and are not a score counterfactual.

`diagnostics/test_contract.py` passes eight tests, including discrete
threshold projection, fractional/cold fallback, garbage and timeout model
answers, stale-answer rejection, and the safety rule that three recent
contradictions cannot be overruled by stale consensus. Development probes of
the native-wall version reached scores 503, 847, 859, 864, 807 and 853 on
seeds 0–5 (the first seed-0 both-model run was 608 with no error); these
wall-clock probes predate the final regime-specific PROBE gate and are not the
official mean. The score variance is why that gate, the runtime geometry
guard, and the catch-first fallback remain conservative.
