# Candidate 0018 — contact evidence, identified actuator lag, and model audits

## Parent and concrete diagnosis

Parent: **0015**, the best evaluated candidate (mean 420.167). I retain its
legal FIRE/episode handling, reflected interception, event bookkeeping, broad
ascending/home interval, and code-owned native-frame action loop. I replace
its descending motor and its model division of labour. The inherited
`IntervalFallback` contains the parent geometry/home controller; `Harness`
implements the new contact tracker, measured dynamics, search, and model bridge.

The selected worst parent trace does not support asking for more frame actions:

* In `0015/trace_seed5.jsonl`, frames **6998–7009** all have fresh NOOP actions.
  The landing contracts are around 114–115, with paddle-left overlap intervals
  around 98–115, yet the life is lost at **7017**. That episode accepted **79
  PRESERVE** Jev contracts and no other contract.
* Seed 2 similarly posts mostly fresh NOOP around **5900–5910** and loses at
  **5919**. Other loss windows contain late LEFT/RIGHT reversals. Freshness is
  necessary, but is not a catch certificate.
* All six parent Astra calls time out. Its **12-second** timeout is below most
  measured archive latencies, so its strategic layer supplied no control evidence.
* The parent's diagnostic `tries/seed3-160842` shows a successful ball at
  **y=186** at frame 191, followed by upward motion at 192. The nominal
  rectangle-overlap plane y=185 is not itself the observed reversal.

**Initial hypothesis:** the broad decoded rectangle is an insufficiently
validated catch set, and the motor is using the wrong dynamics. Aim at an
interior band, measure actual crossings, and let models review accumulated
contradictions instead of repeatedly choosing a home anchor.

**Refinement supported by the local probes:** the clearest remaining failures
were moving-paddle crossings, not proven stationary collision-mask errors.
NOOP could be followed by another 4–5 pixels of motion. A velocity-only motor
fit also confused delayed starts with an actuator that could never move.
Including the *previous posted command* substantially improved the dynamics
fit. Two-frame ball averaging still left a noisy landing for the narrower
band, so the final estimator uses an elapsed-displacement window within a
clean flight. This is an evidence-driven controller/bridge rewrite, not a
claim that the ROM's paddle definitely shrinks or that its exact mask is known.

## What code now owns

1. **Clean-flight tracking.** Up to nine consecutive public position samples
   provide displacement over at most eight elapsed frames. Serve, loss,
   brick/paddle/side events, sign reversals, or observation gaps clear the
   relevant history. The parent's short estimate is the startup fallback.
   Code still recomputes the reflected landing each descending frame after
   the ball leaves the brick guard; it does not repeat 0017's final-target lock.
2. **Observed crossing plane and interior band.** The target plane is paddle
   y minus three (186 in the observed baseline). CORE uses ball-center minus
   paddle-left offsets **2–8**, rather than accepting every edge overlap of a
   reported 16-pixel rectangle. This is a conservative experimental band, not
   a newly asserted ROM rule. Physical paddle bounds are **8–144**, supported
   by the live observations. Reflection initially retains the parent's
   8–150 ball bounds; their discretization remains a possible source of error.
3. **Contact evidence.** A catch requires an observed downward-to-upward
   transition near the paddle. A disappearance records a failed return.
   The tracker chooses the observed sample nearest the crossing plane and
   logs the preceding window, offset, paddle displacement, and landing
   residual. It never equates a planned overlap with a successful catch.
4. **Identified actuator lag.** Interior observations update a ridge-regularized
   three-variable regression:

   `next_paddle_delta = a * current_delta + b * posted_direction + c * previous_direction`

   Directions are signed code features, not raw ALE action enums. A fit to
   the development traces gave roughly **0.65, 0.3, 2.2**; these initialize a
   cold per-episode fit. The fit is recomputed every 120 usable samples and
   accepted only within bounded coefficients. Saturated-edge observations
   do not teach spurious motor gains. All learning uses public observations
   and this harness's own posted actions, not emulator internals.
5. **Short sequence search.** Code evaluates up to `3**4` sequences of
   NOOP/LEFT/RIGHT under nominal and delayed-first-input branches. It scores
   catch-band distance, terminal motion, and the still-pending command's
   predicted neutral continuation. Only the first action is posted, then the
   next observation replans. Reaching a band at five pixels/frame is penalized;
   a quiet position inside the band is not pulled needlessly to its midpoint.
   A far-outside reachability guard prevents a learned delayed startup from
   becoming a permanent NOOP decision. Search and simulation respect observed
   travel limits.
6. **Lifecycle and timing.** FIRE is repeated only while serving; a missing
   ball is not a reset. The loop posts a legal action before launching new
   inference and waits with `ctx.wait_for_tick()`. It plays until the game or
   context ends, including after a wall clears or the score reaches 864.

This is not an unqualified endorsement of MPC: 0012 already tried a short
search and scored 313.333. The tested additions here are **measured command
lag, valid travel limits, settled interior contacts, and evidence-based model
interventions**. The unsuccessful local velocity-only search is recorded below.

## New model division of labour

### Astra: revise the evidence audit, not a live target

After five measured contacts, one asynchronous review receives the current
policy, precomputed diagnostic facts, and recent contact records. Later reviews
are requested after a diagnosed contradiction, with at least 1,800 frames
between starts and at most three starts per episode. Astra returns exactly:

* `VERIFY_COVERAGE`: scrutinize overlap claims; require eight recent reliable
  stationary catches before authorizing an empirical band;
* `VERIFY_MOTOR`: prioritize the measured moving-paddle failure evidence;
* `VERIFY_PROJECTION`: prioritize landing discrepancies;
* `USE_MEASURED`: prefer the evidence-supported band when Jev and code agree.

All tokens enter the next Jev state. `VERIFY_COVERAGE` also raises the
code-computed repeated-success requirement from six to eight. Other tokens
retain six; none can waive a recent miss or invent numeric geometry. A valid
review increments the plan version and requests a fresh audit. The fixed
questions are not generated by Astra, so the runtime prompt does not ask it to
use an unavailable question-construction API.

The timeout is **150 seconds**, not the parent's 12 seconds. A new call needs
more than 150 seconds of wall budget remaining and a reserve of System-2 calls.
A slow physics review can survive a life change, because all current safety
conditions are rechecked; a wall/geometry epoch change invalidates it. Garbage,
extra prose, errors, or expired reviews are ignored. ModelUnavailable still
ends the context through the normal context contract.

### Jev: adjudicate completed-contact evidence

Every request is a small strict-JSON object:

```text
{
  game,
  policy: {astra_review, active_certificate, fallback},
  evidence: {
    coverage_contradiction, motor_motion_at_miss,
    projection_error_at_miss, repeatable_stationary_successes,
    no_recent_misses, progress_stalled
  },
  episodes_are_not_reset_on_life_loss,
  recent_contact_outcomes: [up to six records],
  candidates: {CORE, MEASURED, UNKNOWN},
  review_meanings: {COVERAGE, MOTOR, PROJECTION, UNKNOWN}
}
```

Each contact record includes outcome, crossing offset/y, maximum recent paddle
motion, prediction residual, precomputed support predicates, frame, life, and
wall. Code computes all comparisons: stationary means recent motion at most
one pixel/frame; supported prediction means crossing residual at most three
pixels. The records are evidence for interpreting the flags, not a request to
calculate coordinates or count bricks.

One request batches four independent questions:

1. `certificate_a`: choose CORE, MEASURED, or UNKNOWN.
2. `certificate_b`: the same choice with reversed option order.
3. `cause`: classify COVERAGE, MOTOR, PROJECTION, or UNKNOWN from the named
   evidence flags, with explicit inclusion conditions.
4. `needs_review`: a Noul for a coverage contradiction or projection error.

The paired Choices must agree and each selected-option probability must be
**at least 0.65**. Missing probability maps are uncertainty, not certainty;
`.confidence=None` is never converted to certainty. MEASURED also requires
six recent stationary, prediction-supported catches (eight under
VERIFY_COVERAGE), with no miss among the last twelve outcomes. It substitutes
an interior quantile band of successful stationary offsets from the recent
history, capped inside the reported paddle span. It can change the motor's
admissible position interval; it is not an unused side label. Unsupported or
ambiguous evidence cannot authorize a new measured band.

A cause probability at least 0.65 requests a later Astra review. A Noul value
at least 0.75 can also request a review, but only if code's corresponding
contradiction flag is still true. Neither diagnostic directly posts an action.

S1 timeout is **3 seconds**; requests are asynchronous, at most one at a time,
with a 90-frame scheduling gap. An answer expires after **240 frames** or a
life/wall/plan-version change. Current evidence is rechecked on arrival. A
late or failed answer retains the current executable certificate; every miss
or life/wall change immediately restores CORE without waiting for a model.
There is a reserve of four S1 calls, and no background Python polling loop.

### Observable cooperation

`certificate_audit` records old/new certificate and whether it changed;
`failure_classified` records diagnostic attribution; `astra_review_installed`
records the accepted token/version. `motor_identified`, `contact_evidence`, and
`capture_control` expose the numeric evidence behind actual action selection.
The parent geometric-envelope logs are not the revised band: `capture_control`
contains the final band used by the new motor.

## Local experiments and limitations

Only training seeds and the supplied `metaharness try` interface were used.
There was no game-source inspection, private state access, held-out evaluation,
seed-conditioned logic, model-backend bypass, or persistent cross-episode fit.
All harness imports are standard library or `metaharness.*`.

Early probes were genuinely informative failures:

* The initial interior-band/one-step controller scored **404 on seed 5**,
  versus the parent's archived 382. Its misses still involved motion; this
  did **not** establish a collision-mask error.
* A velocity-only empirical search scored **6 on seed 5 and 7 on seed 2**.
  It learned zero-displacement starts as absorbing actions and stopped moving
  toward reachable targets. This reproduces the lesson from 0010: a nominal
  interval and fresh NOOPs are not sufficient.
* Guarding those starts recovered **415 on seed 5**. The command-lag fit then
  scored **453 on seed 2 / 394 on seed 5**. Correcting travel bounds and
  removing midpoint micro-adjustments scored **432 on seed 0 / 376 on seed 5**.
  These are development probes, not a selected best-of-seed aggregate.
* The final longer clean-flight estimator and full evidence bridge scored
  **848 on seed 3** with both models live, no episode error, and only one stale
  tick in 17,242 ticks. All three Astra calls succeeded (about 5.8–9.4 seconds).
  Jev made real CORE-to-MEASURED changes, including frames 875 and 5295.
* A final-controller seed-1 probe with Astra disabled scored **428**, also
  error-free. Live two-model validation results are recorded below when complete.

`contract_check.py` is a separate synthetic API smoke test, not a game-score
benchmark. It exercised 1,000 ticks with arbitrary model answers, missing
probability maps, a deliberate timeout, and garbage Astra text; all actions
remained legal and the loop returned normally. The official evaluator's
pre-flight and six 900-second evaluations remain authoritative.

## Confirmation and refutation

Confirmation is higher multi-seed score, more first-wall completions and
meaningful second-wall progress, with comparable freshness and no errors.
Contact logs should show predictions stabilizing and the paddle settling
before contact; model logs should demonstrate actual certificate changes.

The live probes support the combined design but are **not an ablation** that
separates the inner band, lag fit, flight window, and model advice. Remaining
misses often still occur during movement, including occasional overlaps in a
single sampled frame that disappear before the true collision. Persistent
stationary misses inside the measured band would refute its adequacy and
justify a collision-order/mask experiment. If model-enabled and fallback-only
runs match, the numerical controller—not the audits—deserves the credit. If
cleanup stalls at a few bricks, a later iteration should learn contact-angle
outcomes rather than restore generic LEFT/RIGHT home-anchor advice.

## Final-controller live validation

These are six local training probes with **both models live**, not formal
selection results. The cap was eight minutes (seven for seed 3), but **every
run ended naturally before its cap**. The final module differs from these
runs only in its explanatory module docstring; the controller and prompts
were frozen throughout these six validations.

| Seed | Score | Parent 0015 score | Frames | S1 / S2 calls | End |
|---|---:|---:|---:|---:|---|
| 0 | 797 | 421 | 19,140 | 37 / 3 | game_done |
| 1 | 853 | 464 | 28,405 | 47 / 3 | game_done |
| 2 | 706 | 418 | 16,613 | 33 / 3 | game_done |
| 3 | 848 | 410 | 17,242 | 35 / 3 | game_done |
| 4 | 864 | 426 | 27,789 | 45 / 3 | game_done |
| 5 | 828 | 382 | 22,106 | 39 / 3 | game_done |
| **Mean** | **816.000** | **420.167** | | | |

All six runs had **zero episode errors and zero S1/S2 call failures**. Each
had exactly one stale tick: **6 / 131,295 total ticks**, approximately
**0.0046%**. The unmodified ALE score matched the decoded cartridge score in
every final observation. Every seed cleared the first wall; seed 4 cleared
both walls and reached the manual maximum of 864. The final second-wall
brick counts were 19, 2, 35, 7, 0, and 9 respectively.

The six final probes used 236 S1 calls and 18 S2 calls in total. Model advice
was not merely logged: the traces contain repeated CORE-to-MEASURED
certificate changes and installed Astra evidence-review tokens. The five
runs inspected before seed 1 finished already contained 25 such certificate
changes and 15 successful Astra calls, with latencies about 5.6–28.9 seconds.
This does not isolate the causal contribution of either model; a same-code
model-off ablation would be needed for that claim.

`local_validation.json` records the final module hash, per-seed provenance,
raw score cross-checks, timings, call failures, and trace paths. The official
pre-flight/evaluation is still the authority for eligibility and selection.
