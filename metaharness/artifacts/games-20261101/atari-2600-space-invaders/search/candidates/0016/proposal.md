# 0016 — review temporal evidence, not target-order labels

Parent: **0008**, structural rewrite. Comparison/control inspiration: **0012** (the current best), plus the published lessons for 0006/0010/0014. No other candidate is modified.

## Hypothesis

The missing capability is not another target-order label or a faster actuator. The object decoder is an imperfect sensor. The parent treats its player-shot identity, invader identity, and disappearance events as facts, while the best candidate treats unexplained shield erosion as an uncertain enemy lane. Neither has established an incoming projectile before many life losses.

**Test:** separate sensor identity from temporal evidence. Track every public projectile position from both `player_shot` and `bombs` in one kinematic ledger. Associate points using bounded motion, not their decoder label; distinguish upward own-shot tracks, downward tracks, new launches, and identity discontinuities. Give Jev the unresolved evidence-association decisions instead of target ordering. Code performs collision prediction and posts full-rate combined FIRE throughout. Astra chooses a small evidence rubric early, not a late route or long prose plan.

This is a falsifiable sensor-repair hypothesis, **not** a claim that the archive proves that bombs appear in `player_shot`. The archived score-change observations are too sparse to establish that. The live per-tick ledger will measure it. If no such tracks occur, this mechanism has failed to find the hidden threats, even if the sweep scores well.

## Concrete archive evidence

- **0008, worst seed 6:** `trace_seed6.jsonl` has 289 ticks: **26 fire, 185 right, 78 left**, no combined FIRE. Its two `life_recovery` events are at **frames 576 and 856**. The final controller event still reports **left_limit=32, right_limit=147**, and its score is **10**. All four logged observation events have empty bombs. The target-policy answers did not compensate for low firing throughput or missing threat evidence.
- **0008's calibration is not reliable hit identification:** lines 96 and 108 of that trace report `shot_lane_calibrated` changing the offset **0→1→−0.5**; the second calibration occurs while score remains **10**. A disappearance without a corresponding score cannot safely teach a laser hit offset.
- **0012, worst seed 7:** **529 ticks, 305 right_fire / 224 left_fire**, score **190**. Its `shield_escape_applied` events occur at **frames 212, 292, 532**. `life_loss_signal` still arrives at **1128 (previous_x=86)** and **1744 (previous_x=88)**, both with `recent_bomb=false`. Its final event reports **jev_changed_actions=0, astra_effective=0**.
- **Public identities demonstrably alias:** in 0012 seed 0, the observation at **frame 964**, still wave 1, removes all six row-1 identities while five previously dead identities reappear; the score increases only **10** from the preceding sampled observation. That is not evidence of seven ordinary kills plus five new invaders. The new harness logs such revivals but does not use them for shot calibration, target lock, or death/boundary learning.
- **0012's best seed 0** still has blind life signals at **1532/2092** and finishes with zero lives. Its score **620** includes a **200-point UFO**, rather than proving a wave clear. The comparison must separate score from incoming-threat recognition and survival.

The generation-3/4 lessons already reject stalled-route/boundary repairs and unchanged semantic policy labels as demonstrated causes. This candidate does not retune those mechanisms or resurrect the reset-position death-lane mistake.

## Structural changes and division of labor

### Code

- Preserve the supported mechanism: continuous combined FIRE and a bounded broad sweep, refreshed once per tick. The ROM enforces the single-shot lock; decoded shot activity never suppresses FIRE.
- Replace target locking and aim-offset calibration with a **label-independent projectile ledger**. Match observations only within small horizontal and physically plausible vertical displacements. A new muzzle-level beam cannot attach to an old upper beam. Direction reversals invalidate cached judgments.
- Infer velocity from observed differences. An explicit bomb or two consecutive downward increments is a code-owned threat; an ascending track is not. Retain confirmed tracks for at most 24 emulator frames across decoder gaps.
- Predict the paths of left/fire/right actions under both immediate switching and one-tick sticky delay. Recompute on the latest observation; never execute coordinates from a model answer.
- Preserve only a bounded version of 0012's weak shield cue: small top-entry erosion, not explained by an outgoing launch, may cause one short detour when cover is thin. Bulk shield clearing is discarded. A chip is **not** extrapolated as a surviving bomb.
- Record control gaps and the last mobile frame separately from the late life counter. Do not interpret no-motion as a boundary or the respawn position as an impact lane.

### Jev

Jev is an **evidence reviewer**, not a target-order adviser. A request contains up to three unresolved cues. It can promote supported ambiguous incoming evidence, suppress a friendly-shot explanation, or leave the cue unresolved. This modifies which threats/cues enter the live collision and detour checks, rather than merely logging a preferred policy label.

### Astra

Make one short, asynchronous request immediately after the first action, so reasoning begins before a life is lost. Ask for exactly one token:

- `CORROBORATE` (default): a model-promoted, singly observed downward increment also needs recent same-lane top shield erosion.
- `MOTION`: a continuous downward increment without own-launch/upward contradiction can be reviewed without the shield corroboration.

The choice changes the explicit rubric in future Jev state **and** the code-side promotion guard. It is a persistent sensor policy, not a stale action tied to the opening coordinates. Invalid output is ignored. There is no second runtime call in this controlled experiment: one episode is usually shorter than two useful Astra turns. For a later iteration, repeated identity conflicts or measured false promotions—not merely an empty bomb list—would justify asking Astra to redesign the reviewer interface. Here the interface remains bounded and auditable.

## Jev state and questions

Schema: `space_invaders_temporal_evidence_v1`.

- `rules`: y decreases for an outgoing laser, increases for incoming bombs; labels can be wrong; empty bombs is not safety; do not invent objects.
- `plan`: current `MOTION`/`CORROBORATE` rubric and its full meaning.
- `life_state`: responsive versus a motion gap.
- `candidates[id]`, for a projectile: decoder label, one-downward-increment/not-measured category, own-launch association, same-lane top-chip corroboration, above/below shield region, continuity category.
- `candidates[id]`, for a chip: top-entry erosion, own-launch match, same-lane ascending/descending track, thin/substantial remaining cover, and the warning that an absorbed projectile need not survive.

Each candidate gets two independent `Choice` questions with reversed option order:

1. `INCOMING`: evidence supports enemy-origin motion/damage, without an outgoing contradiction.
2. `OUTGOING`: own launch or ascending same-lane evidence explains the cue.
3. `UNRESOLVED`: insufficient/discontinuous/conflicting evidence.

The instructions name the candidate directly, spell out the rubric, and forbid movement selection. There are no numeric comparisons or counting tasks for Jev. All geometric features are derived in code.

## Acceptance, freshness, fallback

- Both reordered answers must agree; selected probability must be **at least 0.72** in both. Supplied confidence must be **at least 0.30**; missing confidence is not fabricated.
- Request age at most **36 emulator frames**; same life/teleport epoch and rubric version. Projectile identity revision must still match, and the track must have been observed within **8 frames**.
- Incoming promotion cannot override upward motion, a matched own launch, or absence of a downward increment. A chip promotion requires a still-present matching descending track. Strong two-increment or explicit-bomb evidence is independently handled in code.
- Cache only with that track/cue identity, never globally. A direction contradiction, respawn/teleport, or wave change clears/invalidate evidence. Every resulting maneuver gets a current collision check.
- One outstanding Jev request, **1.0 s timeout**. Observed prior p50 is about 0.17–0.20 s and the parent maximum is about 0.38 s. The main loop never waits for this call. No queued backlog; event-driven queries, at least 16 frames apart.
- One outstanding Astra request, **55 s timeout**, one-token output. Its actual latency remains to be measured; the archive's 30–50 s range makes it unsuitable for individual actions.
- Timeout, failed call, garbage answer, disagreement, low probability, or stale identity retains the per-tick deterministic sweep/visible-motion safety/limited thin-chip fallback. Every posted action includes FIRE. No worker contains an unbounded Python loop.

## Confirmation and refutation

**Confirm** only if the evaluation shows `descending_shot_field` events or accepted ambiguous cue associations **before** otherwise blind impacts, actual `jev_changed_action` events, fewer life losses/longer survival and improved score across training seeds. Stronger evidence is an incoming track continuing coherently after a model association, with the selected path avoiding its predicted collision interval.

**Refute** if the projectile ledger never exposes missing threats; if Jev decisions expire before changing an action; if relabeled tracks are actually new friendly beams; or if false detours reduce score without survival gain. A higher score with zero sensor interventions is evidence for the preserved sweep, not confirmation of the new hypothesis. Continued zero waves cleared remains an explicit failure to solve the main headroom.

Diagnostics distinguish `descending_shot_field`, `unattributed_chip`, `jev_evidence_verdict`, `jev_changed_action` (baseline and chosen direction), `projectile_escape`, `chip_escape`, `lattice_identity_conflict`, `life_evidence`, `astra_rubric_changed`, and final counters.

## Validation status

The module was written before refinement and passes Python syntax compilation. Only standard-library and documented `metaharness.*` imports are present. Local isolated Python cannot import the framework package; contract/garbage-output checks will use explicit stand-in objects, not real model backends or emulator internals. No game experiments or score claims are made in this proposal; the evaluator performs the actual preflight and training evaluation.
