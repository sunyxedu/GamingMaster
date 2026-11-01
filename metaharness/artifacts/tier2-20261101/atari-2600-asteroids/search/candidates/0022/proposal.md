# Candidate 0022 — calibrated intercept turret

## Parent, evidence, and novel hypothesis

Parent: **0018**, the current best (12,815 mean native reward). I retain its
frame-local clock ownership, occupancy forecasts, bounded movement and model
workers, but test a different failure mechanism: **weapon geometry and shot
allocation**, not another relocation or hyperspace threshold.

In `0018/trace_seed6.jsonl`, an escape starts at frame 1643 with TTC 18. DOWN is
issued at 1657 with TTC 0; frames 1659–1677 are nearly all NOOP and the ship is
lost at 1678. At 1667 the player is `(114,84)` and a newly split small rock is
`(115,60)`. Further escape starts at 4543, 4559, 4571, 4578 and 4585 precede
another loss at 4592. The observation at 4581 shows a small rock `(86,106)` by
the ship `(84,96)`. The parent spends warning intervals turning for evacuation
and disables its gun while maneuvering, yet has no calibrated gun-intercept
alternative. In seed 2, defenses at 6918, 8671 and 8943 precede losses at
6936, 8688 and 8968. These correlations do not establish that DOWN itself
caused death; they establish that the current late escape hierarchy is not a
reliable substitute for clearing the approaching rock.

The parent's normal aiming uses evenly spaced trigonometric headings and a
fixed five-frame lead. The interface explicitly says intermediate headings are
uncalibrated. Archived visible missiles have anisotropic velocities such as
`[2.5,0]`, `[2,-3]`, `[-1.5,3]` and `[0.667,-2]`. A general angle-to-position
heuristic is therefore not a verified bullet intersection solver. No evaluated
candidate has learned a per-heading projectile table and used it to allocate
shots to reachable moving targets.

**Hypothesis:** learn actual projectile vectors from public observations, score
turn-plus-flight intersections in code, and prioritize a threatening rock only
when a shot can reach it before contact. This should preserve the low-thrust
parent's long runs while preventing some collisions offensively. Unlike the
refuted patrol/retry experiments, it does not add movement or defense budget.

## Changes

* Track newly appearing missile slots near the ship after a real FIRE press.
  Associate the visible heading at creation, measure displacement across at
  least six frames, reject jumps, implausible speeds and inconsistent directions,
  and learn a separate velocity/error record for each orientation. Slots are not
  permanent identities; disappearance, jumps and life epochs clear association.
  Only public decoded positions are used: no RAM writes, emulator internals,
  seed branching, save states or hidden training data.
* Use a provisional anisotropic velocity only until a heading has observations;
  keep its confidence explicitly provisional. Direct short in-screen rays avoid
  inventing missile wrap periods. Enumerate all sixteen headings, include a
  four-frames-per-code turn estimate, and solve a normalized closest-approach
  intersection with each moving rock. Collision-inflated boxes are reduced for
  gun targeting rather than reused as hit masks.
* A shot that is predicted to arrive before a forecast crossing has priority.
  Otherwise favor clean low-turn-cost hits and small rocks. Current-nose shots
  can hit any supplied rock, not just the intended future target. Waiting for
  a short alignment is bounded; uncertain predictions never silence the gun
  indefinitely. The five/six-frame press/release gate and two observed slots
  remain intact.
* Preserve movement and hyperspace safety. An active escape may now also fire
  if the *existing nose direction* has a computed opportunity; the gun never
  changes the escape turn/thrust or suppresses emergency defense.
* Replace generic Jev regime/target IDs with semantic gun-selection rules.
  Recompute their concrete target every frame, avoiding persistent source-slot
  identity claims. Astra's durable aim preference selects the rule family from
  actual shot/calibration evidence.

## Jev state, questions, and answer bridge

One request every 36 frames, at most one in flight, timeout 2.25 seconds. State:

* `rules`: FIRE level/release semantics, two observed missile slots, inertia,
  mode-0 hyperspace, and explicitly unknown enemy coverage.
* `snapshot`: observed absolute frame, player presence, local danger and slots.
* `gun.opportunities`: at most three code-ranked candidates summarized by size,
  `calibrated|provisional` trajectory, `clean|marginal` intersection,
  `aligned|short|long` turn, `shootable_before_contact|crossing|noncrossing`,
  and `known|uncertain` motion. Jev does no numeric geometry.
* `gun.calibration`, `gun.recent_progress`, emergency-override flag; and the
  durable `plan.focus` and goal.

`focus_a` and reversed-order `focus_b` independently choose `threat`, `clean`,
`value`, or `unknown`. Both must agree with winning probabilities at least
0.60. The rule lasts at most 54 frames; its exact target/intersection is solved
again on the latest observation. A clearable near threat overrides a distracting
value/cleanup suggestion. `clearance` is a Noul confirming whether the supplied
opportunities explicitly support a clean known-motion pre-contact hit; probability
0.80 is logged as semantic support, never treated as a safety proof.

Results older than 36 frames, from another life epoch, malformed results, or
low confidence fall back to code threat-first aiming. A first-option stand-in
cannot force thrust, defense or long action sequences. Every native frame still
gets a local action while either model is pending. Errors, timeouts and arbitrary
Astra text remain advisory misses; terminal/budget conditions end normally.

## Astra and falsification

One asynchronous post-launch review accepts only three aim preferences and uses
a 150-second timeout (the archived slowest answer took 103 seconds). It receives losses, score, escape/anchor counts,
plus shot counts, observed spawns, vector samples/errors and alignment waits.
`threat_first`, `value_first`, `sweep` map to threat, value and clean-intercept
selection respectively. It cannot emit actions or rewrite numeric safety gates.
A change in this interface would be justified if apparent calibration remains
inconsistent or gun-rule changes do not affect the selected targets; extra
model frequency alone is not the hypothesis.

Confirming evidence: `gun_calibration` converges across multiple headings;
`gun_shot` records clean pre-contact opportunities and visible spawns; fewer
life losses / higher native reward without more thrust or stale ticks. Refuting
evidence: frequent sample rejection, persistent provisional trajectories,
miss-biased aiming, loss of scoring while waiting for alignment, or earlier
collisions despite well-calibrated vectors. Increased score is the official
objective; lives/survival and shot records are only explanatory diagnostics.

## Local validation and what it actually establishes

The module compiles and a full-loop smoke check replaying 79 archived public
snapshots survived first-option stand-ins, malformed typed values, malformed
Astra text, and intermittent `TimeoutError`s. This is a contract smoke check,
not the evaluator's official preflight and not a game-score simulation.
The sampled local update/choose computation took 3.65 ms mean / 5.72 ms maximum
on those snapshots after caching the parent's unchanged compass vectors.

Actual live training trials learned all sixteen heading vectors. The dominant
measured decoded-pixel vectors were:

| Code | vx | vy | Code | vx | vy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | -4 | 8 | 0 | 4 |
| 1 | -0.5 | -3 | 9 | 1 | 3 |
| 2 | -1.5 | -3 | 10 | 2 | 3 |
| 3 | -1.5 | -1 | 11 | 2 | 1 |
| 4 | -2 | 0 | 12 | 2.5 | 0 |
| 5 | -1.5 | 1 | 13 | 2 | -1 |
| 6 | -1.5 | 3 | 14 | 2 | -3 |
| 7 | -0.5 | 3 | 15 | 1 | -3 |

These are **observations in the decoder's coordinate system**, not a claim about
exact cartridge collision masks. The asymmetry and nonuniform intermediate
angles directly support replacing the parent's evenly spaced gun-angle prior.
The final harness still learns this table cold in each episode; it does not
load these trial results or hard-code seed-specific trajectories.

The first five Astra-off trials (seeds 0, 2, 4, 6, 10) all reached frame 18000,
with native scores 29,230 / 30,220 / 23,770 / 26,880 / 25,660. The first live
Astra-on trial, seed 8, also reached the horizon at 26,970. They are exploratory
results, not an official six-seed evaluation and not an Astra on/off ablation.
A separate six-seed Astra-on validation is recorded below.

This supports the **combined calibrated gun and semantic-selection bridge**.
It does not isolate velocity learning from better target allocation or removal
of the parent's generic hold/anchor target advice. Likewise, `gun_shot.clearable`
is a prediction, not confirmed causal attribution of a kill or saved life.
Observed extra ships can extend survival because of additional native scoring;
no survival points or life bonuses are added to the objective. The parent
movement/defense limits remain unchanged. Deferred work includes measured muzzle
offsets, shot-creation acknowledgement, and a matched calibration-only ablation.


### Completed six-seed validation with both live models

These are local `metaharness try --astra` results on the archive's six training
seeds, **not official selection results or held-out evaluation**. Every run
reached the native 18,000-frame budget, with no harness errors, invalid trials,
or failed S1/S2 calls.

| Seed | Native score | Final lives | Lives lost | Learned headings |
|---:|---:|---:|---:|---:|
| 0 | 29,230 | 9 | 0 | 16 |
| 2 | 27,990 | 9 | 0 | 16 |
| 4 | 26,800 | 6 | 3 | 16 |
| 6 | 28,260 | 6 | 3 | 16 |
| 8 | 26,970 | 7 | 2 | 16 |
| 10 | 29,040 | 8 | 1 | 16 |

Mean score: **28,048.3**; range **26,800–29,230**.
The archived parent mean is 12,815.0, but this local comparison is not a paired
causal ablation or a replacement for the forthcoming evaluator run.

There were 2,989 Jev calls and 6 Astra calls.
Jev per-episode median latency was approximately
0.183 s, with worst observed call
0.740 s. Astra completed in
9.73–16.30 s in these tries;
the longer archive latency still justifies the 150-second advisory timeout.
All six Astra reviews selected the value-priority default, which was exposed in
subsequent Jev states and used by local target ranking. It never changed safety
limits. Aggregate stale fraction was **0.120%** during parallel local
trials: low, but not zero and not better than the archived parent's 0.058%.

A broader robustness pass exercised **1,227 public parent snapshots**, including
fields with eleven observed rocks. It produced only legal actions, strict JSON
states and no exceptions under malformed advisory answers; local update/choose
cost was 3.99 ms mean and 9.40 ms maximum in that pass. These are sampled timing
checks, not a guarantee of a hard real-time deadline.

`trial_results.json` records every local trial separately, model setting,
termination reason, failures, latency, calibration, native life counters, and the
final harness hash. `summarize_trials.py` regenerates it from this candidate's
traces. The first Astra-on seeds 6 and 8 preceded only the terminal diagnostic
and an exactly equivalent compass-vector cache; their gun and movement rules
were the same. No scoring-controller parameters were tuned to individual seeds,
and no held-out seeds were inspected or run.
