# Candidate 0010 — verified hyperspace retries

## Parents and concrete diagnosis

Parent: `candidates/0008/`; the best reference is `candidates/0007/`.
I read the parent controller, the best controller, and the best candidate's
worst trace (`trace_seed10.jsonl`) plus its long seed-4 trace.

The previous structural improvement—bounded lane relocation—works often enough
to score 5,343 mean, but it has a specific hard failure.  In seed 10,
0007 logged a `defense` at frame **279** with `threat_time: 2`, then lost a
ship at frame **304**.  The observation at frame 293 still showed the player
present at `(116,116)` and `danger: immediate`; the controller posted turns and
firing rather than another defense because `defense_used` was already true.
In seed 4 the same pattern is clearer: after a first defense at frame **2677**,
0007 repeatedly classified the field as immediate around frames 3172–3208 and
lost the ship at **3210**, again because its once-per-life defense gate was
already spent.  All candidates so far either use no defense or cap it at one
pulse per life.  A one-shot gate is a heuristic, not a disclosed cartridge
rule.

The parent 0008 also confirms that more proactive thrust is not the missing
answer: its thrust-heavy seed-4 run used 1,760 thrust/diagonal-thrust ticks and
scored only 680.  The best trace's issue is not stale ticking—0007 had zero
stale ticks—but an emergency action whose result was never verified.

## Hypothesis

Mode-0 hyperspace should be treated as a bounded, observable escape attempt,
not as an irrevocable once-per-life resource.  If a DOWN pulse leaves the player
present in essentially the same dangerous encounter, or relocates into another
immediate contact, a second pulse after a short cooldown can save a ship that
rotation cannot save.  Conversely, blindly repeating DOWN is dangerous because
landing is uncertain.  Therefore a small per-life retry budget plus a
post-pulse verification/recovery state should improve score and survival over
0007 without reproducing the long UP latches of early candidates.

This is a structural test of an untested cartridge/interface assumption.  It
does not import an arcade hyperspace probability or claim a score bonus.  The
controller allows at most three one-frame DOWN pulses in a life, requires a
current code-derived critical collision and a 16-frame inter-pulse gap, clears
an old escape plan after every pulse, and invalidates model advice on life or
motion discontinuities.  The default policy is `retry`; Astra may conservatively
change it to `single` or `never`, but cannot bypass the local gate.

The second change is a distinct bridge for that experiment: a receding-horizon
planner scores eight candidate thrust headings against all currently decoded
tracks and emits only a finite nose-heading/thrust allowance.  It is not the
five fixed horizontal lanes of 0007 or the parent’s periodic sector patrol.
Normal play returns to code-lead aiming and press/release FIRE, so the defense
experiment does not sacrifice all demonstrated scoring behavior.

## What would confirm or refute it

Confirmation would be: (1) retries occur mainly when the post-pulse player
remains present and the critical bucket persists, (2) the frame of the next
life loss moves later or disappears in the seed-10/seed-4-like episodes, and
(3) mean ALE reward rises while stale ticks remains near zero.  A particularly
strong confirmation is a second DOWN followed by a discontinuity and continued
life, rather than a second DOWN followed immediately by death.

Refutation would be: retries are common without a persistent critical state,
landings cause more immediate deaths than they prevent, score falls because
hyperspace interrupts useful firing, or the life-loss frames and mean score are
no better than 0007.  The trace events `defense_pulse`, `life_change`,
`escape_plan`, score changes, and final lives are the intended diagnostics;
official score remains the unclipped ALE reward only.

## Harness changes

* `RecedingAsteroids` owns every native frame and posts a fresh legal action
  while model requests are outstanding.  Malformed observations or model
  output leave local control running.
* Tracks use wrap-aware relative positions, measured velocity when available,
  inflated uncertainty geometry when velocity is null, and a 60-frame sampled
  clearance forecast.  Eight cardinal/intermediate headings are scored in code;
  the controller rotates first in normal danger and uses at most 4–6 thrust
  frames in a finite escape plan.
* `DOWN` is emitted only by `_defense_allowed`: player present, current
  `critical` geometry, no more than three pulses this life, and at least 16
  frames since the last pulse.  After a pulse, the escape plan is discarded and
  a 10-frame recovery window avoids compounding stale inertia.  A life change or
  motion discontinuity resets the per-life defense state and invalidates model
  hints.
* FIRE is a level press followed by a code-owned gap; missile occupancy gates
  another press.  No model answer can hold FIRE, thrust, or DOWN.
* Astra is one delayed aggregate policy review.  Its allow-listed fields change
  only aim, risk, fire spacing, and retry posture.  It cannot post actions.

## System-1 state and questions

Each Jev request is strict JSON and contains:

* `rules`: mode 0, FIRE release semantics, NOOP inertia, DOWN hyperspace,
  orientation mapping, and unknown enemy coverage;
* `snapshot`: absolute frame, lives, score, player presence, danger bucket,
  impact-window bucket, clearance bucket, speed bucket, asteroid count, and
  missile count;
* `threat`: one code-selected target summary (size, bearing, distance bucket,
  and known/uncertain motion);
* `defense`: pulses used this life, retry availability, and age of the last
  pulse;
* `candidate_modes`: named semantic choices (`target`, `sweep_left`,
  `sweep_right`, `hold`, and, only when code generated one, `escape`).

The batched questions are:

1. `mode_a`: choose one named bounded posture from the current candidates.
2. `mode_b`: the same judgment with reversed option order, reducing the
   documented first-option bias.
3. `risk`: classify `clear/watch/urgent/critical/unknown` from supplied buckets;
   this is logged and does not perform arithmetic.
4. `defense`: a Noul asking whether a retry may be considered for a persistent
   critical contact.

A mode is accepted only when the two Choices agree, the selected option is
still legal in the current candidate set, the answer is no more than 42 native
frames old and from the same life epoch, and both selected probabilities meet
0.34 for `escape` or 0.18 for ordinary postures.  Accepted posture advice lasts
50 frames and affects aiming only.  A defense hint needs yes probability at
least 0.76 and is still subordinate to `_defense_allowed` on the latest
observation.  Disagreement, low confidence, stale results, timeouts, garbage
stand-in answers, and life/discontinuity changes are ignored.

## Fallback and Astra revision rule

While Jev (2.5-second timeout, one outstanding request about every 24 frames)
or Astra runs, `wait_for_tick()` advances the game and the local controller
continues.  Critical geometry immediately gets a verified defense retry when
eligible, otherwise a finite escape or aim fallback.  A missing player emits
NOOP.  The local loop catches normal model timeout/cancellation and parse
failures; it never waits for a model to own a native frame.

Astra is scheduled after 360 observed frames and receives only aggregate score,
lives, danger, clearance, impact, rock density, defense count, and current
policy.  It must return a small JSON object with allow-listed `aim`, `risk`,
`fire`, and `defense` values.  It is asked to change the policy only when the
aggregate evidence suggests a persistent scoring/survival imbalance; invalid
or prose output is ignored.  In particular, Astra cannot decide that a stale
old screenshot warrants a DOWN action.  The hypothesis-specific default remains
`retry` if Astra times out.
