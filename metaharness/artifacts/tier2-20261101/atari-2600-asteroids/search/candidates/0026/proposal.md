# Candidate 0026 — acknowledged shot transactions

## Parent and concrete diagnosis

**Requested parent:** `0007` (structural rewrite). **Validated numerical reference:**
`0022`, the archive leader. I read both modules/proposals/evaluations and the
worst traces, `0007/trace_seed10.jsonl` and `0022/trace_seed2.jsonl`, as well as
`0018/trace_seed6.jsonl` and `lessons.md`. Sibling candidates were not inspected.

The parent's useful mechanisms are per-frame control, actual FIRE releases and
low-duty thrust. Its worst trace has `DOWN` at frame **279**, TTC **2**, followed
by loss at **304**; further losses are at **2714**, **3332**, and terminal frame
**3622** (score **2900**). More late movement/defense has repeatedly failed in
later generations, so this final candidate does not expand those budgets.

0022 is the more relevant scoring baseline: all six episodes survive the full
frame budget, mean **28,303.3**. Its worst seed 2 finishes with **24,960**, **6
lives**, and only **2 losses**. The actual `gun_summary` is **1873 posted shots**,
**967 observed births**, **963 calibrated samples**, **0 calibration rejects**,
and **16 learned headings**. Only **1/182** sampled `gun_shot` records is
`clearable:true`. The best seed 0 also survives, with **0 losses**, yet only
**994** observed births. This supports the combined numerical turret, not the
claim that pre-contact kills caused its gain.

**Hypothesis:** some remaining lost score is in the *actuation-to-observation
bridge*: a one-frame FIRE request is counted as a shot without evidence that a
missile was produced, and simultaneous turning can change the sampled heading.
The observed/posted gap does **not** prove that all unobserved attempts failed:
flicker, fast hits, slot reuse and tracker association are alternative causes.
We can test the narrower claim with bounded acknowledged press transactions.
Each starts once, emits FIRE for one observed native frame, releases on a new
locally associated missile (or after its short cap), then enforces an actual
release frame before another request. Aim is held only during ordinary turret
presses. Safety movement and hyperspace always preempt it.

## What remains and what changes

The delivered module is standalone. Its reviewed numerical sections retain
0022's established cold per-heading missile learning, analytic direct-screen
intercepts, target ranking, occupancy paths, bounded anchor/escape executor and
last-resort defense. No new wrap period, missile lifetime, RAM rule, physics
model, or seed-indexed schedule is introduced.

The rewrite is in command execution and model responsibility:

* **Code:** `TransactionPilot` separates a shot *request*, a bounded press,
  *visible birth acknowledgement*, and a mandatory release. It counts attempts
  once rather than counting held frames as distinct shots. One press frame and at
  least one release frame; after four frames without an
  associated birth it logs `unobserved`, not "miss". Existing five-frame
  cadence/alignment limits still apply to new requests. A normal turret FIRE
  removes simultaneous rotation, but no escape thrust/rotation is suppressed.
* **Jev:** chooses a target-*allocation family*, not a target slot, raw action,
  lane, arbitrary posture or defense. A separate typed judgment can authorize a
  small alignment reserve for a clean short-turn shot while the field is clear
  and scoring. Exact current geometry is checked again before every use.
* **Astra:** sees actual ten-second blocks of attempts, visible births, native
  score gains and losses, rather than a single delayed tactical snapshot. One
  asynchronous review, after 1800 frames of control, selects a durable
  allocation default and whether the fixed bounded alignment-reserve interface
  should be selective. It cannot rewrite numerical safety or firing duration.

`shot_request`, `shot_result`, `shot_block`, `jev_allocation`,
`astra_allocation`, and `transaction_summary` expose the causal chain. A Jev
answer logs whether its acceptance actually changed the current target/code;
call counts alone are not claimed as meaningful cooperation. Native score
blocks are descriptive: no exact kill attribution or synthetic reward is used.

## Jev state, questions, and bridge

One batched request at most every **48 frames**, one request in flight, timeout
**2.5 seconds**. The archive's typical Jev latency is approximately 0.185 s
(11 ticks), with occasional longer tails. The per-frame numerical controller
continues throughout model latency.

State is a compact strict-JSON object:

* `rules`: FIRE is a level requiring release, mode-0 DOWN is hyperspace, inertia
  persists under NOOP, and enemy/projectile coverage is unknown.
* `snapshot`: absolute frame, presence, code-computed danger, slots
  `available|full`.
* `gun.opportunities`: at most three current candidates described by size,
  calibrated/provisional trajectory, clean/marginal intersection,
  aligned/short/long turn, clearable/crossing/noncrossing threat and known/uncertain
  motion. No numeric geometry is delegated to Jev.
* `gun.phase`: acknowledgement_pending/release/ready;
  `gun.realization`: unmeasured/usually_observed/many_unobserved, calculated in
  code from a completed block; scoring/stalled progress and calibration status.
* `plan`: version, durable focus and patience, and the native-score objective.

`allocation_a` and reverse-order `allocation_b` select **threat**, **clean**,
**value**, or **unknown**. Agreement and both winning probabilities **>=0.60**
are required. Accepted rules last at most **60 frames**; concrete candidates
are recomputed on the current snapshot. An existing local clearable urgent
threat still overrides cleanup. Unknown/disagreement preserves the numerical
fallback, never freezes the turret.

`reserve` is a **Noul**, accepted at **>=0.80**. Its prompt requires a clean,
known-motion, short-turn opportunity, clear danger and ongoing scoring. Code
again checks those geometry/safety preconditions, never delays an escape, and
cannot wait beyond **14 frames since the last attempt** (the base policy already
permits a bounded 12-frame alignment wait). This is a reversible, small gun
scheduling choice, not permission to hold FIRE or postpone hyperspace.

Responses older than **36 frames**, from another life epoch or another Astra
plan version are rejected. Tactical advice expires; no source slot becomes a
persistent identity. Malformed probabilities, arbitrary text, timeouts and
stand-in first options cannot issue controls. The deterministic calibrated gun
is the fallback whenever advice is absent or invalid.

## Astra scheduling and interface

One review is launched after **1800 control frames** if there is remaining call
budget and enough game horizon. Timeout **150 s**, covering the archive's
observed slow answers up to 103 s. It requests JSON under 50 words:

```json
{"focus":"threat|clean|value","patience":"normal|selective"}
```

The prompt includes the exact fixed Jev feature meanings and the meaning of
these options; Astra is not asked to author arbitrary Jev questions or code.
Only allow-listed strings are applied. A valid answer increments the plan
version and invalidates old tactical advice. Unknown enemy coverage and
uncertain shot attribution are explicit. The default remains usable before,
during and after the call; no waits are introduced in the native loop.

Every action is posted from the latest observation and stamped `ACTION@FRAME`
for a six-frame backstop. `ctx.wait_for_tick()` owns pacing; the harness never
steps the game or sleeps. All work goes through the context, only standard
library and `metaharness.*` are imported, and no episode state persists.

## Confirmation and refutation

Confirming evidence would be a larger observed-birth fraction with similar or
higher missile throughput, higher mean native score while preserving full-horizon
survival, and calibration still converging without heading contamination. The
hold/release transaction should be visibly exercised, not just declared. Jev's
allocation/short reserve and Astra's default changes should be observable through
versioned events and target changes.

Refuting evidence would be that the old observed/posted gap was mostly tracker
artifact, acknowledgements add latency without creating more useful missiles,
aiming freezes suppress reward, or life losses increase. Birth counts alone are
not success: native score and same-horizon missile throughput must improve.
Matching trials with and without the transaction layer are needed for causal
attribution; a score comparison to archived 0022 is not a paired ablation.

## Development notes and validation (updated after tests)

An initial written draft explored decayed per-heading reward credit. Its short
live trials ran, but it changed too much of the best numerical controller and
introduced noisy five-frame calibration. That draft was discarded, not tuned
into a second candidate. The delivered hypothesis above is narrower and retains
the established six-frame calibration and numerical safety.

The current module compiles. A final one-minute live seed-2 smoke trial completed
without an error: score **5700**, **3599** ticks, **1 stale tick**, and **74 Jev
calls**. The transaction summary recorded **369** one-frame requests, **194
visible births**, and no life loss in that shortened run. An earlier two-frame
press variant reached the full horizon at **26,630** (seed 0) and **26,820**
(seed 2), but those are development measurements before the final one-frame
cadence and are not official evaluation. The smoke checks and malformed-output
replay are plumbing evidence, not a score claim.
