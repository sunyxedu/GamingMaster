# Candidate 0019 — verified lane-pulse crossover

## Parents and hypothesis

Parents: **0017** (risk-budget turret, 10,896.7 mean) and **0015** (watch-state lane precommit, 9,360 mean). Both earn score through native-frame ownership, low-duty FIRE, and code-owned aiming. In particular, 0015's seed-6 run reached the frame budget with 14,320 points and only 125 thrust/diagonal-thrust ticks; its strength is bounded movement plus a productive turret, not demonstrated success of the watch label alone. 0017 contributes conservative movement gates and post-discontinuity coasting; 0015 contributes early preparation for a horizontal relocation. This crossover does **not** assume that more thrust, more hyperspace, or an earlier destination label is sufficient.

The worst 0017 trace has **zero `escape_pulse` events**. At frame 1525 the coast candidate has `contact: null` yet `clearance: overlap`; at frame 1555 all four candidates report contact zero. Defenses at 1554, 2999, 3321, and 3793 do not prevent game over at 3820 (3,100 score). The first two defenses precede life changes at 1566 and 3018. In 0015's seed-10 trace, frame 268 already reports watch and left-safe lanes, but the current goal is L114. Defense occurs at 279, and the first life change in the actual JSONL is 304. The archived lesson's different intermediate life-loss frames do not match this JSONL, so this proposal relies on the trace itself.

**Concrete forecast errors explain why copying either safety layer is insufficient.** 0017 uses `min(dx-xgap, dy-ygap)` as clearance, although boxes are separated whenever **either** axis separates them. It also forecasts a one-frame pulse with `0.5*a*t*t`, effectively applying acceleration throughout the horizon. Its zero-contact ranking additionally treats zero as a missing time. These are implementation errors, not evidence that a larger movement budget is beneficial.

Hypothesis: retain the successful turret, correct the forecast arithmetic, and admit a watch-time relocation only when a complete **turn → short pulse → inertial coast** trajectory is reachable and improves clearance. Then explicitly verify the observed outcome before another pulse. This combines 0015's preparation with 0017's restraint without repeating the refuted endpoint-only or repeated-late-defense hypotheses.

## Changes

* Signed box separation uses `max` of the two axis gaps; contact zero is never confused with no contact.
* Single-pulse forecasts use a velocity impulse followed by linear drift. New route forecasts integrate acceleration only for at most eight native frames and coast thereafter.
* Routes compare current nose and the four cardinal headings. Cardinal left/right paths implement the lane relocation intent without pretending that a safe lane can be reached instantaneously. Turn delay is conservatively four frames per orientation code; thrust is forbidden until the heading is actually observed.
* Both a weak (0.035) and strong (0.075 pixels/frame²) response must clear the horizon, with at least one pixel of clearance and two pixels improvement. These bounds are explicitly **engineering hypotheses**, not calibrated cartridge constants. Intermediate vectors and screen-size wrap approximations remain inherited limitations; no RAM is written and source slots are not treated as persistent physical identities.
* A crossing must persist across reviews. A reservation has a fixed expiry, at most eight thrust frames, a rolling cap of sixteen thrust frames per 180 frames, and an eight-frame outcome observation period. New close contacts, discontinuities, uncertainty, disappearance of the hazard, life changes, and defense cancel it.
* `route_start`, `route_pulse`, `route_blocked`, and `route_finish` expose execution and cancellation, unlike the unverified reservations in earlier generations. Outcomes include before/after TTC and whether the danger actually improved.
* Verification compares current TTC with the original TTC **minus elapsed frames**; merely counting down is not success. A completed route must remove the contact or delay it by at least eight frames. Failure halves the assumed thrust response (floor 0.20 of the initial bounds), making later route admission more conservative. This is a model-mismatch guard, not a calibrated claim about cartridge acceleration.
* 0017's FIRE cadence, conservative no-thrust fallback, and at most two defenses separated by 120 frames remain. There are no seed-specific conditions.

## Jev state, questions, and action bridge

Each request carries the parent's compact JSON: explicit mode/fire/hyperspace/unknown-enemy rules; snapshot frame, lives, score, presence, motion, danger, impact-window, object counts and uncertainty; named forecast candidates; selected target size/bearing/motion; current plan. Added fields are `controller.route_stage`, `controller.route_outcome`, `controller.response_scale`, and `plan.version`. The high-rate forecast now computes coast only; the more expensive route search runs only during an actual crossing window, not four redundant forecasts on every tick. Jev never receives raw RAM or has to calculate an intersection.

The batch retains four independent judgments: two reverse-order danger Choices, a posture Choice (`focus`, `sweep_left`, `sweep_right`, `hold`, `coast`), and a defense Noul. Danger requires agreement and selected probability ≥0.45 for immediate, ≥0.18 otherwise. Posture requires ≥0.18 and affects aiming only for 54 frames. Defense probability ≥0.82 supplies a hint; the current local contact, reserve, and cooldown remain mandatory. These permissive posture thresholds are inherited low-consequence settings, not confidence claims about thrust safety.

One request is outstanding at a time, approximately every 30 frames, timeout 2.2 seconds (archived median ≈0.18 seconds, tail up to 1.2 seconds). Replies older than 42 frames or from a different life epoch are rejected. A model answer cannot start, extend, or redirect a thrust reservation. Frame-local code continues while models think. Missing/garbage/late answers retain the deterministic turret and local movement gate. Player absence releases all buttons. Unknown nearby geometry prohibits thrust.

## Astra and interface revision

One asynchronous review starts after 1,200 controlled frames and has a 140-second timeout, allowing the archived 103-second tail. The prompt includes maneuver counters and recent completion/cancellation outcomes as well as score/lives. Only allow-listed preferences are accepted: risk, firing cadence, aim family, and reactive/rare escape. `rare` requires greater route improvement; sweep/target actually changes the fallback aiming mode. Astra cannot alter the collision arithmetic or action durations. Its updated preferences are included in subsequent Jev states. Garbage output is ignored.

A later interface revision is warranted if the new outcome records show that predicted clear routes systematically fail, or Jev's coast choices repeatedly suppress useful aiming without improving survival. Such evidence would call for revised uncertainty/trajectory features, not more danger-classification calls. This candidate deliberately keeps the question interface stable to isolate execution validity.

## Confirmation and refutation

Confirm with higher training mean ALE reward, later life losses, and successful `route_finish` outcomes while thrust stays sparse. In particular, the worst-parent failure should show either a reachable route that actually pulses and reduces danger, or a concrete `route_blocked` reason—not merely a safe endpoint label. Refute if route pulses cause drift deaths, the corrected radar raises false alarms, reservations still never execute, or movement lowers score without extending survival. Survival is diagnostic only; the objective remains native unclipped reward.

## Validation

Implementation and proposal were written before local evaluation. Test results, if available, are appended below. No held-out seed, game implementation, or other archive is inspected.


### Local checks and refinements

* Python compilation and a 600-tick stand-in contract check passed, including timeout replies, malformed/unhashable Jev values, arbitrary Astra prose, a missing player, and exact action-name validation. Geometry assertions separately check axis separation and contact at time zero. These are unit/contract checks, **not** substitute emulator scores.
* First live seed-10 try (four-frame route limit): **1,900**, 2,712 external ticks, **21 stale ticks**, no error. Every candidate route was rejected. The prediction work was too redundant for reliable frame ownership and four-frame routes did not clear the tested crossings.
* After removing redundant every-frame escape forecasts, precomputing each ship path once, and allowing up to eight thrust frames: seed **10 scored 11,950**, 11,910 external ticks, **0 stale ticks**, 397 Jev calls; seed **0 scored 8,760**, 12,044 external ticks, **1 stale tick**, 402 Jev calls. Both ended by true game over without errors. Jev was live; Astra was off. These two runs precede the final TTC-verification feedback fix, so they are developmental evidence, not final-version validation or an official mean.
* The diagnostics do **not** yet establish that the pulse mechanism works. In seed 0, the frame-5291–5298 eight-frame pulse finished at 5306 with TTC changing from 35 to 19 over 16 frames: exactly the unmodified countdown, followed by a life change at 5342. Seed 10 similarly finished a route at 6517 with TTC 2 after starting at TTC 32 thirty frames earlier. This prompted the final elapsed-time-adjusted verification test and response-scale reduction. A later 21-frame TTC is not called improvement merely because it exceeds a fixed threshold.
* The public observation at seed-0 frame 5330 puts the ship at x=82 versus x=84 before the pulse. This is weaker displacement than the initial model suggests; it supports conservative model-mismatch handling, **not** a precise acceleration estimate. The final feedback policy was also tested on training seed 2: **4,370 score**, **8,674 external ticks**, **1 stale tick**, **no error**, true game over. Jev was live and Astra off. Its four reservations were cancelled (hazard cleared or a new close contact); seven thrust frames were actually emitted. No route reached the verification stage, so this run validates execution/contract robustness but does **not** exercise the response-scale reduction.

The broad result is mixed: substantially better on the tested worst-parent seed, worse on its strong seed, and no proof of a training-mean gain. The next iteration should use these explicit failure/outcome records to identify the actual thrust response and reduce quantized-velocity forecast flicker rather than simply allowing longer bursts.
