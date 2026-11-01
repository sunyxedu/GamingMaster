# 0019 — Stable home commitments with an actor-triggered escape controller

**Parents:** 0017 and 0014 (crossover). I retained 0014's whole-frog transit plan (road search through row 7, median as a transit gate, settled final docking) and 0017's complete-home route portfolio, auxiliary-actor layer, matched-home alert, and second median track. The new intervention is at the execution boundary: preserve an actually feasible home commitment instead of selecting a different complete route whenever its instantaneous rank changes, and interrupt waiting when a same-row auxiliary actor is about to meet a carried frog.

## Hypothesis and trace evidence

0017 is the strongest candidate (mean 4433; 53 homes), but still loses the run's full life stock. Its worst episode, seed 127, has two concrete failure signatures that motivate a structural fix rather than a larger generic clearance margin:

* In its worst trace, the home itinerary flaps **1→4 at tick 18714, 4→1 at 18717, then 1→4 at 18724**. These are rapid reversals in a single crossing. The route portfolio helps when a committed bay is truly unavailable, but 0017 lets a frame-local route winner replace a still-usable commitment. Parent 0014's stable whole-crossing commitment was designed to avoid this class of dithering.
* At tick **16280**, the frog dies on river row 8 at x100 after waiting on a safe-looking turtle while the public auxiliary candidate is at row 8/x108. The last plan is a **NOOP** at x93 (tick16265), with predicted margin3.5. The same frog/actor contact pattern recurs at tick26514 (x100/x108). Another row-8 loss at tick22094 has frog x148 and auxiliary x132 although its plan reports margin24.5. Those observations show that a static landing/support margin is not a sufficient *ride-through* certificate: the frog can drift into an actor between replans.
* The other observed losses include a road death at tick19237 and a timer-1 river-row-7 loss at tick28496. These confirm there is residual uncertainty, so this proposal does not claim that the new escape rule eliminates all deaths.

**Testable hypothesis:** (1) a target commitment should remain stable while a current complete route to that open, unalerted bay exists; (2) when the independently tracked auxiliary actor is on the frog's current river row and is close or approaching, immediately choose a *supported* action that exits that row or increases separation, rather than waiting for route search to turn the carrier's future drift into a hard veto. This should reduce target reversals and close actor-contact deaths without the global padding that previously reduced reachability.

## What changed

1. **Retained 0014's transit structure.** Road routes plan through the median and into a useful river corridor (row 7). At harder levels the median is a transit link, not a safe place to dwell. A dock remains a distinct, live-checked UP from a settled top-carrier position.
2. **Retained 0017's route and observation layers.** The controller still searches alternatives among complete home routes, excludes only the currently matched home alert, models the public auxiliary actor separately from carrier support, reserves the observed spawn mouth, forecasts the second median track under its established conditions, and rechecks the chosen action against the latest observation.
3. **Added commitment hysteresis.** In river rows, if the committed bay is open, unalerted, and has a complete route in the current search, route selection is restricted to that bay. If it temporarily has no complete route, an alternate goal is held as a pending option for 18 game frames before the commitment changes; an urgent timer or a newly invalid/alerted home bypasses that delay. During the pending window, code can use the same alternative route consistently without prematurely rewriting the durable target. This uses 0014's itinerary commitment while preserving 0017's real reroute escape hatch.
4. **Added an actor-imminence escape.** After normal route selection, code checks whether the auxiliary actor is on the frog's river row and within 20 pixels, or is predicted to close within 24 frames using observed actor velocity and row drift. It evaluates UP/DOWN/LEFT/RIGHT against the same live support/collision field. A valid vertical transfer that leaves the actor's row is preferred; otherwise a supported lateral move must increase predicted separation. NOOP is not an escape candidate. This does not make a bonus label or unknown actor harmless, and does not authorize an unsupported jump.
5. **Models remain supervisors.** Neither Jev nor Astra is permitted to bypass live geometry, the committed-target invalidation checks, or neutral rearming. Invalid or stale answers preserve code policy.

## System-1 interface and action bridge

Each Jev request keeps 0017's strict-JSON compact state:

* `game`, `phase` (`road_launch`, `bank_transit`, `river`, or `dock`), and the validated Astra `policy` enums;
* `homes`: available bay labels and derived booleans for upstream frontier, nearest now, current commitment, failed entry, matched alert, and complete route now;
* named `facts`: complete launch exists, higher-level/bank threat, auxiliary on a playable row / on the frog row / reserved spawn, target alert, timer urgency, and recent entry/bank/carrier contradiction;
* short explicit rule text explaining that a home choice is a whole-crossing itinerary and unknown crocodile state is not proof of safety.

There are four batched judgments: reversed-order `Choice`s for `itinerary` and `gate`, plus a `Noul` `review` signal. Itinerary candidates are open bay IDs plus `UNKNOWN`; gate candidates are `TRANSIT`, `STAGE`, `RECOVER`, `SCREEN`, and `UNKNOWN`. Jev selects a crossing goal/protocol, not an unverified frame action. The duplicated Choices must agree and each selected-option probability must be at least **0.70**. Results must match the life/home/level epoch and policy version and be at most **90 ticks** old. A review probability of at least **0.85** may request an Astra protocol review. Missing probabilities, disagreement, malformed output, timeout, or stale results are ignored.

Astra remains an asynchronous, low-frequency reviewer of the latest concrete failure witnesses. It can choose only existing itinerary/launch/dock enums; accepted changes increment the version and invalidate prior Jev advice. It cannot change actor-clearance code or invent an action. The route executor and immediate escape operate without model answers, so the measured Jev latency (about 0.2 seconds typically, occasional longer tails) does not block the game clock.

## Fallback and lifecycle

The fallback is the code-owned upstream itinerary and time-expanded route, followed by the latest-state support guard. Direction pulses release on observed movement or at four frames and are rearmed with NOOP. If the auxiliary imminence condition fires, the fastest supported separating action supersedes route preference; if no supported escape exists, ordinary guarded route selection remains. Matched alerts and unavailable bays invalidate a commitment immediately; a merely transient lack of a route does not trigger target thrashing. The main loop runs until the game or context ends. The existing handler propagates lifecycle/budget/backend exceptions normally, catches ordinary inference errors, and ignores arbitrary Astra text unless it parses to allowed enums.

## Confirmation and refutation

Confirm if evaluation traces show (a) fewer rapid changes of `target` within one life/crossing, (b) `auxiliary_evasion` events preceding fewer same-row close-contact losses, (c) more homes per life, and (d) no rise in median dwell, timeout deaths, or no-route stalls. Compare scores and deaths as well as stale fraction; a high evasion count alone is not success. Refute if commitment hysteresis traps the frog while another bay is feasible, if the row/velocity contact prediction fails to trigger before the observed actor contacts, if supported escape actions cause more deaths, or if reduced target changes lower homes/score.

The new code is intentionally limited to these two interacting changes. It does not claim that an auxiliary candidate is a verified ROM snake decoder, that every positive route margin is safe, or that a short trial confirms a leaderboard gain.
