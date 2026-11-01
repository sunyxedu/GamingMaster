# 0016 — Separate semantic hazards from carrier geometry

Parent: **0009**, structural rewrite. Inspected evaluated best **0014**, including its worst seed103 and low-score seed101 traces, all six of its failure records, and the parent’s worst seed113. Retain the proven separated-copy geometry, time-layered route search, median transit, neutral rearming, and asynchronous inference. Replace route-label/policy nudges with **semantic hazard overlays and conditional responses**.

## Concrete diagnosis

The archive's positive-clearance failures are not all evidence of wrong carrier widths or insufficient margins. Some are intersections with an object that the planner never considered hazardous.

* Parent0009 seed113 fails final UP at **6106**, source x46, HOME_1, certificate margin **3.0**. It fails again at7850. Parent geometry padding cannot explain away the failed interior entry.
* Best0014 seed103 fails HOME_2 at **4145** and **4707**, source x77/x76, margins1/0. Immediately before **both** failures, `ram_fields.collision_e3=131` and `collision_row_e5=3`. Low five bits are the occupied-home mask; the raised bit128 is active and the remaining target value equals bay2+1. Seed101 shows the same pattern at **11094**, HOME_1, E3=129/E5=2, and **12878**, HOME_2, E3=131/E5=3. Across all six evaluated0014 traces, **13/13 entry failures** have this exact *matching raised alert* condition. Conversely, a successful entry can occur while the raised alert names another bay: seed109 tick3310 enters HOME_0 while the observation retains E3=129/E5=2. Thus this is a bay-local hazard, not an unconditional global stop.
* Best0014 seed103 fails on river row8 at **14596**, frog x13, while the decoded safe carrier is shape184 and the object's public “fly” fields are `AE=8, B2=2, C7=128`. At14762 the analogous pair is frog97 / sprite86; at16002 frog75 / sprite66. Seed101's15367 failure is frog53 / sprite42; its15647 failure is frog100 / sprite108. Across the six evaluated0014 traces, **12/12 row8 failures** have an active C7-bit128 board sprite on that row with frog offset **−8 through+16** from B2. These failures include interior support, waits, lateral hops and new landings. The named `fly.hazard=false` is not a reliable assertion here: the manual places flies in home bays and later-level snakes on the river.

These are observational associations, not a claim to have inspected the ROM or proved every RAM meaning. The conservative operational rule is to treat a raised matching home phase as unavailable and a board-located bonus sprite as a spatial obstacle. Width and motion uncertainty are explicit. This does not ban an entire carrier, row, or home permanently.

## Hypothesis

The remaining score ceiling is substantially caused by **semantic omissions**: home-alert state is ignored and a snake-like bonus-channel object is treated as harmless. Separating these obstacles from platform support should preserve reachability while avoiding the repeated lethal contacts. Global padding, repeated phase delays, and failed-coordinate quarantines were already unconvincing in0011–0013; none is the primary mechanism here.

## New division of labour

### Code: physical prediction and evidence

Code handles RAM access, epoch/life/score bookkeeping, object speed estimates, separate sprite copies, legal movement, six-frame hop/release scheduling, and time-expanded search. It keeps the successful0014 road endpoint at supported river row7 rather than median5.

A second layer overlays hazards on otherwise safe surfaces:

* `E3 & 0x80`, with `E5` naming the target bay, blocks that entry. The default additionally stages through the warning (`0x40`) phase. Other bays remain available. Live guards cancel even an already-pending final UP if the alert changes.
* An active C7-bit128 bonus object with an on-board AE row becomes a projected obstacle around B2. It is separate from carrier ink. Code estimates its motion from observations and sweeps a bounded uncertainty envelope rather than assuming the bonus name proves safety.
* Home entry checks its supported source as well as its open bay. Every action is rechecked after search using a new observation. A rejected action gets a tested escape/hold, not an unconditional long wait.

Code selects a viable upstream target itself. Models no longer spend each answer choosing the same obvious home label or requesting generic wider clearance.

### Jev: semantic classification and conditional response

A compact strict-JSON state contains:

* objective and explicit rules distinguishing home flies from board intruders;
* the observational evidence summary and the current Astra policy;
* named facts: `home_phase`, `alert_matches_target`, `other_open_home`, `board_sprite_active`, `sprite_on_frog_row`, `sprite_collision_near`, `at_dock`, `river_committed`, `timer_urgent`, and counterexample/bank-contradiction flags;
* the exact answer-to-control bridge.

All distances, bit tests, row matches, occupancy and numerical comparisons are computed in code. Jev is not asked to decode hex or perform a route search.

One request batches:

1. `home` and reversed `home_check`: HOSTILE / CAUTION / CLEAR / UNKNOWN. This validates which *target-local* phase applies; CLEAR is not a global claim that everything is safe.
2. `object` and reversed `object_check`: ROW_HAZARD / ABSENT / UNKNOWN. This records the semantic interpretation separately from geometry.
3. `response` and reversed `response_check`: ESCAPE / BYPASS / STAGE / PROCEED / UNKNOWN. BYPASS can change the current home target; STAGE retains it while code searches supported waiting; ESCAPE selects the tested urgent-escape protocol rather than treating a nearby object as a bonus.
4. `review`: Noul for explicit counterexample/bank contradiction, diagnostic evidence for policy review.

A pair must agree with both winning probabilities **>=0.72**. Missing probabilities, malformed values, disagreement and garbage stand-ins do not become certainty. A response must match life/home/level epoch, policy version and semantic signature (home phase/bay and board-sprite row), and be at most **75 ticks** old. Collision-near and target conditions are rechecked live. The semantic lease expires after90 ticks or an observed signature change. It does not cache an action or moving coordinate.

Requests occur every36 ticks during relevant hazards, otherwise150, with one outstanding request and a **3-second timeout**, well above the archived ~0.2s typical and ~1s tail. Actual latency is logged per response. The motor never waits for it. Raised home and board-object safety overlays remain in force even when Jev is late or uncertain.

### Astra: semantic protocol reviewer, not joystick controller

The first observed anomaly opens a review after motor warmup, rather than waiting for multiple sacrificed frogs. Subsequent calls are separated by4200 ticks unless a new failure opens a review. Astra receives the compact schema/bridge contract, exact observational correlations, and at most two concrete counterexamples with before/after public states. It returns only bounded enums:

* `home_phase`: full_cycle / raised_only;
* `response`: bypass / stage;
* `intruder_response`: escape / detour.

These become the next Jev state and the executable response policy. It cannot invent coordinates, numerical action mappings, carrier widths, or unverified RAM fields. New counterexamples invalidate an in-flight policy revision; arbitrary prose preserves defaults. Calls run asynchronously with120s timeout. The falsifiable revision question is whether a semantic overlay misses a new hazard or overblocks benign behavior, not whether all river margins should grow again.

## Confirmation and refutation

Confirm if matching-raised-alert entry deaths and proximity-to-board-sprite row8 deaths disappear, homes per life rise, and the controller passes levels5–6 without widespread no-route/timeouts. Logs separate `hazard_overlay_changed`, actual target changes, Jev semantic acceptance/discard, Astra compilation, and complete failure witnesses. A higher score alone cannot establish model benefit; look for accepted responses that change the target/protocol.

Refute or narrow the hypothesis if the same contact classes die despite the overlay, if nonmatching-alert deaths dominate, if the presumed board intruder can be approached safely without hazard, or if avoiding it reduces reachability enough to cause timeouts. Median and road failures remain distinct unresolved classes; they must not be misattributed to a carrier merely because the previous plan had positive clearance.

## Delivery / contract

Complete `harness.py` and this proposal were written before live trials. Imports are standard library plus `metaharness.*`, with `build(ctx)` returning `run_episode(game)`. The episode continues to game/context completion. Every pulse is explicitly released, model failures are defensive, and lifecycle exceptions preserve the contract. No emulator source, held-out results, original checkout, sibling candidate, or direct model backend is used.

## Smoke validation after delivery

* A public-observation replay with first-option, garbage-value and timeout stand-ins ran 260 ticks in each case. All posted actions were legal and no parser or contract exception escaped. The replay reproduced the archive witnesses: **13/13** home-entry witnesses were blocked by the target-local alert overlay and **12/12** row-8 witnesses were blocked by the board-sprite overlay.
* The initial live Jev-only trial on seed103 reached **2314 score / 27 homes / level6** at the seven-minute smoke deadline, with no harness error. After removing an over-strict source-support check at docking, a longer seed103 smoke reached **2541** before game completion. These runs retained the intended route throughput rather than stalling immediately on the overlay.
* With the reappearance/phase tracker and urgent escape rearming enabled, the corrected seed101 smoke reached **2918 / 37 homes / level8 / 4 deaths**, ending `game_done`; it had zero failed model calls or harness errors. A separate pre-tracker Astra-enabled seed109 smoke reached **3361 / 40 homes / level9 / 6 deaths**, ending `game_done`; Astra compiled several bounded semantic policies and Jev changed at least one target. This is evidence that the bridge is live, not a six-seed estimate, and the final tracker change still needs evaluation with Astra.
* The corrected seed101 run had no matching-home-alert death. The Astra-enabled run’s remaining deaths were late carrier/median cases rather than the original matching-alert class. Its remaining witnesses were a late row-8 edge case and median/road failures, so the hypothesis is supported for the original home-alert class but not a claim that all late-level deaths are solved. The board-sprite overlay now treats stationary C7=192 phases and the late-level AE=128 reappearance corridor separately; those changes are deliberately narrow rather than global padding.
