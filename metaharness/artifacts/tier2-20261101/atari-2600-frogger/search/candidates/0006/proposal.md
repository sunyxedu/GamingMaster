# 0006 — Sprite-copy routes and durable model goals

Parent: `0002-tier2_idle` (structural rewrite). Evidence also used: evaluated candidates `0003` and best candidate `0005`; no sibling candidates inspected.

## Hypothesis and evidence

The idle parent's only demonstrated mechanism is a nonblocking clock loop: seed 101 idles for 9,711 ticks, scores zero, and exhausts its frogs. Keep that independent clock discipline, not its inaction. The best candidate proves that neutral-rearmed hops work, but its geometry is not a safe model of the cartridge.

Concrete examples from 0005:

* Seed 101: UP issued at tick 13 is labelled feasible with margin 1.08; the frog reaches road row 1 at x=80 at tick 14, beside a vehicle at x=86, and loses its life at tick 16. Treating the frog's left coordinate as a centre with radius five understates its forward horizontal footprint.
* Seed 109 (worst, score 19): UP at tick 371 enters road row 2 at x=96. The lane's object at x=64 has NUSIZ=2. It was treated as a solid 24-pixel interval ending at 88; the second sprite copy instead explains the collision at x=96 and death at tick 374.
* Seed 101: LEFT at tick 233 on river row 2 was labelled `wide`, margin 10.5; death follows at tick 236. The same row and lateral pattern kills frogs repeatedly in seeds 107 and 127. A NUSIZ=6 group is separated copies, not a continuous 32-pixel platform. Bounding-box interiors include water.
* 0005 makes 6 Astra calls but has no completed Astra answer before its short episodes end. Row-specific Jev advice is often already stale after its approximately 0.18-second latency. It is not useful to ask either model to repair an invalid collision predicate frame by frame.

**Hypothesis:** reconstructing separated sprite copies, using a frog-left coordinate, and planning to safe *endpoints* will preserve enough lives to complete repeated homes. Models should choose durable route goals and review prediction failures, rather than pick fleeting joystick directions. This is a division-of-labour rewrite, not a timing tweak.

## New architecture

1. **Code owns perception and prediction.** NUSIZ low bits expand to sprite offsets and scaled widths. Adjacent river intervals are merged; gaps remain water. Road exclusions include the frog's eight-pixel footprint plus uncertainty. River support uses the frog centre inside actual carrier ink, not the bounding box. Unknown/hazard carriers and generic `river_carrier` objects are excluded, even when a generic carrier shares a recognized shape pointer; trial evidence showed that shape matching alone did not establish safety. Object velocity is measured from recent public observations and snapped to the observed activity cadence. No source, ROM, emulator, private state, clone, or seed-specific route is accessed.
2. **Code searches time-expanded routes.** Six-frame hop/release opportunities form a bounded beam search to the median or an unoccupied home. Completed routes dominate partial forward progress. Searches precompute safety arrays, replan from the current observation, and execute only the first hop. A model can never authorize a geometrically rejected move. Death/home animation is detected from the observed frog state; input stays neutral while the frog is inactive.
3. **Jev chooses a durable endpoint.** One asynchronous call at a time chooses among full forecast routes. Its decision may survive row changes, but not a death/home/level transition, road-to-river phase change, policy revision, or age limit. This replaces the best candidate's immediate-action questions and row-based expiry.
4. **Astra reviews counterexamples, not the initial frame.** After repeated prediction failures, or Jev's review flag, a background Astra call sees recent planned moves and before/after public observations. It emits a tiny enum policy selecting balanced/safest route ranking, normal/extra obstacle clearance, and centre/inset carrier support. Valid changes alter deterministic geometry and are carried in later Jev state. Invalid output preserves the executable policy. Astra is not called while an untested initial plan burns its entire latency budget.
5. **Motor output is independent.** A direction is held only until public movement confirmation, with a four-observed-tick cap, then explicit NOOP until the next opportunity. This handles the observed alternating-frame input sampling: one consumed display frame was not always a sampled joystick edge. The harness posts an action every tick and waits with `ctx.wait_for_tick`. While inference is late it searches the newest observation, never repeats a stale directional answer.

## Jev interface and bridge

Each request explicitly carries game objective, phase (`road`/`river`), validated Astra policy, timer urgency, recent prediction-failure boolean, home-crocodile uncertainty, and route candidates. Candidate facts are semantic: goal label, complete current forecast, shortest/detour, widest-available/narrower, comfortable/tight support, and current-goal flag. Jev is not asked to subtract coordinates, count frames, interpret raw RAM, or discover sprite geometry.

Questions:

* `goal`: Choice of full-route endpoint, plus UNKNOWN.
* `reverse`: the same Choice with reversed options, to detect order sensitivity.
* `review`: Noul asking whether explicit prediction-failure evidence merits policy review.

Both Choices must agree and both selected-option probabilities must be at least 0.65. Missing probability is zero, not certainty. A response must be within 90 ticks, with the same episode epoch, phase, and policy version. An accepted goal lasts at most another 180 ticks and only changes action if a fresh complete route to that endpoint still exists and is near the leading route. `review >= 0.8` can trigger Astra when counterexamples exist and its cooldown permits. Request timeout is three seconds, well beyond the measured Jev tail; one outstanding request bounds load. Events record accepted/rejected goals, actual goal-induced action changes, route predictions, failures, search time and observed model latency.

Fallbacks: newest code-verified completed route; otherwise feasible progress; otherwise a feasible one-hop survival move; otherwise NOOP. Budget/lifecycle/unavailable-model exceptions retain their normal semantics. Transient model errors are caught. Model garbage cannot become executable code or an unvalidated direction.

## Confirmation and refutation

Confirmation requires fewer immediate safe-prediction deaths, measured NUSIZ gap avoidance, sustained full road/river crossings, and repeated distinct homes, not merely more forward hops. Durable Jev goals should be adopted despite several intervening hops and sometimes change a route action. Astra should only change policy from concrete outcomes and must not stall motor output.

Refutation: deaths still coincide with predicted supported sprite interiors; full-route search gets stuck despite feasible crossings; excessive computation creates stale ticks; or retained goals systematically prefer paths no longer valid. The NUSIZ interpretation and six-frame cadence are hypotheses checked through public observations, not claims of an exact simulator. Live trial findings and refinements will be appended below.

## Initial delivery

A complete harness and this proposal were written before live trials. No evaluation performance is claimed yet.

## Public-observation trial findings

The standard live-Jev CLI failed before starting an episode because `TYPESAFE_API_KEY` was unavailable. No credentials were inspected. For controller-only smoke tests, the CLI's model factory was replaced in the test process with `None`; the unchanged official runner/context still drove the real-time game. These tests made **zero System-1 and System-2 calls** and are not a live-model evaluation.

The initial six-frame, one-display-frame pulse version scored 128 with two homes on seed 109, ending normally at 48 seconds. Its trace exposed two correctable errors:

* Pulses consumed on ticks 11, 17, 23, 67, 73, etc. did not move the frog, whereas even-tick pulses did. Holding until observed motion (bounded at four observed ticks), then releasing, removed this sampling-phase failure. Search and release remain independent of model latency.
* Deaths at ticks 542 and 900 followed landings on generic `river_carrier` objects, despite matching a known log's shape. The final code excludes these uncertain carriers instead of assuming shape equality proves safety.

Two corrected 72-second, model-free smoke tests then produced:

* Seed 109: score **784.0**, homes 9.0, deaths 2.0, level 2.0; ended `deadline`, error `None`.
* Seed 101: score **696.0**, homes 8.0, deaths 2.0, level 2.0; ended `deadline`, error `None`.

Both reached the five-home bonus and level 2, unlike the archived best candidate. These are capped diagnostic runs on two training seeds, not a six-seed mean or held-out claim. Search costs sampled from archived observations were roughly 4–10 ms on roads and 6–20 ms on river rows; the corrected concurrent smoke tests still had about 17–18% stale ticks, identifying computation scheduling as remaining headroom.

A final small home-entry guard was made after these tests: the allowed frog-left interval is asymmetric around a bay centre, leaving room for rightward drift before the input is sampled. This responds to two observed deaths near x=51 while targeting the x=48 bay; that last guard has been compiled but not separately trialled. Live Jev goal quality, runtime Astra adaptation, longer-run higher-level behavior, and alternate modes remain unverified. The harness itself never patches model factories or accesses anything beyond public ctx/game APIs.
