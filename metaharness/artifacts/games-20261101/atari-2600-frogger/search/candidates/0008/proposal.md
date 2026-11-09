# 0008 — Copy-aware collision certificates

Parent: `candidates/0005/`. Mode: explore.

## Hypothesis and concrete diagnosis

**The parent has solved actuator rearming, but searches a false world model.** It treats each NUSIZ object as a solid `width_px_estimate` rectangle and the frog's x as a point. TIA size codes also specify separated copies. This invents water support between turtles, misses repeated cars, and ignores the right-hand part of the frog sprite. More search or another stale action judgment cannot repair those incorrect feasibility predicates.

Evidence from the parent's evaluated traces:

* Seed 109, tick 23: UP from x72 into road row 1 is certified with margin 0.13. Tick 24's observation has frog x72 and a vehicle at x77, NUSIZ 0. The death is recorded at tick 26: treating x72 as a point misses the overlap of the frog's body with x77.
* Seed 109, tick 371: UP into road row 2 at x96 has claimed margin 3.13. Tick 372 reports object base x64, NUSIZ 2. The death at tick 374 is consistent with a repeated sprite at base +32, not the adaptor's single 24-pixel interval.
* Seed 101, tick 233: LEFT on river row 2 is certified with **10.5 pixels** of clearance, followed by death at tick 236. Seed 109's final failure has frog x49 with turtle bases x25/x33 and NUSIZ 6. Expanded 8-pixel copies at offsets 0/32/64 leave a real water gap there; the parent's solid 32-pixel rectangles claim support.
* The same river lateral failure recurs at ticks 560 (seed 101), 1029/1245/1895 (seed 107), and 373/601/1015 (seed 127). These are geometric contradictions, not evidence that another forward-action prompt is needed.
* Conversely, seed 107's successful row-7 arrival at tick 365 has frog x101 and bases x66/x74: the second pair of NUSIZ-6 copies supports that location. Correct copy geometry explains success as well as failure.

The supplied `lessons.md` summary confirms that rearming and frame service are already established. This candidate tests **copy-aware, body-aware feasibility**, not another debounce or latency tweak. The NUSIZ interpretation is a testable world-model hypothesis, not a claim that the adaptor's estimates are exact. Later trials must check it against observations.

## Changes

1. Expand each of the two public object tracks using its NUSIZ copy offsets and scale. Merge only genuinely adjoining intervals; never fill a turtle gap by replacing several copies with a rectangle.
2. Treat frog x as a sprite origin. Road clearance checks the frog's body plus a small margin; river support requires two pixels of overlap with a real carrier on new landings, retaining one-pixel already-observed support for source/ride checks. Expand unknown/diving carriers into no safe support.
3. Learn discrete per-row motion phases from consecutive public observations and the reported activity mask, rather than smoothing quantized movement into an arbitrary velocity. Protect both the source during input sampling and the destination through the press/release window.
4. Search connected routes with a shared, first-action-diverse beam. Prefer higher rows and reachable open homes, penalizing unnecessary lateral river jumps. Execute only the first action and recertify from the next observation.
5. Keep the parent's frame executor and explicit direction/NOOP release. Ignore input during witnessed death/home-animation state 8. Log the exact safety certificate and later outcome; a road certificate contradicted by death increases the episode-local collision margin.
6. Replace fragile Jev next-action votes with short-lived **semantic route preferences**. They can remain useful after the frog advances one row, but cannot authorize an unsafe move.

No game implementation, emulator internals, reset, clone, external inference, private observation, or seed-specific route is used. Imports are standard library and `metaharness.*` only. Motion/geometry are derived from the public observation. The controller continues until game end or the context deadline.

## System-1 interface

Every request carries a compact named state:

* `rules`: cars/water/copy gaps and the code/model division of responsibility;
* `policy`: Astra's durable home order, route style and strict uncertainty mode;
* `zone`: current named frog zone;
* `candidates`: at most three code-certified alternatives, each with action, interior/edge surface, connected/home/holding route, advances and toward-home predicates;
* `wait_supported`, `forward_route_exists`, `home_route_exists`;
* semantic timer bucket and recent certificate-failure evidence;
* explicit home-crocodile uncertainty and rejected carrier kinds.

Questions are batched in one request:

1. `route`: Choice of `connected`, `interior`, `align`, or `unknown`. This selects a durable route preference, not a joystick action or numerical comparison.
2. `verify`: the same choice with reversed option order to expose order bias.
3. `wait_veto`: Noul identifying the explicit unsupported-wait predicate. Initially diagnostic; it never overrides geometric legality.

Only matching route choices with both winning probabilities at least **0.65** are accepted; absent/malformed distributions mean rejection. Acceptance requires the same life/home/level epoch, plan version, and zone, and an observation age no greater than **90 ticks**. The lease expires 90 ticks after its source observation. It only changes choices among near-equal, currently certified routes with at least the leading route's reachable row. Every action is geometrically rechecked. Events distinguish lease acceptance/discard from actual action changes.

One request can be outstanding. Normal cadence is 60 ticks; timeout is 3 seconds, comfortably above archived ~0.2-second latency. Inference never blocks the frame executor. Late/unknown/transient-failed decisions use the leading current certified route, or NOOP if none exists. `BudgetExceeded`, `EpisodeOver`, and `ModelUnavailable` retain contract semantics.

## System-2 role and revision hypothesis

Astra runs asynchronously and returns only enum JSON: open-home order (`nearest`, `left`, `right`) and route preference (`connected`, `interior`, `align`), with strict uncertainty. It receives certificate/outcome evidence, current homes, timer and policy, plus a description of the fixed Jev interface and probability/freshness bridge. It does not design new questions or output code. Arbitrary stand-in text preserves the executable default.

Reconsider the interface when outcomes contradict supposedly safe certificates, home progress stalls, or completion changes the available bays. Calls are separated by at least 60 seconds, with a 120-second timeout based on the archived Astra tail. Policies from another level are discarded. Astra is not on the motor path and is not expected to rescue a frog in a few frames.

## Confirmation and refutation

Confirm if the repeated NUSIZ-2 road and NUSIZ-6 lateral-river deaths disappear, positive-clearance certificates predict survival, homes per life increase, and all training episodes remain error-free. The main target is sustained home completion, not a superficial increase in issued hops.

Refute if copy-aware intervals still certify the same deaths, or if they exclude observed safe support and cause persistent stalls/timeouts. Inspect support shape/phase and source/landing timing separately. Increased survival with no homes would indicate route-horizon/targeting limitations rather than a rearming failure.

## Validation status

Complete harness and proposal were written before live trials.

* Syntax compilation passed. An 800-tick synthetic context replay using public archived observations, first-option/missing-confidence Jev stand-ins and garbage Astra text completed with 800 legal per-frame actions and no harness exception. A final repeat after the geometry refinement also passed, including injected `TimeoutError` and `RuntimeError` model failures. These are contract smoke tests, not emulator score trials.
* Direct geometry checks reject the seed-109 road collision witnesses at ticks 24 and 372, accept its supported row-7 position at tick 630, and reject the final water-gap position at tick 640. The successful seed-107 row-7 arrival at tick 365 is accepted.
* One observation contradicted the initial centre-inside-carrier support rule: seed107 tick593 has x147 on a NUSIZ-1 copy spanning [140,148], yet survives onward to row10/home. The final harness therefore uses sprite overlap (two pixels for a new landing, one for already-observed source/ride support), not centre containment. It still rejects the fatal NUSIZ-6 gap. NOOP prediction also carries the frog for the full wait, unlike a hop whose destination carry starts after input sampling.
* Search benchmarking on sampled archive states took roughly 0.1–11.5 ms before the support-rule refinement, below one nominal 16.7-ms tick on these samples; this is not a guaranteed runtime bound.
* Live `metaharness try` attempts on training seeds 107 and 109 both failed before gameplay because the CLI could not find `TYPESAFE_API_KEY` from the required worktree directory. A standard environment-loader retry also stopped before gameplay because `python-dotenv` is not installed. No live score, survival improvement, or model latency measurement is claimed. No credentials were printed and no other candidate was edited.

Full training evaluation must establish whether the corrected feasibility model translates into more homes and longer survival. The safety geometry and search remain hypotheses, not an exact emulator.
