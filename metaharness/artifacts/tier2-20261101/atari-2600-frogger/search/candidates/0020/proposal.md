# Candidate 0020 — drift-reserve watchdog for carried frogs

**Parent:** `candidates/0016/`. The route and hazard machinery is informed by the evaluated `0017` crossover, which is the current best reference, but this candidate changes the motor decision rather than adding another semantic overlay.

## Hypothesis

The remaining deaths are often caused by an open-loop wait, not by choosing a geometrically impossible landing. In the best candidate's traces, the last plan before several losses is a `NOOP` while the frog is visibly supported by a safe turtle: seed 127 at tick 16280 has frog x100, the auxiliary actor at x108, and a recorded plan from x93 with margin 3.5; seed 113 has repeated row-8 losses at frog x100 with the actor at x108 despite margins 2.5–6.5; seed 101 loses at x100/104 with the same shape-184 support and active row-8 actor. Other losses occur after the frog drifts to x148 or x152 on a carrier while the previous one-hop plan still looked safe. The route certificate checks a six-frame landing/dwell, but a carrier moves the frog during subsequent neutral frames and an actor can appear or change phase between searches. Thus a positive instantaneous margin is not a safe *dwell* certificate.

Generation 5 already tried larger scalar margins (`0015`) and independent auxiliary/home overlays (`0017`). It did not test a continuous drift budget: predicting the frog's observed carried velocity, reserving an interior band, and refusing a neutral action unless the whole next hold horizon remains safe. A global pad would sacrifice reachability; the new rule is conditional on being on a moving river carrier and has an explicit escape preference.

**Testable hypothesis:** a short-horizon drift-reserve watchdog will prevent a meaningful fraction of late carrier/edge deaths while retaining the portfolio's throughput. It should reduce deaths per home, especially row-8 x~100/actor~108 and x>140 witnesses, without creating a large no-route stall rate.

## What changed

1. Retain the separated-copy route search, matched-home veto, auxiliary actor/spawn overlay, median-track check, complete-home portfolio and neutral rearming from the strong reference. They provide reachability and protect hazards already established by observations.
2. Add an episode-local frog-motion track. While the frog is on rows 6–10 and `moving_with_platform` is true, estimate its actual world-pixel drift from consecutive observations, robustly quantized to the observed carrier speeds. This is deliberately measured from the frog, not inferred from an average of two carriers.
3. Add `drift_reserve`: forecast the frog center over a short 10-frame hold by default (14 frames only if Astra explicitly compiles the less-conservative enum) and measure support at shorter points in the same forecast. An uncertain/lost support forecast means `WATCH`, not an unverified hard veto. `ESCAPE` is reserved for a projected screen-edge loss or same-row actor contact (actor/frog separation at most 25 world pixels over the next short horizon). The controller then chooses a one-frame-verified inward lateral hop, or a safe forward/back hop, before the carrier carries the frog into the edge/contact interval.
4. Put the `edge_watch` and `contact_watch` in the action-selection bridge before returning an ordinary ranked route. They are hard executable guards, independent of Jev/Astra and never disabled by model output. They are not a global padding increase: they apply only to a currently carried frog, use observed frog drift, and return to the normal route portfolio if no one-frame-verified escape exists. During an emergency the auxiliary actor overlay is omitted only for this immediate escape check; carrier landing geometry remains live-checked.
5. Recheck the current observation immediately before posting the selected action. A route may still choose `NOOP` in a stable central carrier corridor, but the controller replans at most every three game frames after a `NOOP`; a stale route does not own a later drifted position.
6. Astra reviews a witnessed carrier/edge failure and can choose only bounded `hold` (`short`/`normal`) and `escape` (`inward`/`forward_first`) enums, alongside the existing itinerary policy enums. Jev judges whether the current reserve is `CLEAR`, `WATCH`, or `ESCAPE`, and chooses a verified protocol (`KEEP`, `INWARD`, `FORWARD`, `BACK`). The code owns all geometry and action legality, so malformed stand-in output cannot become an action.

## System-1 state and questions

Every request carries strict JSON derived from the current observation, not raw RAM: `phase`, `frog_row`, `x_band`, `on_carrier`, `drift_band` (`left`, `stationary`, `right`, or `unknown`), `status` (`CLEAR`, `WATCH`, `ESCAPE`), `actor_min_band` (`contact`, `near`, `far`), `support_future` (`stable`, `uncertain`, `lost`), `edge`, `horizon`, `home_alert`, `complete_route_exists`, `timer_urgent`, the current crossing protocol, and a small recent failure class. It also includes the current epoch. Numeric distances, comparisons, drift estimation and carrier identity are computed in code; Jev is not asked to count pixels or decode flags.

One batched request asks:

* `reserve` (and reversed `reserve_check`): `CLEAR`, `WATCH`, `ESCAPE`, or `UNKNOWN`. `ESCAPE` requires an immediate projected edge or same-row actor threat; `WATCH` means the support forecast is uncertain/lost or drift is meaningful and the next neutral hold should be short; `CLEAR` is only a classification, not permission to ignore code guards.
* `protocol` (and reversed `protocol_check`): `KEEP`, `INWARD`, `FORWARD`, `BACK`, or `UNKNOWN`. It chooses among code-generated, live-verified candidates; it never invents a joystick string.
* `review`: a `Noul` asking whether a newly recorded drift/edge failure should trigger Astra review.

A choice is accepted only when both order-reversed answers agree and each selected-option probability is at least `.70`; missing or malformed probabilities are rejection. Answers must match the life/home/level epoch, policy version, and derived reserve signature `(row,status,x_band,drift_band,actor_min_band,support_future)`, and be at most 70 ticks old. Jev is advisory: hard edge/contact guards and the live carrier landing check always win. While a request is late, code runs the same watchdog; `INWARD`/`FORWARD`/`BACK` can reorder only the already-verified escape options. A fallback `NOOP` is not leased as an emergency—if no verified exit exists, control falls through to the ordinary route portfolio. Requests use a 3-second timeout, every 90 ticks normally and every 24 ticks while a carried frog is in `WATCH`/`ESCAPE`.

## Astra role and scheduling

Astra is a slow protocol reviewer, not a joystick. It receives the compact state, the last two drift witnesses, and observed outcomes. It returns only JSON enums such as `{"hold":"short|normal", "escape":"inward|forward_first"}`. The default is short hold/inward escape. The response is compiled only if it corresponds to the same failure revision; arbitrary prose or stale answers preserve the default. Calls are asynchronous, sparse, and bounded so the real-time motor never waits.

## Fallback and confirmation

The motor continues until `ctx.done`. It sends `NOOP` while a direction pulse is being released, and never repeats a standard-mode direction as a fresh route hop without neutral rearming. A watchdog emergency action is posted immediately when possible. On a model timeout, budget exception, garbage answer, or unavailable semantic opinion, code uses the live field and drift reserve; it does not parse prose as an action. All action candidates pass the same current-observation guard.

The hypothesis is confirmed by fewer `gate_failure` events with `moving_with_platform=true`, fewer losses at x>=140 or near an active actor, higher homes/death and score on the six training seeds, while route counts and no-route intervals remain comparable to `0017`. It is refuted if the watchdog mostly causes water deaths from lateral escapes, stalls at stable carriers, or does not change the same positive-margin drift witnesses. The key diagnostic logs are `drift_watch`, `drift_emergency`, `drift_watchdog_block`, and the post-death `drift_witness`, separated from home-alert and bank failures.

## Contract smoke check

The harness compiled and completed 20- and 80-second virtual-clock episodes on training seed 101 with stand-in System-1/System-2 profiles, without errors. The longer run reached 3,074 score / 35 homes at its cap, handled one System-1 timeout, and safely rejected four unrecognized scripted Astra responses; the shorter run exercised `WATCH`, blocked and accepted edge-rescue paths. These are contract checks, not estimates of live performance.
