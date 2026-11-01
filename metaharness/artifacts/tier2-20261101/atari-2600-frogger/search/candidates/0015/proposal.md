# Candidate 0015 — Prefer clearance within bounded route regret

## Parent and scope

Parent: **0014**, retained intact apart from the route selector in `pick()`. The time-layered search, geometry, transit/stage/recover protocol, rearmed motor, Astra review path, Jev interface, action scheduling, and fallbacks are unchanged. The only behavioral change is that *all* protocols (not only `STAGE`/`wide`) now choose the largest whole-route bottleneck margin among same-stage candidates no more than **5.0 route-value points** behind the best-valued candidate.

## Hypothesis and trace evidence

0014's route search found viable crossings but its default `TRANSIT`/`RECOVER` executor mostly chose the route with the best weighted value, where path cost can dominate a small clearance bonus. The archive's worst 0014 episode, seed 101 (score **1989**, below the candidate's 2130.833 mean), has route certificates repeatedly close to contact:

- At tick **11094**, `gate_failure` records an attempted final `UP` from row 10, x=44, with the plan margin **0.0**; the previous observation identifies a non-hazard turtle carrier. A later crossing plan at tick **12875** reaches the home entry from row 10, x=79 with margin **1.0**, then fails at tick **12878**.
- At tick **15367**, a river failure follows the plan logged at tick **15363**: `origin=[7,52]`, `action="UP"`, `endpoint=11`, `complete=true`, `margin=0.5`. The last-frame ring shows the frog on row 7 at x=51–53 before that UP and row 8 at x=53 when the life loss is detected. The plan was feasible under its model but used a narrow support clearance.
- Seed 109 has the same pattern despite being 0014's best episode (score **2590**): river gate failures at ticks **14173** and **14433** have plan margins **0.5**, and its home failure at **11574** has margin **0.0**. Other 0014 failures occur with margins 0–1.5.

Positive margin is not a safety guarantee; lessons 0006–0013 already refuted global margin inflation and universal clearance pads. This proposal therefore does **not** increase the collision pad or veto low-margin routes. It tests a narrower claim: where the route-value objective considers an alternative nearly equivalent (within five points and the same completion class), selecting the higher *minimum predicted margin across the itinerary* will avoid some marginal carrier/home-entry commits without materially slowing progress. The existing stage protocol already uses this comparison; 0015 makes that tie-break the default for `TRANSIT` and `RECOVER` too.

## Exact change

In `pick()`, start with the highest-value route as the anchor. Form `near` from candidates with the same `complete` status and `value >= anchor.value - 5.0`. Choose `max(near, key=(margin, value))`, then pass the action through the unchanged live guard. The selected plan is annotated with the anchor action/margin, actual value regret, and whether a clearance override survived the guard, so evaluation traces can establish that this branch actually changed the choice. Thus:

- maximum extra route regret is five value points (about 2.5 search-cost units before other terms), not an unbounded preference for scenic safe routes;
- it compares existing whole-route bottleneck margins, including the final home-entry window, rather than adding a new model-derived safety claim;
- if no similarly valued alternative exists, the original top-value route remains selected;
- invalid routes still fail the same latest-state guard and use the existing executable fallback.

There are no changes to Jev questions, Astra prompts or frequency, route geometry, action timings, or object assumptions. That keeps this a single-hypothesis refinement.

## System-1 state, questions, and bridge (inherited from 0014)

Every request supplies a strict-JSON object with the game objective, current phase (`road_launch`, `bank_transit`, `river`, or `dock`), current policy enums, open home facts (frontier/nearest/current commitment/failed-entry-this-level), and named Boolean urgency/failure facts. It omits raw RAM and irrelevant object detail; code handles arithmetic, geometry, routes, and action legality.

One batched request contains:

- `itinerary` and order-reversed `itinerary_check`: `Choice` over currently open bays plus `UNKNOWN`, asking for a whole-crossing bay commitment. Agreement and each winning probability **>=0.70** are required. It can change the target only while the frog is still on the road; the commitment then persists until home/death/level change and is constrained by current occupancy.
- `gate` and order-reversed `gate_check`: `Choice` among `TRANSIT`, `STAGE`, `RECOVER`, `UNKNOWN`, selecting a future crossing protocol from named facts. Agreement and each winning probability **>=0.70** are required. It does not directly post an action.
- `review`: `Noul` asking whether the explicit failure evidence warrants Astra review; **>=0.85** can request it.

Answers are accepted only if their life/home/level epoch and policy version still match and the observation age is at most **90 ticks**. Missing or malformed values, reversed-choice disagreement, stale answers, or low probabilities preserve the code fallback. At most one Jev request is outstanding; timeout is **3 seconds**, above the measured ~0.2-second normal Jev latency. The executor continually replans from observations; it never waits for Jev or Astra. Astra asynchronously reviews at most the latest two concrete failures with a 120-second timeout and bounded enum parser. Invalid Astra text is ignored. Model errors normal to the contract are caught, while lifecycle exceptions propagate.

For 0015's isolated test, a Jev answer still controls only the parent's itinerary/protocol enums. It does not control clearance selection: the new tie-break is deterministic code operating on the routes already produced by search.

## Fallback and contract behavior

The controller posts only `NOOP`, `UP`, `RIGHT`, `LEFT`, or `DOWN`. It retains the parent four-tick maximum press, observed-movement release, explicit neutral rearm, and one-hop latest-state guard. When a model answer is absent, rejected, late, or low-confidence, code retains its upstream open-bay target, recomputes a route, and uses a currently supported escape or neutral action. Malformed observation/model text is handled by the inherited defensive parsers. The episode loop runs until game completion or `ctx.done`, with no private step limit.

## Confirmation / refutation

Confirm if across the six training seeds 0015 reduces carrier and final-entry deaths per life (especially failures after margins 0–1.5) and increases homes/score or preserves them without a meaningful increase in time-to-home or no-route waits. Trace-level confirmation requires seeing `clearance_override=true`, a higher selected-plan bottleneck margin, and lower actual carrier/entry loss after those plans.

Refute if the alternative is rarely selected, if higher predicted bottleneck margin does not reduce these failures, if selected high-margin routes lose substantial time/score or introduce no-route stalls, or if outcomes remain unchanged. In particular, a mere change in aggregate stale-tick percentage does not test this hypothesis. The evaluator's training scores and the gate-failure/crossing traces decide; the one-minute smoke below is only a plumbing check, not efficacy evidence.

## Validation

`py_compile` passed. A one-minute live Jev smoke on training seed 101 ended at the deadline with score 528 after 3,599 ticks (38 S1 calls, no S2 calls, no error). This short run checks executable control and call plumbing only; it does not establish that the clearance tie-break improves survival or score.
