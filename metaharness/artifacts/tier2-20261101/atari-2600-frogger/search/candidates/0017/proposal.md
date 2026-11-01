# 0017 — Route portfolios with an independent auxiliary-hazard layer

**Parents:** 0014 and 0006 (crossover). This candidate retains 0014's through-bank launch, upstream itinerary, settled docking and independent pulse/release motor. It restores 0006's useful choice among complete home routes rather than treating a blocked commitment as the only possible destination. No unfinished sibling was inspected.

## Hypothesis and concrete parent evidence

The best parent's remaining positive-margin failures are not all support/phase errors. It models vehicles and carriers, but ignores auxiliary actors and treats the home-crocodile decoder's `unknown` as if there were no usable *correlated alert evidence*. A supported carrier and an open home are necessary, not sufficient, for a safe route.

In 0014's actual lowest-scoring episode, **seed103, score1902**:

* At ticks **4145 and4707**, final UP into bay2 dies despite margins1 and0. Before both commits the public fields have **E3=131 (occupancy3 plus high bit128), E5=3 (bay2 plus one)**. Frog x77/76 lies inside the proposed docking window. This is not evidence for banning that x or every turtle.
* At **14596**, a forecast complete HOME route with margin0.5 enters row8. The previous frame has frog x13, an apparently safe shape184 turtle, and the auxiliary candidate called `fly` at **row8/x2**, C7=128.
* At **14762**, another certified UP dies at row8/x97 while the same auxiliary candidate is at **x86**. At **16002**, LEFT dies at x75 with that actor at **x66**. The carrier remains classified safe in all three cases.

This repeats across the best parent's training traces: row8 deaths at seed101 tick15367 (frog53/actor42), seed109 tick14433 (frog104/actor108), and seed127 tick15356 (frog64/actor48). The manual independently warns of later-level snakes. We call this an **on-lane auxiliary hazard**, not a proven exact ROM snake decoder. All examined final-UP failure witnesses also have the high E3 bit and the matching one-based E5 bay. An unrelated bay alert must not block every home.

Parent0006 earns its score from time-expanded separated-copy/body-aware routes and durable choices among complete endpoints. Its worst seed109 nevertheless dies at ticks2532/3814 after identical top-row landing plans (margin1.25), and at3054 after a final UP (margin1). Its support model does not settle independent actor or home hazards. Parent0014 improves bank exposure and itinerary order but still exhausts lives. Combining their routing mechanisms without adding the omitted hazard layer would preserve that failure.

**Testable hypothesis:** separately tracking and forecasting on-lane auxiliary actors, and temporarily excluding only the home whose mature alert signature matches, will prevent a substantial share of the remaining deaths without global clearance growth. A complete-route portfolio should maintain throughput when the preferred bay is temporarily excluded.

## Changes

1. Keep the parents' separated sprite copies, shape-local widths, bounded time-expanded search, six-frame motor opportunities and neutral rearming. Keep 0014's road endpoint at river row7 and strict higher-level median transit.
2. Add an independent auxiliary track using only public `fly`/named RAM fields. When the candidate is present on a playable row, estimate velocity from recent same-row positions; reset history on teleport, row change or disappearance, retain the last running speed for reappearance, and recognize observed stationary phases. Exclude its swept contact neighborhood in every time layer, separately from carrier support. Reserve the empirically observed spawn mouth while it is absent, rather than assuming an absent actor cannot appear during a hop. A bonus name or `hazard=false` cannot override the observed lane-actor role.
3. Add a matched-home alert predicate: E3 high bit128 **and** E5 equal to the proposed slot plus one. This is a conservative observational veto, not a claim that other flag combinations prove safety. It expires immediately when the public signature no longer matches. No permanent coordinate quarantine or scalar global padding.
4. Restore complete-home alternatives when the commitment is alerted, when an auxiliary actor or its spawn zone constrains the river, after stalled progress, or under timer urgency. A valid Jev portfolio goal then influences ranking rather than opening every alternative by itself. Code ranks complete routes, slightly favors the committed upstream bay, and rechecks the selected route on the latest observation. A replacement bay is selected from actual complete routes, not arbitrary nearby positions. The median's omitted second position track is also forecast as a local moving hazard once bank hazards are expected; the measured public-position mapping is described below.
5. Add SCREEN to the durable protocol vocabulary: Jev may prefer lower lateral/wait exposure among near-equivalent verified routes when an auxiliary actor is present. It cannot disable either hazard exclusion.

## System-1 interface and bridge

Each asynchronous strict-JSON request contains the game objective, phase, validated Astra policy, open-home candidates, and named derived facts. Candidate fields include upstream-frontier, nearest-now, current commitment, failed-entry-this-level, **matched-home-alert**, and **complete-route-now**. Facts include complete-launch availability, bank threat, urgent timer, recent failure class, **auxiliary-on-playable-row**, **auxiliary-on-frog-row**, and **target-has-matched-alert**, and **auxiliary-spawn-reserved**. No raw arithmetic or hidden history is delegated to Jev.

Batched questions:

* `itinerary` / `itinerary_check`: reversed Choices over open homes and UNKNOWN. Prefer the upstream commitment unless it is alerted; in the river, a replacement requires a presently complete unalerted route.
* `gate` / `gate_check`: reversed Choices over TRANSIT, STAGE, RECOVER, SCREEN and UNKNOWN. SCREEN chooses low-exposure alternatives; STAGE favors wider near-equivalent routes; RECOVER changes a failed approach; TRANSIT preserves through-bank motion.
* `review`: Noul for explicit repeated entry/bank contradictions that merit Astra review.

Both Choices must agree with selected-option probability **>=0.70**. Missing probabilities are zero; confidence is not substituted. Answers require the same life/home/level epoch and policy version, age **<=90 ticks**. River portfolio goals expire after180 ticks and are always geometrically revalidated. One outstanding request, timeout3s, compared with the archive's typical0.2s and worst approximately1s. The motor never waits for an answer. Logs distinguish accepted goals/protocols from actual itinerary changes.

Astra reviews concrete failures asynchronously after warmup/cooldown. The fixed interface is explained in its prompt. It returns only validated itinerary/launch/dock enums, not code or raw frame actions. Its policy is sent in every later Jev state. Malformed output preserves the current policy. Calls have120s timeout and at least3600 ticks between reviews. Local hazard exclusions cannot be disabled by Astra.

## Fallback, freshness and lifecycle

Without model advice, choose the newest complete supported transit/home route; otherwise feasible progress or a live-checked one-hop escape/hold. A matched alert can redirect to another complete home. Directions are held until public motion confirmation or four observed ticks, then explicitly released. The next search starts early by the just-measured computation lag (bounded to two frames), retaining a minimum four-frame scheduling interval and two-frame neutral rearming, to align actual actions with the six-frame forecast rather than silently adding search latency to each hop. Death/home animation stays neutral. Unknown or malformed model output cannot become an action. Budget/episode/backend-unavailable exceptions retain their contract semantics; ordinary inference errors are caught. Run until context/game completion, not an internal step cap.

## Confirmation and refutation

Confirm with fewer row8 deaths near auxiliary actors, fewer final-UP deaths with matching home alerts, more homes per life, and portfolio reroutes rather than long no-route stalls. Report score, homes, deaths, veto/reroute witnesses and real latency, not only stale fraction. Refute if the hazard signatures also reject routine successful contacts, if the actor motion forecast is wrong after respawns, or if the portfolio undoes upstream ordering and leaves the last bay unreachable.

## Delivery and validation

A complete `harness.py` and this proposal were written before live trials. Compilation passes. Further validation and measured outcomes will be appended here; no score improvement is claimed yet.


## Diagnostic evidence and refinements

All tests below use the official `metaharness try` runner, live Jev, primary configuration and training seeds only. The first invocation failed before the episode with `No module named metaharness`; setting `PYTHONPATH` to the documented package location allowed running from the required worktree. No backend or game implementation was inspected or patched.

### Initial hazard layer

* `tries/seed103-001631`: **2314**, 28 homes, level6, 5 recorded deaths, 367.1s, `game_done`, error=None. S1=225, S2=0, stale11.9%; Jev median0.202s/max0.750s. This exceeds parent0014's 1902 on this seed, but it is one diagnostic run.
* `tries/seed109-002036`: **2323**, 28 homes, level6, 4 recorded deaths, 353.5s, `game_done`, error=None. S1=209, S2=2, stale14.0%; Jev median0.205s/max0.860s. One Astra answer completed in18.15s; it validated the existing protocol, not a demonstrated score-improving policy change. This run is **below** parent0014's2590 on this seed.
* Both removed the matching-alert final-UP death class. Remaining failures included road timing, two bank contacts per run, and trapped/late-spawn river contacts. No aggregate win is inferred from these two runs.

### Omitted median track

The public raw position layout was checked against **18,820 decoded object positions** across parent traces (0014 seeds103/107 and 0006 seed109), with **zero mismatches** for `raw_ram[row + 11*object_index]`. The decoded object lists omit row5, but the same second-track position at index16 explains the bank witnesses: parent0014 seed107 tick19049 has frog104 and the last median-track position88 four frames earlier; tick21593 has frog104 and track95 two frames earlier. The initial candidate seed103 loses twice at x104 with the second median track at91. Earlier 0010 witnesses also align after observed drift (seed127 bank losses6849,7015,7401,9467). Nearby first-track contacts can survive; this is not a blanket ban around both raw positions.

The refinement forecasts **only the second median track**, with its episode-measured velocity and a local body interval, when level>=4 or an observed bank contradiction exists. It retains transit routing rather than reverting to a median sanctuary. This is an observation-supported conservative hazard model, not an exact ROM decoder claim.

### Reappearance and execution timing

The median-refined `tries/seed103-002547` finished **2302**, 27 homes, level6, 4 recorded deaths, 295.2s, error=None; stale9.5%, Jev median0.182s/max0.514s. Its bank losses disappeared, but it repeatedly waited near a reappearance mouth: at ticks14420,15060,16340 the auxiliary candidate switches from absent (`row128`) to row8/x0 while the frog is at x12–14, followed immediately by death. This supports reserving an *observed spawn zone*, not another global carrier inset or a failed-frog-coordinate quarantine. Actor coordinates also pause, so zero observed motion is modeled as stationary rather than retaining a false constant speed or wrapping the prediction.

The same traces show actual consecutive road UP onsets predominantly **seven frames apart** despite a six-frame forecast (0014 seed103:48 such pairs at seven frames versus17 at six). Searches run while the clock advances. The final motor schedules its next search early by the measured observation-tick lag, capped at two frames; it still verifies current geometry before posting and retains explicit neutral rearming.

A public-observation replay with first-option responses, malformed values/missing probabilities, and injected model timeouts completed without exceptions or illegal actions (`check_contract.py`, `contract_check.json`). This is a robustness check, **not** a game score evaluation. Final-revision live outcomes will be recorded separately below.
