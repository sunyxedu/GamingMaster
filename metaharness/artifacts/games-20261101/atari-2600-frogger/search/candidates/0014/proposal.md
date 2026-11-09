# 0014 — Treat the median and home entry as transit gates

Parent: **0009**, structural rewrite. Also inspected evaluated best **0010**, its worst seed109 and bank-death seed127, and the working time-expanded search in **0006**. No unfinished sibling was read.

## Hypothesis and trace evidence

The remaining failures are not solved by globally inflating collision margins (0011 already refuted that trade). The route planner is decomposed at the wrong places: the median is treated as a terminal safe haven, and home entry is treated as just another UP. Those are precisely the two places where best-candidate lives disappear.

* Parent0009 seed113 dies at ticks **6106 and7850** after final-UP HOME certificates with **margin3.0**, at frog x46 and78. A generous certificate is not evidence that a home commit is safe. Later deaths at11648 and11968 follow LEFT attempts from road x8 toward x0, while the observed frog remains x8; an executable route also needs reachable motor coordinates.
* Best0010 seed109 has four entry failures at **4270,4900,5552,6192**, all for the last open **HOME_0** at level2. The plans repeatedly press UP around x11 after long rides to the right edge and wraparound. The same bay/x11 eventually succeeds at7144. That is evidence against an unconditional bad-coordinate rule; the approach, phase, and unknown home hazard matter. Merely classifying the failures did not stop the repeated commit.
* Best0010 seed127 dies on the median at **6849,7015,7401,9467**. At6849 the last plan is LEFT from row5/x80 after first arriving on the median at6833; at7015 it is RIGHT from row5/x104 after arriving at6993. At9467 the guard changed an intended UP to NOOP. The public median field still says safe_pause=true. A route to that field is therefore not a sufficient safe endpoint at high levels.
* Parent0009's search also applies its short transition forecast repeatedly without advancing a node's absolute time. Keep its proven asynchronous motor, but use the genuinely time-layered search structure demonstrated in0006 rather than certify a multi-hop path against repeated time-zero geometry.

**Test:** plan the road crossing through the median into a supported river corridor, and choose the whole-frog upstream itinerary before entering the river. Treat docking as a separate supported, settled commit. This should reduce bank exposure and avoid repeatedly leaving an up-current last bay that requires a wraparound approach. It should preserve reachability without global margin growth.

## Structural changes and division of labour

1. **Code:** precompute time-layered collision/support arrays, separated copies and empirically supported shape163/170 tails; retain eight-pixel frog/body arithmetic and neutral-rearmed input. Road search now ends at **river row7**, not median5. Its cost includes alignment with the committed home corridor. At level4+, or after a witnessed bank contradiction, row5 is a transit-only link: no lateral/NOOP search waypoint there. A latest-state guard can choose an immediate supported exit or safe retreat rather than blindly waiting.
2. **Code:** keep world positions inside the empirically reachable interior. Docking checks the current carrier as well as the bay, uses the interior entry window, and waits for top-row/lateral movement to settle. Failure records preserve recent public frame summaries and source contact, rather than automatically teaching that the carrier was water. No global padding escalation or permanent blanket carrier ban.
3. **Jev:** select a **whole-crossing itinerary while still on the road**, plus an executable **TRANSIT/STAGE/RECOVER protocol**. It no longer repeatedly nudges a nearest home label from a short-lived river route list. Its selected home affects road alignment and is retained through the river crossing. Recovery changes the approach, not the collision constants.
4. **Astra:** asynchronously review concrete gate failures and select a small protocol specification: upstream/nearest itinerary, transit/wide launch, settle/different-carrier docking. This specification is explicitly supplied to Jev's subsequent questions and changes code's control protocol. Invalid text is ignored. It does not generate code or per-frame actions.

## System-1 schema and answer bridge

One strict-JSON state contains game objective and field meanings; phase (`road_launch`, `bank_transit`, `river`, `dock`); Astra protocol enums; open home candidates with code-computed `upstream_frontier`, `nearest_now`, `current_commitment`, and `failed_entry_this_level`; and named facts for complete launch availability, timer urgency, higher-level bank threat, recent entry/bank/carrier failure and repeated entry failure. There is no raw RAM, numerical comparison, hidden context, or requested route algorithm.

Batched questions:

* `itinerary` and `itinerary_check`: reversed Choices over open bays plus UNKNOWN. Agreement and both winning probabilities **>=0.70** are required. A fresh result can commit a bay only while the frog remains on the road. The commitment then persists to a home/death/level change and is always constrained by live occupancy.
* `gate` and `gate_check`: reversed Choices over TRANSIT, STAGE, RECOVER, UNKNOWN. Agreement at **>=0.70** chooses the routing protocol. TRANSIT traverses the bank; STAGE prefers wider near-equivalent launches without declaring the bank safe; RECOVER, with matching evidence revision, temporarily avoids a specifically failed top-carrier approach.
* `review`: Noul for explicit repeated-entry/bank contradiction. **>=0.85** can request an eligible Astra review; it is not permission to change geometry.

Responses require matching life/home/level epoch, policy version, and age **<=90 ticks**. Missing probabilities, disagreement, malformed output or absent confidence never become certainty. At most one request is outstanding; timeout **3s** is above the measured ~0.2s normal latency. Accepted choices and actual protocol changes are logged. Inference never blocks the motor.

## Astra revision and fallbacks

After a concrete failure and initial warmup, Astra receives at most the latest two failed protocols, before/after public states and a short frame ring. Calls are separated by3600 ticks, timeout120s. A bounded JSON parser accepts only known enums; policy version changes invalidate old Jev results. The prompt explicitly explains the fixed Jev schema, reversed-Choice thresholds and code-owned legality bridge; it does not ask Astra to invent a new interface without its contract.

Without a valid model answer: code commits the upstream open bay, searches a current transit route, executes only its first rechecked hop, or uses a supported one-hop escape/hold. Every press is held only until observed movement or four ticks, followed by explicit NOOP release. Death/home animation stays neutral. Ordinary transient model errors are caught; lifecycle exceptions retain normal semantics. Run until context/game completion, not a private step limit. Imports are standard library plus metaharness only.

## Confirmation / refutation

Confirm with higher homes per life, fewer row5 deaths, shorter median dwell, and disappearance of repeated same-bay/same-approach entry deaths without widespread no-route stalls. Itinerary adoption must visibly change the crossing target; gate protocol changes must be logged. Refute if transit-only search cannot reach river corridors, harder-level road exposure replaces bank deaths, upstream commitment traps the frog, or settled interior docking still fails repeatedly. Stale-tick fraction alone is not the test.

## Delivery and validation

A complete harness and this proposal were written before live trials. A defensive public-observation replay with first-option, malformed and timeout stand-ins ran to completion in all three cases; it posted only legal actions and raised no parser/contract exception.

A live Jev-only trial on training seed109 initially reached score1179, 15 homes, level4, one death at the 4-minute smoke deadline, with no errors. Profiling exposed an avoidable 40 ms search cost from building all five river rows during a road launch; the harness was tightened to build only rows relevant to the current phase and to use slice fills. A 7-minute run then reached **1903 score, 23 homes, level5, 5 deaths**, ended `game_done`, had **10.5% stale ticks**, and had zero failed model calls or harness errors. Its five gate failures were three repeated/initial dock failures and two median failures late in levels 4–5; importantly, it continued through 23 homes rather than stalling. This is one training-seed smoke result, not a six-seed leaderboard claim. In response to those remaining witnesses, the submitted code additionally forbids downward re-entry to the median at strict levels and rejects a recently failed bay approach for a short recovery window; the final multi-seed evaluation should test whether that reduces the last two failure classes without causing no-route stalls.
