- The baseline's main bottleneck is call bandwidth: 20–27 one-action Astra calls consumed 1800 seconds and left games mid-turn.
- Batch plans must be guarded by the current decision ID and option set; Dominion attacks and nested trash/gain prompts make an unchecked suffix stale.
- `ctx.progress()` is useful evidence for replanning: rejected actions or an unchanged observation should clear the plan instead of looping.
- Model output is untrusted during preflight; parse only explicit `choose` lines and keep a legal pass/first-option fallback.
- The naive leader spent one slow Astra call per prompt (179 calls, seven deadline endings); a
  harness must reserve inference for strategic replanning and execute routine prompts locally.
- Parent traces showed Chapel trash answers losing IDs; every multi-card response must be rebuilt
  from the current prompt and validated against its min/max bounds.
- Public VP and Province counts are useful race signals, but a Chapel line must retain enough
  Copper to buy cards; the parent passed a Chapel trash prompt with four Coppers available.
- `ctx.progress()` is the post-action failure detector; unchanged observations and rising illegal
  counts should invalidate a cached plan and request fresh planning.
- The backend can leave Astra unresolved past its advertised timeout. A separately bounded future
  wait let both dev episodes finish legally at +9 and +1 VP margin after one unanswered call each.
- Routine choice logic survived all 179 public prompt snapshots from the parent, including
  multi-card prompt bounds; pre-flight alone did not exercise model planning or parsing.
- A board-level regime is a safer model interface than a long command suffix: attacks and nested effects can branch after any action.
- Local acquisition must remain receding-horizon: recompute legal IDs and supply counts on every prompt, especially after Chapel, Sentry, Workshop, or Militia.
- Province-race transitions are a cheap strategic interrupt; they should switch from engine pieces to green cards without waiting for a periodic review.
- A compact JSON plan needs a deterministic fallback regime because preflight Astra output is arbitrary and can be non-JSON.
- Public Witch/Curse events are a useful strategic interrupt: they reveal opponent pressure without leaking hidden zones and justify changing acquisition priorities.
- Preserve at least two Copper-equivalents while Chapel thins; a fixed “trash every Copper” rule can strand the next buy even when VP is temporarily unchanged.
- Regime reviews should be advisory and bounded; a malformed or slow Astra response must leave a complete local policy in control.
- Gardens detection belongs at the board level (Gardens plus Workshop), while Province scarcity should be an independent late-game interrupt.
- Candidate 0004: a tie on a Gardens board does not establish that a full Gardens rush is better. The concrete trace defect is missed late VP: 0001 bought Silver at $4/three Provinces, then passed a spare $2 buy.
- Search complete buy bundles before choosing the first card; exact Gardens thresholds and remaining buys can change the best single purchase. Recompute after each prompt.
- Exact cost/VP/pile arithmetic and estimated future throughput are different layers. Label multiset forecasts as approximate and verify the point model against public VP before using it.
- Keep a timed-out review future tracked. Permit at most one additional request for fresh failure evidence; obsolete occurrence IDs must not enter a new plan, even if that response also contains valid strategic fields.
- Candidate 0001's archived Astra call timed out at 200 s on every seed; its strong score came from the fallback, so model latency can hide which policy is actually being tested.
- Gardens+Workshop is a genuine alternate regime: a generic Gold/Province script bought no Gardens on both Gardens seeds and tied BMU-1 despite excellent Witch/Chapel performance.
- A regime controller must let Province pressure override Gardens building; Workshop gains can accelerate a three-pile end but can also hand BMU-1 the race if card mass is built too long.
- Recovery inference belongs behind evidence from `ctx.progress()` and unchanged observations; local prompt handling should remain safe when Astra is unavailable or returns malformed JSON.
- Candidate 0006's dev trace exposed a structural Gardens failure: second seat ended 24 VP/3 Provinces while BMU-1 reached 42 VP/5 Provinces after the controller kept building.
- Gardens+Workshop should be treated as a provisional regime; public VP and Province counts can justify a one-way switch back to payload/Province racing.
- Seat-aware opening constraints matter: an early Workshop is much harder to repay when BMU-1 has already taken the first turn.
- Keep the local executor and prompt-ID validation unchanged while changing only the regime trigger, so score changes can be attributed to race adaptation.
- Candidate 0008 tests a structural alternative to bundle forecasting: public-multiset rollouts cannot see delayed shuffles or hidden opponent hands, so exact receding-horizon buy ranks plus regime interrupts may be safer.
- Approximate continuation values should not override immediate pile pressure unless the model has evidence from public ownership and current legal buys.
- A model can choose a regime without supplying direct option IDs; this keeps stale nested prompts out of long-lived plans while retaining strategic adaptation.
- Candidate 0009: inspect the executor/model boundary before crediting a strategy label. In 0002/1300, Astra requested Silver/Gold but decisions 1 and 2 bought Chapel; by turn 14 ownership was six Chapels and two Coppers, eventually losing by 75. Legal progress alone did not mean the strategic plan was working.
- Ordered acquisition agendas with TOTAL ownership caps provide a testable contract: show purchases at several coin amounts, record the revision behind each acquisition, and verify the expected gain in public ownership/events after every action.
- The 0009 first dev run completed at +11/+3 with no illegal actions, but both opening worker futures reached the 260-second cutoff without an answer. These results validate only the fallback; model-plan effectiveness remains unproven until an agenda is accepted and executed.
- A synchronous tier-1 planner can use the direct ctx.think path that succeeded in 0000; 0009 tests it with a 240-second timeout, exception handling, eight-call ceiling, and enough remaining time to finish locally. Separate future waits can hide an unresponsive planning path.
- Audit strategic failures as well as protocol failures: an exhausted agenda with affordable options, repeated sub-$3 turns, or inadequate remaining deck money should trigger fresh advice even while decision IDs continue to advance.
- Candidate 0008 refinement: the final experiment retains 0004's opening and compares its stochastic continuation with exact late-game liquidation bundles. In 0004/1101 decision 25, Cellar beat a certain Estate point by only 0.04 forecast margin; distinguish numerical precision from predictive accuracy.
- A direct 0008 dev inference remained blocked well beyond its requested timeout. Restoring a tracked future with an independent 200-second wait completed the repaired games at +11/+3; a requested model timeout alone is not sufficient evidence of a bounded harness.
- The 0008 liquidation dev result (+11/+3) is below parent 0004's +14/+18, although legal completion and cheap-point collection work. Keep that negative evidence: removing approximate forecasts is a testable ablation, not an established improvement, and unanswered review calls do not validate strategic adaptation.
- Candidate 0009 queue finding: request-to-response time can dwarf reported model latency (0002/1300: 363 vs 81 seconds; 0005/1200: 1,325 vs 51). Its synchronous second dev diagnostic also remained pending beyond 900 seconds and was stopped without a score. Final code restores an independent 900-second opening / 300-second later wait, reserves completion time, expires model agendas, and permits one fresh failure-recovery request behind an unresolved call; synchronous invocation alone was not a fix.
- Candidate 0012 tests a public-race interrupt rather than another Astra acquisition recipe: a visible Province/VP deficit can justify liquidating one turn earlier even when the model plan is stale.
- Keep race detection separate from the model plan; slow or malformed Astra output must not suppress an urgent, observable endgame switch.
- A fixed liquidation threshold is a useful ablation target: candidate 0008's exact bundle arithmetic can be retained while changing only when it becomes authoritative.
- Public attack events are evidence for raising Witch/Militia/Moat, but should never be treated as hidden-hand information.
- Candidate 0010 hypothesis: candidate 0001's one 200-second Astra wait timed out in every archived episode, so model latency was not strategic value; make reviews asynchronous and keep local play moving.
- A concrete late-game miss is seed 1400: at two Provinces remaining an affordable Estate in an extra-buy turn was passed, ending 30–30. Add a narrow late green guard while preserving Province priority.
- Direct option IDs from a completed background plan are valid only if the decision ID is unchanged; stale plans may still update regime fields but must not post old IDs.
- A Gardens threshold should be evaluated by actual VP delta from total owned cards, not by Gardens count alone; a nominal Gardens purchase can be worth zero.

- Candidate 0011 tests shuffle timing against 0004: in 0001/1400 decision 24, Silver was bought with three Provinces left and 18 cards still in the draw deck. A stationary ownership sample can credit that purchase several turns too early.
- Public own-deck composition is ownership minus hand, discard and played cards; verify its count against deck_count before sampling. Full-turn rollout checks conserved zones and nonnegative supply across 529 archived buy states without importing the game implementation.
- Search uncertainty can dominate tactical-looking recommendations: 0011's first dev decision at $11/one Buy favored Merchant with 32 samples, but 192 samples favored Province. More search precision and paired player-specific random streams address estimator variance; they do not establish policy accuracy.
- 0011's first dev result was -4/+12, below 0004's recorded +14/+18. Keep this negative evidence alongside the mechanism: legal completion and coherent shuffle simulation are not sufficient evidence of improved play.
- End conditions take effect after the whole turn. Once an ending pile is empty, optimize remaining buys for exact VP, including free Copper crossing a Gardens boundary; ordinary economy rankings can still miss those points.
- Candidate 0011 final dev: +24/0, mean +12, two returned Astra calls per game, no errors/illegal actions/truncation; still below 0004's recorded +16 dev mean. Search took about 42/32 seconds. Compare matched forecasts and outcomes: current VP margin is not the observed outcome of a terminal-margin forecast, even when Astra calls that difference optimism.
- Candidate 0013 tests phase gating: public-zone rollouts should not choose early action attacks while hidden deck order and shuffle delay dominate; reserve them for auditable scoring decisions.
- A spawned Astra future is not asynchronous if review immediately calls `future.result`; model queue latency can still consume the game deadline even when the future is tracked.
- Midgame rollout candidate sets should separate scoring cards from action acquisitions; exchangeable opponent zones are a poor basis for selecting speculative engine pieces.
- Candidate 0015: 0012/1301 bought Duchy at eight Provinces immediately after trashing Estates; self-inflicted VP loss is not opponent scoring pressure. Require actual opponent Province gains and scarcity before a race override.
- Candidate 0015: apply total support ownership caps before every regime branch. In 0012/1301, race ranking bypassed the Militia cap and ignored Astra's later Silver/Gold repair.
- Candidate 0015: Chapel consumes an Action, so five owned Militia are not $10 of independently spendable thinning reserve. Use stable Treasure money to guard Copper removal and audit any resulting shortfall.
- Candidate 0015 retains bounded waits so Astra can actually affect fast local episodes; purely background calls otherwise finish after the game. Current bundle comparisons must be cleared before reviewing a new decision, since stale search rows can induce false model diagnoses.

- Candidate 0014 branches from 0001 to test Action/trash/discard search instead of refining 0011's full-game purchase rollouts. Match each trace action to the preceding observation: the observation embedded in a step is its resulting next prompt.
- A concrete fixed-priority miss: 0001/1401 decision 18 chooses Militia for $4 with one Province left and a six-VP deficit. Ownership minus known zones leaves Copper×4, Gold×2, Silver in the draw deck, so Smithy guarantees at least $5. The attack still has value; this is a tactical opportunity, not proof of a counterfactual win.
- Root action evaluation must anticipate sensible nested choices. Pricing Chapel with the old “trash all eligible Copper” continuation can incorrectly reject playing Chapel even though trashing only an Estate preserves the purchase. Align the simulated acquisition continuation with the executable buying policy as well.
- Current-turn cash forecasts are directly auditable at the first buy prompt. Utility scores and terminal VP are different targets. Two envelope misses or three low-economy turns can trigger Astra recovery even when every action remains legal.
- Candidate 0014's first dev +9/+1 matches parent 0001: no searched action changed the baseline, despite 17 cash checks staying within sampled ranges. Treat this as execution evidence, not a demonstrated scoring gain. One review returned; two independently bounded waits expired.

- Candidate 0018: late terminal collisions admit exact current-turn draw distributions when only one Action remains; drawn Actions are unplayable, avoiding a full action simulator.
- In 0015/1100 decision 26, Militia gives $6 with two Buys; Smithy can cross scoring thresholds. Attack delay is uncertain and should be shown separately from exact cash/VP arithmetic.
- Retain the public draw/discard boundary when enumerating Smithy draws. Only the discard can reshuffle after the existing draw pile is exhausted.
- An exact cash-support miss is stronger execution evidence than a terminal forecast differing from the current VP margin; disable the local method and send that evidence to Astra.
- Candidate 0016: public event logs can prove lower bounds on the opponent's discard, but exchangeable hidden-zone forecasts may still be poorly calibrated; archive-only cash diagnostics showed conditioned MAE 1.58 versus 1.44 unconditioned. Treat this as a falsifiable structural experiment, not automatic evidence of score improvement.
- Candidate 0016: a just-bought Smithy can be impossible in the opponent's current hand when its draw pile is exhausted and the purchase follows a completed turn. Reserve only cards that survive a no-shuffle count proof; forget the ledger at possible shuffle boundaries.
- Candidate 0016: public Treasure plays constrain hidden cleanup slots because BMU automatically plays every Treasure. Sampling those type-restricted slots requires separate conservation checks; otherwise adding known discard cards can silently double-count zones.
- Candidate 0016: forecast audits must use the next opponent turn's public Treasure plays and must clear rows whenever the decision ID changes. Spare buy prompts and current VP margins are separate targets from next-turn cash forecasts.

- Candidate 0017 branches from 0015 to isolate attack scheduling: equal Witch/Militia priorities repeatedly select Militia by option order with live Curses. At 0015/1300 decision 12, both attacks guarantee $4 from public zones, but only Witch delivers a permanent Curse. This is an opportunity, not proof of better terminal margin.
- A terminal Witch with one Action permits a small auditable calculation: enumerate its two draws from ownership minus hand/discard/play, preserve a possible discard reshuffle, and include Merchant first-Silver money. No full-game simulator is needed to forecast this turn's cash.
- Distinguish permanent Curse VP/dilution from temporary Militia disruption. Preserve guaranteed Provinces and compare actual scoring bundles near endings; immediate VP arithmetic still does not price Militia preventing the opponent's next purchase.
- Candidate 0017 audits scheduled attacks against the next same-turn buy cash and opponent Curse count; contradiction disables scheduling and sends evidence to Astra. Re-enabled plans use the same audit. Archive checks matched 248 real terminal cash/Curse transitions and validated 1,893 legal choices.
- Candidate 0016 refinement: named discard membership alone biased cash forecasts because unobserved cleanup cards were still exchangeable Treasures. Reserve anonymous non-Treasure cleanup slots too (also exclude Smithy if none was played). On 143 matched archived turns this reduced MAE from parent 1.443 to 1.293; named-only conditioning was 1.576. This is predictive evidence, not a complete-game result.
- Candidate 0017 dev finished +11/+3, matching parent 0015, with four returned reviews and zero errors/illegal actions/truncations. Its Gardens-only dev fixtures cannot test Witch scheduling; preserve that limitation instead of crediting unchanged scores to the new mechanism. A stale or malformed model reply must not re-enable a scheduler disabled by newer effect evidence.
- Candidate 0018's single dev run matched parent 0015 at +11/+3. Its returned tactical review correctly retained a guaranteed final-Province Militia over risky Smithy; this validates execution, not improved scoring. Unanswered opening/endgame calls still required independent wait bounds.
