# 0006 — Round-local perception repair, not another direction voter

**Mode:** rewrite. **Parent:** `0002-tier2_idle`. Comparison/evidence: evaluated candidates `0004` and `0005`. No sibling candidate was read or modified.

## Hypothesis and concrete evidence

The idle parent establishes that neutral input is only a temporary brake: seed 0 reaches tick 1000 with score 0 and all four lives lost. Retain its non-spinning `wait_for_tick` clock discipline, not its inactivity.

The best controller has a more specific failure than inadequate survival strategy. In `0005/trace_seed0.jsonl`:

* Tick 220: score 850, all **21 raw cube colours equal 26**, no unfinished cubes.
* Ticks 222–259: the whole board flashes synchronously through many palette values and score rises from 950 to 3950 in **31 increments of 100**. `completed_rounds` nevertheless remains zero. The score breakdown calls those increments green-ball awards, although observations show no green ball. This is strong evidence of an unrecognized round-award animation, not 31 separate catches.
* Tick 267: a landing on node 1 earns 25 and changes its raw colour **26→148**. The decoder now marks that successfully changed cube `S`, while the twenty untouched cubes remain `D`.
* Tick 275: the same happens at node 0. From ticks 280–330 the posted actions repeatedly alternate DOWN and UP. At tick 417 only node 2 adds another 25. The final score is 4025 at tick 900, with all lives gone.

Thus fixed global palette roles reverse after the first completed pyramid. Following `needed_landings` literally turns a completion controller into a top-cube oscillator. The archive's reported zero rounds and green-ball composition are not reliable evidence that the cartridge never completed a round. The new harness preserves the evaluator's actual score and does **not** modify or replace its metrics; it logs a separately inferred round epoch.

## Structural change

1. **Code owns a round-local evidence model.** Learn an ordered colour path from sparse local landing transitions. Detect simultaneous whole-board changes, quarantine the award flash, and rebase the start colour after a stable uniform board with Q*bert back at the apex. The manual's one/two-step cycle selects the number of stages; raw observed colour changes assign palette roles afresh each round. Keep decoder disagreement explicit rather than silently trusting stale named labels.
2. **Code owns hop transactions.** A nearest cube in the middle of an arc is not a landing. Public `ram_x`/`ram_y` and fixed cube coordinates establish landing anchors; coordinates are calibrated from archived scoring observations (cube x+9, y=25+28×row). Post a direction briefly, release to NOOP after departure, and replan at arrival. A bounded resynchronization handles missed launches, sticky inputs, or unexpected anchors. Geometry and current safe moves remain mandatory.
3. **Code computes paths and hazards.** The exact 21-node graph, BFS distances, useful colour transitions, recent repeated branches, occupied nodes and possible next enemy nodes produce a small ranked move set. No FIRE, speculative disc exits, emulator access, or seed-specific routes.
4. **Jev judges evidence and bounded alternatives asynchronously.** It no longer receives raw RAM or a full unfiltered observation and does not own the action clock.
5. **Astra audits the policy, not a stale target list.** A short asynchronous response selects a coverage policy and hazard interpretation. Requests are bounded, use a 110-second timeout consistent with archived latency, and never block action posting. Policy is explicitly carried into each Jev request. Malformed text is ignored.

## Jev schema, questions and answer bridge

The state contains `rules`, the current Astra `policy`, `round_evidence` (inferred/reported completion, learned palette roles and three recent local transitions), `position`, the code `fallback`, semantic `alternatives`, and `decoder_disagreement`. Alternative descriptions include danger (`clear`, `adjacent_enemy`, `predicted_intersection`, `occupied`), useful landing versus transit, single/multiple escape choices, and whether a branch was repeatedly visited. Code does arithmetic and coordinate work.

One request batches:

* `route_a` and `route_b`: identical safety/coverage Choice instructions with reversed option ordering. `unknown` explicitly defers to code.
* `perception_conflict`: Noul asking whether the evidence contradicts globally fixed decoder colour labels. A probability ≥0.8 schedules an Astra evidence audit; it cannot overwrite geometry or palette observations.

Move advice requires agreement and both selected-option probabilities ≥0.62. Metadata bind each response to tick, epoch, node and policy version. It expires after ten ticks or any epoch/node/policy change, and fresh code risk must be no worse than the fallback. Logs distinguish a genuinely changed hop from model calls that did not affect control. Requests have a 1.5-second timeout; only one of each model can be outstanding. Transient failures leave the deterministic controller running.

## Fallback and falsification

Every clock tick posts a fresh legal action or deliberate NOOP. Flash, fall and unanchored flight states receive NOOP; ordinary progress never waits for a model. Call counts are kept below budget exhaustion. Background tasks are individual bounded model calls, not persistent Python loops.

Confirm the hypothesis if independently logged flash/rebase events occur, post-flash local colour changes again reduce the internal unfinished set, and score materially exceeds the old 4025 plateau across training seeds. Compare raw scoring transitions, life loss, inferred epochs and adaptor counters separately. Refute it if the supposed flash is not a round transition, learned colour roles do not predict subsequent useful landings, the controller remains trapped on the top cubes, or landing anchors stall/lose lives. Later-level two-stage/reversible cycles and enemy forecasting remain additional hypotheses, not claims established by the first-round trace.

Astra should revise its coverage/risk policy when repeated branches persist without local palette progress, after life losses, or when Jev identifies new perception contradictions. Its short schema constrains revisions to validated controller choices; it cannot generate executable code or authorize an unsafe edge.

## Validation

Initial complete implementation and proposal were written before running real-time trials.

**Training seed 0, live Jev, Astra disabled, 120-second deadline:** score **18,550**, 1,799 ticks, **one stale tick**, 205 Jev calls, zero failed calls, no episode error, one life remaining; ended by the trial deadline, not game over. Five whole-board award/rebase sequences were observed, each yielding exactly 3,100. The second-round palette path was learned as `[26,148]`, and the fifth pyramid correctly established a two-stage path `[148,26,214]`. This supports round-local calibration and also verifies the first two-step colour cycle. The adaptor still reported zero completed rounds and classified 15,500 as green-ball points; these counters must not be used to refute the observed award sequences. This is a short training trial, not the official six-seed evaluation.

After that trial, the Jev request was moved from the just-departed cube to a **conditional projected next landing**. The state names that future node and code-generated options; its metadata already require arriving at that exact node. Live risk and legality are rechecked before consumption. This uses the roughly 0.5-second hop as the inference window instead of discarding an answer about the old cube. The hazard policy selected by Astra now explicitly changes whether a predicted intersection or only current occupation receives the larger penalty. Seed 0's trial predates these two small bridge corrections; a separate seed-1 trial checks the revised bridge. No claim of Astra benefit follows from Astra-disabled trials.


**Revised bridge, training seed 1, live Jev, Astra disabled, 60-second deadline:** score **8,075**, 899 ticks, one stale tick, 104 Jev calls, no failed model calls and no episode error; ended by deadline. There were 3 logged Jev-induced hop changes and 2 inferred round rebases. Remaining lives: 2.0. The revised bridge demonstrably influences actions, but these unequal-duration, different-seed trials are not an ablation and do not establish a score benefit from Jev. A deterministic replay smoke test over 267 public observations also completed with arbitrary non-JSON Astra text and empty Jev answers; this is not a substitute for evaluator pre-flight.
