# 0007 — Replica-aware safety, neutral rearming, durable Jev preferences

Parents: **0005** and **0003** (generation-2 crossover).

## Hypothesis and evidence

0003's useful mechanism is an intentional direction pulse followed by an observed neutral release. This fixes the original one-point held-UP failure. Its worst evaluated seed, 127, reaches road row 1 by observation tick 37, is dead by tick 51, and ultimately scores 10 without a home. The motor works; raw synchronous Jev choices do not preserve lives. 0005 adds the valuable independent motor clock and predictive geometric search: 1% stale ticks, mean 44.5, and one home on two seeds. Its remaining failure is not a lack of hop throughput.

**Hypothesis:** 0005 mistakes the decoder's estimated NUSIZ hulls for actual solid sprite support, and treats the frog's x anchor as a centre. Consequently, its search confidently validates car collisions and sideways water landings. Fixing this representation, retaining neutral rearming, and using Jev preferences that survive its approximately 0.2-second latency should preserve crossings long enough to complete multiple homes.

Concrete trace evidence:

* `0005/trace_seed109.jsonl`: tick 23 issues UP from x=72 with margin 0.13; tick 24 reaches road row 1, whose nearest vehicle anchor is x=77; tick 26 loses the life. An eight-pixel frog extends to the right of its anchor, unlike the parent's symmetric five-pixel point clearance.
* The same trace: tick 371 issues UP with margin 3.13; tick 372 reaches road row 2 at x=96 while a NUSIZ-2 vehicle anchor is x=64 (offset **32**); tick 374 loses the life. A NUSIZ-2 repeated sprite at offset 32 is outside the decoder's 24-pixel width estimate.
* `0005/trace_seed107.jsonl`: row-7 LEFT hops at ticks 1025, 1242 and 1892 have predicted margins **10.48, 9.52 and 12.52**, but deaths follow at 1029, 1245 and 1895. NUSIZ-6's separated copies explain why a large estimated solid hull is not safe support. Successful row-7 landings occur around offsets 33–36 from the first object, consistent with the next copy, not a filled interval.
* 0005 also issues many directions while frog state is 8 (death animation); the playable observations use state 128. Its six Astra calls are all cancelled before providing a policy. This proposal suppresses futile animation hops without waiting for Astra to save a dying frog.

These claims are from public observations/traces and general TIA NUSIZ semantics, not game source inspection. The shape-core margins remain conservative hypotheses to test in live trials. `lessons.md` confirms rearming and asynchronous execution; it does not establish that the prior geometry is correct.

## Crossover and changes

1. Retain 0003's electrical press/release separation and 0005's per-frame execution. A direction is released after movement is observed or four frames, then at least four neutral frames are consumed. Ten-frame decision spacing is independent of inference latency. No commands are issued during observed non-playable animation.
2. Expand NUSIZ into actual copy offsets and widths, including stretched sprites. Merge touching supports, but never fill water gaps. Road checks use the frog's eight-pixel extent plus a margin. River checks require the central core to have support; lateral hops also validate intermediate positions. Unknown/diving carriers remain excluded; explicitly nonhazardous `river_carrier` objects can support the frog rather than being rejected solely for unfamiliar colour.
3. Rebuild a bounded six-hop predictive search at every action boundary. Validate early and late input sampling and the neutral recovery interval. Replan after every move, rather than executing a long stale sequence. Open-home alignment uses the frog centre, not its anchor.
4. Keep asynchronous Jev, but ask for a maneuver **template** that remains relevant across multiple hops in the same crossing phase. This addresses the parent's repeated discarded one-frame advice. A template only selects among freshly feasible near-best proposals; it cannot authorize unsafe geometry.
5. Keep Astra's asynchronous durable policy role. It selects home ordering, conservative clearance, and progress/wide-support/home-alignment focus. It sees outcomes and can change policy after repeated misses or stalls. Unparseable output retains the executable defaults. No runtime question generation is requested.

## System-1 interface

Every request includes a small named JSON state:

* `rules`: task, actual-sprite geometry, safe landing, motor ownership, template semantics;
* `policy`: Astra's route, safety, and focus;
* `phase`: road crossing, median, or river crossing;
* `facts`: safe-wait, escape-needed, timer urgency, near-home and home-alignment booleans computed by code;
* `proposals_in_progress_order`: legal action, maneuver category, complete-home-path flag, towards-home predicate and named support quality; no coordinate arithmetic is delegated;
* `uncertainty`: unknown crocodiles are not absent; no optional bonus detours.

One batched request asks two reordered Choices over the available `ADVANCE`, `REALIGN`, `RIDE`, `RETREAT`, `UNKNOWN` templates and an independent escape Noul. The duplicate Choices must agree, each with winning probability at least **0.65**. Missing probabilities mean rejection, not confidence. The escape question is diagnostic; geometric escape checks retain authority.

A preference expires after **90 frames**, a death/home/level change, crossing-phase change, or Astra policy revision. Unlike a coordinate action, this semantic template may survive a row change within the same crossing. At execution, it can only select a current candidate within nine search-value units of the leader, after current safety validation. Events log acceptance and actual action changes separately. At most one Jev future is outstanding, with a three-second timeout and nominal 30-frame cadence; none blocks the motor.

## Astra and fallback

Astra receives public snapshot/outcome summaries and the exact fixed Jev interface semantics, returning only validated enum JSON. Its 120-second timeout covers archived slow tails. Reconsider after 60 seconds when failures or stalls supply new evidence. A result from an incompatible level is rejected. The default route is nearest open home, balanced clearance, and progress.

While models are late, code uses the current highest-ranked safe proposal; when none exists, it posts NOOP and observes again. All action posting uses `ctx.act`, all waiting uses `ctx.wait_for_tick`, and all inference uses context calls. Transient inference failures preserve the controller; budget/lifecycle and unavailable-model errors retain normal termination. Background jobs contain only bounded model calls and need no persistent Python loops. Model garbage cannot change legal actions or the geometry table.

## Confirmation / refutation

Confirm if replica-aware filtering rejects the documented fatal car/water offsets, road deaths decline, successful river lateral moves cease entering replica gaps, and multiple distinct homes become feasible. Strong evidence is improved six-seed mean score and home count without errors. Also measure Jev accepted/action-changing preferences and Astra policy completions, not just call counts.

Refute if the new copy/core representation excludes viable routes so often that frogs time out, if predicted-safe water deaths persist, or if home-centre alignment is systematically wrong. Such outcomes require correcting the observation model rather than merely increasing confidence thresholds. Full snapshots around action boundaries and crossing-outcome events support that diagnosis.

## Validation status

Initial complete harness and proposal written before live testing.

### Offline checks completed

* Python compilation and import through the installed framework succeeded. The module uses only standard-library and `metaharness.*` imports.
* Replaying public observation snapshots through the new geometry marks the documented road-row-1 position at seed 109/tick 24 unsafe (margin -5), and the road-row-2 replica collision at tick 372 unsafe (margin -10).
* Projecting the public row-7 snapshots forward at their observed velocity rejects all three formerly fatal LEFT proposals at seed 107/ticks **1025, 1242 and 1892**. These are counterfactual geometric checks, not emulator rollouts.
* Four sampled planning calls took approximately **3–14 ms** each. This is an offline execution-time measurement, not a model-latency measurement or a guarantee of every-frame scheduling.
* A 150-tick fake-context run with malformed Choice values, absent probabilities, and arbitrary Astra text completed without exceptions or illegal actions. Separate 100-tick fake-context checks completed through repeated `TimeoutError`, repeated `RuntimeError`, and UNKNOWN responses with malformed Astra enum fields. These check our bridge; they are not the evaluator's official preflight.
* Level transitions clear fitted velocities so the next level starts from its public movement masks and fresh measurements.

### Live-trial limitation

The supported `metaharness try` command was attempted for training seeds 109 and 101, but model initialization failed **before any episode** with `ModelUnavailable: TYPESAFE_API_KEY missing`. The worktree also does not have the expected project environment loader/configuration available. No credentials were displayed and no inference was attempted outside context calls. These startup failures provide no candidate score, survival, Jev latency, or home-count evidence. No live-performance improvement is claimed; the evaluator must test the geometry hypothesis and its conservative river restrictions.
