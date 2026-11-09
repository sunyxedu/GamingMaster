# Candidate 0003 — Jev approves visible timestamp batches

## Parent and isolated hypothesis

Parent: `../0000-tier2_jev_only/` (mean training score 7,564.804).

**Hypothesis:** the parent's reactive *action abstraction*, rather than the absence of Astra, explains its worst failures. Choosing one untimestamped lane event after a 175–263 ms model call cannot reliably serve ±18 ms judgements, simultaneous chords, or long notes. Keeping its synchronous Jev-only `observe -> one Choice -> act` architecture, replace the one-event action vocabulary with approval of a code-compiled, timestamped batch covering the currently visible notes. The game's own documented event queue can execute that approved action while the next Jev call is pending.

This is one change to the answer-to-action bridge, not a new asynchronous controller or an independent autoplay loop. There are no Astra calls, no background inference, no hidden-chart access, and no seed-specific rules. The full observation remains part of Jev's state. Every posted batch needs a fresh Jev approval of that specific proposal. Code implements the exact, already-studied timing and LN rules.

## Archive evidence

There is no `lessons.md` in this archive.

* `../0000-tier2_jev_only/trace_seed1003.jsonl`, line 1: at clock 0 the observation already exposes a chord in lanes **4 and 7 at 668 ms**, then three LN heads at 748 ms. These targets need not be discovered at the last moment.
* The same trace, lines 18, 30, 42, 55, 68 and 84: **all six model answers are `noop`**. Their latencies are 257.253, 177.769, 174.507, 197.404, 191.967 and 232.145 ms. In particular, the request at line 43 sees clock **612 ms**, but line 55 returns `noop` with probability **0.84**, after 197.404 ms. It is followed by the next request at clock **810 ms** (line 56), long after the first chord's Marvelous window. Line 85 ends at **1242 ms**, score **0**, health **0**, with **17 Miss**, **17 automatic judgements**, and **0 player timing samples**.
* `../0000-tier2_jev_only/trace_seed1008.jsonl`, lines 18/31/44/57/69/82 likewise show six `noop` answers, each 186–263 ms. Line 83 ends at **1206 ms** with **17 Miss**, no player timing samples, and score **0**.
* The LN-heavy seed also demonstrates why merely encouraging more immediate presses is insufficient: `trace_seed1002.jsonl`, line 63 posts `press 1` at tick 50; line 74 reports **"press lane 1 was 216 ms early for note 5"**. Line 97 reports **"press lane 2 was 1255 ms early for note 11"**. The final record (line 179) has **17 Miss**, **15 automatic judgements**, and the two player errors average **-158 ms**. Replayed untimestamped actions can target subsequent objects at the wrong times.

The public adaptor documentation explicitly permits `events <json>` batches. That existing mechanism is the treatment under test. Its statement that duplicate events are ignored does **not** establish that replayed future batches are idempotent; the first live test below disproved that stronger interpretation. The bridge therefore uses one-shot batch delivery. Rate and timing offset are already incorporated in the observed target times; the harness does not apply them twice.

## What changed

1. Expand new visible pending notes into press events at `start_ms`; link each LN head with a release at its observed `end_ms`. A held LN contributes only its release. Taps never get synthetic releases. Keep simultaneous times identical and order releases before presses at equal timestamps.
2. Give Jev one Choice over `queue_visible`, `wait`, and `unknown`, instead of the parent's raw lane-action Choice. Explain explicitly that queueing a future event **does not press early**.
3. Translate approval into the documented event-batch string. Track submitted `(note id, part, lane, start, end)` identities so overlapping lookahead observations do not generate new copies. Deliver each batch for **one tick**, then immediately post `noop` before another model call: a repeated future batch can otherwise enqueue duplicate presses against subsequent notes.

Freshness validation, probability checks, and the tick-aligned one-shot handoff are bridge safety mechanics. No new model role or scheduling thread is introduced. The clearing `noop` is a transport action, not a new Jev decision; the evaluator's overall Jev-to-action fraction will therefore be below 100%, although every nonempty batch is freshly approved.

## System-1 interface and control

### State schema

* `observation`: the complete current public observation, parsed from JSON if necessary. It contains visible note identities, lanes, targets, states, holds, judgement windows and feedback, but not the hidden chart.
* `schedule.revision` and `schedule.observed_clock_ms`: identify the proposal and its source observation.
* `schedule.status`: code-derived `ready` or `empty`; unsupported interfaces are handled by waiting, without guessing an action.
* `schedule.events`: only new timestamped events generated from that visible snapshot.
* `schedule.semantics`: the queue meaning, LN dependency, and fresh-state revalidation rule. Jev is not asked to calculate timestamps or count objects.

### Single question

`act` asks whether to submit this schedule. `queue_visible` means approve the new known events now; `wait` means no new events are needed; `unknown` means evidence contradicts the expected interface/semantics. Future note times alone are explicitly **not** a reason to wait. This preserves one synchronous typed judgment per decision, with no Astra or independent recognizer.

### Thresholds, freshness and fallback

* A batch requires `answer.value == "queue_visible"` and `answer.probabilities["queue_visible"] >= 0.60`. Confidence is logged, not substituted for probability; missing probabilities do not become certainty. This low-consequence approval is also constrained by exact code validation.
* Bind approval to the source chart identity and its immutable note-event keys. Re-observe after inference and rebuild eligible events; only keys in the approved snapshot may execute. New notes discovered while the call was pending need another Jev answer. Judged/missed notes are discarded.
* Future targets keep their exact timestamps. An overdue but still-pending event can be salvaged at the next tick only if that execution time is inside its observed Miss window (release multiplier included); there is no backdating. An LN release is eligible only with its pending authorized/submitted head or an observed held state.
* Valid `wait`, `unknown`, low probability, missing/garbage answers, or an empty validated batch map to `noop`, leaving earlier queued commitments intact. A timeout/transient model failure retains the last action (now `noop` after the one-shot handoff), waits a tick, then retries. The timeout remains **5 seconds**, not below the measured latency tail. A multi-second outage can exhaust lookahead; that remains a documented limitation of the deliberately synchronous parent architecture.
* Budget exhaustion / episode closure propagates as a normal end; `ModelUnavailable` propagates. Unsupported observations keep waiting until the actual episode ends rather than returning at an arbitrary step count. No work is spawned.
* `jev_batch_action` events record proposal revision, source/action clocks, selected answer, probability, confidence, proposed/posted event counts, and the latest action error. These expose actual answer-to-action consequences rather than inferring usefulness from call counts.

## Astra policy

Preserve the parent's **zero-System-2** architecture. The study already established the rules, and measured Astra latency exceeds the entirety of these worst episodes. There is no runtime Astra revision in this targeted refinement. A subsequent proposal would be justified in adding an off-path Astra interface revision only if live traces showed persistent `unknown`/`wait` votes on verified ready batches or systematic semantic errors despite timely queued execution; lateness alone is not evidence for adding a slower model.

## Confirmation / refutation and validation plan

Confirm if the formerly zero-score dense/LN seeds queue their first visible batch before the first note, produce zero-error chord/LN judgements, survive substantially longer, and score far above the parent **without** Astra or background execution. The strongest result would be full completion with predominantly/all Marvelous judgements. Inspect first-call latency, `jev_batch_action`, duplicate/late action errors, automatic judgement counts and actual timing errors.

Refute if Jev still waits on ready schedules, batching is rejected by the documented interface, queued tails are not preserved, or known timestamps still yield late judgements. High stale-tick fractions may persist: the loop is still synchronous. The prediction is that stale **posted-action** ticks cease to imply stale **scheduled note** execution; reducing the evaluator's stale-tick statistic is not a separate treatment in this candidate.

The harness and this proposal were written before running live tries. No held-out seeds or game source were inspected.

## Live validation and the one bridge repair

### Initial probe: batch enqueue is not idempotent

`tries/seed1003-195710/trace_seed1003.jsonl` tested the initial bridge. Jev immediately approved the 48-event first visible batch with probability **1.0**, latency **224.583 ms**; it was posted at tick 13. The first batch contained correct target times, but leaving it as the repeated last action enqueued copies during the following synchronous model call. Line 65's next request shows note 6 hit Marvelous at **829 ms**, then the following same-lane note 8 hit **Good at -80 ms**, and note 11 hit **Miss at -160 ms**, all at the repeated event's **829 ms** timestamp. The run ended at **1872 ms**, score **50,464.807**, with no harness error.

Thus the docs' duplicate-event wording was insufficient to assume a future batch could be left in the repeating last-action register. The sole repair was to the same batch transport: align to a tick boundary, post the approved batch, wait for its one consuming tick, and immediately post `noop` before inference resumes. This is not a second strategy or independent scheduler; it makes the proposed macro action execute once. The initial probe is retained rather than hidden.

### Corrected harness: all training seeds, no further code changes

The corrected seed-1003 rerun and the remaining eleven training seeds used the same final harness. These are **local `metaharness try` measurements**, not the official selection evaluation or unseen-seed results.

| Seed | Trial directory under `tries/` | Score | Marvelous / total | S1 calls |
|---|---|---:|---:|---:|
| 1001 | `seed1001-200741` | 1,000,000 | 77 / 77 | 61 |
| 1002 | `seed1002-200754` | 1,000,000 | 85 / 85 | 64 |
| 1003 | `seed1003-200623` | 1,000,000 | 198 / 198 | 37 |
| 1004 | `seed1004-200807` | 1,000,000 | 107 / 107 | 53 |
| 1005 | `seed1005-200818` | 1,000,000 | 37 / 37 | 91 |
| 1006 | `seed1006-200835` | 1,000,000 | 74 / 74 | 52 |
| 1007 | `seed1007-200846` | 1,000,000 | 100 / 100 | 81 |
| 1008 | `seed1008-200903` | 1,000,000 | 135 / 135 | 34 |
| 1009 | `seed1009-200911` | 1,000,000 | 115 / 115 | 68 |
| 1010 | `seed1010-200925` | 1,000,000 | 36 / 36 | 80 |
| 1011 | `seed1011-200940` | 1,000,000 | 81 / 81 | 56 |
| 1012 | `seed1012-200952` | 1,000,000 | 86 / 86 | 74 |

Across these trials:

* Mean score **1,000,000**, up from the parent's **7,564.804**; all 12 ended `game_done`, with no episode errors.
* All **1,131 / 1,131** judgements were Marvelous, every episode had **100% accuracy**, zero automatic judgements, and maximum absolute timing error **0 ms**. No action errors appeared in the bridge's logged revalidation checkpoints.
* **751 S1 calls**, **0 S2 calls**. Pooled successful Jev latency median **166.119 ms**, maximum **546.099 ms**. The 2-second visible lookahead absorbed this tail without moving the target event times.
* **475 nonempty batches**, each freshly approved by Jev, posted 1,131 events. Clearing noops make the overall fresh-Jev-to-action statistic lower than 100%; they must not be presented as extra independent model decisions.
* Weighted stale-tick fraction **85.400%** remains high, as predicted for the unchanged synchronous architecture. Nevertheless all queued note events were exact. This confirms the distinction between a stale last-action register and stale execution timing.

`test_harness.py` adds five local bridge tests using only the parent's public initial observation and hand-written stand-ins, not game implementation code. They passed: visible-only chord/LN expansion and deduplication; no release for an expired unpressed LN head; one-shot delivery with zero-, one-, and ten-tick model latency; garbage/missing/invalid/NaN answers and transient TimeoutError/RuntimeError handling; and JSON observation parsing. These are local robustness tests, not a claim that the evaluator's separate preflight has already run.

### Lesson for the next proposer

The score hypothesis is strongly supported on training seeds: the existing event queue removes model latency from the judgement path without adding Astra or concurrency. **Never leave a future event batch as the repeated action.** Use one-tick submission followed by noop, even if interface prose says duplicates are ignored. A later stale-tick optimization would be a separate architectural experiment, not part of this refinement. Long inference outages, scheduling stalls spanning multiple ticks during the handoff, and zero-lead external charts remain limitations; the successful training trials do not prove immunity to those cases.
