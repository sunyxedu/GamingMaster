# 0004 — Compile the visible chart; reason only about protocol exceptions

## Parent and hypothesis

Parent: `0002-tier2_idle` (structural rewrite). Comparison: `0000-tier2_jev_only`, the current best; `0001-tier2_astra_only` supplies the slow-model latency counterexample. No `lessons.md` was present in this archive when inspected.

**Hypothesis:** the baseline is failing at the decision/execution boundary, not because Quaver pattern recognition needs a stronger model. Exact visible timestamps make local event generation a compiler problem. A model round trip should never gate an input. Compile all newly visible heads and tails into timestamped batches immediately, let the simulator own timing, and keep models out of the normal control loop. Model intervention is reserved for interpreting unexpected protocol error text.

### Concrete archived evidence

* The best baseline scores 7,564.804/1,000,000 on average with 89.85% stale ticks despite 263 Jev calls. Its action Choice contains only plain single-lane actions, so it cannot express the documented lookahead event batches or simultaneous chords.
* `0000-tier2_jev_only/trace_seed1003.jsonl` begins with two visible taps at 668 ms in lanes 4 and 7, followed by three LN heads at 748 ms in lanes 1, 3, and 4. Their tails, already visible, are 1915, 1186, and 1624 ms. Yet all six recorded Jev answers select `noop`. The first call takes 257 ms; subsequent calls take 175–232 ms. State input grows from 8,894 to 11,606 tokens. The episode ends after 69 ticks with score zero and 64 stale ticks.
* The idle parent also ends seed 1003 after 69 ticks with score zero. Waiting correctly follows the clock, but earns nothing.
* Astra-only's seed 1003 call is cancelled when the game ends; no action is posted. The separate latency calibration measures 69 seconds even for a one-word Astra answer. Starting an obligatory planner would forfeit this short map.

## Structural change

1. Read only `ctx.observe()` and the public action contract. No chart files, seed-specific plans, game internals, or hidden remainder are accessed.
2. Validate active lane count, note state, and finite exposed timestamps. Visible `start_ms` and `end_ms` are authoritative: never reapply the chart's already-incorporated rate or offset.
3. Expand each pending tap into one press; each pending LN into a press and release; each held LN into its release. Queue a tail as soon as its parent note is visible, even if the tail is later than the head lookahead. It is exposed information, not an inferred hidden note.
4. Deduplicate by observed note identity and part. Sort simultaneous releases before heads, preserve chords in one batch, and schedule all available future events, not merely the next one.
5. Post one action and wait for the next tick before posting again. With no new notes, post `noop` to replace the last batch. This heartbeat both reduces stale ticks and avoids replaying a batch unnecessarily. The game queue, not Python sleeps or model latency, delivers exact timestamps.
6. If an event is newly observed late, never backdate it: try a still-valid future tick. Normal two-second lookahead should avoid this path entirely.
7. Track protocol errors separately. A background Jev call can diagnose an unfamiliar error; a slow Astra call is conditional on an unresolved, high-confidence diagnosis. The scheduler remains active throughout.

The default policy is the research/study plan compiled into deterministic code. Runtime Astra is not called merely to restate verified rules. It can choose a documented transport repair, not change timing or issue lane actions. No model answer is required for successful ordinary play.

## Jev interface and answer-to-control bridge

**State:** a small named JSON object containing `goal`, the applicable `protocol` (timestamp/duplicate/transport rules), current `plan.transport`, the literal new `error`, `input_mapping`, aggregate `timing_error_summary`, and a version `(chart_hash, transport, error_text)`. It excludes the raw chart and irrelevant history. Exact arithmetic and lane legality stay in code.

**Questions:** two independent `Choice`s classify the same error with reversed option order: `replay` (benign duplicate/already resolved/no target), `late` (expired timestamp), `encoding` (explicit whole-batch parse/serialization failure), or `unknown`. The second question checks option-order sensitivity rather than chaining a dependent inference.

**Thresholds:** both choices must agree, each winning probability must be at least 0.85, and each reported confidence at least 0.60. Missing confidence is not certainty. A malformed, timed-out, conflicting, or low-confidence response leaves the executable compiler unchanged. Jev timeout is 1.5 seconds, safely beyond the archived warm latency tail, and there is at most one outstanding diagnostic.

**Effects:** `replay` and `late` retain the future-only scheduler; `encoding` switches between the two documented transports (`events <JSON>` and a direct event list), then reconstructs still-pending/held parts from fresh observations. `unknown` may trigger one background Astra call asking for one of `string`, `list`, or `keep`. Astra cannot generate Python or modify offsets. Garbage Astra text is ignored. Trace events distinguish accepted/discarded diagnoses and actual transport changes; call counts are not claimed as evidence of useful collaboration.

**Freshness:** accept a response only if the chart hash, transport and literal error are still current. A no-longer-present error invalidates its diagnosis. Never apply a delayed answer to individual notes. Each distinct error is queried at most once. Model failures cannot block event compilation; budget and model-unavailability signals are allowed to propagate as specified by the context contract.

**When should Astra revise the interface?** Only when observation-backed errors contradict the known transport interpretation and Jev agrees that its available classifications do not explain the error. On these short charts, it is expected that ordinary episodes need no runtime Astra at all. The current experiment constrains repair to documented transports; persistent semantic failures should inform the next research iteration rather than inviting unverified within-song timing changes.

## Confirmation and falsification

Confirm with training-seed tries: full map completion, predominantly/all Marvelous judgements, zero auto-expiry, near-zero signed timing error, intact LN releases and chords, substantially fewer stale ticks, and no model calls unless a real protocol fault occurs. Verify that future batches survive subsequent `noop`s and that queued tails do not require a second observation.

Refute if batches are overwritten before intake, timestamp queueing is not persistent, tails cannot be queued with heads, duplicate replay changes outcomes, or the public timestamp semantics produce nonzero errors. Those are protocol hypotheses to test through observations, not assumptions to bury. A successful result should generalize across modes, rates, offsets and pattern families because no seed-specific chart data is stored.

## Validation

The initial harness and proposal were written before any live tries. All twelve public training seeds were then run through `metaharness try`, in wall-clock mode, with the normal local simulator and no model-backend bypass. The package was made importable with `PYTHONPATH` while retaining the required worktree directory. These are development tries, not an official leaderboard evaluation or held-out result.

| Training seed | Score | Marvelous / total judgements | Stale / total ticks |
|---|---:|---:|---:|
| 1001 | 1,000,000 | 77 / 77 | 0 / 659 |
| 1002 | 1,000,000 | 85 / 85 | 0 / 703 |
| 1003 | 1,000,000 | 198 / 198 | 0 / 475 |
| 1004 | 1,000,000 | 107 / 107 | 0 / 610 |
| 1005 | 1,000,000 | 37 / 37 | 0 / 950 |
| 1006 | 1,000,000 | 74 / 74 | 0 / 583 |
| 1007 | 1,000,000 | 100 / 100 | 0 / 907 |
| 1008 | 1,000,000 | 135 / 135 | 0 / 403 |
| 1009 | 1,000,000 | 115 / 115 | 0 / 756 |
| 1010 | 1,000,000 | 36 / 36 | 0 / 845 |
| 1011 | 1,000,000 | 81 / 81 | 0 / 628 |
| 1012 | 1,000,000 | 86 / 86 | 0 / 803 |

Mean score: **1,000,000**. Every episode ended by `game_done`, with no error, grade X, 100% accuracy, full combo and full health. All **1,131** judgements had zero error; automatic judgement count was zero. There were **0 stale ticks out of 8,322**, **0 Jev calls**, and **0 Astra calls**. Raw results and traces are retained under `candidates/0004/tries/`.

In particular, seed 1003's final observation has `completed_judgements = total_judgements = 198`, `judgements.Marvelous = 198`, `timing_error_summary.max_abs_ms = 0`, and `automatic_judgement_count = 0`. This verifies both sides of its LNs and simultaneous chord scheduling. The earliest batch includes known tails beyond the initial two-second head window; later heartbeat noops do not cancel the queued events.

Nine synthetic tests using stdlib `unittest` in `test_harness.py` additionally exercise chord/LN expansion and release-before-press ordering, identity deduplication, held/judged/missed filtering, illegal lane rejection, no backdating, heartbeat/model-free execution, garbage Jev/Astra outputs, confidence/freshness gates, and reversed-first-option stand-ins. All pass. Refinement was limited to rejecting malformed probability/confidence values and removing an unused member; the successful scheduling policy was not tuned to seeds. A final-revision repeat of seed 1003 also scored 1,000,000 with 0/475 stale ticks and no model calls or errors (`tries/seed1003-201026/`).

**Interpretation and limit:** the primary compiler hypothesis is strongly supported on this training set. These runs do **not** establish that the exception supervisor improves performance: no protocol exception occurred, so its recovery branch is covered synthetically but not by live-model fault data. There is no new runtime-model latency measurement. No held-out seed or source implementation was inspected. With the normal path already at the score ceiling and zero stale ticks, the next useful experiment would be observation/transport fault injection, not a larger routine model prompt.
