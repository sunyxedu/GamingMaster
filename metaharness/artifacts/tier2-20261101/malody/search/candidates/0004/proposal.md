# Candidate 0004: observed-prefix compiler with semantic supervision

## Lineage and hypothesis

Parent: `0002-tier2_idle` (rewrite). Comparator: `0000-tier2_jev_only`, the best evaluated archive entry. Preserve the idle parent's reliable `wait_for_tick` lifecycle and neutral no-op, not its passive policy. Replace the best baseline's model-per-action loop with a deterministic timing thread and a separate asynchronous semantic supervisor.

**Hypothesis:** the bottleneck is the division of labor, not model intelligence. The observation already supplies displayed lanes and absolute head/tail timestamps. Asking Jev to select a key from a raw observation both discards this information and blocks the actuator across several deadlines. Code should compile complete chronological prefixes of observed events; Jev should certify semantic compatibility rather than choose individual keys. Astra is reserved for persistent interface contradictions that justify its multi-second latency.

## Concrete archive evidence

- `0000-tier2_jev_only/trace_seed102.jsonl`, `episode_end`: score **0.075**, **27 misses**, **11 broken holds**, max combo **1**, **140 stale ticks / 181 ticks**, and **42 Jev calls**. The per-episode Jev median is **0.1705169678 s**, maximum **0.2211732864 s**, against a **0.04 s** game cadence. All calls succeed, so this is an architectural failure rather than backend instability.
- That trace's initial observation exposes a hold in displayed column 0 at `start_ms: 848.486`, `end_ms: 1355.243`. Ticks **19 through 23** all apply `noop`, with only tick 19 fresh. Ticks 24–27 again repeat `noop`. This directly demonstrates an already visible, exactly timed hold passing while model-paced control does nothing.
- Its later observation records note 19's hold head as `Big P`, `offset_ms: -9.027`, at `3880.0`, followed by an active hold with `end_ms: 4395.784`. Yet all eleven holds eventually break. A timely head alone is insufficient; releases need explicit, independent scheduling.
- The parent on seed 102 scores **0**, with **31 misses**, **11 broken holds**, **181/181 stale ticks**, zero model calls, and no error. Waiting correctly is safe but earns no score.
- The Astra-only archive has six successful episode executions but zero score and 100% stale ticks. Calibration reports **9.63 s** for a tiny Astra answer versus approximately **6–9 s** natural chart durations. Startup Astra planning is therefore deliberately omitted.
- No `lessons.md` exists in the supplied archive at inspection time. The study explicitly recommends deterministic timestamp scheduling and warns against serializing chords, double-transforming lanes, or treating model latency as a tick.

## Structural changes

1. **Execution is model-independent.** At each tick, read the public observation, validate finite times and legal displayed lanes, build the next action, post it, then `ctx.wait_for_tick()`. Never wait for inference on this thread. No implementation source, chart seed reconstruction, hidden chart state, or training-specific branches are used.
2. **Use observed absolute time.** `start_ms` and `end_ms` already include chart offset/BPM; `column` already includes Flip. Code neither re-integrates BPM nor mirrors lanes again. It accepts either strict JSON text or an observation dict.
3. **Compile a complete chronological prefix.** Before certification, service only near-current events; after certification, compile at most 1.2 seconds of all visible heads and due hold tails. Sort by timestamp, old-hold release before new head, then lane. Chords share timestamps.
4. **Do not attach a distant tail blindly to its head.** Only tails inside the prefix execute. Active holds in later observations supply remaining tails. This prevents a long hold's end from advancing the chart past unseen heads. If the 64-note cap is reached, stop before the last potentially truncated group with a cadence margin.
5. **Serialization and replay safeguards.** Public timer examples are rounded to three decimals, so add a 0.002 ms guard to both heads and tails, preserving same-time release/repress ordering while never releasing fractionally early. Recompute from fresh observations rather than replaying an old batch. In gaps, post fresh `noop`; certified empty prefixes may use documented `advance`.
6. **Layered failure behavior.** Garbage or low-confidence model output leaves deterministic tick execution running. A reported JSON transport rejection invalidates the certificate and selects textual tick actions. Repeated explicit protocol errors may request one asynchronous Astra diagnosis. No unconditional Astra call, no joins, no uncontrolled background loops.

## Jev interface and bridge

One asynchronous request contains three independent Choices. State is a named JSON object:

- `plan`: the current concise execution policy.
- `rules`: displayed input indexing; already converted times; hold release rule; complete-prefix semantics; 2.2-second/64-note visibility; code's responsibility for arithmetic and safety.
- `facts`: code-derived `schema` (`valid`/`invalid`), explicit lane indexing, observed mod, absolute tail availability, visibility guard, JSON transport status, latest protocol error. No raw chart or seed is sent.
- `candidates`: complete-prefix execution, deterministic current-tick execution, and unknown/undecided.

Questions:

1. `route`: decide whether the named schema, lane and transport preconditions support complete-prefix execution.
2. `route_reordered`: identical condition, reversed option order. The pair mitigates first-option bias.
3. `lane`: choose the displayed `column` rather than `raw_column` or unknown, explicitly including Flip semantics.

**Control:** only unanimous `prefix`, `prefix`, `column` enables the 1.2-second prefix compiler. Otherwise the exact current-tick compiler remains active. This is an actual change in action granularity and chart-time progression, not a ceremonial model call. Every application/rejection is logged.

**Confidence:** each selected option needs probability at least 0.85. If confidence is supplied, it must also be at least 0.6. Missing confidence does not imply certainty: probability remains mandatory. Missing keys, malformed values, arbitrary first-option stand-ins, model failure and timeout all retain the executable fallback.

**Freshness:** each result carries a schema signature and plan generation. Only the matching generation/signature can apply. A certificate describes stable semantics, not a note action, so it can outlive the precise note horizon. Its scope and observation age are logged; changed chart identity, column count, mod, indexing, schema validity or transport invalidates it. Every action uses a new observation.

**Latency and budget:** timeout 2.0 s comfortably exceeds observed 0.17–0.22 s calls; there is at most one request in flight and at most three requests for startup/genuine schema changes. The executor continues throughout inference. No timeout is used as a reason to retain an old press.

## Astra interface and revision hypothesis

Astra should revise the interface only after repeated *observed protocol errors*, not isolated note timing or hypothetical uncertainty. Two explicit errors can trigger one asynchronous call with the exact error and current stats. The response grammar is `TICK` or `KEEP`; only the safe `TICK` response is actionable. It creates a new semantic-plan generation and retains textual tick mode. Arbitrary prose cannot generate code or unvalidated actions. This is a narrow policy diagnosis, not runtime question generation; the Jev interface is authored in this harness. Healthy episodes intentionally use zero Astra calls.

## Confirmation / refutation

Confirmation: major improvement over 0.167 mean accuracy, near-zero misses and broken holds on seed 102, simultaneous chord judgments, high completion, and mostly fresh ticks, with one or a few Jev calls rather than hundreds. Trace evidence must show a fresh certificate changing subsequent action granularity while deterministic control remains active during the request.

Refutation: timestamped batches reject, tails break at rounded boundaries, a prefix misses an unseen/truncated chord, old batches repeat, or the tick fallback fails arbitrary model outputs. Any fatal episode makes the candidate ineligible regardless of score. Tests should cover every allowed training seed, rounded hold tails, equal-time release/new-head ordering, cap-limited visibility, and garbage/failed/late semantic responses.

## Validation status

Initial implementation and proposal were written before live testing. Primary metric is weighted accuracy, never the labelled `rm_proxy_2025` score.

### Live training-seed checks (not the official evaluation)

| Seed | Weighted accuracy | Misses | Broken holds | Stale / ticks | Jev / Astra calls | End |
|---|---:|---:|---:|---:|---:|---|
| 100 | 1.000 | 0 | 0 | 0 / 175 | 1 / 0 | game_done |
| 101 | 1.000 | 0 | 0 | 0 / 174 | 1 / 0 | game_done |
| 102 | 1.000 | 0 | 0 | 0 / 181 | 1 / 0 | game_done |
| 103 | 1.000 | 0 | 0 | 0 / 146 | 1 / 0 | game_done |
| 104 | 1.000 | 0 | 0 | 0 / 153 | 1 / 0 | game_done |
| 105 | 1.000 | 0 | 0 | 0 / 210 | 1 / 0 | game_done |

All six complete at 100% with no error. Mean score is **1.000**, versus the best archived **0.166746**; stale ticks are **0 / 1039**. Each episode used one asynchronous Jev request and zero Astra requests. Observed Jev latencies range from **0.155836 to 0.193958 seconds**. Candidate-only traces are under `tries/`; `validation.json` records the compact results. The required working directory was retained by adding the repository to `PYTHONPATH` for the documented `metaharness try` command. No game implementation source was inspected.

### Important negative result

**All six semantic certificates were rejected.** Jev chose `raw_column` on seeds 100 and 102 and was uncertain on the other seeds despite the explicit displayed-lane rule. The probability gates prevented those answers from changing execution. These runs confirm deterministic timing and nonblocking fallbacks, but **do not demonstrate that Jev improves score or that wide prefixes work in the live adaptor**. The main result must not be attributed to model cooperation. The optional wide-prefix branch is covered by synthetic contract tests only; runtime Astra's rare error-diagnosis branch was not exercised by healthy live episodes.

The implementation is intentionally not weakened to force the model to appear useful, and thresholds were not lowered just to obtain an acceptance. Future iterations should either establish a genuinely ambiguous supervisory decision or remove the redundant semantic audit; they should not put known lane/timing arithmetic back into model control.

### Contract and safety checks

`test_safety.py`: **17 passing tests** using synthetic public observations and local stand-in answers, without importing game implementation. Cases cover effective lanes, same-time chords, bounded prefixes, deferred long-hold tails, active-hold releases, release-before-repress ordering, the serialization guard, a capped partial chord, expired heads, bad schema, JSON-string observations, textual fallback, successful semantic activation, stale-generation rejection, first-option bias, malformed model answers, timeouts, transport-error fallback, and a whole-loop stand-in/no-model execution.

These checks are supplemental, **not a claim that the evaluator's official pre-flight has run**. Python compilation succeeds. No formal `eval.json` or leaderboard was edited. No sibling candidate was changed.
