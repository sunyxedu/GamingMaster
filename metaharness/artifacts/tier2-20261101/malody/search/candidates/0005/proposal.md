# Candidate 0005: timestamped chart executor

## Parents and diagnosis

This is a crossover of `0000-tier2_jev_only` and `0002-tier2_idle`. The Jev parent establishes that acting is better than idling (mean weighted accuracy 0.167 versus 0.000), while the idle parent establishes the safest fallback: let the real-time clock advance with no malformed or illegal actions. The parent Jev harness asks one unconstrained action question on the full observation and then waits synchronously for Jev.

The traces make the failure concrete. On seed 102, Jev requests occur at logical times 0, 200, 360, 560, and 760 ms, while the game advances on ticks 1--5, 6--9, 10--13, and so on. The first five ticks are noops before the first answer; only 41 of 181 ticks are fresh-action ticks (140 stale). The final observation has 27 misses and 11 broken holds, with only 0.075 weighted accuracy. On seed 100, 138 of 175 ticks are stale and 34 of 44 heads are misses. The model often returns `noop` even when a note is already in the visible horizon, and a one-tick action cannot represent a chord or a hold tail. Seed 104's sparse chart still has eight misses because the 0.17 s response cadence is unrelated to the chart timestamps.

## Hypothesis

The dominant failure is not Jev's ability to recognize a lane. It is the control granularity: a per-tick model decision arrives after several chart ticks, and a plain action has no durable plan for future notes. The strict JSON observation already contains exact effective columns, absolute start times, group membership, and hold ends. Therefore a code-owned event queue should substantially outperform per-tick choices. If the batch grammar accepts absolute `at_ms` events as documented, posting the visible horizon once should remove model latency from the timing-critical path, preserve chords, and release holds correctly. If a batch is rejected or an answer is unavailable, the idle/noop path keeps the episode alive rather than repeating a stale key action.

## What changed

* The harness extracts only trusted numeric fields from the structured observation; it does not parse beats or ask a model to do arithmetic.
* It builds exactly-once `{press, column, at_ms}` events for visible heads and discovers `{release, column, at_ms}` events from `active_holds`, sorted by timestamp. Events at an equal start time remain simultaneous in one batch. A hold tail is posted 1 ms after its reported end, avoiding the observed equality tie that was classified `released_before_end`.
* It tracks note IDs so repeated observations do not duplicate events. It never posts an active-hold tail beyond the 2.2-second observation frontier: doing so could advance the batch past an unseen head. Hold charts therefore use one earliest group at a time; tap/chord-only charts may safely use the visible horizon.
* It asks one compact Jev Choice about **dispatch horizon**, not about an individual key: `full_horizon` or `next_group`. The answer is accepted only when it is the expected typed value with confidence at least 0.60; malformed, late, or ambiguous output selects the deterministic incremental fallback. Thus arbitrary stand-in model output cannot crash pre-flight.
* Between event batches it posts `noop`, inheriting the idle parent's safe behavior and preventing a previous batch from being repeatedly applied.
* Astra is deliberately not called in this candidate. Exact chart ingestion and scheduling are verified mechanical work exposed in the observation, and a 9--94 s Astra call cannot improve a known timestamp after the episode starts. The interface that would be useful for Astra is nevertheless explicit: a future revision can give Astra a short summary of signed offsets, head/tail histograms, chord versus jack misses, and the current dispatch policy, then have it author a new Jev policy or choose a fixed timing correction for the next episode. No such correction is applied here because Jev/Astra should not replace exact arithmetic with an unverified numeric guess.

## Jev contract and bridge

Every Jev request contains this JSON schema:

```json
{
  "task": "Dispatch known Malody Key chart events using exact timestamps.",
  "observation": {
    "time_ms": "number",
    "tick_s": "number",
    "columns": "integer",
    "judge": "string",
    "windows_ms": "object",
    "mod": "string",
    "offset_ms": "number",
    "bpm_segments": "array",
    "column_mapping": "array",
    "upcoming": [
      {"id": "id", "column": "integer", "start_ms": "number",
       "end_ms": "number|null", "kind": "string", "pattern": "string",
       "group_id": "id", "group_size": "integer", "group_columns": "array"}
    ],
    "active_holds": "array",
    "recent_judgments": "array"
  },
  "execution_constraints": {
    "events_are_scheduled_at_their_absolute_start_ms": true,
    "equal_start_group_columns_must_be_pressed_together": true,
    "hold_release_ms_must_not_precede_end_ms": true,
    "fallback_between_event_batches": "noop"
  }
}
```

The single Choice question asks whether to commit the full visible horizon or only the nearest equal-start group. `full_horizon` is accepted only above confidence 0.60; `next_group` is the fallback and still emits all lanes in the earliest group together. Even when Jev selects `full_horizon`, any chart with a hold head or active hold is forced into safe incremental dispatch by code. The answer never becomes a raw action string. It changes only how many already validated timestamped events the code posts. If Jev times out, raises a normal transient error, returns an unexpected value, or has low confidence, the code chooses incremental scheduling. `BudgetExceeded`, `EpisodeOver`, and `ModelUnavailable` end normally when the context is already done; no exception from garbage output is allowed to escape.

The event queue is refreshed after each tick from the latest observation. A note/head or discovered tail is marked scheduled before posting only after it has a finite time and legal column. This freshness rule prevents stale model state from changing an event already posted. A no-event iteration posts `noop`; `wait_for_tick()` is used rather than a busy loop. The event batch is the only timing-critical action, so the 0.17--0.22 s Jev latency is paid once for the dispatch policy rather than once per note. Preliminary live `try` runs on seeds 100--105 reached 1.0 score on every seed, including the hold seed 102 after the 1 ms tail margin and horizon guard, with 4--6 stale ticks and one Jev call per episode. A virtual evaluation with the stand-in typed model also completed seed 102 at 1.0 with no failed calls.

## Astra revision trigger (future iterations)

This candidate has no Astra call, but the diagnostic boundary is intentional. A later candidate should ask Astra to revise the Jev interface only after a trace shows one of these falsifiable conditions: (1) many correctly timed batches but systematic wrong columns despite valid `column_mapping`, (2) correct heads but broken tails while release events are present, (3) a consistent signed-offset bias across batches, or (4) batch actions are rejected by the adaptor. Astra's input should be the derived counts and signed offsets, not the full chart. A revision is confirmed only if it reduces that category on multiple pattern families without increasing misses or stale ticks.

## Evidence that would confirm or refute the hypothesis

Confirmation would be a sharp increase in weighted accuracy, lower misses and broken holds, and a high completion rate on both sparse and dense seeds, with actions concentrated in timestamped batches and few stale ticks. Exact head/tail histograms should improve together; chords should no longer be serialized. A particularly strong confirmation is near-perfect performance on seed 102's holds and on chord families while Jev remains at one planning call.

Refutation would be any of: batch events are treated as immediate rather than scheduled, future `at_ms` events are ignored, duplicate observations cause duplicate judgments, or the score remains near 0.167 despite valid batches. If only holds fail, the likely refutation is the assumed release semantics and the next harness should test release-at-end versus release one cadence after end. If all events are late by one or more ticks, use a measured fixed timestamp lead only after traces demonstrate that the adaptor interprets `at_ms` relative to posting time.
