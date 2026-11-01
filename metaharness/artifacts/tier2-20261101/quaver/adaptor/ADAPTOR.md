# Quaver adaptor

## Backend and lifecycle

This adaptor uses the study's selected **local `.qua` chart simulator**, not the
Steam client. It is a tier-2 `RealtimeGame`: each `tick(action)` consumes one
18 ms virtual cadence and the chart clock advances even when the action is
`noop`. No process, server, display, GPU, audio device, account, or network
connection is started; `close()` is therefore idempotent and clears the local
episode queue.

The supplied `QUAVER_CHART_ROOT` is optional. If set, it must be a directory
inside which `.qua` files are selected deterministically by seed. The parser
validates `Mode`, `TimingPoints`, `StartTime`, `Lane`, and `EndTime`, and rejects
mines, timing groups/scroll-velocity groups, malformed objects, and future
`QuaVersion` values rather than treating unsupported features as taps. Without
a chart root, deterministic native-shaped `.qua` charts are generated under
`runtime/charts/`; those files are runtime artifacts, not source code.

`QUAVER_RULESET` defaults to and must equal `standard-star-v1`. There are no
credentials. See `env.example` for the optional chart-root and ruleset
settings.

## Scenarios and seeds

`reset(seed)` uses a seeded PRNG. Seed families are shared between train and
held-out groups (`seed % 5`): calibration, mixed, LN-heavy, dense-chord, and
stream/jack. The seed changes 4K/7K mode, generated length, BPM/rate, timing
offset, density, lane arrangements, chord/stream/jack/chordjack mix, long-note
placement, and difficulty while preserving valid `.qua` semantics. Held-out
seeds use different random lane layouts and chart hashes, but do not introduce
a scenario family absent from training.

The normal observation exposes only a fair `clock_ms - 164` through
`clock_ms + 2000` lookahead, not the hidden remainder. It does expose exact
visible target timestamps because that is the structured equivalent of the
client's scroll position. `info()` records seed, chart hash, ruleset, source,
scenario metadata, and explicit modifiers for evaluator diagnostics.

Player modifiers are opt-in constructor kwargs: `mirror=True`, `rate=<positive
number>`, `timing_offset_ms=<number>`, and `custom_window_ms=<six values or
mapping>`. The observation and `info()` report them; defaults do not silently
apply mirror, rate, or custom judgement windows. Scenario-generated rate and
offset are part of the chart scenario and are also reported.

## Actions and observations

The action spec lists `noop`, `press 1..7`, and `release 1..7`. The compact
parameter grammar is:

* `press <lane> [time_ms]`
* `release <lane> [time_ms]`
* a JSON event/list, or `events <json>`, for timestamped batches such as
  `events [{"type":"press","lane":2,"time_ms":1000}]`.

An omitted timestamp means the current tick's clock. Timestamped events are
queued in stable order, so code can schedule a lookahead batch without a model
sleeping per note. Press selects the earliest pending object in that lane;
release selects the earliest held long note. Impossible, duplicate, malformed,
late, or out-of-mode actions are ignored and described by
`last_action_error` in the next observation; they do not raise from `tick()`.

Observations are strict JSON and contain mode/lane count, virtual clock and
ticks, a `lane_actions` mapping from each active lane to its virtual press/release
actions, and an `input_mapping` declaration that this backend uses virtual
lane events rather than physical keys. A real client would expose its configured
4K/7K key layout here; there is no physical keyboard in this headless backend.
A fair visible note window follows. Each visible note is decoded into
id/lane/kind (tap/long), vertical motion, start/end timestamp, relative timers,
state (`pending`, `held`, `judged`, or `missed`), simultaneous group, and
pattern labels (`tap`, `stream`, `jack`, `chord`, or `chordjack`) plus long-note
role. A `timing` object gives current BPM/signature, beat period, upcoming timing
changes, scenario rate/offset, and the explicit timestamp-authoritative vertical
scroll model. The hidden remainder of the note chart is not exposed.

The observation also contains held lanes, recent signed timing errors, a compact
aggregate player-action timing-error mean/median/absolute-error, variance, recent-window drift,
and early/near/late histogram, plus the count of automatic expiry judgements, chart progress/remaining judgements, resources (health/combo/held
lanes), all judgement counts, misses, score, accuracy, grade, combo/max/final
combo, full-combo state, health, difficulty and performance rating, chart hash,
seed, ruleset, scenario, modifiers, and terminal status. `score_breakdown()`
returns numeric accuracy, normalized-score, combo, health, and survival
components.

## Scoring fidelity

The simulator follows the documented Quaver Keys Standard* constants:

* press windows are Marvelous 18, Perfect 43, Great 76, Good 106, Okay 127,
  and Miss 164 ms;
* accuracy weights are 100, 98.25, 65, 25, -100, and -50;
* score weights are 100, 50, 25, 10, 5, and 0, with the official 10-hit
  multiplier progression and normalized one-million maximum;
* a long note contributes press and release judgements; release windows are
  1.5x; late missed releases use Good by default and do not break combo;
* Miss breaks combo, health is bounded to 0..100, and performance rating is
  `DifficultyRating * (Accuracy / 98)^6`.

A tap is missed when its press deadline expires. A missed long-note start also
records its required release Miss. A held long note with no release is scored
as the configured `LNMissJudgement` after the release deadline. An explicit
early release outside the release window is a Miss. Boundaries are inclusive.

The fidelity limit is intentional: this does not emulate rendering, audio
latency, visual scroll quantisation, keyboard/OS scheduling, client skins,
Steam, server validation, replay serialization, every custom timing group, or
unsupported mines. It is a deterministic rules/format simulator for offline
headless evaluation and is labelled `local_sim` in every observation and
`info()` result; it must not be used for online/ranked play.
