# osu!mania adaptor

## Connection and fidelity

This adaptor uses the study's selected **local open-chart simulator**. The real
osu! client has no supported headless gameplay API and this server has no
window, GPU, audio device, or account, so no client, network connection, or
credential is started. `game.py` parses native Mode 3 `.osu` charts with the
standard-library parser. By default it generates a seeded native-format chart
in memory, serializes it through the native format, and parses it again. An
optional `OSU_MANIA_CHART_ROOT` selects a deterministic chart from local
`.osu` files instead.

The simulator is deterministic and real-time at a fixed `tick_s = 0.016` s.
Each `tick(action)` queues the submitted lane transitions and advances the
chart clock by exactly 16 chart milliseconds. It does not sleep, run a worker,
or inject OS keyboard events. Future events can be scheduled in the visible
planning horizon; an event submitted after its requested time is applied at the
current time and judged late rather than being retimed perfectly.

This is the documented `stable-scorev1-native-mania` profile, not osu!lazer. It
uses native Mania windows (MAX 16 ms, then GREAT/GOOD/OK/MEH/MISS at
`64/97/127/151/188 - 3*OD` ms), one stable judgement per hold, and ScoreV1's
equal base/bonus halves. MAX and GREAT both have 300 accuracy weight.

## Observation decoding

`observe()` is compact structured JSON, not a screenshot or raw memory dump.
It exposes:

- `key_count`, `od`, `bpm`, `ruleset`, `timers`, `current_time_ms`, the fixed
  `lookahead_ms`, planning horizon, chart remainder, and `tick_s`.
- `lanes`: each zero-based lane's current `key_down` state and upcoming notes.
  Each note has `id`, `lane`, `kind` (`tap`/`hold`), `motion` (falling toward
  the judgement line), absolute `start_ms`/`end_ms`, relative time-to-head and
  time-to-tail, the head judgement window, and for holds the duration and
  release window/`release_due` flag.
- `visible_patterns`: temporal rows with `lanes`, pending object IDs,
  simultaneous/chord status, interval, and decoded `pattern` cues (`single`,
  `chord`, `stream`, `jack`, `anchor`, or `hold`). Per-note fields also carry
  chord lanes/size, stream status, same-lane run index/length, and lane/row
  intervals. These are calculated from the parsed chart objects used by the
  judge, so Astra/Jev do not need to infer simultaneous lanes or jacks from
  pixels.
- `active_holds`: held lane, head/tail timers, elapsed duration, key state,
  signed head error, release window, `release_due`, and `hold_break_risk`.
- `recent_results`, `timing_errors_ms`, and `timing_error_summary` provide
  signed head/tail timing errors; `counts`, `score`, `accuracy`, `combo`,
  `final_combo`, `max_combo`, `miss_rate`, and `hold_break_rate` provide
  progress and reliability.
- `resources` explicitly reports `health: null` and
  `health_model: "disabled; chart runs to completion"`: this profile has no
  health/fail resource, opponent, inventory, or other persistent resource.
  `chart_metrics` reports the seeded scenario without exposing hidden note
  positions outside the lookahead.

There is no graphical rendering in this headless profile. The note's
`motion` and time-to-head/tail fields are the decoded equivalent of a player
watching notes fall toward the judgement line; timing and lane values come
from the native parser and are also the authoritative judge state.

## Actions

Lanes are zero-based. `action_spec` provides these grammar forms:

- `noop`
- `event TIME LANE down|up`, e.g. `event 1325 2 down`
- `events TIME LANE down|up TIME LANE down|up ...`, with nondecreasing times
- `press LANE`, `release LANE`, or `tap LANE` at the current chart time

A down transition near a tap head scores that object. A hold needs a down near
its head, continuous key-down state, and release near its tail. Repeated key
transitions that do not correspond to a valid state change are harmless.
Timestamped event submissions are idempotent when the Tier 2 mailbox repeats a
stale action on later ticks. Malformed actions are ignored and described in
`action_errors` in the next observation. `act()` and `advance()` are also
available as deterministic local helpers, although the evaluator uses the Tier
2 `tick()` interface.

## Seeds and scenarios

`SEEDS` contains eight disjoint training seeds (`100..107`) and eight held-out
seeds (`1000..1007`). Every seed makes a fresh chart. Independent stable SHA-256
streams choose key count (3--8), OD, BPM and subdivision grid, length, density,
difficulty, dominant stream/jack/chord/anchor/mixed family, hold fraction and
hold duration. The generator rejects lane choices that would overlap an
existing hold, while allowing cross-lane holds and release overlap. The chart
metrics in observations and `info()` report the selected scenario, note/hold
counts, density, interval, pattern mix, and release-overlap count. Held-out
seeds use the same generator and scenario family distribution, not a separate
level set.

## Score and limits

`score` is the rounded stable native ScoreV1 score in `[0, 1_000_000]`.
`score_breakdown()` separately reports native score, accuracy, judged progress,
combo reliability, miss rate, and hold-break rate. `info()` additionally reports
counts, final/maximum combo, and aggregate/per-object signed head/tail timing
errors. The simulator runs until all objects are resolved; it intentionally
omits rendering, audio, skin/input polling, notelock, health/fail behavior,
mods, online scores, lazer's distinct MAX/hold rules, and client-specific
floating-point/input timing details.

No external process or connection is owned. `close()` is idempotent and clears
queued transient state.
