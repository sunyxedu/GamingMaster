# Malody adaptor

## Connection and lifecycle

This is the offline/headless adaptor selected by the study. It does not start the
Malody client, a server, a display, an audio device, or a network connection. The
`game.py` implementation is an in-process deterministic RealtimeGame and uses only
Python's standard library plus `metaharness.games.base.ActionSpec`. `close()` is
idempotent and releases the only owned resource (the in-memory game state).

Each `reset(seed)` creates a fresh native-mode-0 `.mc`-shaped chart, validates its
`meta`, `time`, and `note` fields, converts fractional beats through piecewise BPM
segments, and builds a new Key-mode episode. No chart download or account is needed.

## Interface

The real-time cadence is `tick_s = 0.04` seconds. A call to `tick(action)` applies
the action at the current logical chart time and advances the clock by one cadence;
there is no wall-clock sleep. `ticks` counts calls, including calls after the chart
has completed. `noop` advances without input. The action vocabulary includes
`press N`, `release N`, and representative `press N+M` chords for zero-based
**displayed** columns. JSON actions are also accepted:

```json
{"type": "press", "column": 2}
{"type": "release", "column": 2, "at_ms": 1840}
{"type": "batch", "events": [
  {"type": "press", "column": 0, "at_ms": 1000},
  {"type": "press", "column": 3, "at_ms": 1000}
]}
{"type": "advance", "to_ms": 2400}
```

A press resolves the nearest unjudged head in that column when it is within the
selected judge's miss window. Taps are complete after their head. A hold head
creates an active held column; its tail must receive a release at or after
`end_ms`. Early or timed-out releases are broken holds. Malformed actions do not
raise from `tick`; they become a no-op and the next observation contains
`action_feedback.error`.

## Observation and score

Observations are strict JSON and include:

- episode seed, mode, key count, judge, mod, logical time, tick count/cadence, explicit judgment windows,
  and terminal state;
- `controls.column_mapping` for each zero-based displayed lane and an explicit `resources` block (the
  emulator has no lives, health, or power-item resource);
- chart ID, difficulty, BPM segments, signed offset, snap grid, and mod;
- up to 64 upcoming notes for a 2.2-second horizon, including raw/effective lane, fractional beat,
  start/end time, kind, decoded pattern (`tap`, `stream`, `jack`, `chord`, `anchor`, or `hold`),
  falling motion, BPM, exact-start group ID/size/columns, and time until the receptor;
- held columns plus active hold objects with head judgment, falling/held motion, tail timer, and
  remaining time;
- recent signed-offset head/tail judgments with raw/effective columns;
- judgment histogram, tail histogram, pending-head/hold counts, misses, broken holds, combo, max combo,
  accuracy, completion, clear/terminal status, and the two score values.

The primary `score` and `score_breakdown()["weighted_accuracy"]` are the weighted
judgment accuracy: Big P 1.00, p1 0.90, p2 0.85, p3 0.80, G 0.40, and M 0.00.
The score is intentionally separate from completion so unresolved notes do not
silently become perfect points. `score_breakdown()` also reports numeric misses,
broken holds, max combo, completion, and raw/normalized `rm_proxy_2025`.

`rm_proxy_2025` is explicitly only the documented RM/no-mod-compatible reference
calculation: Big-P combo bases 200/332/466/600, judge multipliers E 1.20, D 1.10,
C 1.00, B 0.85, A 0.78, and the supported scenario mod factors (none/Flip 1.00,
Const 0.95). It is not claimed to be native Malody Key scoring. Accuracy,
histograms, misses, holds, combo, and completion are the authoritative emulator
outputs.

## Seed scenarios

`SEEDS` declares eight train and eight disjoint held-out seeds. Both groups use the
same eight families (`seed % 8`), so held-out episodes are new deterministic chart
instances rather than an unseen family:

| family | keys | features | mod |
|---|---:|---|---|
| isolated taps | 4 | spacious taps, constant BPM | none |
| alternation | 5 | alternating lanes, occasional chord, BPM change | none |
| streams | 6 | dense 1/4-beat stream and chords, BPM changes | Flip |
| jacks | 7 | repeated same-lane notes and lane changes | Const |
| chords | 4 | two-/three-lane chords, changing BPM | none |
| anchors | 5 | outer-lane anchors and inner responses | Flip |
| holds | 6 | interleaved holds, taps, final release | Const |
| mixed dense | 7 | taps, chords, stream, hold and final release | none |

The generator varies BPM, positive/negative chart offset, snap denominator, lane
patterns, density, and exact note placements from the complete integer seed. The
native chart parser supports nested and flattened timing-point forms and keeps
fractional beats until conversion.

## Fidelity limits

The official Malody V desktop client is not used because the target machine has no
display, GPU, audio device, network, or account. This adaptor therefore does not
simulate rendering, audio latency, keyboard-device jitter, skin effects, online
ranking, or client-specific hold behavior. The public timing/judgment and RM combo
values are community documentation for RM mode; they are a transparent proxy for
this native Key emulator and should be calibrated against an authorized real client
before treating numeric scores as official Key scores.
