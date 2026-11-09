# Quaver brief

## Identity, mode, and objective

Quaver is an open-source, community-driven competitive vertical-scrolling rhythm game. The relevant ruleset here is **Keys**: notes fall (or scroll upward) in lanes and the player presses the corresponding lane key at the correct time. The game has distinct **4K** and **7K** modes with separate leaderboards; a map's mode determines the lane count. A regular note has one hit judgement. A long note (LN) has a press/start and a release/end, so it contributes two judgements. Sources: [official gameplay wiki](https://quavergame.com/docs/gameplay), [official API chart model](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Qua.cs), and [official game README](https://github.com/Quaver/Quaver#readme).

The play objective is to finish the chart while hitting every object in its lane and timing window, maximize accuracy and score, and avoid combo breaks/failure. There is no opponent/build order in solo Keys play. In ranked play, the meaningful performance measure is accuracy/performance rating, not merely raw score: `PerformanceRating = DifficultyRating * (Accuracy / 98)^6`. Overall rating is a decayed weighted sum across plays. Raw score is capped at 1,000,000 per map and is described by the game as XP rather than a global-ranking metric. Sources: [official scoring documentation](https://quavergame.com/docs/gameplay#performance-rating), [official scoring source](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs), [official gameplay wiki](https://quavergame.com/docs/gameplay#score).

## Judgements, accuracy, grade, and combo

Use the ranked **Standard\*** window unless a scenario explicitly selects another preset. The official Standard\* windows are absolute hit-error windows:

| judgement | absolute error | accuracy weight |
|---|---:|---:|
| Marvelous | <= 18 ms | 100% |
| Perfect | <= 43 ms | 98.25% |
| Great | <= 76 ms | 65% |
| Good | <= 106 ms | 25% |
| Okay | <= 127 ms | -100% |
| Miss | <= 164 ms / otherwise a missed object | -50% |

Judgement classification uses the absolute difference between the action time and the target start/end time, choosing the tightest matching judgement. Accuracy is the weighted average over all judgements, with an LN counted twice. The official page lists eight presets (Peaceful, Lenient, Chill, Standard\*, Strict, Tough, Extreme, Impossible) and permits custom windows, but ranked scores are standardized to Standard\*. Sources: [official timing and weighting table](https://quavergame.com/docs/gameplay#accuracy), [official timing-window table](https://quavergame.com/docs/gameplay#judgement-timing-windows), [official scoring implementation](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs#L80-L145).

Grades are X (100%, all Marvelous), SS (>=99%), S (95–98%), A (90–94%), B (80–89%), C (70–79%), D (60–69%), and F (failed). Ranked combo breaks on Miss; non-ranked/custom combo-break settings can differ. Not pressing in time produces a Miss. Not releasing an LN is, by default, converted to an LN-miss judgement of Good and does not break combo; an early release can be a miss. Sources: [official grades and combo rules](https://quavergame.com/docs/gameplay#grading), [official LN/miss behavior](https://quavergame.com/docs/gameplay#combo), [official manager code for missed releases](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Screens/Gameplay/Rulesets/Keys/HitObjects/HitObjectManagerKeys.cs#L663-L715), and [release scoring code](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Screens/Gameplay/Rulesets/Input/InputManagerKeys.cs#L332-L435).

For a faithful simulator, model the default score processor as well as accuracy: each judgement has score weights Marvelous 100, Perfect 50, Great 25, Good 10, Okay 5, Miss 0; combo/multiplier rises with consecutive hits and resets/decreases on a combo-break judgement. The exact normalized-score calculation is in [ScoreProcessorKeys](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs#L247-L345). Health is also judgement/density dependent in the official processor; an adaptor may report it, but accuracy, miss count, combo, and score are the stable evaluation fields.

## Chart and timing model

The native `.qua` chart is YAML. The minimum simulator-relevant data is:

```yaml
Mode: Keys4              # or Keys7 (the official model uses a GameMode)
TimingPoints:
  - StartTime: 0
    Bpm: 180
HitObjects:
  - StartTime: 1000
    Lane: 1
    EndTime: 0             # tap; EndTime > StartTime is an LN
  - StartTime: 1200
    Lane: 2
    EndTime: 1600
```

Field spelling/casing follows the official C# model: `TimingPoints`, `HitObjects`, `StartTime`, `Bpm`, `Signature`, `Hidden`, `Lane`, `EndTime`, `HitSound`, `KeySounds`, `EditorLayer`, plus metadata. The official parser sorts objects and validates that LNs end after they start. A chart can contain mines and newer timing groups/SV features; the first simulator should reject unsupported mines or implement them explicitly rather than silently treating them as taps. Sources: [official `Qua` parser/model](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Qua.cs#L27-L215), [official hit-object structure](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Structures/HitObjectInfo.cs), [official Go schema documentation](https://pkg.go.dev/github.com/Quaver/api2/qua).

Timing points/BPM and scroll velocity affect visual placement and song timing, but note target times are already millisecond timestamps in the `.qua`; a score simulator does not need to synthesize audio or render falling notes. The adaptor should normalize lane indices to 1..4 or 1..7, sort by `(StartTime, Lane)`, and treat simultaneous same-time notes as a chord. For an LN, emit a press at `StartTime`, keep that lane held, then emit a release at `EndTime`. The game accepts physical key press/release events; a held lane must remain down while other lanes can be independently pressed.

## Controls and actions

The client lets the user rebind every lane. Current official defaults are generated as:

- 4K: `A S K L`
- 7K: `A S D Space J K L`

The defaults and config names are in [ConfigManager.DefaultKeyLayout](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Config/ConfigManager.cs#L876-L890). The programmatic adaptor should avoid relying on physical letters: represent actions as JSON events such as `{"type":"press","lane":2,"time_ms":1198}` and `{"type":"release","lane":2,"time_ms":1602}`. A real-client backend maps lane numbers to the configured keys and sends key-down/up events. A simulator backend consumes the same events directly.

Useful simulator operations:

- `reset(seed, mode/difficulty optional)`: generate or select a deterministic chart and reset clock/score/health/combo.
- `observe()`: return JSON with `now_ms`, mode/key count, visible/upcoming objects (target time, lane, end time, pressed/held state), judgement settings, recent judgements/error, score, accuracy, combo, max combo, health, and terminal status.
- `step(actions)`: apply timestamped press/release events; score events in timestamp order and advance the clock. Reject impossible duplicate presses/releases deterministically.
- `run_until(t_ms)` / `finish()`: advance time and assign misses/LN-missed-release judgements when their windows expire.

Do not make the model wait for one note at a time. The chart clock advances continuously; the code scheduler should prepare actions from a lookahead window and send them ahead of the target timestamp. In a real client, the action is penalized by timing error, not by a turn boundary.

## Pace and latency

This is **tier 2**. The song clock and note windows continue while Astra/Jev reason. A new note can arrive every few tens of milliseconds in dense charts; at 180 BPM, a sixteenth is about 83 ms and simultaneous chords require several lane events at once. For the top judgement, an action must land within **18 ms** of the target; it remains non-Miss only through the default 164 ms press window, but Okay/late LN behavior damages accuracy and can break combo. Therefore set the study's `decision_latency_s` to `0.018` (the no-accuracy-loss deadline), while engineering the adaptor around pre-scheduling rather than model round trips. A Jev call at roughly 0.5 s is too slow for per-note decisions; Astra is far too slow. Code must parse the chart and schedule deterministic events, while Jev can classify a pattern or diagnose a recent error batch and Astra can choose a chart/modifier/calibration plan.

The local simulator is intentionally faster and more deterministic than a real audio/UI client: its clock can be virtual, and it has no audio latency, GPU/render-frame quantization, keyboard debounce, or OS scheduling jitter. The fidelity boundary is documented in [interfaces.md](interfaces.md).

## Sources

- [Quaver official Gameplay wiki](https://quavergame.com/docs/gameplay)
- [Quaver official open-source client](https://github.com/Quaver/Quaver)
- [Quaver.API official map/replay/scoring library](https://github.com/Quaver/Quaver.API)
- [Quaver API `.qua` package documentation](https://pkg.go.dev/github.com/Quaver/api2/qua)
