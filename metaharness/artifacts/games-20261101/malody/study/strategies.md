# Malody Key-mode strategies and harness priors

## What “strong play” means here

Strong play is not a build order: it is accurate, low-jitter execution of a time-ordered, lane-specific event stream. The official Malody V listing identifies Key as one of several modes and advertises effects including Random, Flip, Const, Rush, Hide, Origin, and Death ([Steam listing](https://store.steampowered.com/app/1512940/Malody_V/)). This document targets native Key charts (`mode == 0`) and distinguishes three layers:

- **Code** must do exact arithmetic, chart parsing, event scheduling, collision/hold bookkeeping, and deterministic transforms.
- **Astra prompts** should choose chart/judge/mod policy, interpret performance summaries, diagnose systematic errors, and adapt difficulty or preprocessing.
- **Jev questions** should be short typed recognitions/choices over a JSON horizon; Jev must not be asked to add offsets, parse BPM math, or schedule every key in a dense chart.

The public stable-chart criteria are a draft and are mostly chart-quality rules, not a player manual. They require technically sound notes, accurate metadata, OGG audio, reasonable note counts, and a chart long enough for a stable submission; they also say Auto Mod should obtain full score ([Key stable criteria draft](https://m.mugzone.net/wiki/1984)). Use it as a sanity check for generated charts, not as proof of live scoring behavior.

## Strategy table

| Situation / heuristic | Strong-player behavior | Put it in |
|---|---|---|
| Chart ingestion | Validate Key mode and column count; parse each `[measure,numerator,denominator]`; construct piecewise BPM/time-signature map; add chart offset; sort stable note IDs. | **Code** |
| Basic timing | Aim for the target timestamp, not for a visual frame. Use one monotonic clock and absolute timestamps so BPM changes cannot accumulate drift. | **Code** |
| Chords | Recognize equal-start notes as one chord and press all required columns at the same timestamp. Never serialize a chord merely because the model output is sequential. | **Code**, with Jev recognition |
| Streams/rolls | Read the rhythm and alternate suitable fingers across lanes; keep a regular internal subdivision instead of mashing. For a known chart, pre-schedule the complete stream. | **Code** for known events; **Astra** for pattern diagnosis; **Jev** for “stream vs chord vs jack” |
| Jacks / repeated same-column notes | Use controlled repeated taps with enough separation; avoid key-repeat and accidental double presses. If stamina is the issue, lower speed/difficulty before sacrificing accuracy. A transferable mania guide stresses that reading/control, not random mashing, is the core of chordjack play ([Makise guide](https://note.com/makise24/n/na347dbf291b5?hl=en)). | **Code** for action timing; **Astra** for difficulty policy; **Jev** for classification |
| Long notes | Press the head in its timing window, keep the lane held, and release at the tail. Treat early release, lost hold, and missing tail as separate diagnostics. The `.mc` convention is `endbeat` ([malody2osu](https://github.com/Jakads/malody2osu/blob/master/convert.py)); the public RM article says a broken hold stops combo accumulation without resetting it ([RM scoring article](https://m.mugzone.net/wiki/2443?lang=sc)). | **Code** |
| Lane/finger assignment | Keep a stable column-to-input mapping; use symmetric hands and avoid remapping mid-chart. If a chart has a systematic weak lane, Astra can select a deliberate mirrored mapping before reset. | **Code** for mapping; **Astra** for policy |
| Reading ahead | Keep at least a 1--2 second JSON horizon. Classify density, chords, holds, jacks, and BPM changes before their deadline. Do not make an LLM call for each note: the C Big-P window is ±40 ms and G/miss cutoff is ±200 ms ([RM scoring article](https://m.mugzone.net/wiki/2443?lang=sc)). | **Code** for horizon/scheduling; **Jev** for fast recognition; **Astra** for longer-term plan |
| Judgment selection | C is the documented standard multiplier (1.0x); A/B widen timing at a score penalty and are appropriate for practice, while D/E tighten windows and increase multiplier. Select once per episode, not reactively per note. | **Astra** |
| Accuracy vs combo | Prioritize correct notes and stable timing over reckless combo chasing. Report weighted accuracy, histogram, max combo, misses, broken holds, and score separately. The documented weights are Big P 100%, p1 90%, p2 85%, p3 80%, G 40%, M 0% ([RM scoring article](https://m.mugzone.net/wiki/2443?lang=sc)). | **Code** |
| Score optimization | In the documented D/no-mod RM example, hit value rises from 200 to 332, 466, and 600 as combo reaches 21, 51, and 101+. Do not trade many misses for a theoretical high-combo multiplier; compute expected accuracy/combo from the actual chart. | **Code** for score; **Astra** for risk policy |
| Mods | Use no mod for calibration. `Flip` is a deterministic lane mirror; `Const` changes scroll presentation but not note timestamps; speed mods change pacing, not the cited timing windows. Random/Hide/Death should be separate scenarios because they change reading or failure risk. The official listing confirms the broad effect inventory, while exact multipliers are only documented in the RM article. | **Code** for transform; **Astra** for selection; **Jev** for “mod safe/unsafe” |
| Recovery | A miss is already resolved; do not send repeated presses hoping to repair it. Resume at the next target and preserve hold state. If the simulator permits early reset, Astra should choose reset only when expected remaining score exceeds continuing. | **Code** for state; **Astra** for reset policy |
| Practice progression | Start with stable 4K charts below the agent’s limit, then increase density, BPM, chord width, jacks, holds, and key count separately. The Malody beginner list spans roughly Lv.2--Lv.9 and provides several songs at multiple difficulties, supporting same-song gradual progression ([beginner chart list](https://m.mugzone.net/wiki/1892)). | **Astra** |

## Deterministic implementation plan

### 1. Parse and normalize before play

1. Require `meta.mode == 0`; read `meta.mode_ext.column` and reject columns outside `0..K-1`.
2. Convert the chart’s beat triples with the parser convention and preserve exact rational values until timing conversion. The rconv reference describes beat arrays and mode/column fields ([rconv format](https://prefixaut.github.io/rconv/rconv/malody.html)); `malody-convert` documents normalization for negative snaps ([parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)).
3. Build ordered timing segments from `time`, using each segment’s BPM and cumulative elapsed milliseconds. Apply chart/audio offset exactly once. A common converter formula is `1000 * (60 / bpm) * beats + offset`, with separate cumulative offsets at BPM changes ([malody2osu source](https://github.com/Jakads/malody2osu/blob/master/convert.py)).
4. For each note, produce `{id,column,start_ms,end_ms|null,kind}`. Presence of `endbeat` means hold. Group notes whose converted start times are equal within an explicit epsilon; do not merge notes in different columns.
5. Validate generated fixtures: no impossible same-column simultaneous tap/hold heads, hold end after head, finite BPM, and sensible offset. Keep source metadata for debugging.

### 2. Schedule known charts without model latency

For a fixed chart, code can know every future event. Build a priority queue of head presses and hold releases and emit timestamped actions at exact target times (or a fixed calibrated offset if an input device needs it). Use a logical clock in the emulator. This is the only credible way to achieve 25--40 ms windows while Astra takes seconds and Jev takes about 0.5 s.

Astra can select a timing policy such as “target 0 ms, C judge, no mod” or diagnose “all late by +18 ms; next episode use -18 ms calibration.” Jev can answer independent typed questions such as:

```json
{
  "pattern": "jack",
  "risk": "high",
  "recommended": "continue",
  "confidence": 0.83
}
```

Jev should not calculate `beat -> ms`, pick an arbitrary release timestamp, or emit an untyped prose action. The harness should validate Jev's enum and ignore low-confidence suggestions.

### 3. Use feedback diagnostically

After every short segment or chart, Astra receives:

- signed-offset mean/variance and early/late histogram;
- counts of Big P/p1/p2/p3/G/M;
- chord-specific versus single-note accuracy;
- jack/stream/hold-specific miss rates;
- max combo and where it broke;
- score proxy and weighted accuracy.

Interpretation rules:

- Same sign across all patterns means clock/offset calibration, not poor reading.
- Good singles but poor chords means simultaneous event handling or finger assignment.
- Good heads but broken holds means release bookkeeping or hold policy.
- Good streams but poor jacks means repeated-lane stamina/timing; lower density or speed before increasing key count.
- High score with low accuracy is not a win: score is combo- and judge-dependent, while the weighted judgment metric directly reflects timing quality.

## Common mistakes and counters

1. **Treating Malody as turn-based.** A model waits for its next call while notes pass. Counter: precompute and batch events; reserve model calls for policy/classification.
2. **Treating beat triples as seconds.** Counter: parse BPM sections and offset once; unit-test a constant-BPM chart and a two-BPM chart.
3. **Drifting at BPM changes.** Counter: use cumulative piecewise integration and rational beats, not repeated rounded frame intervals.
4. **Serializing chords.** Counter: group coincident starts and emit one same-time press per lane.
5. **Relying on OS key-repeat.** Counter: explicit down/up events and a per-column pressed set.
6. **Releasing holds immediately after the head.** Counter: schedule `endbeat` releases and record broken holds separately.
7. **Using a visual renderer as the game API.** The replay renderer reconstructs `.mrv` results and exports video but is post-hoc, not a live controller ([renderer](https://github.com/jk21lcl/malody-v-replay-renderer)).
8. **Claiming the RM formula is official Key scoring.** Counter: report the model name (`rm_proxy_2025`) and make accuracy/histogram the authoritative metrics.
9. **Generating unfair charts.** Counter: enforce the draft Key stable criteria and vary one difficulty axis at a time; do not put duplicate same-column notes or unresolvable holds in fixtures ([stable criteria](https://m.mugzone.net/wiki/1984)).
10. **Overfitting to one chart/seed.** Counter: vary BPM, offset, key count, density, pattern families, holds, and BPM changes while keeping the generator deterministic.

## Scenario prior for held-out seeds

A useful seed family should contain:

- 4K, 5K, 6K, and 7K charts (the format supports variable columns; read the chart rather than hard-code 4K);
- constant and piecewise BPM, positive/negative audio offsets, and different snap resolutions;
- isolated taps, left/right alternation, dense streams, same-lane jacks, two/three/four-note chords, anchors, and holds mixed with taps;
- easy/medium/hard density and a few short stress windows, but no impossible same-column overlaps;
- no-mod baseline plus deterministic Flip/Const variants; Random/Hide/Death only when the rules are explicitly configured;
- a mix of chart lengths and a terminal tail that tests final holds/releases.

Hold out entire pattern families or BPM layouts, not just random note timestamps, so a policy cannot memorize a chart. Evaluate both perfect scripted play (parser/scorer sanity) and deliberately early/late/noisy actions (judgment-window sanity).

## References

- [Malody V feature/mode/format listing](https://store.steampowered.com/app/1512940/Malody_V/)
- [RM timing, accuracy, combo, and score reference](https://m.mugzone.net/wiki/2443?lang=sc)
- [Malody Key stable criteria draft](https://m.mugzone.net/wiki/1984)
- [Beginner Key chart progression](https://m.mugzone.net/wiki/1892)
- [Malody `.mc` format reference](https://prefixaut.github.io/rconv/rconv/malody.html)
- [Malody-to-osu conversion logic](https://github.com/Jakads/malody2osu/blob/master/convert.py)
- [Python `.mc` parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)
- [Malody V replay renderer](https://github.com/jk21lcl/malody-v-replay-renderer)
- [Transferable mania chordjack guide](https://note.com/makise24/n/na347549eb448?hl=en)
