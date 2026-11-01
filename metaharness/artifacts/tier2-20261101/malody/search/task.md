Design a harness for a **real-time** game (tier 2).

Game: `Malody Key (offline native .mc emulator)` — Malody Key mode, offline native .mc emulator. Objective: clear the generated seeded chart with accurate presses, simultaneous chord presses, and released holds. The observation is strict structured JSON: episode clock/seed/columns/judge/mod/tick cadence, explicit judgment windows, controls.column_mapping, and resources (this Key emulator models no lives or power items); chart metadata has ID, difficulty, BPM segments, signed offset, snap grid, and mod. The next 2.2 seconds (up to 64) of notes carries raw/effective lane, fractional beat and millisecond timers, kind, decoded pattern (tap/stream/jack/chord/anchor/hold), falling motion, BPM, exact-start group ID/size/columns, and time until the receptor. held_columns and active_holds carry hold motion, head judgment, tail timer, and remaining time. last_judgments has signed offsets and raw/effective columns; stats has pending counts, judgment histograms, combo, misses, broken holds, accuracy, completion, clear/terminal status, and score. The logical real-time cadence is tick_s seconds (default 40 ms); each tick applies the supplied current action and then advances the clock, with no wall-clock sleep inside the adaptor. Action grammar is 'noop', 'press N', 'release N', or 'press N+M' for a chord, where N is the zero-based displayed column. JSON press/release actions may add at_ms, and JSON batches contain timestamp-sorted events; advance may use to_ms. A press within the active judge window resolves the nearest unjudged head. C-judge windows are Big P 40 ms, p1 56, p2 81, p3 111, G 200, then M; the selected chart judge is reported. Holds must be released at or after their end, otherwise they are broken. Score is weighted accuracy (Big P 1.00, p1 .90, p2 .85, p3 .80, G .40, M 0); score_breakdown and observations also report misses, broken holds, max combo, completion, and the explicitly labelled rm_proxy_2025 using the documented 200/332/466/600 RM combo bases and multipliers. That proxy is not an officially verified Key score. Seed families cover 4K-7K, taps, streams, jacks, chords, anchors, BPM changes, offsets, snap grids, holds/final releases, and none/Flip/Const mods.

Actions:
- noop: Do not press or release a lane; the chart clock advances one 40 ms tick.
- press 0: Press the displayed Key column 0 at this tick; starts a tap or hold head.
- release 0: Release the displayed Key column 0; resolves its active hold tail.
- press 1: Press the displayed Key column 1 at this tick; starts a tap or hold head.
- release 1: Release the displayed Key column 1; resolves its active hold tail.
- press 2: Press the displayed Key column 2 at this tick; starts a tap or hold head.
- release 2: Release the displayed Key column 2; resolves its active hold tail.
- press 3: Press the displayed Key column 3 at this tick; starts a tap or hold head.
- release 3: Release the displayed Key column 3; resolves its active hold tail.
- press 4: Press the displayed Key column 4 at this tick; starts a tap or hold head.
- release 4: Release the displayed Key column 4; resolves its active hold tail.
- press 5: Press the displayed Key column 5 at this tick; starts a tap or hold head.
- release 5: Release the displayed Key column 5; resolves its active hold tail.
- press 6: Press the displayed Key column 6 at this tick; starts a tap or hold head.
- release 6: Release the displayed Key column 6; resolves its active hold tail.
- press 0+1: Press these columns simultaneously (a chord) at this tick.
- press 0+2: Press these columns simultaneously (a chord) at this tick.
- press 0+3: Press these columns simultaneously (a chord) at this tick.
- press 0+4: Press these columns simultaneously (a chord) at this tick.
- press 0+5: Press these columns simultaneously (a chord) at this tick.
- press 0+6: Press these columns simultaneously (a chord) at this tick.
- press 1+2: Press these columns simultaneously (a chord) at this tick.
- press 1+3: Press these columns simultaneously (a chord) at this tick.
- press 1+4: Press these columns simultaneously (a chord) at this tick.
- press 1+5: Press these columns simultaneously (a chord) at this tick.
- press 1+6: Press these columns simultaneously (a chord) at this tick.
- press 2+3: Press these columns simultaneously (a chord) at this tick.
- press 2+4: Press these columns simultaneously (a chord) at this tick.
- press 2+5: Press these columns simultaneously (a chord) at this tick.
- press 2+6: Press these columns simultaneously (a chord) at this tick.
- press 3+4: Press these columns simultaneously (a chord) at this tick.
- press 3+5: Press these columns simultaneously (a chord) at this tick.
- press 3+6: Press these columns simultaneously (a chord) at this tick.
- press 4+5: Press these columns simultaneously (a chord) at this tick.
- press 4+6: Press these columns simultaneously (a chord) at this tick.
- press 5+6: Press these columns simultaneously (a chord) at this tick.
- press 0+1+2: Press these columns simultaneously (a chord) at this tick.
- press 0+1+3: Press these columns simultaneously (a chord) at this tick.
- press 0+1+4: Press these columns simultaneously (a chord) at this tick.
- press 0+1+5: Press these columns simultaneously (a chord) at this tick.
- press 0+1+6: Press these columns simultaneously (a chord) at this tick.
- press 0+2+3: Press these columns simultaneously (a chord) at this tick.
- press 0+2+4: Press these columns simultaneously (a chord) at this tick.
- press 0+2+5: Press these columns simultaneously (a chord) at this tick.
- press 0+2+6: Press these columns simultaneously (a chord) at this tick.
- press 0+3+4: Press these columns simultaneously (a chord) at this tick.
- press 0+3+5: Press these columns simultaneously (a chord) at this tick.
- press 0+3+6: Press these columns simultaneously (a chord) at this tick.
- press 0+4+5: Press these columns simultaneously (a chord) at this tick.
- press 0+4+6: Press these columns simultaneously (a chord) at this tick.
- press 0+5+6: Press these columns simultaneously (a chord) at this tick.
- press 1+2+3: Press these columns simultaneously (a chord) at this tick.
- press 1+2+4: Press these columns simultaneously (a chord) at this tick.
- press 1+2+5: Press these columns simultaneously (a chord) at this tick.
- press 1+2+6: Press these columns simultaneously (a chord) at this tick.
- press 1+3+4: Press these columns simultaneously (a chord) at this tick.
- press 1+3+5: Press these columns simultaneously (a chord) at this tick.
- press 1+3+6: Press these columns simultaneously (a chord) at this tick.
- press 1+4+5: Press these columns simultaneously (a chord) at this tick.
- press 1+4+6: Press these columns simultaneously (a chord) at this tick.
- press 1+5+6: Press these columns simultaneously (a chord) at this tick.
- press 2+3+4: Press these columns simultaneously (a chord) at this tick.
- press 2+3+5: Press these columns simultaneously (a chord) at this tick.
- press 2+3+6: Press these columns simultaneously (a chord) at this tick.
- press 2+4+5: Press these columns simultaneously (a chord) at this tick.
- press 2+4+6: Press these columns simultaneously (a chord) at this tick.
- press 2+5+6: Press these columns simultaneously (a chord) at this tick.
- press 3+4+5: Press these columns simultaneously (a chord) at this tick.
- press 3+4+6: Press these columns simultaneously (a chord) at this tick.
- press 3+5+6: Press these columns simultaneously (a chord) at this tick.
- press 4+5+6: Press these columns simultaneously (a chord) at this tick.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Grammar also accepts JSON actions {type: press|release, column: N, at_ms: T}, {type: batch, events: [...]}, or {type: advance, to_ms: T}. Columns are zero-based after the active Flip option. A press is a tap unless it matches a hold head; a hold stays down until release. Malformed actions are treated as no-op and reported in the next observation.

Models available through `ctx`:
* `ctx.think(prompt, session_id=...)` — Astra (System-2, slow). Deep reasoning, can plan, can explain
  rules. Returns `.text` and `.session_id` (pass it back to keep conversation context).
  Budget: 40 calls per episode. Astra never sees the game unless you show it.
* `ctx.decide(state, questions)` — Jev (System-1, fast typed judgments). Budget: 4000 calls per
  episode. Read the complete Jev model context and local usage contract below before designing
  its state, questions and answer-to-action bridge. Actual latency must be measured.
* `ctx.spawn(fn)` — background thread, so Astra can think while Jev keeps acting.

Model economics (facts, not instructions): one Jev request answers *many independent typed questions
about one state in parallel* for the price and latency of one, and the Jev budget is two orders of
magnitude larger than Astra's; an Astra call takes far longer and its latency grows with the length
of its answer. Design timeouts and pipelining from the measured latencies (below, if calibrated, and on
the leaderboard). Whatever the design, the evaluator reports how many posted actions were preceded by a
fresh Jev / Astra answer, so the division of labour is visible.

The game ticks every 0.04s **on its own clock**; `ctx.act(a)` posts the current action (instant).
A tick with no fresh action re-applies the last one. Each episode has a wall-clock deadline of
900 s (`ctx.budget.remaining()["wall_s"]`); the game's own end, the deadline or a used-up call
budget end the episode with the score reached. Each episode starts cold: fresh `ctx.memory` and fresh module globals. Clock mode: wall.

Evaluation: mean score over the training seeds (primary), then fraction of stale ticks (lower is
better), then System-2 calls (fewer is better); real-time ties favour longer survival. A candidate can
only be selected if none of its training episodes ends with an error (see the contract). The idle
baseline (never acts) shows what doing nothing scores.

What the harness must design (this is the whole point): how Astra's slow output (a plan, a rule
summary, a policy sketch, a subgoal) is turned into *state + questions* for Jev when needed, how
Jev's typed answer is turned into a game action, when to go back to Astra, and what to do when a
decision is late.

**Evaluation scope.** The harness is selected on the training seeds of this game and then reported on
unseen seeds of the same game; there is no held-out game. Rules, strategies and search that the game
description or the study establish may be implemented as code once they are verified against
observations — but nothing specific to the training seeds.

## Prior knowledge from the study phase
The study phase researched this game (rules, interface, community strategies). Use it: implement exact
rules, bookkeeping and search as code, put planning and adaptation into Astra prompts, and fast
recognition into Jev questions — but verify every claim against observations, because sources can be
wrong or outdated for this build of the game.

The full documents are readable in this archive under `prior/` (`prior/strategies.md`, `prior/brief.md`, `prior/interfaces.md`, `prior/study.json`, `prior/ADAPTOR.md`). Below: excerpts within a size budget; every cut is marked `…(truncated; full text in prior/<file>)`.

### study/study.json (summary)
- strategy_summary: Preprocess the complete chart in code, convert fractional beats through every BPM segment, and schedule timestamped lane presses/releases; do not ask Astra or Jev to react to individual notes. Group equal-time notes into chords, alternate/pace streams and jacks, and keep holds down through endbeat, with C judge/no mod as the calibration baseline. Astra should select difficulty/judge/mod and diagnose signed timing, chord, jack, or hold-specific errors from structured summaries; Jev should only answer fast typed pattern/risk choices over a JSON horizon. Evaluate accuracy and combo across seeded charts, while treating the documented RM score formula as a labelled proxy rather than verified Key scoring.
- scoring: Report weighted judgment accuracy as the primary score: Big P=1.00, p1=0.90, p2=0.85, p3=0.80, G=0.40, M=0.00, plus judgment histogram, misses, broken holds, max combo, and completion. Also report the documented D/no-mod RM-compatible score proxy using 200/332/466/600 Big-P combo bases and judge/mod multipliers, explicitly labelled rm_proxy_2025; the source is https://m.mugzone.net/wiki/2443?lang=sc and applies to RM mode, so it must not be presented as verified native Key scoring.
- interface.observation: Structured JSON, not pixels: episode seed/mode/key count/judge/mod/logical time, chart BPM segments and offset, at least the next 1-2 seconds or 64 notes grouped by start time with column/start/end/kind, held columns, last judgments with signed offsets, and score/accuracy/combo/miss/broken-hold statistics.
- interface.actions: Timestamped JSON actions: press(column, at_ms) for tap/hold heads, release(column, at_ms) for hold tails, advance(to_ms) to progress and resolve misses, or a sorted batch of these events. Chords require simultaneous presses in their columns; the adaptor schedules exact chart timestamps rather than waiting in real time.

### study/strategies.md
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

### study/brief.md
# Malody study brief

## Scope and game identity

This study treats **Malody Key mode** (native `.mc` charts, `meta.mode == 0`) as the playable target. Malody is not one ruleset: the official Malody V listing names Key, Catch, Pad, Taiko, Ring, Slide Live, and Cube, and lists osu/sm/bms/pms/mc/tja chart formats ([Steam listing](https://store.steampowered.com/app/1512940/Malody_V/)). The community scoring article used below is explicitly for **RM mode**, a distinct falling-note mode, so its timing/scoring table is a documented proxy for a headless Key simulator rather than evidence that every Malody mode shares one score formula ([RM scoring article](https://m.mugzone.net/wiki/2443?lang=sc)). This distinction must remain visible in evaluation reports.

The user-supplied target is an offline, headless Linux server with no account, display, GPU, or audio device. The real client is therefore not the baseline. The baseline is a deterministic local emulator of Key play, using the native chart representation and the documented timing/judgment table; fidelity limits are listed below and in `interfaces.md`.

## Objective and state

A player clears a chart by pressing the correct key column when each falling note reaches the receptor. A tap is one press. A hold requires a correctly timed head press and, in the simulator, a release at/after its end; the chart converter convention is a note with an `endbeat` field ([Malody-to-osu converter](https://github.com/Jakads/malody2osu/blob/master/convert.py), [Python converter parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)). A chord is several columns with the same start time and must be pressed together. The single-player objective is to maximize accuracy and score while completing the chart, with as few misses and broken holds as possible.

Important reported state is:

- chart seed/id, mode, key count, difficulty, BPM/timing sections, current absolute time;
- upcoming notes grouped by start time, each note's column and hold end;
- currently held columns and pending notes;
- judgment histogram, last judgment and signed hit offset;
- current/max combo, miss/broken-hold counts, accuracy, score, and terminal/clear status.

## Native chart representation

A `.mc` chart is JSON with top-level `meta`, `time`, `note`, and optional `effect`/`extra` data. The format documentation identifies `meta.mode` 0 as Key and `meta.mode_ext.column` as the number of columns ([rconv Malody format reference](https://prefixaut.github.io/rconv/rconv/malody.html)). A note has a fractional beat triple `[measure, numerator, denominator]`, a zero-based `column`, and optionally `endbeat`; converters also encounter `type`, `sound`, `vol`, and `offset` ([malody2osu source](https://github.com/Jakads/malody2osu/blob/master/convert.py)).

For a normal 4/4 chart, the commonly implemented beat value is `measure + numerator / denominator`; timing points in `time` carry a beat and `bpm`, with an optional `sign`. Convert beat positions to milliseconds piecewise between BPM points, adding the chart/audio offset. Do not use the first BPM for the whole song: the converter source explicitly walks BPM changes and computes cumulative offsets ([converter timing code](https://raw.githubusercontent.com/Jakads/malody2osu/master/convert.py)). Validate unusual time signatures and negative/empty beats rather than silently guessing; the Python parser documents normalization of negative snaps and 4-beat bars ([malody-convert parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)).

## Judgments, accuracy, combo, and score

The most concrete public timing table found is the Malody community RM article. It gives six result buckets: Big P, p1, p2, p3, G, and M. For desktop-style timing, the windows are symmetric around the target as follows:

| Judge | Big P | p1 | p2 | p3 | G | Miss |
|---|---:|---:|---:|---:|---:|---:|
| A | ±60 ms | ±105 ms | ±120 ms | ±135 ms | ±200 ms | >200 ms |
| B | ±47 ms | ±63 ms | ±88 ms | ±118 ms | ±200 ms | >200 ms |
| C (standard) | ±40 ms | ±56 ms | ±81 ms | ±111 ms | ±200 ms | >200 ms |
| D | ±34 ms | ±50 ms | ±75 ms | ±105 ms | ±200 ms | >200 ms |
| E | ±25 ms | ±50 ms | ±75 ms | ±105 ms | ±200 ms | >200 ms |

The same source assigns accuracy contributions Big P 100%, p1 90%, p2 85%, p3 80%, G 40%, and M 0%. Report this weighted accuracy as the primary quality metric; it is more comparable than raw score across judge levels and charts. The source says the timing windows are not changed by speed mods, while judge choice changes both tolerance and multiplier ([RM scoring article](https://m.mugzone.net/wiki/2443?lang=sc)).

For the article's D-judge/no-item/no-mod example, a Big P is worth 200 points at combo 0--20, 332 at 21--50, 466 at 51--100, and 600 at 100+. Its listed judge multipliers are E 1.20x, D 1.10x, C 1.00x, B 0.85x, A 0.78x; listed mod multipliers are Slow 0.55x, Dash 1.05x, Rush 1.12x, Const 0.95x, Death/Flip 1.00x, with active items/ST applying 0.8x. These exact base values and multipliers are documented for RM, not guaranteed to be the native Key score formula. The emulator should therefore report weighted accuracy, judgments, misses, combo, and a clearly labelled **RM-compatible score proxy**; it must not call the proxy an official Key score. The article also says a tap/flick miss resets combo, while a broken hold stops combo accumulation without resetting it ([same source](https://m.mugzone.net/wiki/2443?lang=sc)).

## Controls and action semantics

Real Key play uses one configurable input per column, with simultaneous key-down events for chords and key-up events for hold releases. Do not assume an undocumented default keyboard layout: expose a zero-based column-to-key mapping and allow the adaptor to set it. The local interface should accept timestamped actions, preferably in chart time rather than wall-clock time:

```text
press(column, t_ms)
release(column, t_ms)
advance(t_ms)
```

An action batch may contain several same-time presses. The simulator should resolve each press to the nearest unjudged note in that column, judge by signed `t_press - t_note`, auto-miss notes after the selected G/miss deadline, and make reset(seed) deterministic. Hold releases must be represented separately so an adaptor cannot obtain perfect holds by pressing a tap once. This is a simulator contract, not a claim that the official client exposes JSON actions.

## Pace and tier

This is **tier 2**. Notes continue to arrive while a player thinks; a chart can contain simultaneous notes, dense streams, jacks (repeated notes in one column), and holds. Under the standard C table, a hit must be within 200 ms to avoid a miss, but a high-quality hit needs about 40 ms and the strict E Big-P window is 25 ms. A decision made once per note by Astra (seconds) or even by a 0.5-second Jev call is too slow for dense charts. The adaptor should parse and schedule the known chart in code, using Astra for chart/setting plans and Jev only for fast typed classifications or bounded policy choices. `decision_latency_s` is set to `0.04` for the standard-C high-accuracy target; the hard miss deadline is about `0.20` seconds.

## Fidelity limits

- The official Steam build is documented for Windows and requires a Mugzone third-party account; it is not a headless Linux API ([Steam requirements](https://store.steampowered.com/app/1512940/Malody_V/)).
- Public `.mc` references document chart parsing well, but do not provide a complete official Key scoring specification. The numerical judgment table and combo score above come from the RM article and need version/mode validation if an actual client is later available.
- Real rendering, scroll effects, audio latency, key-repeat behavior, device jitter, skin effects, online ranking, multiplayer, and anti-cheat are not simulated.
- The default simulator should support taps, chords, holds, BPM changes, chart offset, C judge, and no mod. Flip/Const and the other listed options can be added as deterministic scenario options, but unsupported real-client features must be reported rather than silently approximated.

## Sources

- [Malody V on Steam](https://store.steampowered.com/app/1512940/Malody_V/)
- [Malody RM-mode scoring and judgment (community reference)](https://m.mugzone.net/wiki/2443?lang=sc)
- [Malody chart format reference from rconv](https://prefixaut.github.io/rconv/rconv/malody.html)
- [Malody-to-osu converter source](https://github.com/Jakads/malody2osu/blob/master/convert.py)
- [Python `.mc` parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)
- [Malody Key stable-chart criteria (draft)](https://m.mugzone.net/wiki/1984)

### adaptor/ADAPTOR.md
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

### study/interfaces.md
# Malody interfaces and adaptor plan

## Recommendation in one line

Use a **local deterministic Key-mode emulator** driven by native `.mc` JSON and timestamped actions. It is the only option that fits the supplied offline/headless/no-account/no-display/no-GPU/no-audio constraints and still lets the harness vary charts by seed. Treat the emulator's numerical score as an RM-compatible proxy unless the real Key client is later used to calibrate it.

## Interface inventory

| Route | What it exposes / controls | Maturity and latency | Cost and legal/ToS risk | Verdict |
|---|---|---|---|---|
| **Local `.mc` parser + Key simulator (chosen)** | Structured chart JSON becomes beats, BPM segments, note/hold events, judgments, score, combo and accuracy. Adaptor sends timestamped `press`/`release` actions or a batch. | No network or render loop; deterministic and effectively sub-millisecond in Python. Chart format and parser behavior are supported by multiple open-source references, but full Key scoring is not officially specified publicly. | No account, service, or paid software. The next phase writes its own simulator; do not copy code without checking each dependency's license. Fidelity risk is explicit: RM table is a proxy for Key. | **Choose.** Works on this server and is reproducible for held-out seeds. |
| **Official Malody V Steam/desktop client + Wine/Proton** | Real gameplay, chart loader, skin, audio and result screen. Automation would inject keyboard events and observe screenshots/OCR or a plugin. | Steam client is mature; input timing can be 1--20+ ms with a real event device but Wine/Xvfb/render/audio scheduling adds jitter. Only documented desktop requirement is Windows 7 SP1+ 64-bit, 2 GB RAM, DX10, 500 MB; the listing also says a Mugzone third-party account is required ([Steam](https://store.steampowered.com/app/1512940/Malody_V/)). | Paid Steam product/account and likely network/login; violates the no-account/offline constraints. Running a Windows binary under Wine is unvalidated and may fail without GPU/audio. Game automation may violate service rules; do not use on online rankings. | Useful only for later calibration on an authorized local account. Not the evaluator interface. |
| **Screen capture + keyboard injection** (`Xvfb`, `xdotool`, `ydotool`, `uinput`, evdev, OCR) | Screenshot pixels, optionally audio waveform; sends key-down/up events. It works with whatever client can run in a virtual display but has no semantic chart state. | Capture/OCR is tens to hundreds of ms and has frame/clock jitter; direct uinput is faster, but timing still depends on the client. | Open-source OS tools are free; root/uinput permissions and X/Wayland setup are hazards. Same client account/ToS risk as above, and visual automation can look like botting. | A fallback only; it cannot meet 25--40 ms accuracy reliably or expose deterministic state. |
| **Android/iOS client in an emulator** (ADB/touch or virtual keyboard) | Real client and rendered play; ADB can inject key/touch events and take screenshots. | Emulator/display startup and input transport add large, variable latency; audio/GPU/device emulation are unavailable or expensive on this server. | Client/account/network constraints remain; emulator license and game automation/ToS must be checked. | Not practical for this headless Linux evaluator. |
| **Malody V overlay/plugin/reverse-engineering hook** | Potentially internal chart, timing, input and result objects rather than pixels. The [MalodyV Rework Overlay](https://github.com/MCPsDE/malodyv-rework-overlay) is an unofficial BepInEx-style overlay project; the [hit-window study](https://github.com/Hxjjxg/malody-hitwindow) reverse-engineers desktop/mobile judge functions. | Potentially low-latency internal events, but version/ABI dependent and not a stable public API. The hit-window project targets Windows x86 `malody.exe` and Android arm64 `libmalodycpp.so`, not a Linux client. | DLL injection/reverse engineering may breach client terms, trigger anti-cheat, or break after updates. No Linux/headless path is documented. | Valuable for offline rule calibration by an authorized researcher; not the production adaptor. |
| **Replay inspection/rendering** ([Malody V replay renderer](https://github.com/jk21lcl/malody-v-replay-renderer)) | Reads `.mrv` replay plus `.mcz/.osz` chart package, pairs notes/inputs/judgments, and exports a local MP4; has a browser UI and `--no-browser`. | Good offline forensic/visualization utility; it is post-hoc, not a live action API. Its documented limitations include No Mod/Const/Flip only and special long-note pairing behavior. | MIT project; no account needed for local files. FFmpeg/browser are extra dependencies and video generation is slower than real time. | Use as an optional validation tool, not as control. |

…(truncated; full text in prior/interfaces.md)

## Credentials and environment variables

The chosen local emulator needs **no credentials and no required environment variables**. In particular, do not put a Mugzone UID/key/token in files. If the optional Store API is ever enabled, obtain `uid`, server-generated `key`, API version, and server base URL at runtime from a secret manager/environment; never hard-code them. Suggested optional names are `MALODY_STORE_URL`, `MALODY_UID`, `MALODY_API_KEY`, and `MALODY_API_VERSION`, all disabled by default and not needed by the evaluator.

## Real-client fallback setup (not selected)

For a future authorized calibration run: install Steam/Malody V on a Windows-capable host, create/login to a Mugzone account, launch a private/offline chart, and capture a replay with the same chart/judge/mod settings. On Linux, an experiment could use Wine/Proton + Xvfb + a virtual input device, but this is not a supported Malody API and must not be assumed to work. Use the local replay renderer with `python main.py --port 8766 --no-browser` only for post-hoc chart/replay visualization; it is not a way to feed actions to a live game.

## Sources

- [Malody V Steam listing](https://store.steampowered.com/app/1512940/Malody_V/)
- [Malody V App Store listing](https://apps.apple.com/us/app/malody-v/id1566873256)
- [Malody RM timing/scoring article](https://m.mugzone.net/wiki/2443?lang=sc)
- [Malody chart format reference](https://prefixaut.github.io/rconv/rconv/malody.html)
- [Malody-to-osu converter](https://github.com/Jakads/malody2osu/blob/master/convert.py)
- [malody-convert parser](https://raw.githubusercontent.com/kangalio/malody-convert/master/mc.py)
- [Malody V replay renderer](https://github.com/jk21lcl/malody-v-replay-renderer)
- [Malody hit-window reverse-engineering](https://github.com/Hxjjxg/malody-hitwindow)
- [Malody Store API](https://github.com/Mugzone/Malody_Store_API)
- [MorePracticeMalodyServer](https://github.com/RhythmCodec/MorePracticeMalodyServer)
- [Malody Data Studio](https://github.com/ChuanYuanNotBoat/Malody_Data_Studio)


## Measured model latency (at archive creation, 2026-10-31 08:54)
- System-2 (Astra), one-word answer: 9.6 s
- System-2, an answer of ~1192 output tokens: 94 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 171 ms (= 4.3 ticks), p90 188 ms, max 188 ms over 12 warm requests; first request on a new connection 224 ms — a timeout below the tail turns slow answers into failed calls
Later measurements per candidate are on the leaderboard (latency columns, a latency fit).

## Jev model context and usage for Meta-Harness

Snapshot: 2026-10-29 (docs.typesafe.ai: API, models, primitives, confidence, patterns, cookbooks, request
rules and the jev-1.13 known-failure page; plus the design of deployed open-source Jev agents). This is design context for the Astra proposer and for runtime Astra
when it designs Jev questions. The Python interface below is GamingMaster's wrapper, which differs
from the provider SDK. The model profile and episode budgets in the experiment manifest are
authoritative.

### What Jev supplies

Jev is TypeSafe's System One model for fast, focused judgments. It produces
typed answers and probability distributions, not prose, code, plans or reasoning
traces. It reads text and structured JSON. (The provider has an experimental image
input; this wrapper does not send images, so a visual game needs a separate perception
step — code, or Astra via `ctx.think(images=[...])` — before Jev can judge its state.)
Calibration describes aggregate predictions, not correctness of each answer.
See [System One](https://docs.typesafe.ai/concepts/system-one).

Model ids: `jev-latest` and `jev-preview` both resolve to jev-1.14.0 at this snapshot (released
2026-10-14; jev-1.13.0 remains available). An alias can move between calls; every answer records
the version that produced it, and the leaderboard warns when an archive mixes versions. Limits:
64k tokens per request, 32k per question; English is the primary training language.

In this architecture, Astra can interpret unfamiliar rules, choose subgoals and
design the decision interface. Jev can recognize situations, assess whether a
subgoal's preconditions still hold, or choose among well-described alternatives.
Code should handle exact arithmetic, legal-action checks, coordinate transforms,
bookkeeping and execution. Do not ask Jev to invent an unknown numerical action
mapping or perform an entire planning algorithm in one classification.

### Known failure modes (provider's list for jev-1.13) and what to do instead

* **Literal reading.** It answers the question written, not the one meant: state the
  exact condition in `instructions` and put boundary cases in the criteria.
* **Math, counting, comparisons of numbers and dates.** Unreliable: compute them in
  code and send the result or a named bucket ("adjacent", "two cells away"), never ask
  Jev to count, add or compare magnitudes.
* **Numeric representations.** Semantic words beat raw numbers: colour names over RGB/hex
  or palette indices, "wall to the left" over coordinates. Translate in code first.
* **Indirection.** Several hops (a property of a property, double negatives) cost
  accuracy: name the relevant state field directly, e.g. `threats.nearest`.
* **Large states with irrelevant detail.** Unrelated content distracts and costs accuracy
  ("context rot"): filter in code and send only the fields this decision needs.
* **Contradictory instructions and criteria.** Keep them aligned; a Noul whose `true`
  means "no" performs worse.
* **Choice option order.** It leans toward the option listed first: do not always list a
  default/no-op first; when a decision matters, ask the same question twice with the
  options reordered (one request, two question ids) and act only if both agree.
* **Generation.** It is not a generator: turn extraction into a Choice over candidates
  that code (or Astra) produced.

### Context that each request must carry

Jev does not inherit the proposer's prompt, Astra's conversation, `ctx.memory`,
previous requests, game description or observations automatically. This wrapper
sends only the supplied `state` and `questions` with the configured model.
Construct the necessary context explicitly on every call.

Prefer a named JSON object containing the current observation summary, the
relevant rule/field meanings, Astra's current subgoal and constraints, and any
short recent evidence needed for this decision. Include candidate descriptions
when asking Jev to compare candidates. Supply actual values, not variable names
or a reference to an inaccessible file. Keep irrelevant history out; preserve
evidence that distinguishes the options. English instructions are preferred for
this model. A request must fit 64k tokens, and the state plus any single question
32k. See [State](https://docs.typesafe.ai/concepts/state).

Batch independent judgments about that state into one call. They share the
input but do not see each other's answers. If a question really requires an
earlier answer, put that answer in the next call's state. Otherwise ask both
speculatively and combine them in code. Put the complete judgment in
`instructions`: question IDs are invisible to Jev. Refer to input fields
explicitly, e.g. `plan.preconditions`. See
[Questions](https://docs.typesafe.ai/primitives).

### Patterns that deployed Jev systems share (provider cookbooks, open-source agents)

* **Code computes facts, Jev judges, code decides.** Successful controllers send a small JSON
  state of *derived facts* (distances as named buckets, what is adjacent, time to contact,
  the last action and its measured outcome, the measured age of the observation) rather than
  raw grids or numbers, then ask atomic questions and combine the probabilities in code.
* **One request, several atomic questions.** A per-tick request typically carries 3–8
  independent questions (e.g. a Choice over the legal actions, a Score of immediate danger, a
  Noul per precondition or per candidate), including speculative ones whose answers code may
  ignore; a request with many questions costs about the same latency as one with one.
* **Structured criteria.** Option descriptions as objects (`what`, `not_for`, `examples`) and
  instructions that name the state fields (`` `threats.nearest` ``) measurably beat prose.
* **Freshness and fallbacks.** Every answer is tagged with the observation / plan version it
  judged; a late answer is dropped, and an executable fallback (the last safe action, a
  conservative rule) runs while waiting. Deadlines per question are short and explicit.
* **Confidence gates per consequence.** Irreversible or lethal actions need a higher top
  probability than cheap ones; a top probability under ~0.6 is treated as "undecided" (route to
  the fallback or to System-2), not as a weak yes. Ask high-stakes Choices twice with the options
  reordered and act only when both agree.
* **Large option sets** (more than 255, or many near-identical options) are handled in two
  stages: Score or Noul each candidate independently, then one Choice over the shortlist.
* **System-2 authors, System-1 runs.** The slow model designs the state schema, the questions,
  the thresholds and the fallback policy — and re-authors them when the evidence says the
  questions are wrong — rather than being asked per tick. Harnesses that let System-2 revise
  the System-1 interface on observed failures improved most across iterations.

### Exact local call and answer contract

Import `Choice`, `Score`, `Noul` from `metaharness.models.jev`. Call
`ctx.decide(state, questions, timeout_s=5.0)`; it returns
`dict[str, System1Answer]` directly, keyed by your question IDs.

| Question constructor | Meaning of `answer.value` | Other local fields |
|---|---|---|
| `Choice(instructions, criteria={option: description, ...})` | Selected option key, a string | `.probabilities`, `.confidence` |
| `Score(instructions, criteria=[low_description, ..., high_description])` | Fractional level index, from 0 to number of levels minus 1 | `.probabilities` with string index keys, `.confidence` |
| `Noul(instructions, criteria={"true": description, "false": description})` | Probability of yes, between 0 and 1 | `.probabilities` contains `"true"`/`"false"`; `.confidence` is normally `None` |

All answers also have `.type` and `.top(k)`. There is no `.answers` container,
`.choice`, `.score`, `.noul`, generated explanation, or Score `.legend` on the
local return objects. Keep the Score level descriptions in your own code.
`Noul` criteria are optional. JSON-serializable question dictionaries with
`type`, `instructions` and appropriate `criteria` also work. Credentials and
the HTTP client are managed outside the harness; do not instantiate `JevClient`
or bypass `ctx.decide`. See the [HTTP API](https://docs.typesafe.ai/api) for the
provider's different wire format.

For Choice, define distinguishable options and include an explicit unknown or
no-match option when needed; there can be at most 255 options. Structured
descriptions can spell out inclusion, exclusion and examples when labels are
ambiguous. See [Choice](https://docs.typesafe.ai/primitives/choice).
For Score, use 2–10 concrete ordered levels for one dimension (the API rejects more
than 10; the constructor raises); the returned number is an expected level, not
necessarily an integer or a probability — use it for thresholds, not to interpolate
magnitudes. See [Score](https://docs.typesafe.ai/primitives/score).

Confidence summarizes the distribution and is not interchangeable with the
winning option's probability. Inspect both when useful. A Noul near 0.5 is
uncertainty about a yes/no proposition, not medium severity. Handle absent
confidence explicitly; never turn `None` into certainty. Choose thresholds
using training evidence and action consequences, with a fallback or Astra
escalation for ambiguity. See [Confidence](https://docs.typesafe.ai/confidence).

What `ctx.decide` raises: `TimeoutError` when the answer (including retries of
rate-limited or transient server errors) does not arrive within `timeout_s`;
`RuntimeError` for any other failed call (e.g. a malformed question rejected by
the API); `BudgetExceeded` / `EpisodeOver` when the System-1 budget is used up or
the episode is over; `ModelUnavailable` when the key is rejected or the account is
persistently throttled (the run stops; catching it does not hide it).

### Small usage example, not a game adapter

This helper illustrates one batched request about an Astra plan and an
observation summary. The proposer still designs the summaries, questions,
thresholds, action mapping and scheduling for its candidate.

```python
from metaharness.models.jev import Choice, Noul, Score


def assess_plan(ctx, plan, observation_summary):
    state = {"plan": plan, "observation": observation_summary}
    answers = ctx.decide(state, {
        "condition": Choice(
            "Do the facts in `observation` support `plan.preconditions`?",
            {
                "supported": "All required preconditions are supported by the observations.",
                "contradicted": "An observed fact contradicts a required precondition.",
                "unknown": "The observations cannot establish all required preconditions.",
            },
        ),
        "progress": Score(
            "How much of `plan.goal` is visibly achieved in `observation`?",
            ["No visible progress.", "Some progress, but the goal is incomplete.",
             "The stated goal is visibly complete."],
        ),
        "finished": Noul(
            "Does `observation` explicitly establish completion of `plan.goal`?"
        ),
    }, timeout_s=5.0)
    condition = answers["condition"]
    return {
        "condition": condition.value,
        "condition_probability": condition.probabilities.get(condition.value, 0.0),
        "condition_confidence": condition.confidence,
        "progress_level": answers["progress"].value,
        "completion_probability": answers["finished"].value,
    }
```

### Timing and the generated bridge

One admitted `ctx.decide` request consumes one System-1 call, even with several
questions. Requests are synchronous unless submitted via `ctx.spawn`. Batching
reduces round trips but does not promise constant latency. Measure actual
latency from traces and the task's calibration: Jev can take longer than a game
tick, and the first request on a new connection (each thread keeps its own) pays
the connection setup. Keep an executable fallback while waiting, limit outstanding
requests, and associate each response with the observation and plan version that
produced it. Recheck relevant preconditions and legality on the latest observation
before acting. A cached semantic interpretation may outlive an individual action
decision; define its invalidation rule rather than treating every answer as
permanently reusable.

The context caps each request's timeout by the remaining wall budget. Handle
transient failures within that budget, let `ModelUnavailable` propagate, and stop
background work when `ctx.done`. The standard inference calls already log request,
response, latency and token usage. Add harness events explaining when a Jev
answer changes a chosen action, invalidates a plan, or triggers Astra; call
counts alone do not establish useful cooperation.

In `proposal.md`, specify the Jev state schema, each question's role, how its
answer affects control, freshness/confidence handling, and the hypothesis about
when Astra should revise the interface. These choices belong to the generated
harness; this guide supplies no environment-specific adapter.

When runtime Astra is asked to create or revise Jev questions, include this
guide (or its relevant interface sections) in that `ctx.think` prompt too.
The proposer context is not automatically forwarded to runtime Astra. A harness
can obtain the exact same bundled text with
`from metaharness.models.jev import usage_guide` and append `usage_guide()` to
the interface-design prompt, together with public observations and action
specifications. Do not send the full guide to Jev on every tick; send the
decision-specific state and questions it describes.
