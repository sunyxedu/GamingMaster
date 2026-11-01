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
