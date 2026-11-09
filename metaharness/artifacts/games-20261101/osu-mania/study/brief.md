# osu!mania brief

## Scope and tier

**Canonical game:** osu!mania, the key-based mode of osu!.

**Benchmark profile used below:** native osu!mania `.osu` charts, stable/ScoreV1 rules, no online account, no mods, and a deterministic local simulator. This is a deliberate profile choice: the stable judgement and score formulas are documented precisely, while osu!lazer has materially different MAX windows and hold-note scoring. The fidelity boundary is recorded in [`interfaces.md`](interfaces.md).

This is **tier 2**. The chart clock advances continuously and each object is judged at a real timestamp; the agent cannot pause while Astra reasons. There is no turn structure or opponent. In the benchmark, Astra should plan a horizon and Jev should classify the next visible patterns while a low-level scheduler executes already-approved key events.

## Objective and game rules

Notes travel down a vertical playfield toward a judgement line. The objective is to press the lane corresponding to each note at its timestamp and, for a hold, keep that lane down until its tail. The mode has a variable number of columns/keys; in a native chart, `CircleSize` is the key count. A chart contains ordinary notes and hold notes; sliders and spinners are not used in mania. Empty judgement lines do not break combo.

The benchmark treats each tap or hold as one stable ScoreV1 object. A press in the wrong lane is not a substitute for the required lane. Chords are multiple objects with the same start time and must be pressed simultaneously. A lane cannot contain two overlapping holds in a valid chart.

### Stable native judgement windows

For a native mania chart, with overall difficulty `OD`, the absolute hit-error limits are:

| Result | Common name | Value | Accuracy weight | Maximum absolute error |
|---|---:|---:|---:|---:|
| PERFECT / MAX | rainbow 300 | 320 | 300 | 16 ms |
| GREAT | gold 300 | 300 | 300 | `64 - 3*OD` ms |
| GOOD | 200 | 200 | 200 | `97 - 3*OD` ms |
| OK | 100 | 100 | 100 | `127 - 3*OD` ms |
| MEH | 50 | 50 | 50 | `151 - 3*OD` ms |
| MISS | miss | 0 | 0 | `188 - 3*OD` ms |

Converted-from-osu charts use a different fixed stable table (PERFECT 16; GREAT 34 when OD > 4 otherwise 47; GOOD 67 when OD > 4 otherwise 77; OK 97; MEH 121; MISS 158). The selected benchmark should generate or accept native mania charts only, or mark a converted chart and apply that table explicitly.

The listed values are half-widths. The documented implementation rounds hit error and truncates the maximum to an integer, so a simulator should define its rounding policy explicitly and use integer milliseconds for reproducibility. A hit before the MISS window has no effect; a late MEH hit/release is impossible and becomes a miss. Double Time, Half Time, and Nightcore do not change stable mania hit-window durations. These details come from the [official osu!mania judgement page](https://osu.ppy.sh/wiki/en/Gameplay/Judgement/osu!mania) and [Overall difficulty](https://osu.ppy.sh/wiki/en/Beatmap/Overall_difficulty).

For a hold, stable ScoreV1 gives **one** judgement from the head press error and tail release error. Let `h` and `t` be their non-negative absolute errors and `c = h + t`. PERFECT requires `h <= 1.2*16` and `c <= 2.4*16`; GREAT requires `h <= 1.1*GREAT_window` and `c <= 2.2*GREAT_window`; GOOD and OK require, respectively, `h <= their_window` and `c <= 2*their_window`. Anything else that is not a miss is MEH. A hold is a MISS when its lane is not held from the tail's early MEH-window start through the late OK-window end; late MEH releases are also misses. Releasing during the body caps the result at MEH. A hold break resets combo and prevents a high tail result. See [hold notes](https://osu.ppy.sh/wiki/en/Gameplay/Hit_object/Hold_note) and the [judgement table](https://osu.ppy.sh/wiki/en/Gameplay/Judgement/osu!mania).

### Score, accuracy, combo, and progress

Stable mania score is capped at 1,000,000 and has equal base and bonus halves. For each scored object:

```text
Score = BaseScore + BonusScore
BaseScore = (1,000,000 * ModMultiplier * 0.5 / TotalObjects) * (HitValue / 320)
BonusScore = (1,000,000 * ModMultiplier * 0.5 / TotalObjects)
              * (HitBonusValue * sqrt(Bonus) / 320)
Bonus = clamp(Bonus_before + HitBonus - HitPunishment / ModDivider, 0, 100)
Bonus starts at 100; ModDivider is currently 1, and the score implementation should use the source-defined per-hit ordering consistently.
```

`HitValue` is 320/300/200/100/50/0 for PERFECT/GREAT/GOOD/OK/MEH/MISS. `HitBonusValue` is 32/32/16/8/4/0; `HitBonus` is 2 for PERFECT and 1 for GREAT; `HitPunishment` is 8/24/44/infinity for GOOD/OK/MEH/MISS. The final total is rounded to the nearest integer. With no mods, a full PERFECT play is the 1,000,000 reference score. The complete formula is in [ScoreV1 osu!mania](https://osu.ppy.sh/wiki/en/Gameplay/Score/ScoreV1/osu!mania).

For stable accuracy, MAX and gold 300 both have weight 300:

```text
accuracy = (300*(MAX + GREAT) + 200*GOOD + 100*OK + 50*MEH)
           / (300*(MAX + GREAT + GOOD + OK + MEH + MISS))
```

The evaluator should report score, accuracy, maximum combo, final combo, result counts, and per-object timing errors. Accuracy and score are the primary progress measures; combo is a useful secondary measure. Health drain is disabled in mania, although hit objects affect the health bar and holding restores it slowly. The simulator should not claim an exact client fail threshold unless it implements the client’s health model; by default it should run the chart to completion so scores remain comparable. See [mania mode](https://osu.ppy.sh/wiki/en/Game_mode/osu!mania) and [mania accuracy](https://osu.ppy.sh/wiki/en/Gameplay/Accuracy#osu!mania).

## Chart and controls

The minimal parser reads `[General]` (`Mode: 3`), `[Difficulty]` (`CircleSize`, `OverallDifficulty`), `[TimingPoints]`, and `[HitObjects]`. A hit object is `x,y,time,type,hitSound,objectParams,hitSample`; mania ignores `y`. A normal note has the circle bit; a hold has type bit 7 and uses `endTime:hitSample`. For `K` columns, compute the lane as `clamp(floor(x*K/512), 0, K-1)`. Note times and hold ends are milliseconds from audio start. These are the [official `.osu` format rules](https://osu.ppy.sh/wiki/en/Client/File_formats/osu_%28file_format%29).

The simulator should abstract physical keys away as lane transitions:

```json
{"time_ms": 1234.0, "lane": 2, "down": true}
{"time_ms": 1450.0, "lane": 2, "down": false}
```

A tap is a down/up pair; a hold is down near its head and up near its tail. Multiple lanes may transition at the same timestamp for a chord. The real client has configurable bindings (for example A/S/D/F and J/K/L/; in the legacy symmetric layout), but the benchmark should use lane numbers so it does not depend on a keyboard, X11, or an audio device.

## Pace and decision latency

The world advances in chart milliseconds. There is no single BPM or note rate: a map may have widely separated notes, simultaneous chords, streams, jacks, and holds. As examples, a quarter note at 120 BPM is 500 ms apart, an eighth is 250 ms, and a sixteenth is 125 ms; dense charts can put multiple lane events in one interval. At OD 10, the stable MAX window is only ±16 ms and the miss boundary is about ±158 ms. Therefore `decision_latency_s` is **0.016**: a fresh decision arriving later than 16 ms from the target can lose MAX, even though it may still earn a lower judgement. Astra must not be called once per note. Jev’s ~0.5 s response is also too slow for direct per-note control; ask it independent recognition questions over a lookahead and schedule future events in code.

## Limits of this profile

This profile does not emulate audio, rendering, skin/layout, client input polling, online rankings, pp, or account state. It uses stable ScoreV1 rather than current lazer: in lazer, MAX timing scales with OD and hold heads/tails are separate judgements; hold ticks were also removed. If the adaptor later targets lazer, it must select that profile explicitly rather than silently mixing formulas. The source for those differences is [lazer gameplay differences](https://osu.ppy.sh/wiki/en/Client/Release_stream/Lazer/Gameplay_differences_in_osu!%28lazer%29).
