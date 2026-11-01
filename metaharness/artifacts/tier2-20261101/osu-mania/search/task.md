Design a harness for a **real-time** game (tier 2).

Game: `osu!mania (stable native ScoreV1 simulator)` — This is a deterministic offline osu!mania Mode 3 simulator using the native .osu chart format and the stable ScoreV1 profile (not current osu!lazer). The chart clock advances 16 ms for every tick, independent of model latency; there are no opponents, audio, display, account, health fail, mods, or network. Objective: score each visible note. A tap needs a down transition in its lane near its start. A hold needs a down near its head, continuous key-down state, and an up near its tail. Chords are simultaneous notes in different lanes; jacks repeat a lane and anchors combine a repeated lane with another lane. Stable native windows are MAX ±16 ms, GREAT ±(64-3*OD), GOOD ±(97-3*OD), OK ±(127-3*OD), MEH ±(151-3*OD), with MISS at ±(188-3*OD), all in chart milliseconds. Hold notes receive one judgement: head error and tail error are combined; an early release breaks the hold and caps it at MEH, while a late MEH-window release is a miss. Score is native ScoreV1 out of 1,000,000; MAX and GREAT both weigh 300 for mania accuracy. The observation is structured JSON: chart time and timers, per-lane upcoming notes with lane, kind, falling motion, head/tail times and windows, decoded chord lanes, stream/jack/anchor cues and intervals, visible pattern rows, key state, active-hold release windows and break risk, recent signed head/tail errors, judgement windows, result counts, score, accuracy, combo, maximum combo, miss/hold-break rates, and seeded chart metrics. Health is explicitly disabled and there are no other resources. Use zero-based lanes. Action grammar: `event TIME LANE down|up` or a batch `events TIME LANE down|up TIME LANE down|up ...`; `press`, `release`, and `tap` use the current time, and `noop` only advances. Event times must be nondecreasing within a batch and future events may be scheduled only within the planning horizon. Malformed actions are ignored.

Actions:
- noop: Advance one 16 ms chart tick without submitting a key transition.
- event <time_ms> <lane> <down|up>: Queue one timestamped lane transition; concrete example: event 1325 2 down.
- events <time_ms> <lane> <down|up> ...: Queue a space-separated batch in nondecreasing time order; concrete example: events 1325 2 down 1360 2 up.
- press <lane>: Press a lane at the current chart time (shorthand for a down transition).
- release <lane>: Release a lane at the current chart time (shorthand for an up transition).
- tap <lane>: Press and release a lane at the current chart time; useful for a quick local tap.
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Lane numbers are zero-based.  Event times are chart milliseconds, not wall-clock seconds. Only future events within the visible planning horizon are accepted; a past event is applied at the current time and is judged late. A down transition must be near a note head; holding a key does not invent a hit. Malformed actions are ignored and reported in the next observation.

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

The game ticks every 0.016s **on its own clock**; `ctx.act(a)` posts the current action (instant).
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
- strategy_summary: Strong play reads a lookahead in chunks, distinguishes chords, streams, jacks/anchors, and holds, and schedules lane transitions near the centre of each judgement window. Holds require a correct head, continuous key-down state, and accurately timed tail release; stable ScoreV1 gives the hold one combined judgement. Deterministic code should parse, schedule, judge, and score, while Astra selects horizon policies and adapts to recent errors and Jev rapidly labels visible patterns and hold risks. Do not call either model per note: batch Jev questions and let the event scheduler run while the models think.
- scoring: Report stable native ScoreV1 score out of 1,000,000, stable mania accuracy, MAX/GREAT/GOOD/OK/MEH/MISS counts, final and maximum combo, signed head/tail timing errors, miss rate, and hold-break rate. Score and accuracy reflect timing quality; combo and hold-break statistics expose reliability that a single aggregate score can hide.
- interface.observation: Structured JSON containing current_time_ms, key_count, OD, fixed lookahead_ms, per-lane key state, upcoming object IDs with lane/start/end/kind, active holds, recent result/error records, judgement windows, score, accuracy, combo, maximum combo, and remaining object count. No screenshot, display, audio, account, or network state is required.
- interface.actions: Submit batches of future events of the form {time_ms, lane, down}; a tap is a down/up pair, a hold remains down until its tail, and simultaneous lane events represent a chord. advance(to_time_ms) processes queued transitions and deadlines in timestamp order. The adaptor translates model-approved policies into exact events but must not auto-generate perfect events for objects the model did not select.

### study/strategies.md
# osu!mania strategies and model allocation

## What “strong play” means here

There are no openings, build orders, opponents, or persistent resources in osu!mania. Strong play is continuous prediction and precisely timed lane coordination. A player maximizes score by producing the correct lane transitions near each object’s timestamp, preserving holds, and keeping timing errors consistent. In the benchmark, exact timing arithmetic and bookkeeping belong in the adaptor; Astra and Jev should reason about patterns and policy, not emulate a 16 ms clock.

The engine facts below come from the [mania mode rules](https://osu.ppy.sh/wiki/en/Game_mode/osu!mania), [judgement system](https://osu.ppy.sh/wiki/en/Gameplay/Judgement/osu!mania), [hold-note rules](https://osu.ppy.sh/wiki/en/Gameplay/Hit_object/Hold_note), [ScoreV1 formula](https://osu.ppy.sh/wiki/en/Gameplay/Score/ScoreV1/osu!mania), and [accuracy formula](https://osu.ppy.sh/wiki/en/Gameplay/Accuracy#osu!mania). Pattern names are community/mapping vocabulary, not separate engine rules; see the official [Jack](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu!mania/Jack), [Stream](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu!mania/Stream), [Hold note](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu!mania/Hold_note), and [Anchor](https://osu.ppy.sh/wiki/en/Beatmap/Pattern/osu!mania/Anchor) pages.

## Tactics and placement

| Strong-player tactic | Why it works / common failure | Put it in |
|---|---|---|
| **Read ahead and chunk the chart.** Recognise a short section as a stream, chord, jack, anchor, bracket, or LN sequence instead of treating every glyph as unrelated. Community guidance describes horizontal reading as sequential blocks and vertical reading as simultaneous rhythmic layers; neither is universally superior ([improvement guide](https://osu.ppy.sh/community/forums/topics/146615)). | Bottom-only reading causes late reactions. Over-memorising one chart does not transfer to unseen seeds. | Pattern extraction in **code**; horizon plan and reading mode in **Astra prompts**; fast labels in **Jev questions**. |
| **Aim for the centre of the judgement window.** MAX/300 accuracy is much more valuable than barely surviving: MAX and GREAT have 100% stable accuracy weight, while lower judgements reduce it and reduce bonus. Estimate systematic early/late bias from recent signed errors, then adjust the next scheduled target rather than reacting to one outlier. | Chasing a late note with a second corrective press creates another error. | Signed-error aggregation and target-time arithmetic in **code**; bias policy and adaptation in **Astra**; Jev may answer whether a visible section is drifting early/late, but must not calculate a correction. |
| **Treat holds as two planned transitions with a continuous invariant.** Press the correct lane at the head, keep it down, release at the tail. Read the release shape, not just the head; stable ScoreV1 combines head and tail error, and an early body release caps the result at MEH. | The most common mechanical failure is a good head followed by an early/late release or a forgotten held lane. | Hold state, overlap checks, deadlines, and release timing in **code**; Astra plans hand/lane load; Jev answers `release_due`/`hold_break_risk` recognition. |
| **Play chords simultaneously.** A chord is multiple notes at the same time, not a rapid sequence. Keep each lane independent while a different lane is held. | Sequentialising a chord shifts one or more notes outside their window; pressing two notes in one lane or overlapping same-lane holds is an invalid chart/action. | Group equal timestamps and emit multi-lane events in **code**; Astra selects a safe hand/finger assignment; Jev answers `is_chord` and `which_lanes` from JSON. |
| **Streams and rolls: preserve the pulse.** A stream is a constant snap interval; alternate available fingers/lanes and follow the rhythm rather than visually chasing individual notes. A roll traverses lanes in sequence; jump/hand/quad/chord streams add simultaneous load. | Losing the pulse causes a cascade of late hits. Tension and trying to over-speed a stream drain stamina. | Detect interval and lane transitions in **code**; Astra chooses a chunk policy and fallback; Jev classifies `stream_kind` and `pulse_continuity`. |
| **Jacks/anchors: commit to repeated-lane timing.** A jack is at least three consecutive notes in one column; an anchor is a high concentration in one column at a common snap interval. Use the designated finger deliberately and relax between bursts. | Alternating fingers when the chart repeats one lane causes misses; holding a key instead of making distinct down transitions loses jacks. | Count same-lane runs and enforce down/up transitions in **code**; Astra chooses a comfortable policy; Jev answers `jack_present` and `strain_risk`. |
| **Adjust scroll/read settings before a run when using a real client.** Choose a speed that exposes enough future structure without making it visually noisy; the official mode page documents adjustable scrolling. For the headless benchmark there is no rendering, so replace this with a fixed `lookahead_ms` and do not allow per-seed tuning after seeing the answer. | A horizon that is too short makes Astra react too late; a changing horizon leaks difficulty information. | Fixed visibility in **code**; one-time horizon selection in **Astra**; Jev only recognises visible structure. |
| **Use consistent technique and take recovery breaks.** Community advice emphasises reading, speed, accuracy, stamina, finger independence, and avoiding tension; it also warns against specialising in only 4K when later playing other key counts ([guide](https://osu.ppy.sh/community/forums/topics/146615)). | A benchmark has no fatigue physiology, so do not invent random fatigue unless it is an explicit scenario variable. | Physical advice is an **Astra prompt** for strategy selection; any simulated fatigue is **code** state, not Jev arithmetic. |

## Pattern-specific playbook

### Taps, chords, and dense streams

1. Code groups objects by start timestamp and identifies simultaneous lanes, inter-onset intervals, same-lane repetitions, and alternating lane paths.
2. Astra receives a short horizon and selects a plan such as `single_stream`, `chord_stream`, `jack`, or `mixed` plus a target timing bias of zero unless telemetry supports calibration.
3. Jev receives independent typed questions: `pattern_type` (enum), `requires_simultaneous` (yes/no), `next_lane_group` (choice), and `strain_or_reading_risk` (choice). It does not add, multiply, or choose raw milliseconds.
4. Code turns that plan into scheduled lane transitions and reports result/error. It should never delay a transition for a new model call when a valid event is already scheduled.

### Long notes

A hold occupies one lane from head to tail and may coexist with notes in other lanes. Preserve the held-lane bit while acting elsewhere. Schedule the release as soon as the tail is known, but leave it cancellable if the visible state says the hold broke. A release is not cosmetic: it is part of the stable judgement. This is a good Jev task because “is the key still safely held?” and “is the tail next?” are recognition/yes-no questions; exact tail error and score remain code.

### Jacks, anchors, and alternating patterns

A run of repeated lanes is mechanically different from an alternating stream even when the timestamps match. Code should expose run length, interval, and whether a down/up cycle is required. Astra can choose whether to treat a long run as a stamina risk or to reserve attention for a concurrent hold. Jev can rapidly identify the run category and whether an upcoming lane changes. Do not ask Jev to infer the number of milliseconds from an image-like description; give it the structured times and ask a typed label.

### Reading and adaptation

Use a stable signed error history per lane and globally. If the agent is consistently early, Astra can choose a small global delay for the next horizon; if one lane is systematically bad, it can choose a lane-specific calibration only if the benchmark explicitly permits it. Code should clamp corrections, avoid changing target times after their deadline, and keep the actual reported error. A one-off miss should not cause a large oscillating correction.

For an unseen seed, Astra should first identify key count, OD, density, hold fraction, and pattern mix from the visible metadata. It should select a conservative plan for unfamiliar high-density sections, then increase planned concurrency only when the horizon confirms the lanes. Jev is useful for rapid recognition, not global policy, arithmetic, or long-horizon planning.

## Common mistakes and counters

* **Calling the slow model per note.** Counter: Astra plans a horizon; Jev answers many independent questions in one call; the scheduler executes events without waiting.
* **Using stable formulas for lazer or vice versa.** Counter: put `ruleset` in every state and reject mixed profiles. Stable ScoreV1 has one hold judgement; lazer/ScoreV2 separates head and tail and changes MAX timing. See [lazer gameplay differences](https://osu.ppy.sh/wiki/en/Client/Release_stream/Lazer/Gameplay_differences_in_osu!%28lazer%29).
* **Releasing holds during the body.** Counter: explicit active-hold state and a release deadline; report hold breaks separately from tap misses.
* **Confusing a chord with a fast stream.** Counter: group equal start times before pattern classification and emit simultaneous lane events.
* **Forgetting that early presses have no effect.** Counter: never “pre-hit” a note just because it is visible; schedule a transition near its timestamp. An already-held lane also does not create the distinct down event needed for a new same-lane note.
* **Reading only the bottom of the playfield.** Counter: fixed lookahead and pattern labels; let Astra plan chunks rather than react to a single note.
* **Tension, overcorrection, and single-key-count overfitting.** Counter: vary keys, density, intervals, and patterns by seed; score both accuracy and miss/hold-break rate.
* **Assuming an opponent or counter-meta exists.** There are no opponents or openings in this mode. “Counter” means a technique or chart-selection response to a pattern, not a competitive matchup.

## What the harness should ask

### Deterministic code questions

Implement these without a foundation-model call:

* Parse and validate `[General]`, `[Difficulty]`, `[TimingPoints]`, and `[HitObjects]`; map `x` to a lane with `floor(x*K/512)` and clamp.
* Resolve per-lane down/up transitions in timestamp order, including simultaneous chords and active holds.
* Apply the selected stable timing windows and hold combined-error table, including early-press/no-effect and late-MEH behavior.
* Maintain score, bonus, accuracy counts, combo, maximum combo, signed head/tail errors, and result history.
* Extract interval/lane features and generate/replay only the model-approved event plan; do not use code to silently auto-play all known future objects.

### Astra prompt questions

Give Astra JSON state plus the visible horizon and ask it to:

* choose a high-level plan for the next horizon (`stream`, `jack`, `chord`, `LN`, `mixed`);
* decide whether to preserve the current timing bias, make a bounded calibration change, or prioritise hold safety;
* select a lane/finger allocation policy and fallback for a dense or unfamiliar section;
* interpret recent accuracy, combo breaks, and hold breaks and change strategy between sections;
* explain a plan in terms of the observed chart, without doing final score arithmetic.

Astra should not be the clock, parser, or authoritative judge. It has seconds-to-a-minute latency and must return a plan far enough ahead to be useful.

### Jev typed questions

Batch independent questions against one JSON horizon, for example:

* `pattern_type`: enum `{single, chord, stream, jack, anchor, hold, mixed}`;
* `is_chord`: boolean;
* `next_lane_group`: a finite list of lane sets already presented in state;
* `hold_release_due`: boolean;
* `hold_break_risk`: enum `{low, medium, high}`;
* `readability_risk`: enum `{low, medium, high}`;
* `preferred_policy`: choice among named policies Astra has supplied.

Do not ask Jev to derive OD formulas, calculate score, invent timestamps, or make one call for every 16 ms event. Its output should select or label a pre-defined policy; code schedules and judges the concrete lane transitions.

## Seeded scenario design

There is no opponent. `reset(seed)` should vary:

* key count (for example 4–7 in the normal benchmark, with held-out 8/9-key cases if the agent is expected to generalise);
* OD and density independently;
* BPM/interval grids and small timing perturbations;
* proportions and placements of taps, chords, streams, jacks, anchors, and holds;
* hold durations and cross-lane release overlaps;
* short isolated sections versus long mixed sections.

Keep every generated chart valid: no same-lane overlap, no two objects in one column at one time, and no impossible hold release. Split by seed before evaluation so a model cannot memorise chart IDs. Report per-scenario score, accuracy, miss rate, hold-break rate, maximum combo, and timing-error distribution—not only a single aggregate score.

### study/brief.md
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

### adaptor/ADAPTOR.md
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

### study/interfaces.md
# osu!mania programmatic interfaces

## Recommendation

Use a **local open-chart simulator** driven by native mania `.osu` files, implemented with the evaluator’s Python 3.12.3 standard library. It is the only option here that is headless, offline, deterministic, account-free, and naturally produces structured observations and timestamped actions. It also lets the benchmark vary key counts, OD, density, pattern families, and hold placement by seed. The simulator should implement the stable native ScoreV1 profile specified in [`brief.md`](brief.md), not silently claim to be current lazer.

The real osu! client is useful as a reference or offline spot-check, but it is a poor adaptor target on this server: there is no display, GPU, audio device, account, or documented headless control API. Live gameplay automation is also prohibited by the [osu! rules](https://osu.ppy.sh/wiki/en/Rules), so screen/input or bot routes must not be used against the online service.

## Interface comparison

| Route | Maturity | Latency and observation | Control | Legal/ToS risk | Cost and credentials |
|---|---|---|---|---|---|
| **Local `.osu` parser + simulator (chosen)** | The file format and stable formulas are official and well documented; the simulator itself must be tested by the adaptor phase | Deterministic local JSON; event processing can be sub-millisecond and does not depend on wall-clock scheduling | Timestamped lane down/up events; reset/advance; no OS input | Low when used only with local/generated charts; no online score or account | Free; evaluator Python only; no credentials |
| **osu! API v2** | Mature official web API | Network request latency and rate limits; metadata, beatmap, score/profile/leaderboard data, not the live playfield | No gameplay input, pause, or real-time state endpoint | Official API use is lower risk if terms/rate limits are followed, but it cannot play | Free registration/OAuth; client ID/secret or token for protected routes; see [API docs](https://osu.ppy.sh/docs/index.html) |
| **Official `.osu`/`.osr` file and replay pipeline** | Stable formats; replay tooling is widely used | Local file parsing is fast; `.osr` is post-hoc replay data, not a live observation | Can create or inspect replay frames, but a replay does not control a running client by itself | Low for offline analysis; submitting automated play online conflicts with fair play | Free; no credential for local files. `osrparse` and [osu-parsers](https://github.com/kionell/osu-parsers) are optional libraries |
| **Built-in Auto/replay playback** | Useful official-client validation path | Produces a rendered play/result, but not an agent-visible structured state | Auto/replay drives the client itself; it offers no decision-action hook for Astra/Jev | Do not submit automated results online; safe only as an offline chart/simulator spot-check | osu! client and local chart/replay; still needs a display-capable client |
| **Embed osu!lazer ruleset/source or its tests** | Very high fidelity to current open-source lazer code, but APIs are internal and change with master | Local deterministic ruleset calls can be fast once embedded; building the full client is slow/heavy | Possible only after writing a custom integration around ruleset objects/events; no documented public gameplay-agent API | Code/framework is MIT, but branding/resources have separate terms; live automation remains prohibited | .NET SDK and a large source checkout; no account for tests. [Repository](https://github.com/ppy/osu), [lazer gameplay differences](https://osu.ppy.sh/wiki/en/Client/Release_stream/Lazer/Gameplay_differences_in_osu!%28lazer%29) |
| **Real client + screen capture + key injection** | Mature desktop automation primitives, but brittle for a rhythm game | Screenshot/vision and OS input add frame, capture, compositor, and injection jitter—bad against 16 ms MAX windows | X11/Wayland virtual keyboard or `uinput`; physical lane bindings | Highest anti-cheat/ToS risk; osu! rules prohibit software that plays for a user. Use only for an explicitly isolated local experiment, never online | Client, virtual display, capture, input-injection tools, and likely audio/graphics dependencies; no account is acceptable only if the client supports offline local play |

…(truncated; full text in prior/interfaces.md)

At a tap head, a down transition in the lane is the candidate hit. At a hold, the lane must be down near the head and remain down until a release near the tail; a premature release records a hold break and caps the stable result at MEH. Simultaneous objects require simultaneous transitions in their respective lanes. The low-level adaptor may translate a model’s selected target time into events, but it must not invent perfect events for objects the model did not select.

### Software and environment

Required:

* `/home/hemlix/GamingMaster-mh-tier2/runs/osu-mania/.venv/bin/python` (Python 3.12.3), using only the standard library for the selected path.
* A native `.osu` chart corpus is optional; generated charts avoid accounts and downloads.

Optional environment variables, with no credential values:

* `OSU_MANIA_CHART_ROOT`: directory of local `.osu` files to use instead of generated charts; omit for seed-generated scenarios.
* `OSU_MANIA_RULESET`: explicit profile name, default `stable-scorev1-native-mania`; reject unknown profiles rather than silently using lazer rules.
* `OSU_MANIA_LOOKAHEAD_MS`: visible planning horizon, default 2000 ms; keep this fixed within an evaluation split.

No server address, API key, OAuth token, osu! account, display, GPU, or audio device is required. If a future adaptor deliberately chooses the official API instead, keep `OSU_CLIENT_ID`, `OSU_CLIENT_SECRET`, and `OSU_ACCESS_TOKEN` in its process environment only; never write values here.

## Validation checklist

* Compare parser lane assignment against `floor(x*K/512)` and clamp it.
* Test each tap at exactly 0, just inside, exactly on, and just outside every stable window.
* Test early press (no effect), late MEH, misses, chords, repeated same-lane jacks, and a hold with early release.
* Cross-check ScoreV1 counts and score against the formula in the official [score page](https://osu.ppy.sh/wiki/en/Gameplay/Score/ScoreV1/osu!mania); keep MAX/300 accuracy weights distinct from their 320/300 score values.
* Keep a separate test profile if lazer compatibility is later added; do not combine stable one-judgement holds with lazer two-judgement holds.


## Measured model latency (at archive creation, 2026-10-31 09:17)
- System-2 (Astra), one-word answer: 21 s
- System-2, an answer of ~1292 output tokens: 115 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 434 ms (= 27.1 ticks), p90 462 ms, max 465 ms over 12 warm requests; first request on a new connection 471 ms — a timeout below the tail turns slow answers into failed calls
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
