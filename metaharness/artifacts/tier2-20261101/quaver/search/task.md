Design a harness for a **real-time** game (tier 2).

Game: `Quaver Keys (local .qua simulator)` — Quaver Keys is a real-time vertical-scrolling rhythm chart. The objective is to finish the generated or supplied .qua map with the highest timing accuracy, normalized score, and combo. Each tap has one judgement; each long note has a press and a release judgement. The local simulator uses the official Standard* windows in milliseconds: Marvelous ±18, Perfect ±43, Great ±76, Good ±106, Okay ±127, and Miss through ±164 (outside that window is an automatic miss). Release windows are 1.5 times those press windows; an early release outside them is a Miss, while an unreleased long note is converted to the configured LN-miss judgement (Good by default). Judgement weights are Marvelous 100, Perfect 98.25, Great 65, Good 25, Okay -100, Miss -50. Accuracy is the clamped weighted mean; score is the official-style normalized score out of 1,000,000 and combo breaks on Miss. Performance rating is difficulty rating * (accuracy / 98)^6.

This is RealtimeGame tier 2: every tick is 18 ms and the virtual song clock advances whether a new decision is supplied. Plain actions are 'noop', 'press <lane>', or 'release <lane>'; an optional integer after the lane is a timestamp in milliseconds. For lookahead scheduling, pass one event dict/list or a string beginning with 'events ' followed by JSON, for example events [{"type":"press","lane":2,"time_ms":1000}]. Press selects the earliest pending object in that lane; release selects the earliest held long note. Duplicate or impossible events are ignored and reported in the next observation.

Observations expose the fair visible window (clock - 164 ms through clock + 2000 ms), with note id, lane, kind, vertical motion, start/end time, relative timers, simultaneous group, decoded tap/stream/jack/chord/chordjack pattern, and pending/held/judged/missed state. The top-level timing object gives current BPM, beat period, rate, offset, upcoming timing changes, and the timestamp-authoritative vertical movement model. lane_actions maps every active lane to its press/release action, and input_mapping identifies the local virtual lane interface; there are no physical keys in this headless backend. Observations also expose held lanes, resources, chart progress, recent judgement errors, aggregate player-action early/late timing-error statistics, variance, recent-window drift and histogram, automatic expiry count, full judgement counts, accuracy, score, grade, combo, max combo, health, terminal state, seed, chart hash, ruleset, scenario metadata, and explicit modifiers. The hidden remainder of the chart is not included. The built-in charts vary mode, length, BPM/rate, timing offset, density, streams, jacks, chords, chordjacks, and long-note patterns by seed; external charts must be valid supported .qua YAML. Mines, timing groups, and unsupported future chart versions are rejected rather than scored as ordinary notes. This backend is an offline local_sim: it omits Quaver rendering, audio, OS input latency, Steam/network validation, and other client-side effects.

Actions:
- noop: Apply no lane event for this 18 ms tick.
- press 1: Press lane 1 at the current tick time (or use press 1 <time_ms>).
- press 2: Press lane 2 at the current tick time (or use press 2 <time_ms>).
- press 3: Press lane 3 at the current tick time (or use press 3 <time_ms>).
- press 4: Press lane 4 at the current tick time (or use press 4 <time_ms>).
- press 5: Press lane 5 at the current tick time (or use press 5 <time_ms>).
- press 6: Press lane 6 at the current tick time (or use press 6 <time_ms>).
- press 7: Press lane 7 at the current tick time (or use press 7 <time_ms>).
- release 1: Release a held long note in lane 1 at the current tick time (or use release 1 <time_ms>).
- release 2: Release a held long note in lane 2 at the current tick time (or use release 2 <time_ms>).
- release 3: Release a held long note in lane 3 at the current tick time (or use release 3 <time_ms>).
- release 4: Release a held long note in lane 4 at the current tick time (or use release 4 <time_ms>).
- release 5: Release a held long note in lane 5 at the current tick time (or use release 5 <time_ms>).
- release 6: Release a held long note in lane 6 at the current tick time (or use release 6 <time_ms>).
- release 7: Release a held long note in lane 7 at the current tick time (or use release 7 <time_ms>).
(real-time games: ticks apply 'noop' until the first action is posted; afterwards a tick without a fresh action repeats the last posted action)
Parameterized grammar: 'press <lane> [time_ms]' or 'release <lane> [time_ms]'; a JSON event/list or 'events <json>' may batch timestamped events. Timestamped events are queued ahead of the virtual clock. Lanes outside the current 4K/7K chart are ignored and reported in the next observation.

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

The game ticks every 0.018s **on its own clock**; `ctx.act(a)` posts the current action (instant).
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
- strategy_summary: Strong Quaver players optimize timing accuracy rather than raw combo, use a comfortable lane layout/readable skin, and practice charts at or slightly above their reliable skill level. They recognize streams, chords, jacks, chordjacks, and long notes, with deliberate practice targeted at the weakest pattern and careful LN press/hold/release state. The adaptor should parse the chart and schedule exact lane events in code, grouping chords and preserving holds; Jev may classify a partial upcoming pattern or timing bias, while Astra chooses difficulty, modifiers, and between-run calibration. Report per-judgement errors and adjust a global offset only between runs; never put a model call on the ±18 ms event path.
- scoring: Report final normalized score out of 1,000,000, accuracy_percent, grade, full judgement counts, miss count, max_combo, final combo, and performance_rating when a map difficulty is available. Accuracy is the principal measure because the official weighted judgements count Marvelous 100, Perfect 98.25, Great 65, Good 25, Okay -100, and Miss -50, long notes count twice, and performance rating uses DifficultyRating * (Accuracy / 98)^6; score and combo remain useful secondary diagnostics.
- interface.observation: JSON containing interface version, virtual clock_ms, mode and lane count, Standard* or selected judgement windows, fair visible/upcoming notes with id/lane/start_ms/end_ms/state, held_lanes, recent judgements and signed errors, judgement counts, score, accuracy_percent, combo, max_combo, health, terminal status, seed, and fidelity. Default visible lookahead should be about 2000 ms rather than the whole hidden chart; an explicit offline chart-planning mode may expose the full chart to deterministic scheduling code.
- interface.actions: JSON batches of timestamped lane events such as {type: press|release, lane: 1..4/7, time_ms: integer}. A press hits the earliest pending object in that lane; a release resolves the earliest held long note. Process simultaneous events together, keep other lanes independent, reject/report impossible duplicate state transitions, and advance the virtual clock until finish assigns expired misses and default Good missed-LN releases. The real-client fallback, if ever used, maps lanes to configured keys and sends key-down/up events, but is not the chosen backend.

### study/strategies.md
# Quaver strategies for the agent

Quaver has no opening/build order or opponent counterplay in solo Keys. Strong play is a combination of exact event timing, reading the upcoming lane pattern, appropriate finger technique, and deliberate practice. The agent should therefore move almost all per-note work into deterministic code; Astra plans the run and diagnoses trends, and Jev answers small typed recognition/choice questions only when they cannot be computed more reliably.

## What strong players do

### 1. Configure for reliable reading and timing

- Pick a readable skin and a comfortable lane layout. The official defaults are 4K `A S K L` and 7K `A S D Space J K L`, but the game allows rebinding. Community improvement guidance emphasizes comfort, hand separation, and a readable skin rather than a universally optimal bind. Sources: [official key-layout code](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Config/ConfigManager.cs#L876-L890), [Byte's in-depth community guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2381686124), [community accuracy guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2578042176).
- Set scroll speed high enough that notes do not visually stack, but choose what is readable; there is no magic value. A human player also calibrates audio/visual/map offset. In the adaptor, this becomes an explicit measured `timing_offset_ms`, never a hidden per-chart fudge. Sources: [community improvement guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2192574185), [official client offset controls/source](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Input/Global/GlobalInputHandler.cs#L107-L145).
- For the simulator, discard visual skin/scroll settings for judgement; target timestamps in `.qua` are authoritative. For a real client, measure the offset and input jitter before attempting a score.

**Implementation:** code. Keep `lane -> key` (or lane -> virtual event) as configuration, validate a one-to-one mapping, and use a monotonic scheduler. Astra may choose a readable configuration for a real-client experiment; Jev can answer a typed `readable_state: yes/no` question from a screenshot, but neither should replace a fixed mapping in the action loop.

### 2. Prioritize accuracy over flashy difficulty or raw combo

The official accuracy weights make this concrete: Marvelous is 100%, Perfect 98.25%, Great 65%, Good 25%, Okay -100%, and Miss -50%; LNs count twice. Performance rating raises accuracy to the sixth power. A high-combo low-accuracy pass is not strong ranked play. Sources: [official gameplay/scoring wiki](https://quavergame.com/docs/gameplay#accuracy), [performance-rating formula](https://quavergame.com/docs/gameplay#performance-rating), [official score processor](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs#L132-L145).

Community advice is to play maps in or just above the current skill range, move up after consistently high accuracy (one guide suggests a consistent ~97% as a practical signal), and use easier/slower maps when accuracy is unstable. That is a heuristic, not a rule of the game. Sources: [Flameflan's accuracy guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2578042176), [Rain UK's improvement guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2192574185).

**Implementation:** code reports exact accuracy, per-judgement counts, error histogram, misses, max combo, and final score; Astra chooses the next difficulty from the report. Jev can choose a typed progression label (`stay`, `slightly_harder`, `easier`) from compact result JSON, but must not calculate the percentage.

### 3. Read and execute common pattern families

The following community definitions are useful labels for chart analysis, not separate Quaver rules:

- **Tap/single:** one lane at a target time.
- **Chord:** two or more lanes at the same time; press all lanes together.
- **Stream:** successive notes moving across fingers/lane positions; use alternating fingers and preserve the beat instead of mashing.
- **Jumpstream/handstream:** streams containing two-note or three-note chords.
- **Jack:** the same lane repeated at successive times; use a controlled repeated finger action and avoid double-triggering.
- **Chordjack:** repeated chords; keep simultaneous lanes synchronized while distributing repeated work between hands.
- **Long note (LN):** press at the head, hold that lane, release at the tail. The official game counts the press and release as two judgements. Sources for these community labels and progression advice: [Byte's guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2381686124); official LN semantics: [gameplay wiki](https://quavergame.com/docs/gameplay#accuracy) and [input manager](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Screens/Gameplay/Rulesets/Input/InputManagerKeys.cs#L223-L435).

Strong players deliberately practice the weak family: stream-tagged maps for streams, jack/chordjack maps for repeated lanes, LN-heavy maps for holds/releases, and mixed tech maps for reading. They do not “counter” a chart by randomly moving keys; they adjust finger allocation, rhythm, or reading approach while preserving target timing. Sources: [community pattern/progression guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2192574185), [in-depth guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2381686124).

**Implementation:** code should classify pattern features and precompute all lane events; a chart-known simulator can schedule an entire chord/LN sequence without a model call. Jev is suitable for a typed label such as `pattern_family: stream|jack|chord|ln|mixed` when the state is intentionally partial or visual. Astra can decide which weakness to practice after seeing aggregate errors, but should not issue each key event.

### 4. Treat LNs as state, not as two independent taps

At an LN head, press once and mark the lane held. Keep it held through its body even while pressing other lanes. Release near the end timestamp. A late/unperformed release is normally converted to Good, while an early release can miss; a dropped LN can therefore cost the second judgement and, depending on the event, health/score. The official processor uses a 1.5 release-window multiplier and the official manager records an automatic LN-miss judgement when the release window expires. Sources: [official gameplay rules](https://quavergame.com/docs/gameplay#combo), [official score code](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/ScoreProcessorKeys.cs#L179-L237), [official missed-release code](https://github.com/Quaver/Quaver/blob/develop/Quaver.Shared/Screens/Gameplay/Rulesets/Keys/HitObjects/HitObjectManagerKeys.cs#L663-L715).

**Implementation:** code owns `held_lanes`, schedules releases, checks overlapping same-lane LNs, and never releases a lane merely because another note is due. Jev may answer `is_lane_already_held` for an intentionally noisy visual state, but this is better as a deterministic set lookup. Astra can decide whether a seed is LN-focused practice.

### 5. Use feedback to correct systematic timing error

A human uses the hit-error bar/result graph: repeated early hits need a timing/offset correction in one direction; repeated late hits need the other. The official game exposes hit differences in replay/result data, and a community accuracy guide specifically recommends inspecting the error bar and post-song graph. Source: [community accuracy guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2578042176); [official `HitStat` data](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Processors/Scoring/Data/HitStat.cs); [official replay model](https://github.com/Quaver/Quaver.API/tree/master/Quaver.API/Replays).

**Implementation:** code computes median/mean error and a robust histogram. Astra decides whether to change a global calibration between runs and bounds the change; Jev can answer a typed question from a binned histogram (`early`, `late`, `neutral`, `inconclusive`). Do not let a model change offset within a song: that creates inconsistent timing and makes held-out evaluation noisy.

### 6. Build stamina without sacrificing safety or accuracy

Community guides recommend short, manageable stamina files first, then gradually longer/more demanding streams and jacks; they also recommend taking breaks when hands tire. Do not copy advice to push through pain: fatigue creates timing drift and risk of injury. Source: [accuracy guide's rest advice](https://steamcommunity.com/sharedfiles/filedetails/?id=2578042176), [in-depth progression guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2381686124).

…(truncated; full text in prior/strategies.md)

- **Hidden/lane cover:** visual-reading practice only; it should not change judgement math. Community players report it can help reading streams/LNs, but it is preference-dependent: [community guide](https://steamcommunity.com/sharedfiles/filedetails/?id=2192574185).
- **Autoplay:** never use it as an agent strategy or score. It bypasses the skill being measured and third-party automation is prohibited for server play by [Quaver's rules](https://quavergame.com/docs/rules).

**Implementation:** code applies modifiers and records them in the result; Astra chooses them for practice; Jev can choose a typed modifier from a small allowed set after seeing the current weakness. Held-out scenarios should not permit an unreported modifier to make a run easier.

## Common mistakes and corrections

| Mistake | Why it hurts | Correction / owner |
|---|---|---|
| Asking Astra/Jev for every press | 0.5 s Jev and multi-second Astra calls miss 18–164 ms windows | Preparse and schedule events in code; model only plans/labels |
| Screenshot-only OCR in the headless setup | Requires display/GPU-ish rendering and adds frame/OCR jitter | Use `.qua` JSON in the simulator; real client is fallback only |
| Treating an LN release as a new tap or dropping the held lane | Loses the second judgement and can add a miss | Code maintains held-lane state and tail events |
| Pressing a chord sequentially | Lane events land at different times and lower accuracy | Code groups equal target timestamps and dispatches together |
| Releasing a lane when another note is due | Breaks an unrelated LN | Per-lane hold set/queue in code |
| Using a wrong lane map or duplicate key | Every otherwise-timed note becomes a miss | Validate lane count and mapping before `run` |
| Chasing max combo/raw score | Good/Okay judgements can preserve combo but destroy accuracy; rating rewards accuracy strongly | Optimize accuracy first; report score and combo separately |
| Playing only charts far above ability | Generates misses and teaches bad timing | Astra chooses skill-appropriate or slightly-above charts; move up after stable high accuracy |
| Ignoring early/late bias | A constant offset turns every note into a worse judgement | Code histogram; Astra changes calibration between runs; Jev only returns early/late/neutral |
| Randomly changing scroll speed/offset mid-run | Makes visual timing and feedback nonstationary | Freeze settings within a run; adapt only on reset |
| Mashing jacks or forcing fatigued hands | Extra/late presses and variance; possible injury | Pattern-specific controlled practice and rest; code can stop on degradation |
| Letting unsupported mines/timing groups pass silently | Simulator diverges from client semantics | Reject with an explicit fidelity error or implement and version the rule |

## Suggested model/harness division

1. **Astra pre-run:** choose 4K/7K, seed/chart family, target difficulty, optional practice modifier, and a calibration policy from prior result JSON. It can explain why and revise after a complete run.
2. **Code pre-run:** parse YAML, validate the chart, classify density/patterns, expand taps/LNs into timestamped lane events, apply lane/rate modifiers, and initialize score state.
3. **Jev during run only when useful:** ask independent typed questions over compact JSON such as `pattern_family`, `timing_bias`, `should_abort_on_degradation`, or `next_practice_family`. Never ask it to add/subtract timestamps or pick a physical key from a screenshot when the chart already says the lane.
4. **Code during run:** run the virtual/monotonic clock, emit press/release events, preserve holds, advance misses, and compute judgement/score/accuracy. No model round trip belongs on the critical 18 ms path.
5. **Astra post-run:** inspect judgement/error/pattern summaries, choose the next seed/difficulty, and keep a small persistent calibration record. The output should include `score`, `accuracy_percent`, `grade`, `judgements`, `misses`, `max_combo`, `timing_error_summary`, `seed`, `mode`, `modifier`, and `fidelity`.

This follows the actual game: the world ticks while the player reads and acts, so Quaver remains tier 2 even though a chart-aware deterministic scheduler can make the local simulation almost perfectly repeatable.

### study/brief.md
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

…(truncated; full text in prior/brief.md)

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

### adaptor/ADAPTOR.md
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

### study/interfaces.md
# Quaver interfaces and adaptor choice

## Recommendation for this run

**Choose a local `.qua` chart simulator with a JSON observation/action protocol.** It is the only option that works with all stated constraints (offline, headless Linux, no GPU/audio device, no account) and still tests timing, lanes, long-note state, judgement, combo, accuracy, score, and held-out chart variation. It uses the same open chart representation and the official Standard\* windows/weights, rather than pretending that a screenshot bot can meet an 18 ms window.

This is not a claim that a simulator is the same as the competitive client. It does not reproduce audio/UI/Steam/network latency, server validation, visual scroll, or every current modifier/timing-group edge case. It must label results `local_sim` and keep a fidelity version in its output. Use generated charts or user-supplied/openly licensed chart files where possible: community map music/assets can have separate copyright/licensing terms. Use the native client backend only when the user supplies a Steam installation/account and a display/input path.

## Chosen interface: local simulator contract

### Inputs and observations

The adaptor reads a native YAML `.qua` file, validates the fields below, and runs a virtual song clock. It should expose JSON (one object per observation) with at least:

```json
{
  "interface": "quaver-local-sim-v1",
  "clock_ms": 1932,
  "mode": "Keys4",
  "lanes": 4,
  "judgement_window": "Standard*",
  "visible_notes": [
    {"id": 12, "lane": 2, "start_ms": 2000, "end_ms": 0, "state": "pending"},
    {"id": 13, "lane": 4, "start_ms": 2020, "end_ms": 2600, "state": "pending"}
  ],
  "held_lanes": [],
  "recent": [{"id": 11, "part": "press", "judgement": "Marvelous", "error_ms": 1}],
  "judgements": {"Marvelous": 11, "Perfect": 1, "Great": 0, "Good": 0, "Okay": 0, "Miss": 0},
  "score": 12345,
  "accuracy_percent": 99.85,
  "combo": 12,
  "max_combo": 12,
  "health": 100.0,
  "terminal": false
}
```

`visible_notes` must be a fair visual lookahead, not the entire hidden chart. A sensible default is objects whose target start/end is between `clock_ms - 164` and `clock_ms + 2000` (or a configurable screen/lookahead duration selected by the scenario). Include target timestamps because the real game visually encodes time via scroll; do not require Astra/Jev to infer milliseconds from pixels in the simulator. Include only notes already visible to a hypothetical client; a separate offline chart-planning mode may expose the full chart to code for evaluation of scheduling.

The input action is a JSON array of timestamped events:

```json
[
  {"type":"press", "lane":2, "time_ms":1999},
  {"type":"press", "lane":4, "time_ms":2021},
  {"type":"release", "lane":4, "time_ms":2598}
]
```

…(truncated; full text in prior/interfaces.md)

- **Cost/legal:** Free source but requires .NET 8, submodules/assets, build time, and likely graphics/audio runtime. MPL code obligations and separate asset licensing apply. The [official rules](https://quavergame.com/docs/rules) prohibit edited clients for server login; restrict a fork to local offline testing.

## Real-client fallback setup (not chosen)

Only use this if the user later supplies an account and accepts offline-only testing:

1. Install Steam and Quaver (app 980610), plus a compatible Linux graphics/audio stack; do not attempt server login with a modified client.
2. Install `xvfb`, `xdotool`/XTest, and `mss`; set `DISPLAY=:99` and launch `Xvfb :99 -screen 0 1280x720x24` with software rendering. Verify the client can load a local `.qua` without audio/GPU.
3. Configure lane keybinds in-game and keep a lane-to-key map outside secrets. Focus the game window; inject only key-down/up events from a monotonic scheduler.
4. Capture screenshots at a known cadence, crop the playfield, and detect lane/receptor/note positions. Calibrate audio/visual offset against a known chart; report the measured offset and timing jitter.
5. Run offline only and compare result-screen counts against the local simulator. Stop if capture/injection jitter exceeds 18 ms; do not use an online account or submit a result.

The simulator remains the chosen interface because it is deterministic, installable in the supplied Python 3.12.3 venv, and exposes exact JSON state without credentials.


## Measured model latency (at archive creation, 2026-10-31 18:06)
- System-2 (Astra), one-word answer: 69 s
- System-2, an answer of ~1015 output tokens: 769 s — latency grows with the length of the answer; size timeouts for the answers your prompts actually request
- System-1 (Jev), one request with the game's initial observation and a Choice over its actions: median 167 ms (= 9.3 ticks), p90 181 ms, max 207 ms over 12 warm requests; first request on a new connection 174 ms — a timeout below the tail turns slow answers into failed calls
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
