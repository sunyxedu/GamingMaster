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

**Implementation:** Astra chooses session length and difficulty progression; code records timing drift over time, failure/health, and fatigue proxy (rising error variance). Jev can classify a run as `stable` versus `degrading` from a small summary. In a simulator, expose stamina as chart density/length, not as a hidden random penalty.

## Modifiers and adaptation

- **Standard\*:** evaluation baseline. Use the official windows; do not train against a different preset and report it as ranked-equivalent.
- **Strict/Tough/Extreme/custom judgement windows:** useful deliberate-practice options for a human, because they reduce leniency, but simulator scenarios should identify the selected preset and score separately. The official wiki documents eight presets and custom windows: [judgement windows](https://quavergame.com/docs/gameplay#judgement-timing-windows).
- **Speed/rate:** changes the temporal density and stamina/readability demand. A simulator should transform all target times consistently or generate a chart at that rate; do not just label a faster difficulty while leaving timestamps unchanged.
- **Mirror:** deterministic lane transformation. It is a legitimate practice counter for a pattern that feels asymmetric, but it must be recorded in the seed/scenario and applied before scheduling. The official map processor implements lane transforms: [Mirror/Randomize source](https://github.com/Quaver/Quaver.API/blob/master/Quaver.API/Maps/Qua.cs#L1020-L1082).
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
