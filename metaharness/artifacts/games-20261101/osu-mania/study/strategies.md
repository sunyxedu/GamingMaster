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
