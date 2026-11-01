# Observation review

## Scope and method

The study selects the local `.qua` simulator because this run has no Quaver
client, display, GPU, audio device, Steam account, or network. Therefore there
is no native rendered frame to compare against. I compared observations with
the simulator's authoritative parsed `.qua` state, timing-point state, queued
events, hold state, judgement counters, and score state. This is the appropriate
state comparison for the selected interface; native rendering/audio limits are
listed below rather than hidden.

Checks performed after the observation changes:

- 2,160 observations across seeds `1001, 1003, 1005, 1008, 2002, 2013`, with
  360 realtime ticks per seed. Every observation was strict JSON and remained
  below 200 kB.
- Every visible note's lane, start/end, timer, pattern context, state, and long-
  note role was compared to the simulator's parsed note object and state; the
  state-specific pass covered 62,006 visible-note records.
- 1,800 additional noop observations across five seeds verified that every
  visible note stayed inside the `clock - 164 ms` through `clock + 2000 ms`
  window and that the hidden remainder stayed hidden.
- 1,680 random-action observations across four seeds compared resources,
  judgement counts, score, accuracy, and combo fields with the live scorer.
- A two-timing-point native `.qua` fixture verified BPM/signature changes at the
  exact timing-point boundary.
- A long-note test verified `pending -> held -> judged`, `held_lanes`, and the
  release timer. A +10 ms scheduled run verified the timing-error summary; a
  noop-only run verified automatic expiry judgements are separated from player
  timing errors.

## Player-facing review

| Strong-player perception/action | Decoded observation field(s) | State/value check | Still missing or limited |
|---|---|---|---|
| Select 4K/7K and know which lanes/actions exist | `mode`, `lane_count`, `lanes`, `lane_actions`, `input_mapping`; also `action_spec` and `describe()` | Across train/held-out seeds, mode and active lane range matched parsed chart mode. Each mapping names the exact virtual `press N`/`release N` action and hold status. | Physical key bindings and skin layout are not applicable to the headless virtual-event backend. |
| Know what is where in the playfield | `visible_notes[].lane`, `simultaneous_lanes`, `group_id`, `group_size`, `start_ms`, `end_ms` | Compared every visible item against the sorted parsed `.qua` objects across 2,160 frames. Chords had the same-time lane group and no hidden notes were substituted. | No pixel x-coordinate, receptor position, skin, or screen crop exists in `local_sim`; lane numbers are the structured equivalent. |
| Distinguish taps from long notes and know what action is required | `visible_notes[].kind`, `note_role`, `end_ms`, `time_to_end_ms`, `state`; `held_lanes` | Pressing an LN changed its decoded state to `held` and added its lane; releasing at the parsed tail removed it and produced the release judgement. | Real visual LN body rendering is not reproduced. |
| Read stream, jack, chord, and chordjack context | `visible_notes[].pattern`, `group_id`, `group_size`, `simultaneous_lanes`; `scenario.pattern_mix` | Per-note pattern contexts were compared with the simulator's preceding/same-time lane groups across all 2,160 checked frames. Labels are `tap`, `stream`, `jack`, `chord`, and `chordjack`. | Community labels are deterministic chart-context labels, not a pixel/skin classifier; jumpstream/handstream remain represented by the chord sequence rather than separate labels. |
| Know how notes move and how fast the chart is progressing | `visible_notes[].motion`, `motion_detail`; `timing.current_bpm`, `beat_period_ms`, `current_signature`, `upcoming_points`, `rate`, `timing_offset_ms`, and `timing.scroll` | The two-point fixture reported 120 BPM before 1000 ms and 180 BPM/signature 3 at 1000 ms. Generated chart BPM matched its timing point. | No native frame rate, audio latency, renderer scroll velocity, or unsupported SV/timing-group behavior; `timestamp_authoritative` and rejection of SV groups are explicit. |
| Know the active press/release judgement windows | `judgement_window`, `judgement_window_ms`, `release_window_multiplier`, `ruleset` | Standard* values were checked against the study/API constants (18/43/76/106/127/164 ms; release multiplier 1.5), including exact boundary actions in the simulator. | No client-specific visual hit-burst or audio cue is rendered. |
| Time head/tail presses precisely while the world keeps ticking | `clock_ms`, `tick_s`, `ticks`, `time_to_start_ms`, `time_to_end_ms`, `chart_progress.elapsed_ms` | Over 2,160 frames, `clock_ms == ticks * 18`; every note timer equaled parsed target minus clock. | The clock is virtual and has no OS/audio jitter. |
| Schedule chords together and prepare a lookahead | `visible_notes`, `lookahead_ms`, group fields, timestamped action grammar in `action_spec`/`describe()` | Every checked frame obeyed the 164 ms back-window and 2000 ms lookahead; all six seeds retained hidden chart notes beyond the visible window. | Models cannot see the hidden remainder by design; code may schedule supplied timestamped events, but no real keyboard dispatch occurs. |
| Preserve LN holds while acting on other lanes | `held_lanes`, `lane_actions[*].held`, note `state`, `time_to_end_ms` | Explicit LN head/tail test matched the live held set and note state; random-action checks also compared held lanes every frame. | Overlapping same-lane LNs are rejected/handled by earliest-held selection rather than rendered visually; unsupported client edge cases are outside fidelity. |
| See timing feedback and correct systematic early/late bias | `recent[].error_ms`, `recent[].automatic`, `timing_error_summary` (`mean_ms`, `median_ms`, `variance_ms2`, recent drift, early/late histogram) | A chart-wide +10 ms scheduled run reported mean and median +10 ms for every player event. Noop-only runs reported zero player samples plus nonzero `automatic_judgement_count`, avoiding false timing bias from expiry misses. | Error summary is simulator judgement data, not the native result graph/audio error bar. |
| Track resources and danger while playing | `health`, `resources`, `held_lanes`, `combo`, `max_combo`, `chart_progress`, `terminal`, `last_action_error` | Across 1,680 random-action frames, health/resources/counts/score/accuracy/combo matched live scorer state. Noop misses reduced health and terminal state was reported. | There is no native animation, fail screen, or fatigue sensation. |
| Optimize accuracy rather than only combo/raw score | `judgements`, `miss_count`, `completed_judgements`, `accuracy_percent`, `grade`, `score`, `performance_rating`, `full_combo`, `max_combo` | Perfect scheduled charts reached all Marvelous, 100% accuracy, X, and 1,000,000 score; random-action fields matched scorer calculations. | Normalized score/health intentionally follow the documented local rules implementation, not client/network presentation. |
| Choose difficulty, stamina, and next practice family | `difficulty_rating`, `scenario` (`family`, `chart_length_ms`, `density_objects_per_second`, `pattern_mix`), `chart_progress`, timing variance/drift | Train and held-out seeds shared all five scenario families while varying mode, length, BPM/rate, density, patterns, offset, and difficulty; chart hashes were distinct. | No physiological fatigue signal; variance/drift is only a timing-based proxy, as documented by the study. |
| Know the active rules/modifiers and calibration | `judgement_window`, `judgement_window_ms`, `release_window_multiplier`, `ruleset`, `modifiers`, `timing`, `scenario`, `chart_hash`, `seed` | Reset and explicit modifier tests reported Standard*, custom windows, mirror/rate/offset, chart hash, and seed without changing code files. | Real client skins, key rebinding, audio offset, and server validation are not present. |
| Handle bad input and terminal state safely | `last_action_error`, `terminal`, `last_action`, `done`/`info()` | Generic adaptor validation exercised every listed action and a malformed action without an exception; lifecycle/close checks passed. | None for the selected JSON protocol; infrastructure failure reporting is only relevant if an external process is added. |

## Changes made during this review

`game.py` now exposes decoded lane/action mapping, timing-point/BPM and
movement metadata, per-note pattern/group context, explicit long-note role and
motion fields, chart progress/resources, and compact player-action timing
statistics including variance and drift. `describe()` and `ADAPTOR.md` now
explain those fields and distinguish virtual timestamp-authoritative movement
from native rendering.

## Remaining fidelity limits

The selected backend cannot show a native Quaver skin, receptor geometry, frame
scroll position, audio beat, physical key layout, OS injection latency, Steam or
server state. Mines and timing groups/SVs are rejected rather than silently
scored. These are documented local-simulator limits; the observation contains
the structured state available to a strong player and does not pretend that
pixels or audio were observed.

## Final validation

`validate_adaptor(..., expected_tier=2)` completed with **PASS** after these
changes, including strict-JSON, timing, action, determinism, lifecycle, and
source-integrity checks.
