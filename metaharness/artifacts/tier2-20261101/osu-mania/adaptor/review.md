# Observation review

Reviewed against `study/brief.md` and `study/strategies.md` before harness
construction. The adaptor is the selected headless native-chart simulator, so
there is no real osu! client renderer, audio stream, GPU, or process memory to
sample. The authoritative comparison target is the parsed `Chart`/`Note` state
used by the simulator's judge. I wrote an independent observation oracle in a
one-off standard-library check: it grouped chart objects by timestamp and
compared lane, hold, timing, chord, and pending-object data rather than calling
the observation feature helper.

## Player-perception checklist

| What a strong player perceives/acts on | Observation field(s) | Check and result | Still missing / limit |
|---|---|---|---|
| Current song clock and continuous cadence | `current_time_ms`, `time_ms`, `timers.current_time_ms`, `tick_s`, `ticks` | Across 4,560 observations on eight train/held-out seeds, the aliases matched the simulator clock and `ticks`; each tick advanced exactly `0.016 * 1000` chart ms. | No wall-clock/audio latency is modeled; this is intentional deterministic Tier 2 virtual chart time. |
| Fixed read-ahead and planning deadline | `lookahead_ms`, `planning_horizon_ms`, `timers.lookahead_ms`, `timers.chart_remaining_ms` | Checked at every long-trajectory observation against `chart.duration_ms - current_time_ms`; all values matched. Future events are bounded by the same planning horizon. | No visual scroll-speed setting exists; the benchmark fixes the horizon as required. |
| What is where: lane/column location | `lanes[*].lane`, `lanes[*].upcoming[*].lane`, `visible_patterns[*].lanes`, `pending_lanes` | 76,214 exposed notes were compared to their native `x -> floor(x*K/512)` parser lane and parent lane. Every ID appeared in the correct lane, with no duplicates or omissions. | Physical keyboard labels are intentionally absent; actions use stable zero-based lane numbers. |
| Note kind and target time | `upcoming[*].kind`, `start_ms`, `end_ms`, `time_to_start_ms`, `time_to_tail_ms`, `head_window_ms` | Every visible note across all eight seeds matched its parsed tap/hold type, absolute timestamp, and relative timer. Tap/hold fields were checked at 16 ms steps and around deadlines. | No pixel glyph/skin representation; the decoded kind and timestamp are the actionable equivalent. |
| Note motion toward the judgement line | `upcoming[*].motion = "falling_toward_judgement_line"`, relative time fields | Checked that motion is present on every visible object and that its time-to-head/tail equals native target minus current chart time. | The simulator has no rendered y-coordinate or actual visual velocity. The profile documents timestamp motion as the faithful structured substitute; audio/render/input-polling effects remain out of scope. |
| Simultaneous chords | `visible_patterns[*].is_chord`, `lanes`, `pending_lanes`, `object_ids`; per-note `chord_size`, `chord_lanes`, `is_chord` | Independently grouped notes by rounded native start timestamp. 49,925 visible pattern rows matched their exact lane sets and pending IDs; chord booleans matched lane-set cardinality. | “Chord” is a chart cue, not a separate ruleset judgement. |
| Streams and rhythm pulse | `visible_patterns[*].pattern`, `interval_from_previous_ms`; per-note `stream`, `interval_from_previous_ms`, `lane_interval_ms` | Row intervals and lane intervals were compared with the parsed chronological rows over all seeds. The code exposes interval values so the harness does not need to calculate pulse timing. | Stream labels are deterministic mapping cues/heuristics, not a client-provided community annotation; unusual external maps may be labeled `single`/`mixed`. |
| Jacks and repeated-lane strain | Per-note `same_lane_run_index`, `same_lane_run_length`; row `pattern`, `interval_from_previous_ms` | Independently computed consecutive single-lane row runs and compared their membership/length to observations over hundreds of ticks. | No physiological stamina/fatigue resource is simulated, as the study explicitly says not to invent fatigue. |
| Anchors and lane allocation | `visible_patterns[*].pattern`, `anchor_lane`, `lanes`; per-note `anchor_lane` | Repeated-lane neighborhood cues were recomputed from adjacent parsed rows and inspected across anchor-family seeds (including seeds 103, 1001, and 1002). | Anchor is community mapping vocabulary, so `anchor_lane` is a transparent deterministic cue, not an engine fact; the exact rendered hand/finger assignment is not represented. |
| Long-note head, continuous hold, and tail release | Upcoming `hold_duration_ms`, `release_window_ms`, `release_due`; `active_holds[*]` with lane, head/tail, `time_to_tail_ms`, `key_down`, `head_error_ms`, release window, risk | 959 active-hold snapshots were checked against the judge's active-hold map; all IDs, lanes, key bits, tail timers, head errors, and release bounds matched. Exact schedulers produced perfect results with maximum simultaneous holds of 1–3. | No separate visual hold-body geometry; continuous state and release timing are explicit. Stable one-judgement holds are intentional and are not lazer's head/tail model. |
| Current key/input state | `lanes[*].key_down`, `active_holds[*].key_down` | Compared every lane bit and active hold bit with the simulator's key-state array over exact and random trajectories. Malformed/random actions did not produce an observation mismatch or exception. | No OS key bindings are shown because the interface abstracts physical keys to lanes. |
| Recent outcome and timing bias | `recent_results`, `timing_errors_ms`, `timing_error_summary` | Checked result counts and signed head/tail records after exact, random, late, and hold-break actions. The four exact-play charts reached MAX-only counts and score 1,000,000; synthetic hold boundary tests produced signed head/tail errors. | Error history is bounded to the latest 128 in observations (full latest 2,000 in `info()`), which keeps JSON compact. |
| Score, accuracy, combo, reliability | `score`, `accuracy`, `counts` (MAX/GREAT/GOOD/OK/MEH/MISS), `combo`, `final_combo`, `max_combo`, `miss_rate`, `hold_break_rate`, `score_breakdown()` | Every checked observation satisfied `sum(counts) == judged_objects`; perfect schedules on seeds 100, 101, 103, and 1004 reached 1,000,000, 1.0 accuracy, zero misses, and full max combo. Random trajectories produced finite, nontrivial score/counts. | Score is the documented stable ScoreV1 profile, not current lazer/ScoreV2. |
| Resources and failure state | `resources.health = null`, `resources.health_model`, score/combo fields; `done` | Checked that health is explicitly absent/disabled in every oracle observation and that charts run to completion. | There is no health, fail threshold, opponent, inventory, or persistent resource in this benchmark profile; this is a declared game rule, not an omitted observation. |
| Difficulty/chart context for planning | `key_count`, `od`, `bpm`, `chart_metrics` (density, interval, length, family, pattern mix, hold fraction, release overlap) | Compared key count/OD/BPM and chart metrics with each seeded `Chart`; all declared seeds produce different scenario starts and the train/held-out distributions share the generator. | External `.osu` charts may report `external` pattern mix where mapper metadata is unavailable; native timing/object fields remain decoded. |

## Compactness and validation

The long-trajectory oracle covered 4,560 JSON observations, 76,214 visible
objects, 49,925 visible pattern rows, 959 active-hold states, and 1,760 random
actions (including malformed actions). The largest serialized observation was
35,866 bytes, below the 200 kB contract limit. The four exact schedules also
verified that adding the decoded fields did not change judgement or scoring.

The required validation was re-run after the observation changes and printed:

```text
# Adaptor validation: PASS
```

It passed tier agreement, strict JSON, reset determinism, train/held-out seed
variety, action application, tick pacing, malformed-action handling, and
idempotent `close()`.

## Remaining fidelity boundaries

Nothing actionable from the selected study is left as an undecoded raw field:
lanes, timing, motion-as-time, note kind, chords, pattern cues, holds, key
state, outcomes, errors, score, and the deliberately empty resource model are
named in JSON. The remaining limits are the selected simulator profile itself:
no real client rendering/scroll position, audio, input polling, health model,
notelock, skin, online account, or current lazer rules. Pattern labels for
streams/jacks/anchors are deterministic derived cues rather than a claim that
the native `.osu` format stores those community labels.
