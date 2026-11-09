# Malody observation review

## Scope and method

The study selects the local deterministic native-`.mc` Key emulator, not the
official Malody client. The emulator has no renderer, display, audio device, or
client memory to inspect. Therefore the authoritative comparison for this review
is the emulator's own decoded chart/state (`_notes`, `_timing`, `_scenario`) plus
independent checks of beat-to-millisecond conversion and action outcomes. A real
client rendering comparison is not possible under the supplied offline/headless
constraints; that limit is listed for every affected item below.

I audited all 16 declared seeds (8 train and 8 held-out), with up to 250 logical
clock ticks per episode and clock-only completion. The audit made **251,800
state assertions**, included every declared family, and observed:

- all 4K, 5K, 6K, and 7K charts;
- all eight pattern families and all decoded patterns (`tap`, `stream`, `jack`,
  `chord`, `anchor`, `hold`);
- `none`, `Flip`, and `Const` mods;
- 562 native notes checked for raw/effective lane transforms;
- maximum serialized observation size 10,081 bytes (under the 200 kB contract);
- exact 40 ms logical cadence on every non-terminal tick.

## Player-state coverage

| What a strong player perceives and acts on | Observation field(s) | How it was checked | Still missing / limit |
|---|---|---|---|
| Episode identity and chart selection | `episode.seed`, `chart.id`, `chart.native_mode`, `episode.mode` | Reset all 16 seeds; IDs and modes matched the generated native chart and stayed deterministic on fresh instances. | No official client chart/package metadata or audio title is loaded. |
| Key layout and lane mapping | `episode.columns`, `controls.column_indexing`, `controls.column_mapping`, note `column` and `raw_column` | Checked mapping length for every seed and all 562 notes. For Flip, verified `column = K - 1 - raw_column`; for none/Const, verified identity. | There is no physical keyboard: mapping is the zero-based action grammar, not a device scan code. |
| Difficulty and chart-level context | `chart.difficulty`, `chart.pattern_family`, `chart.snap_resolution`, `chart.mod` | Compared observation metadata to `_scenario` on every seed; all eight families, key widths, snap labels, and supported mods appeared. | Difficulty is the generator's deterministic proxy, not an official Malody level rating. |
| Current absolute timer and real-time pace | `episode.time_ms`, `episode.tick`, `episode.tick_s` | Compared every audit observation with the internal logical clock; 40 ms cadence error was 0.0 ms on non-terminal ticks. | It is logical chart time, not audio/playback or display-vsync time. After terminal state the clock intentionally freezes. |
| BPM and timing changes | `chart.bpm_segments`; each note `bpm`, `start_beat`, `end_beat`, `start_ms`, `end_ms` | Recomputed every note with `_beat_to_ms` across constant and changing-BPM seeds; all start/end values matched within floating-point tolerance. | Time-signature/audio-clock behavior outside the generated subset is not modeled. |
| Positive/negative chart offset | `chart.offset_ms`; converted note milliseconds | Compared the chart offset with native `meta.offset` and recomputed note times for all seeds. Both signs occurred. | No external audio latency calibration exists. |
| What is where | Each `upcoming` note's `column`, `raw_column`, `group_columns`, `start_ms`, `in_ms` | At every audit tick, upcoming IDs, lanes, and timers were compared with the unresolved internal notes. | There is no pixel x/y position; Key lanes are represented semantically by zero-based columns. |
| What kind of object it is | `upcoming[].kind`, `upcoming[].pattern`, `active_holds[].kind/pattern` | Compared each decoded object to `_Note.kind` and the parser's pattern annotations over all seed families. | Pattern labels are emulator classifications, not a client skin's visual glyphs. |
| How notes move toward the receptor | `upcoming[].motion = falling_toward_receptor`, `in_ms`; active `motion = held_at_receptor` | Verified motion labels and ETA against `start_ms - episode.time_ms` at every non-terminal audit tick; active hold motion was forced and checked on hold seeds. | No scroll speed, screen coordinate, animation frame, or audio-driven movement is available. ETA and logical motion are the faithful available representation. |
| Exact-time chords | `group_id`, `group_size`, `group_columns`, equal `start_ms` values; chord `pattern` | Checked group IDs/columns against the internal equal-start groups at every tick; generated chord families produced two-/three-lane groups. | No rendered chord flash or input-device simultaneity jitter is simulated. |
| Streams and rhythm subdivisions | `pattern`, `start_beat`, `start_ms`, `bpm`, `in_ms` | All stream families were observed; note timing/ordering was checked against the native beat triples and BPM map. | Human finger ergonomics and visual subdivision cues are not modeled. |
| Same-lane jacks | `pattern = jack`, `column`, ordered `start_beat`/`start_ms` | Jack-family notes and pattern annotations were present in both seed groups; unresolved-note order matched internal lane order. | Stamina, key-repeat, and physical key bounce are not simulated. |
| Anchors / outer-lane references | `pattern = anchor`, `column`, `group_columns` | Anchor-family observations contained the expected boundary-lane annotations and their internal columns. | No hand/finger assignment or rendered lane geometry is inferred. |
| Holds and final releases | `upcoming[].end_beat/end_ms`, `held_columns`, `active_holds[].end_ms`, `remaining_ms`, `head_judgment`, `stats.pending_holds` | Exact head presses created active holds in four hold-containing audit episodes. Perfect schedules cleared all 16 seeds with score 1.0; an early release produced `broken_hold` and incremented `broken_holds`. | Client-specific hold-body/tail rules are not verified; this emulator uses the documented explicit release rule. |
| Current held input state | `held_columns`, `active_holds` | Compared these fields to all internal `note.active` values at every audit tick and after forced hold heads. | Physical key-down state outside an active chart hold is not exposed because taps are instantaneous events. |
| Read-ahead timers | `upcoming[].in_ms`, `start_ms`, `end_ms`, `active_holds[].remaining_ms`, `episode.judgment_windows_ms` | Checked ETA arithmetic and tail remaining-time arithmetic at every audit tick; C timing boundaries were tested at 0/40/41/56/57/81/82/111/112/200/201 ms. | The displayed game's visual receptor distance is unavailable; timing values are the control truth. |
| Judgment feedback and calibration | `last_judgments[].judgment`, `offset_ms`, `at_ms`, `note_id`, `column`, `raw_column`, `combo_after`; `stats.judgment_histogram` and `tail_judgment_histogram` | Controlled timing actions produced the expected C buckets: Big P, p1, p2, p3, G, and M at the tested boundaries. | Public windows are RM documentation used as a transparent Key-emulator proxy, not verified official Key judgment. |
| Combo and recovery state | `stats.combo`, `stats.max_combo`, `stats.misses`, `stats.broken_holds`, recent judgment `combo_after` | Perfect and noisy/early/late actions were run; miss and broken-hold outcomes matched the emulator rules and all noop episodes terminated. | Official client combo edge cases may differ. |
| Accuracy and score objective | `score`, `stats.accuracy`, `stats.completion`, `stats.rm_proxy_2025`, `stats.rm_proxy_normalized`, `score_breakdown()` | Perfect timestamped batches scored 1.0 on all 16 seeds; timing-bucket tests produced the documented weights; proxy values stayed finite. | `rm_proxy_2025` is explicitly RM-only reference math, not native Key scoring. |
| Completion / clear result | `episode.done`, `stats.clear`, `stats.terminal_status`, `stats.completion` | Clock-only runs reached terminal state on all 16 seeds; `clear` agreed with `done and misses == 0 and broken_holds == 0`. | There is no official result screen or rank/grade. |
| Resources and failure risk | `controls.resources = {"modeled": false, "available": []}` | Checked the explicit resource declaration on every seed. The chosen Key simulator has no lives, health, stamina meter, power items, or consumable resources. | Real Malody effects such as Death are outside the selected scenario set; no unmodeled resource is silently implied. |
| Action errors / infrastructure feedback | `action_feedback.error`, `action_feedback.last_action`, `info().valid_trial` | Malformed actions were applied without raising and produced feedback; local episodes report `valid_trial: true`. | No external server/connection failure exists to exercise; the adaptor starts no infrastructure. |

## Changes made from the review

- Added decoded fractional beat fields, BPM-at-note, pattern class, exact-start
group ID/size/columns, and semantic motion to upcoming notes.
- Added `active_holds` with effective/raw lane, head judgment, tail deadline, and
remaining time.
- Added explicit zero-based lane mapping, judgment windows, tick cadence, and the
no-resource declaration.
- Added pending-head/hold counts, clear/terminal status, and raw/effective columns
to judgment feedback.
- Updated `describe()` and `ADAPTOR.md` to enumerate these fields and their limits.

## Remaining fidelity gaps

The observation now exposes every state variable available in the selected local
emulator. What remains unavailable is deliberately documented: official Malody
rendering and scroll coordinates, audio timing/latency, physical keyboard jitter,
client-specific hold behavior, official native Key judgment/score calibration, and
real client effects such as Random/Hide/Death. These cannot be decoded from a
headless local emulator without violating the supplied no-account/no-display/
no-audio constraints.
