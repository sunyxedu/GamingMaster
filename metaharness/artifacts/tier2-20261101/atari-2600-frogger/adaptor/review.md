# Observation review — Atari 2600 Frogger

The review used the exact ALE ROM loaded by the adaptor (MD5
`081e2c114c9c20b61acf25fc95c71bf4`), headless RGB frames only as a calibration
oracle, and the production RAM decoder for observations. The object alignment
run is retained at `runtime/calibration/object_render_alignment.json`: three
seeds (101, 139, and 211), modes 0/1, and samples at 0, 30, 90, and 180 agent
ticks, covering 240 decoded objects per sample set and more than 1,000 live
emulator frames. The raw x/TIA positions, row bands, alternating directions,
and palette/shape forms were checked against the rendered board.

## Strong-player perception and action checklist

| Moment / what a strong player needs | Observation field | Check and result | Remaining gap |
|---|---|---|---|
| Start: know that the frog is at the bottom and how much time/resources remain | `frog.zone`, `frog.row`, `frog.x`, `timer_s`, `reserve_frogs`, `frogs_remaining_including_active`, `frame`, `startup_frames` | Reset across seeds 101/139 and all three secondary configurations. The cartridge starts with `AC=0xff`, `B0=80`, `D0=4`, and `E0` near 30. The adaptor waits for `DE=0x80` before exposing agent ticks, so the first action is not lost in the attract phase. | Frog pixels are not needed for control; exact frog sprite bounding width is not exposed. |
| Road: what is in each of five lanes, where it is, its direction/speed/size, and whether it is dangerous | `road[row].objects[]`: `x`, `x_intervals`, `velocity_px_per_tick`, `direction`, `kind=vehicle`, `hazard`, `nusiz`, `shape_ptr`, `color_code`, `tia_x` | The ROM update loop maps `$80-$84/$8b-$8f` to road rows 0--4 and `$96-$9a/$a1-$a5` to render positions. Across 240 sampled objects, raw x moved in the rendered direction and the TIA bytes placed the corresponding colored vehicle envelopes in the expected lane. Render row centers are recorded as `screen_y_px`. | `width_px_estimate` is deliberately conservative for NUSIZ copies; the exact sprite collision mask is retained as `shape_ptr`/`nusiz` rather than claimed as a rectangle. |
| Median: safe pause between traffic and river | `median.row=5`, `median.safe_pause=true`, frog `zone=median` | The rendering loop treats Y=5 specially and the screen's yellow median is between road and river. RAM AC transitions through 5 when a real UP hop is accepted. | No missing gameplay field. |
| River: find a carrier row, distinguish log/turtle/diving turtle, predict drift, and detect being carried | `river[row].objects[]`: x/interval, measured velocity, direction, `kind`, `color`, `shape_ptr`, `nusiz`, `hazard`; frog `moving_with_platform`; `difficulty` | The ROM's C0/C2 sprite, B8/BA NUSIZ, BC/BE color, C4 activity, and FD47 direction tables are decoded per row. RGB checks showed brown color code 9 as logs, olive code 6 as turtles, and blue code 10 as diving turtles. Consecutive samples showed river object x changing while NOOP was applied; `E7&0x80` is surfaced as the carry flag. | Exact alligator/snake names for unfamiliar later-level shapes are not guaranteed; they are emitted as `river_hazard_or_carrier` with raw shape/color. |
| Keep frog on a broad platform and away from screen edge | frog `x`, `moving_with_platform`, river object `x_intervals`, `difficulty`, raw `hazard_e7` | World x is the cartridge's direct 0--160 coordinate; the TIA render byte is retained for the edge/wrap case. The object report includes wrapped intervals when an estimated envelope crosses 160. Difficulty is recorded and is not inferred from a generic strategy. | Carrier support is a ROM flag plus geometric candidates, not a perfect per-object collision ID. |
| Home approach: choose an unoccupied bay and avoid a bad target | `homes[]` with slot, center x, occupied/available/target; frog row/zone; `ram_fields.home_occupancy_mask`, `collision_row_e5` | Disassembly and controlled ALE RAM probes confirm that `E3&0x1f` is the five-bit occupancy mask checked against `$1f`; bay centers are the five fixed top openings at x 16/48/80/112/144. `target` is only asserted while AC=11, avoiding a false target during ordinary play. | The Parker Brothers ROM does not expose a separate home-crocodile/head byte through ALE RAM. `homes[].crocodile` is explicitly `unknown` and `bonus_flags.home_crocodile` says so; no unsafe “safe bay” claim is fabricated. |
| Lady frog and fly bonuses | `lady_frog`, `fly`, `bonus_flags`, raw B1/B2/AD/AE/F9/C7 | The disassembly renders bonus objects from AD/AE and B1/B2/B5/B6. Reset and later NOOP samples showed absent sentinels (outside rows 0--10) and valid row/x candidates when the fields became active. | A fly in a home bay is not separately named; the active fly object and raw C7 flags are exposed. |
| Timer and urgent hazards | `timer_s`, `time_bonus_available`, `terminal_reason`, river `hazard`, raw E0/E5/E7 | E0 starts at 30 and decrements during live frames; the timer was observed changing during hundreds of NOOP frames. Blue river carriers are flagged as diving hazards. Death probes with random actions showed D0 decrement and `deaths`/`life_lost` updates. | A collision's exact manual label (car, water, mouth, timeout) is not always distinguishable from ALE's RAM-only terminal transition; timeout is identified when the old E0 was <=1, otherwise `life_lost`. |
| Score and progress | `score`, `score_delta`, `reward`, `total_reward`, `frogs_home`, `homes_occupied_current_level`, `deaths`, `reserve_frogs`, `level`, `score_breakdown()` | ALE Frogger.cpp defines score at CC/CE and reward as score change. A real UP hop produced score/reward 1. Random episodes showed reward accumulation matching decoded score. E3 occupancy transitions are counted cumulatively for homes; raw D2/variation DD are retained for progression. | Time bonus is reported both as awarded estimate and currently available `2*timer`; ALE does not label individual manual bonus events in its Python observation. |
| Clock/action semantics | `action_spec`, `tick_s`, `tick`, `last_action`, `invalid_action`, `frame` | Generic validation passed for all five actions, malformed input, one-frame ticks, and pace. The emulator itself continued moving objects during NOOP. | None for the fixed five-action interface. |

## Calibration conclusions

- The production observation is structured object state, not screenshot-only:
  five road rows, five river rows, two ROM-tracked objects per row, frog,
  homes, bonus candidates, timer, resources, and score are named fields.
- `raw_ram` remains in every observation so a controller can inspect an
  unfamiliar later-level shape and so future decoder repairs do not require a
  new interface.
- The only intentionally unresolved item is a separate home-crocodile/head
  variable: the exact cartridge/RAM interface does not provide one. The
  observation and `describe()` state that limitation explicitly.
- `validate_adaptor(..., expected_tier=2)` passes after these changes. A
  1,000-tick random-action run produced finite JSON, changing traffic, score
  from forward hops, four life-loss events, and a valid `no_frogs` terminal.
