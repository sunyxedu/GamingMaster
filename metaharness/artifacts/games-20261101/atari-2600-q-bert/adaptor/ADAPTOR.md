# Atari 2600 Q*bert adaptor

## Connection and setup

`game.py` owns one local `ALE/Qbert-v5` Gymnasium environment backed by
ALE/Stella. It is headless (`obs_type="ram"`, `render_mode=None`); no display,
GPU, account, server, or credential is needed. `requirements.txt` installs the
pinned ALE/Gymnasium pair and OCAtari 2.2.1's maintained Q*bert RAM decoder.
The constructor verifies that `ale_py.roms.get_rom_path("qbert")` is a regular
file with MD5 `484b0076816a104875e00467d431c2d2`. A private legal ROM can be
selected with the optional `ALE_ROMS_DIR` environment variable shown in
`env.example`; the adaptor never downloads or accepts an unverified ROM.

The default is the study's robust evaluation configuration:

- `mode=0`, `difficulty=0` (novice), `frameskip=4`;
- `repeat_action_probability=0.25` (ALE v5 sticky actions);
- minimal six-action ALE space, no full action space;
- `tick_s = frameskip / 60`, so the default tick quantum is about 66.7 ms.

`difficulty=1` is supported separately for expert/A play and adds the Red Ball
hazard. `frameskip=1, repeat_action_probability=0.0` is supported for a
separate deterministic decoder calibration run, not the robust default.

The Atari cartridge spends about 150 emulated frames in its attract/boot
sequence. `warm_start=True` (the declared evaluator default) sends bounded
NOOPs during reset until Q*bert's playable Level 1 Round 1 opening is present.
Those bootstrap frames do not increment the public `ticks` counter and do not
change the score. Set `warm_start=False` to expose the literal frame-zero
cartridge state instead.

## Interface

The returned object is Tier 2 `RealtimeGame`. The evaluator calls
`tick(action)` on its own clock; `tick` never sleeps or starts a worker thread.
One call advances one ALE `frameskip` step and increments `ticks` once. `None`
means a missed decision: it repeats the last valid posted action, or uses
`NOOP` before the first action. A malformed action is converted to `NOOP`, does
not replace the posted action, and is reported in `action_error` rather than
raising. Names and integer conveniences are:

```
NOOP=0, FIRE=1, UP=2, RIGHT=3, LEFT=4, DOWN=5
```

`FIRE` is deliberately exposed because it is part of ALE's legal minimal set,
but it is not a normal movement action and can restart the cartridge. The
four directional names are joystick edges on the Q*bert diamond. `safe_moves`
is derived from the ROM's 21-node graph and excludes pyramid edges that do not
exist at the estimated `qbert_node`; a hop is animated, so the state may be
between nodes and the estimate should be rechecked before committing a long
macro.

## Observation and decoding

Every observation is strict JSON and contains the complete 128-byte RAM view as
`raw_ram` plus compact decoded fields:

- `frame`, `global_frame`, `tick`/`ticks`, `lives`, the `resources` map (life losses,
  extra lives, and next scheduled extra-life round), `difficulty`, `level`, `round`,
  `round_in_level`, `completed_rounds`, and `completed_levels`;
- cumulative `score`/`cumulative_ale_reward`, the BCD `decoded_score`, and
  `score_match`;
- `qbert` rectangle and pixel motion, estimated `qbert_node`, `qbert_in_hop`,
  `qbert_motion` (`standing`/`hopping`/`falling`/`not_visible`), and `timers`
  containing the verified hop-animation timer, object scheduler, enemy spawn
  scheduler bytes, falling flag, and sprite-state bytes;
- 21 `cube_state`/`cube_phase` labels (`S`, `I`, `D`, or `unknown`), raw cube
  palette bytes, fixed-node coordinates, color names, `destination_cubes`,
  `intermediate_cubes`, `incomplete_cubes`, and `needed_landings`;
- named `red_ball`, `purple_ball`, `coily`, `sam`, and `green_ball` objects
  with kind, position and motion when visible; absent objects are `null`;
- `hazards` with lethal/non-lethal classification, estimated node, pixel
  distance to Q*bert, and a conservative risk label; `legal_moves` is the
  geometry-only graph set, while `blocked_moves` and `safe_moves` account for
  an immediately occupied hazard node;
- `disc_exits` with rendered position/availability, plus `last_reward`, event
  counters, `last_action`, malformed-action diagnostics, `terminal`/`terminated`,
  `truncated`, `done`, and termination reason.

The observation is normally about 5--6 kB, far below the 200 kB contract.

OCAtari supplies the ROM-specific 21-cube/player/disc/Purple Ball/Coily/Sam/
Green Ball extraction from RAM. Its Player/character slots can retain a
placeholder during an animated edge fall, so the adaptor cross-checks the
current native RGB frame internally against Atari palette components and uses
the rendered component to correct Q*bert/enemy visibility, position, and
motion. OCAtari's current Q*bert list has no Red Ball slot, so expert mode also
uses the yellow/red-ball palette component and exposes only its rectangle/motion,
never the binary frame. If a sprite is between palette frames it is reported
as absent rather than as a stale RAM object. The two disc entries report
availability but intentionally leave `validated_exit` unset: the Atari port's
correct Coily lure exit must be learned/calibrated rather than guessed from the
arcade game.

The cube tracker follows the manual's five finite cycles. It uses the actual
RAM palette transition and keeps raw bytes beside the named phase, so a Sam
reversion or Level 3-5 reversible cycle does not get flattened into a boolean
"visited" flag. `needed_landings` is the number of transitions to the next D
phase (or `-1` when the cartridge is in an uncalibrated/blank state).

## Seeds and scenarios

`SEEDS` provides eight train and eight disjoint held-out integer seeds. All
seeds use the same fixed Level 1 Round 1 opening and 21-cube board; Q*bert has
no normal randomized board layout. The seed selects ALE's seeded stochastic
stream, principally the v5 sticky-action realization, so held-out episodes
measure control robustness rather than puzzle-layout transfer. The same seed
mix is available for `difficulty=1` by constructing with that keyword; novice
and expert results should be recorded separately.

The adaptor's ordinary scenario is a fresh Level 1 Round 1 reset. It tracks
progress into Levels 1-5, four rounds per level, the repeating Level 5, lives,
colour cycles, Coily/discs, Sam, Green Ball, and life loss as they occur during
play. It does not use RAM writes, cheats, or a save state during ordinary play.
Advanced later-level experiments should use a separately recorded,
ALE-version-matched safe action prefix or an RNG-inclusive ALE clone/restore
calibration harness; no unverified warm-start state is silently substituted by
this adaptor. `warm_start` only skips the cartridge attract screen and never
claims to select a later level.

## Score and limits

`score` is cumulative unmodified ALE reward. `decoded_score` independently
uses the Q*bert wrapper's verified BCD bytes (`RAM[0x5b]`, `[0x5a]`, `[0x59]`)
and `score_match` is the consistency check. `score_breakdown()` reports ALE
reward, decoded score, cube/Sam/Green/Coily/round event points, completed
rounds, lives, life losses, extra-life awards, and terminal/truncation flags. ALE deliberately
returns zero on Q*bert's terminal game-over step; the observation and
`info()` expose `terminal_step_reward_zero` so a final life loss is not
mistaken for a negative score event.

`info()["valid_trial"]` is false only for an emulator/connection/decoder
infrastructure failure and includes `invalid_reason`; bad actions are handled
inside the episode. `close()` closes the Gymnasium/ALE instance, releases its
native emulator state, and is idempotent. The adaptor starts no external
process. No screenshots, logs, caches, or generated code are written while it
runs.
