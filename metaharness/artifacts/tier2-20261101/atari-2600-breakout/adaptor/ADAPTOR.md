# Atari 2600 Breakout adaptor

This adaptor connects the real `ALE/Breakout-v5` Atari 2600 cartridge to the
Meta-Harness tier-2 `RealtimeGame` interface. It uses the official
`ale-py`/Gymnasium environment as the sole emulator, reward, reset, seed, life,
and termination owner. The emulator runs headlessly with RAM observations; no
window, display server, GPU, account, network service, or screenshot is used.

## Configuration and connection

`game.py` calls `gym.register_envs(ale_py)` and constructs one
`ALE/Breakout-v5` environment per game instance with:

- `obs_type="ram"`, `render_mode=None`, `mode=0`, `difficulty=0`;
- `full_action_space=False`, `frameskip=1`;
- `repeat_action_probability=0.25` (the v5 sticky-action baseline); and
- `max_num_frames_per_episode=108000`.

The optional `ALE_ROMS_DIR` setting is not required with the tested
`ale-py==0.12.1` wheel, which supplies the packaged ALE ROM. If that package
cannot use its packaged ROM, set `ALE_ROMS_DIR` to a directory containing a
legally supplied, ALE-compatible ROM; do not copy a ROM into this adaptor.
The ROM's distribution and use remain subject to the applicable Atari and ALE
licenses/terms.

No credentials are read. `env.example` contains the optional ROM override.
`requirements.txt` pins the tested Python packages for the evaluator venv.

## Timing and actions

This is tier 2. `tick_s` is exactly `1/60` seconds, the native Atari cadence
for the selected `frameskip=1` configuration. `tick(action)` synchronously
applies one action and advances exactly one native ALE frame; the Meta-Harness
runner owns the wall-clock ticker and calls it at that cadence. This is
intentional: adding an adaptor-owned stepping thread would race the runner and
advance the single emulator twice. The emulator itself is protected by a
lock for evaluator/harness access.

The legal action names are `NOOP`, `FIRE`, `RIGHT`, and `LEFT`. Their Gym
indices are respectively `0`, `1`, `2`, and `3`; the adaptor asserts this
mapping at construction. `FIRE` serves an initial or replacement ball. `NOOP`
removes directional input but is not promised to brake the paddle immediately.
The runner's mailbox repeats its current/last action on stale ticks, while a
malformed direct action is converted to `NOOP` and recorded in the next
observation's `action_error` instead of raising.

## Observation and decoding

Every observation is a fresh strict-JSON-compatible dictionary under 200 kB.
It includes:

- native and global frame counters, tick count, last reward, cumulative raw
  ALE reward (`score`), and the separately decoded cartridge score;
- `score_match`, lives from official Gym info plus a RAM cross-check, paddle
  rectangle, and ball rectangle or `null` while waiting for a serve;
- ball `dx`/`dy` and a validity flag (velocity is invalid on an
  absent-to-present serve transition);
- the exact six-by-18 `brick_bitmap`, occupied brick records with row values
  `7,7,4,4,1,1`, remaining-brick count, wall index, walls cleared, and the
  two-wall completion flag; and
- action/serve/termination flags. The evaluator-only `info()` reports the
  reset seed and full configuration; the model observation does not expose the
  scenario identifier.

The normal decoder path calls the pinned, read-only OCAtari 2.2.1 Breakout
RAM helpers (`_make_block_bitmap` and `detect_objects_ram`) after every reset
and ALE step. The exact bitmap is authoritative; OCAtari's grouped variable-
width block objects are not used for counting. A small equivalent bitmap
fallback is retained for deployments where OCAtari cannot import. That fallback
preserves the Breakout bit permutation from OCAtari's MIT-licensed decoder; the
normal execution path uses the installed package rather than a second emulator.

The primary `score` is the unmodified cumulative ALE reward. The
`cartridge_score` uses the Breakout BCD RAM formula and is reported as a
cross-check rather than replacing reward. `score_breakdown()` and `info()`
also report lives, bricks, wall progress, elapsed emulated frames, and whether
both walls were cleared. A complete two-wall ordinary game is expected to
reach 864 points, but ALE's natural termination is controlled by lives; the
108000-frame cap is reported as truncation.

## Seeds and limits

`SEEDS` declares train seeds `0..7` and held-out seeds `100..107`. All are
ordinary mode-0/difficulty-0 episodes; the brick layout is fixed by the ROM,
so seed variation primarily affects the seeded serve/sticky-action random
stream rather than producing new maps. `reset(seed)` passes the integer
straight to Gymnasium/ALE and recreates the decoder state. With the pinned
configuration, a fresh instance reset with the same seed reproduces the first
RAM observation and trajectory.

Difficulty 1 and other ALE modes are intentionally not folded into these
baseline seeds: they change paddle or cartridge-variant rules and should be
run as separately labelled robustness experiments. The adaptor accepts
`mode`, `difficulty`, and other constructor settings as keyword overrides for
such diagnostics, but it requires `frameskip=1`; the published seed contract
and observation geometry describe the baseline.

There is no adaptor-owned process or server. `close()` closes the one ALE
instance, is idempotent, and is called after every evaluator episode.
