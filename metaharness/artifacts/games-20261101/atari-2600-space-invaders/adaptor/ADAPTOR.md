# Atari 2600 Space Invaders adaptor

## Connection and setup

`game.py` uses the maintained local `ale-py`/Gymnasium `ALE/SpaceInvaders-v5`
environment.  Stella runs in-process and headlessly; no display, GPU, account,
server, or credential is needed.  The current `ale-py` package supplies the
verified `space_invaders.bin` ROM.  If an evaluator intentionally keeps ALE ROMs
elsewhere, it may set the optional `ALE_ROMS_DIR` shown in `env.example`.

The adaptor's default benchmark configuration is explicit:

- `mode=0`, `difficulty=0` (canonical one-player rules)
- `frameskip=4` (four NTSC frames per action, about 66.7 ms)
- `repeat_action_probability=0.25` (ALE v5 sticky-action RNG)

`reset(seed)` calls Gymnasium's seeded reset on a fresh episode.  The initial
cartridge opening is normally identical, but the seed controls ALE's seeded
random/sticky-action stream and therefore bomb/shot/action trajectories.  The
module declares train seeds `0,1,2,3` and held-out seeds `100,101`; they are
scenario RNG seeds, not level numbers.  Optional constructor kwargs can expose
other ALE modes/difficulty settings for experiments, but those are not covered
by the canonical decoder and are recorded in `info()`.

## Interface

The returned object is a tier-2 `RealtimeGame`.  Each `tick(action)` posts a
valid action and synchronously advances exactly one configured ALE step.  If the
action is missing, the last valid action is repeated (and `noop` is used before
the first action).  A malformed action is ignored and reported in the next
observation rather than raising an evaluator-fatal exception.  `ticks` counts
successful emulator steps.  `close()` closes the Gym/ALE instance and is
idempotent.

Legal action names are `noop`, `fire`, `right`, `left`, `right_fire`, and
`left_fire`.  Gym's six-action mapping is used: NOOP, FIRE, RIGHT, LEFT,
RIGHTFIRE, LEFTFIRE.  The combined actions move and fire on the same tick.

## Observation and decoder

Gym receives the 128-byte RAM observation, while the adaptor reads ALE's current
RGB buffer internally to build object-level JSON.  It does not expose a binary
screenshot or raw RAM to the harness.  The compact observation contains:

- mode, difficulty, frame, ALE score, lives, game-over/terminal cause, inferred wave,
  waves cleared, and invaders killed;
- player x/y and one decoded player laser;
- all 36 invader row/column entries, current lattice x/y, alive flag, and row
  value (5 through 30 from top to bottom);
- gray bomb positions, three shield positions/occupancy masks, and the purple
  UFO position when present;
- a malformed-action warning field when applicable.

The decoder is calibrated to the packaged NTSC ROM: invader sprites are matched
against their known 6x6 lattice, shield masks are read from their fixed regions,
and projectile motion between observations distinguishes rising lasers from
descending bombs.  Coordinates are 160x210 screen coordinates.  RGB decoding is
ROM-specific; PAL, altered palettes, invisible-invader modes, moving shields,
and multiplayer modes may make object fields approximate.  `info()` always
records `mode`, `difficulty`, `frameskip`, sticky-action probability, terminal
cause, and validity diagnostics.

## Scoring and limits

`score` is the sum of ALE rewards, i.e. the Atari score: invaders award 5, 10,
15, 20, 25, or 30 by initial row, a full wave is 630, and the normal UFO is 200.
`score_breakdown()` additionally reports lives remaining, inferred waves cleared,
invaders killed, UFO hits, and survival ticks.  `info()` reports the terminal
cause (`all_lives_lost`, `game_over`, `time_limit`, or an infrastructure error).

The adaptor starts no child process and writes no run-time files.  ALE's packaged
ROM remains a local dependency; this adaptor does not redistribute it.
