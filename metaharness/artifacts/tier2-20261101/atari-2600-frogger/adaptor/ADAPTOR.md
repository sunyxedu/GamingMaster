# Atari 2600 Frogger adaptor

## Connection and lifecycle

`game.py` uses `ale-py==0.12.1` and `gymnasium==1.4.0` through the
headless ALE/Stella `ALE/Frogger-v5` environment. It loads the bundled Parker
Brothers Atari 2600 Frogger ROM and verifies MD5
`081e2c114c9c20b61acf25fc95c71bf4`. No display, GPU, server, account, network,
or credential is needed. `close()` closes the emulator and is idempotent.

Each instance is isolated. `reset(seed)` resets ALE, waits through the
cartridge's own attract/start phase (the five-action minimal set has no FIRE),
then advances the live emulator with NOOP for a deterministic 1--90 frame
seed phase. The startup and phase lengths are exposed in both `info()` and the
observation; they are not counted in agent `ticks`. This matters because a bare
ALE reset has the same initial traffic arrangement for every seed.

## Tier and actions

This is Tier 2. `tick(action)` applies one action to one Atari display frame;
`tick_s = 1/60` and `ticks` counts agent-applied frames. The runner owns the
clock. Legal action strings are:

- `NOOP` -- centered joystick / wait for this frame.
- `UP` -- one hop toward the river and homes.
- `RIGHT`, `LEFT`, `DOWN` -- one lateral or backward hop.

In standard games (modes 0 and 1), a direction is edge-triggered as one hop;
controllers should use NOOP between deliberate hops. Mode 2 is Speedy Frogger,
where holding a direction can repeat. A malformed action is safely converted
to NOOP and recorded as `invalid_action` in the next observation.

## Observation and decoder

Every observation is strict JSON and currently about 10 kB. `raw_ram` retains
all 128 bytes for `$80--$FF`, while named fields decode the state needed for
play:

- `frog`: row/zone, world x, TIA render x, platform-carry flag, and state.
  Rows 0--4 are road lanes, 5 is the median, 6--10 are river rows, `0xff` is
'the start sidewalk, and 11 is the home approach.
- `road` and `river`: five rows each, with two tracked objects per row. Each
  object has the cartridge world x, wrapped collision interval estimate, TIA
  position byte, direction, measured velocity in pixels/tick, shape pointer,
  NUSIZ size code, palette/color code, kind, and hazard flag. River color/shape
  decoding identifies brown `log`, olive `turtle`, and blue `diving_turtle`
  carriers; unfamiliar later-level forms remain explicitly
  `river_hazard_or_carrier` rather than being silently called safe.
- `homes`: five fixed bay centers (x 16, 48, 80, 112, 144), exact occupancy
  from `E3 & 0x1f`, availability, and the current target during home entry.
  `bonus_flags` and raw `C7/F9` are retained because this ROM does not expose a
  separate home-crocodile byte. `crocodile` is therefore `unknown`, never a
  false safety claim.
- `lady_frog` and `fly`: presence, row, x, and render position from the ROM's
  bonus-object fields.
- `score`, `score_delta`, `reward`, and `total_reward`: score is the on-screen
  packed-BCD value from `$CC/$CE`, and reward is ALE's per-frame score change.
- `reserve_frogs`, `frogs_remaining_including_active`, `deaths`, `timer_s`,
  `time_bonus`, `time_bonus_available`, `frogs_home`, `level`, terminal flags,
  and `terminal_reason`.
- `mode`, `difficulty`, `rom_hash`, `frameskip`, `sticky_action_probability`,
  requested seed, ALE RNG seeds, startup/phase frames, and frame counters.

Object tables are decoded from the exact ROM pointers held at `C0/C2`, `B8/BA`,
`BC/BE`, and `C4`; the alternating direction table is the cartridge's
`$FD47` table. `B0/B1/B2` are the direct 0--160 world x values and `B4/B5/B6`
are the render-position bytes. The raw bytes and table codes remain alongside
the friendly names so a later mode or level can be handled without losing
information. RAM/object positions were compared with ALE RGB rendering for
three seeds, modes 0/1, and 0/30/90/180 ticks; see `review.md` and the
runtime calibration report.

## Score and scenarios

`score` is the decoded on-screen score and is the primary evaluator score;
`total_reward` is cumulative ALE reward and should agree with score changes.
The manual events are 1 per forward hop, 5 per home, 100 for five homes, 20
for a lady frog, 20 for a fly, and 2 per second remaining. `score_breakdown()`
reports score, ALE reward, score delta, cumulative homes, deaths, reserve frogs,
awarded/available time bonus, and level. The current timer-derived bonus is
also exposed separately so a controller does not confuse possible points with
points already awarded.

The declared `SEEDS` train/held-out split is mode 0, difficulty 0. Every seed
uses a distinct seed-selected live traffic phase. `SCENARIO_SEEDS` provides
separate eight-seed groups for mode 1 (harder single-player), mode 2 (Speedy
Frogger), and difficulty 1 (floating objects wrap rather than killing the frog).
Use `make_game(mode=..., difficulty=...)` for those groups; scores are never
mixed between configurations. ALE mode and difficulty, ROM hash, frameskip,
sticky-action probability, and all seed metadata are recorded per episode.

## Limits

ALE's public RAM interface does not provide a first-party object graph or a
separate home-crocodile variable. The adaptor exposes exact RAM flags and
marks that one home hazard as unknown rather than inventing it. NUSIZ intervals
are conservative render/collision estimates; controllers can use the exact
world x, TIA byte, shape pointer, and raw RAM for their own margins. The
bundled cartridge remains subject to the user's applicable ROM distribution
rights.
