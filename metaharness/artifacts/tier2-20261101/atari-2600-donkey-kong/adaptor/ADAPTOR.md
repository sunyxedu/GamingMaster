# Atari 2600 Donkey Kong adaptor

## Connection

`game.py` runs the maintained local ALE/Stella emulator through Gymnasium's
`ALE/DonkeyKong-v5` environment. It creates the environment with RAM
observations, fixed `frameskip=4`, no renderer, and no sound. OCAtari 2.2.1 is
used in RAM mode as a named-object decoder; the adaptor steps OCAtari's
underlying Gymnasium wrapper directly so Gymnasium's `(terminated, truncated)`
flags are not confused with OCAtari 2.2.1's swapped public return order.

The adaptor verifies the cartridge before construction. The bundled or
`ALE_ROMS_DIR` supplied `donkey_kong.bin` must have MD5
`36b20c427975760cb9cf4a47e41369e4`. It does not copy or redistribute the ROM.
There is no server, account, display, GPU, credential, or network dependency.

## Real-time contract

The returned object is Tier 2. One `tick(action)` applies one ALE action for
four emulated NTSC frames and returns immediately; the runner owns wall-clock
scheduling. `tick_s` is `4 / 60.0` seconds (about 66.7 ms), and `ticks` counts
successfully applied emulator quanta. The default `repeat_action_probability`
is the standard ALE v5 value, `0.25`, so Stella may repeat the previous action.
For deterministic calibration, construct with
`repeat_action_probability=0.0`.

`reset(seed)` always begins at the cartridge's same attract/idle opening. The
cartridge exposes only mode 0 and difficulty 0, so seeds do not choose boards
or levels. Seeds initialize ALE/Gymnasium randomness; with sticky actions
enabled they can produce different traces even though the first observation
is identical. The declared train and held-out seeds are separate RNG streams
of this same fixed scenario, not different level families.

## Actions

`action_spec` exposes all 18 ALE actions as lowercase strings:

- `noop`, `fire`, `up`, `right`, `left`, `down`
- `upright`, `upleft`, `downright`, `downleft`
- `upfire`, `rightfire`, `leftfire`, `downfire`
- `uprightfire`, `upleftfire`, `downrightfire`, `downleftfire`

`fire` is required to leave the initial idle state. UP/DOWN climb only when
aligned with a usable ladder; directional FIRE actions jump/steer. Integer
indices 0--17 are accepted as a convenience. Passing `None` means that no new
controller decision arrived, so the last posted action is repeated (or `noop`
before the first post). A malformed or illegal action is converted to `noop`
and reported in the next observation rather than raising.

## Observation

Every observation is a strict JSON dictionary under 200 kB. It includes:

- `stage` (`idle`, `ramp`, `rivet`, `transition`, `game_over`, or `unknown`),
  frame/tick counters, terminal flags, last reward/action/error;
- `player`, four-slot `hazards` (OCAtari's `Barrel` slots are called barrels
  on the ramp and fireballs on the rivet screen), ten fixed `ladders`, and
  `hammer`, with integer boxes and motion where available;
- `rivet_rows`, the four raw row status bytes (0, 6, 12, or 18 in the normal
  cartridge), and `rivets_removed` (`sum(row // 6)`);
- `bonus_display_bcd`, `current_bonus` (the BCD timer converted to points;
  `$50` is the 5,000-point starting bonus), HUD score bytes, cumulative
  `score_total`/`episode_return`, `screens_completed`, and
  `lives_remaining`; and
- the complete 128-byte `raw_ram` snapshot.

The verified RAM map used by the JSON decoder is: obstacle Y `$80--$83`, jump
state `$85`, hammer state `$86`, score bytes `$87--$88`, game state `$8C`, life
loss `$8D`, screen `$90`, player X `$93`, player Y `$9B`, rivets
`$9F--$A2`, lives `$A3`, bonus `$A4`, obstacle X `$C1--$C4`, directions
`$C5--$C8`, and ladder numbers `$C9--$CC`. OCAtari supplies the player,
hazard, ladder, hammer, and HUD object geometry. Its HUD `Score.value` is not
used as the primary score because it is only a display/timer value.

## Score and limits

`score` is the cumulative **unclipped ALE reward** from every four-frame
step, including point awards and screen-completion bonus. It is not derived
from the one-byte bonus display. `score_breakdown()` additionally reports ALE
reward, current bonus, rivets removed, lives, screens completed, and terminal
indicators.

The adaptor does not provide save-state cloning to the harness and does not
start on the rivet screen; normal evaluation uses full episodes from reset.
ALE's own clone/restore facilities remain available only inside local
calibration experiments, outside this public interface. No screenshots are
required or generated.
