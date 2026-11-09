# Atari 2600 Space Invaders

## Scope and canonical game

This study targets the Atari VCS/2600 cartridge version of **Space Invaders** (the ROM bundled by current ALE identifies itself as `Space Invaders (1978) (Atari) [!]`, 4 KiB, NTSC). It is not the arcade ROM and not an Atari 8-bit port. The cartridge's scanned Atari manual is dated 1980, while the game itself is the 1978 Atari release. The practical emulator target is `ALE/SpaceInvaders-v5`.

Primary sources:

- [Atari manual, HTML transcription](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)
- [Atari manual, scanned PDF](https://atarionline.org/wp-content/uploads/2021/11/space-invaders-1978-atari-2600-game-instructions-manual.pdf)
- [ALE SpaceInvaders environment](https://ale.farama.org/environments/space_invaders/)
- [ALE installation and interfaces](https://ale.farama.org/getting-started/)
- [Computer Archaeology Atari 2600 technical pages](https://www.computerarcheology.com/Atari2600/SpaceInvaders/)

## Objective and termination

The player controls a laser cannon at the bottom of the screen. Destroy the formation before the lowest invader reaches Earth/the bottom, and avoid losing the cannon to enemy laser bombs. A normal wave contains **36 invaders in six rows of six**. When all 36 are destroyed, another set appears; the manual says each reset set starts closer to Earth, so surviving waves become progressively more urgent. The game ends when the invaders reach Earth or the player has taken the three enemy hits/lost all lives. In Gymnasium/ALE this is reported as `terminated=True`; `info` also exposes a `lives` count.

The long-term objective is the score, but survival and clearing waves are the useful progress measures for an agent. Keep total reward/score, waves cleared or inferred from formation resets, invaders killed, lives remaining, and terminal cause where the adaptor can obtain them.

## Screen, controls, and actions

A one-player game uses the joystick in the left controller port:

- Joystick left/right moves the cannon horizontally along the bounded bottom lane.
- The red controller button fires the laser beam.
- Only one player shot can be in flight: after firing, the cannon cannot fire again until that beam hits a target or leaves the top of the screen.
- Game Select chooses one of 112 cartridge variations; Game Reset starts the selected game.
- The difficulty switch changes cannon size: the manual describes the `B` position as a smaller cannon and `A` as a larger cannon that is more exposed to enemy fire.

The recommended ALE Gymnasium action space is `Discrete(6)`:

| Gym action | Meaning |
|---:|---|
| 0 | NOOP |
| 1 | FIRE |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | RIGHTFIRE |
| 5 | LEFTFIRE |

Do not confuse this compressed Gym mapping with the numeric values of the full ALE `Action` enum. If using `ALEInterface` directly, use `getMinimalActionSet()`/the enum values rather than assuming Gym's indices.

## Formation, fire, shields, and UFO

The invader block traverses horizontally and steps downward at the sides. Destroying aliens makes the remaining block move faster (a standard and observed Atari-2600 pressure mechanic); consequently the end of a wave is more dangerous than its opening. Enemy bombs fall toward the cannon, and the cannon must dodge while maintaining firing alignment.

Three shields sit between the formation and the cannon. They provide temporary cover, but both player fire and enemy bombs wear them away. Gaps eventually let both kinds of beam/bomb through, and the shields disappear as the formation gets close. The command alien/UFO periodically crosses the top: it is harmless and does not drop bombs, but is a bonus target. In the normal one-player games it is worth 200 points; the manual specifies 100 in the two-player competitive variants C and D.

The manual's helpful hints specifically warn not to concentrate so hard on the UFO that an enemy bomb hits the cannon, and warn that the bottom movement bars are the cannon's limits, where it can be trapped by fire.

## Scoring

The value is fixed by the invader's **initial row**, not by how far it has descended:

| Initial row (bottom to top) | Points |
|---:|---:|
| 1 | 5 |
| 2 | 10 |
| 3 | 15 |
| 4 | 20 |
| 5 | 25 |
| 6 | 30 |

A complete 36-invader wave is 630 points. The four-digit display tops out at 9999 even though scoring can continue. ALE reward is the score delta, so summing rewards is the primary score. The adaptor should not substitute the arcade game's variable UFO scoring for this fixed Atari-2600 table.

## Variations and difficulty

The cartridge includes 112 game variations, including moving shields, zigzagging bombs, fast bombs, invisible invaders, and one-/two-player competitive or partnership modes. The normal single-player game is the best canonical benchmark. ALE exposes `mode` values 0–15 and `difficulty` values 0–1; document the selected values in an episode record because these settings change the task. `mode=0`, `difficulty=0` should be the baseline unless the evaluator explicitly asks for a variation.

## Pace and tier

This is **tier 2**: the game is a continuously animated action game, not a turn-based world. NTSC Atari video is nominally about 59.92 Hz ([Atari 2600 specifications](https://problemkaputt.de/2k6specs.htm)). ALE v5's default `frameskip=4` advances four emulator frames per `step` (about 66.8 ms of game time), and its default sticky-action probability is 0.25 ([ALE environment specification](https://ale.farama.org/environments/)). The formation, bombs, player shot, and UFO can all change while the agent is deciding.

Use a tight low-level control loop: Jev should choose a short action/hold and be queried again at roughly one ALE step or a small multiple of it; Astra should plan between batches rather than blocking the loop. A practical safe reaction budget is about **0.25 s** of game time; at a dangerous bomb or late-wave alignment, even 0.1 s can matter. The JSON deliverable uses 0.25 seconds as the conservative model-level decision budget, while the adaptor may use `frameskip=1` for calibration and `frameskip=4` for normal operation.

## References

The manual is the authoritative source for the 36-invader wave, controls, one-shot limitation, hits, shields, UFO, scoring, and cartridge variations. The [AtariAge strategy discussion](https://forums.atariage.com/topic/15341-space-invaders-strategies/) and [StrategyWiki walkthrough](https://strategywiki.org/wiki/Space_Invaders/Walkthrough) provide player heuristics such as clearing outer columns; treat those as community strategy rather than additional ROM specification. ALE's [SpaceInvaders page](https://ale.farama.org/environments/space_invaders/) supplies the machine interface, action mapping, observation shapes, frame skip, reward description, modes, and difficulty values.
