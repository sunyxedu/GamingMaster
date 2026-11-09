# Atari 2600 Frogger — study brief

## Identity and objective

This is the Parker Brothers Atari 2600/VCS release of *Frogger* (1982), not the later Starpath “Official Frogger” cartridge. The objective is to move as many frogs as possible from the bottom sidewalk, across five lanes of traffic and five river rows, into the five home bays at the top while maximising score. The official ALE description is the same objective: cross a highway and a perilous river without being crushed or eaten ([ALE Frogger](https://ale.farama.org/environments/frogger/)).

The board, from bottom to top, is:

1. sidewalk/start area;
2. five road lanes whose traffic alternates direction and has different speeds;
3. a riverbank/median;
4. five river rows containing logs, turtles, diving turtles, alligators and, at harder stages, snakes; and
5. five separate home bays.

The original manual is the authoritative rules source for these mechanics ([Parker Brothers manual, AtariAge](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)).

## Win/progress and failure

- A frog is safe only after entering an unoccupied home bay. Five bays can be filled, but a bay cannot be reused while occupied.
- The player starts with five frogs: one active frog and four reserve indicators. A life is lost on a vehicle collision, water, a diving turtle, an alligator/snake mouth, timeout, or (on Difficulty A) being carried off the screen by a floating object.
- The game ends when no frogs remain. Every five frogs brought home advances to a harder level: traffic becomes harder, floating objects become fewer or change speed, and snakes appear/proliferate.
- The ALE implementation exposes the game’s reserve/life field at RAM address `0xD0` and treats `0xD0 == 0xFF` as terminal; its reset value is 4 because the active frog is not counted in that field ([ALE Frogger implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)). Report both the raw ALE `info["lives"]` and a separately named “active + reserve” interpretation; do not silently call the raw value “five lives.”

## Score

The manual gives the following score events:

| Event | Points |
|---|---:|
| Successful forward jump | 1 |
| Frog reaches a home bay | 5 |
| Five frogs reach home | 100 bonus |
| Lady frog carried home | 20 |
| Fly eaten in a home bay | 20 |
| Each second remaining at home | 2 |
| Extra frog at 1,000 points when fewer than four frogs remain | one extra frog |

ALE decodes the score from RAM addresses `0xCE` and `0xCC` as decimal/BCD and emits reward as the change in that score, so cumulative ALE reward is the most convenient machine score ([Frogger.cpp](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)). The adaptor should additionally expose `raw_score`, `score_delta`, `frogs_home`, `deaths`, `level`, and `terminal`; the latter fields distinguish safe progress from a lucky or incomplete run.

## Controls and actions

The Atari joystick makes a single frog hop in four directions. In games 1–4, move the joystick once for one hop; games 5–6 are “Speedy Frogger” and repeat hops while a direction is held. The manual also defines the console Select, Reset, and left Difficulty switch; the adaptor should start/reset through the emulator rather than fake keyboard input.

ALE’s minimal action set is exactly ([ALE Frogger docs](https://ale.farama.org/environments/frogger/)):

| ALE action | Meaning |
|---:|---|
| 0 | `NOOP` |
| 1 | `UP` |
| 2 | `RIGHT` |
| 3 | `LEFT` |
| 4 | `DOWN` |

Fire is not needed. The full Atari 18-action set is unnecessary and makes action selection less safe.

## Pace and decision budget

This is not turn based. Traffic and river objects move on every emulator frame while the agent is thinking. A run has only 30 seconds from sidewalk to home; the manual gives a warning at five seconds remaining. The ALE v5 default is four emulator frames per agent step with 25% sticky-action probability, but the recommended control configuration for this adaptor is `frameskip=1` and `repeat_action_probability=0.0`, so the harness sees each frame and can make deterministic one-hop/no-op decisions ([ALE Frogger environment source](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)).

Use an engineering decision budget of about **0.5 seconds** for a normal decision. A frog can wait on the sidewalk or median, but a moving car, a river platform edge, or a diving-turtle transition can punish a stale decision in a fraction of a second. Jev should therefore issue short typed action judgments continuously while Astra plans at a slower cadence; the adaptor can advance one frame or a very small fixed batch while Astra is unavailable. A long Astra call is acceptable only when Jev is holding `NOOP` or another already-safe action and the current platform/traffic gap has been checked.

For reproducible experiments, seed `reset(seed)` and report the ALE episode frame number. The official Gymnasium/ALE interface returns RAM observations of 128 bytes, lives, episode-frame number, and frame number; it does not require a display ([ALE Python interface](https://ale.farama.org/python-interface/), [ALE Gymnasium environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)).

## State representation to build

Use the 128-byte RAM observation rather than screenshots. ALE maps the console’s `$80–$FF` RAM into `ram[0:128]`, so a 6502 address `a` is read as `ram[a - 0x80]`. The official game integration gives the high-confidence fields:

- score: `0xCC`/`0xCE` (BCD/decimal score pair);
- reserve/life/terminal: `0xD0`;
- selected single-player game mode: `0xDD` is used by ALE while selecting modes.

A public 2025 community disassembly supplies useful, but **not official**, starting hypotheses for decoding the rest: frog position/state candidates around `0xB0`/`0xB4`, time counter `0xE0`, and collision/object flags around `0xE3`/`0xE5`/`0xE7` ([Parker Brothers Frogger disassembly gist](https://gist.github.com/erik-nilcoast/cc55644c916c729e6a82ae07b681ac2b)). Validate those hypotheses against the exact ALE ROM hash and controlled probes before relying on them. The production JSON observation should contain named fields plus the raw RAM array for auditability, not a screenshot-only representation.

## Sources

- [AtariAge Parker Brothers Frogger manual](https://atariage.com/manual_html_page.php?SoftwareLabelID=194)
- [ALE Frogger environment](https://ale.farama.org/environments/frogger/)
- [ALE Python interface](https://ale.farama.org/python-interface/)
- [ALE Frogger-specific scoring, lives, modes and actions](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/Frogger.cpp)
- [ALE Gymnasium environment implementation](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/python/env.py)
- [Parker Brothers Frogger community disassembly](https://gist.github.com/erik-nilcoast/cc55644c916c729e6a82ae07b681ac2b)
