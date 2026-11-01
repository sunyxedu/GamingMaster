# Atari 2600 Ms. Pac-Man — game brief

## Identity and objective

The target is the Atari 2600 cartridge **Ms. Pac-Man (Atari CX2675, 8-KB ROM)**, not the arcade/MAME Ms. Pac-Man ROM. The AtariAge manual describes the home version as having multiple mazes, floating fruit and pretzels, four ghosts, and a short end-of-game cartoon. The Internet Archive entry identifies the 1982 Atari/GCC cartridge as CX2675; the ALE ROM used by the recommended local interface is an 8,192-byte ROM with MD5 `87e79cd41ce136fd4f72cc6e2c161bee` (the checksum is declared by ALE's MsPacman settings).

The player steers Ms. Pac-Man through the maze, eating every dot. Clearing all dots flashes the maze and starts a new maze. The score-maximising objective also rewards energy pills, moving fruit/pretzels, and ghosts eaten while frightened. A normal ghost kills Ms. Pac-Man on contact; the game consumes a life and restarts her after the death animation unless the game is over.

*Sources:* [AtariAge Atari 2600 manual](https://atariage.com/manual_html_page.php?SoftwareID=1169), [Internet Archive cartridge entry](https://archive.org/details/atari_2600_ms._pac-man_1982_atari_-_gcc_mark_ackerman_glenn_parker_cx2675), [ALE MsPacman settings and checksum](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.hpp).

## Rules and scoring

The manual's scoring table is the authoritative scoring prior for this port:

| Object/event | Points |
|---|---:|
| Ordinary dot | 10 |
| Energy pill | 50 |
| Cherry | 100 |
| Strawberry | 200 |
| Orange | 500 |
| Pretzel | 700 |
| Apple | 1,000 |
| Pear | 2,000 |
| Banana | 5,000 |
| First ghost eaten on one energy pill | 200 |
| Second ghost on that pill | 400 |
| Third ghost on that pill | 800 |
| Fourth ghost on that pill | 1,600 |

The four ghost awards are consecutive on the **same** energy pill; the multiplier resets for a later pill. Ms. Pac-Man starts with three lives and receives one bonus life after the first 10,000 points. The Atari manual says energy pills turn ghosts blue only briefly; uneaten ghosts blink before returning to normal, while eaten ghosts' eyes return to the centre chamber.

The variation selected on the cartridge is a ghost-count choice:

* **Cherries / mode 0:** standard/arcade-style variation with four ghosts.
* **One Bear / mode 1:** one ghost, intended for beginners.
* **Two Bears / mode 2:** two ghosts.
* **Three Bears / mode 3:** three ghosts.

ALE's source confirms four available modes, maps mode 0 to the standard variation, and maps modes 1–3 to one, two, and three ghosts. All variations start at the easiest cherry level and advance through the fruit levels to banana. After banana, the manual says mazes alternate between maze three and maze four; after every two maze clears, fruit selection becomes random and frightened-ghost time decreases.

*Sources:* [AtariAge manual, sections 2, 5 and 6](https://atariage.com/manual_html_page.php?SoftwareID=1169), [ALE MsPacman C++ settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/master/src/ale/games/supported/MsPacman.cpp).

## Controls and action semantics

On the physical Atari 2600, use the joystick in the **left controller port**. Up, down, left and right steer Ms. Pac-Man. The joystick also enters the warp tunnels. Pressing GAME RESET or the red joystick button starts the selected game; GAME SELECT cycles the variation. The difficulty switches are unused and TV TYPE only selects colour/black-and-white display.

The local ALE interface exposes the minimal 9-action set:

| ALE action | Meaning |
|---:|---|
| 0 | NOOP |
| 1 | UP |
| 2 | RIGHT |
| 3 | LEFT |
| 4 | DOWN |
| 5 | UP+RIGHT |
| 6 | UP+LEFT |
| 7 | DOWN+RIGHT |
| 8 | DOWN+LEFT |

Use the five cardinal/no-op actions in the adaptor. The diagonal actions exist because they are part of the ALE minimal set, but the game is a four-direction joystick game and there is no need to plan diagonals. There is no FIRE action required for normal Ms. Pac-Man play. The full 18-action Atari set can be requested, but it adds irrelevant buttons.

Ms. Pac-Man moves continuously once started: this is not a turn-based game. A direction is held through the emulator frames until another direction is sent. At a junction, a turn must be selected before the character reaches the turn; waiting for a slow model response can therefore cause a wall hit, a missed escape turn, or a ghost collision.

*Sources:* [AtariAge manual, sections 3 and 4](https://atariage.com/manual_html_page.php?SoftwareID=1169), [official ALE MsPacman environment page](https://ale.farama.org/environments/ms_pacman/).

## Maze, tunnel, and ghost behaviour

The manual says the tunnels are on the two **sides** of the maze: leaving on the right re-enters on the left and vice versa. It explicitly warns that different mazes have different tunnel locations. Treat a tunnel as a wrap edge, not as a normal dead-end corridor.

A strong default is to clear dots safely rather than chase ghosts. Stay near (but not directly on) an energy pill, draw ghosts toward it, then consume it when several are close enough to chain the 200/400/800/1,600 awards. If survival matters more than score, take the pill earlier and use a side tunnel to break pursuit. The manual recommends using the openings as escape tunnels and practising dot-clearing before attempting ghost hunts.

The published Atari 2600 strategy guide by Robert Guerra describes four distinct maze layouts with different tunnel placements and dangerous passages. Its useful port-specific priors include: learn the tunnel locations rather than assuming the arcade map; use tunnels for escape because ghosts seldom choose them; and clear the vulnerable lower/inverted-T section early in the third-maze situation before all ghosts are roaming. The guide also warns against waiting beside a lower energy pill without watching the lower corners. These are priors, not facts to hard-code blindly: confirm them against the exact CX2675 ROM and the selected mode.

Do not import arcade ghost target algorithms, arcade level-speed tables, or an arcade maze map without validation. Atari 2600 Ms. Pac-Man has its own layouts, timing, ghost-count variations, and port-specific quirks. The AtariAge manual and the ALE ROM are the appropriate rules sources; the strategy guide is a useful human-play supplement.

*Sources:* [AtariAge manual survival hints](https://atariage.com/manual_html_page.php?SoftwareID=1169), [Robert Guerra's Atari 2600 strategy guide](https://gamefaqs.gamespot.com/atari2600/584919-ms-pac-man/faqs/2632), [AtariAge strategy discussion](https://forums.atariage.com/topic/7503-whats-your-ms-pac-man-strategy/).

## Progress and evaluation signals

Use these signals together rather than treating a single pixel or object count as the whole objective:

1. **Primary score:** cumulative ALE reward, which is the score delta, or the decoded HUD score. It reflects dots, pills, fruit, pretzels, and ghost chains.
2. **Survival:** remaining lives, deaths, and whether the episode is terminal.
3. **Maze progress:** number of remaining visible pill objects / dots eaten and completed maze transitions. A maze completion is more meaningful than a brief score spike.
4. **Level/maze context:** current fruit/maze colour and the learned map identifier. Do not assume an object decoder's two pill-grid templates are a four-maze index.
5. **Risk-adjusted performance:** score at game over, mazes completed, and lives remaining. Ghost/fruit hunting can increase score while reducing expected completion, so report both score and progress.

ALE's 128-byte RAM and its reward wrapper are sufficient for score, lives, actor positions, fruit, power-pill presence, and dot bitfields. However, OCAtari's own paper notes that Pac-Man/Ms. Pac-Man maze geometry is not encoded in RAM. A symbolic adaptor must therefore retain a calibrated wall/turn map (learned once from RGB frames or supplied as a verified CX2675 map) while using RAM for dynamic objects. OCAtari's MsPacman decoder is useful but should be cross-checked: its `Life.value` is the displayed spare-life count, while ALE's `info["lives"]` includes the life currently being played; its ghost objects do not expose a reliable frightened-state flag.

*Sources:* [OCAtari paper](https://arxiv.org/html/2306.08649v2), [OCAtari MsPacman RAM decoder](https://github.com/k4ntz/OC_Atari/blob/master/ocatari/ram/mspacman.py), [ALE Python interface](https://ale.farama.org/python-interface/).

## Pace and tier

Atari emulation runs at 60 frames per second. The documented `ALE/MsPacman-v5` default repeats an action for four frames, so one environment step represents about 4/60 = **0.067 seconds** of game time; v5 also has 0.25 sticky-action probability unless overridden. With `frameskip=1`, the adaptor can observe every 0.0167 seconds. Real decisions are most urgent near an intersection, tunnel entrance, or approaching ghost; a safe corridor can tolerate a held action for longer.

Use **tier 2**. The world ticks while Astra reasons, and a slow call cannot block the emulator. A practical danger horizon is roughly **0.2 seconds** (about 12 native frames, or three default ALE steps): a nearby ghost or missed turn can punish a stale direction within that interval. Jev should issue typed cardinal-direction choices rapidly from the current JSON state, while the harness keeps a short action queue running and Astra replans asynchronously. A 0.5-second Jev call is too slow for unbuffered frame-by-frame control, so the adaptor should batch a direction for a few frames and interrupt the batch on a hazard/turn event.

*Sources:* [ALE MsPacman defaults](https://ale.farama.org/environments/ms_pacman/), [ALE environment specification](https://ale.farama.org/env-spec/).