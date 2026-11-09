# Atari 2600 Q*bert — playing brief

## Scope and version

This study is for the Atari 2600 Parker Brothers cartridge, not the original arcade game or the later Atari-branded release. The ALE-supported `qbert.bin` is the 4 KiB Atari 2600 ROM identified by MD5 `484b0076816a104875e00467d431c2d2`; the ALE source binds its Q*bert reward/life wrapper to that checksum ([ALE QBert settings](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). The scanned Parker Brothers manual is dated 1983 ([manual scan/OCR](https://archive.org/details/Q-bert_1988_Atari)); use the ROM supplied by ALE rather than silently substituting an arcade ROM.

## Objective and board

Q*bert starts on the top cube of a pyramid. Hop onto every cube until the whole pyramid reaches its destination colour; completing a pyramid ends the round and produces a new destination colour. The home conversion has a six-row pyramid (21 cubes), rather than the arcade game's 28-cube pyramid; this port-specific count is documented by a contemporary play review ([Carstairs, Atari 2600 review](https://carstairsconsiders.blogspot.com/2013/07/video-game-review-qbert-for-atari-2600.html)). The manual itself states the objective and progression but does not give the row count ([Parker Brothers manual](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=1224&itemTypeID=HTMLMANUAL)).

There are five levels, four rounds per level. After Level 5, Level 5 Round 1 repeats. Q*bert, the enemies, and the frequency of enemy appearances become faster/more frequent as rounds advance. The game starts with four Q*berts (lives); one extra is awarded after round 5 and another after every four rounds thereafter. The game ends when the supply is exhausted ([manual, progression and scoring pages](https://archive.org/details/Q-bert_1988_Atari)).

## Cube colour rules

Let `S` be the starting colour, `I` an intermediate colour, and `D` the destination colour. A jump applies the following state transition to the cube landed on:

| Level | Rounds | Transition when landing | Consequence |
|---|---:|---|---|
| 1 | 1–4 | `S -> D`, `D -> D` | Each cube normally needs one useful landing. |
| 2 | 1–4 | `S -> I -> D`, `D -> D` | Each cube needs two useful landings. |
| 3 | 1–4 | `S -> D -> S` | A landing on an already-complete cube undoes it; revisit deliberately. |
| 4 | 1–4 | `S -> I -> D -> I` | A complete cube cycles back to `I`; revisit to finish it. |
| 5 | 1–4 and repeat | `S -> I -> D -> S` | A complete cube cycles back to `S`; revisit until all are `D`. |

This table is transcribed from the manual's “Reaching the Destination Color” page ([manual scan/OCR](https://archive.org/details/Q-bert_1988_Atari)). Do not assume the arcade version's colour schedule.

## Characters and hazards

* **Red Ball:** rolls down the pyramid and kills Q*bert on contact. Red Balls do not appear in the manual's novice setting.
* **Purple Ball:** a lethal rolling hazard; if it reaches the bottom it hatches Coily.
* **Coily:** the snake pursues Q*bert. The reliable counter is to lure him toward a flying-disc exit, then board the disc as Coily reaches the exit so Coily falls off. Each disc has one correct exit cube; boarding from the wrong cube kills Q*bert.
* **Sam:** cannot catch Q*bert, but changes landed cubes back toward their original colour. Run into Sam to catch him and score a bonus.
* **Green Ball:** cannot catch Q*bert. Catching it freezes every other character (not Q*bert) briefly and awards a bonus.
* **Flying discs:** emergency transport to the pyramid top. Their number and placement vary by level/round; they are also the main Coily trap. These behaviours and the one-correct-exit warning are in the manual ([manual pages 3–5](https://archive.org/details/Q-bert_1988_Atari)).

The Atari port omits some arcade-only enemies and has a simplified pyramid; do not import arcade Ugg/Wrong-Way/Slick behaviour into the state model. A port review confirms the omitted enemies and notes that the home controls require anticipating rather than reacting to the slower/unresponsive movement ([VGMuseum review](https://www.vgmuseum.com/reviews/2600/qbert/)).

## Score and progress

| Event | Points |
|---|---:|
| Change a cube to its destination colour | 25 |
| Catch Sam | 300 |
| Catch Green Ball | 100 |
| Lure Coily off the pyramid | 500 |
| Complete a round | 3,100 |

The manual gives these values and the extra-life schedule ([manual scoring page](https://archive.org/details/Q-bert_1988_Atari)). The adaptor should report:

1. cumulative score (the sum of ALE reward deltas, or the decoded in-game score),
2. current lives, level and round when decoded,
3. number of cubes at `D` and the colour state of each cube, and
4. whether the episode ended by game over or by an evaluator truncation.

ALE's Q*bert wrapper reads the score from the game RAM and returns the score delta as reward; it deliberately returns zero on the terminal step ([QBert.cpp](https://github.com/Farama-Foundation/Arcade-Learning-Environment/blob/main/src/ale/games/supported/QBert.cpp)). Thus the cumulative non-terminal reward is the meaningful score, not the final terminal-step reward. Score is a better primary measure than survival alone, while lives and completed rounds prevent a high-score report from hiding reckless play.

## Controls and actions

The manual says to hold the joystick with its four corners making a baseball diamond and the fire button at the top; Q*bert moves in the four diagonal board directions. In ALE, the physical joystick directions are represented by these action IDs:

| ALE action | Meaning | Q*bert use |
|---:|---|---|
| 0 | `NOOP` | Hold position; normally only for an intentional wait. |
| 1 | `FIRE` | Fire/reset input; do not use during play unless the adaptor explicitly wants a reset. |
| 2 | `UP` | One board-diagonal direction. |
| 3 | `RIGHT` | One board-diagonal direction. |
| 4 | `LEFT` | One board-diagonal direction. |
| 5 | `DOWN` | One board-diagonal direction. |

The mapping is the official ALE mapping ([Q*bert environment](https://ale.farama.org/environments/qbert/), [action specification](https://ale.farama.org/environments/#action-space)). The board's “up-right”/“down-left” names depend on the manual's rotated joystick orientation; calibrate the four action IDs once from the top cube and retain that mapping. ALE's minimal Q*bert set is exactly these six actions; the 18-action full Atari set is unnecessary.

A reset switch starts a new manual game. The Gymnasium adaptor should use `reset()` rather than injecting `FIRE`; the native ALE API exposes `reset_game()`.

## Pace and tier

The Atari 2600 world advances continuously at roughly one NTSC frame per 1/60 second. The game samples joystick input while Q*bert and enemies animate, and the manual explicitly says both speeds increase with rounds. With `ALE/Qbert-v5`, one environment action is repeated for four emulated frames and is therefore about 66.7 ms of game time; ALE documents the fixed four-frame skip and 25% sticky-action default ([ALE environment defaults](https://ale.farama.org/environments/qbert/), [stochasticity settings](https://ale.farama.org/environments/)). A bad direction can be fatal in a single hop, so the practical safe synchronous decision budget is about **0.10 s at fast rounds** (roughly one or two four-frame decisions), not the seconds Astra may take to reason.

This is **tier 2**: the world does not wait for a model response. Jev should run the short-horizon action queue and answer recognition/choice questions quickly; Astra should plan routes, colour schedules, and adaptation asynchronously. A 0.5-second Jev call cannot be placed directly on every hop without buffering several safe actions or running the emulator faster than wall-clock. For exact calibration/search, `frameskip=1` exposes every frame; for the normal harness, fixed `frameskip=4` is a useful compromise.

## References

* [ALE Q*bert environment specification](https://ale.farama.org/environments/qbert/)
* [ALE Python interface: RAM, ROMs, state cloning](https://ale.farama.org/python-interface/)
* [Parker Brothers Atari 2600 Q*bert manual](https://atariage.com/manual_html_page.php?SystemID=2600&SoftwareID=1224&itemTypeID=HTMLMANUAL)
* [Searchable manual scan/OCR](https://archive.org/details/Q-bert_1988_Atari)
* [Atari 2600 review and port-specific board details](https://carstairsconsiders.blogspot.com/2013/07/video-game-review-qbert-for-atari-2600.html)
* [Atari 2600 control/response review](https://www.vgmuseum.com/reviews/2600/qbert/)
