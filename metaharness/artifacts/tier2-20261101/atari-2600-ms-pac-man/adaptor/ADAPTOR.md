# Atari 2600 Ms. Pac-Man adaptor

## Connection and setup

`game.py` runs a local, headless Atari Learning Environment (ALE) emulator using the
Ms. Pac-Man ROM bundled by `ale-py==0.12.1`. It does not contact a server, require an
account, or read credentials. Install `requirements.txt` into the evaluator's Python
3.12 environment. `SDL_VIDEODRIVER=dummy` is an optional headless setting; normal
operation requests no display and no RGB frames.

OCAtari 2.2.1 is used in RAM Extraction Mode. It decodes the raw 128-byte Atari RAM
into Ms. Pac-Man, ghost, fruit, power-pill, pill, HUD score, and life objects. The
adaptor retains the raw RAM in every observation for cross-checking reverse-engineered
fields.

## Interface and timing

This is a Tier 2 `RealtimeGame`. Each `tick(action)` advances one fixed ALE action
quantum, and the evaluator's clock calls it at `tick_s`. The default is four Atari
frames at 60 Hz (`frameskip=4`, `tick_s=0.066666...` seconds), with the documented
v5 sticky-action probability fixed at `repeat_action_probability=0.25`. ALE's seeded
sticky-action draws make otherwise identical train and held-out seeds diverge while
`reset(seed)` remains reproducible. The emulator runs faster than wall time; `tick()`
does not sleep. Before the first action, `noop` is current; a missing action repeats the
previous action. Malformed actions are ignored and the current action is applied, so a
late command cannot crash the episode.

The action vocabulary is:

- `noop`, `up`, `right`, `left`, `down`
- `up_right`, `up_left`, `down_right`, `down_left`

Cardinal directions are recommended. The Atari action is a joystick direction rather
than a turn-based move, and Ms. Pac-Man and the ghosts continue moving between model
decisions. The ROM's automatic opening sequence can consume the first several action
quanta before movement begins.

## Observations

Observations are strict JSON-sized dictionaries. They include:

- episode frame, tick count, cumulative ALE reward (`score`), decoded HUD score,
  last reward, lives, deaths, terminal state, and last action;
- player position, dimensions, velocity, and direction (RAM byte 56);
- four stable ghost slots with position/velocity/colour/visibility;
- fruit position/visibility and a best-effort fruit kind decoded from RAM byte 123;
- four stable power-pill slots and currently visible ordinary pills with OCAtari
  `row`/`col` grid coordinates;
- visible collectible count, eaten count, maze counters, configured ghost mode, and
  the original 128-byte RAM array;
- a maze note. Dynamic objects are in RAM, but wall geometry is not. `maze.map_confidence`
  is therefore `0.0` and `known_walls_or_turn_graph` says that geometry must be inferred
  or calibrated from RGB rather than pretending that OCAtari's pill grid is a wall map.

`score` is cumulative ALE reward, which is the score delta for dots, pills, fruit and
ghost chains in this ROM. `hud_score` is reported as a separate cross-check. The
optional `score_breakdown()` reports score, visible progress, mazes cleared, lives,
deaths, mode, and terminal state so a score-only strategy is not mistaken for strong
survival or maze completion.

## Seeds and scenarios

`DETERMINISTIC` is true. `reset(seed)` reseeds ALE and starts the fixed Atari opening
state; it does not select a later maze. The declared train seeds are `0..7` and held-out
seeds are `100..103`, disjoint reproducible scenario seeds. With the main
`EVAL_KWARGS`, mode `0` is the scored four-ghost stratum. The constructor accepts
`ghost_mode=1`, `2`, or `3` for separate one-, two-, and three-ghost curriculum/
robustness strata. ALE's mode mapping is:

| `ghost_mode` | configured ghosts | meaning |
|---:|---:|---|
| 0 | 4 | main evaluation |
| 1 | 1 | curriculum |
| 2 | 2 | curriculum |
| 3 | 3 | robustness |

Later mazes are reached only by actually clearing the current maze (or by an explicitly
verified emulator checkpoint, which this minimal adaptor does not silently create).
The `maze_number` and `mazes_cleared` fields are bookkeeping for such progression.
The Atari RAM/OCAtari decoder does not reliably expose frightened/blue ghost state;
normal ghost colour is retained but should not be interpreted as frightened status.

## Limits and legal asset note

This adaptor uses the ROM supplied by the installed ALE package and does not copy or
redistribute ROM bytes. Use it only under the applicable Atari/Namco/ALE terms. The
object decoder is version-pinned and its RAM coordinates are reverse engineered; raw
RAM is included so downstream consumers can detect decoder changes. There is no
screenshot path in normal observations and no GUI, GPU, server, or account dependency.
