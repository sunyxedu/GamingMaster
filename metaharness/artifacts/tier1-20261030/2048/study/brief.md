# 2048 brief

## Rules and objective

2048 is a single-player, turn-based puzzle on a 4x4 grid. A move slides every tile as far as possible in one of four directions (up, down, left, right); adjacent equal tiles on that line merge once, in the direction of travel, into a tile whose value is their sum. A tile cannot merge twice during one move. After a move that changes the board, one empty cell receives a new tile: 2 with probability 0.90 or 4 with probability 0.10 in the reference implementation. The game starts with two randomly placed tiles. The objective is to create a 2048 tile, while continuing to maximize score and the largest tile when possible. The original implementation and its rules are in the open-source repository: [gabrielecirulli/2048](https://github.com/gabrielecirulli/2048), especially [`js/grid.js`](https://github.com/gabrielecirulli/2048/blob/master/js/grid.js) and [`js/game_manager.js`](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js). A concise rules overview is also available at [Wikipedia: 2048](https://en.wikipedia.org/wiki/2048_(video_game)).

The score starts at zero. Every merge adds the value of the newly created tile to the score (2+2 adds 4, 4+4 adds 8, and so on); ordinary slides and spawned tiles add nothing. The reference score bookkeeping is in [`game_manager.js`](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js). Play ends when no empty cell exists and no adjacent equal pair can merge. Reaching 2048 does not force termination in the original game: the player may continue.

## Controls and state changes

The browser game accepts arrow keys and, on supported devices, swipe gestures. The canonical keyboard mapping is Up/Down/Left/Right; its input manager is documented in [`js/keyboard_input_manager.js`](https://github.com/gabrielecirulli/2048/blob/master/js/keyboard_input_manager.js). For an adaptor, represent actions as the four strings `up`, `down`, `left`, and `right` (optionally reject an action if it would be a no-op). An observation should contain the 16 integer cells, current score, move count, game-over flag, and RNG state/seed when deterministic replay is needed.

This is tier 1: the world waits between moves. A move is processed synchronously, then the random tile is spawned and the next decision is available. There is no real-time clock or timeout in the normal game, so a decision can take seconds or minutes without changing the board. The evaluator may cap wall-clock time externally, but game quality is determined by the selected move sequence over a fixed number of turns, not reaction speed.

## Evaluation-relevant progress

Report both the cumulative score and the maximum tile. Score rewards making many high-value merges; maximum tile captures the main objective and prevents a strategy that farms small merges from appearing strong. For held-out seeds, reset should reproduce the same initial board and random stream so that runs are comparable while still testing generalization.

Sources: [official 2048 repository](https://github.com/gabrielecirulli/2048), [Game Manager source](https://github.com/gabrielecirulli/2048/blob/master/js/game_manager.js), [Grid source](https://github.com/gabrielecirulli/2048/blob/master/js/grid.js), and [Wikipedia](https://en.wikipedia.org/wiki/2048_(video_game)).
