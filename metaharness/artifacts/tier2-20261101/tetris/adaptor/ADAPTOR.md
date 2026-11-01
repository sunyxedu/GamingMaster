# Tetris adaptor

## Connection and engine

This adaptor is a faithful, self-contained local implementation of a modern
Guideline-style falling-blocks game. It does not open a display, use a GPU,
connect to a server, or require an account. `game.py` owns the board, deterministic
seven-bag randomizer, SRS wall kicks, hold, gravity, lock delay, line clearing,
T-spin classification, and scoring. The evaluator's Python process is the game
client, so `close()` only closes the in-process instance and is idempotent.

The implementation intentionally exposes the symbolic state directly rather than
converting it to pixels. It uses no third-party runtime package or credentials.

## Tier and clock

This is Tier 2 (`RealtimeGame`). `tick_s` is 0.05 seconds (20 Hz). Every evaluator
call to `tick(action)` counts one applied clock tick: the named action is processed,
then gravity and lock delay advance by exactly 0.05 seconds. Thus a slow model does
not pause the game; the runner's clock decides when the next tick arrives. There is
no hidden background thread. A malformed action is ignored, counted, and the last
valid action is used for that tick; it never raises from `tick`.

The normal `standard` scenario starts with an empty 10x20 board at level 1. The
piece sequence is a deterministic seven-bag stream derived from `reset(seed)`, and
held-out seeds use the same rules without queue-specific tuning. `make_game` also
accepts `scenario="low"`, `"medium"`, `"high"`, or `"recovery"` for controlled
scenario evaluations: medium starts at level 5, high at level 10, and recovery
starts at level 3 with a fixed uneven board containing holes and a right-side well.
These optional starts do not alter the standard seed semantics.

Episodes end on top-out, after 200 placed pieces, or after 20,000 ticks by default;
`max_pieces` and `max_ticks` are constructor options. The observation reports the
terminal reason, score, cumulative lines, level, pieces, survival duration, top-out,
and board-quality metrics.

## Actions

`action_spec` contains:

- `left`, `right`: one-column movement.
- `soft_drop`: one cell downward, worth one point if it moves.
- `rotate_cw`, `rotate_ccw`: SRS rotation with wall/floor kicks.
- `hard_drop`: move to the ghost position, worth two points per cell, and lock now.
- `hold`: store or swap the piece, once per falling piece.
- `noop`: no input; gravity and lock delay still advance.

Actions are normally strings. `{"action": "left"}` is accepted as a convenience.
There are no parameterised actions, so no additional grammar is needed.

## Observation

`observe()` returns strict JSON under 200 kB. `board` is a settled 20-row board in
top-to-bottom order (`0` is empty and `1..7` identify I/J/L/O/S/T/Z). `active` and
`ghost_cells` contain exact absolute x/y cells (y increases upward), with active
piece type, orientation, x, and y. `hold`, `can_hold`, and the next five `queue`
pieces are explicit. The state also includes score components, lines, level, combo,
back-to-back status, gravity interval, lock timer, ticks, survival time, events from
the last tick, seed, scenario, illegal-action count, stack holes/aggregate height/
bumpiness, top-out height, and terminal status.

## Score policy

The score follows the requested Guideline-compatible policy: 100/300/500/800 times
level for one/two/three/four lines; 100/200 for mini T-spin with zero/one line;
400/800/1200/1600 for T-spin with zero/one/two/three lines; one point per soft-
dropped cell; two per hard-dropped cell. A combo starts at -1, becomes 0 after the
first clear, and adds `50 * combo * level` for later consecutive clears. Tetris and
line-clearing T-spins are difficult clears; consecutive difficult clears receive a
1.5x back-to-back multiplier. A non-difficult line clear breaks back-to-back. There
is no perfect-clear bonus, garbage, or multiplayer mode. `score_breakdown()` and
`info()` expose numeric components and quality metrics so drop points are not the
only evaluation signal.

## Seeds and limits

Training seeds are 0--7 and held-out seeds are 101--107; the lists are disjoint and
all use the standard empty level-1 start. `DETERMINISTIC=True`: a fresh instance
reset with the same seed has the same first observation. Controlled level/recovery
starts are opt-in constructor scenarios rather than hidden seed cases. The engine
is deliberately offline and does not need `env.example` credentials.
