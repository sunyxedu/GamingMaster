# Sokoban adaptor

This is a Tier 1, turn-based Sokoban game. `game.py` implements a small offline
XSB engine and loads a pinned subset of the [Boxoban levels](https://github.com/google-deepmind/boxoban-levels)
repository. The bundled corpus is `runtime/boxoban/{train_000,valid_000}.txt`,
from Boxoban revision `78011d8c86ad423bfb2e1005012393b4c3c24409`; its license
notice is retained beside the levels. The two files contain 2,000 fixed levels
and their SHA-256 values are recorded in `info()` at runtime.

`reset(seed)` chooses a level with SHA-256 of the decimal seed modulo the
corpus size. This mapping is stable across processes and is included in
`info()` as `level_index` and `level_id`. The declared train, held-out, dev,
and validation seed groups are disjoint and use the same corpus distribution.
Set `SOKOBAN_LEVELSET` to a directory of `.txt`/`.xsb` Boxoban/XSB files (or one
XSB file) to use another pinned offline corpus; when unset, the bundled subset is
used. `SOKOBAN_MAX_ACTIONS` or the `max_actions` constructor argument can set
the per-episode action cap.

The observation is strict JSON: an ASCII board, dimensions, player/box/goal
coordinates, reachable player cells, valid actions, transition event, counters,
and termination state. Actions are `up`, `down`, `left`, `right`, and `noop`.
Walking moves one square; pushing moves an adjacent box when its destination is
free and increments `pushes`. Walls and blocked pushes consume an action and
leave the board unchanged. The episode ends when every goal is occupied or the
action cap is reached. Malformed actions are ignored and reported in the next
observation rather than raising.

The numeric score gives solved levels priority, then boxes on goals, then fewer
pushes and actions. `score_breakdown()` reports `solved`, `boxes_on_goals`,
`pushes`, `walk_steps`, `total_actions`, and `action_cap_terminated`.
There are no servers, accounts, credentials, or background processes; `close()`
is idempotent and only clears the in-memory episode.

The staged data hashes are `train_000.txt` =
`674dda499261deface252c90077ae67181b5c7704c235baf0dd0bd2d16bf072e` and
`valid_000.txt` =
`52cd811af721d69eef71a4e3cfc40cd4f635269a8e26d42ec017aa43af02b8a9`.
