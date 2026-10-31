# Minesweeper adaptor

This is a dependency-free, offline Tier 1 `TurnGame` implementing the local deterministic JSON interface selected in the study. It starts no process, server, GUI, account, or network connection. `game.py:make_game()` returns a fresh instance and `reset(seed)` begins an episode.

The seed selects one of six board families: classic 9x9/10, 16x16/40, and 30x16/99, plus 12x10/18, 20x12/45, and 10x10/15. Selection uses a stable modulo-six mapping, and mine coordinates are sampled with `random.Random(seed)`. The declared train, validation, and held-out groups each contain two examples of every family. A first reveal is excluded from the mine candidates, so it is always safe; flags made before the first reveal do not affect layout generation. Supplying `width`, `height`, and `mines` to `make_game` fixes a custom family while retaining seeded placement.

The observation is a JSON dictionary containing `width`, `height`, `mines`, `status`, `cells`, `safe_revealed`, `safe_total`, `flags`, `remaining_mines`, `turn`, `last_action`, `error`, `won`, and `terminal_reason`. During play, cells are `unknown`, `flagged`, or `revealed:N`; after loss or win, mine cells are shown as `mine` for terminal feedback. The mine map is never exposed during play. `reveal` flood-fills zero regions, `flag` toggles hidden cells, and `chord` opens neighbours only when adjacent flags equal the revealed number. Invalid or illegal actions are ignored and described in `error`.

Actions can be compact strings (`reveal ROW COL`, `flag ROW COL`, `chord ROW COL`, or `noop`) or objects such as `{"type":"reveal","row":3,"col":4}`. Coordinates are zero-based. The public score is `safe_revealed / (width*height - mines)`, so progress remains useful after a loss. `score_breakdown()` reports progress, counts, and a numeric `won` component; `info()` records the seed, board family, and terminal state.

`SEEDS` declares disjoint train, dev, validation, and held-out seed groups. The adaptor is deterministic and needs no environment variables or third-party packages.
