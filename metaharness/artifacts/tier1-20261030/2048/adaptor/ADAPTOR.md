# 2048 adaptor

This adaptor embeds a small Python implementation of the MIT-licensed 2048 rules. It runs fully
offline and needs no browser, server, account, credentials, or external package. `make_game()`
returns a Tier 1 `TurnGame`; `step()` applies one complete move synchronously.

`reset(seed)` creates a fresh `random.Random(seed)` stream, clears the 4x4 board, and places two
tiles. Every changed move uses the same stream to select an empty cell and then chooses 2 with
probability 0.90 or 4 with probability 0.10. Rows/columns are compressed toward the action and
merged from the travel edge, so `[2, 2, 2, 2]` becomes `[4, 4, 0, 0]` toward the left. No-op and
malformed actions are harmless and do not spawn a tile. Play continues after a 2048 tile.

The observation is strict JSON with the board, raw cumulative merge `score`, move count,
`max_tile`, terminal status, seed, last-move metadata, and currently legal actions. The scalar
`score` property is `merge_score + max_tile`, while `score_breakdown()` and `info()` expose the
individual components so evaluation can distinguish merge quality from tile progress. Episodes
default to 200 changed moves (`EVAL_KWARGS`), leaving room for policies to differ; callers may pass
`max_moves` to `make_game`.

Scenario seeds are declared in `SEEDS`; train, held-out, development, and validation groups are
disjoint and use the same seed-stream rules. `close()` is idempotent and only clears local state,
since this adaptor starts no process or connection.

