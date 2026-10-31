# Baba Is AI adaptor

This adaptor runs the open-source `nacloos/baba-is-ai` engine in process, headlessly and offline. The adaptor vendors the inspected upstream `baba` package source (version 0.0.1) under `baba/`; its runtime dependencies are listed in requirements.txt. No GUI, server, account, or credentials are used.

`reset(seed)` maps declared seeds to a versioned, sorted manifest of native environment IDs. Train, held-out, dev, and validation groups use disjoint environment IDs (held-out and validation use the same difficulty family but different from training). Seeds outside the declaration select from the full manifest by stable modulo mapping. A seed therefore selects a native puzzle template/configuration; the engine's own RNG, seeded with that integer, instantiates the board.

The game is Tier 1. `step` accepts `noop`, `up`, `right`, `down`, `left` (or native integers 0–4), applies exactly one native turn, and returns the next structured observation. Invalid actions are ignored and reported in `last_error`. Observations preserve every object stack in `(x,y)` cells, active rules, agent position/direction, step count, and terminal state.

Score is one point only when the engine's `is_win` predicate is true. Losses and step-limit termination score zero. `score_breakdown()` reports solved count, attempted count, and solved fraction; `info()` includes seed, level identity, step count, and termination reason. The finite manifest is intentionally limited, so arbitrary seeds can collide modulo its size.

The adaptor does not start background processes. `close()` calls the engine close hook when present and is idempotent.
