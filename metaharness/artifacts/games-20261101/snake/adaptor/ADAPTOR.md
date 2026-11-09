# Snake adaptor

This adaptor implements the study-selected **local symbolic Snake API**. There is
no external game binary, display, account, server, browser, or network connection:
`game.py` is the faithful local reference implementation used by the evaluator.

## Interface and timing

`make_game(**kwargs)` returns a `RealtimeGame` named `Snake (local classic)`.
The runner owns the real-time clock and calls `tick(action)` at the declared
cadence; `tick()` does not sleep and advances exactly one grid cell. The default
cadence is `tick_s = 0.10`. `observe()` is a side-effect-free deep snapshot.
`step(action)` is also provided for deterministic replay/unit tests, but does not
turn the game into a turn-based benchmark.

`ActionSpec` names the five advertised actions:

- `UP`, `RIGHT`, `DOWN`, `LEFT`: absolute steering directions.
- `NOOP`: keep the current heading.

Integer aliases `0..3` map to `UP, RIGHT, DOWN, LEFT`. A dictionary with
`action` and optional `state_tick` is accepted too. A stale `state_tick`, malformed
input, or immediate reverse is ignored, recorded in diagnostics, and the snake
continues moving; malformed input never raises from `tick()`.

## Rules and observations

The reference scenario is a 12x12 zero-based `[row, column]` bounded board, with
no wrapping, one food cell, a valid length-three opening selected near the centre,
and a private `random.Random(seed)` stream. Food is uniformly selected from the
row-major list of currently empty cells, including after every successful eat.
The body in `observe()["snake"]` is head-first. A non-food move may enter the
current tail because it vacates on that transition; a food move retains the tail
and grows by one. Wall/self collision terminates. Filling the board terminates
successfully without sampling another food. A live episode at `max_ticks` is
truncated as `time_limit`.

The symbolic observation contains the complete body, head/tail, food, direction,
score, length, free cells, legal one-step safety aid, timing metadata, terminal
flags/reason, last reward/event/error, and the attempted head. It is strict JSON
and contains no screenshots or binary assets.

## Seeds and scenarios

`SEEDS` declares training seeds `0..7` and held-out seeds `101..107`. A seed
selects both the valid central opening and all future food placements; it is not a
hard-coded level or a policy. Same seed, profile, and applied actions reproduce
the same trajectory on a fresh instance. Held-out seeds use the same rules but
unseen RNG streams/routes.

The default `EVAL_KWARGS` is the reference profile:

```json
{"rows": 12, "cols": 12, "tick_s": 0.10, "max_ticks": 4000}
```

For explicitly declared generalisation experiments, `make_game` accepts fixed
constructor options such as:

- `rows=8, cols=8, tick_s=0.08, max_ticks=4000`
- `rows=10, cols=14, tick_s=0.15, max_ticks=6000`
- `rows=14, cols=10, tick_s=0.25, max_ticks=6000`
- `rows=11, cols=11, tick_s=0.10, max_ticks=5000` (odd-by-odd stress case;
  no Hamiltonian-cycle assumption is made)

These options alter only board dimensions, cadence metadata, and the explicit
external tick budget. Wall wrapping, growth, reverse handling, food sampling,
and score semantics never vary with seed or profile. Dimensions must be at least
4 and the maximum tick budget prevents an infinite tail loop.

## Score and limits

`score` is exactly the cumulative number of food items eaten, one point each.
`score_breakdown()` and `info()` report food count, final length, ticks survived,
`terminal_reason`, `filled_board`, `max_score`, and score fraction. Survival
without eating is diagnostic only and does not increase score. The reference
maximum is `12*12-3 = 141`; other profiles report their own `rows*cols-3`.

The adaptor is process-free and resource-free. `close()` is idempotent, clears
pending input, and releases the in-process state. No environment variables or
credentials are required.
