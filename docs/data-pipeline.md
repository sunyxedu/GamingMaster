# Data Pipeline

`scripts/prepare_data.py` runs four stages: **download → filter → simulate →
convert**. The output of `convert` is what `train_il.py` consumes.

```bash
uv run python -m scripts.prepare_data --players Human.exe \
    --output data/output --max-replays 1000 --min-turns 80
```

## What gets rejected, and why

Training on states the simulator cannot reproduce faithfully is worse than
having less data — every diverged frame is a mislabeled sample. The pipeline
therefore drops:

| Rejected | Where | Reason |
|----------|-------|--------|
| Games other than `1v1` / two-player `custom` | `filter.py` | FFA and >2p games have different dynamics; solo customs would give the value head a constant +1 outcome |
| Maps larger than 25×25 | `ReplaySimulator` (ValueError) | The model grid is 25×25; cropping hides tiles (even generals) and clamps move labels onto wrong tiles |
| Replays with swamps / deserts / lookouts / observatories / lights / neutrals | `ReplaySimulator` | These features change army/production/vision rules the simulator does not implement |
| Replays with game modifiers (Torus, Defenseless, City-State, …) | `ReplaySimulator` | Modifiers change the rules themselves |

`prepare_data.py` catches these `ValueError`s and reports skipped replays.
Model-independent viewers (`scripts/replay_to_visual.py`) construct the
simulator with `strict=False` and can still render such replays.

## Label semantics

The converter (`data_processing/parsers/generals_json.py`) pairs the
observation at time *t* with the action decided at *t+1* and emits a per-step
`count` label with **three** states:

| `count` | Meaning | Loss treatment |
|---------|---------|----------------|
| `1` | the player moved, and the server executed it | source/dir/split + pass(target 0) |
| `0` | the player deliberately made no move | pass (target 1) |
| `-1` | the player *tried* to move but the server dropped it (stale source, mountain target) | excluded from action **and** pass supervision |

Server-dropped moves are detected by replaying every move through the
simulator's validation: a move whose source is no longer owned (or has ≤1
army), or whose destination is a mountain, is not executed and never becomes
a label. This mirrors the real server, which silently drops such queued
moves.

## Winner resolution

Games that end with a general capture get the winner from the board. Games
that end by **surrender/disconnect** (a large share of decisive 1v1s) never
see a capture — the winner is resolved from the replay's `afks` array (first
AFK entry loses). Without this, all surrendered games would train the value
head with no outcome signal.

## Known approximation: within-tick move ordering

When both players move in the same tick, the real server executes them in
the recorded order (with a priority rule that has changed over the game's
history). The simulator instead orders them with a heuristic
(chase > defensive > larger army, general attacks last). This is a **known,
deliberate approximation**: the ordering rule only decides who wins
same-tick races, and RL self-play (which controls its own environment) is
expected to absorb whatever convention it is trained under. If you ever see
systematic army-count divergence between a simulated replay and the real
game, this is the first place to look.

## Observation encoding (must stay identical everywhere)

Three places build model observations and must agree exactly:
IL data (`generals_json.py`), the RL env (`env/generals_env.py`), and the
eval scripts (`scripts/play_model.py`). The shared convention:

- 12 grid channels: one-hot of the 11-state encoding (0 unseen-blank,
  1 unseen-obstacle, 2 seen-blank, 3 neutral city, 4 mountain, 5/6/7 own
  land/city/general, 8/9/10 enemy land/city/general) + `log1p(army)/10`.
- Globals: `log1p([myArmy, myLand, opArmy, opLand]) / 10`.
- Directions: 0=down, 1=right, 2=up, 3=left (row-major, `y` = row).
- Split: 0 = move all-but-one, 1 = 50% split.
- Value heuristic: `tanh(adj_ratio − 1)` with the city-difference army
  adjustment — the same formula in the IL converter and the env's dense
  reward.
