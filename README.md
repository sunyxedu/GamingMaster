# GamingMaster — Game Agents and Meta-Harness

Games use different control paths according to latency pressure:

| Tier | Control path | Implemented experiment environments |
|---|---|---|
| 1: world waits | Astra with a harness designed by an outer code-search loop | Rule puzzles, offline ARC-AGI-3 |
| 2: world keeps ticking | Astra plans, Jev classifies, a generated harness connects them to actions | Snake, Lane Runner |
| 3: very tight latency | Trained neural network | See the per-game neural adapters below |

The [Meta-Harness guide](docs/meta-harness.md) explains tiers 1 and 2. The
[live-model audit](docs/meta-harness-audit-2026-09-19.md) records real Astra/Jev
experiments, cold-start and shared-memory controls, and current transfer limits.
Minecraft and Genshin are target use cases; their integrations are not implemented.

An AlphaStar-inspired Transformer architecture with a TRM (Tiny Recursive Model) pondering core, trained by Imitation Learning (IL) followed by PPO self-play (RL).

The core network is **game-agnostic**: a fixed two-stage trunk (spatiotemporal history encoder + TRM pondering) with pluggable per-game observation encoders and action decoders. [generals.io](https://generals.io) is the fully-trained reference game (data pipeline + RL environment + trained checkpoints); model adapters also exist for chess, Go, Othello, Gomoku, Shogi (pieces-in-hand drops), Riichi Mahjong, Pokémon (Showdown singles), Lux AI–style simultaneous multi-unit games, Clash Royale, Hearthstone, and StarCraft-style RTS — see [docs/multi-game.md](docs/multi-game.md) and the per-game data/environment survey in [docs/game-data-sources.md](docs/game-data-sources.md).

```python
from model.games import build_game_network

net = build_game_network("chess", d_model=256)   # generals / chess / go / othello / gomoku / shogi /
net = build_game_network("go", board_h=9)        # mahjong / pokemon / lux / clash_royale / hearthstone / starcraft
```

Start with the [documentation index](docs/README.md). Published Meta-Harness
programs and measurements are catalogued in the
[experiment archive](metaharness/artifacts/README.md).

## Repository layout

```text
GamingMaster/
├── metaharness/          # Tiers 1/2: models, harness search, games, evaluation
│   └── artifacts/        # Published experiment programs and measurements
├── model/                # Tier 3: shared neural network and per-game adapters
├── env/                  # Neural-agent self-play environments
├── data_processing/      # Replay ingestion, conversion and datasets
├── training/             # Python modules for IL, RL and checkpoint evaluation
├── scripts/              # Data, visualization and analysis tools
│   └── slurm/            # Cluster jobs grouped into data/, il/, rl/, eval/
├── tests/                # Unit, integration and regression tests
├── docs/                 # Architecture, guides and experiment reports
├── pyproject.toml        # Package metadata, dependencies and tool configuration
└── uv.lock               # Dependency lockfile
```

See [repository conventions](docs/repository-layout.md) for file placement and
entrypoint migration, and [Slurm jobs](scripts/slurm/README.md) for cluster usage.

## Quick Start

### Install

```bash
uv sync

# (Node.js is NOT required — .gior decompression is pure Python)
```

### Foundation-model agents (tiers 1 and 2)

Check the pipeline locally with explicit offline backends:

```bash
uv run python -m metaharness loop --tier 2 --game snake \
    --system1 heuristic --system2 scripted --proposer scripted --clock virtual --iters 1
```

For live experiments, use the existing Codex account login for Astra and set
`TYPESAFE_API_KEY` in `.env` using [.env.example](.env.example) as a reference:

```bash
uv run python -m metaharness loop --tier 2 --game snake \
    --system1 jev --system2 astra --proposer astra
```

See the [Meta-Harness guide](docs/meta-harness.md) for tier 1, ARC setup, budgets,
held-out evaluation and the distinction between offline checks and live results.

### Neural agent workflow (tier 3)

#### Visualise a replay

```bash
# From a generals.io replay URL — produces an interactive HTML file
uv run python scripts/replay_to_visual.py \
    --url "https://generals.io/replays/asS2S5O_y?p=%7EHemlix%7E"

# Open in browser (← → keys to step, Space to play/pause)
open viz_output/replay.html
```

#### Prepare training data

```bash
uv run python -m scripts.prepare_data --players Human.exe --output data/output --target 1000 --min-turns 80
# (wraps the Rust pipeline; all games: data_processing/README.md, e.g. `gmdata generals ladder --min-stars 70`)
```

#### Train with imitation learning

```bash
# Recommended: recurrent neural memory + world-model objective
uv run python -m training.train_il \
    --data_dir data/output/training/Human.exe \
    --temporal_mixer recurrent --world_model_weight 0.2 \
    --tf_start 0.4 --tf_end 0.0 \
    --batch_size 4 --grad_accum 4
```

Effective batch = `batch_size × GPUs × grad_accum`; see
[docs/training.md](docs/training.md) for memory-based batch sizing and the
full from-scratch recipe.

#### Train with reinforcement learning

```bash
# Load from IL checkpoint, fine-tune with PPO self-play
uv run python -m training.train_rl \
    --il_checkpoint checkpoints/checkpoint_epoch50.pt \
    --il_data_dir data/output/training/MeltedToast
```

> RL training uses PPO self-play with joint IL regularisation (AlphaStar-style).
> Environment: `env/generals_env.py` provides a two-player simultaneous-move Gym-like environment with fog of war.
> Stage-2 (mixed multi-game) RL: `env/vec_game.py` defines the `VecGameEnv` protocol — N games stepped at once,
> observations in each game's gm-shard obs schema (pinned bit-for-bit to the IL data by
> `tests/test_vec_game_golden.py`; adapters for generals, othello and gomoku) — and `env/league.py` the seat / opponent
> interface (mirror self-play, callables, network snapshots).
> `training/train_rl_joint.py` is the Stage-2 trainer: PPO with decoupled ratios on the WHOLE multi-game model
> (a `train_multi_game` checkpoint is the initial policy and the frozen KL reference), mixing every simulator
> game's rollouts with an IL/KL replay stream of every configured game in each update (algorithm pieces in
> `training/rl_joint/`; Slurm: `scripts/slurm/multi/rl_joint_job.sh`).
> Model-side support: `model.step()` single-step inference with KV cache, `ValueHead`, `RLLoss` (PPO), `rl_utils.py` (GAE, RolloutBuffer, KL regularisation).

## Development and documentation

```bash
uv run pytest tests/ -q
uv run pytest tests/metaharness -q
```

CI runs the test suite without live model credentials. See
[repository conventions](docs/repository-layout.md) before adding entrypoints,
and use the [documentation index](docs/README.md) to find:

- [Neural agent walkthrough](docs/neural-agent.md): data flow, architecture and verification.
- [Training guide](docs/training.md): IL, PPO, distributed training and memory sizing.
- [Multi-game adapters](docs/multi-game.md) and [data sources](docs/game-data-sources.md).
- [Meta-Harness guide](docs/meta-harness.md) and [live-model audit](docs/meta-harness-audit-2026-09-19.md).
- [Known limitations](docs/known-limitations.md).
