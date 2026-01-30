# GamingMaster

AI for [generals.io](https://generals.io) - a real-time strategy game.

Train a neural network to play generals.io through imitation learning and self-play reinforcement learning, using direct policy sampling guided by a Transformer-based policy-value network.

## Quick Start

```bash
# Install dependencies
uv sync

# Start RL training (self-play)
uv run python scripts/run_rl.py --iterations 100 --games 5

# Or with IL pre-training (requires replay data)
uv run python train.py --il_epochs 10 --rl_iterations 100 --replay_dir data/replays
```

## How It Works

```
Self-Play ──▶ Trajectory Buffer ──▶ Training ──▶ Better Network ──┐
    ▲                                                              │
    └──────────────────────────────────────────────────────────────┘
```

1. **Imitation Learning** (Optional): Learn from human replays
2. **Self-Play**: The AI plays against itself using direct policy sampling
3. **V-trace Correction**: Off-policy correction for stable RL
4. **Train Network**: Policy learns to win, Value learns to predict game outcome
5. **Repeat**: Stronger network → better play → even stronger network

## Architecture

**TRM (Tiny Recursive Model) architecture (~14.2M parameters):**

Inputs (per tick):
- `grid_seq`: [B, T, 12, 25, 25] one-hot state channels + army channel
- `globals_seq`: [B, T, 4] global features
- `tick_bucket_seq`: [B, T] tick bucket in [0..25]

History encoder (model/history_encoder.py):
- Spatial conv encoder -> temporal transformer (causal) -> history latent [B, 256]

Input embedding (model/input_embedding.py):
- Grid conv encoder + global proj + history proj + 2D pos encoding (sin + learned) + tick embedding
- Summed tokens -> [B, 625, 512]

TRM backbone (model/backbone.py):
- Dual state recursion: y = answer (prediction), z = reasoning (latent)
- TRMLayer x2, n_inner_cycles=6, n_supervision_steps=3 (total 18 cycles)
- Cycle embeddings + halting head on z
- all_y stored for deep supervision / halting loss

Policy heads (model/policy_head.py):
- MoveHead: pass(0) / move(1)
- SourceHead: 625 positions (no masking - learns validity via penalty)
- DirectionHead: 4 directions (no masking - learns validity via penalty)
- RatioHead: 2 ratios (all-but-1, 50%)
- Autoregressive sampling: move -> source -> direction -> ratio

Value head (model/value_head.py):
- Mean pool combined tokens (y + z) + globals -> MLP -> tanh in [-1, 1]

### No-Masking Approach (Generalization)

Instead of masking illegal actions to -inf, the model learns action validity through:
- **`is_legal(grid, action)`**: Checks if action is valid based on grid state (game_env.py)
- **Illegal action penalty**: Added to loss when model predicts invalid actions
- This enables better generalization to unseen game states

## Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--iterations` | RL training iterations | 100 |
| `--games` | Self-play games per iteration | 5 |
| `--epochs` | IL training epochs | 5 |
| `--map_width/height` | Map dimensions | 25×25 |
| `--lr` | Learning rate | 1e-4 (IL), 1e-5 (RL) |

## Project Structure

```
├── scripts/              # Training scripts (run_il.py, run_rl.py)
├── train.py    # Main training entry point (IL + RL)
├── model/                # Neural network
├── training/             # Training infrastructure (RLTrainer, V-trace)
└── data_processing/      # Game environment
```

## Replay Data Format (gio)

Processed replay files are JSON with per-tick sequences:

- `grid_seq`: `[T, 12, 25, 25]` one-hot state channels + army channel
- `globals_seq`: `[T, 4]` global features (myArmy, myLand, opArmy, opLand), log-normalized
- `prev_action_seq`: `[T, 8]` previous action embedding (currently zeroed)
- `count`: `[T]` number of actions at this tick (0 or 1 in current pipeline)
- `src_y`, `src_x`: `[T, K]` source tile coordinates (padded, `K = max_actions_per_tick`)
- `dir_id`: `[T, K]` direction id (0:down, 1:right, 2:up, 3:left)
- `ratio_id`: `[T, K]` ratio id (0: all-but-1, 1: 50%)
- `value`: `[T]` heuristic value target in [-1, 1]
- `enemy_heatmap`: `[T, 1, 25, 25]` placeholder (zeros)
- `next_visibility`: `[T, 1, 25, 25]` placeholder (zeros)
- `tick_seq`: `[T]` absolute tick index
- `tick_bucket_seq`: `[T]` tick bucket, computed as `((t % 50) + 1) // 2`
- `max_actions_per_tick`: integer `K` (default 4)

When a tick has no action, `count[t] = 0` and the action arrays are zero-padded.

Note: `legal_source_mask_seq` is no longer used as input. Action validity is checked via `is_legal()` for penalty computation.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- ~8GB RAM for training

## License

MIT
