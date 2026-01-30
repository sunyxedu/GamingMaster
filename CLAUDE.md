# Generals.io AI - Claude Code Guide

## Project Overview

AI for generals.io game using Tiny Recursive Model (TRM) Transformer.

**Key Features:**
- TRM backbone with adaptive computation (2-8 recursive cycles)
- Hierarchical action space: move → source → direction → ratio
- No-masking approach: learns action validity via penalty (better generalization)
- Two-phase training: Imitation Learning (IL) + Reinforcement Learning (RL)
- RL with direct policy sampling and V-trace off-policy correction
- Entropy bonus for exploration

## Directory Structure

```
GamingMaster/
├── model/                      # Neural network architecture
│   ├── components/             # Building blocks (RMSNorm, SwiGLU, TRMAttention)
│   ├── backbone.py             # TRM recursive Transformer
│   ├── policy_head.py          # MoveHead, SourceHead, DirectionHead, RatioHead
│   ├── value_head.py           # Value estimation
│   ├── model.py                # GeneralsModel main model
│   ├── config.py               # Model/Training configs
│   ├── history_encoder.py      # Temporal history encoding
│   └── input_embedding.py      # Grid/global feature embedding
├── training/                   # Training infrastructure
│   ├── rl_trainer.py           # RLTrainer (main trainer)
│   ├── vtrace.py               # V-trace off-policy correction
│   ├── loss.py                 # ImitationLoss
│   └── replay_buffer.py        # ReplayBuffer, GameRecord
├── data_processing/            # Replay data processing
│   ├── processing/
│   │   ├── game_env.py         # GameEnvironment for self-play
│   │   ├── models.py           # Tile, Player, Turn dataclasses
│   │   ├── simulator.py        # ReplaySimulator
│   │   └── filter.py           # Replay filtering utilities
│   ├── download/               # API client, replay fetcher, deserializer
│   ├── parsers/                # JSON parsing (generals_json.py)
│   └── dataset_seq.py          # SeqReplayDataset for IL
├── tests/                      # Unit tests
├── scripts/                    # Training scripts (run_il.py, run_rl.py)
├── train.py          # Main training script (IL + RL)
└── visualize_game.py           # Game visualization (GIF generation)
```

## Key Components

### Model Architecture (~12.6M params)

```
Input: grid[12,25,25] + globals[4] + history_latent[256] + tick
    ↓
 HistoryEncoder → latent[B, 256]
    ↓
InputEmbedding → tokens[B, 625, 512]
    ↓
TRM Backbone (2 layers, 8 heads, 2-8 recursive cycles)
    ↓
├── MoveHead → [B, 2] (pass/move)
├── SourceHead → [B, 625] (which tile)
├── DirectionHead → [B, 4] (down/right/up/left)
├── RatioHead → [B, 2] (all-1 / 50%)
└── ValueHead → [B, 1] (win probability)
```

### Action Format

```python
action = (move_id, source_idx, dir_id, ratio_id)
# move_id: 0=pass, 1=move
# source_idx: 0-624 (25x25 grid, row*25+col)
# dir_id: 0=down, 1=right, 2=up, 3=left
# ratio_id: 0=all-but-1, 1=50%
```

### Training Pipeline

```
Phase 1: Imitation Learning
    - Dataset: SeqReplayDataset (human replays)
    - Loss: ImitationLoss (cross-entropy + value MSE)

Phase 2: Reinforcement Learning
    - Self-play with direct policy sampling
    - V-trace off-policy correction
    - Actor-Critic loss (policy gradient + value MSE + entropy bonus)
    - Optional KL penalty to stay close to supervised policy
```

## Complete Workflow

### Step 1: Environment Setup

```bash
# Clone and setup
cd GamingMaster
uv sync  # Install dependencies

# Verify installation
uv run python -c "from model.model import GeneralsModel; print('OK')"
```

### Step 2: Data Collection (Optional, for IL)

```bash
# Download replays from generals.io API
uv run python scripts/prepare_data.py --username "top_player" --num_replays 1000

# Or manually place replay JSON files in:
mkdir -p data/replays
# Copy your .json replay files to data/replays/
```

**Replay JSON format** (output from data processing):
```json
{
  "grid_seq": [...],           // [T, 12, H, W] state sequence
  "globals_seq": [...],        // [T, 4] global features
  "src_y": [...],              // [T, K] action source y
  "src_x": [...],              // [T, K] action source x
  "dir_id": [...],             // [T, K] direction
  "ratio_id": [...],           // [T, K] ratio
  "value": [...],              // [T] game outcome
  "tick_bucket_seq": [...]     // [T] tick buckets, formula: ((t % 50) + 1) // 2
}
```
Note: `legal_source_mask_seq` is no longer used. Action validity is checked via `is_legal()` for penalty.

### Step 3: Training

**Option A: Full Pipeline (IL → RL)**
```bash
# Phase 1: Learn from human replays
# Phase 2: Improve via self-play
uv run python train.py \
    --il_epochs 10 \
    --rl_iterations 100 \
    --replay_dir data/replays \
    --games_per_iter 10 \
    --output_dir checkpoints
```

**Option B: IL Only (Imitation Learning)**
```bash
uv run python scripts/run_il.py \
    --epochs 20 \
    --data_dir data/replays \
    --batch_size 32 \
    --lr 1e-4
```

**Option C: RL Only (Self-Play with Direct Policy Sampling)**
```bash
uv run python scripts/run_rl.py \
    --iterations 200 \
    --games 10 \
    --entropy_weight 0.01
```

### Step 4: Evaluation & Visualization

```bash
# Watch AI play against itself
uv run python visualize_game.py --live --max_turns 100

# Generate GIF of a game
uv run python visualize_game.py \
    --save game.json \
    --gif game.gif

# Load trained model for visualization
uv run python visualize_game.py \
    --checkpoint checkpoints/model_final.pt \
    --save game.json \
    --gif trained_game.gif
```

### Step 5: Resume Training

```bash
# Resume from checkpoint
uv run python scripts/run_rl.py \
    --checkpoint checkpoints/rl/model_rl_iter_50.pt \
    --iterations 50
```

### Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
│  generals.io API → download/ → parsers/ → data/replays/*.json   │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: IMITATION LEARNING                  │
│  SeqReplayDataset → Model → ImitationLoss → Optimizer           │
│  (Learn to mimic human players from replay data)                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 2: REINFORCEMENT LEARNING                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Self-Play   │ →  │ Trajectory   │ →  │   Training   │       │
│  │ (direct pol) │    │   Buffer     │    │  (V-trace)   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│  (Improve beyond human level through self-play)                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        EVALUATION                               │
│  visualize_game.py → Watch games / Generate GIFs                │
│  (Evaluate model performance)                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Common Commands

```bash
# IL only (requires replay data)
uv run python scripts/run_il.py --epochs 10 --data_dir data/replays

# RL only
uv run python scripts/run_rl.py --iterations 100 --games 5

# Full IL + RL
uv run python train.py --il_epochs 5 --rl_iterations 100 --replay_dir data/replays

# Visualize a game
uv run python visualize_game.py --save game.json --gif game.gif

# Quick test
uv run python -c "from model.model import GeneralsModel; print('OK')"

# Run tests
uv run pytest tests/
```

## Key Files to Modify

| Task | Files |
|------|-------|
| Change model architecture | `model/backbone.py`, `model/config.py` |
| Modify action space | `model/policy_head.py`, `training/rl_trainer.py` |
| Change reward/loss | `training/loss.py`, `training/rl_trainer.py` |
| Add game features | `data_processing/processing/game_env.py` |
| Modify V-trace | `training/vtrace.py` |
| Change history encoding | `model/history_encoder.py` |
| Modify input features | `model/input_embedding.py` |
| Tune RL training | `training/rl_trainer.py` (RLTrainer) |

## Important Implementation Details

### TRM Recursive Mechanism

The TRM backbone performs iterative refinement:
```python
z = x  # Initialize with input
for cycle in range(max_cycles):
    z_input = z + cycle_embedding[cycle]  # Add cycle embedding
    z = transformer_layers(z_input)       # z ← net(z)
    if halt_head(z) > 0.5 and cycle >= min_cycles:
        break  # Early stopping
```

### Legal Action Handling (No-Masking Approach)

**Design Choice**: Instead of masking illegal actions to -inf, the model learns action validity through penalty. This enables better generalization to unseen game states.

- `is_legal(grid, action)`: Checks if action is valid based on grid state (game_env.py)
- `is_legal_batch(grid, source_idx, dir_idx, ratio_idx)`: Batch version for training
- `compute_illegal_action_penalty()`: Adds penalty to loss when model predicts illegal actions
- Direction validity: bounds check + mountain check (channel 4)
- Source validity: owned tile (channels 5-7) + army > 1 (channel 11)

### V-trace Off-Policy Correction

```python
# Key formula (in training/vtrace.py)
importance_weight = exp(target_log_prob - behavior_log_prob)
clipped_rho = clamp(importance_weight, max=rho_bar)  # default 1.0
td_error = clipped_rho * (reward + gamma * v_next - v_current)
# V-trace target computed via backward accumulation
```

### Log Probability Calculation

```python
# Autoregressive factorization (in training/vtrace.py)
if move == 0:
    log_prob = log_prob(move=0)
else:
    log_prob = log_prob(move=1) + log_prob(src) + log_prob(dir|src) + log_prob(ratio)
```

## Debugging Tips

1. **NaN in loss**: Check for `-inf` in logits, replace with `-100` before softmax
2. **MPS slow**: Training steps on MPS can be slow; use CPU for quick tests
3. **No legal moves**: Environment returns empty legal_actions → pass action
4. **Game termination**: `_check_winner()` detects general capture → done=True
5. **Import errors**: Run `uv sync` to install dependencies

## State Encoding (12 channels)

```
Channel 0:  Unseen blank
Channel 1:  Unseen not blank
Channel 2:  Seen blank
Channel 3:  Seen uncaptured city
Channel 4:  Seen mountain
Channel 5:  Your land
Channel 6:  Your city
Channel 7:  Your general
Channel 8:  Opponent land
Channel 9:  Opponent city
Channel 10: Opponent general
Channel 11: Army count (log normalized)
```

## Hyperparameters

### Model
| Parameter | Default | Description |
|-----------|---------|-------------|
| d_model | 512 | Transformer dimension |
| n_heads | 8 | Attention heads |
| n_layers | 2 | Transformer layers |
| max_cycles | 8 | Max recursive cycles |
| min_cycles | 2 | Min recursive cycles |

### Training (IL)
| Parameter | Default | Description |
|-----------|---------|-------------|
| il_lr | 1e-4 | Adam learning rate for IL |
| il_batch_size | 32 | Batch size for IL |
| max_seq_len | 100 | Max sequence length |

### Training (RL)
| Parameter | Default | Description |
|-----------|---------|-------------|
| gamma | 0.99 | RL discount factor |
| rl_lr | 1e-5 | Adam learning rate (lower than IL) |
| entropy_weight | 0.01 | Entropy bonus for exploration |
| value_loss_weight | 0.5 | Value loss weight |
| rho_bar | 1.0 | V-trace rho clipping |
| c_bar | 1.0 | V-trace c clipping |
| kl_weight | 0.0 | KL penalty (0 to disable) |
| rl_seq_len | 16 | Sequence length for RL training |
