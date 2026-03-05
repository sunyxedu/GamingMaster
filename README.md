# GamingMaster — Generals.io Neural Network Bot

An AlphaStar-inspired Transformer architecture for [generals.io](https://generals.io), supporting Imitation Learning (IL) and future Reinforcement Learning (RL).

## Project Structure

```
├── data_processing/          # Replay data pipeline
│   ├── download/             #   S3 download + LZ-string decompression
│   ├── processing/           #   Simulator, filter, data models
│   ├── parsers/              #   Convert simulator output → training format
│   ├── dataset_seq.py        #   PyTorch Dataset + collate
│   └── mock/                 #   Mock data generators for testing
├── model/                    # Neural network
│   ├── config.py             #   ModelConfig dataclass
│   ├── network.py            #   GeneralsNetwork (assembles everything)
│   ├── losses.py             #   ImitationLoss, RLLoss
│   └── components/           #   Reusable building blocks
│       ├── encoder.py        #     SpatialEncoder, HistoryEncoder, ResBlock
│       ├── transformer.py    #     TransformerBlock (shared-weight pondering)
│       ├── heads.py          #     Source / Direction / Split / Value heads
│       └── world_model.py    #     WorldModel (for RL curiosity reward)
├── scripts/
│   ├── prepare_data.py       # CLI: batch download → filter → simulate → convert
│   └── replay_to_visual.py   # CLI: download one replay → simulate → HTML viewer
├── train_il.py               # Imitation Learning training script
├── tests/
│   ├── test_verify_simulator.py   # Simulator correctness (real replay)
│   ├── test_model.py              # 19 unit tests for every model component
│   └── test_training_e2e.py       # End-to-end training loop tests
└── pyproject.toml
```

## Quick Start

### 1. Install

```bash
pip install -e .

# Node.js dependency (auto-installed on first use, or manually):
cd data_processing/download && npm install lz-string
```

### 2. Visualise a Replay

```bash
# From a generals.io replay URL — produces an interactive HTML file
python3 scripts/replay_to_visual.py \
    --url "https://generals.io/replays/asS2S5O_y?p=%7EHemlix%7E"

# Open in browser (← → keys to step, Space to play/pause)
open viz_output/replay.html
```

### 3. Prepare Training Data

```bash
python3 scripts/prepare_data.py \
    --players MeltedToast \
    --output data/output \
    --max-replays 1000 \
    --min-turns 80
```

### 4. Train (Imitation Learning)

```bash
python3 train_il.py \
    --data_dir data/output/training/MeltedToast \
    --epochs 50 \
    --batch_size 4 \
    --d_model 256 \
    --n_ponder_steps 4
```

---

## How It Works — Step by Step

### Step 1: Data Pipeline

The data pipeline turns raw generals.io replays into training-ready tensors.

**Download** — `data_processing/download/`

Replays are stored on AWS S3 as `.gior` files (LZ-string compressed). The pipeline:
1. Fetches replay metadata from `generals.io/api/replaysForUsername`
2. Downloads `.gior` files from `generalsio-replays-na.s3.amazonaws.com`
3. Decompresses using Node.js `lz-string` (auto-installed if missing)
4. Deserialises into a structured dict: map size, generals, cities, mountains, moves

**Filter** — `data_processing/processing/filter.py`

Keeps only useful replays: 1v1 games, minimum turn count, optionally minimum opponent rating.

**Simulate** — `data_processing/processing/simulator.py`

Replays the game tick-by-tick from one player's perspective, producing per-turn:
- **State encoding** (11 categories × 25×25 grid): what the player sees through fog of war
- **Army counts** (25×25 grid): troop numbers on visible tiles
- **Global stats**: my army, my land, opponent army, opponent land
- **Actions**: the move the player made (source, destination, split flag)

Key game rules implemented:
| Rule | Implementation |
|------|---------------|
| Army growth | Generals/cities: +1 every 2 ticks. All land: +1 every 50 ticks |
| Fog of war | Owned tiles + 8 adjacent tiles visible |
| Move priority | Chase > Defensive > Army size (largest first), general attacks last |
| Capture | Attacker wins if moving army > defender army; difference remains |
| 50% split | Moves ⌊army/2⌋ instead of army−1 |

**Convert** — `data_processing/parsers/generals_json.py`

Transforms simulator output into the training format. **Critical detail**: the converter correctly pairs observation at time *t* with the action at time *t+1*, because the player decides their next move *after* seeing the current state. (This off-by-one was a bug that was found and fixed — verified by checking that 100% of action sources fall within the legal-move mask of the paired observation.)

Training format per timestep:
| Field | Shape | Description |
|-------|-------|-------------|
| `grid_seq` | `[T, 12, 25, 25]` | 11 one-hot state channels + 1 log-normalised army channel |
| `globals_seq` | `[T, 4]` | log-normalised (myArmy, myLand, opArmy, opLand) |
| `legal_source_mask_seq` | `[T, 1, 25, 25]` | bool mask — owned tiles with army > 1 |
| `count` | `[T]` | 0 or 1 — whether a move was made |
| `src_y`, `src_x` | `[T, K]` | source tile coordinates |
| `dir_id` | `[T, K]` | direction: 0=↓ 1=→ 2=↑ 3=← |
| `ratio_id` | `[T, K]` | 0=all-but-1, 1=50% split |
| `value` | `[T]` | heuristic game value ∈ \[−1, 1\] |

### Step 2: Neural Network Architecture

The model follows the architecture diagram — a Transformer with shared-weight pondering, autoregressive policy heads, and optional world model.

```
Input (Grid Types + Grid Number + Globals)
  │
  ▼
┌─────────────────────┐
│   SpatialEncoder     │  Conv + ResBlocks: [B, 12, 25, 25] → [B, d, 25, 25]
└─────────┬───────────┘
          │ + broadcast global features [B, d, 1, 1]
          ▼
┌─────────────────────┐
│   HistoryEncoder     │  GRU over temporally-pooled spatial embeddings
└─────────┬───────────┘
          │ flatten to [B·T, 625, d] + append 2 decision tokens
          ▼
┌─────────────────────┐
│  TransformerBlock    │  Self-Attention → Add & Norm → MLP → Add & Norm
│       × k steps      │  (same weights applied k times — pondering)
└─────────┬───────────┘
          │
    ┌─────┼──────────┐
    ▼     ▼          ▼
 spatial  HL token   LL token
 tokens
    │     │          │
    ▼     │     ┌────┴─────┐
┌────────┐│     ▼          ▼
│ Source  ││  ┌──────┐  ┌──────┐
│  Head   ││  │ Dir  │  │Split │
└───┬────┘│  │ Head │  │ Head │
    │     │  └──────┘  └──────┘
    │     ▼
    │  ┌──────┐
    │  │Value │
    │  │ Head │
    │  └──────┘
    ▼
  source_logits [B, T, 625]
```

**Key design choices:**

- **Autoregressive policy heads**: Source is predicted first; its embedding is then fed to the Direction and Split heads. During training, the ground-truth source is used; during inference, the model samples.
- **Shared-weight pondering** (×k): A single `TransformerBlock` is applied k times (default 4). This lets the model iteratively refine its representation, similar to Universal Transformer / ACT.
- **2D positional encoding**: Separate learnable row and column embeddings, summed additively.
- **Decision tokens**: Two special learnable tokens — High-Level (used for value prediction) and Low-Level (used for direction/split prediction) — are appended to the spatial tokens and participate in self-attention.
- **World model** (optional): Predicts next-state embedding from (current state, action). Disabled during IL (receives zeros); during RL, prediction error serves as intrinsic curiosity reward.

Default config:
| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_heads` | 8 |
| `d_ff` | 512 |
| `n_ponder_steps` | 4 |
| `encoder_channels` | 128 |
| `gru_layers` | 1 |

### Step 3: Training

**Imitation Learning** (`train_il.py`)

Supervised learning from expert replays:

| Loss component | Weight | Description |
|----------------|--------|-------------|
| Source CE | 1.0 | Cross-entropy over 625 grid positions |
| Direction CE | 1.0 | Cross-entropy over 4 directions |
| Split CE | 0.5 | Cross-entropy over 2 split options |
| Value MSE | 0.5 | Mean squared error on game value |

All losses are masked: timesteps with no move (`count=0`) are excluded from source/direction/split loss; timesteps beyond the actual sequence length are excluded from all losses.

**Reinforcement Learning** (`model/losses.py` — `RLLoss` placeholder)

PPO-style loss structure is ready:
- Clipped policy gradient objective
- Value function loss
- Entropy bonus
- World model prediction error as intrinsic reward

### Step 4: Visualisation

`scripts/replay_to_visual.py` provides the full pipeline from URL to interactive viewer:

```bash
python3 scripts/replay_to_visual.py \
    --url "https://generals.io/replays/asS2S5O_y?p=%7EHemlix%7E"
```

Produces a self-contained HTML file (~2-4 MB) with:
- Canvas-based grid rendering with the full colour scheme
- **← →** arrow keys to step forward/backward
- **Space** to play/pause, **[ ]** to adjust speed
- **Home/End** to jump, range slider for scrubbing
- Move arrows (yellow) showing the player's action each turn
- Army/land stats HUD for both players
- `--god` flag for fog-free full-map view

---

## Verification

Every component has been verified against the real replay `asS2S5O_y` (~Hemlix~ vs Human.exe):

| Check | Method | Result |
|-------|--------|--------|
| Army growth | Traced general army: 1→2→3→…→25 at tick 48, +25 at tick 50 | ✓ |
| Fog of war | Owned + 8-neighbour visibility; distant tiles hidden | ✓ |
| Move priority | Chase > defensive > army size; general attacks last | ✓ |
| State encoding | 11 categories correct for all terrain/ownership/visibility combos | ✓ |
| Obs-action pairing | 0/1225 legal-mask violations (was 1160/1225 before fix) | ✓ |
| Full simulation | Both players' perspectives: correct winner, land/army progression | ✓ |
| Model forward | All output shapes correct for batch + sequence dims | ✓ |
| Gradient flow | All 64 parameters receive gradients through every component | ✓ |
| Training loop | Loss decreases on mock and real data; checkpoint save/load works | ✓ |

Run all tests:

```bash
python3 tests/test_verify_simulator.py   # Simulator correctness
python3 tests/test_model.py              # 19 model unit tests
python3 tests/test_training_e2e.py       # End-to-end training
```
