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
│       ├── transformer.py    #     TransformerBlock, SpatiotemporalBlock
│       ├── heads.py          #     Source / Direction / Split / Value heads
│       └── world_model.py    #     WorldModel (for RL curiosity reward)
├── scripts/
│   ├── prepare_data.py       # CLI: batch download → filter → simulate → convert
│   └── replay_to_visual.py   # CLI: download one replay → simulate → HTML viewer
├── train_il.py               # Imitation Learning training script
├── tests/
│   ├── test_verify_simulator.py   # Simulator correctness (real replay)
│   ├── test_model.py              # 24 unit tests for every model component
│   └── test_training_e2e.py       # End-to-end training loop tests
└── pyproject.toml
```

## Quick Start

### 1. Install

```bash
uv sync

# Node.js dependency (auto-installed on first use, or manually):
(cd data_processing/download && npm install lz-string)
```

### 2. Visualise a Replay

```bash
# From a generals.io replay URL — produces an interactive HTML file
uv run python scripts/replay_to_visual.py \
    --url "https://generals.io/replays/asS2S5O_y?p=%7EHemlix%7E"

# Open in browser (← → keys to step, Space to play/pause)
open viz_output/replay.html
```

### 3. Prepare Training Data

```bash
uv run python scripts/prepare_data.py \
    --players MeltedToast \
    --output data/output \
    --max-replays 1000 \
    --min-turns 80
```

### 4. Train (Imitation Learning)

```bash
uv run python train_il.py \
    --data_dir data/output/training/MeltedToast \
    --epochs 50 \
    --batch_size 4 \
    --d_model 256
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

The model uses **factored spatiotemporal attention**: each layer first does spatial self-attention within the current timestep, then causal temporal self-attention across all past timesteps at each grid position. This replaces both the CNN-era history encoder and the spatial Transformer into a single unified backbone.

Two temporal modes are available via `ModelConfig.temporal_type`:

| Mode | `temporal_type` | Description |
|------|----------------|-------------|
| **Spatiotemporal** (default) | `"spatiotemporal"` | Factored spatial + causal temporal attention, stacked `n_layers` times |
| GRU + Pondering | `"gru"` | GRU history encoder → shared-weight spatial TransformerBlock ×k |

```
Input (Grid Types 11ch + Grid Number 1ch + Globals 4-dim)
  │
  ▼
┌─────────────────────┐
│   SpatialEncoder     │  Conv + ResBlocks: [B, 12, 25, 25] → [B, d, 25, 25]
└─────────┬───────────┘
          │ + broadcast global features [B, d, 1, 1]
          │ + sinusoidal temporal PE
          │ flatten to [B, T, 625, d] + append HL & LL decision tokens
          ▼
┌──────────────────────────────────┐
│    SpatiotemporalBlock × n_layers │
│                                  │
│  ┌─ Spatial Self-Attention ────┐ │  627 tokens within each timestep
│  │  Add & Norm                 │ │
│  ├─ Temporal Self-Attention ───┤ │  per cell across all past timesteps (causal)
│  │  Add & Norm                 │ │
│  ├─ MLP ───────────────────────┤ │
│  │  Add & Norm                 │ │
│  └─────────────────────────────┘ │
└─────────────┬────────────────────┘
              │
        ┌─────┼──────────┐
        ▼     ▼          ▼
     spatial  HL token   LL token
     tokens
        │     │          │
        ▼     │     ┌────┴─────┐
   ┌────────┐ │     ▼          ▼
   │ Source  │ │  ┌──────┐  ┌──────┐
   │  Head   │ │  │ Dir  │  │Split │
   └───┬────┘ │  │ Head │  │ Head │
       │      │  └──────┘  └──────┘
       │      ▼
       │   ┌──────┐
       │   │Value │
       │   │ Head │
       │   └──────┘
       ▼
  source_logits [B, T, 625]
```

**Key design choices:**

- **Factored spatiotemporal attention**: Each `SpatiotemporalBlock` first runs spatial self-attention (627 tokens within the same timestep interact), then causal temporal self-attention (each of the 627 token positions attends to its own history across all past timesteps). This gives every grid cell a per-cell temporal memory — it knows exactly *where* and *when* it saw something before, unlike a pooled GRU that discards spatial structure.
- **Causal temporal masking**: Timestep *t* can only attend to timesteps 0..*t*, never the future. Verified by test: processing T=6 vs T=3 produces identical outputs for the first 3 steps (diff = 0).
- **KV cache for inference**: During single-step inference, each block caches past key/value tensors. Only the current timestep's tokens are computed; temporal attention queries the growing cache. No re-processing of past timesteps.
- **Autoregressive policy heads**: Source is predicted first from spatial tokens; its embedding is extracted and concatenated with the Low-Level decision token to predict Direction and Split. During training, ground-truth source is used; during inference, the model samples.
- **Decision tokens**: Two learnable tokens — High-Level (→ value prediction) and Low-Level (→ direction/split) — are appended to the 625 spatial tokens and participate in both spatial and temporal attention.
- **2D spatial PE + sinusoidal temporal PE**: Spatial positions use learnable row + column embeddings (summed). Temporal positions use sinusoidal encoding, allowing generalisation to arbitrary game lengths.
- **World model** (optional): Predicts next-state embedding from (current state, action). Disabled during IL; during RL, prediction error serves as intrinsic curiosity reward.

Default config:
| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_heads` | 8 |
| `d_ff` | 512 |
| `n_layers` | 4 |
| `encoder_channels` | 128 |
| `temporal_type` | `"spatiotemporal"` |

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

**Reinforcement Learning** (`model/losses.py` — `RLLoss`)

IL 训练出一个合理的初始策略后，用 RL 做 fine-tune 来超越人类 replay 的水平。整体流程：

```
IL pretrained model
  │
  ▼
┌────────────────────────────────────────────┐
│  RL Training Loop (PPO)                    │
│                                            │
│  1. model.step() 生成动作 (用 KV cache)     │
│  2. 环境执行动作，返回 next_obs + reward     │
│  3. 收集 trajectory (obs, action, reward)  │
│  4. 计算 advantage (GAE)                   │
│  5. PPO update:                            │
│     - clipped policy gradient              │
│     - value function loss                  │
│     - entropy bonus                        │
│  6. (可选) World Model prediction error    │
│     作为 intrinsic reward                   │
└────────────────────────────────────────────┘
```

`RLLoss` 已实现 PPO 的核心组件：

| 组件 | 说明 |
|------|------|
| Policy gradient | 对 source / direction / split 三个 head 的 log-prob 求和，用 clipped ratio (0.8–1.2) |
| Value loss | MSE(predicted value, discounted returns) |
| Entropy bonus | 鼓励探索，防止策略过早收敛 |
| World model | 预测下一步状态 embedding，prediction error 作为 curiosity intrinsic reward |

World model 的作用：generals.io 的 reward 非常稀疏（只有胜/负），world model 提供 dense 的 intrinsic reward —— 模型对未知区域的预测越差，说明那里信息越多，鼓励模型去探索迷雾区域。

RL 阶段需要一个可交互的 generals.io 环境（self-play 或 vs bot），这部分尚未实现。架构上模型已经支持：
- `model.step()` 单步推理 + KV cache，适配 RL rollout
- `ValueHead` 输出 \[-1, 1\] 的局势评估
- `WorldModel` 可选开启 (`use_world_model=True`)

### Step 4: Visualisation

`scripts/replay_to_visual.py` provides the full pipeline from URL to interactive viewer:

```bash
uv run python scripts/replay_to_visual.py \
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
| Temporal causality | T=6 vs T=3 prefix outputs match exactly (diff = 0) | ✓ |
| KV cache inference | Cache grows correctly step-by-step (1→2→3→…→t) | ✓ |
| Model forward | All output shapes correct for batch + sequence dims | ✓ |
| Gradient flow | All parameters receive gradients through every component | ✓ |
| Training loop | Loss decreases on mock and real data; checkpoint save/load works | ✓ |

Run all tests:

```bash
uv run python tests/test_verify_simulator.py   # Simulator correctness
uv run python tests/test_model.py              # 24 model unit tests
uv run python tests/test_training_e2e.py       # End-to-end training
```
