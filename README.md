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
│   ├── rl_utils.py           #   GAE, RolloutBuffer, KL reg, freeze helpers
│   └── components/           #   Reusable building blocks
│       ├── encoder.py        #     SpatialEncoder, HistoryEncoder, ResBlock
│       ├── transformer.py    #     TransformerBlock, SpatiotemporalBlock
│       ├── trm.py            #     TRM (Tiny Recursive Model) two-state pondering
│       ├── heads.py          #     Source / Direction / Split / Pass / Value heads
│       └── world_model.py    #     WorldModel (for RL curiosity reward)
├── env/                      # RL environment
│   └── generals_env.py       #   Two-player simultaneous-move Gym-like env
├── scripts/
│   ├── prepare_data.py       # CLI: batch download → filter → simulate → convert
│   └── replay_to_visual.py   # CLI: download one replay → simulate → HTML viewer
├── train_il.py               # Imitation Learning training script
├── train_rl.py               # RL training script (PPO self-play + IL regularisation)
├── tests/
│   ├── test_verify_simulator.py       # Simulator correctness (real replay)
│   ├── test_model.py                  # 26 unit tests for every model component
│   ├── test_audit_fixes.py            # 11 bug-fix verification tests
│   ├── test_comprehensive_audit.py    # 48 comprehensive audit tests (all modules)
│   ├── test_training_e2e.py           # End-to-end training loop tests
│   └── test_rl.py                     # RL utilities and rollout tests
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
uv run python -m scripts.prepare_data --players Human.exe --output data/output --max-replays 1000 --min-turns 80
```

### 4. Train (Imitation Learning)

```bash
uv run python train_il.py \
    --data_dir data/output/training/Human.exe \
    --epochs 50 \
    --batch_size 4 \
    --d_model 256
```

### 5. Train (Reinforcement Learning)

```bash
# Load from IL checkpoint, fine-tune with PPO self-play
uv run python train_rl.py \
    --il_checkpoint checkpoints/checkpoint_epoch50.pt \
    --il_data_dir data/output/training/MeltedToast
```

> RL training uses PPO self-play with joint IL regularisation (AlphaStar-style).
> Environment: `env/generals_env.py` provides a two-player simultaneous-move Gym-like environment with fog of war.
> Model-side support: `model.step()` single-step inference with KV cache, `ValueHead`, `RLLoss` (PPO), `WorldModel` (curiosity reward), `rl_utils.py` (GAE, RolloutBuffer, KL regularisation).

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
| **Spatiotemporal** (default) | `"spatiotemporal"` | Factored spatial + causal temporal attention (`n_st_layers`) → TRM pondering |
| GRU + TRM | `"gru"` | GRU history encoder → TRM two-state recursive pondering |

```
  grid_seq [B,T,12,25,25]              globals_seq [B,T,4]
  (11 one-hot state + 1 army)          (myArmy, myLand, opArmy, opLand)
          │                                     │
          ▼                                     ▼
  ┌───────────────────┐                ┌─────────────────┐
  │  SpatialEncoder   │                │   GlobalProj    │
  │  Conv + ResBlocks │                │  Linear→ReLU    │
  │  ×2               │                │  →Linear        │
  └────────┬──────────┘                └────────┬────────┘
           │ [B*T, d, 25, 25]                   │ [B*T, d]
           │                          broadcast │ → [B*T, d, 1, 1]
           └──────────── + ─────────────────────┘
                         │ [B, T, d, 25, 25]
                         ▼
                  flatten to [B, T, 625, d]
                         │
                  append learnable tokens:
                    HL (High-Level → value)
                    LL (Low-Level  → direction/split)
                         │ [B, T, 627, d]
                         │
                  + 2D spatial PE  (learnable row + col embeddings)
                  + sinusoidal temporal PE
                         │
                    ┌─────────┐
                    │pre_norm │ LayerNorm
                    └────┬────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Stage 1: SpatiotemporalBlock × n_st_layers  (separate wts)  │
  │                                                              │
  │  ┌─ Spatial Self-Attention ──────┐   627 tokens within each  │
  │  │  Add & Norm                   │   timestep interact       │
  │  ├─ Temporal Self-Attention ─────┤   per position across all │
  │  │  (causal: t sees 0..t only)   │   past timesteps          │
  │  │  Add & Norm                   │                           │
  │  ├─ MLP (d → d_ff → d) ──────────┤                           │
  │  │  Add & Norm                   │                           │
  │  └───────────────────────────────┘                           │
  └──────────────────────┬───────────────────────────────────────┘
                         │ [B, T, 627, d]
                    ┌─────────┐
                    │mid_norm │ LayerNorm
                    └────┬────┘
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  Stage 2: TRM Pondering  (shared reasoning module)           │
  │                                                              │
  │  z_H = H_init (fixed buffer),  z_L = L_init (fixed buffer)    │
  │                                                              │
  │  for h in H_cycles:                    ┌──────────────────┐  │
  │  │  for l in L_cycles:                 │ ReasoningModule  │  │
  │  │  │  z_L = reason(z_L, z_H + x)      │ (TRMBlock × N)   │  │
  │  │  z_H = reason(z_H, z_L)             │  Self-Attn→Norm  │  │
  │  (first H_cycles−1 with no_grad)       │  MLP→Norm        │  │
  │                                        └──────────────────┘  │
  │  q_halt = linear(z_H[:,0])  ── raw halt logit                │
  └──────────────────────┬───────────────────────────────────────┘
                         │ z_H [B*T, 627, d],  q_halt [B, T]
                    ┌──────────┐
                    │post_norm │ LayerNorm
                    └─────┬────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
      spatial[:625]    hl [pos 625]   ll [pos 626]
           │              │              │
           ▼              │              │
    ┌────────────┐        │              │
    │ SourceHead │        │              │
    └─────┬──────┘        │              │
          │               │              │
          ├──→ source_logits [B,T,625]  (raw, unmasked)
          │               │              │
          ▼               │              │
     extract src          │              │
     embedding at         │              │
     predicted pos        │              │
     (train: GT,          │              │
      infer: mask →       │              │
       sample)            │              │
          │               │              │
          │               ▼              │
          │         ┌───────────┐        │
          │         │ ValueHead │        │
          │         │ → tanh    │        │
          │         └─────┬─────┘        │
          │               │              │
          │          value [B,T]         │
          │          ∈ [−1, 1]           │
          │                              │
          │   cat(src_embed, ll_out)     │
          └──────────────┬───────────────┘
                         │ [B*T, 2d]
                    ┌────┴────┐
                    ▼         ▼
             ┌───────────┐ ┌───────────┐
             │  DirHead  │ │ SplitHead │
             └─────┬─────┘ └─────┬─────┘
                   │             │
            dir_logits      split_logits
            [B,T,4]         [B,T,2]
            (↓→↑←)          (all|50%)
                   │             │
                   └──────┬──────┘
                          ▼
                    ┌───────────┐
                    │ PassHead  │
                    │ ll_token  │
                    │ src_embed │
                    │ dir_logits│
                    │split_logits│
                    │ sin time  │
                    │ PE (32-d) │
                    └─────┬─────┘
                          │
                    pass_logits
                    [B,T] (>0 = pass/no move)
```

**Key design choices:**

- **Factored spatiotemporal attention**: Each `SpatiotemporalBlock` first runs spatial self-attention (627 tokens within the same timestep interact), then causal temporal self-attention (each of the 627 token positions attends to its own history across all past timesteps). This gives every grid cell a per-cell temporal memory — it knows exactly *where* and *when* it saw something before, unlike a pooled GRU that discards spatial structure.
- **Causal temporal masking**: Timestep *t* can only attend to timesteps 0..*t*, never the future. Verified by test: processing T=6 vs T=3 produces identical outputs for the first 3 steps (diff = 0).
- **KV cache for inference**: During single-step inference, each block caches past key/value tensors. Only the current timestep's tokens are computed; temporal attention queries the growing cache. No re-processing of past timesteps.
- **Autoregressive policy heads**: Source is predicted first from spatial tokens; its embedding is extracted and concatenated with the Low-Level decision token to predict Direction, Split, and Pass. During training, ground-truth source is used; during inference, the model samples.
- **Legal mask at sampling only**: `SourceHead` outputs raw logits over all 625 positions. The legal-move mask (owned tiles with army > 1) is applied externally only during inference sampling — not during training. This lets the model learn board understanding from the state encoding itself, rather than relying on a hard mask.
- **Decision tokens**: Two learnable tokens — High-Level (→ value prediction) and Low-Level (→ direction/split/pass) — are appended to the 625 spatial tokens and participate in both spatial and temporal attention.
- **2D spatial PE + sinusoidal temporal PE**: Spatial positions use learnable row + column embeddings (summed). Temporal positions use sinusoidal encoding, allowing generalisation to arbitrary game lengths.
- **TRM two-state pondering** ([arxiv 2510.04871](https://arxiv.org/abs/2510.04871)): Two state vectors — `z_H` (answer) and `z_L` (reasoning scratchpad) — are initialized from fixed buffers and iteratively refined by a single shared reasoning module. Inner loop refines `z_L` using `z_H + x` as injection; outer loop updates `z_H` from `z_L`. First `H_cycles−1` outer iterations run without gradient for memory efficiency; only the last iteration propagates gradients. At inference, ACT (Adaptive Computation Time) loops up to `halt_max_steps` times — `q_halt` head on `z_H` outputs a confidence logit; when positive, the model stops pondering early. Carry state `(z_H, z_L)` is passed between ACT steps so each round builds on the previous thinking.
- **World model** (optional): Predicts next-state embedding from (current state, action). Disabled during IL; during RL, prediction error serves as intrinsic curiosity reward.

Default config:
| Parameter | Value |
|-----------|-------|
| `d_model` | 256 |
| `n_heads` | 8 |
| `d_ff` | 512 |
| `n_st_layers` | 2 |
| `trm_H_cycles` | 3 |
| `trm_L_cycles` | 6 |
| `trm_block_layers` | 2 |
| `halt_max_steps` | 5 |
| `encoder_channels` | 128 |
| `temporal_type` | `"spatiotemporal"` |

### Step 3: Training

**Imitation Learning** (`train_il.py`)

Supervised learning from expert replays:

| Loss component | Weight | Mask | Description |
|----------------|--------|------|-------------|
| Source CE | 1.0 | valid (count>0) | Cross-entropy over 625 grid positions |
| Direction CE | 1.0 | valid (count>0) | Cross-entropy over 4 directions |
| Split CE | 0.5 | valid (count>0) | Cross-entropy over 2 split options (0=all-but-1, 1=50%) |
| Pass BCE | 1.0 | seq_mask (all) | Binary CE — target=1 when count=0, target=0 when count>0 |
| Value MSE | 0.5 | seq_mask (all) | Mean squared error on game value |
| Q-halt BCE | 0.5 | valid (count>0) | Binary CE — target 1 if source prediction correct, 0 otherwise |
| Move-Mismatch | 2.0 | valid & mismatch | Extra penalty on src+dir+split CE when any head predicts wrong |

Source/direction/split losses are masked to timesteps with moves (count>0). Pass and value losses apply to all valid sequence timesteps. Move-mismatch loss re-penalizes timesteps where any of source/direction/split/pass predictions are wrong.

**Reinforcement Learning** (`model/losses.py` — `RLLoss`)

After IL produces a reasonable initial policy, RL fine-tunes it to surpass human replay quality. Overall flow:

```
IL pretrained model
  │
  ▼
┌───────────────────────────────────────────────┐
│  RL Training Loop (PPO)                       │
│                                               │
│  1. model.step() generates action (KV cache)  │
│  2. Environment executes, returns next_obs    │
│     + reward                                  │
│  3. Collect trajectory (obs, action, reward)  │
│  4. Compute advantage (GAE)                   │
│  5. PPO update:                               │
│     - clipped policy gradient                 │
│     - value function loss                     │
│     - entropy bonus                           │
│  6. (Optional) World Model prediction error   │
│     as intrinsic reward                       │
└───────────────────────────────────────────────┘
```

`RLLoss` implements the core PPO components:

| Component | Description |
|-----------|-------------|
| Policy gradient | Sum of log-probs across source / direction / split heads, clipped ratio (0.8–1.2) |
| Value loss | MSE(predicted value, discounted returns) |
| Entropy bonus | Encourages exploration, prevents premature convergence |
| World model | Predicts next-state embedding; prediction error serves as curiosity intrinsic reward |

Why the world model matters: generals.io has extremely sparse reward (only win/loss at game end). The world model provides dense intrinsic reward — the worse the model's prediction for an unseen area, the more information lies there, encouraging the agent to explore through fog of war.

RL training components:
- `env/generals_env.py` — two-player simultaneous-move environment with fog of war, random map generation, and dense reward shaping
- `train_rl.py` — full PPO self-play loop with joint IL regularisation and KL constraint
- `model/rl_utils.py` — GAE, RolloutBuffer, encoder freeze/unfreeze helpers
- `model.step()` — single-step inference with KV cache, suitable for RL rollouts
- `ValueHead` outputs game value in \[-1, 1\]
- `WorldModel` can be enabled via `use_world_model=True`

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
uv run python tests/test_verify_simulator.py        # Simulator correctness
uv run python tests/test_model.py                   # 26 model unit tests
uv run python tests/test_audit_fixes.py             # 11 bug-fix verification tests
uv run python tests/test_comprehensive_audit.py     # 48 comprehensive audit tests
uv run python tests/test_training_e2e.py            # End-to-end training
```
