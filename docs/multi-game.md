# Multi-Game Architecture

The network core is game-agnostic. What used to be a generals.io-only model is now a
fixed two-stage core plus per-game plug-ins:

```
                 obs dict (game-specific tensors)
                          │
                ┌─────────▼──────────┐
                │ ObservationEncoder │   per game  (model/games/<game>.py)
                │  → [B,T,S_game,d]  │
                └─────────┬──────────┘
                          │  + HL/LL decision tokens, + PE
                ┌─────────▼──────────┐
                │ Stage 1            │   shared    (model/game_network.py)
                │ SpatiotemporalBlock│
                │ × n_st_layers      │
                ├────────────────────┤
                │ Stage 2            │   shared
                │ TRM pondering      │
                └─────────┬──────────┘
                          │  z_H
              ┌───────────┼───────────────┐
    ┌─────────▼─────────┐ │      ┌────────▼────────┐
    │   ActionDecoder   │ │      │    ValueHead    │  shared (reads HL token)
    │     per game      │ │      └─────────────────┘
    └───────────────────┘ │
     logits / sample /    q_halt
     log_prob_entropy / il_loss
```

Everything between the two boundaries — decision tokens, positional-embedding
assembly, Stage-1 blocks, TRM recursion/ACT, value head, weight init — lives in
`GameNetwork` and never changes per game. The TRM's q_halt token index is rewired
automatically to `obs_encoder.n_tokens` (the HL token slot).

## The two interfaces (`model/games/base.py`)

**`ObservationEncoder`** — raw per-frame observation dict → token sequence.

| Contract point | Why |
|---|---|
| `n_tokens` is a constant | the core's PE and the TRM q-token index are static |
| `forward(obs, frame_mask) → [B,T,S,d]` | `frame_mask` must exclude padded frames from any batch statistics (BatchNorm) |
| `encode_step(obs) → [B,S,d]` | single-frame path for cached rollout inference (default: T=1 forward) |
| `positional_embedding() → [1,S,d]` | game-token PE; the core appends HL/LL PE after it |
| variable-count entities → fixed padded slots | zero absent features + learnable "empty" embedding; never shrink the sequence |
| optional `supports_next_obs` / `next_obs_outputs` | hosts the self-supervised world-model objective (`cfg.predict_next_obs`) |

**`ActionDecoder`** — pondered tokens → hierarchical action distribution.

| Contract point | Why |
|---|---|
| `decode(tokens, hl, ll, B, T, t0, legal, labels)` | returns logits `[B,T,…]`; commits conditioning choices (teacher-forced from `labels`, or self-sampled) and echoes them in the pred dict (like generals' `src_idx`) |
| `sample(pred, legal)` | rollout: samples leaf heads and returns `action_log_prob` |
| `log_prob_entropy(pred, actions, legal)` | PPO replay: differentiable log-prob + entropy for stored actions |
| **PPO-exactness invariant** | `sample()`'s log-prob must equal `log_prob_entropy()`'s when `decode()` is re-run with labels = the stored conditioning choices — rollout and update must define the same policy (verified per game by a rollout-vs-replay test at fixed `act_steps`) |
| `il_loss(pred, tgt, seq_mask)` | per-game imitation loss; dependent heads are condition-weighted by "was the conditioning index the ground truth" (generals keeps its original `ImitationLoss`) |

Shared building blocks in `base.py`: `PointerHead` (score tokens against query
vectors — the universal "pick an entity/tile" head), `MLPHead`, masked
sampling/log-prob/entropy utilities with degenerate-row (all-masked, NaN) guards.

## Registered games (`model/games/registry.py`)

```python
from model.games import build_game_network

net = build_game_network("chess", d_model=256)
net = build_game_network("go", board_h=9, board_w=9)
```

| Game | Encoder | Decoder | Action structure |
|---|---|---|---|
| `generals` | `GeneralsEncoder` — grid conv + globals | `GeneralsDecoder` | source tile → direction/split, pass |
| `chess` | `BoardEncoder` — piece-plane conv | `FromToDecoder` | from-square → to-square → promotion |
| `go` | `BoardEncoder` | `PlacementDecoder` | point + pass |
| `othello` | `BoardEncoder` (8×8, 3 planes) | `PlacementDecoder` | point + pass |
| `gomoku` | `BoardEncoder` (15×15, 3 planes) | `PlacementDecoder` | point (+ pass, maskable) |
| `shogi` | `ShogiEncoder` — board conv + 14 pieces-in-hand tokens | `ShogiDecoder` | move (from → to → promote?) OR drop (piece → square) |
| `lux` | `LuxEncoder` — resource-map conv + unit slots | `LuxDecoder` | **simultaneous** per-unit categorical (factored joint policy) |
| `clash_royale` | `ClashRoyaleEncoder` — arena conv + hand-card tokens | `CardPlacementDecoder` | wait / card → placement cell |
| `hearthstone` | `HearthstoneEncoder` — 27 entity slots | `EntityActionDecoder` | action-type → actor → target → position (per-type `ARG_USES` gating) |
| `starcraft` | `StarCraftEncoder` — minimap conv + unit slots | `RTSDecoder` | action-type → actor unit → target unit / map location + queued |

`lux` demonstrates the second decoder pattern: instead of one hierarchical
action per turn, every unit token gets its own categorical head and the joint
log-prob is the sum over present units — replay stays exact with no
teacher-forcing chain. Use it as the template for games where many pieces act
at once.

`GeneralsNetwork` (model/network.py) is now a thin adapter over the same core: it
keeps the historical tensor-argument API (`forward(grid_seq, globals_seq, …)`,
`step`, `step_rl`, `compute_action_log_prob[_rl]`), the legacy attribute names
(`model.spatial_encoder`, `model.source_head`, …), and remaps pre-refactor
checkpoint keys on `load_state_dict` (`LEGACY_KEY_MAP`) — old checkpoints,
training scripts and all pre-existing tests work unchanged.

## Adding a new game

1. **Config**: subclass `ModelConfig` (dataclass) with your game's dimensions.
   Core fields (`d_model`, `n_heads`, `d_ff`, `n_st_layers`, `trm_*`,
   `temporal_mixer`, `halt_max_steps`, `dropout`) are consumed by the core.
2. **Encoder**: fixed token layout; conv stacks for grids (mirror
   `GeneralsEncoder`, including the masked-BatchNorm `frame_mask` handling),
   embedding+projection for entity slots (zero absent features + empty
   embedding). Provide the PE for your tokens.
3. **Decoder**: build the action hierarchy from `PointerHead`/`MLPHead`.
   Sample conditioning heads inside `decode()` (with teacher-forcing labels for
   IL/PPO replay), leaf heads inside `sample()`. Gate per-action-type argument
   heads with an `ARG_USES`-style buffer if your action space is typed.
4. **Register** it in `model/games/registry.py`.
5. **Test** (see `tests/test_games_*.py` for the template): forward shapes,
   gradient flow, `step()` cache chaining, **the rollout-vs-replay PPO-exactness
   test**, legal-mask respect, `il_loss` finiteness.

What a new game does *not* need: any change to Stage 1, the TRM, ACT handling,
the value head, checkpointing, or the training-loop mechanics.

## Data & environments

Model adapters are only half the story — training needs human replays (IL) and a
fast simulator (RL). See [game-data-sources.md](game-data-sources.md) for the
per-game survey of datasets, APIs and simulators, with feasibility grades.
Summary: chess and Go have S-grade infrastructure (Lichess CC0 dumps, OGS/Tygem,
python-chess/PettingZoo/KataGo); StarCraft II is feasible but compute-heavy
(Blizzard replay packs + PySC2); Hearthstone lacks public bulk replay data;
Clash Royale currently lacks both frame-level data and an accurate simulator.
