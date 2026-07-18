"""Model configuration for the Generals.io neural network."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Grid dimensions
    grid_h: int = 25
    grid_w: int = 25

    # Input channels: 11 one-hot state types + 1 army channel
    grid_channels: int = 12
    n_grid_types: int = 11
    global_features: int = 4  # myArmy, myLand, opArmy, opLand
                              # (7 with turn_globals: + production-cycle
                              #  phase, half-turn parity, scaled turn)

    # Generals-adapter input features (both default OFF for byte-compatible
    # behavior with existing checkpoints; enabling them zero-pads the first
    # conv / global projection — function-preserving — via the
    # shape-adaptive loader in GeneralsNetwork.load_state_dict):
    #   derive_army_channels — GeneralsEncoder derives 3 LINEAR army
    #     channels (own/enemy/neutral-city, army/50 clipped) from the grid
    #     before the spatial encoder. The single log1p(army)/10 channel
    #     makes exact army comparisons nearly invisible in input space
    #     (200-vs-250 differs by ~0.02); combat arithmetic is linear.
    #   turn_globals — the env appends (turn%50)/50, turn%2, turn/500 to
    #     the globals (set global_features=7 together with this): the
    #     temporal attention has no positional encoding, so the policy
    #     otherwise cannot see the 50-tick production cycle at all.
    derive_army_channels: bool = False
    turn_globals: bool = False

    # Core model dimensions
    d_model: int = 384
    n_heads: int = 12
    d_ff: int = 1024
    dropout: float = 0.1

    # Stage 1: Spatiotemporal encoding (separate weights per layer)
    n_st_layers: int = 3

    # Stage 2: TRM pondering
    trm_H_cycles: int = 3    # outer loop — refine answer
    trm_L_cycles: int = 6    # inner loop — refine reasoning
    trm_block_layers: int = 3  # transformer layers inside the shared reasoning module
    halt_max_steps: int = 5   # ACT pondering steps at inference (1 = no ACT)

    # Temporal modelling ("spatiotemporal" is the only supported value; the
    # field is kept so old checkpoints' configs load unchanged)
    temporal_type: str = "spatiotemporal"

    # Temporal mixer inside SpatiotemporalBlock:
    #   "attention"  – causal temporal self-attention over the full history
    #                  (KV cache grows O(T) at inference)
    #   "recurrent"  – token-wise gated linear recurrence: episode memory
    #                  lives in a fixed-size state ("in the neurons"),
    #                  O(1) per step, full BPTT during training
    temporal_mixer: str = "attention"

    # Spatial encoder
    encoder_channels: int = 192

    # Action space
    n_directions: int = 4
    n_split: int = 2  # 0=all-but-1, 1=50%

    # Self-supervised next-observation prediction from the Stage-1 belief
    # tokens (game-agnostic memory pressure: the belief must retain whatever
    # is needed to predict the next frame — e.g. fogged tiles re-entering
    # vision). Spatiotemporal path only.
    predict_next_obs: bool = False

    # HL-Gauss distributional value head (phase-6, AverageJoe-style):
    # the critic outputs a categorical distribution over hl_gauss_bins
    # support points in [-vmax, vmax] and is trained with cross-entropy
    # against a Gaussian-smeared target (histogram loss). The scalar value
    # exposed to GAE/eval is the distribution's expectation, so every
    # consumer of out["value"] is unchanged. vmax=1.5 covers the sparse
    # ±1 terminal plus the ±0.35-scaled truncation score with margin;
    # sigma≈2.5 bin widths mirrors AverageJoe's 128-bin/σ0.04 on [-1,1].
    # Sticky via checkpoint config like the input-feature flags.
    hl_gauss_value: bool = False
    hl_gauss_bins: int = 128
    hl_gauss_vmax: float = 1.5
    hl_gauss_sigma: float = 0.06

