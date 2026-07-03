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

    # Temporal modelling
    #   "spatiotemporal" – factored spatiotemporal → TRM
    #   "gru"            – GRU history encoder → TRM
    temporal_type: str = "spatiotemporal"

    # Temporal mixer inside SpatiotemporalBlock:
    #   "attention"  – causal temporal self-attention over the full history
    #                  (KV cache grows O(T) at inference)
    #   "recurrent"  – token-wise gated linear recurrence: episode memory
    #                  lives in a fixed-size state ("in the neurons"),
    #                  O(1) per step, full BPTT during training
    temporal_mixer: str = "attention"

    # GRU-specific (only used when temporal_type="gru")
    gru_layers: int = 1

    # Spatial encoder
    encoder_channels: int = 192

    # Action space
    n_directions: int = 4
    n_split: int = 2  # 0=all-but-1, 1=50%

    # World model (optional, for RL)
    use_world_model: bool = False
    world_model_hidden: int = 256
