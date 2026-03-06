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
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 512
    dropout: float = 0.1

    # Stage 1: Spatiotemporal encoding (separate weights per layer)
    n_st_layers: int = 2

    # Stage 2: TRM pondering
    trm_H_cycles: int = 4    # outer loop — refine answer
    trm_L_cycles: int = 4    # inner loop — refine reasoning
    trm_block_layers: int = 2  # transformer layers inside the shared reasoning module

    # Temporal modelling
    #   "spatiotemporal" – Stage1: factored spatiotemporal → Stage2: TRM
    #   "gru"            – GRU history encoder → TRM
    temporal_type: str = "spatiotemporal"

    # GRU-specific (only used when temporal_type="gru")
    gru_layers: int = 1

    # Spatial encoder
    encoder_channels: int = 128

    # Action space
    n_directions: int = 4
    n_split: int = 2

    # World model (optional, for RL)
    use_world_model: bool = False
    world_model_hidden: int = 256
