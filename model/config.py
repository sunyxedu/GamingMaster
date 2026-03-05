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
    d_ff: int = 512  # feed-forward hidden dim
    dropout: float = 0.1

    # Number of transformer layers
    n_layers: int = 4

    # Temporal modelling
    #   "gru"             – GRU history encoder + spatial-only TransformerBlock ×k (pondering)
    #   "spatiotemporal"  – factored spatial+temporal attention (SpatiotemporalBlock ×n_layers)
    temporal_type: str = "spatiotemporal"

    # GRU-specific (only used when temporal_type="gru")
    n_ponder_steps: int = 4  # shared-weight pondering iterations
    gru_layers: int = 1

    # Spatial encoder
    encoder_channels: int = 128

    # Action space
    n_directions: int = 4  # down, right, up, left
    n_split: int = 2  # all-but-1, 50%

    # World model (optional, for RL)
    use_world_model: bool = False
    world_model_hidden: int = 256
