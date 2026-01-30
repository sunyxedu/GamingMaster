"""
Model configuration for TRM architecture.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the GeneralsModel."""

    # Grid dimensions
    grid_height: int = 25
    grid_width: int = 25
    grid_channels: int = 12  # 11 state one-hot + 1 army channel
    global_features: int = 4  # myArmy, myLand, opArmy, opLand

    # History Encoder
    history_d_model: int = 256
    history_n_heads: int = 4
    history_n_layers: int = 2
    history_max_seq_len: int = 512

    # Input Embedding
    d_model: int = 512
    n_tokens: int = 625  # 25 x 25

    # TRM Backbone (from "Less is More" paper)
    n_heads: int = 8
    d_ff: int = 2048  # 4 * d_model
    n_layers: int = 2  # Single tiny network with 2 layers
    n_inner_cycles: int = 6  # Inner recursion cycles per supervision step
    n_supervision_steps: int = 3  # Number of supervision steps (T in paper)
    dropout: float = 0.0  # TRM uses no dropout

    # Policy Head
    n_directions: int = 4  # down, right, up, left
    n_ratios: int = 2  # all-but-1, 50%

    # Tick embedding
    n_tick_buckets: int = 26  # 0-25

    # Training
    ema_decay: float = 0.999

    @property
    def n_positions(self) -> int:
        """Total number of source positions."""
        return self.grid_height * self.grid_width


@dataclass
class TrainingConfig:
    """Configuration for training."""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    replay_buffer_size: int = 100000

    # Loss weights
    value_loss_weight: float = 1.0
    halt_loss_weight: float = 0.01

    # Self-play
    selfplay_games_per_iteration: int = 100
    training_steps_per_iteration: int = 1000

    # Logging
    log_interval: int = 100
    checkpoint_interval: int = 1000
