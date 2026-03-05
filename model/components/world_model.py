"""World model — predicts next-state embedding for RL curiosity reward."""

import torch
import torch.nn as nn

from model.config import ModelConfig


class WorldModel(nn.Module):
    """
    Predicts next-state embedding from (current embedding, action).

    During IL training this module is unused (caller feeds zeros).
    During RL it provides intrinsic reward via prediction error.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        action_dim = cfg.grid_h * cfg.grid_w + cfg.n_directions + cfg.n_split
        self.encoder = nn.Sequential(
            nn.Linear(cfg.d_model + action_dim, cfg.world_model_hidden),
            nn.ReLU(),
            nn.Linear(cfg.world_model_hidden, cfg.d_model),
        )

    def forward(
        self,
        state_embed: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([state_embed, action_onehot], dim=-1)
        return self.encoder(x)
