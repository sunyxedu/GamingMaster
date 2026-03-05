"""Generals.io neural network model."""

from model.config import ModelConfig
from model.losses import ImitationLoss, RLLoss
from model.network import GeneralsNetwork

__all__ = [
    "ModelConfig",
    "GeneralsNetwork",
    "ImitationLoss",
    "RLLoss",
]
