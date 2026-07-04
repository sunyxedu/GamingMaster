"""Generals.io neural network model."""

from model.components import (
    DirectionHead,
    ResBlock,
    SourceHead,
    SpatialEncoder,
    SplitHead,
    TransformerBlock,
    ValueHead,
)
from model.config import ModelConfig
from model.losses import ImitationLoss, RLLoss
from model.network import GeneralsNetwork

__all__ = [
    "ModelConfig",
    "GeneralsNetwork",
    "ImitationLoss",
    "RLLoss",
    # components
    "ResBlock",
    "SpatialEncoder",
    "TransformerBlock",
    "SourceHead",
    "DirectionHead",
    "SplitHead",
    "ValueHead",
]
