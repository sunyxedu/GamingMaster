"""
Training infrastructure.

Key components:
- RLTrainer: Main RL trainer using direct policy sampling with V-trace
- ImitationLoss: Loss for supervised pre-training
- V-trace utilities for off-policy correction
- EMA: Exponential Moving Average for stable evaluation (from TRM paper)
"""

from training.loss import ImitationLoss
from training.replay_buffer import ReplayBuffer
from training.rl_trainer import RLTrainer
from training.vtrace import (
    compute_importance_weights,
    vtrace_targets,
    compute_vtrace_loss,
    VTraceFromLogits,
    compute_log_prob_from_policy_heads
)
from training.ema import EMA, EMATrainer, create_ema_model

__all__ = [
    # Losses
    'ImitationLoss',
    # Trainers
    'RLTrainer',
    'EMATrainer',
    # EMA
    'EMA',
    'create_ema_model',
    # Buffers
    'ReplayBuffer',
    # V-trace utilities
    'compute_importance_weights',
    'vtrace_targets',
    'compute_vtrace_loss',
    'VTraceFromLogits',
    'compute_log_prob_from_policy_heads'
]
