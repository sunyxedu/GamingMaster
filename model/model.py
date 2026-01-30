"""
Complete TRM model for generals.io.

Implements the Tiny Recursive Model architecture from:
"Less is More: Recursive Reasoning with Tiny Networks" (arXiv:2510.04871)

Assembles all components:
- History Encoder
- Input Embedding
- TRM Recursive Backbone (with y/z dual state)
- Policy Heads (Move, Source, Direction, Ratio)
- Value Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .config import ModelConfig
from .history_encoder import HistoryEncoder
from .input_embedding import InputEmbedding
from .backbone import TRMBackbone
from .policy_head import CombinedPolicyHead
from .value_head import ValueHead


class GeneralsModel(nn.Module):
    """
    TRM model for generals.io.

    Architecture:
        History Encoder: Compresses sequence into latent state
        Input Embedding: Creates spatial tokens from current frame + latent
        TRM Backbone: Recursive Transformer with y/z dual state
        Policy Head: Move/Source/Direction/Ratio action prediction
        Value Head: Win probability estimation

    Args:
        config: Model configuration (optional, uses defaults if None)
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        self.config = config or ModelConfig()

        # History Encoder
        self.history_encoder = HistoryEncoder(
            grid_channels=self.config.grid_channels,
            global_features=self.config.global_features,
            d_model=self.config.history_d_model,
            n_heads=self.config.history_n_heads,
            n_layers=self.config.history_n_layers,
            max_seq_len=self.config.history_max_seq_len
        )

        # Input Embedding
        self.input_embedding = InputEmbedding(
            grid_channels=self.config.grid_channels,
            global_features=self.config.global_features,
            history_dim=self.config.history_d_model,
            d_model=self.config.d_model,
            height=self.config.grid_height,
            width=self.config.grid_width,
            n_tick_buckets=self.config.n_tick_buckets
        )

        # TRM Backbone with dual state (y, z)
        self.backbone = TRMBackbone(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            n_layers=self.config.n_layers,
            n_inner_cycles=self.config.n_inner_cycles,
            n_supervision_steps=self.config.n_supervision_steps,
            dropout=self.config.dropout
        )

        # Policy Head
        self.policy_head = CombinedPolicyHead(
            d_model=self.config.d_model,
            grid_height=self.config.grid_height,
            grid_width=self.config.grid_width
        )

        # Value Head
        self.value_head = ValueHead(
            d_model=self.config.d_model,
            d_global=self.config.global_features
        )

    def forward(
        self,
        grid_seq: torch.Tensor,
        globals_seq: torch.Tensor,
        tick_bucket_seq: torch.Tensor,
        target_source_idx: Optional[torch.Tensor] = None,
        sample_action: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass with full history sequence.

        Uses TRM's deep supervision: computes loss at each supervision step.
        NO masking applied - model learns valid actions through loss penalty.

        Args:
            grid_seq: Grid sequence [B, T, C, H, W]
            globals_seq: Global features sequence [B, T, G]
            tick_bucket_seq: Tick buckets [B, T]
            target_source_idx: Ground truth source indices [B, T] (for training)
            sample_action: Whether to sample actions

        Note:
            Action validity is checked via game_env.is_legal() for penalty computation,
            not via masking during forward pass.

        Returns:
            Dictionary with outputs for each timestep and supervision step
        """
        B, T, C, H, W = grid_seq.shape
        device = grid_seq.device

        # Encode history (using full sequence)
        history_latent = self.history_encoder(grid_seq, globals_seq)  # [B, D_h]

        # Process each timestep
        all_outputs = {
            'source_logits': [],
            'source_probs': [],
            'dir_logits': [],
            'dir_probs': [],
            'ratio_logits': [],
            'ratio_probs': [],
            'move_logits': [],
            'move_probs': [],
            'value': [],
            'halt_probs': [],
            'step_source_preds': [],  # Predictions at each supervision step
            'actions': []
        }

        for t in range(T):
            # Get current frame inputs
            grid_t = grid_seq[:, t]  # [B, C, H, W]
            globals_t = globals_seq[:, t]  # [B, G]
            tick_bucket_t = tick_bucket_seq[:, t].long()  # [B]

            # Input embedding
            tokens = self.input_embedding(
                grid_t, globals_t, history_latent, tick_bucket_t
            )  # [B, N, D]

            # TRM Backbone with deep recursion
            # Returns y (answer), z (latent), halt_probs, and all_y (intermediate answers)
            y, z, halt_probs, all_y = self.backbone(tokens, return_all_steps=True)
            # y: [B, N, D], z: [B, N, D]
            # halt_probs: [B, n_supervision_steps]
            # all_y: list of [B, N, D] for each supervision step

            # Policy prediction (on final y - the answer state)
            if target_source_idx is not None:
                # Training: use target source for direction/ratio
                target_src_t = target_source_idx[:, t]
                policy_out = self.policy_head.forward_training(
                    y, target_src_t, grid_t
                )

                # For halting loss: compute predictions at each supervision step
                step_source_preds = []
                if all_y is not None:
                    for step_y in all_y:
                        step_source_logits = self.policy_head.source_head(step_y)
                        step_source_pred = step_source_logits.argmax(dim=-1)  # [B]
                        step_source_preds.append(step_source_pred)

                    # Stack: [B, n_supervision_steps]
                    step_source_preds = torch.stack(step_source_preds, dim=1)
                    all_outputs['step_source_preds'].append(step_source_preds)
            else:
                # Inference: sample or argmax
                policy_out = self.policy_head(
                    y, grid_t, sample=sample_action
                )

            # Value prediction (uses both y and z for richer representation)
            # Combine y and z for value estimation
            combined_repr = y + z  # Or could concatenate
            value = self.value_head(combined_repr, globals_t)  # [B]

            # Store outputs
            all_outputs['source_logits'].append(policy_out['source_logits'])
            all_outputs['source_probs'].append(policy_out['source_probs'])
            all_outputs['dir_logits'].append(policy_out['dir_logits'])
            all_outputs['dir_probs'].append(policy_out['dir_probs'])
            all_outputs['ratio_logits'].append(policy_out['ratio_logits'])
            all_outputs['ratio_probs'].append(policy_out['ratio_probs'])
            all_outputs['move_logits'].append(policy_out['move_logits'])
            all_outputs['move_probs'].append(policy_out['move_probs'])
            all_outputs['value'].append(value)
            all_outputs['halt_probs'].append(halt_probs)

            if 'action' in policy_out:
                all_outputs['actions'].append(policy_out['action'])

        # Stack outputs along time dimension
        all_outputs['source_logits'] = torch.stack(all_outputs['source_logits'], dim=1)  # [B, T, N]
        all_outputs['source_probs'] = torch.stack(all_outputs['source_probs'], dim=1)
        all_outputs['dir_logits'] = torch.stack(all_outputs['dir_logits'], dim=1)  # [B, T, 4]
        all_outputs['dir_probs'] = torch.stack(all_outputs['dir_probs'], dim=1)
        all_outputs['ratio_logits'] = torch.stack(all_outputs['ratio_logits'], dim=1)  # [B, T, 2]
        all_outputs['ratio_probs'] = torch.stack(all_outputs['ratio_probs'], dim=1)
        all_outputs['move_logits'] = torch.stack(all_outputs['move_logits'], dim=1)  # [B, T, 2]
        all_outputs['move_probs'] = torch.stack(all_outputs['move_probs'], dim=1)
        all_outputs['value'] = torch.stack(all_outputs['value'], dim=1)  # [B, T]
        all_outputs['halt_probs'] = torch.stack(all_outputs['halt_probs'], dim=1)  # [B, T, n_steps]

        # Stack step predictions if available (for halting loss)
        if all_outputs['step_source_preds']:
            all_outputs['step_source_preds'] = torch.stack(
                all_outputs['step_source_preds'], dim=1
            )  # [B, T, n_supervision_steps]

        return all_outputs

    def inference(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        history_latent: torch.Tensor,
        tick_bucket: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Single-frame inference with adaptive halting.

        Args:
            grid: Current frame [B, C, H, W]
            globals_vec: Global features [B, G]
            history_latent: History latent state [B, D_h]
            tick_bucket: Tick bucket [B]
            sample: Whether to sample actions
            temperature: Sampling temperature

        Returns:
            Dictionary with policy, value, and action
        """
        # Input embedding
        tokens = self.input_embedding(
            grid, globals_vec, history_latent, tick_bucket.long()
        )

        # TRM Backbone (deterministic halting for inference)
        y, z, halt_probs, _ = self.backbone(tokens, deterministic=True, return_all_steps=False)

        # Policy
        policy_out = self.policy_head(
            y, grid, sample=sample, temperature=temperature
        )

        # Value
        combined_repr = y + z
        value = self.value_head(combined_repr, globals_vec)

        return {
            'move_probs': policy_out['move_probs'],
            'source_probs': policy_out['source_probs'],
            'dir_probs': policy_out['dir_probs'],
            'ratio_probs': policy_out['ratio_probs'],
            'value': value,
            'action': policy_out['action'],
            'halt_probs': halt_probs
        }

    def update_history_latent(
        self,
        latent_prev: torch.Tensor,
        grid: torch.Tensor,
        globals_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Update history latent with new observation.

        Args:
            latent_prev: Previous latent state [B, D_h]
            grid: Current frame [B, C, H, W]
            globals_vec: Current global features [B, G]

        Returns:
            Updated latent state [B, D_h]
        """
        return self.history_encoder.update_latent(latent_prev, grid, globals_vec)

    def get_initial_latent(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Get initial history latent state (zeros).

        Args:
            batch_size: Batch size
            device: Target device

        Returns:
            Zero latent state [B, D_h]
        """
        return self.history_encoder.get_initial_latent(batch_size, device)


