"""
RL Trainer.

Features:
- Direct policy sampling
- Actor-Critic with V-trace off-policy correction
- Entropy bonus for exploration
- Optional KL penalty to stay close to supervised policy
"""

import copy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Any, List, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from model.config import ModelConfig, TrainingConfig
from model.model import GeneralsModel
from training.vtrace import (
    compute_importance_weights,
    vtrace_targets,
    compute_log_prob_from_policy_heads
)
from data_processing.processing.game_env import GameEnvironment
from data_processing.processing.models import MapData


@dataclass
class Trajectory:
    """Single trajectory from self-play."""
    # Observations
    grids: List[torch.Tensor]  # [12, H, W] each
    globals_: List[torch.Tensor]  # [4] each
    legal_masks: List[torch.Tensor]  # [1, H, W] each
    tick_buckets: List[int]

    # Actions taken
    source_idxs: List[int]
    dir_ids: List[int]
    ratio_ids: List[int]

    # Behavior log probs (for off-policy correction)
    behavior_log_probs: List[float]

    # Rewards and done flags
    rewards: List[float]
    dones: List[bool]

    # Value for bootstrap (final state value)
    bootstrap_value: float = 0.0

    # Player index (for assigning final rewards)
    player_idx: int = 0


@dataclass
class TrajectoryBuffer:
    """Buffer for storing trajectories."""
    trajectories: List[Trajectory] = field(default_factory=list)
    max_size: int = 1000

    def add(self, trajectory: Trajectory):
        """Add a trajectory to buffer."""
        self.trajectories.append(trajectory)
        if len(self.trajectories) > self.max_size:
            self.trajectories.pop(0)

    def sample_batch(self, batch_size: int, seq_len: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of trajectory segments.

        Args:
            batch_size: Number of segments to sample
            seq_len: Length of each segment

        Returns:
            Batched tensors for training
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories in buffer")

        # Filter trajectories that are long enough
        valid_trajs = [t for t in self.trajectories if len(t.grids) >= seq_len]

        # If no trajectories are long enough, use all trajectories but truncate seq_len
        if len(valid_trajs) == 0:
            # Find the max length among all trajectories
            max_len = max(len(t.grids) for t in self.trajectories)
            seq_len = min(seq_len, max_len)
            valid_trajs = [t for t in self.trajectories if len(t.grids) >= 1]

        # Pad by reusing trajectories if needed
        if len(valid_trajs) < batch_size:
            valid_trajs = valid_trajs * (batch_size // len(valid_trajs) + 1)

        # Sample trajectories and starting points
        segments = []
        for _ in range(batch_size):
            traj = random.choice(valid_trajs)
            max_start = len(traj.grids) - seq_len
            start_idx = random.randint(0, max(0, max_start))
            segments.append((traj, start_idx))

        # Collate into batch
        return self._collate_segments(segments, seq_len)

    def _collate_segments(
        self,
        segments: List[Tuple[Trajectory, int]],
        seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """Collate trajectory segments into batch tensors."""
        B = len(segments)

        # Get dimensions from first trajectory
        first_traj = segments[0][0]
        C, H, W = first_traj.grids[0].shape

        # Initialize tensors
        grids = torch.zeros(B, seq_len, C, H, W)
        globals_ = torch.zeros(B, seq_len, 4)
        legal_masks = torch.zeros(B, seq_len, 1, H, W, dtype=torch.bool)
        tick_buckets = torch.zeros(B, seq_len, dtype=torch.long)

        source_idxs = torch.zeros(B, seq_len, dtype=torch.long)
        dir_ids = torch.zeros(B, seq_len, dtype=torch.long)
        ratio_ids = torch.zeros(B, seq_len, dtype=torch.long)

        behavior_log_probs = torch.zeros(B, seq_len)
        rewards = torch.zeros(B, seq_len)
        dones = torch.zeros(B, seq_len, dtype=torch.bool)
        bootstrap_values = torch.zeros(B)

        masks = torch.zeros(B, seq_len, dtype=torch.bool)

        for b, (traj, start_idx) in enumerate(segments):
            actual_len = min(seq_len, len(traj.grids) - start_idx)

            for t in range(actual_len):
                idx = start_idx + t
                grids[b, t] = traj.grids[idx]
                globals_[b, t] = traj.globals_[idx]
                legal_masks[b, t] = traj.legal_masks[idx]
                tick_buckets[b, t] = traj.tick_buckets[idx]

                source_idxs[b, t] = traj.source_idxs[idx]
                dir_ids[b, t] = traj.dir_ids[idx]
                ratio_ids[b, t] = traj.ratio_ids[idx]

                behavior_log_probs[b, t] = traj.behavior_log_probs[idx]
                rewards[b, t] = traj.rewards[idx]
                dones[b, t] = traj.dones[idx]

                masks[b, t] = True

            bootstrap_values[b] = traj.bootstrap_value

        return {
            'grids': grids,
            'globals': globals_,
            'legal_masks': legal_masks,
            'tick_buckets': tick_buckets,
            'source_idxs': source_idxs,
            'dir_ids': dir_ids,
            'ratio_ids': ratio_ids,
            'behavior_log_probs': behavior_log_probs,
            'rewards': rewards,
            'dones': dones,
            'bootstrap_values': bootstrap_values,
            'masks': masks
        }

    def __len__(self):
        return len(self.trajectories)

    def clear(self):
        """Clear all trajectories."""
        self.trajectories.clear()


class RLTrainer:
    """
    RL Trainer.

    Uses:
    - Direct policy sampling
    - Actor-Critic with V-trace off-policy correction
    - Entropy bonus for exploration
    """

    def __init__(
        self,
        model: GeneralsModel,
        config: Optional[TrainingConfig] = None,
        device: Optional[torch.device] = None,
        map_size: Tuple[int, int] = (25, 25),
        max_turns: int = 100,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        value_loss_weight: float = 0.5,
        entropy_weight: float = 0.01,
        kl_weight: float = 0.0,  # Optional KL penalty
        ema_decay: float = 0.999,  # EMA decay for target model
    ):
        self.model = model
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.map_size = map_size
        self.max_turns = max_turns

        # V-trace parameters
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.ema_decay = ema_decay

        # Move model to device
        self.model = self.model.to(self.device)

        # Create target model (EMA) for stable value targets
        self.target_model = copy.deepcopy(model)
        self.target_model.eval()
        for param in self.target_model.parameters():
            param.requires_grad = False

        # Optional: reference policy for KL penalty (frozen copy of initial policy)
        self.reference_model = None
        if kl_weight > 0:
            self.reference_model = copy.deepcopy(model)
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=1e-6
        )

        # Trajectory buffer
        self.trajectory_buffer = TrajectoryBuffer(max_size=1000)

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')

    def _update_ema(self):
        """Update target model with exponential moving average."""
        with torch.no_grad():
            for p, p_target in zip(
                self.model.parameters(),
                self.target_model.parameters()
            ):
                p_target.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

    def self_play(self, num_games: int = 1) -> List[Dict[str, Any]]:
        """
        Run self-play games with direct policy sampling.

        Args:
            num_games: Number of games to play

        Returns:
            List of game statistics
        """
        self.model.eval()
        game_stats = []

        for game_idx in range(num_games):
            trajectories, stats = self._play_one_game()
            for traj in trajectories:
                self.trajectory_buffer.add(traj)
            game_stats.append(stats)

        return game_stats

    def _play_one_game(self) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """
        Play one self-play game with direct policy sampling.

        Returns:
            Tuple of (trajectories for each player, game statistics)
        """
        map_data = MapData.create_random(
            width=self.map_size[0],
            height=self.map_size[1]
        )
        env = GameEnvironment()
        state = env.reset(map_data)

        # Initialize trajectories for both players
        trajectories = [
            Trajectory(
                grids=[], globals_=[], legal_masks=[], tick_buckets=[],
                source_idxs=[], dir_ids=[], ratio_ids=[],
                behavior_log_probs=[], rewards=[], dones=[],
                player_idx=i
            )
            for i in range(2)
        ]

        # History latents for both players
        history_latents = [
            self.model.get_initial_latent(1, self.device),
            self.model.get_initial_latent(1, self.device)
        ]

        for turn in range(self.max_turns):
            current_player = state.current_player

            # Get state tensors
            grid = state.get_grid_tensor().unsqueeze(0).to(self.device)
            globals_vec = state.get_globals_tensor().unsqueeze(0).to(self.device)
            legal_mask = state.get_legal_mask_tensor().unsqueeze(0).to(self.device)
            history_latent = history_latents[current_player]
            # FIX: Use same formula as dataset: ((t % 50) + 1) // 2
            tick_bucket = torch.tensor([((state.turn % 50) + 1) // 2], device=self.device)

            # Get legal actions
            legal_actions = env.get_legal_actions()

            if not legal_actions:
                # No legal moves, skip turn (pass)
                state.current_player = 1 - state.current_player
                state.turn += 1
                continue

            # Sample action from policy (direct sampling)
            with torch.no_grad():
                action, log_prob, value = self._sample_action(
                    grid, globals_vec, legal_mask, history_latent, tick_bucket
                )

            # Validate action is in legal actions
            if action not in legal_actions:
                # Fallback to random legal action
                action = random.choice(legal_actions)
                log_prob = 0.0  # Approximate

            # Store experience in current player's trajectory
            traj = trajectories[current_player]
            traj.grids.append(grid.squeeze(0).cpu())
            traj.globals_.append(globals_vec.squeeze(0).cpu())
            traj.legal_masks.append(legal_mask.squeeze(0).cpu())
            traj.tick_buckets.append(tick_bucket.item())
            traj.source_idxs.append(action[0])
            traj.dir_ids.append(action[1])
            traj.ratio_ids.append(action[2])
            traj.behavior_log_probs.append(log_prob)

            # Update history latent
            with torch.no_grad():
                history_latents[current_player] = self.model.update_history_latent(
                    history_latent, grid, globals_vec
                )

            # Execute action
            state, reward, done, info = env.step(action)

            # Store reward (will be 0 except at game end)
            traj.rewards.append(0.0)  # Placeholder, assign at end
            traj.dones.append(done)

            if done:
                winner = info.get('winner', -1)
                # Assign final rewards
                for i, t in enumerate(trajectories):
                    if len(t.rewards) > 0:
                        # Final reward based on outcome
                        if winner == -1:
                            t.rewards[-1] = 0.0  # Draw
                        elif i == winner:
                            t.rewards[-1] = 1.0  # Win
                        else:
                            t.rewards[-1] = -1.0  # Loss
                        t.bootstrap_value = 0.0  # Terminal state
                break
        else:
            # Game didn't finish, use value estimate as bootstrap
            for i, t in enumerate(trajectories):
                if len(t.grids) > 0:
                    with torch.no_grad():
                        grid = t.grids[-1].unsqueeze(0).to(self.device)
                        globals_vec = t.globals_[-1].unsqueeze(0).to(self.device)
                        history_latent = history_latents[i]
                        tick_bucket = torch.tensor([t.tick_buckets[-1]], device=self.device)

                        tokens = self.model.input_embedding(
                            grid, globals_vec, history_latent, tick_bucket
                        )
                        y, z, _, _ = self.model.backbone(tokens)
                        value = self.model.value_head(y + z, globals_vec)
                        t.bootstrap_value = value.item()

        # Filter out empty trajectories
        trajectories = [t for t in trajectories if len(t.grids) > 0]

        stats = {
            'turns': turn + 1,
            'winner': info.get('winner', -1) if done else -1,
            'done': done
        }

        return trajectories, stats

    def _sample_action(
        self,
        grid: torch.Tensor,
        globals_vec: torch.Tensor,
        legal_mask: torch.Tensor,
        history_latent: torch.Tensor,
        tick_bucket: torch.Tensor
    ) -> Tuple[Tuple[int, int, int], float, float]:
        """
        Sample action from policy using direct sampling.

        Args:
            grid: Grid tensor [1, C, H, W]
            globals_vec: Global features [1, 4]
            legal_mask: Legal source mask [1, 1, H, W]
            history_latent: History latent state
            tick_bucket: Tick bucket index

        Returns:
            Tuple of (action, log_prob, value)
        """
        # Forward pass
        tokens = self.model.input_embedding(
            grid, globals_vec, history_latent, tick_bucket
        )
        y, z, _, _ = self.model.backbone(tokens)

        # Get policy output (with sampling)
        policy_out = self.model.policy_head(
            y, legal_mask, grid, sample=True
        )

        # Get value
        value = self.model.value_head(y + z, globals_vec)

        # Extract action from nested 'action' dict
        action_dict = policy_out['action']
        source_idx = action_dict['source_idx'].item()
        dir_id = action_dict['dir_id'].item()
        ratio_id = action_dict['ratio_id'].item()
        action = (source_idx, dir_id, ratio_id)

        # Compute log probability
        log_prob = self._compute_action_log_prob(
            policy_out['source_logits'],
            policy_out['dir_logits'],
            policy_out['ratio_logits'],
            source_idx, dir_id, ratio_id
        )

        return action, log_prob, value.item()

    def _compute_action_log_prob(
        self,
        source_logits: torch.Tensor,
        dir_logits: torch.Tensor,
        ratio_logits: torch.Tensor,
        source_idx: int,
        dir_id: int,
        ratio_id: int
    ) -> float:
        """Compute log probability of action."""
        # Handle -inf in logits
        source_logits = torch.where(
            torch.isinf(source_logits),
            torch.full_like(source_logits, -100.0),
            source_logits
        )
        dir_logits = torch.where(
            torch.isinf(dir_logits),
            torch.full_like(dir_logits, -100.0),
            dir_logits
        )

        # Log probabilities
        source_log_probs = F.log_softmax(source_logits, dim=-1)
        dir_log_probs = F.log_softmax(dir_logits, dim=-1)
        ratio_log_probs = F.log_softmax(ratio_logits, dim=-1)

        # Total log prob
        log_prob = (
            source_log_probs[0, source_idx].item() +
            dir_log_probs[0, dir_id].item() +
            ratio_log_probs[0, ratio_id].item()
        )

        return log_prob

    def train_step(self, batch_size: int = 32, seq_len: int = 16) -> Dict[str, float]:
        """
        Single training step with V-trace.

        Args:
            batch_size: Batch size
            seq_len: Sequence length for training

        Returns:
            Dictionary of loss values
        """
        if len(self.trajectory_buffer) < 1:
            return {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'entropy': 0.0}

        self.model.train()

        # Sample batch from trajectory buffer
        batch = self.trajectory_buffer.sample_batch(batch_size, seq_len)

        # Move to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        B, T = batch['grids'].shape[:2]

        # Initialize history latent
        history_latent = self.model.get_initial_latent(B, self.device)

        # Lists to collect per-timestep outputs
        all_source_logits = []
        all_dir_logits = []
        all_ratio_logits = []
        all_values = []

        # Process sequence
        for t in range(T):
            grid_t = batch['grids'][:, t]  # [B, C, H, W]
            globals_t = batch['globals'][:, t]  # [B, 4]
            legal_mask_t = batch['legal_masks'][:, t]  # [B, 1, H, W]
            tick_bucket_t = batch['tick_buckets'][:, t]  # [B]

            # Forward pass
            tokens = self.model.input_embedding(
                grid_t, globals_t, history_latent, tick_bucket_t
            )
            y, z, _, _ = self.model.backbone(tokens)

            # Policy (training mode with teacher forcing on source)
            source_idx_t = batch['source_idxs'][:, t]
            policy_out = self.model.policy_head.forward_training(
                y, legal_mask_t, source_idx_t, grid_t
            )

            # Value
            value = self.model.value_head(y + z, globals_t)

            all_source_logits.append(policy_out['source_logits'])
            all_dir_logits.append(policy_out['dir_logits'])
            all_ratio_logits.append(policy_out['ratio_logits'])
            all_values.append(value)

            # Update history latent
            history_latent = self.model.update_history_latent(
                history_latent, grid_t, globals_t
            )

        # Stack outputs [B, T, ...]
        source_logits = torch.stack(all_source_logits, dim=1)
        dir_logits = torch.stack(all_dir_logits, dim=1)
        ratio_logits = torch.stack(all_ratio_logits, dim=1)
        values = torch.stack(all_values, dim=1).squeeze(-1)  # [B, T]

        # Compute target log probs
        target_log_probs = self._compute_batch_log_probs(
            source_logits, dir_logits, ratio_logits,
            batch['source_idxs'], batch['dir_ids'], batch['ratio_ids']
        )

        # Behavior log probs (from rollout)
        behavior_log_probs = batch['behavior_log_probs']

        # Compute V-trace
        rho, c = compute_importance_weights(
            target_log_probs, behavior_log_probs,
            self.rho_bar, self.c_bar
        )

        vs, advantages = vtrace_targets(
            values.detach(),
            batch['rewards'],
            batch['dones'].float(),
            rho, c,
            self.gamma,
            batch['bootstrap_values']
        )

        # Mask for valid timesteps
        mask = batch['masks'].float()

        # === Policy Loss ===
        # Policy gradient: -A * log_prob
        policy_loss = -(advantages.detach() * target_log_probs * mask).sum() / mask.sum().clamp(min=1)

        # === Value Loss ===
        value_loss = F.mse_loss(values * mask, vs.detach() * mask, reduction='sum') / mask.sum().clamp(min=1)

        # === Entropy Bonus ===
        entropy = self._compute_entropy(source_logits, dir_logits, ratio_logits, mask)

        # === KL Penalty (optional) ===
        kl_loss = torch.tensor(0.0, device=self.device)
        if self.kl_weight > 0 and self.reference_model is not None:
            kl_loss = self._compute_kl_penalty(
                batch, source_logits, dir_logits, ratio_logits, mask
            )

        # === Total Loss ===
        total_loss = (
            policy_loss +
            self.value_loss_weight * value_loss -
            self.entropy_weight * entropy +
            self.kl_weight * kl_loss
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        # EMA update
        self._update_ema()

        self.global_step += 1

        return {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'entropy': entropy.item(),
            'kl': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'mean_rho': rho.mean().item(),
            'mean_advantage': advantages.mean().item()
        }

    def _compute_batch_log_probs(
        self,
        source_logits: torch.Tensor,
        dir_logits: torch.Tensor,
        ratio_logits: torch.Tensor,
        source_idxs: torch.Tensor,
        dir_ids: torch.Tensor,
        ratio_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for batch of actions.

        Args:
            source_logits: [B, T, 625]
            dir_logits: [B, T, 4]
            ratio_logits: [B, T, 2]
            source_idxs: [B, T]
            dir_ids: [B, T]
            ratio_ids: [B, T]

        Returns:
            log_probs: [B, T]
        """
        B, T = source_idxs.shape

        # Handle -inf
        source_logits = torch.where(
            torch.isinf(source_logits),
            torch.full_like(source_logits, -100.0),
            source_logits
        )
        dir_logits = torch.where(
            torch.isinf(dir_logits),
            torch.full_like(dir_logits, -100.0),
            dir_logits
        )

        # Log softmax
        source_log_probs = F.log_softmax(source_logits, dim=-1)  # [B, T, 625]
        dir_log_probs = F.log_softmax(dir_logits, dim=-1)  # [B, T, 4]
        ratio_log_probs = F.log_softmax(ratio_logits, dim=-1)  # [B, T, 2]

        # Gather
        source_lp = source_log_probs.gather(-1, source_idxs.unsqueeze(-1)).squeeze(-1)
        dir_lp = dir_log_probs.gather(-1, dir_ids.unsqueeze(-1)).squeeze(-1)
        ratio_lp = ratio_log_probs.gather(-1, ratio_ids.unsqueeze(-1)).squeeze(-1)

        return source_lp + dir_lp + ratio_lp

    def _compute_entropy(
        self,
        source_logits: torch.Tensor,
        dir_logits: torch.Tensor,
        ratio_logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute entropy bonus.

        Args:
            source_logits: [B, T, 625]
            dir_logits: [B, T, 4]
            ratio_logits: [B, T, 2]
            mask: [B, T]

        Returns:
            Average entropy
        """
        # Handle -inf
        source_logits = torch.where(
            torch.isinf(source_logits),
            torch.full_like(source_logits, -100.0),
            source_logits
        )
        dir_logits = torch.where(
            torch.isinf(dir_logits),
            torch.full_like(dir_logits, -100.0),
            dir_logits
        )

        # Entropy: -sum(p * log(p))
        def entropy_from_logits(logits):
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            return -(probs * log_probs).sum(dim=-1)

        source_entropy = entropy_from_logits(source_logits)
        dir_entropy = entropy_from_logits(dir_logits)
        ratio_entropy = entropy_from_logits(ratio_logits)

        total_entropy = source_entropy + dir_entropy + ratio_entropy
        return (total_entropy * mask).sum() / mask.sum().clamp(min=1)

    def _compute_kl_penalty(
        self,
        batch: Dict[str, torch.Tensor],
        source_logits: torch.Tensor,
        dir_logits: torch.Tensor,
        ratio_logits: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from reference policy.

        This helps prevent the policy from diverging too far from
        the supervised pre-trained policy.
        """
        if self.reference_model is None:
            return torch.tensor(0.0, device=self.device)

        B, T = batch['grids'].shape[:2]

        # Get reference policy outputs
        with torch.no_grad():
            history_latent = self.reference_model.get_initial_latent(B, self.device)
            ref_source_logits_list = []
            ref_dir_logits_list = []
            ref_ratio_logits_list = []

            for t in range(T):
                grid_t = batch['grids'][:, t]
                globals_t = batch['globals'][:, t]
                legal_mask_t = batch['legal_masks'][:, t]
                tick_bucket_t = batch['tick_buckets'][:, t]
                source_idx_t = batch['source_idxs'][:, t]

                tokens = self.reference_model.input_embedding(
                    grid_t, globals_t, history_latent, tick_bucket_t
                )
                y, _, _, _ = self.reference_model.backbone(tokens)
                policy_out = self.reference_model.policy_head.forward_training(
                    y, legal_mask_t, source_idx_t, grid_t
                )

                ref_source_logits_list.append(policy_out['source_logits'])
                ref_dir_logits_list.append(policy_out['dir_logits'])
                ref_ratio_logits_list.append(policy_out['ratio_logits'])

                history_latent = self.reference_model.update_history_latent(
                    history_latent, grid_t, globals_t
                )

            ref_source_logits = torch.stack(ref_source_logits_list, dim=1)
            ref_dir_logits = torch.stack(ref_dir_logits_list, dim=1)
            ref_ratio_logits = torch.stack(ref_ratio_logits_list, dim=1)

        # Compute KL divergence: sum(p_ref * (log_p_ref - log_p))
        def kl_from_logits(logits, ref_logits):
            # Handle -inf
            logits = torch.where(torch.isinf(logits), torch.full_like(logits, -100.0), logits)
            ref_logits = torch.where(torch.isinf(ref_logits), torch.full_like(ref_logits, -100.0), ref_logits)

            p_ref = F.softmax(ref_logits, dim=-1)
            log_p = F.log_softmax(logits, dim=-1)
            log_p_ref = F.log_softmax(ref_logits, dim=-1)
            return (p_ref * (log_p_ref - log_p)).sum(dim=-1)

        kl_source = kl_from_logits(source_logits, ref_source_logits)
        kl_dir = kl_from_logits(dir_logits, ref_dir_logits)
        kl_ratio = kl_from_logits(ratio_logits, ref_ratio_logits)

        total_kl = kl_source + kl_dir + kl_ratio
        return (total_kl * mask).sum() / mask.sum().clamp(min=1)

    def train_from_buffer(
        self,
        num_steps: int,
        batch_size: int = 32,
        seq_len: int = 16,
        log_interval: int = 10
    ) -> List[Dict[str, float]]:
        """
        Train for multiple steps from trajectory buffer.

        Args:
            num_steps: Number of training steps
            batch_size: Batch size
            seq_len: Sequence length
            log_interval: How often to log

        Returns:
            List of loss dictionaries
        """
        all_losses = []

        for step in range(num_steps):
            losses = self.train_step(batch_size, seq_len)
            all_losses.append(losses)

            if (step + 1) % log_interval == 0:
                avg_losses = {
                    k: sum(d[k] for d in all_losses[-log_interval:]) / log_interval
                    for k in losses.keys()
                }
                print(f"Step {self.global_step}: "
                      f"total={avg_losses['total']:.4f}, "
                      f"policy={avg_losses['policy']:.4f}, "
                      f"value={avg_losses['value']:.4f}, "
                      f"entropy={avg_losses['entropy']:.4f}")

        return all_losses

    def save_checkpoint(
        self,
        path: str,
        extra_state: Optional[Dict[str, Any]] = None
    ):
        """Save training checkpoint."""
        state = {
            'model': self.model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }

        if self.reference_model is not None:
            state['reference_model'] = self.reference_model.state_dict()

        if extra_state:
            state.update(extra_state)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load training checkpoint."""
        state = torch.load(path, map_location=self.device)

        self.model.load_state_dict(state['model'])
        self.target_model.load_state_dict(state['target_model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.scheduler.load_state_dict(state['scheduler'])
        self.global_step = state['global_step']
        self.best_loss = state['best_loss']

        if 'reference_model' in state and self.reference_model is not None:
            self.reference_model.load_state_dict(state['reference_model'])

        print(f"Loaded checkpoint from {path} (step {self.global_step})")

        extra_keys = set(state.keys()) - {
            'model', 'target_model', 'optimizer', 'scheduler',
            'global_step', 'best_loss', 'reference_model'
        }
        return {k: state[k] for k in extra_keys}

    def get_target_model(self) -> GeneralsModel:
        """Get the EMA target model for evaluation."""
        return self.target_model
