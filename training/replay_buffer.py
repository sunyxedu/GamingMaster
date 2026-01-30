"""
Replay Buffer for storing and sampling training data.
"""

import random
import numpy as np
import torch
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class GameRecord:
    """
    Record of a single game for training.

    Attributes:
        states: List of game states (grid, globals, etc.)
        actions: List of actions taken
        policies: List of policy distributions (action -> probability)
        outcome: Game outcome (1 for win, -1 for loss, 0 for draw)
        player_id: Which player's perspective
    """
    states: List[Dict[str, torch.Tensor]]
    actions: List[Tuple[int, int, int]]  # (source_idx, dir_id, ratio_id)
    policies: List[Dict[Tuple[int, int, int], float]]
    outcome: float
    player_id: int

    # Backwards compatibility alias
    @property
    def mcts_policies(self):
        return self.policies


class ReplayBuffer:
    """
    Replay buffer for storing and sampling training data.

    Stores game records and samples random batches for training.
    Uses a circular buffer with fixed maximum size.

    Note: For RL training, use TrajectoryBuffer in rl_trainer.py instead.

    Args:
        max_size: Maximum number of game records to store
        batch_size: Default batch size for sampling
    """

    def __init__(
        self,
        max_size: int = 100000,
        batch_size: int = 256
    ):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer: deque = deque(maxlen=max_size)

    def add(self, record: GameRecord):
        """Add a game record to the buffer."""
        self.buffer.append(record)

    def add_game(
        self,
        states: List[Dict[str, torch.Tensor]],
        actions: List[Tuple[int, int, int]],
        policies: List[Dict[Tuple[int, int, int], float]],
        outcome: float,
        player_id: int
    ):
        """
        Add a game as individual components.

        Convenience method that creates GameRecord internally.
        """
        record = GameRecord(
            states=states,
            actions=actions,
            policies=policies,
            outcome=outcome,
            player_id=player_id
        )
        self.add(record)

    def sample(
        self,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of training examples.

        Args:
            batch_size: Number of samples (default: self.batch_size)

        Returns:
            Dictionary with batched tensors
        """
        batch_size = batch_size or self.batch_size

        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Sample random game records
        records = random.sample(list(self.buffer), batch_size)

        # For now, sample one random timestep from each game
        # Could also implement sequence sampling
        batch = self._create_batch(records)

        return batch

    def _create_batch(
        self,
        records: List[GameRecord]
    ) -> Dict[str, torch.Tensor]:
        """
        Create batch tensors from game records.

        Samples one random timestep from each game.
        """
        batch_grids = []
        batch_globals = []
        batch_legal_masks = []
        batch_ticks = []
        batch_source_idx = []
        batch_dir_id = []
        batch_ratio_id = []
        batch_values = []
        batch_source_probs = []  # Policy distribution for source

        for record in records:
            # Sample random timestep
            t = random.randint(0, len(record.states) - 1)

            state = record.states[t]
            action = record.actions[t]
            policy = record.policies[t]

            batch_grids.append(state['grid'])
            batch_globals.append(state['globals'])
            batch_legal_masks.append(state['legal_mask'])
            batch_ticks.append(state['tick'] % 50 // 2)

            # Action targets
            source_idx, dir_id, ratio_id = action
            batch_source_idx.append(source_idx)
            batch_dir_id.append(dir_id)
            batch_ratio_id.append(ratio_id)

            # Value target (same for all timesteps in a game)
            batch_values.append(record.outcome)

            # Policy target (convert to dense tensor)
            source_probs = torch.zeros(625)
            for (src, _, _), prob in policy.items():
                source_probs[src] += prob
            source_probs = source_probs / source_probs.sum().clamp(min=1e-8)
            batch_source_probs.append(source_probs)

        # Stack into tensors
        return {
            'grid': torch.stack(batch_grids),
            'globals': torch.stack(batch_globals),
            'legal_mask': torch.stack(batch_legal_masks),
            'tick_bucket': torch.tensor(batch_ticks, dtype=torch.long),
            'source_idx': torch.tensor(batch_source_idx, dtype=torch.long),
            'dir_id': torch.tensor(batch_dir_id, dtype=torch.long),
            'ratio_id': torch.tensor(batch_ratio_id, dtype=torch.long),
            'value': torch.tensor(batch_values, dtype=torch.float32),
            'source_probs': torch.stack(batch_source_probs)
        }

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int
    ) -> Dict[str, torch.Tensor]:
        """
        Sample sequences of consecutive timesteps.

        Args:
            batch_size: Number of sequences
            seq_len: Length of each sequence

        Returns:
            Dictionary with sequence tensors [B, T, ...]
        """
        # Filter games with enough timesteps
        valid_records = [r for r in self.buffer if len(r.states) >= seq_len]

        if len(valid_records) < batch_size:
            batch_size = len(valid_records)

        records = random.sample(valid_records, batch_size)

        # Sample starting positions
        batch = {
            'grid_seq': [],
            'globals_seq': [],
            'legal_mask_seq': [],
            'tick_bucket_seq': [],
            'source_idx': [],
            'dir_id': [],
            'ratio_id': [],
            'value': [],
            'action_mask': []
        }

        for record in records:
            max_start = len(record.states) - seq_len
            start_t = random.randint(0, max_start)

            grids = []
            globals_list = []
            legal_masks = []
            ticks = []
            source_idxs = []
            dir_ids = []
            ratio_ids = []
            action_masks = []

            for t in range(start_t, start_t + seq_len):
                state = record.states[t]
                action = record.actions[t]

                grids.append(state['grid'])
                globals_list.append(state['globals'])
                legal_masks.append(state['legal_mask'])
                ticks.append(state['tick'] % 50 // 2)

                source_idx, dir_id, ratio_id = action
                source_idxs.append(source_idx)
                dir_ids.append(dir_id)
                ratio_ids.append(ratio_id)

                # Action mask: True if action was taken
                action_masks.append(source_idx >= 0)

            batch['grid_seq'].append(torch.stack(grids))
            batch['globals_seq'].append(torch.stack(globals_list))
            batch['legal_mask_seq'].append(torch.stack(legal_masks))
            batch['tick_bucket_seq'].append(torch.tensor(ticks))
            batch['source_idx'].append(torch.tensor(source_idxs))
            batch['dir_id'].append(torch.tensor(dir_ids))
            batch['ratio_id'].append(torch.tensor(ratio_ids))
            batch['value'].append(record.outcome)
            batch['action_mask'].append(torch.tensor(action_masks))

        # Stack
        return {
            'grid_seq': torch.stack(batch['grid_seq']),
            'globals_seq': torch.stack(batch['globals_seq']),
            'legal_mask_seq': torch.stack(batch['legal_mask_seq']),
            'tick_bucket_seq': torch.stack(batch['tick_bucket_seq']),
            'source_idx': torch.stack(batch['source_idx']),
            'dir_id': torch.stack(batch['dir_id']),
            'ratio_id': torch.stack(batch['ratio_id']),
            'value': torch.tensor(batch['value']),
            'action_mask': torch.stack(batch['action_mask'])
        }

    def add_single(self, data: Dict[str, torch.Tensor]):
        """
        Add a single timestep data directly (simpler interface).

        Args:
            data: Dictionary with grid, globals, legal_mask, tick_bucket,
                  source_idx, dir_id, ratio_id, value
        """
        self.buffer.append(data)

    def sample_simple(
        self,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from simple single-timestep buffer.

        Args:
            batch_size: Number of samples

        Returns:
            Batched dictionary
        """
        batch_size = batch_size or self.batch_size
        batch_size = min(batch_size, len(self.buffer))

        samples = random.sample(list(self.buffer), batch_size)

        # Check if buffer contains dicts (simple mode) or GameRecords
        if isinstance(samples[0], dict):
            # Stack all fields
            batch = {}
            for key in samples[0].keys():
                vals = [s[key] for s in samples]
                if isinstance(vals[0], torch.Tensor):
                    batch[key] = torch.stack(vals)
                else:
                    batch[key] = torch.tensor(vals)
            return batch
        else:
            # Old GameRecord mode
            return self._create_batch(samples)

    def __len__(self) -> int:
        return len(self.buffer)

    def clear(self):
        """Clear all records."""
        self.buffer.clear()


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with prioritized sampling.

    Samples based on TD error or loss magnitude.
    """

    def __init__(
        self,
        max_size: int = 100000,
        batch_size: int = 256,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        super().__init__(max_size, batch_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.priorities: deque = deque(maxlen=max_size)
        self.max_priority = 1.0

    def add(self, record: GameRecord):
        """Add record with max priority."""
        super().add(record)
        self.priorities.append(self.max_priority)

    def sample(
        self,
        batch_size: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample with priorities.

        Returns:
            batch: Sampled batch
            indices: Indices of sampled records
            weights: Importance sampling weights
        """
        batch_size = batch_size or self.batch_size
        n = len(self.buffer)

        if n < batch_size:
            batch_size = n

        # Compute sampling probabilities
        priorities = np.array(list(self.priorities))
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # Sample indices
        indices = np.random.choice(n, batch_size, p=probs, replace=False)

        # Compute importance weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Create batch
        records = [self.buffer[i] for i in indices]
        batch = self._create_batch(records)

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled records."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)
