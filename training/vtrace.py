"""
V-trace off-policy correction for RL training.

Reference: IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures
https://arxiv.org/abs/1802.01561

V-trace corrects for the discrepancy between the behavior policy (used to collect data)
and the target policy (being learned), enabling efficient off-policy learning.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def compute_importance_weights(
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    rho_bar: float = 1.0,
    c_bar: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute clipped importance sampling weights for V-trace.

    Args:
        target_log_probs: Log probabilities under target policy [B, T]
        behavior_log_probs: Log probabilities under behavior policy [B, T]
        rho_bar: Clipping threshold for rho (default 1.0)
        c_bar: Clipping threshold for c (default 1.0)

    Returns:
        rho: Clipped importance weights for value targets [B, T]
        c: Clipped importance weights for trace cutting [B, T]
    """
    # Importance weight: pi(a|s) / mu(a|s) = exp(log_pi - log_mu)
    log_rho = target_log_probs - behavior_log_probs
    rho = torch.exp(log_rho)

    # Clip rho and c
    rho_clipped = torch.clamp(rho, max=rho_bar)
    c_clipped = torch.clamp(rho, max=c_bar)

    return rho_clipped, c_clipped


def vtrace_targets(
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    rho: torch.Tensor,
    c: torch.Tensor,
    gamma: float = 0.99,
    bootstrap_value: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute V-trace targets and advantages.

    V-trace target:
        v_s = V(s) + sum_{t=s}^{s+n-1} gamma^{t-s} (prod_{i=s}^{t-1} c_i) * delta_t
    Where:
        delta_t = rho_t * (r_t + gamma * V(s_{t+1}) - V(s_t))

    Args:
        values: Value estimates [B, T]
        rewards: Rewards [B, T]
        dones: Episode termination flags [B, T]
        rho: Clipped importance weights [B, T]
        c: Trace cutting coefficients [B, T]
        gamma: Discount factor
        bootstrap_value: Value estimate for state after last timestep [B]

    Returns:
        vs: V-trace value targets [B, T]
        advantages: Policy gradient advantages [B, T]
    """
    B, T = values.shape
    device = values.device

    # Handle bootstrap value
    if bootstrap_value is None:
        bootstrap_value = torch.zeros(B, device=device)

    # Append bootstrap value for indexing convenience
    values_plus_1 = torch.cat([values, bootstrap_value.unsqueeze(1)], dim=1)  # [B, T+1]

    # Compute TD errors: delta_t = rho_t * (r_t + gamma * V(s_{t+1}) - V(s_t))
    # Account for episode boundaries
    not_done = 1.0 - dones.float()
    next_values = values_plus_1[:, 1:] * not_done  # Zero out next value if done
    td_errors = rho * (rewards + gamma * next_values - values)  # [B, T]

    # Compute V-trace targets using reverse accumulation
    # v_s - V(s) = delta_s + gamma * c_s * (v_{s+1} - V(s+1))
    vs_minus_v = torch.zeros_like(values)  # [B, T]

    # Backward pass to compute cumulative trace
    for t in reversed(range(T)):
        if t == T - 1:
            vs_minus_v[:, t] = td_errors[:, t]
        else:
            vs_minus_v[:, t] = td_errors[:, t] + gamma * c[:, t] * not_done[:, t] * vs_minus_v[:, t + 1]

    # V-trace targets
    vs = values + vs_minus_v

    # Advantages for policy gradient
    # A_t = rho_t * (r_t + gamma * v_{t+1} - V(s_t))
    vs_plus_1 = torch.cat([vs[:, 1:], bootstrap_value.unsqueeze(1)], dim=1)
    advantages = rho * (rewards + gamma * not_done * vs_plus_1 - values)

    return vs, advantages


def compute_vtrace_loss(
    target_log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    values: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
    value_loss_weight: float = 0.5,
    entropy_weight: float = 0.01,
    entropy: Optional[torch.Tensor] = None,
    bootstrap_value: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """
    Compute complete V-trace loss for actor-critic training.

    Args:
        target_log_probs: Log probs under current policy [B, T]
        behavior_log_probs: Log probs under behavior policy [B, T]
        values: Value predictions [B, T]
        rewards: Rewards [B, T]
        dones: Done flags [B, T]
        gamma: Discount factor
        rho_bar: Rho clipping threshold
        c_bar: C clipping threshold
        value_loss_weight: Weight for value loss
        entropy_weight: Weight for entropy bonus
        entropy: Policy entropy [B, T] (optional)
        bootstrap_value: Bootstrap value [B] (optional)

    Returns:
        total_loss: Combined loss scalar
        metrics: Dictionary of loss components
    """
    # Compute importance weights
    rho, c = compute_importance_weights(
        target_log_probs, behavior_log_probs, rho_bar, c_bar
    )

    # Compute V-trace targets
    vs, advantages = vtrace_targets(
        values, rewards, dones, rho, c, gamma, bootstrap_value
    )

    # Policy gradient loss: -advantage * log_prob
    # Note: advantages already include rho, so we use unweighted log_probs
    policy_loss = -(advantages.detach() * target_log_probs).mean()

    # Value loss: MSE between predicted values and V-trace targets
    value_loss = F.mse_loss(values, vs.detach())

    # Entropy bonus (encourages exploration)
    entropy_loss = torch.tensor(0.0, device=values.device)
    if entropy is not None:
        entropy_loss = -entropy.mean()

    # Total loss
    total_loss = (
        policy_loss +
        value_loss_weight * value_loss +
        entropy_weight * entropy_loss
    )

    metrics = {
        'total': total_loss,
        'policy': policy_loss,
        'value': value_loss,
        'entropy': -entropy_loss if entropy is not None else torch.tensor(0.0),
        'mean_rho': rho.mean(),
        'mean_advantage': advantages.mean(),
        'vs_mean': vs.mean()
    }

    return total_loss, metrics


class VTraceFromLogits:
    """
    V-trace computation helper for models that output logits.

    Handles the full pipeline from logits to loss computation.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        value_loss_weight: float = 0.5,
        entropy_weight: float = 0.01
    ):
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight

    def __call__(
        self,
        target_logits: torch.Tensor,
        behavior_logits: torch.Tensor,
        actions: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute V-trace loss from logits.

        Args:
            target_logits: Logits from current policy [B, T, num_actions]
            behavior_logits: Logits from behavior policy [B, T, num_actions]
            actions: Actions taken [B, T]
            values: Value predictions [B, T]
            rewards: Rewards [B, T]
            dones: Done flags [B, T]
            bootstrap_value: Bootstrap value [B]

        Returns:
            loss: Total loss
            metrics: Loss components
        """
        B, T, num_actions = target_logits.shape

        # Compute log probabilities
        target_log_probs_all = F.log_softmax(target_logits, dim=-1)
        behavior_log_probs_all = F.log_softmax(behavior_logits, dim=-1)

        # Gather log probs for taken actions
        actions_expanded = actions.unsqueeze(-1)  # [B, T, 1]
        target_log_probs = target_log_probs_all.gather(-1, actions_expanded).squeeze(-1)  # [B, T]
        behavior_log_probs = behavior_log_probs_all.gather(-1, actions_expanded).squeeze(-1)  # [B, T]

        # Compute entropy
        probs = F.softmax(target_logits, dim=-1)
        entropy = -(probs * target_log_probs_all).sum(dim=-1)  # [B, T]

        return compute_vtrace_loss(
            target_log_probs=target_log_probs,
            behavior_log_probs=behavior_log_probs,
            values=values,
            rewards=rewards,
            dones=dones,
            gamma=self.gamma,
            rho_bar=self.rho_bar,
            c_bar=self.c_bar,
            value_loss_weight=self.value_loss_weight,
            entropy_weight=self.entropy_weight,
            entropy=entropy,
            bootstrap_value=bootstrap_value
        )


def compute_log_prob_from_policy_heads(
    move_logits: torch.Tensor,
    source_logits: torch.Tensor,
    dir_logits: torch.Tensor,
    ratio_logits: torch.Tensor,
    move_id: torch.Tensor,
    source_idx: torch.Tensor,
    dir_id: torch.Tensor,
    ratio_id: torch.Tensor
) -> torch.Tensor:
    """
    Compute log probability for hierarchical action space.

    For generals.io action format: (move_id, source_idx, dir_id, ratio_id)

    Log prob factorization:
        if move_id == 0 (pass):
            log_prob = log P(move=0)
        else:
            log_prob = log P(move=1) + log P(src) + log P(dir|src) + log P(ratio)

    Args:
        move_logits: [B, 2] logits for move/pass decision
        source_logits: [B, 625] logits for source tile selection
        dir_logits: [B, 4] logits for direction
        ratio_logits: [B, 2] logits for ratio
        move_id: [B] move decisions (0=pass, 1=move)
        source_idx: [B] source tile indices
        dir_id: [B] direction indices
        ratio_id: [B] ratio indices

    Returns:
        log_prob: [B] total log probability
    """
    B = move_logits.shape[0]
    device = move_logits.device

    # Replace -inf with large negative value for numerical stability
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

    # Compute log softmax for each head
    move_log_probs = F.log_softmax(move_logits, dim=-1)
    source_log_probs = F.log_softmax(source_logits, dim=-1)
    dir_log_probs = F.log_softmax(dir_logits, dim=-1)
    ratio_log_probs = F.log_softmax(ratio_logits, dim=-1)

    # Gather log probs for selected actions
    move_log_prob = move_log_probs.gather(-1, move_id.unsqueeze(-1)).squeeze(-1)  # [B]
    source_log_prob = source_log_probs.gather(-1, source_idx.unsqueeze(-1)).squeeze(-1)  # [B]
    dir_log_prob = dir_log_probs.gather(-1, dir_id.unsqueeze(-1)).squeeze(-1)  # [B]
    ratio_log_prob = ratio_log_probs.gather(-1, ratio_id.unsqueeze(-1)).squeeze(-1)  # [B]

    # Compute total log prob based on move decision
    # If pass (move_id=0): only move log prob
    # If move (move_id=1): sum of all log probs
    is_move = (move_id == 1).float()

    log_prob = move_log_prob + is_move * (source_log_prob + dir_log_prob + ratio_log_prob)

    return log_prob
