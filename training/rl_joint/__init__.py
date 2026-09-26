"""Stage-2 RL algorithm library: mixed-game PPO on the whole MultiGameNetwork.

Pure PyTorch — no environments, no trainer, no model code.  Every piece works
on ONE game's tensors plus a validity mask, or keeps per-game state with
``state_dict`` / ``load_state_dict``; the joint trainer composes them:

per iteration (weights and MoE routing bias frozen through rollouts + epochs)
    quotas = sched.quotas(tokens)                  learner tokens per game
    ... actors roll out; store behaviour log-probs μ and values ...
    V      = popart.denormalize(g, head_out)       reward units
    A, G   = gae(r, V, terminated, gamma=γ_g, lam=λ_g, truncated=..., bootstrap=...)
    popart.update({g: (G, mask)}, mirrors=[ema])   Pop: head rescaled, outputs kept
    advnorm.update({g: (A, mask)}); A = advnorm.normalize(g, A)
    keep   = top_q_mask(A, mask, q=0.25)           optional
    π_old  = learner recompute (same weights, frozen routing), before epoch 1
    per minibatch, per game g:
        pol = decoupled_ppo_loss(logπ, logπ_old, logμ, A, mask, policy_mask=keep)
        val = value_loss(v, popart.normalize(g, G), mask)  (hl_gauss_value_loss
              for HL-Gauss heads)
        ent = entropy_loss(H, mask, coef=entcoef[g])
        kl  = kl_penalty(logπ, logπ_ref, mask, beta=klbeta[g])
    total = combine_game_losses(...)  (+ the IL replay stream)
    after the epochs: entcoef.update(H_g), klbeta.update(KL_g), sched.record(n_g)

Modules
    losses        decoupled_ppo_loss, value_loss, hl_gauss_value_loss,
                  entropy_loss, kl_penalty, categorical_kl, combine_game_losses
    gae           gae (per-game γ/λ, terminated vs truncated bootstrap, padding)
    normalizers   RunningMoments, AdvantageNormalizer (per-game EMA mean/std)
    popart        PopArt (multi-task, output-preserving; HL-Gauss passthrough)
    controllers   AdaptiveEntropyCoef, AdaptiveKLBeta
    filters       top_q_mask
    quota         TokenQuotaScheduler, project_shares, largest_remainder,
                  ShareProvider (bandit hook), LearningProgressBandit
    stats         masked_mean / masked_sum, per_game_moments (one all-reduce)
    diagnostics   explained_variance, expert_load_entropy

Multi-GPU: statistics updates (normaliser, PopArt) all-reduce per-game
moments of every configured game in one collective — call them on every
rank.  Controllers and the scheduler are deterministic given their inputs:
feed them all-reduced measurements and they stay identical across ranks.
"""

from training.rl_joint.controllers import AdaptiveEntropyCoef, AdaptiveKLBeta
from training.rl_joint.diagnostics import expert_load_entropy, explained_variance
from training.rl_joint.filters import top_q_mask
from training.rl_joint.gae import gae
from training.rl_joint.losses import (
    categorical_kl,
    combine_game_losses,
    decoupled_ppo_loss,
    entropy_loss,
    hl_gauss_value_loss,
    kl_penalty,
    value_loss,
)
from training.rl_joint.normalizers import AdvantageNormalizer, RunningMoments
from training.rl_joint.popart import (
    PASSTHROUGH,
    POPART,
    PopArt,
    PopArtTransform,
    final_linear,
    popart_rescale_,
)
from training.rl_joint.quota import (
    LearningProgressBandit,
    ShareProvider,
    TokenQuotaScheduler,
    largest_remainder,
    project_shares,
)
from training.rl_joint.stats import masked_mean, masked_sum, per_game_moments

__all__ = [
    "AdaptiveEntropyCoef", "AdaptiveKLBeta",
    "expert_load_entropy", "explained_variance",
    "top_q_mask",
    "gae",
    "categorical_kl", "combine_game_losses", "decoupled_ppo_loss", "entropy_loss",
    "hl_gauss_value_loss", "kl_penalty", "value_loss",
    "AdvantageNormalizer", "RunningMoments",
    "PASSTHROUGH", "POPART", "PopArt", "PopArtTransform", "final_linear", "popart_rescale_",
    "LearningProgressBandit", "ShareProvider", "TokenQuotaScheduler", "largest_remainder",
    "project_shares",
    "masked_mean", "masked_sum", "per_game_moments",
]
