"""
Reinforcement-learning utilities.

- GAE (Generalized Advantage Estimation)
- RolloutBuffer (trajectory storage for PPO)
- KL-divergence between policies (regularisation)
- Encoder freeze / unfreeze helpers
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# GAE
# ======================================================================

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (Schulman et al., 2016).

    Args:
        rewards      [T]  rewards collected during rollout
        values       [T]  value-head predictions (detached)
        dones        [T]  bool — True if episode ended at this step
        gamma             discount factor
        lam               GAE lambda (bias–variance trade-off)
        last_value        V(s_{T+1}) bootstrap; 0 if final step is terminal

    Returns:
        advantages   [T]
        returns      [T]  (advantages + values)
    """
    T = len(rewards)
    advantages = torch.zeros(T)
    last_gae = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = last_value
        else:
            next_value = values[t + 1]

        # If episode ended at step t, no bootstrapping
        if dones[t]:
            next_value = 0.0
            last_gae = 0.0

        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# ======================================================================
# Rollout buffer
# ======================================================================

class RolloutBuffer:
    """Stores a single trajectory for one player during self-play.

    Usage::

        buf = RolloutBuffer()
        for each step:
            buf.add(obs, action, log_prob, value, reward, done)
        buf.compute_returns_and_advantages(last_value=…)
        batch = buf.to_tensor_batch(device)
    """

    def __init__(self) -> None:
        self.grids: List[torch.Tensor] = []          # [C, H, W]
        self.globals_: List[torch.Tensor] = []        # [G]
        self.legal_masks: List[torch.Tensor] = []    # [1, H, W]
        self.source_idxs: List[int] = []
        self.directions: List[int] = []
        self.splits: List[int] = []
        self.passes: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.valid_directions: List[List[bool]] = []  # [4] per step
        # Capture-feasibility supervision (optional obs keys): [4, H, W]
        self.capture_labels: List[Optional[torch.Tensor]] = []
        self.capture_masks: List[Optional[torch.Tensor]] = []
        # Direction-validity grids (optional obs key): [4, H, W] bool
        self.dir_masks: List[Optional[torch.Tensor]] = []

        self.advantages: Optional[torch.Tensor] = None
        self.returns: Optional[torch.Tensor] = None

    # ---- collection ----------------------------------------------------

    def add(
        self,
        obs: Dict[str, torch.Tensor],
        action: Dict[str, int],
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        valid_directions: Optional[List[bool]] = None,
    ) -> None:
        self.grids.append(obs["grid"])
        self.globals_.append(obs["globals"])
        self.legal_masks.append(obs["legal_mask"])
        self.source_idxs.append(action["source_idx"])
        self.directions.append(action["direction"])
        self.splits.append(action["split"])
        self.passes.append(action.get("passed", False))
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.valid_directions.append(valid_directions or [True, True, True, True])
        self.capture_labels.append(obs.get("capture_label"))
        self.capture_masks.append(obs.get("capture_mask"))
        self.dir_masks.append(obs.get("dir_mask"))

    # ---- advantage computation -----------------------------------------

    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        lam: float = 0.95,
        last_value: float = 0.0,
    ) -> None:
        self.advantages, self.returns = compute_gae(
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            values=torch.tensor(self.values, dtype=torch.float32),
            dones=torch.tensor(self.dones, dtype=torch.bool),
            gamma=gamma,
            lam=lam,
            last_value=last_value,
        )

    # ---- convert to tensors for PPO update -----------------------------

    def to_tensor_batch(self, device: torch.device) -> Dict:
        """Stack trajectory into a single-sample batch ``[1, T, …]``."""
        assert self.advantages is not None, "call compute_returns_and_advantages first"
        T = len(self.grids)
        batch = {
            "grid_seq": torch.stack(self.grids).unsqueeze(0).to(device),
            "globals_seq": torch.stack(self.globals_).unsqueeze(0).to(device),
            "legal_mask": torch.stack(self.legal_masks).unsqueeze(0).to(device),
            "actions": {
                "source_idx": torch.tensor(
                    self.source_idxs, dtype=torch.long, device=device,
                ).unsqueeze(0),
                "direction": torch.tensor(
                    self.directions, dtype=torch.long, device=device,
                ).unsqueeze(0),
                "split": torch.tensor(
                    self.splits, dtype=torch.long, device=device,
                ).unsqueeze(0),
                "passed": torch.tensor(
                    self.passes, dtype=torch.bool, device=device,
                ).unsqueeze(0),
            },
            "old_log_probs": torch.tensor(
                self.log_probs, dtype=torch.float32, device=device,
            ).unsqueeze(0),
            "advantages": self.advantages.to(device).unsqueeze(0),
            "returns": self.returns.to(device).unsqueeze(0),
            "valid_directions": torch.tensor(
                self.valid_directions, dtype=torch.bool, device=device,
            ).unsqueeze(0),  # [1, T, 4]
            "length": T,
        }
        # Capture-feasibility supervision — only when every step carried it
        # (older callers / tests build obs without these keys).
        if all(t is not None for t in self.capture_labels):
            batch["capture_label"] = (
                torch.stack(self.capture_labels).unsqueeze(0).to(device))
            batch["capture_mask"] = (
                torch.stack(self.capture_masks).unsqueeze(0).to(device))
        if all(t is not None for t in self.dir_masks):
            batch["dir_mask"] = (
                torch.stack(self.dir_masks).unsqueeze(0).to(device))  # [1,T,4,H,W]
        return batch

    # ---- housekeeping --------------------------------------------------

    def clear(self) -> None:
        self.__init__()  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self.rewards)


# ======================================================================
# KL divergence between policies (source-only, for regularisation)
# ======================================================================

def compute_policy_kl(
    current_source_logits: torch.Tensor,
    reference_source_logits: torch.Tensor,
    legal_mask: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL( π_ref ‖ π_current ) over the source-action distribution.

    Used as regularisation to prevent the RL policy from drifting too far
    from the IL-pretrained policy.  Only computed over the source head
    because it dominates the action space (625 positions vs 4+2).

    Args:
        current_source_logits    [B, T, N]
        reference_source_logits  [B, T, N]  (detached)
        legal_mask               [B, T, 1, H, W] or None
        mask                     [B, T] bool – valid timesteps

    Returns:
        scalar KL divergence (mean over valid positions)
    """
    B, T, N = current_source_logits.shape

    # Apply legal mask
    if legal_mask is not None:
        lm = legal_mask.reshape(B, T, N)
        cur_masked = current_source_logits.masked_fill(~lm, float("-inf"))
        ref_masked = reference_source_logits.masked_fill(~lm, float("-inf"))
    else:
        cur_masked = current_source_logits
        ref_masked = reference_source_logits

    # Handle fully-masked rows
    all_inf = cur_masked.isinf() & (cur_masked < 0)
    bad = all_inf.all(dim=-1)
    if bad.any():
        cur_masked = cur_masked.clone()
        ref_masked = ref_masked.clone()
        cur_masked[bad, 0] = 0.0
        ref_masked[bad, 0] = 0.0

    log_p_ref = F.log_softmax(ref_masked, dim=-1)
    log_p_cur = F.log_softmax(cur_masked, dim=-1)
    p_ref = torch.exp(log_p_ref)

    # KL = Σ p_ref * (log p_ref - log p_cur)
    kl = (p_ref * (log_p_ref - log_p_cur)).nan_to_num(0.0).sum(-1)  # [B, T]

    if mask is not None:
        n_valid = mask.float().sum().clamp(min=1)
        kl = (kl * mask.float()).sum() / n_valid
    else:
        kl = kl.mean()

    return kl


def compute_policy_kl_rl(
    current_source_logits: torch.Tensor,
    reference_source_logits: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """KL( π_ref ‖ π_current ) without legal mask (RL mode).

    Same as ``compute_policy_kl`` but operates on raw logits over all 625
    positions without masking.
    """
    log_p_ref = F.log_softmax(reference_source_logits, dim=-1)
    log_p_cur = F.log_softmax(current_source_logits, dim=-1)
    p_ref = torch.exp(log_p_ref)

    kl = (p_ref * (log_p_ref - log_p_cur)).nan_to_num(0.0).sum(-1)

    if mask is not None:
        n_valid = mask.float().sum().clamp(min=1)
        kl = (kl * mask.float()).sum() / n_valid
    else:
        kl = kl.mean()

    return kl


# ======================================================================
# Distributed gradient sync (manual DDP replacement)
# ======================================================================

def sync_gradients(model: nn.Module, world_size: int) -> None:
    """All-reduce gradients across ranks with a RANK-INVARIANT collective
    schedule.

    The naive per-parameter loop guarded by ``p.grad is not None`` posts a
    different number of collectives on a rank whose autograd left some
    parameter without a gradient. NCCL pairs collectives by order, so the
    streams shift: same-size pairs complete with silently corrupted data
    and the first size mismatch deadlocks until the 600 s watchdog kills
    the job. (Observed in production: one rank missing the two TRM q_head
    gradients for a single accumulation window desynced a 4-GPU phase-2
    run at NCCL work #648.)

    Schedule invariance: a fixed-size presence mask is MAX-reduced first,
    so every rank agrees which parameters participate this round. A rank
    missing a gradient for a participating parameter contributes zeros —
    and prints which ones, since that asymmetry is an autograd anomaly
    worth investigating. Parameters with no gradient on ANY rank are
    skipped by all ranks alike, preserving the previous semantics (heads
    untouched by the objective are never allreduced, never stepped).
    """
    named = list(model.named_parameters())
    device = named[0][1].device
    local = torch.tensor([p.grad is not None for _, p in named],
                         dtype=torch.float32, device=device)
    participating = local.clone()
    dist.all_reduce(participating, op=dist.ReduceOp.MAX)

    gap = (participating - local) > 0
    if bool(gap.any()):
        names = [named[i][0] for i in gap.nonzero(as_tuple=True)[0].tolist()]
        print(f"  [grad-sync] rank {dist.get_rank()}: no local gradient for "
              f"{len(names)} participating parameter(s), contributing zeros "
              f"— autograd anomaly, e.g. {names[:6]}", flush=True)

    for has_grad, (_, p) in zip(participating.tolist(), named):
        if not has_grad:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(world_size)


# ======================================================================
# Optimizer parameter groups
# ======================================================================

def optim_param_groups(model: nn.Module, weight_decay: float) -> List[Dict]:
    """Parameter groups excluding recurrence memory parameters from decay.

    AdamW's decoupled decay shrinks ``retention_bias`` toward 0 every step
    (a → 0.5), and the shrinkage hits the longest-memory channels hardest
    (half-life ≈ ln2·e^b, so a constant relative shrink of b multiplies the
    half-life down). Quantified: IL defaults (lr=1e-3, wd=1e-4) erode the
    slowest channel from 1024 to ~370 ticks over ~1.5M steps. ``h_init``
    is likewise decayed against a weak one-sample-per-sequence gradient.
    """
    no_decay, decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("retention_bias") or name.endswith("temporal_rnn.h_init"):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "weight_decay": weight_decay}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def optim_state_mismatch(optimizer, saved: Dict) -> Optional[str]:
    """Why ``saved`` (an optimizer ``state_dict``) cannot be loaded safely,
    or ``None`` if it lines up with ``optimizer``'s current parameters.

    ``torch.optim`` matches saved state to parameters purely by POSITION.
    A checkpoint from a model with a different parameter registration
    order (e.g. saved before the game-adapter refactor renamed submodules)
    has identical counts, so a bare count check passes and the load
    silently attaches Adam moments to the wrong tensors. Comparing the
    per-position moment shapes catches any reordering between
    differently-shaped parameters.
    """
    saved_groups = saved.get("param_groups", [])
    if len(saved_groups) != len(optimizer.param_groups):
        return (f"{len(saved_groups)} saved vs "
                f"{len(optimizer.param_groups)} current param groups")
    for gi, (sg, cg) in enumerate(zip(saved_groups, optimizer.param_groups)):
        if len(sg["params"]) != len(cg["params"]):
            return (f"group {gi}: {len(sg['params'])} saved vs "
                    f"{len(cg['params'])} current params")
        for sid, p in zip(sg["params"], cg["params"]):
            state = saved.get("state", {}).get(sid)
            if not state:
                continue  # parameter never stepped — nothing to misalign
            for name, v in state.items():
                if torch.is_tensor(v) and v.dim() > 0 and v.shape != p.shape:
                    return (f"group {gi}: saved {name} shape {tuple(v.shape)} "
                            f"!= param shape {tuple(p.shape)} — parameter "
                            f"order changed since the checkpoint was saved")
    return None


def is_legacy_state_dict(state_dict: Dict, model: nn.Module) -> bool:
    """True if ``state_dict`` uses pre-refactor key names for ``model``.

    Checks the model class's ``LEGACY_KEY_MAP`` old prefixes; new-format
    keys never match them (the map is verified idempotent), so this is an
    exact detector for checkpoints saved before the rename.
    """
    legacy_map = getattr(type(model), "LEGACY_KEY_MAP", None) or {}
    if not legacy_map:
        return False
    old_prefixes = tuple(legacy_map)

    def _clean(k: str) -> str:
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k

    return any(_clean(k).startswith(old_prefixes) for k in state_dict)


# ======================================================================
# Checkpoint loading
# ======================================================================

def load_model_state(model: nn.Module, state_dict: Dict, ctx: str = "checkpoint",
                     reinit_prefixes: tuple = ()) -> None:
    """Load a state dict, normalizing DDP/compile prefixes.

    Fails loudly if any model parameter is missing from the checkpoint —
    a silent partial load (strict=False with ignored return value) leaves
    whole modules randomly initialized while training continues.
    Unused checkpoint keys only warn (e.g. optional world-model weights).

    ``reinit_prefixes`` opts specific submodules out of the strict check:
    matching checkpoint keys are dropped and the model's fresh init is kept.
    For hot-starting across an architecture change (e.g. the neighbor-gather
    DirectionHead replacing the cross-attention one).
    """
    def _clean(k: str) -> str:
        for prefix in ("module.", "_orig_mod."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k

    # Models that renamed submodules (e.g. GeneralsNetwork's game-adapter
    # refactor) expose LEGACY_KEY_MAP; accept reinit prefixes in either the
    # old or the new spelling by expanding each prefix into BOTH spellings —
    # the checkpoint-side drop must match old-format checkpoints and the
    # model-side missing filter must match the current (new) key names,
    # regardless of which spelling the caller used.
    prefixes = tuple(reinit_prefixes)
    legacy_map = getattr(type(model), "LEGACY_KEY_MAP", None) or {}
    if prefixes and legacy_map:
        extra = []
        for p in prefixes:
            for old, new in legacy_map.items():
                if p.startswith(old):
                    extra.append(new + p[len(old):])
                elif old.startswith(p):
                    extra.append(new)
                if p.startswith(new):
                    extra.append(old + p[len(new):])
                elif new.startswith(p):
                    extra.append(old)
        prefixes = prefixes + tuple(extra)

    cleaned = {_clean(k): v for k, v in state_dict.items()}
    if prefixes:
        dropped = [k for k in cleaned if k.startswith(prefixes)]
        for k in dropped:
            del cleaned[k]
        if dropped:
            print(f"  [reinit] {ctx}: dropped {len(dropped)} checkpoint keys under "
                  f"{tuple(reinit_prefixes)} — fresh initialization kept")
        else:
            # Zero checkpoint matches is legitimate when the module is new
            # (absent from the checkpoint — fresh init is kept either way),
            # but a prefix that also matches nothing on the MODEL is a typo
            # and would otherwise be silently ignored.
            if not any(k.startswith(prefixes) for k in model.state_dict()):
                raise ValueError(
                    f"{ctx}: reinit prefixes {tuple(reinit_prefixes)} match no "
                    f"checkpoint key and no model key — typo, or a module "
                    f"disabled by the model config?"
                )
            print(f"  [reinit] {ctx}: no checkpoint keys under "
                  f"{tuple(reinit_prefixes)} — module keeps fresh initialization")
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if prefixes:
        missing = [k for k in missing if not k.startswith(prefixes)]
    if missing:
        raise RuntimeError(
            f"{ctx}: {len(missing)} model keys missing from checkpoint "
            f"(would stay randomly initialized), e.g. {missing[:8]}"
        )
    if unexpected:
        print(f"  [warn] {ctx}: {len(unexpected)} unused checkpoint keys, "
              f"e.g. {unexpected[:8]}")


# ======================================================================
# Encoder freeze / unfreeze
# ======================================================================

def freeze_encoder(model: nn.Module) -> None:
    """Freeze Stage-1 encoder weights (obs encoder + ST layers).

    Useful during early RL fine-tuning to stabilise representations
    learned during IL pre-training.  GeneralsNetwork exposes the legacy
    attribute layout (spatial_encoder / global_proj / row+col embeds);
    every other GameNetwork freezes its obs_encoder, which owns the
    game's spatial PE.
    """
    if hasattr(model, "spatial_encoder"):
        for param in model.spatial_encoder.parameters():
            param.requires_grad = False
        for param in model.global_proj.parameters():
            param.requires_grad = False
        # Positional embeddings
        model.row_embed.requires_grad = False
        model.col_embed.requires_grad = False
    else:
        for param in model.obs_encoder.parameters():
            param.requires_grad = False
    if hasattr(model, "st_layers"):
        for param in model.st_layers.parameters():
            param.requires_grad = False
    if hasattr(model, "history_encoder"):
        for param in model.history_encoder.parameters():
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters."""
    for param in model.parameters():
        param.requires_grad = True
