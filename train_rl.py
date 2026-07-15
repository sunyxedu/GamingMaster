"""
Reinforcement Learning training script for the generals.io bot.

Implements distributed PPO self-play with:
  - Dense reward (heuristic delta + event rewards + terminal ±5.0)
  - No legal mask (model learns legality through reward)
  - IL auxiliary regularisation (AlphaStar-style)
  - KL regularisation against frozen reference policy
  - Opponent pool (league training)
  - Encoder freeze / unfreeze schedule

Usage:
    torchrun --nproc_per_node=8 --nnodes=4 ... train_rl.py \\
        --il_checkpoint checkpoints/finetune/checkpoint_epoch67.pt \\
        --il_data_dir data-1k/training/Human.exe \\
        --iterations 10000
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import ImitationLoss, RLLoss
from model.network import GeneralsNetwork
from model.rl_utils import (
    RolloutBuffer,
    compute_policy_kl_rl,
    freeze_encoder,
    is_legacy_state_dict,
    load_model_state,
    optim_param_groups,
    optim_state_mismatch,
    sync_gradients,
    unfreeze_all,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RL training for generals.io bot")
    # Checkpoint & data
    p.add_argument("--il_checkpoint", type=str, default=None,
                   help="Path to IL-pretrained checkpoint (.pt)")
    p.add_argument("--il_data_dir", type=str, nargs="+", default=None,
                   help="Directory(s) with .json IL training data (for joint training)")
    p.add_argument("--output_dir", type=str, default="checkpoints_rl")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from RL checkpoint")

    # RL hyper-parameters
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--rollout_steps", type=int, default=1000,
                   help="Max steps per self-play episode")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--act_steps", type=int, default=1,
                   help="Fixed TRM ACT depth used for BOTH rollout and PPO update "
                        "(the two must match for the PPO ratio to be valid)")

    # Loss weights
    p.add_argument("--value_weight", type=float, default=0.5)
    p.add_argument("--entropy_weight", type=float, default=0.01)
    p.add_argument("--entropy_weight_decay", type=float, default=1.0,
                   help="Per-iteration decay for entropy weight (1.0 = no decay)")
    p.add_argument("--entropy_weight_min", type=float, default=0.008,
                   help="Minimum entropy weight (floor for decay)")
    p.add_argument("--il_weight", type=float, default=0.5,
                   help="Weight of IL auxiliary loss")
    p.add_argument("--il_decay", type=float, default=0.9999,
                   help="Per-iteration decay for il_weight")
    p.add_argument("--kl_weight", type=float, default=0.01,
                   help="KL regularisation against reference policy")
    p.add_argument("--kl_ref", type=str, default=None,
                   help="Checkpoint whose weights define the frozen KL "
                        "reference policy. Default: this run's starting "
                        "weights — on --resume that is the RESUMED policy, "
                        "so pass the IL checkpoint here to keep the human "
                        "anchor across restarts.")
    p.add_argument("--invalidity_weight", type=float, default=0.1,
                   help="Weight for source/direction invalidity auxiliary loss")
    p.add_argument("--capture_aux_weight", type=float, default=0.05,
                   help="Weight of the auxiliary capture-feasibility BCE "
                        "head (0 = disabled). Supervises 'would a full move "
                        "from tile X in direction d capture the target' so "
                        "the encoder learns exact army comparisons.")
    p.add_argument("--city_budget", type=float, default=0.8,
                   help="Total neutral-city reward budget per map; the "
                        "per-city payout unit is budget/n_cities (capped "
                        "at 0.25), keeping aggregate city shaping below "
                        "the terminal win/loss reward")
    p.add_argument("--city_reward_decay", type=float, default=0.9999,
                   help="Per-iteration decay for city capture reward scale")
    p.add_argument("--dense_reward_decay", type=float, default=0.9999,
                   help="Per-iteration decay for dense heuristic reward scale")
    p.add_argument("--reset_reward_decay", action="store_true",
                   help="On --resume, restart the city/dense reward decay "
                        "schedules at 1.0 instead of decay^done_iters. Use "
                        "when a phase boundary changes the reward structure "
                        "and the new shaping should not enter pre-annealed.")

    # Encoder freeze schedule
    p.add_argument("--freeze_encoder_iters", type=int, default=1000,
                   help="Freeze encoder for the first N iterations (0 = never freeze)")

    # Multi-episode collection
    p.add_argument("--episodes_per_gpu", type=int, default=1,
                   help="Episodes to collect per GPU before PPO update (gradient accumulation)")

    # Opponent pool
    p.add_argument("--pool_size", type=int, default=10,
                   help="Max opponents in pool (excluding IL baseline)")
    p.add_argument("--pool_update_interval", type=int, default=100,
                   help="Check pool update eligibility every N iterations")
    p.add_argument("--pool_update_winrate", type=float, default=0.0,
                   help="Min winrate over recent iterations to update pool (0 = always update)")
    p.add_argument("--pool_update_window", type=int, default=50,
                   help="Number of recent iterations to compute winrate for pool update")

    # Map pool (use real replay maps instead of random)
    p.add_argument("--replay_dir", type=str, default=None,
                   help="Directory with raw replay JSON files")
    p.add_argument("--filtered_file", type=str, default=None,
                   help="Path to filtered replay list JSON")

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_decay", type=float, default=0.9997,
                   help="Per-iteration multiplicative LR decay")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Model (used when no checkpoint is provided)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=3)
    p.add_argument("--temporal_mixer", type=str, default="attention",
                   choices=["attention", "recurrent"],
                   help="Temporal mixer for from-scratch models (checkpoints "
                        "carry their own setting in the saved config)")

    # Fixed-opponent evaluation (progress signal independent of shaping
    # rewards and of the moving opponent pool)
    p.add_argument("--eval_interval", type=int, default=50,
                   help="Every N iterations play eval games vs the frozen "
                        "KL reference policy (0 = never; requires --kl_weight"
                        " > 0 so the reference model exists)")
    p.add_argument("--eval_games", type=int, default=4,
                   help="Eval games per GPU per evaluation round")

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--save_replay_interval", type=int, default=50,
                   help="Save a game replay every N iterations (0 = never)")
    p.add_argument("--device", type=str, default="auto")

    args = p.parse_args()
    # An empty path (e.g. a shell glob that matched nothing) is falsy at
    # every `if args.resume:` check and would silently start from scratch.
    for flag in ("resume", "il_checkpoint", "kl_ref"):
        val = getattr(args, flag)
        if val is not None and not val.strip():
            p.error(f"--{flag} given an empty path")
    return args


# ======================================================================
# Opponent pool
# ======================================================================

class OpponentPool:
    """Maintains a pool of past model checkpoints for league training.

    Index 0 is always the IL baseline (never removed).
    """

    @staticmethod
    def _to_cpu(state_dict: dict) -> dict:
        """Detached CPU copy — keeps up to max_size+1 models off the GPU."""
        return {k: v.detach().to("cpu", copy=True) for k, v in state_dict.items()}

    def __init__(self, initial_state: dict, max_size: int = 10):
        self.pool: List[dict] = [self._to_cpu(initial_state)]
        self.max_size = max_size

    def add(self, state_dict: dict) -> None:
        self.pool.append(self._to_cpu(state_dict))
        if len(self.pool) > self.max_size + 1:  # +1 for IL baseline
            self.pool.pop(1)  # keep index 0 (IL baseline)

    def sample(self) -> dict:
        """50% from latest 3, 50% random from pool."""
        if len(self.pool) > 1 and random.random() < 0.5:
            return random.choice(self.pool)
        # Pick from the latest 3 (excluding IL baseline at index 0)
        recent = self.pool[-min(3, len(self.pool) - 1):]
        return random.choice(recent) if recent else self.pool[-1]

    def __len__(self) -> int:
        return len(self.pool)


# ======================================================================
# Auxiliary capture-feasibility head
# ======================================================================

class CaptureHead(torch.nn.Module):
    """Per-tile, per-direction capture-feasibility predictor.

    Reads the Stage-1 belief tokens and predicts, for each owned tile and
    direction onto a defended target (neutral city or enemy tile), whether
    a full-strength move would capture it — labels come exactly from the
    env (``capture_label``/``capture_mask`` observation keys).

    The policy's army input is a single log-compressed channel, so the
    "src.army - 1 > dst.army" threshold is nearly invisible in input space;
    this head's BCE backpropagates into the encoder and forces it to
    represent the comparison. Kept OUTSIDE ``GeneralsNetwork`` so model
    checkpoints, the opponent pool and the KL reference stay byte-
    compatible; it is trained by its own small optimizer.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 4),
        )

    def forward(self, belief_tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, N, d] → capture logits [B, T, N, 4]."""
        return self.net(belief_tokens)


# ======================================================================
# Self-play rollout collection
# ======================================================================

def collect_selfplay_episode(
    model: GeneralsNetwork,
    opponent: GeneralsNetwork,
    env: GeneralsEnv,
    device: torch.device,
    max_steps: int = 1000,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    record: bool = False,
    act_steps: int = 1,
) -> Tuple[RolloutBuffer, Optional[List[Dict]]]:
    """Run one self-play episode: model (player 0) vs opponent (player 1).

    Only collects trajectory for player 0 (the model being trained).
    Uses step_rl (no legal mask) for both players.

    If ``record=True``, returns a list of per-step dicts for replay viewing.
    """
    model.eval()
    opponent.eval()
    obs = env.reset()
    buffer = RolloutBuffer()
    caches = {0: None, 1: None}    # spatiotemporal KV caches
    replay_log: Optional[List[Dict]] = [] if record else None
    reward_totals: Dict[str, float] = {
        "r_invalid": 0.0, "r_city": 0.0, "r_dense": 0.0, "r_winlose": 0.0,
        "c_cap": 0.0, "c_bash": 0.0, "c_steal": 0.0, "c_loss": 0.0,
    }

    # Record initial state before any moves
    if replay_log is not None:
        snapshot = env.get_state_snapshot()
        replay_log.append({
            "turn": -1,
            "p0_action": None, "p1_action": None,
            "p0_reward": 0.0, "p1_reward": 0.0,
            "illegal": [False, False],
            "p0_stats": env._player_stats(0),
            "p1_stats": env._player_stats(1),
            "state": snapshot["state"],
            "army": snapshot["army"],
            "done": False, "winner": None,
        })

    for _step in range(max_steps):
        step_out = {}

        for pid, net in [(0, model), (1, opponent)]:
            grid = obs[pid]["grid"].unsqueeze(0).to(device)
            glb = obs[pid]["globals"].unsqueeze(0).to(device)

            with torch.no_grad():
                out = net.step_rl(grid, glb,
                                  st_cache=caches[pid], step_idx=_step,
                                  act_steps=act_steps)

            caches[pid] = out.get("st_cache")
            step_out[pid] = out

        # Build env actions
        actions_for_env = {}
        for pid in [0, 1]:
            did_pass = step_out[pid]["action_pass"].item()
            if not did_pass:
                actions_for_env[pid] = {
                    "source_idx": step_out[pid]["action_source"].item(),
                    "direction": step_out[pid]["action_direction"].item(),
                    "split": step_out[pid]["action_split"].item(),
                }
            else:
                actions_for_env[pid] = None

        # Environment step
        new_obs, rewards, done, info = env.step(
            actions_for_env.get(0), actions_for_env.get(1),
        )

        # Record step for replay (snapshot AFTER env.step)
        if replay_log is not None:
            snapshot = env.get_state_snapshot()
            replay_log.append({
                "turn": info["turn"],
                "p0_action": actions_for_env.get(0),
                "p1_action": actions_for_env.get(1),
                "p0_reward": rewards[0],
                "p1_reward": rewards[1],
                "illegal": [info["illegal"][0], info["illegal"][1]],
                "p0_stats": env._player_stats(0),
                "p1_stats": env._player_stats(1),
                "state": snapshot["state"],
                "army": snapshot["army"],
                "done": done,
                "winner": info.get("winner"),
            })

        # Accumulate reward breakdown
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                reward_totals[k] += v

        # Store transition for player 0 only
        action_dict = {
            "source_idx": step_out[0]["action_source"].item(),
            "direction": step_out[0]["action_direction"].item(),
            "split": step_out[0]["action_split"].item(),
            "passed": step_out[0]["action_pass"].item(),
        }
        # Compute valid directions for chosen source
        valid_dirs = env.get_valid_directions(action_dict["source_idx"])
        buffer.add(
            obs=obs[0],
            action=action_dict,
            log_prob=step_out[0]["action_log_prob"].item(),
            value=step_out[0]["value"].item(),
            reward=rewards[0],
            # Truncation (env turn limit) is a SOFT terminal: the env already
            # scored it by material, so GAE must not bootstrap on top.
            done=done,
            valid_directions=valid_dirs,
        )

        obs = new_obs
        if done:
            break

    # Rollout budget exhausted mid-game: the env never saw a terminal, so
    # apply the same material-based truncation score here and close the
    # episode — mirrors the env's own turn-limit handling.
    if not done and len(buffer) > 0:
        buffer.rewards[-1] += env.truncation_score(0)
        buffer.dones[-1] = True
        reward_totals["r_winlose"] += env.truncation_score(0)

    buffer.compute_returns_and_advantages(
        gamma=gamma, lam=gae_lambda, last_value=0.0,
    )

    return buffer, replay_log, reward_totals


# ======================================================================
# PPO update
# ======================================================================

def ppo_update(
    model: GeneralsNetwork,
    ref_model: Optional[GeneralsNetwork],
    buffers: List[RolloutBuffer],
    rl_criterion: RLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ModelConfig,
    world_size: int = 1,
    kl_weight: float = 0.01,
    invalidity_weight: float = 0.1,
    grad_clip: float = 1.0,
    act_steps: int = 1,
    capture_head: Optional[CaptureHead] = None,
    capture_optimizer: Optional[torch.optim.Optimizer] = None,
    capture_weight: float = 0.0,
) -> Dict[str, float]:
    """Run one PPO update epoch with gradient accumulation over multiple trajectories."""
    # eval mode: rollout log-probs were collected in eval mode, so the update
    # must disable dropout and use BatchNorm running stats — otherwise
    # old/new log-probs come from different policies and the ratio is invalid.
    # Gradients still flow in eval mode.
    model.eval()
    optimizer.zero_grad()
    use_capture = (capture_head is not None and capture_optimizer is not None
                   and capture_weight > 0)
    if use_capture:
        capture_optimizer.zero_grad()

    n_buffers = len(buffers)
    metrics_accum: Dict[str, float] = {}

    for buf in buffers:
        batch = buf.to_tensor_batch(device)

        grid_seq = batch["grid_seq"]        # [1, T, C, H, W]
        globals_seq = batch["globals_seq"]  # [1, T, G]
        actions = batch["actions"]
        old_lp = batch["old_log_probs"]     # [1, T]
        advantages = batch["advantages"]    # [1, T]
        returns = batch["returns"]          # [1, T]

        # Forward with stored source positions (no legal mask)
        source_y = actions["source_idx"] // cfg.grid_w
        source_x = actions["source_idx"] % cfg.grid_w

        pred = model(
            grid_seq, globals_seq,
            legal_mask=None,
            source_y=source_y,
            source_x=source_x,
            act_steps=act_steps,
        )

        log_probs, entropy, head_entropy = model.compute_action_log_prob_rl(pred, actions)

        # RL loss
        rl_losses = rl_criterion(
            log_probs=log_probs,
            entropy=entropy,
            values=pred["value"],
            returns=returns,
            advantages=advantages,
            old_log_probs=old_lp,
        )

        total = rl_losses["loss"]

        # KL regularisation against reference policy (no legal mask)
        kl_loss = torch.tensor(0.0, device=device)
        if ref_model is not None and kl_weight > 0:
            with torch.no_grad():
                ref_pred = ref_model(
                    grid_seq, globals_seq,
                    legal_mask=None,
                    source_y=source_y,
                    source_x=source_x,
                    act_steps=act_steps,
                )
            kl_loss = compute_policy_kl_rl(
                pred["source_logits"],
                ref_pred["source_logits"].detach(),
            )
            total = total + kl_weight * kl_loss

        # Invalidity loss: push probability toward legal sources & valid directions
        inv_loss = torch.tensor(0.0, device=device)
        if invalidity_weight > 0:
            T = grid_seq.size(1)
            not_passed = ~actions["passed"]  # [1, T]

            # Source validity: encourage prob mass on legal source positions
            legal_mask_flat = batch["legal_mask"].reshape(1, T, -1).float()
            source_probs = F.softmax(pred["source_logits"], dim=-1)
            prob_on_valid_src = (source_probs * legal_mask_flat).sum(-1)
            src_val_loss = -torch.log(prob_on_valid_src.clamp(min=1e-8))
            src_mask = not_passed.float()
            n_src = src_mask.sum().clamp(min=1)
            src_val_loss = (src_val_loss * src_mask).sum() / n_src

            # Direction validity: encourage prob mass on valid directions
            valid_dirs = batch["valid_directions"].float()
            dir_probs = F.softmax(pred["direction_logits"], dim=-1)
            prob_on_valid_dir = (dir_probs * valid_dirs).sum(-1)
            dir_val_loss = -torch.log(prob_on_valid_dir.clamp(min=1e-8))
            source_was_valid = legal_mask_flat.gather(
                -1, actions["source_idx"].unsqueeze(-1),
            ).squeeze(-1)
            dir_mask = source_was_valid * not_passed.float()
            n_dir = dir_mask.sum().clamp(min=1)
            dir_val_loss = (dir_val_loss * dir_mask).sum() / n_dir

            inv_loss = src_val_loss + dir_val_loss
            total = total + invalidity_weight * inv_loss

        # Auxiliary capture-feasibility BCE: exact env labels, masked to
        # (owned movable tile, defended neighbour) pairs. Backpropagates
        # through the belief tokens into the encoder.
        cap_loss = torch.tensor(0.0, device=device)
        cap_acc = 0.0
        if use_capture and "capture_label" in batch:
            T = grid_seq.size(1)
            cap_logits = capture_head(pred["belief_tokens"])  # [1, T, N, 4]
            # [1, T, 4, H, W] → [1, T, N, 4] matching the token layout
            cap_labels = (batch["capture_label"]
                          .reshape(1, T, 4, -1).permute(0, 1, 3, 2))
            cap_mask = (batch["capture_mask"]
                        .reshape(1, T, 4, -1).permute(0, 1, 3, 2)).float()
            n_lab = cap_mask.sum().clamp(min=1)
            bce = F.binary_cross_entropy_with_logits(
                cap_logits, cap_labels, reduction="none")
            cap_loss = (bce * cap_mask).sum() / n_lab
            total = total + capture_weight * cap_loss
            with torch.no_grad():
                cap_acc = ((((cap_logits > 0).float() == cap_labels).float()
                            * cap_mask).sum() / n_lab).item()

        # Scale loss for gradient accumulation
        (total / n_buffers).backward()

        # Accumulate metrics
        step_metrics = {
            "total_loss": total.item(),
            "policy_loss": rl_losses["policy_loss"].item(),
            "value_loss": rl_losses["value_loss"].item(),
            "entropy": rl_losses["entropy"].item(),
            "approx_kl": rl_losses["approx_kl"].item(),
            "clip_frac": rl_losses["clip_fraction"].item(),
            "kl_reg": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "inv_loss": inv_loss.item() if isinstance(inv_loss, torch.Tensor) else inv_loss,
            "ent_pass": head_entropy["ent_pass"].item(),
            "ent_src": head_entropy["ent_src"].item(),
            "ent_dir": head_entropy["ent_dir"].item(),
            "ent_spl": head_entropy["ent_spl"].item(),
            "cap_loss": cap_loss.item() if isinstance(cap_loss, torch.Tensor) else cap_loss,
            "cap_acc": cap_acc,
        }
        for k, v in step_metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v

    if world_size > 1:
        sync_gradients(model, world_size)
        if use_capture:
            sync_gradients(capture_head, world_size)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if use_capture:
            torch.nn.utils.clip_grad_norm_(capture_head.parameters(), grad_clip)
    optimizer.step()
    if use_capture:
        capture_optimizer.step()

    return {k: v / n_buffers for k, v in metrics_accum.items()}


def il_auxiliary_update(
    model: GeneralsNetwork,
    il_batch: Dict,
    il_criterion: ImitationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    il_weight: float,
    world_size: int = 1,
    grad_clip: float = 1.0,
    act_steps: int = 1,
) -> Dict[str, float]:
    """One gradient step on IL expert data (auxiliary regularisation).

    IL path keeps legal_mask (model learns from expert data with mask).
    Runs at the same ACT depth as PPO so the regularizer constrains the
    policy that is actually being trained.
    """
    model.train()

    grid_seq = il_batch["obs"]["grid_seq"].to(device)
    globals_seq = il_batch["obs"]["globals_seq"].to(device)
    legal_mask = il_batch["obs"]["legal_source_mask_seq"].to(device)
    tgt = {k: v.to(device) for k, v in il_batch["tgt"].items()}
    lengths = il_batch["meta"]["lengths"]

    pred = model(
        grid_seq, globals_seq,
        legal_mask=legal_mask,
        source_y=None,
        source_x=None,
        act_steps=act_steps,
    )

    il_losses = il_criterion(pred, tgt, lengths)

    loss = il_weight * il_losses["loss"]
    optimizer.zero_grad()
    loss.backward()
    if world_size > 1:
        sync_gradients(model, world_size)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "il_loss": il_losses["loss"].item(),
        "il_source": il_losses["source_loss"].item(),
    }


# ======================================================================
# Main training loop
# ======================================================================

def main():
    args = parse_args()

    # Distributed setup
    distributed = "LOCAL_RANK" in os.environ
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rank0 = not distributed or dist.get_rank() == 0
    world_size = dist.get_world_size() if distributed else 1
    if rank0:
        suffix = f" (world_size={world_size})" if distributed else ""
        print(f"Device: {device}{suffix}")

    # ---- Build or load model -------------------------------------------
    # A resume checkpoint defines the architecture: rebuilding from CLI
    # defaults would silently change the run's lineage (drop the world-model
    # head, switch temporal mixer) or crash at opponent-pool load.
    rl_ckpt = None
    if args.resume:
        # Load once, on CPU: the checkpoint embeds the whole opponent pool,
        # and load_state_dict copies to the right devices on its own.
        rl_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    if rl_ckpt is not None and "config" in rl_ckpt:
        cfg = ModelConfig(**{k: v for k, v in rl_ckpt["config"].items()
                             if k in ModelConfig.__dataclass_fields__})
        model = GeneralsNetwork(cfg).to(device)
        if rank0:
            print(f"Architecture from resume checkpoint {args.resume}")
            if args.il_checkpoint:
                print("  [note] --il_checkpoint ignored on resume (weights and "
                      "architecture come from the resume checkpoint)")
    elif args.il_checkpoint:
        ckpt = torch.load(args.il_checkpoint, map_location=device, weights_only=False)
        cfg = ModelConfig(**ckpt["config"])
        model = GeneralsNetwork(cfg).to(device)
        load_model_state(model, ckpt["model_state"], ctx="IL checkpoint")
        if rank0:
            print(f"Loaded IL checkpoint from {args.il_checkpoint}")
    else:
        cfg = ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            trm_H_cycles=args.trm_H_cycles,
            trm_L_cycles=args.trm_L_cycles,
            trm_block_layers=args.trm_block_layers,
            temporal_mixer=args.temporal_mixer,
        )
        model = GeneralsNetwork(cfg).to(device)
        if rank0:
            print("No IL checkpoint — training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    if rank0:
        print(f"Model parameters: {n_params:,}")

    # ---- Resume: load weights & counters FIRST -------------------------
    # Must happen before the KL reference model and opponent pool are
    # captured, otherwise a resume without --il_checkpoint regularizes
    # toward (and plays against) a randomly initialized network.
    start_iter = 1
    il_weight = args.il_weight
    wins = {0: 0, 1: 0, "draw": 0}
    if args.resume:
        # (checkpoint already loaded above, where it defined the architecture)
        # Support both RL checkpoints (model_state) and IL checkpoints
        state_key = "model_state" if "model_state" in rl_ckpt else "model_state_dict"
        load_model_state(model, rl_ckpt[state_key], ctx="resume checkpoint")
        start_iter = rl_ckpt.get("iteration", 0) + 1
        il_weight = rl_ckpt.get("il_weight", args.il_weight)
        wins = rl_ckpt.get("wins", wins)
        if rank0:
            print(f"Resumed weights from {args.resume} (starting iteration {start_iter})")

    # ---- Auxiliary capture head (own optimizer, outside the model) ------
    # Separate module + optimizer so model checkpoints, the opponent pool
    # and the KL reference stay byte-compatible with head-less runs.
    capture_head: Optional[CaptureHead] = None
    capture_optimizer: Optional[torch.optim.Optimizer] = None
    if args.capture_aux_weight > 0:
        capture_head = CaptureHead(cfg.d_model).to(device)
        capture_optimizer = torch.optim.AdamW(
            capture_head.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if rl_ckpt is not None and "capture_head_state" in rl_ckpt:
            capture_head.load_state_dict(rl_ckpt["capture_head_state"])
            if "capture_optim_state" in rl_ckpt:
                capture_optimizer.load_state_dict(rl_ckpt["capture_optim_state"])
            if rank0:
                print("  Capture head resumed from checkpoint")
        elif rank0:
            print(f"Capture head enabled (weight={args.capture_aux_weight})")

    # ---- Sync initial state across ranks --------------------------------
    # Gradients are manually all-reduced, but nothing else aligns the ranks:
    # a from-scratch multi-GPU run would average gradients over different
    # random initializations. Broadcast rank 0's parameters and buffers
    # (DDP does this at construction; the manual setup must too).
    if distributed:
        for t in model.state_dict().values():
            if isinstance(t, torch.Tensor):
                dist.broadcast(t, src=0)
        if capture_head is not None:
            for t in capture_head.state_dict().values():
                if isinstance(t, torch.Tensor):
                    dist.broadcast(t, src=0)

    # ---- Reference model (frozen copy for KL regularisation) -----------
    ref_model: Optional[GeneralsNetwork] = None
    if args.kl_weight > 0:
        ref_model = copy.deepcopy(model)
        if args.kl_ref:
            ref_ckpt = torch.load(args.kl_ref, map_location="cpu", weights_only=False)
            load_model_state(ref_model, ref_ckpt["model_state"],
                             ctx="KL reference checkpoint")
            del ref_ckpt
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        if rank0:
            src = args.kl_ref if args.kl_ref else "this run's starting weights"
            print(f"Created frozen reference model for KL regularisation ({src})")

    # ---- Opponent pool -------------------------------------------------
    opponent_pool = OpponentPool(
        initial_state=model.state_dict(),
        max_size=args.pool_size,
    )
    if rl_ckpt is not None and "opponent_pool" in rl_ckpt:
        opponent_pool.pool = [
            OpponentPool._to_cpu(sd) for sd in rl_ckpt["opponent_pool"]
        ]
    # Opponent model (loaded with pool checkpoint each iteration)
    opponent_model = copy.deepcopy(model)
    opponent_model.eval()
    for p in opponent_model.parameters():
        p.requires_grad = False
    if rank0:
        print(f"Opponent pool initialised (max_size={args.pool_size})")

    # ---- Encoder freeze ------------------------------------------------
    # Phase decided by start_iter so a resume past the boundary builds the
    # optimizer over the full parameter set from the start.
    frozen_phase = args.freeze_encoder_iters > 0 and start_iter <= args.freeze_encoder_iters
    if frozen_phase:
        freeze_encoder(model)
        if rank0:
            print(f"Encoder frozen for the first {args.freeze_encoder_iters} iterations")

    # ---- DDP -----------------------------------------------------------
    # No DDP wrapper — use manual gradient all-reduce instead.
    # DDP conflicts with gradient checkpointing + varying computation graphs
    # (PPO vs IL forward paths). Manual sync is simpler and reliable.

    # ---- Losses --------------------------------------------------------
    rl_criterion = RLLoss(
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
        clip_range=args.clip_range,
    )
    # AlphaStar-style auxiliary regularization targets the POLICY heads only:
    # the replay-outcome value labels live on a different scale and state
    # distribution than the PPO returns and would fight the critic every step.
    il_criterion = ImitationLoss(grid_w=cfg.grid_w, value_weight=0.0)

    # ---- Optimiser -----------------------------------------------------
    # optim_param_groups keeps weight decay off the recurrence memory
    # parameters (retention_bias / h_init) and respects requires_grad.
    optimizer = torch.optim.AdamW(
        optim_param_groups(model, args.weight_decay),
        lr=args.lr,
    )

    # ---- IL data loader (optional) -------------------------------------
    il_loader = None
    il_iter = None
    if args.il_data_dir:
        if len(args.il_data_dir) == 1:
            il_dataset = SeqReplayDataset(args.il_data_dir[0], max_sequence_length=200)
        else:
            from torch.utils.data import ConcatDataset
            il_dataset = ConcatDataset([
                SeqReplayDataset(d, max_sequence_length=200)
                for d in args.il_data_dir
            ])
        il_sampler = DistributedSampler(il_dataset, shuffle=True) if distributed else None
        il_loader = DataLoader(
            il_dataset, batch_size=1, shuffle=(il_sampler is None),
            sampler=il_sampler,
            collate_fn=collate_variable_length_sequences,
        )
        il_iter = iter(il_loader)
        if rank0:
            print(f"IL dataset: {len(il_dataset)} replays from {args.il_data_dir}")

    # ---- Environment (each rank gets its own) --------------------------
    env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
    env.city_budget_total = args.city_budget
    if args.replay_dir and args.filtered_file:
        n_maps = env.load_map_pool(args.replay_dir, args.filtered_file)
        if rank0:
            print(f"Loaded {n_maps} replay maps for environment")

    # ---- Resume: optimizer state & schedules (weights loaded earlier) ---
    if args.resume:
        if "iteration" in rl_ckpt and "optimizer_state" in rl_ckpt:
            # Skip states saved for a different trainable set (e.g. a
            # checkpoint from the frozen phase resumed past the boundary) —
            # loading them raises inside torch.optim. Also skip states saved
            # before the game-adapter refactor: identical counts but a
            # different parameter order, which a bare count check misses.
            saved = rl_ckpt["optimizer_state"]
            saved_model = rl_ckpt.get(
                "model_state", rl_ckpt.get("model_state_dict", {}))
            if is_legacy_state_dict(saved_model, model):
                reason = "pre-refactor checkpoint (parameter order changed)"
            else:
                reason = optim_state_mismatch(optimizer, saved)
            if reason is None:
                optimizer.load_state_dict(saved)
            elif rank0:
                print(f"  [warn] optimizer state skipped ({reason}); "
                      f"starting with a fresh optimizer")

        # Restore the LR schedule: base LR decayed for the iterations already
        # done (like every other decayed quantity below), halved once the
        # unfreeze boundary has been crossed.
        done_iters = start_iter - 1
        resume_lr = args.lr * (args.lr_decay ** done_iters)
        if args.freeze_encoder_iters > 0 and start_iter > args.freeze_encoder_iters:
            resume_lr *= 0.5
        for pg in optimizer.param_groups:
            pg["lr"] = resume_lr
        if capture_optimizer is not None:
            for pg in capture_optimizer.param_groups:
                pg["lr"] = resume_lr
        if rank0:
            print(f"  LR restored to {resume_lr:.3e} (decay^{done_iters})")

        # Restore reward scales. Checkpoints that carry them (saved by this
        # code) continue exactly — --reset_reward_decay must NOT re-zero the
        # clock on every within-phase crash recovery. Older checkpoints
        # (previous phase) fall back to the flag / recompute.
        if "city_reward_scale" in rl_ckpt:
            env.city_reward_scale = rl_ckpt["city_reward_scale"]
            env.dense_reward_scale = rl_ckpt["dense_reward_scale"]
        elif args.reset_reward_decay:
            env.city_reward_scale = 1.0
            env.dense_reward_scale = 1.0
        else:
            env.city_reward_scale = args.city_reward_decay ** done_iters
            env.dense_reward_scale = args.dense_reward_decay ** done_iters
        rl_criterion.entropy_weight = max(
            args.entropy_weight * (args.entropy_weight_decay ** done_iters),
            args.entropy_weight_min,
        )

        if rank0:
            print(f"Resumed from {args.resume}, iteration {start_iter}, "
                  f"pool={len(opponent_pool)}")
        rl_ckpt = None  # release checkpoint tensors

    os.makedirs(args.output_dir, exist_ok=True)

    # Win rate history for conditional pool updates
    from collections import deque
    winrate_history: deque = deque(maxlen=args.pool_update_window)
    # Fresh runs may add to the pool immediately; a resume waits a full
    # interval so a 1-sample winrate window can't evict a historical
    # opponent on the first iteration after every restart.
    last_pool_update_iter = start_iter if args.resume \
        else start_iter - args.pool_update_interval

    # ---- Training loop -------------------------------------------------
    for iteration in range(start_iter, args.iterations + 1):
        t0 = time.time()

        # Unfreeze encoder after schedule. Only fires when THIS process
        # started in the frozen phase — a resume landing exactly past the
        # boundary already applied the 0.5x factor in the resume block and
        # would otherwise be halved a second time.
        if frozen_phase and iteration == args.freeze_encoder_iters + 1:
            raw = model
            unfreeze_all(raw)
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer = torch.optim.AdamW(
                optim_param_groups(model, args.weight_decay),
                lr=current_lr * 0.5,
            )
            if rank0:
                print(f"Iteration {iteration}: encoder unfrozen, LR reduced to {current_lr * 0.5:.3e}")

        # ---- 1. Load opponent from pool ----
        # Each GPU picks one opponent for all its local episodes.
        # Local games share the same opponent (clean gradient), but different
        # GPUs may face different opponents (diversity via all-reduce).
        opp_state = opponent_pool.sample()
        opponent_model.load_state_dict(opp_state)

        # ---- 2. Collect multiple self-play episodes per GPU ----
        should_record = (
            rank0
            and args.save_replay_interval > 0
            and iteration % args.save_replay_interval == 0
        )
        ep_buffers: List[RolloutBuffer] = []
        replay_log = None
        reward_totals: Dict[str, float] = {
            "r_invalid": 0.0, "r_city": 0.0, "r_dense": 0.0, "r_winlose": 0.0,
            "c_cap": 0.0, "c_bash": 0.0, "c_steal": 0.0, "c_loss": 0.0,
        }
        local_ep_len = 0.0
        local_ep_reward = 0.0
        local_p0 = 0
        local_p1 = 0
        local_draw = 0

        replay_env_snapshot = None  # env state for the recorded episode
        for ep_i in range(args.episodes_per_gpu):
            record_this = should_record and ep_i == 0
            buf, ep_replay, ep_rw = collect_selfplay_episode(
                model, opponent_model, env, device,
                max_steps=args.rollout_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                record=record_this,
                act_steps=args.act_steps,
            )
            if ep_replay is not None:
                replay_log = ep_replay
                replay_env_snapshot = {
                    "winner": env.winner,
                    "map_pool_idx": env.current_map_idx,
                    "replay_id": env.map_pool[env.current_map_idx]["replay_id"] if env.map_pool and env.current_map_idx is not None else None,
                    "real_h": env.real_h, "real_w": env.real_w,
                    "generals": env.generals, "cities": env.cities,
                    "mountains": env.mountains,
                }
            ep_buffers.append(buf)
            local_ep_len += len(buf)
            local_ep_reward += sum(buf.rewards)
            for k in reward_totals:
                reward_totals[k] += ep_rw[k]

            if env.winner == 0:
                wins[0] += 1
                local_p0 += 1
            elif env.winner == 1:
                wins[1] += 1
                local_p1 += 1
            else:
                wins["draw"] += 1
                local_draw += 1

        n_local_eps = args.episodes_per_gpu
        local_ep_len /= n_local_eps
        local_ep_reward /= n_local_eps
        for k in reward_totals:
            reward_totals[k] /= n_local_eps

        # ---- 3. PPO update with gradient accumulation ----
        # Normalize advantages across all local buffers + all GPUs
        all_advs = []
        for buf in ep_buffers:
            if len(buf) > 0:
                all_advs.append(buf.advantages)
        if all_advs:
            cat_adv = torch.cat(all_advs)
            if dist.is_initialized():
                local_sum = cat_adv.sum()
                local_sq_sum = (cat_adv ** 2).sum()
                local_count = torch.tensor(cat_adv.numel(), dtype=torch.float32, device=device)
                stats = torch.stack([local_sum.to(device), local_sq_sum.to(device), local_count])
                dist.all_reduce(stats)
                g_mean = stats[0] / stats[2]
                g_var = (stats[1] / stats[2] - g_mean ** 2).clamp(min=1e-8)
                for buf in ep_buffers:
                    if len(buf) > 0:
                        buf.advantages = (buf.advantages - g_mean.cpu()) / g_var.sqrt().cpu()
            else:
                adv_mean = cat_adv.mean()
                adv_std = cat_adv.std().clamp(min=1e-8)
                for buf in ep_buffers:
                    if len(buf) > 0:
                        buf.advantages = (buf.advantages - adv_mean) / adv_std
            del cat_adv, all_advs
            torch.cuda.empty_cache()

        valid_buffers = [buf for buf in ep_buffers if len(buf) > 0]
        ppo_metrics_agg = {}
        if valid_buffers:
            for _epoch in range(args.ppo_epochs):
                metrics = ppo_update(
                    model, ref_model, valid_buffers, rl_criterion,
                    optimizer, device, cfg,
                    world_size=world_size,
                    kl_weight=args.kl_weight,
                    invalidity_weight=args.invalidity_weight,
                    grad_clip=args.grad_clip,
                    act_steps=args.act_steps,
                    capture_head=capture_head,
                    capture_optimizer=capture_optimizer,
                    capture_weight=args.capture_aux_weight,
                )
                for k, v in metrics.items():
                    ppo_metrics_agg[k] = ppo_metrics_agg.get(k, 0.0) + v

        n_updates = args.ppo_epochs if valid_buffers else 1
        ppo_avg = {k: v / n_updates for k, v in ppo_metrics_agg.items()}

        # ---- 4. IL auxiliary update ----
        il_metrics = {}
        if il_loader is not None and il_weight > 1e-6:
            try:
                il_batch = next(il_iter)
            except StopIteration:
                if il_sampler is not None:
                    il_sampler.set_epoch(iteration)
                il_iter = iter(il_loader)
                il_batch = next(il_iter)

            il_metrics = il_auxiliary_update(
                model, il_batch, il_criterion, optimizer, device,
                il_weight=il_weight,
                world_size=world_size,
                grad_clip=args.grad_clip,
                act_steps=args.act_steps,
            )
            il_weight *= args.il_decay

        # Decay shaping rewards
        env.city_reward_scale *= args.city_reward_decay
        env.dense_reward_scale *= args.dense_reward_decay

        # Decay learning rate
        for pg in optimizer.param_groups:
            pg["lr"] *= args.lr_decay
        if capture_optimizer is not None:
            for pg in capture_optimizer.param_groups:
                pg["lr"] *= args.lr_decay

        # Decay entropy weight
        rl_criterion.entropy_weight = max(
            rl_criterion.entropy_weight * args.entropy_weight_decay,
            args.entropy_weight_min,
        )

        elapsed = time.time() - t0

        # ---- 5. Aggregate stats across all GPUs ----
        total_games = args.episodes_per_gpu * world_size
        if dist.is_initialized():
            rw_keys = ["r_invalid", "r_city", "r_dense", "r_winlose",
                       "c_cap", "c_bash", "c_steal", "c_loss"]
            local_stats = torch.tensor([
                float(local_p0),
                float(local_p1),
                float(local_draw),
                local_ep_reward,
                local_ep_len,
            ] + [reward_totals[k] for k in rw_keys], device=device)
            dist.all_reduce(local_stats)
            global_p0 = int(local_stats[0].item())
            global_p1 = int(local_stats[1].item())
            global_draw = int(local_stats[2].item())
            global_avg_r = local_stats[3].item() / world_size
            global_avg_len = local_stats[4].item() / world_size
            for i, k in enumerate(rw_keys):
                reward_totals[k] = local_stats[5 + i].item() / world_size
        else:
            global_p0 = local_p0
            global_p1 = local_p1
            global_draw = local_draw
            global_avg_r = local_ep_reward
            global_avg_len = local_ep_len

        # ---- 6. Update opponent pool (conditional on winrate + min interval) ----
        winrate_history.append(global_p0 / max(global_p0 + global_p1 + global_draw, 1))
        iters_since_update = iteration - last_pool_update_iter
        if iters_since_update >= args.pool_update_interval:
            recent_wr = sum(winrate_history) / len(winrate_history) if winrate_history else 0
            if recent_wr >= args.pool_update_winrate:
                raw = model
                opponent_pool.add(raw.state_dict())
                last_pool_update_iter = iteration
                if rank0:
                    print(f"  Pool updated: {len(opponent_pool)} opponents (winrate={recent_wr:.2f})")
            elif rank0:
                print(f"  Pool update skipped: winrate={recent_wr:.2f} < {args.pool_update_winrate}")

        # ---- 6b. Fixed-opponent evaluation vs the KL anchor ----
        # Progress signal independent of shaping rewards and of the moving
        # opponent pool: winrate here separates "actually stronger" from
        # "better at farming shaped reward".
        if (args.eval_interval > 0 and ref_model is not None
                and iteration % args.eval_interval == 0):
            ev = {0: 0, 1: 0, None: 0}
            for _ in range(args.eval_games):
                collect_selfplay_episode(
                    model, ref_model, env, device,
                    max_steps=args.rollout_steps,
                    gamma=args.gamma, gae_lambda=args.gae_lambda,
                    act_steps=args.act_steps,
                )
                ev[env.winner if env.winner in (0, 1) else None] += 1
            if dist.is_initialized():
                ev_t = torch.tensor(
                    [ev[0], ev[1], ev[None]], dtype=torch.float32, device=device)
                dist.all_reduce(ev_t)
                ev = {0: int(ev_t[0]), 1: int(ev_t[1]), None: int(ev_t[2])}
            if rank0:
                n_ev = max(ev[0] + ev[1] + ev[None], 1)
                print(f"  Eval vs anchor: {ev[0]}/{ev[1]}/{ev[None]} "
                      f"(W/L/D of {n_ev})  wr={ev[0] / n_ev:.2f}")

        # ---- 7. Logging ----
        if rank0 and iteration % args.log_interval == 0:
            print(
                f"Iter {iteration:>6d} | "
                f"ep_len={global_avg_len:>6.0f}  "
                f"r={global_avg_r:+.3f}  "
                f"p0/p1/draw={global_p0}/{global_p1}/{global_draw}({total_games}) | "
                f"pol={ppo_avg.get('policy_loss', 0):.4f}  "
                f"val={ppo_avg.get('value_loss', 0):.4f}  "
                f"ent={ppo_avg.get('entropy', 0):.4f}  "
                f"kl={ppo_avg.get('approx_kl', 0):.4f}  "
                f"clip={ppo_avg.get('clip_frac', 0):.3f}  "
                f"kl_reg={ppo_avg.get('kl_reg', 0):.4f}  "
                f"inv={ppo_avg.get('inv_loss', 0):.4f}  "
                f"e_p={ppo_avg.get('ent_pass', 0):.3f}  "
                f"e_s={ppo_avg.get('ent_src', 0):.3f}  "
                f"e_d={ppo_avg.get('ent_dir', 0):.3f}  "
                f"e_sp={ppo_avg.get('ent_spl', 0):.3f} | "
                f"cap={ppo_avg.get('cap_loss', 0):.3f}/"
                f"{ppo_avg.get('cap_acc', 0):.2f} | "
                f"il={il_metrics.get('il_loss', 0):.4f}  "
                f"il_w={il_weight:.4f} | "
                f"pool={len(opponent_pool)}  "
                f"t={elapsed:.1f}s\n"
                f"    rw: inv={reward_totals['r_invalid']:.3f}  "
                f"city={reward_totals['r_city']:.3f}  "
                f"dense={reward_totals['r_dense']:.3f}  "
                f"wl={reward_totals['r_winlose']:.2f} | "
                f"city/game: cap={reward_totals['c_cap']:.2f}  "
                f"bash={reward_totals['c_bash']:.2f}  "
                # Extra hits per capture — direct measure of garrison-number
                # judgement (0 = every city taken in one blow).
                f"fail={reward_totals['c_bash'] / max(reward_totals['c_bash'] + reward_totals['c_cap'], 1e-6):.2f}  "
                f"steal={reward_totals['c_steal']:.2f}  "
                f"loss={reward_totals['c_loss']:.2f}"
            )

        # ---- 7. Save game replay ----
        if replay_log is not None:
            import json as _json
            replay_path = os.path.join(
                args.output_dir, f"replay_iter{iteration}.json",
            )
            snap = replay_env_snapshot
            _json.dump({
                "iteration": iteration,
                "ep_len": len(ep_buffers[0]),
                "total_reward_p0": sum(ep_buffers[0].rewards),
                "winner": snap["winner"],
                "map_pool_idx": snap["map_pool_idx"],
                "replay_id": snap["replay_id"],
                "map": {
                    "real_h": snap["real_h"],
                    "real_w": snap["real_w"],
                    "generals": snap["generals"],
                    "cities": snap["cities"],
                    "mountains": snap["mountains"],
                },
                "steps": replay_log,
            }, open(replay_path, "w"), indent=2)
            print(f"  Saved replay {replay_path}")

        # ---- 8. Save checkpoint ----
        if rank0 and iteration % args.save_interval == 0:
            raw = model
            ckpt_path = os.path.join(
                args.output_dir, f"rl_checkpoint_iter{iteration}.pt",
            )
            ckpt_data = {
                "iteration": iteration,
                "model_state": raw.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.__dict__,
                "il_weight": il_weight,
                "wins": wins,
                "opponent_pool": opponent_pool.pool,
                # Shaping decay state — lets a resume continue the schedule
                # exactly instead of recomputing from the iteration counter.
                "city_reward_scale": env.city_reward_scale,
                "dense_reward_scale": env.dense_reward_scale,
            }
            if capture_head is not None:
                ckpt_data["capture_head_state"] = capture_head.state_dict()
                ckpt_data["capture_optim_state"] = capture_optimizer.state_dict()
            torch.save(ckpt_data, ckpt_path + ".tmp")
            # Atomic publish: a reader (or a resume glob) must never see a
            # half-written checkpoint.
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print(f"  Saved {ckpt_path}")

    if rank0:
        print("RL training complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()