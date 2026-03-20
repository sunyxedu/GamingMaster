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
    p.add_argument("--il_data_dir", type=str, default=None,
                   help="Directory with .json IL training data (for joint training)")
    p.add_argument("--output_dir", type=str, default="checkpoints_rl")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from RL checkpoint")

    # RL hyper-parameters
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--rollout_steps", type=int, default=1000,
                   help="Max steps per self-play episode")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    # Loss weights
    p.add_argument("--value_weight", type=float, default=0.5)
    p.add_argument("--entropy_weight", type=float, default=0.03)
    p.add_argument("--il_weight", type=float, default=0.5,
                   help="Weight of IL auxiliary loss")
    p.add_argument("--il_decay", type=float, default=0.9999,
                   help="Per-iteration decay for il_weight")
    p.add_argument("--kl_weight", type=float, default=0.01,
                   help="KL regularisation against reference policy")

    # Encoder freeze schedule
    p.add_argument("--freeze_encoder_iters", type=int, default=1000,
                   help="Freeze encoder for the first N iterations (0 = never freeze)")

    # Opponent pool
    p.add_argument("--pool_size", type=int, default=10,
                   help="Max opponents in pool (excluding IL baseline)")
    p.add_argument("--pool_update_interval", type=int, default=100,
                   help="Add current model to pool every N iterations")

    # Map pool (use real replay maps instead of random)
    p.add_argument("--replay_dir", type=str, default=None,
                   help="Directory with raw replay JSON files")
    p.add_argument("--filtered_file", type=str, default=None,
                   help="Path to filtered replay list JSON")

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Model (used when no checkpoint is provided)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=3)

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--save_replay_interval", type=int, default=50,
                   help="Save a game replay every N iterations (0 = never)")
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


# ======================================================================
# Opponent pool
# ======================================================================

class OpponentPool:
    """Maintains a pool of past model checkpoints for league training.

    Index 0 is always the IL baseline (never removed).
    """

    def __init__(self, initial_state: dict, max_size: int = 10):
        self.pool: List[dict] = [copy.deepcopy(initial_state)]
        self.max_size = max_size

    def add(self, state_dict: dict) -> None:
        self.pool.append(copy.deepcopy(state_dict))
        if len(self.pool) > self.max_size + 1:  # +1 for IL baseline
            self.pool.pop(1)  # keep index 0 (IL baseline)

    def sample(self) -> dict:
        """50% latest, 50% random from pool."""
        if len(self.pool) > 1 and random.random() < 0.5:
            return random.choice(self.pool)
        return self.pool[-1]

    def __len__(self) -> int:
        return len(self.pool)


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
    caches = {0: None, 1: None}
    replay_log: Optional[List[Dict]] = [] if record else None

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
                out = net.step_rl(grid, glb, st_cache=caches[pid], step_idx=_step)

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

        # Store transition for player 0 only
        action_dict = {
            "source_idx": step_out[0]["action_source"].item(),
            "direction": step_out[0]["action_direction"].item(),
            "split": step_out[0]["action_split"].item(),
            "passed": step_out[0]["action_pass"].item(),
        }
        buffer.add(
            obs=obs[0],
            action=action_dict,
            log_prob=step_out[0]["action_log_prob"].item(),
            value=step_out[0]["value"].item(),
            reward=rewards[0],
            done=done,
        )

        obs = new_obs
        if done:
            break

    # Compute GAE. If truncated, bootstrap with value estimate.
    if not done:
        grid = obs[0]["grid"].unsqueeze(0).to(device)
        glb = obs[0]["globals"].unsqueeze(0).to(device)
        with torch.no_grad():
            last_out = model.step_rl(grid, glb, st_cache=caches[0])
        last_value = last_out["value"].item()
    else:
        last_value = 0.0
    buffer.compute_returns_and_advantages(
        gamma=gamma, lam=gae_lambda, last_value=last_value,
    )

    return buffer, replay_log


# ======================================================================
# Gradient sync (replaces DDP)
# ======================================================================

def sync_gradients(model: torch.nn.Module, world_size: int) -> None:
    """All-reduce gradients across GPUs (manual DDP replacement)."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


# ======================================================================
# PPO update
# ======================================================================

def ppo_update(
    model: GeneralsNetwork,
    ref_model: Optional[GeneralsNetwork],
    buffer: RolloutBuffer,
    rl_criterion: RLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ModelConfig,
    world_size: int = 1,
    kl_weight: float = 0.01,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one PPO update epoch on a single trajectory (no legal mask)."""
    model.train()
    batch = buffer.to_tensor_batch(device)

    grid_seq = batch["grid_seq"]        # [1, T, C, H, W]
    globals_seq = batch["globals_seq"]  # [1, T, G]
    actions = batch["actions"]
    old_lp = batch["old_log_probs"]     # [1, T]
    advantages = batch["advantages"]    # [1, T]
    returns = batch["returns"]          # [1, T]

    # Advantages are already normalized before the PPO loop (in main)

    # Forward with stored source positions (no legal mask)
    source_y = actions["source_idx"] // cfg.grid_w
    source_x = actions["source_idx"] % cfg.grid_w

    pred = model(
        grid_seq, globals_seq,
        legal_mask=None,
        source_y=source_y,
        source_x=source_x,
    )

    log_probs, entropy = model.compute_action_log_prob_rl(pred, actions)

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
            )
        kl_loss = compute_policy_kl_rl(
            pred["source_logits"],
            ref_pred["source_logits"].detach(),
        )
        total = total + kl_weight * kl_loss

    optimizer.zero_grad()
    total.backward()
    if world_size > 1:
        sync_gradients(model, world_size)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "total_loss": total.item(),
        "policy_loss": rl_losses["policy_loss"].item(),
        "value_loss": rl_losses["value_loss"].item(),
        "entropy": rl_losses["entropy"].item(),
        "approx_kl": rl_losses["approx_kl"].item(),
        "clip_frac": rl_losses["clip_fraction"].item(),
        "kl_reg": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
    }


def il_auxiliary_update(
    model: GeneralsNetwork,
    il_batch: Dict,
    il_criterion: ImitationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    il_weight: float,
    world_size: int = 1,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """One gradient step on IL expert data (auxiliary regularisation).

    IL path keeps legal_mask (model learns from expert data with mask).
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
    if args.il_checkpoint:
        ckpt = torch.load(args.il_checkpoint, map_location=device, weights_only=False)
        cfg = ModelConfig(**ckpt["config"])
        model = GeneralsNetwork(cfg).to(device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if rank0:
            print(f"Loaded IL checkpoint from {args.il_checkpoint}")
    else:
        cfg = ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            trm_H_cycles=args.trm_H_cycles,
            trm_L_cycles=args.trm_L_cycles,
            trm_block_layers=args.trm_block_layers,
        )
        model = GeneralsNetwork(cfg).to(device)
        if rank0:
            print("No IL checkpoint — training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    if rank0:
        print(f"Model parameters: {n_params:,}")

    # ---- Reference model (frozen copy for KL regularisation) -----------
    ref_model: Optional[GeneralsNetwork] = None
    if args.kl_weight > 0:
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        if rank0:
            print("Created frozen reference model for KL regularisation")

    # ---- Opponent pool -------------------------------------------------
    opponent_pool = OpponentPool(
        initial_state=model.state_dict(),
        max_size=args.pool_size,
    )
    # Opponent model (loaded with pool checkpoint each iteration)
    opponent_model = copy.deepcopy(model)
    opponent_model.eval()
    for p in opponent_model.parameters():
        p.requires_grad = False
    if rank0:
        print(f"Opponent pool initialised (max_size={args.pool_size})")

    # ---- Encoder freeze ------------------------------------------------
    if args.freeze_encoder_iters > 0:
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
    il_criterion = ImitationLoss(grid_w=cfg.grid_w)

    # ---- Optimiser -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ---- IL data loader (optional) -------------------------------------
    il_loader = None
    il_iter = None
    if args.il_data_dir:
        il_dataset = SeqReplayDataset(args.il_data_dir, max_sequence_length=200)
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
    if args.replay_dir and args.filtered_file:
        n_maps = env.load_map_pool(args.replay_dir, args.filtered_file)
        if rank0:
            print(f"Loaded {n_maps} replay maps for environment")

    # ---- Resume --------------------------------------------------------
    start_iter = 1
    il_weight = args.il_weight
    wins = {0: 0, 1: 0, "draw": 0}

    if args.resume:
        rl_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw = model
        raw.load_state_dict(rl_ckpt["model_state"])
        start_iter = rl_ckpt["iteration"] + 1
        il_weight = rl_ckpt.get("il_weight", args.il_weight)
        wins = rl_ckpt.get("wins", wins)
        if "opponent_pool" in rl_ckpt:
            opponent_pool.pool = rl_ckpt["opponent_pool"]

        # If resuming past the unfreeze point, unfreeze encoder and rebuild optimizer
        if args.freeze_encoder_iters > 0 and start_iter > args.freeze_encoder_iters:
            unfreeze_all(raw)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            if rank0:
                print(f"Resumed past unfreeze point — encoder unfrozen, LR={args.lr * 0.1}")

        optimizer.load_state_dict(rl_ckpt["optimizer_state"])

        if rank0:
            print(f"Resumed from {args.resume}, iteration {start_iter}, "
                  f"pool={len(opponent_pool)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Training loop -------------------------------------------------
    for iteration in range(start_iter, args.iterations + 1):
        t0 = time.time()

        # Unfreeze encoder after schedule
        if args.freeze_encoder_iters > 0 and iteration == args.freeze_encoder_iters + 1:
            raw = model
            unfreeze_all(raw)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,
                weight_decay=args.weight_decay,
            )
            if rank0:
                print(f"Iteration {iteration}: encoder unfrozen, LR reduced to {args.lr * 0.1}")

        # ---- 1. Load opponent from pool ----
        opp_state = opponent_pool.sample()
        opponent_model.load_state_dict(opp_state)

        # ---- 2. Collect self-play episode (each GPU independently) ----
        should_record = (
            rank0
            and args.save_replay_interval > 0
            and iteration % args.save_replay_interval == 0
        )
        buffer, replay_log = collect_selfplay_episode(
            model, opponent_model, env, device,
            max_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            record=should_record,
        )
        ep_len = len(buffer)
        ep_reward = sum(buffer.rewards)

        # Local wins tracking (for checkpoint)
        if env.winner == 0:
            wins[0] += 1
        elif env.winner == 1:
            wins[1] += 1
        else:
            wins["draw"] += 1

        # ---- 3. PPO update (DDP syncs gradients across all GPUs) ----
        # Normalize advantages once before the PPO loop (not inside ppo_update)
        if len(buffer) > 0:
            batch_for_norm = buffer.to_tensor_batch(device)
            adv = batch_for_norm["advantages"]
            if dist.is_initialized():
                local_sum = adv.sum()
                local_sq_sum = (adv ** 2).sum()
                local_count = torch.tensor(adv.numel(), dtype=torch.float32, device=device)
                stats = torch.stack([local_sum, local_sq_sum, local_count])
                dist.all_reduce(stats)
                g_mean = stats[0] / stats[2]
                g_var = (stats[1] / stats[2] - g_mean ** 2).clamp(min=1e-8)
                buffer.advantages = (buffer.advantages - g_mean.cpu()) / g_var.sqrt().cpu()
            else:
                adv_mean = buffer.advantages.mean()
                adv_std = buffer.advantages.std().clamp(min=1e-8)
                buffer.advantages = (buffer.advantages - adv_mean) / adv_std

        ppo_metrics_agg = {}
        if len(buffer) > 0:
            for _epoch in range(args.ppo_epochs):
                metrics = ppo_update(
                    model, ref_model, buffer, rl_criterion,
                    optimizer, device, cfg,
                    world_size=world_size,
                    kl_weight=args.kl_weight,
                    grad_clip=args.grad_clip,
                )
                for k, v in metrics.items():
                    ppo_metrics_agg[k] = ppo_metrics_agg.get(k, 0.0) + v

        n_updates = args.ppo_epochs if len(buffer) > 0 else 1
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
            )
            il_weight *= args.il_decay

        elapsed = time.time() - t0

        # ---- 5. Update opponent pool ----
        if iteration % args.pool_update_interval == 0:
            raw = model
            opponent_pool.add(raw.state_dict())
            if rank0:
                print(f"  Pool updated: {len(opponent_pool)} opponents")

        # ---- 6. Aggregate stats across all GPUs ----
        if dist.is_initialized():
            local_stats = torch.tensor([
                1.0 if env.winner == 0 else 0.0,
                1.0 if env.winner == 1 else 0.0,
                1.0 if env.winner is None else 0.0,
                ep_reward,
                float(ep_len),
            ], device=device)
            dist.all_reduce(local_stats)
            global_p0 = int(local_stats[0].item())
            global_p1 = int(local_stats[1].item())
            global_draw = int(local_stats[2].item())
            global_avg_r = local_stats[3].item() / world_size
            global_avg_len = local_stats[4].item() / world_size
        else:
            global_p0 = 1 if env.winner == 0 else 0
            global_p1 = 1 if env.winner == 1 else 0
            global_draw = 1 if env.winner is None else 0
            global_avg_r = ep_reward
            global_avg_len = ep_len

        # ---- 7. Logging ----
        if rank0 and iteration % args.log_interval == 0:
            print(
                f"Iter {iteration:>6d} | "
                f"ep_len={global_avg_len:>6.0f}  "
                f"r={global_avg_r:+.3f}  "
                f"p0/p1/draw={global_p0}/{global_p1}/{global_draw} | "
                f"pol={ppo_avg.get('policy_loss', 0):.4f}  "
                f"val={ppo_avg.get('value_loss', 0):.4f}  "
                f"ent={ppo_avg.get('entropy', 0):.4f}  "
                f"kl={ppo_avg.get('approx_kl', 0):.4f}  "
                f"clip={ppo_avg.get('clip_frac', 0):.3f}  "
                f"kl_reg={ppo_avg.get('kl_reg', 0):.4f} | "
                f"il={il_metrics.get('il_loss', 0):.4f}  "
                f"il_w={il_weight:.4f} | "
                f"pool={len(opponent_pool)}  "
                f"t={elapsed:.1f}s"
            )

        # ---- 7. Save game replay ----
        if replay_log is not None:
            import json as _json
            replay_path = os.path.join(
                args.output_dir, f"replay_iter{iteration}.json",
            )
            _json.dump({
                "iteration": iteration,
                "ep_len": ep_len,
                "total_reward_p0": ep_reward,
                "winner": env.winner,
                "map_pool_idx": env.current_map_idx,
                "replay_id": env.map_pool[env.current_map_idx]["replay_id"] if env.map_pool and env.current_map_idx is not None else None,
                "map": {
                    "real_h": env.real_h,
                    "real_w": env.real_w,
                    "generals": env.generals,
                    "cities": env.cities,
                    "mountains": env.mountains,
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
            torch.save({
                "iteration": iteration,
                "model_state": raw.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.__dict__,
                "il_weight": il_weight,
                "wins": wins,
                "opponent_pool": opponent_pool.pool,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    if rank0:
        print("RL training complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
