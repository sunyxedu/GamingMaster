"""
Reinforcement Learning training script for the generals.io bot.

Implements PPO self-play with joint IL regularisation (AlphaStar-style).

Key features:
  - IL pre-trained weights shared: same GeneralsNetwork, same parameters
  - Joint IL+RL loss with KL regularisation against frozen reference policy
  - Encoder freeze / unfreeze schedule
  - Self-play via GeneralsEnv

Usage:
    python3 train_rl.py \\
        --il_checkpoint checkpoints/checkpoint_epoch50.pt \\
        --il_data_dir data/output/training/PlayerName \\
        --iterations 10000
"""

from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import ImitationLoss, RLLoss
from model.network import GeneralsNetwork
from model.rl_utils import (
    RolloutBuffer,
    compute_policy_kl,
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
    p.add_argument("--rollout_steps", type=int, default=200,
                   help="Max steps per self-play episode")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)

    # Loss weights
    p.add_argument("--value_weight", type=float, default=0.5)
    p.add_argument("--entropy_weight", type=float, default=0.01)
    p.add_argument("--il_weight", type=float, default=0.5,
                   help="Weight of IL auxiliary loss")
    p.add_argument("--il_decay", type=float, default=0.9999,
                   help="Per-iteration decay for il_weight")
    p.add_argument("--kl_weight", type=float, default=0.01,
                   help="KL regularisation against reference policy")

    # Encoder freeze schedule
    p.add_argument("--freeze_encoder_iters", type=int, default=500,
                   help="Freeze encoder for the first N iterations (0 = never freeze)")

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Model (used when no checkpoint is provided)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=2)

    # Environment
    p.add_argument("--env_max_turns", type=int, default=500)
    p.add_argument("--dense_reward", action="store_true")

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--device", type=str, default="auto")

    return p.parse_args()


# ======================================================================
# Self-play rollout collection
# ======================================================================

def collect_selfplay_episode(
    model: GeneralsNetwork,
    env: GeneralsEnv,
    device: torch.device,
    max_steps: int = 200,
) -> Dict[int, RolloutBuffer]:
    """Run one self-play episode, collecting trajectories for both players.

    The **same model** is used for both players (weight sharing).
    """
    model.eval()
    obs = env.reset()
    buffers = {0: RolloutBuffer(), 1: RolloutBuffer()}
    caches = {0: None, 1: None}

    for _step in range(max_steps):
        step_out = {}
        actions_for_env = {}

        for pid in [0, 1]:
            grid = obs[pid]["grid"].unsqueeze(0).to(device)       # [1, C, H, W]
            glb = obs[pid]["globals"].unsqueeze(0).to(device)     # [1, G]
            lm = obs[pid]["legal_mask"].unsqueeze(0).to(device)   # [1, 1, H, W]

            with torch.no_grad():
                out = model.step(grid, glb, legal_mask=lm,
                                 st_cache=caches[pid], step_idx=_step)

            caches[pid] = out.get("st_cache")
            step_out[pid] = out

            # Check if player has any legal move and model doesn't pass
            has_legal = lm.any().item()
            did_pass = out["action_pass"].item()
            if has_legal and not did_pass:
                actions_for_env[pid] = {
                    "source_idx": out["action_source"].item(),
                    "direction": out["action_direction"].item(),
                    "split": out["action_split"].item(),
                }
            else:
                actions_for_env[pid] = None

        # Environment step
        new_obs, rewards, done, info = env.step(
            actions_for_env.get(0), actions_for_env.get(1),
        )

        # Store transitions (always store sampled actions for PPO re-evaluation)
        for pid in [0, 1]:
            action_dict = {
                "source_idx": step_out[pid]["action_source"].item(),
                "direction": step_out[pid]["action_direction"].item(),
                "split": step_out[pid]["action_split"].item(),
                "passed": step_out[pid]["action_pass"].item(),
            }
            buffers[pid].add(
                obs=obs[pid],
                action=action_dict,
                log_prob=step_out[pid]["action_log_prob"].item(),
                value=step_out[pid]["value"].item(),
                reward=rewards[pid],
                done=done,
            )

        obs = new_obs
        if done:
            break

    # Compute GAE.  If truncated (not done), bootstrap with value estimate.
    for pid in [0, 1]:
        if not done:
            grid = obs[pid]["grid"].unsqueeze(0).to(device)
            glb = obs[pid]["globals"].unsqueeze(0).to(device)
            with torch.no_grad():
                last_out = model.step(
                    grid, glb,
                    legal_mask=obs[pid]["legal_mask"].unsqueeze(0).to(device),
                    st_cache=caches[pid],
                )
            last_value = last_out["value"].item()
        else:
            last_value = 0.0
        buffers[pid].compute_returns_and_advantages(last_value=last_value)

    return buffers


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
    kl_weight: float = 0.01,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one PPO update epoch on a single trajectory."""
    model.train()
    batch = buffer.to_tensor_batch(device)

    grid_seq = batch["grid_seq"]        # [1, T, C, H, W]
    globals_seq = batch["globals_seq"]  # [1, T, G]
    legal_mask = batch["legal_mask"]    # [1, T, 1, H, W]
    actions = batch["actions"]
    old_lp = batch["old_log_probs"]     # [1, T]
    advantages = batch["advantages"]    # [1, T]
    returns = batch["returns"]          # [1, T]

    # Normalise advantages
    adv_mean = advantages.mean()
    adv_std = advantages.std().clamp(min=1e-8)
    advantages = (advantages - adv_mean) / adv_std

    # Forward with stored source positions
    source_y = actions["source_idx"] // cfg.grid_w
    source_x = actions["source_idx"] % cfg.grid_w

    pred = model(
        grid_seq, globals_seq,
        legal_mask=legal_mask,
        source_y=source_y,
        source_x=source_x,
    )

    # Compute current log-probs and entropy
    log_probs, entropy = model.compute_action_log_prob(
        pred, actions, legal_mask=legal_mask,
    )

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

    # KL regularisation against reference policy
    kl_loss = torch.tensor(0.0, device=device)
    if ref_model is not None and kl_weight > 0:
        with torch.no_grad():
            ref_pred = ref_model(
                grid_seq, globals_seq,
                legal_mask=legal_mask,
                source_y=source_y,
                source_x=source_x,
            )
        kl_loss = compute_policy_kl(
            pred["source_logits"],
            ref_pred["source_logits"].detach(),
            legal_mask=legal_mask,
        )
        total = total + kl_weight * kl_loss

    optimizer.zero_grad()
    total.backward()
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
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """One gradient step on IL expert data (auxiliary regularisation)."""
    model.train()

    grid_seq = il_batch["obs"]["grid_seq"].to(device)
    globals_seq = il_batch["obs"]["globals_seq"].to(device)
    legal_mask = il_batch["obs"]["legal_source_mask_seq"].to(device)
    tgt = {k: v.to(device) for k, v in il_batch["tgt"].items()}
    lengths = il_batch["meta"]["lengths"]

    # No teacher forcing — let the model sample its own source
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

    # Disable MHA fast path to avoid checkpoint recomputation mismatch
    torch.backends.mha.set_fastpath_enabled(False)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ---- Build or load model -------------------------------------------
    if args.il_checkpoint:
        ckpt = torch.load(args.il_checkpoint, map_location=device, weights_only=False)
        cfg = ModelConfig(**ckpt["config"])
        model = GeneralsNetwork(cfg).to(device)
        model.load_state_dict(ckpt["model_state"])
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
        print("No IL checkpoint — training from scratch")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---- Reference model (frozen copy for KL regularisation) -----------
    ref_model: Optional[GeneralsNetwork] = None
    if args.kl_weight > 0:
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print("Created frozen reference model for KL regularisation")

    # ---- Encoder freeze ------------------------------------------------
    if args.freeze_encoder_iters > 0:
        freeze_encoder(model)
        print(f"Encoder frozen for the first {args.freeze_encoder_iters} iterations")

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
        il_loader = DataLoader(
            il_dataset, batch_size=1, shuffle=True,
            collate_fn=collate_variable_length_sequences,
        )
        il_iter = iter(il_loader)
        print(f"IL dataset: {len(il_dataset)} replays from {args.il_data_dir}")

    # ---- Environment ---------------------------------------------------
    env = GeneralsEnv(
        grid_h=cfg.grid_h,
        grid_w=cfg.grid_w,
        max_turns=args.env_max_turns,
        dense_reward=args.dense_reward,
    )

    # ---- Resume --------------------------------------------------------
    start_iter = 1
    if args.resume:
        rl_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(rl_ckpt["model_state"])
        optimizer.load_state_dict(rl_ckpt["optimizer_state"])
        start_iter = rl_ckpt["iteration"] + 1
        print(f"Resumed from {args.resume}, iteration {start_iter}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Training loop -------------------------------------------------
    il_weight = args.il_weight
    wins = {0: 0, 1: 0, "draw": 0}

    for iteration in range(start_iter, args.iterations + 1):
        t0 = time.time()

        # Unfreeze encoder after schedule
        if args.freeze_encoder_iters > 0 and iteration == args.freeze_encoder_iters + 1:
            unfreeze_all(model)
            # Rebuild optimiser to include newly-unfrozen params
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr * 0.1,   # lower LR when unfreezing
                weight_decay=args.weight_decay,
            )
            print(f"Iteration {iteration}: encoder unfrozen, LR reduced to {args.lr * 0.1}")

        # ---- 1. Collect self-play episode ----
        buffers = collect_selfplay_episode(
            model, env, device, max_steps=args.rollout_steps,
        )
        ep_len = len(buffers[0])
        ep_reward_0 = sum(buffers[0].rewards)
        ep_reward_1 = sum(buffers[1].rewards)

        if env.winner is not None:
            wins[env.winner] += 1
        else:
            wins["draw"] += 1

        # ---- 2. PPO update (both players' trajectories) ----
        ppo_metrics_agg = {}
        for pid in [0, 1]:
            if len(buffers[pid]) == 0:
                continue
            for _epoch in range(args.ppo_epochs):
                metrics = ppo_update(
                    model, ref_model, buffers[pid], rl_criterion,
                    optimizer, device, cfg,
                    kl_weight=args.kl_weight,
                    grad_clip=args.grad_clip,
                )
                for k, v in metrics.items():
                    ppo_metrics_agg[k] = ppo_metrics_agg.get(k, 0.0) + v

        n_updates = args.ppo_epochs * 2
        ppo_metrics_avg = {k: v / max(n_updates, 1) for k, v in ppo_metrics_agg.items()}

        # ---- 3. IL auxiliary update (if data available) ----
        il_metrics = {}
        if il_loader is not None and il_weight > 1e-6:
            try:
                il_batch = next(il_iter)
            except StopIteration:
                il_iter = iter(il_loader)
                il_batch = next(il_iter)

            il_metrics = il_auxiliary_update(
                model, il_batch, il_criterion, optimizer, device,
                il_weight=il_weight,
                grad_clip=args.grad_clip,
            )
            il_weight *= args.il_decay  # decay IL weight

        elapsed = time.time() - t0

        # ---- 4. Logging ----
        if iteration % args.log_interval == 0:
            print(
                f"Iter {iteration:>6d} | "
                f"ep_len={ep_len:>3d}  "
                f"r0={ep_reward_0:+.2f} r1={ep_reward_1:+.2f} | "
                f"pol={ppo_metrics_avg.get('policy_loss', 0):.4f}  "
                f"val={ppo_metrics_avg.get('value_loss', 0):.4f}  "
                f"ent={ppo_metrics_avg.get('entropy', 0):.4f}  "
                f"kl={ppo_metrics_avg.get('approx_kl', 0):.4f}  "
                f"clip={ppo_metrics_avg.get('clip_frac', 0):.3f} | "
                f"il={il_metrics.get('il_loss', 0):.4f}  "
                f"il_w={il_weight:.4f} | "
                f"wins={wins[0]}/{wins[1]}/{wins['draw']}  "
                f"t={elapsed:.1f}s"
            )

        # ---- 5. Save checkpoint ----
        if iteration % args.save_interval == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"rl_checkpoint_iter{iteration}.pt",
            )
            torch.save({
                "iteration": iteration,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.__dict__,
                "il_weight": il_weight,
                "wins": wins,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    print("RL training complete.")


if __name__ == "__main__":
    main()
