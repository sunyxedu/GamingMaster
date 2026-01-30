#!/usr/bin/env python3
"""
Training for generals.io AI.

Two-phase training:
1. Imitation Learning (IL): Supervised learning from human replays
2. Reinforcement Learning (RL): Self-play with direct policy sampling and V-trace

Features:
- Direct policy sampling for action selection
- Actor-Critic with V-trace off-policy correction
- Entropy bonus for exploration
- Optional KL penalty to stay close to supervised policy

Usage:
    # IL only (requires replay data)
    uv run python train_alphastar.py --il_epochs 10 --rl_iterations 0 --replay_dir data/replays

    # RL only
    uv run python train_alphastar.py --il_epochs 0 --rl_iterations 100 --games_per_iter 5

    # Full IL + RL
    uv run python train_alphastar.py --il_epochs 5 --rl_iterations 100 --replay_dir data/replays
"""

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.config import ModelConfig, TrainingConfig
from model.model import GeneralsModel
from training.ema import EMA


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training (IL + RL)"
    )

    # Training phases
    parser.add_argument("--il_epochs", type=int, default=5,
                        help="Number of imitation learning epochs (0 to skip)")
    parser.add_argument("--rl_iterations", type=int, default=100,
                        help="Number of RL iterations (0 to skip)")

    # Data
    parser.add_argument("--replay_dir", type=str, default="data/replays",
                        help="Directory containing replay JSON files for IL")

    # IL settings
    parser.add_argument("--il_batch_size", type=int, default=32,
                        help="Batch size for imitation learning")
    parser.add_argument("--il_lr", type=float, default=1e-4,
                        help="Learning rate for IL")
    parser.add_argument("--max_seq_len", type=int, default=100,
                        help="Maximum sequence length for IL")

    # RL settings
    parser.add_argument("--games_per_iter", type=int, default=10,
                        help="Self-play games per RL iteration")
    parser.add_argument("--train_steps_per_iter", type=int, default=50,
                        help="Training steps per RL iteration")
    parser.add_argument("--rl_batch_size", type=int, default=64,
                        help="Batch size for RL")
    parser.add_argument("--rl_lr", type=float, default=1e-5,
                        help="Learning rate for RL (typically lower than IL)")
    parser.add_argument("--rl_seq_len", type=int, default=16,
                        help="Sequence length for RL training")

    # V-trace and Actor-Critic settings
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--rho_bar", type=float, default=1.0,
                        help="V-trace rho clipping threshold")
    parser.add_argument("--c_bar", type=float, default=1.0,
                        help="V-trace c clipping threshold")
    parser.add_argument("--entropy_weight", type=float, default=0.01,
                        help="Entropy bonus weight for exploration")
    parser.add_argument("--value_loss_weight", type=float, default=0.5,
                        help="Value loss weight")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="KL divergence penalty weight (0 to disable)")

    # Environment
    parser.add_argument("--map_width", type=int, default=15,
                        help="Map width for self-play")
    parser.add_argument("--map_height", type=int, default=15,
                        help="Map height for self-play")
    parser.add_argument("--max_turns", type=int, default=100,
                        help="Maximum turns per game")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers for IL")

    # Checkpointing
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N iterations/epochs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    # Logging
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps/batches")

    # EMA (from TRM paper)
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate for model weights (0 to disable)")

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def train_il_epoch(
    model: GeneralsModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 10,
    ema: EMA = None
) -> dict:
    """
    Train one epoch of imitation learning.

    Args:
        model: The model to train
        dataloader: DataLoader for replay data
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        log_every: Log frequency
        ema: Optional EMA instance for weight averaging

    Returns:
        Dictionary of average metrics
    """
    from training.loss import ImitationLoss

    model.train()
    loss_fn = ImitationLoss(value_weight=0.5)

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_halting_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        obs = {k: v.to(device) for k, v in batch['obs'].items()}
        tgt = {k: v.to(device) for k, v in batch['tgt'].items()}
        meta = batch['meta']

        # Get sequence lengths for masking
        lengths = meta['lengths']
        B, T = obs['grid_seq'].shape[:2]

        # Create sequence mask
        seq_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        for i, length in enumerate(lengths):
            seq_mask[i, :length] = True

        # Forward pass
        outputs = model(
            grid_seq=obs['grid_seq'],
            globals_seq=obs['globals_seq'],
            legal_mask_seq=obs['legal_source_mask_seq'],
            tick_bucket_seq=meta['tick_bucket_seq'].to(device).long(),
            target_source_idx=tgt['src_y'][:, :, 0] * 25 + tgt['src_x'][:, :, 0]
        )

        # Prepare targets
        targets = {
            'source_idx': tgt['src_y'][:, :, 0] * 25 + tgt['src_x'][:, :, 0],
            'dir_id': tgt['dir_id'][:, :, 0],
            'ratio_id': tgt['ratio_id'][:, :, 0],
            'value': tgt['value'],
            'action_mask': tgt['count'] > 0
        }

        # Compute loss
        losses = loss_fn(outputs, targets, mask=seq_mask)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update EMA weights
        if ema is not None:
            ema.update()

        # Accumulate metrics
        total_loss += losses['total'].item()
        total_policy_loss += losses['policy'].item()
        total_value_loss += losses['value'].item()
        total_halting_loss += losses.get('halting', torch.tensor(0.0)).item()
        num_batches += 1

        # Log
        if (batch_idx + 1) % log_every == 0:
            halting_val = losses.get('halting', torch.tensor(0.0)).item()
            print(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"loss={losses['total'].item():.4f}, "
                  f"policy={losses['policy'].item():.4f}, "
                  f"value={losses['value'].item():.4f}, "
                  f"halt={halting_val:.4f}")

    return {
        'loss': total_loss / max(num_batches, 1),
        'policy_loss': total_policy_loss / max(num_batches, 1),
        'value_loss': total_value_loss / max(num_batches, 1),
        'halting_loss': total_halting_loss / max(num_batches, 1)
    }


def run_imitation_learning(
    model: GeneralsModel,
    args,
    device: torch.device
):
    """
    Run imitation learning phase.
    """
    from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences

    print("\n" + "=" * 60)
    print("Phase 1: Imitation Learning")
    print("=" * 60)

    # Check if replay directory exists
    replay_dir = Path(args.replay_dir)
    if not replay_dir.exists():
        print(f"Warning: Replay directory {replay_dir} does not exist.")
        print("Skipping IL phase. Create replay data first using data processing tools.")
        return

    # Create dataset
    print(f"Loading replay data from {replay_dir}...")
    dataset = SeqReplayDataset(
        root=str(replay_dir),
        max_sequence_length=args.max_seq_len
    )

    if len(dataset) == 0:
        print("No replay files found. Skipping IL phase.")
        return

    print(f"Loaded {len(dataset)} replay sequences")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.il_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_variable_length_sequences,
        pin_memory=(device.type == 'cuda')
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.il_lr,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.il_epochs * len(dataloader),
        eta_min=1e-6
    )

    # EMA for stable evaluation (from TRM paper)
    ema = None
    if args.ema_decay > 0:
        ema = EMA(model, decay=args.ema_decay, device=device)
        print(f"EMA enabled with decay={args.ema_decay}")

    # Training loop
    for epoch in range(1, args.il_epochs + 1):
        print(f"\n--- IL Epoch {epoch}/{args.il_epochs} ---")
        start_time = time.time()

        metrics = train_il_epoch(
            model, dataloader, optimizer, device, epoch, args.log_every, ema
        )

        scheduler.step()
        elapsed = time.time() - start_time

        print(f"Epoch {epoch} complete in {elapsed:.1f}s: "
              f"avg_loss={metrics['loss']:.4f}, "
              f"policy={metrics['policy_loss']:.4f}, "
              f"value={metrics['value_loss']:.4f}, "
              f"halt={metrics['halting_loss']:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"model_il_epoch_{epoch}.pt"
            )
            checkpoint_state = {
                'epoch': epoch,
                'phase': 'IL',
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            if ema is not None:
                checkpoint_state['ema_state_dict'] = ema.state_dict()
            torch.save(checkpoint_state, ckpt_path)
            print(f"Saved: {ckpt_path}")


def run_reinforcement_learning(
    model: GeneralsModel,
    args,
    device: torch.device
):
    """
    Run reinforcement learning phase.

    Uses direct policy sampling with V-trace off-policy correction.
    """
    from training.rl_trainer import RLTrainer

    print("\n" + "=" * 60)
    print("Phase 2: Reinforcement Learning")
    print("=" * 60)
    print("  Method: Direct policy sampling with V-trace")
    print(f"  Entropy weight: {args.entropy_weight}")
    print(f"  KL penalty weight: {args.kl_weight}")

    # Create trainer config
    training_config = TrainingConfig(
        learning_rate=args.rl_lr,
        batch_size=args.rl_batch_size
    )

    # Create RL trainer
    trainer = RLTrainer(
        model=model,
        config=training_config,
        device=device,
        map_size=(args.map_width, args.map_height),
        max_turns=args.max_turns,
        gamma=args.gamma,
        rho_bar=args.rho_bar,
        c_bar=args.c_bar,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        kl_weight=args.kl_weight,
        ema_decay=args.ema_decay
    )

    # RL training loop
    for iteration in range(1, args.rl_iterations + 1):
        print(f"\n--- RL Iteration {iteration}/{args.rl_iterations} ---")
        start_time = time.time()

        # Self-play phase (direct policy sampling)
        print(f"  Self-play: {args.games_per_iter} games...")
        game_stats = trainer.self_play(num_games=args.games_per_iter)

        # Report game statistics
        wins = sum(1 for s in game_stats if s['winner'] == 0)
        avg_turns = sum(s['turns'] for s in game_stats) / len(game_stats) if game_stats else 0
        print(f"  Games played: {len(game_stats)}, P0 wins: {wins}, Avg turns: {avg_turns:.1f}")
        print(f"  Trajectory buffer size: {len(trainer.trajectory_buffer)}")

        # Training phase
        if len(trainer.trajectory_buffer) >= 1:
            print(f"  Training: {args.train_steps_per_iter} steps...")
            losses = trainer.train_from_buffer(
                num_steps=args.train_steps_per_iter,
                batch_size=args.rl_batch_size,
                seq_len=args.rl_seq_len,
                log_interval=args.log_every
            )

            if losses:
                avg_loss = sum(l['total'] for l in losses) / len(losses)
                avg_policy = sum(l['policy'] for l in losses) / len(losses)
                avg_value = sum(l['value'] for l in losses) / len(losses)
                avg_entropy = sum(l['entropy'] for l in losses) / len(losses)
                print(f"  Avg loss: total={avg_loss:.4f}, policy={avg_policy:.4f}, "
                      f"value={avg_value:.4f}, entropy={avg_entropy:.4f}")
        else:
            print(f"  No trajectories yet, skipping training")

        elapsed = time.time() - start_time
        print(f"  Iteration {iteration} complete in {elapsed:.1f}s")

        # Save checkpoint
        if iteration % args.save_every == 0:
            ckpt_path = os.path.join(
                args.output_dir, f"model_rl_iter_{iteration}.pt"
            )
            trainer.save_checkpoint(ckpt_path, extra_state={
                'iteration': iteration,
                'phase': 'RL',
                'args': vars(args)
            })
            print(f"  Saved: {ckpt_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Training for Generals.io")
    print("=" * 60)

    # Setup device
    device = get_device(args.device)
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    model_config = ModelConfig()
    model = GeneralsModel(model_config)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded from {ckpt.get('phase', 'unknown')} phase")

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print training plan
    print(f"\nTraining Plan:")
    print(f"  IL epochs: {args.il_epochs}")
    print(f"  RL iterations: {args.rl_iterations}")
    print(f"  EMA decay: {args.ema_decay}")
    if args.il_epochs > 0:
        print(f"  Replay dir: {args.replay_dir}")
    if args.rl_iterations > 0:
        print(f"  Games per iter: {args.games_per_iter}")
        print(f"  Method: Direct policy sampling")
        print(f"  V-trace: gamma={args.gamma}, rho_bar={args.rho_bar}")

    # Phase 1: Imitation Learning
    if args.il_epochs > 0:
        run_imitation_learning(model, args, device)

    # Phase 2: Reinforcement Learning
    if args.rl_iterations > 0:
        run_reinforcement_learning(model, args, device)

    # Save final model
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_config.__dict__
    }, final_path)
    print(f"\nSaved final model: {final_path}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
