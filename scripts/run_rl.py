#!/usr/bin/env python3
"""
Reinforcement Learning Training Script.

Uses direct policy sampling with V-trace off-policy correction.

Usage:
    uv run python scripts/run_rl.py
    uv run python scripts/run_rl.py --iterations 100 --games 10
    uv run python scripts/run_rl.py --checkpoint checkpoints/il/model_il_epoch_5.pt
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="RL Training (direct policy sampling)"
    )

    # Training parameters
    parser.add_argument("--iterations", type=int, default=20,
                        help="Number of RL iterations")
    parser.add_argument("--games", type=int, default=5,
                        help="Self-play games per iteration")
    parser.add_argument("--train_steps", type=int, default=20,
                        help="Training steps per iteration")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--seq_len", type=int, default=16,
                        help="Sequence length for training")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")

    # Environment
    parser.add_argument("--map_size", type=int, default=10,
                        help="Map size (width and height)")
    parser.add_argument("--max_turns", type=int, default=50,
                        help="Max turns per game")

    # V-trace and Actor-Critic
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--rho_bar", type=float, default=1.0,
                        help="V-trace rho clipping")
    parser.add_argument("--c_bar", type=float, default=1.0,
                        help="V-trace c clipping")
    parser.add_argument("--entropy_weight", type=float, default=0.01,
                        help="Entropy bonus weight")
    parser.add_argument("--value_loss_weight", type=float, default=0.5,
                        help="Value loss weight")
    parser.add_argument("--kl_weight", type=float, default=0.0,
                        help="KL penalty weight (0 to disable)")

    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rl",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, cuda, mps")

    return parser.parse_args()


def get_device(device_str: str):
    import torch
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def main():
    args = parse_args()

    import torch
    from model.config import ModelConfig, TrainingConfig
    from model.model import GeneralsModel
    from training.rl_trainer import RLTrainer

    print("=" * 60)
    print("Reinforcement Learning Training")
    print("=" * 60)
    print("Method: Direct policy sampling with V-trace")
    print()
    print("Parameters:")
    print(f"  Iterations:       {args.iterations}")
    print(f"  Games/iter:       {args.games}")
    print(f"  Train steps/iter: {args.train_steps}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Sequence length:  {args.seq_len}")
    print(f"  Learning rate:    {args.lr}")
    print(f"  Map size:         {args.map_size}x{args.map_size}")
    print(f"  Max turns:        {args.max_turns}")
    print()
    print("V-trace settings:")
    print(f"  Gamma:            {args.gamma}")
    print(f"  Entropy weight:   {args.entropy_weight}")
    print(f"  KL weight:        {args.kl_weight}")
    print()
    print(f"  Output dir:       {args.output_dir}")
    print(f"  Checkpoint:       {args.checkpoint or '(none)'}")
    print("=" * 60)

    # Device
    device = get_device(args.device)
    print(f"\nDevice: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    model_config = ModelConfig()
    model = GeneralsModel(model_config)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training config
    training_config = TrainingConfig(
        learning_rate=args.lr,
        batch_size=args.batch_size
    )

    # Create trainer
    trainer = RLTrainer(
        model=model,
        config=training_config,
        device=device,
        map_size=(args.map_size, args.map_size),
        max_turns=args.max_turns,
        gamma=args.gamma,
        rho_bar=args.rho_bar,
        c_bar=args.c_bar,
        value_loss_weight=args.value_loss_weight,
        entropy_weight=args.entropy_weight,
        kl_weight=args.kl_weight
    )

    # Training loop
    print("\nStarting training...")
    print()

    for iteration in range(1, args.iterations + 1):
        print(f"--- Iteration {iteration}/{args.iterations} ---")

        # Self-play
        print(f"  Self-play: {args.games} games...")
        game_stats = trainer.self_play(num_games=args.games)

        wins = sum(1 for s in game_stats if s['winner'] == 0)
        avg_turns = sum(s['turns'] for s in game_stats) / len(game_stats) if game_stats else 0
        print(f"  Games: {len(game_stats)}, P0 wins: {wins}, Avg turns: {avg_turns:.1f}")
        print(f"  Buffer size: {len(trainer.trajectory_buffer)}")

        # Training
        if len(trainer.trajectory_buffer) >= 1:
            print(f"  Training: {args.train_steps} steps...")
            losses = trainer.train_from_buffer(
                num_steps=args.train_steps,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                log_interval=args.log_every
            )

            if losses:
                avg_loss = sum(l['total'] for l in losses) / len(losses)
                avg_policy = sum(l['policy'] for l in losses) / len(losses)
                avg_value = sum(l['value'] for l in losses) / len(losses)
                avg_entropy = sum(l['entropy'] for l in losses) / len(losses)
                print(f"  Loss: total={avg_loss:.4f}, policy={avg_policy:.4f}, "
                      f"value={avg_value:.4f}, entropy={avg_entropy:.4f}")

        # Save checkpoint
        if iteration % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_rl_iter_{iteration}.pt")
            trainer.save_checkpoint(ckpt_path, extra_state={
                'iteration': iteration,
                'args': vars(args)
            })
            print(f"  Saved: {ckpt_path}")

        print()

    # Save final model
    final_path = os.path.join(args.output_dir, "model_final.pt")
    trainer.save_checkpoint(final_path)
    print(f"Saved final model: {final_path}")

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
