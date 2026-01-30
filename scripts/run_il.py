#!/usr/bin/env python3
"""
Imitation Learning Training Script.

Trains the model using human replay data (supervised learning).

Usage:
    uv run python scripts/run_il.py --data_dir data/replays
    uv run python scripts/run_il.py --epochs 20 --data_dir data/my_replays/training
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Imitation Learning Training (supervised pre-training)"
    )

    # Training parameters
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_seq_len", type=int, default=50,
                        help="Maximum sequence length")

    # Data
    parser.add_argument("--data_dir", type=str, default="data/replays",
                        help="Directory containing replay JSON files")

    # Checkpointing
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--output_dir", type=str, default="checkpoints/il",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--log_every", type=int, default=5,
                        help="Log every N batches")

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


def count_files(directory: str, extension: str = ".json") -> int:
    """Count files with given extension in directory (recursive)."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(extension):
                count += 1
    return count


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, log_every):
    """Train for one epoch."""
    import torch

    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        grid_seq = batch['obs']['grid_seq'].to(device)
        globals_seq = batch['obs']['globals_seq'].to(device)
        legal_mask_seq = batch['obs']['legal_source_mask_seq'].to(device)
        tick_bucket_seq = batch['meta']['tick_bucket_seq'].to(device).long()

        # Targets
        src_y = batch['tgt']['src_y'].to(device)
        src_x = batch['tgt']['src_x'].to(device)
        dir_id = batch['tgt']['dir_id'].to(device)
        ratio_id = batch['tgt']['ratio_id'].to(device)
        value = batch['tgt']['value'].to(device)
        count = batch['tgt']['count'].to(device)

        B, T = grid_seq.shape[:2]
        H, W = grid_seq.shape[3], grid_seq.shape[4]

        # Initialize history latent
        history_latent = model.get_initial_latent(B, device)

        # Collect outputs
        all_source_logits = []
        all_dir_logits = []
        all_ratio_logits = []
        all_values = []

        # Process sequence
        for t in range(T):
            grid_t = grid_seq[:, t]
            globals_t = globals_seq[:, t]
            legal_mask_t = legal_mask_seq[:, t]
            tick_bucket_t = tick_bucket_seq[:, t]

            # Forward
            tokens = model.input_embedding(grid_t, globals_t, history_latent, tick_bucket_t)
            y, z, _, _ = model.backbone(tokens)

            # Get source index for this timestep (first action)
            source_idx_t = src_y[:, t, 0] * W + src_x[:, t, 0]

            # Policy (training mode)
            policy_out = model.policy_head.forward_training(
                y, legal_mask_t, source_idx_t, grid_t
            )

            # Value
            val = model.value_head(y + z, globals_t)

            all_source_logits.append(policy_out['source_logits'])
            all_dir_logits.append(policy_out['dir_logits'])
            all_ratio_logits.append(policy_out['ratio_logits'])
            all_values.append(val)

            # Update history
            history_latent = model.update_history_latent(history_latent, grid_t, globals_t)

        # Stack outputs
        source_logits = torch.stack(all_source_logits, dim=1)
        dir_logits = torch.stack(all_dir_logits, dim=1)
        ratio_logits = torch.stack(all_ratio_logits, dim=1)
        values = torch.stack(all_values, dim=1).squeeze(-1)

        # Prepare targets
        source_idx = src_y[:, :, 0] * W + src_x[:, :, 0]
        action_mask = count > 0

        outputs = {
            'source_logits': source_logits,
            'dir_logits': dir_logits,
            'ratio_logits': ratio_logits,
            'value': values
        }
        targets = {
            'source_idx': source_idx,
            'dir_id': dir_id[:, :, 0],
            'ratio_id': ratio_id[:, :, 0],
            'value': value,
            'action_mask': action_mask
        }

        # Compute loss
        losses = loss_fn(outputs, targets)

        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += losses['total'].item()
        total_policy_loss += losses['policy'].item()
        total_value_loss += losses['value'].item()
        num_batches += 1

        if (batch_idx + 1) % log_every == 0:
            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}]: loss={avg_loss:.4f}")

    return {
        'total': total_loss / max(num_batches, 1),
        'policy': total_policy_loss / max(num_batches, 1),
        'value': total_value_loss / max(num_batches, 1)
    }


def main():
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from model.config import ModelConfig
    from model.model import GeneralsModel
    from training.loss import ImitationLoss
    from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences

    print("=" * 60)
    print("Imitation Learning Training")
    print("=" * 60)
    print("Parameters:")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Learning rate:  {args.lr}")
    print(f"  Max seq len:    {args.max_seq_len}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Output dir:     {args.output_dir}")
    print(f"  Checkpoint:     {args.checkpoint or '(none)'}")
    print("=" * 60)

    # Check data directory
    if not os.path.exists(args.data_dir):
        print(f"\nError: Data directory '{args.data_dir}' does not exist.")
        print("\nPlease either:")
        print("  1. Prepare replay data with the data processing pipeline")
        print("  2. Specify correct path: --data_dir /path/to/processed/replays")
        sys.exit(1)

    # Count files
    file_count = count_files(args.data_dir, ".json")
    print(f"\nFound {file_count} replay files in {args.data_dir}")

    if file_count == 0:
        print("Error: No .json replay files found")
        sys.exit(1)

    # Device
    device = get_device(args.device)
    print(f"Device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset
    print("\nLoading dataset...")
    dataset = SeqReplayDataset(
        root=args.data_dir,
        max_sequence_length=args.max_seq_len,
        recursive=True
    )
    print(f"Dataset size: {len(dataset)} sequences")

    if len(dataset) == 0:
        print("Error: Dataset is empty. Check if files are in correct format (need 'grid_seq' key)")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length_sequences,
        num_workers=0
    )

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

    # Optimizer and loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader), eta_min=1e-6)
    loss_fn = ImitationLoss(value_weight=0.5)

    # Training loop
    print("\nStarting training...")
    print()

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---")
        start_time = time.time()

        metrics = train_epoch(
            model, dataloader, optimizer, loss_fn,
            device, epoch, args.log_every
        )
        scheduler.step()

        elapsed = time.time() - start_time
        print(f"  Loss: total={metrics['total']:.4f}, policy={metrics['policy']:.4f}, "
              f"value={metrics['value']:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        # Save checkpoint
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"model_il_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': metrics['total'],
                'args': vars(args)
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # Track best
        if metrics['total'] < best_loss:
            best_loss = metrics['total']
            best_path = os.path.join(args.output_dir, "model_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'loss': best_loss
            }, best_path)
            print(f"  New best model saved!")

        print()

    # Save final model
    final_path = os.path.join(args.output_dir, "model_final.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'loss': metrics['total']
    }, final_path)
    print(f"Saved final model: {final_path}")

    print()
    print("=" * 60)
    print("Imitation Learning Complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
