"""
Imitation Learning training script for the generals.io bot.

Usage:
    python3 train_il.py --data_dir <path_to_gio_data> [options]
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from model.config import ModelConfig
from model.losses import ImitationLoss
from model.network import GeneralsNetwork


def parse_args():
    p = argparse.ArgumentParser(description="IL training for generals.io bot")
    p.add_argument("--data_dir", type=str, required=True, help="Directory with .json training data")
    p.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Checkpoint output dir"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_seq_len", type=int, default=2000, help="Max sequence length per sample")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--trm_H_cycles", type=int, default=4)
    p.add_argument("--trm_L_cycles", type=int, default=4)
    p.add_argument("--trm_block_layers", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    p.add_argument("--save_interval", type=int, default=1, help="Save every N epochs")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, epoch, args):
    model.train()
    total_loss = 0.0
    total_steps = 0
    log_losses = {"source": 0, "direction": 0, "split": 0, "value": 0, "q_halt": 0}

    for step, batch in enumerate(loader):
        grid_seq = batch["obs"]["grid_seq"].to(device)
        globals_seq = batch["obs"]["globals_seq"].to(device)
        legal_mask = batch["obs"]["legal_source_mask_seq"].to(device)

        tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
        lengths = batch["meta"]["lengths"]

        source_y = tgt["src_y"][:, :, 0]
        source_x = tgt["src_x"][:, :, 0]

        pred = model(
            grid_seq, globals_seq,
            legal_mask=legal_mask,
            source_y=source_y,
            source_x=source_x,
        )

        losses = criterion(pred, tgt, lengths)
        loss = losses["loss"]

        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        total_steps += 1
        for k in log_losses:
            log_losses[k] += losses[f"{k}_loss"].item()

        if (step + 1) % args.log_interval == 0:
            avg = total_loss / total_steps
            parts = {k: v / total_steps for k, v in log_losses.items()}
            print(
                f"  Epoch {epoch} Step {step+1}/{len(loader)}  "
                f"loss={avg:.4f}  src={parts['source']:.4f}  "
                f"dir={parts['direction']:.4f}  split={parts['split']:.4f}  "
                f"val={parts['value']:.4f}  qhalt={parts['q_halt']:.4f}"
            )

    avg_loss = total_loss / max(total_steps, 1)
    return avg_loss


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        trm_H_cycles=args.trm_H_cycles,
        trm_L_cycles=args.trm_L_cycles,
        trm_block_layers=args.trm_block_layers,
    )
    print(f"Model config: d={cfg.d_model}, heads={cfg.n_heads}, "
          f"TRM={cfg.trm_H_cycles}H×{cfg.trm_L_cycles}L×{cfg.trm_block_layers}blk")

    model = GeneralsNetwork(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = ImitationLoss(grid_w=cfg.grid_w)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    dataset = SeqReplayDataset(
        args.data_dir,
        max_sequence_length=args.max_seq_len,
    )
    print(f"Dataset: {len(dataset)} replays from {args.data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        avg_loss = train_epoch(model, loader, criterion, optimizer, device, epoch, args)
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}  time={elapsed:.1f}s")

        if epoch % args.save_interval == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": cfg.__dict__,
                "loss": avg_loss,
            }
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
            torch.save(ckpt, path)
            print(f"  Saved {path}")

    print("Training complete.")


if __name__ == "__main__":
    main()
