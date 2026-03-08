"""
Imitation Learning training script for the generals.io bot.

Usage:
    # Single GPU:
    python3 train_il.py --data_dir <path_to_gio_data> [options]

    # Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_il.py --data_dir <path_to_gio_data> [options]
"""

import argparse
import os
import random
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    p.add_argument("--max_seq_len", type=int, default=1500, help="Max sequence length per sample")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=2)
    p.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    p.add_argument("--save_interval", type=int, default=1, help="Save every N epochs")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps within epoch")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--tf_start", type=float, default=0.0, help="Teacher forcing ratio at epoch 1")
    p.add_argument("--tf_end", type=float, default=0.0, help="Teacher forcing ratio at last epoch")
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, device, epoch, args,
                rank0=True, save_fn=None, start_step=0, tf_ratio=1.0):
    model.train()
    from collections import deque
    window = args.log_interval
    recent_loss = deque(maxlen=window)
    recent_parts = {k: deque(maxlen=window) for k in
                    ["source", "direction", "split", "pass", "value", "q_halt", "move_match"]}
    epoch_loss_sum = 0.0
    epoch_steps = 0
    use_amp = device.type == "cuda"

    for step, batch in enumerate(loader):
        if step < start_step:
            continue

        grid_seq = batch["obs"]["grid_seq"].to(device)
        globals_seq = batch["obs"]["globals_seq"].to(device)
        legal_mask = batch["obs"]["legal_source_mask_seq"].to(device)

        tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
        lengths = batch["meta"]["lengths"]

        source_y = tgt["src_y"][:, :, 0]
        source_x = tgt["src_x"][:, :, 0]

        use_tf = random.random() < tf_ratio
        if use_tf:
            # Mark pass timesteps (count=0) with -1 so model samples its own source
            move_mask = tgt["count"] > 0
            source_y = torch.where(move_mask, source_y, torch.full_like(source_y, -1))
            source_x = torch.where(move_mask, source_x, torch.full_like(source_x, -1))
        else:
            source_y = None
            source_x = None

        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
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
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if rank0:
                T_cur = grid_seq.shape[1]
                print(f"  [OOM] Skipped step {step+1} (T={T_cur})")
            continue

        recent_loss.append(loss.item())
        epoch_loss_sum += loss.item()
        epoch_steps += 1
        for k in recent_parts:
            recent_parts[k].append(losses[f"{k}_loss"].item())

        if rank0 and (step + 1) % args.log_interval == 0:
            avg = sum(recent_loss) / len(recent_loss)
            parts = {k: sum(v) / len(v) for k, v in recent_parts.items()}
            print(
                f"  Epoch {epoch} Step {step+1}/{len(loader)}  "
                f"loss={avg:.4f}  src={parts['source']:.4f}  "
                f"dir={parts['direction']:.4f}  split={parts['split']:.4f}  "
                f"pass={parts['pass']:.4f}  val={parts['value']:.4f}  "
                f"qhalt={parts['q_halt']:.4f}  mmatch={parts['move_match']:.4f}"
            )

        if save_fn and rank0 and (step + 1) % args.save_steps == 0:
            save_fn(epoch, step + 1)

    avg_loss = epoch_loss_sum / max(epoch_steps, 1)
    return avg_loss


def main():
    args = parse_args()

    # Distributed setup (auto-detected via torchrun)
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
    if rank0:
        suffix = f" (world_size={dist.get_world_size()})" if distributed else ""
        print(f"Device: {device}{suffix}")

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        trm_H_cycles=args.trm_H_cycles,
        trm_L_cycles=args.trm_L_cycles,
        trm_block_layers=args.trm_block_layers,
    )
    if rank0:
        print(f"Model config: d={cfg.d_model}, heads={cfg.n_heads}, "
              f"TRM={cfg.trm_H_cycles}H\u00d7{cfg.trm_L_cycles}L\u00d7{cfg.trm_block_layers}blk")

    model = GeneralsNetwork(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if rank0:
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
    if rank0:
        print(f"Dataset: {len(dataset)} replays from {args.data_dir}")

    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_variable_length_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    start_epoch = 1
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if distributed else model
        raw_model.load_state_dict(ckpt["model_state"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = ckpt.get("step", 0)
        if start_step > 0:
            start_epoch = ckpt["epoch"]
        else:
            start_epoch = ckpt["epoch"] + 1
        if rank0:
            msg = f"Resumed from {args.resume} (epoch {ckpt['epoch']}, loss={ckpt['loss']:.4f}"
            if start_step > 0:
                msg += f", step {start_step}"
            print(msg + ")")

    def save_checkpoint(epoch, step=0):
        raw = model.module if distributed else model
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": raw.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg.__dict__,
            "loss": 0.0,
        }
        if step > 0:
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
        else:
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(ckpt, path)
        print(f"  Saved {path}")

    for epoch in range(start_epoch, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        # Linear decay: tf_start at epoch 1 → tf_end at last epoch
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        tf_ratio = args.tf_start + (args.tf_end - args.tf_start) * progress
        t0 = time.time()
        avg_loss = train_epoch(
            model, loader, criterion, optimizer, device, epoch, args,
            rank0=rank0, save_fn=save_checkpoint, start_step=start_step,
            tf_ratio=tf_ratio,
        )
        start_step = 0  # only skip steps for the first epoch after resume
        scheduler.step()
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        if rank0:
            print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}  "
                  f"tf={tf_ratio:.2f}  time={elapsed:.1f}s")

        if epoch % args.save_interval == 0 and rank0:
            save_checkpoint(epoch)

    if rank0:
        print("Training complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
