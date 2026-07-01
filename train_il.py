"""
Imitation Learning training script for the generals.io bot.

Usage:
    # Single GPU:
    python3 train_il.py --data_dir <path_to_gio_data> [options]

    # Multi-GPU (DDP):
    torchrun --nproc_per_node=2 train_il.py --data_dir <path_to_gio_data> [options]
"""

import argparse
import math
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
    p.add_argument("--data_dir", type=str, nargs="+", required=True, help="Directory(s) with .json training data")
    p.add_argument(
        "--output_dir", type=str, default="checkpoints", help="Checkpoint output dir"
    )
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=200, help="Linear LR warmup steps")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_seq_len", type=int, default=1000, help="Max sequence length per sample")
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=3)
    p.add_argument("--log_interval", type=int, default=10, help="Log every N steps")
    p.add_argument("--save_interval", type=int, default=1, help="Save every N epochs")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps within epoch")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--tf_start", type=float, default=0.0, help="Teacher forcing ratio at epoch 1")
    p.add_argument("--tf_end", type=float, default=0.0, help="Teacher forcing ratio at last epoch")
    p.add_argument("--focal_gamma", type=float, default=1.5, help="Focal loss gamma for source CE")
    p.add_argument("--reset_scheduler", action="store_true", help="Don't load scheduler state from checkpoint")
    p.add_argument(
        "--max_act_steps", type=int, default=0,
        help="Sample TRM ACT depth uniformly from [1, K] per batch so inference-time "
             "pondering states are in-distribution (0 = use model halt_max_steps, 1 = off)",
    )
    return p.parse_args()


def train_epoch(model, loader, criterion, optimizer, scheduler, device, epoch, args,
                rank0=True, save_fn=None, start_step=0, tf_ratio=1.0):
    model.train()
    from collections import deque
    window = args.log_interval
    recent_loss = deque(maxlen=window)
    recent_parts = {k: deque(maxlen=window) for k in
                    ["source", "direction", "split", "pass", "value", "q_halt"]}
    recent_acc = {k: deque(maxlen=window) for k in ["src_acc", "dir_acc", "full_acc", "complete_acc"]}
    epoch_loss_sum = 0.0
    epoch_parts_sum = {k: 0.0 for k in
                       ["source", "direction", "split", "pass", "value", "q_halt"]}
    epoch_acc_sum = {k: 0.0 for k in ["src_acc", "dir_acc", "full_acc", "complete_acc"]}
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
        # True for real frames — keeps zero-padded frames out of BatchNorm stats
        frame_mask = (
            torch.arange(grid_seq.shape[1], device=device)[None, :]
            < torch.as_tensor(lengths, device=device)[:, None]
        )

        source_y = tgt["src_y"][:, :, 0]
        source_x = tgt["src_x"][:, :, 0]

        # Per-timestep teacher forcing: each timestep independently decides
        # whether to use ground-truth source or let the model sample its own.
        move_mask = tgt["count"] > 0  # [B, T]
        if tf_ratio > 0:
            tf_mask = torch.rand_like(source_y, dtype=torch.float32) < tf_ratio  # [B, T]
            # Use GT source where both: (1) TF coin says yes, (2) there's a move
            use_gt = tf_mask & move_mask
            source_y = torch.where(use_gt, source_y, torch.full_like(source_y, -1))
            source_x = torch.where(use_gt, source_x, torch.full_like(source_x, -1))
        else:
            use_gt = None
            source_y = None
            source_x = None

        # Deep-supervision-style ACT: train at a random pondering depth so
        # carry states reached by the inference ACT loop are in-distribution.
        act_steps = random.randint(1, args.max_act_steps) if args.max_act_steps > 1 else 1

        try:
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(
                    grid_seq, globals_seq,
                    legal_mask=legal_mask,
                    source_y=source_y,
                    source_x=source_x,
                    frame_mask=frame_mask,
                    act_steps=act_steps,
                )
                losses = criterion(pred, tgt, lengths, tf_mask=use_gt)
            loss = losses["loss"]

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            else:
                grad_norm = torch.tensor(0.0)

            # Skip step if NaN detected — prevents weight corruption
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                if rank0:
                    T_cur = grid_seq.shape[1]
                    print(f"  [NaN] Skipped step {step+1} (T={T_cur}, loss={loss.item():.4f})")
                continue

            optimizer.step()
            scheduler.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            optimizer.zero_grad(set_to_none=True)
            if rank0:
                T_cur = grid_seq.shape[1]
                print(f"  [OOM] Skipped step {step+1} (T={T_cur})")
            continue

        T_cur = grid_seq.shape[1]
        recent_loss.append(loss.item())
        epoch_loss_sum += loss.item()
        epoch_steps += 1
        for k in recent_parts:
            v = losses[f"{k}_loss"].item()
            recent_parts[k].append(v)
            epoch_parts_sum[k] += v
        for k in recent_acc:
            v = losses[k].item()
            recent_acc[k].append(v)
            epoch_acc_sum[k] += v

        if rank0 and (step + 1) % args.log_interval == 0:
            avg = sum(recent_loss) / len(recent_loss)
            parts = {k: sum(v) / len(v) for k, v in recent_parts.items()}
            acc = {k: sum(v) / len(v) for k, v in recent_acc.items()}
            ep_avg = epoch_loss_sum / epoch_steps
            ep_parts = {k: epoch_parts_sum[k] / epoch_steps for k in epoch_parts_sum}
            ep_acc = {k: epoch_acc_sum[k] / epoch_steps for k in epoch_acc_sum}
            print(
                f"  Epoch {epoch} Step {step+1}/{len(loader)}  T={T_cur}  loss={avg:.4f}[{ep_avg:.4f}]:\n"
                f"    src={parts['source']:.4f}[{ep_parts['source']:.4f}]  "
                f"dir={parts['direction']:.4f}[{ep_parts['direction']:.4f}]  split={parts['split']:.4f}[{ep_parts['split']:.4f}]  "
                f"pass={parts['pass']:.4f}[{ep_parts['pass']:.4f}]  val={parts['value']:.4f}[{ep_parts['value']:.4f}]  "
                f"qhalt={parts['q_halt']:.4f}[{ep_parts['q_halt']:.4f}]\n"
                f"    acc: src={acc['src_acc']:.1%}[{ep_acc['src_acc']:.1%}]  "
                f"dir={acc['dir_acc']:.1%}[{ep_acc['dir_acc']:.1%}]  "
                f"full={acc['full_acc']:.1%}[{ep_acc['full_acc']:.1%}]  "
                f"complete={acc['complete_acc']:.1%}[{ep_acc['complete_acc']:.1%}]"
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
    if args.max_act_steps == 0:
        args.max_act_steps = cfg.halt_max_steps
    if rank0:
        print(f"Model config: d={cfg.d_model}, heads={cfg.n_heads}, "
              f"TRM={cfg.trm_H_cycles}H\u00d7{cfg.trm_L_cycles}L\u00d7{cfg.trm_block_layers}blk, "
              f"ACT depth 1..{args.max_act_steps}")

    model = GeneralsNetwork(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        model._set_static_graph()
    if rank0:
        print(f"Model parameters: {n_params:,}")

    criterion = ImitationLoss(grid_w=cfg.grid_w, focal_gamma=args.focal_gamma)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if len(args.data_dir) == 1:
        dataset = SeqReplayDataset(args.data_dir[0], max_sequence_length=args.max_seq_len)
    else:
        from torch.utils.data import ConcatDataset
        dataset = ConcatDataset([
            SeqReplayDataset(d, max_sequence_length=args.max_seq_len)
            for d in args.data_dir
        ])
    if rank0:
        print(f"Dataset: {len(dataset)} replays from {args.data_dir}")

    world_size = dist.get_world_size() if distributed else 1
    steps_per_epoch = len(dataset) // (args.batch_size * world_size)

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
        # RL checkpoints don't have optimizer_state / epoch / loss — handle gracefully
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0)
        if "epoch" in ckpt:
            if start_step > 0:
                start_epoch = ckpt["epoch"]
            else:
                start_epoch = ckpt["epoch"] + 1
        if rank0:
            if "epoch" in ckpt:
                msg = f"Resumed from {args.resume} (epoch {ckpt['epoch']}, loss={ckpt.get('loss', 0):.4f}"
                if start_step > 0:
                    msg += f", step {start_step}"
                print(msg + ")")
            else:
                # RL checkpoint — just report iteration if available
                it = ckpt.get("iteration", "?")
                print(f"Loaded weights from RL checkpoint {args.resume} (iteration {it})")

    # Build LR scheduler after resume so we know start_epoch
    remaining_epochs = args.epochs - start_epoch + 1
    if args.reset_scheduler or not args.resume:
        # Cosine covers only the remaining epochs
        total_steps = steps_per_epoch * remaining_epochs
    else:
        total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    # Force optimizer to use the CLI lr (load_state_dict restores old initial_lr)
    if args.reset_scheduler:
        for pg in optimizer.param_groups:
            pg["lr"] = args.lr
            pg["initial_lr"] = args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume and not args.reset_scheduler:
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])

    if rank0:
        print(f"LR schedule: {remaining_epochs} epochs, {total_steps} steps, "
              f"warmup={warmup_steps}, reset={args.reset_scheduler}")

    def save_checkpoint(epoch, step=0, loss=0.0):
        raw = model.module if distributed else model
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": raw.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg.__dict__,
            "loss": loss,
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
            model, loader, criterion, optimizer, scheduler, device, epoch, args,
            rank0=rank0, save_fn=save_checkpoint, start_step=start_step,
            tf_ratio=tf_ratio,
        )
        start_step = 0  # only skip steps for the first epoch after resume
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        if rank0:
            print(f"Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  lr={lr:.6f}  "
                  f"tf={tf_ratio:.2f}  time={elapsed:.1f}s")

        if epoch % args.save_interval == 0 and rank0:
            save_checkpoint(epoch, loss=avg_loss)

    if rank0:
        print("Training complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
