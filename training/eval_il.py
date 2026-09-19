"""
Validation-set evaluation for IL checkpoints (generals.io, training/train_il.py lineage).

Runs each checkpoint over held-out *-val replays with NO teacher forcing and
full ACT depth, reporting the same loss parts and accuracies the training
loop logs, so numbers are directly comparable across epochs.

Usage:
    python3 -m training.eval_il \
        --checkpoints ckpts/checkpoint_epoch*.pt \
        --data_dir data/PlayerA-val data/PlayerB-val \
        [--csv results.csv]
"""

import argparse
import csv
import os

import torch
from torch.utils.data import ConcatDataset, DataLoader

from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from model.config import ModelConfig
from model.losses import ImitationLoss, next_obs_loss
from model.network import GeneralsNetwork
from model.rl_utils import load_model_state

PART_KEYS = ["source", "direction", "split", "pass", "value", "q_halt", "wm"]
ACC_KEYS = ["src_acc", "dir_acc", "full_acc", "complete_acc"]


def parse_args():
    p = argparse.ArgumentParser(description="IL validation evaluation")
    p.add_argument("--checkpoints", type=str, nargs="+", required=True,
                   help="Checkpoint .pt files to evaluate (shell glob ok)")
    p.add_argument("--data_dir", type=str, nargs="+", required=True,
                   help="Directory(s) with held-out .json replays")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Eval is grad-free; can run larger than training micro-batch")
    p.add_argument("--max_seq_len", type=int, default=1000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--focal_gamma", type=float, default=1.5,
                   help="Match the training run's --focal_gamma")
    p.add_argument("--world_model_weight", type=float, default=0.2,
                   help="Match the training run's --world_model_weight so the "
                        "total loss is comparable (0 = skip wm loss)")
    p.add_argument("--act_steps", type=int, default=0,
                   help="ACT pondering depth (0 = checkpoint's halt_max_steps)")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--csv", type=str, default=None, help="Append per-checkpoint rows here")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, wm_weight, act_steps):
    model.eval()
    use_amp = device.type == "cuda"
    sums = {"loss": 0.0, **{k: 0.0 for k in PART_KEYS}, **{k: 0.0 for k in ACC_KEYS}}
    n_batches = 0
    for batch in loader:
        grid_seq = batch["obs"]["grid_seq"].to(device)
        globals_seq = batch["obs"]["globals_seq"].to(device)
        legal_mask = batch["obs"]["legal_source_mask_seq"].to(device)
        tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
        lengths = batch["meta"]["lengths"]
        frame_mask = (
            torch.arange(grid_seq.shape[1], device=device)[None, :]
            < torch.as_tensor(lengths, device=device)[:, None]
        )
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred = model(
                grid_seq, globals_seq,
                legal_mask=legal_mask,
                source_y=None, source_x=None,  # no teacher forcing at eval
                frame_mask=frame_mask,
                act_steps=act_steps,
            )
            losses = criterion(pred, tgt, lengths)
            if wm_weight > 0 and "next_state_logits" in pred:
                wm = next_obs_loss(pred, grid_seq, frame_mask)
                wm_total = wm["state_loss"] + wm["army_loss"]
                losses["loss"] = losses["loss"] + wm_weight * wm_total
                losses["wm_loss"] = wm_total.detach()
            else:
                losses["wm_loss"] = losses["loss"].new_tensor(0.0)
        if not torch.isfinite(losses["loss"]):
            continue
        sums["loss"] += losses["loss"].item()
        for k in PART_KEYS:
            sums[k] += losses[f"{k}_loss"].item()
        for k in ACC_KEYS:
            sums[k] += losses[k].item()
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}, n_batches


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    dataset = ConcatDataset([
        SeqReplayDataset(d, max_sequence_length=args.max_seq_len)
        for d in args.data_dir
    ])
    print(f"Val dataset: {len(dataset)} replays from {len(args.data_dir)} dir(s)")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length_sequences,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = None
    prev_cfg = None
    criterion = None
    results = []
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                             if k in ModelConfig.__dataclass_fields__})
        if model is None or cfg != prev_cfg:
            model = GeneralsNetwork(cfg).to(device)
            criterion = ImitationLoss(grid_w=cfg.grid_w, focal_gamma=args.focal_gamma)
            prev_cfg = cfg
        load_model_state(model, ckpt["model_state"], ctx=f"eval {path}")
        act_steps = args.act_steps or cfg.halt_max_steps

        import time
        t0 = time.time()
        stats, n_batches = evaluate(model, loader, criterion, device,
                                    args.world_model_weight, act_steps)
        name = os.path.basename(path)
        print(
            f"{name}  loss={stats['loss']:.4f}  ({n_batches} batches, {time.time()-t0:.0f}s)\n"
            f"    src={stats['source']:.4f}  dir={stats['direction']:.4f}  "
            f"split={stats['split']:.4f}  pass={stats['pass']:.4f}  "
            f"val={stats['value']:.4f}  qhalt={stats['q_halt']:.4f}  wm={stats['wm']:.4f}\n"
            f"    acc: src={stats['src_acc']:.1%}  dir={stats['dir_acc']:.1%}  "
            f"full={stats['full_acc']:.1%}  complete={stats['complete_acc']:.1%}",
            flush=True,
        )
        results.append((name, stats))
        if args.csv:
            write_header = not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(["checkpoint", "loss", *PART_KEYS, *ACC_KEYS])
                w.writerow([name, f"{stats['loss']:.6f}",
                            *(f"{stats[k]:.6f}" for k in PART_KEYS),
                            *(f"{stats[k]:.6f}" for k in ACC_KEYS)])

    if len(results) > 1:
        best_loss = min(results, key=lambda r: r[1]["loss"])
        best_acc = max(results, key=lambda r: r[1]["full_acc"])
        print(f"\nBest val loss:     {best_loss[0]}  ({best_loss[1]['loss']:.4f})")
        print(f"Best full-move acc: {best_acc[0]}  ({best_acc[1]['full_acc']:.1%})")


if __name__ == "__main__":
    main()
