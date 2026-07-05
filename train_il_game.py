"""Generic imitation-learning trainer for any registered game adapter.

Trains ``build_game_network(<game>)`` on shards produced by the
per-game converters (see data_processing/games/), using the decoder's
own ``il_loss`` plus the shared value-head MSE.

    uv run python train_il_game.py --game othello \
        --data_dir data/othello/train --epochs 10 --batch_size 8

generals keeps its dedicated ``train_il.py`` (teacher-forcing schedule,
world-model objective and cluster plumbing live there); this trainer
covers every other adapter with one code path.

Single-process on purpose: the multi-node story (manual grad allreduce,
NaN voting) stays in train_il.py until a non-generals game actually
trains at that scale.
"""

from __future__ import annotations

import argparse
import ast
import os
import random
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.games.dataset import GameShardDataset, collate_game_sequences
from model.games import build_game_network
from model.rl_utils import load_model_state, optim_param_groups

# Teacher-forcing conditioning labels per game: the tgt keys the decoder
# reads from ``labels`` in decode() (−1 → sample from its own policy).
LABEL_KEYS = {
    "chess": ("from_idx", "to_idx"),
    "othello": (),
    "gomoku": (),
    "go": (),
    "shogi": ("dropped", "from_idx", "to_idx", "piece_idx"),
    "lux": (),
    "starcraft": ("action_type", "actor_idx"),
    "hearthstone": ("action_type", "actor_idx"),
    "clash_royale": ("card_slot",),
    "mahjong": ("action_type",),
    # active_idx is state (which own slot is on the field), delivered via
    # labels because the observer-view converter stores it in tgt.
    "pokemon": ("action_type", "active_idx"),
}

# Per-game legality derived from the observation itself: some decoders
# need ``legal`` at decode time even during IL (lux's il_loss masks by
# pred["unit_present"], which decode() reads from legal["unit"]).
# Applied inside run_batch, so training and evaluate() share the mask.
LEGAL_FROM_OBS = {
    "lux": lambda obs: {"unit": obs["unit_mask"].bool()},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, choices=sorted(LABEL_KEYS))
    p.add_argument("--data_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--max_len", type=int, default=None,
                   help="random-crop sequences longer than this")
    p.add_argument("--max_act_steps", type=int, default=1,
                   help="train at a random TRM ACT depth in [1, N] per batch")
    p.add_argument("--value_weight", type=float, default=0.2)
    p.add_argument("--tf_start", type=float, default=1.0,
                   help="teacher-forcing prob at epoch 0 (conditioning labels)")
    p.add_argument("--tf_end", type=float, default=1.0)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", default="checkpoints/il_game")
    p.add_argument("--resume", default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="config overrides, e.g. d_model=256 board_h=9")
    return p.parse_args()


def parse_overrides(pairs) -> Dict:
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"--override expects KEY=VALUE, got {pair!r}")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw  # plain string (e.g. temporal_mixer=recurrent)
    return out


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_labels(
    game: str, tgt: Dict[str, torch.Tensor], tf_prob: float,
    frame_mask: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    """Teacher-forcing labels: keep the ground-truth conditioning with
    probability ``tf_prob`` per frame, else −1 (decoder self-samples)."""
    keys = [k for k in LABEL_KEYS[game] if k in tgt]
    if not keys:
        return None
    labels = {}
    if tf_prob >= 1.0:
        for k in keys:
            labels[k] = tgt[k]
        return labels
    keep = (torch.rand(frame_mask.shape, device=frame_mask.device) < tf_prob)
    for k in keys:
        labels[k] = torch.where(keep, tgt[k], torch.full_like(tgt[k], -1))
    return labels


def run_batch(
    net, game: str, batch: Dict, device: torch.device,
    value_weight: float, tf_prob: float, act_steps: int, bf16: bool,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    obs = {k: v.to(device) for k, v in batch["obs"].items()}
    tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
    frame_mask = batch["meta"]["frame_mask"].to(device)
    labels = build_labels(game, tgt, tf_prob, frame_mask)
    legal_fn = LEGAL_FROM_OBS.get(game)
    legal = legal_fn(obs) if legal_fn is not None else None

    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    with autocast:
        pred = net(obs, legal=legal, labels=labels, act_steps=act_steps,
                   frame_mask=frame_mask)
        losses = net.action_decoder.il_loss(pred, tgt, frame_mask)
        loss = losses["loss"]
        if "value" in tgt:
            mask_f = frame_mask.float()
            v_loss = (F.mse_loss(pred["value"].float(), tgt["value"].float(),
                                 reduction="none") * mask_f).sum() / mask_f.sum().clamp(min=1)
            loss = loss + value_weight * v_loss
            losses["value_loss"] = v_loss.detach()

    stats = {k: float(v) for k, v in losses.items() if k != "loss"}
    stats["loss"] = float(loss.detach())
    return loss, stats


def evaluate(net, game, loader, device, value_weight, bf16) -> Dict[str, float]:
    net.eval()
    agg: Dict[str, float] = {}
    n = 0
    with torch.no_grad():
        for batch in loader:
            _, stats = run_batch(net, game, batch, device, value_weight,
                                 tf_prob=1.0, act_steps=1, bf16=bf16)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    overrides = parse_overrides(args.override)

    net = build_game_network(args.game, **overrides).to(device)
    cfg = net.cfg
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[init] game={args.game} device={device.type} params={n_params:,}")

    dataset = GameShardDataset(args.data_dir, game=args.game, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers,
                        collate_fn=collate_game_sequences, drop_last=False)
    val_loader = None
    if args.val_dir:
        val_loader = DataLoader(
            GameShardDataset(args.val_dir, game=args.game, max_len=args.max_len),
            batch_size=args.batch_size, collate_fn=collate_game_sequences,
        )
    print(f"[data] {len(dataset)} games, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(
        optim_param_groups(net, args.weight_decay), lr=args.lr,
    )
    total_steps = max(1, args.epochs * len(loader) // args.grad_accum)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        left = max(1, total_steps - args.warmup_steps)
        return max(0.05, 1.0 - (step - args.warmup_steps) / left)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("game") != args.game:
            raise ValueError(
                f"checkpoint is for game {ckpt.get('game')!r}, not {args.game!r}"
            )
        load_model_state(net, ckpt["model_state"], ctx=args.resume)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[resume] {args.resume} → epoch {start_epoch}")

    os.makedirs(args.save_dir, exist_ok=True)
    net.train()
    opt_step = 0
    for epoch in range(start_epoch, args.epochs):
        frac = epoch / max(1, args.epochs - 1)
        tf_prob = args.tf_start + (args.tf_end - args.tf_start) * frac
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(loader):
            act_steps = (random.randint(1, args.max_act_steps)
                         if args.max_act_steps > 1 else 1)
            loss, stats = run_batch(net, args.game, batch, device,
                                    args.value_weight, tf_prob, act_steps,
                                    args.bf16)
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
            if i % args.log_every == 0:
                head = " ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items()))
                print(f"[e{epoch} b{i}/{len(loader)}] tf={tf_prob:.2f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {head}", flush=True)

        if val_loader is not None:
            val = evaluate(net, args.game, val_loader, device,
                           args.value_weight, args.bf16)
            print(f"[e{epoch} val] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(val.items())))

        ckpt_path = os.path.join(args.save_dir, f"{args.game}_epoch{epoch}.pt")
        torch.save({
            "game": args.game,
            "epoch": epoch,
            "model_state": net.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg.__dict__,
            "config_overrides": overrides,
        }, ckpt_path)
        print(f"[e{epoch}] saved {ckpt_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
