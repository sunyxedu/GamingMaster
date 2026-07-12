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
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.games.dataset import (
    GameShardDataset,
    ShardLocalitySampler,
    collate_game_sequences,
)
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

# STATE keys are facts about the position, not decisions: the decoder
# must ALWAYS receive them.  build_labels never drops these to −1 under
# the teacher-forcing schedule — pokemon's decode() would clamp −1 to
# slot 0 and condition the move/tera heads on the wrong Pokémon.
STATE_LABEL_KEYS = {
    "pokemon": ("active_idx",),
}

# Per-game legality derived from the observation itself: some decoders
# need ``legal`` at decode time even during IL (lux's il_loss masks by
# pred["unit_present"], which decode() reads from legal["unit"]; the
# starcraft actor mask keeps tf_prob<1 self-sampling off empty unit
# slots — at tf_prob=1.0 it is never consulted).
# Applied inside run_batch, so training and evaluate() share the mask.
LEGAL_FROM_OBS = {
    "lux": lambda obs: {"unit": obs["unit_mask"].bool()},
    "starcraft": lambda obs: {"actor": obs["unit_mask"].bool()},
    # entity_mask (True = slot occupied) keeps tf_prob<1 actor
    # self-sampling off empty entity slots, same as starcraft above.
    "hearthstone": lambda obs: {"actor": obs["entity_mask"].bool()},
    # move_ids is state (revealed moves of the active mon) consumed by
    # the move head at decode time; guarded for pre-move_ids shards.
    "pokemon": lambda obs: (
        {"move_ids": obs["move_ids"]} if "move_ids" in obs else None
    ),
}

# Shard-level meta facts (written by the converters) that must match the
# model config: {game: {meta_key: (config_field, ...)}}.  Fresh runs
# default the config from the data; every run hard-fails on a mismatch.
META_ARCH_KEYS = {
    "gomoku": {"board_size": ("board_h", "board_w")},
    "pokemon": {"move_buckets": ("pokemon_move_buckets",),
                "species_buckets": ("pokemon_species_buckets",)},
}

# Converter generations with semantic changes: shards older than this
# train, but against re-converted data they are a different game (chess
# v2 = rank-mirror a7f5242; v1 frames are 180°-rotated for black).
MIN_CONVERTER_VERSION = {"chess": 2}


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


def resume_arch_overrides(resume_ckpt: Dict, cli_overrides: Dict) -> Dict:
    """Architecture kwargs for build_game_network on resume.

    The checkpoint's full saved config wins — rebuilding from saved
    *overrides* + current registry defaults would silently reshape the
    net whenever a registry default drifts between save and resume.
    Explicit CLI overrides still apply on top (loudly): they are the
    escape hatch when the saved config itself must be corrected.
    """
    arch = resume_ckpt.get("config")
    if arch is None:
        arch = resume_ckpt.get("config_overrides", {})
        print(f"[warn] checkpoint predates full-config saving; rebuilding "
              f"from saved overrides + current defaults: {arch}")
    arch = dict(arch)
    if cli_overrides:
        changed = {k: v for k, v in cli_overrides.items() if arch.get(k) != v}
        if changed:
            print(f"[warn] --override replaces checkpoint arch values: {changed}")
        arch.update(cli_overrides)
    return arch


def check_meta_arch(game: str, cfg, shard_meta: Dict) -> None:
    """Hard-fail when shard-level converter meta contradicts the model
    config (e.g. 20×20 gomoku shards vs a 15×15 net)."""
    for meta_key, cfg_keys in META_ARCH_KEYS.get(game, {}).items():
        if meta_key not in shard_meta:
            continue
        for ck in cfg_keys:
            if getattr(cfg, ck) != shard_meta[meta_key]:
                raise ValueError(
                    f"shards were converted with {meta_key}="
                    f"{shard_meta[meta_key]} but the model has "
                    f"{ck}={getattr(cfg, ck)}; re-convert the data or "
                    f"use a matching checkpoint/--override"
                )


_UNEXPECTED_KW = re.compile(r"unexpected keyword argument '([^']+)'")


def build_net_tolerant(game: str, arch: Dict) -> Tuple[torch.nn.Module, List[str]]:
    """build_game_network, dropping saved-config keys the current config
    class no longer accepts (checkpoints carry fields from the code that
    wrote them).  Returns (net, dropped_keys)."""
    arch = dict(arch)
    dropped: List[str] = []
    while True:
        try:
            return build_game_network(game, **arch), dropped
        except TypeError as e:
            m = _UNEXPECTED_KW.search(str(e))
            if m is None or m.group(1) not in arch:
                raise
            dropped.append(m.group(1))
            arch.pop(m.group(1))


def build_labels(
    game: str, tgt: Dict[str, torch.Tensor], tf_prob: float,
    frame_mask: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    """Teacher-forcing labels: keep the ground-truth conditioning with
    probability ``tf_prob`` per frame, else −1 (decoder self-samples).
    STATE_LABEL_KEYS entries are copied unconditionally — they are game
    state, not decisions, and must never be dropped to −1."""
    keys = [k for k in LABEL_KEYS[game] if k in tgt]
    if not keys:
        return None
    state_keys = set(STATE_LABEL_KEYS.get(game, ()))
    labels = {}
    if tf_prob >= 1.0:
        for k in keys:
            labels[k] = tgt[k]
        return labels
    keep = (torch.rand(frame_mask.shape, device=frame_mask.device) < tf_prob)
    for k in keys:
        if k in state_keys:
            labels[k] = tgt[k]
        else:
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

    if args.game == "clash_royale" and args.max_len is not None:
        raise SystemExit(
            "--max_len random-crops sequences, but the clash_royale wait "
            "head derives absolute time from t0=0 (model/games/"
            "clash_royale.py decode), so cropped frames get wrong time "
            "features; train clash_royale without --max_len"
        )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_ckpt.get("game") != args.game:
            raise ValueError(
                f"checkpoint is for game {resume_ckpt.get('game')!r}, not {args.game!r}"
            )

    dataset = GameShardDataset(args.data_dir, game=args.game, max_len=args.max_len)
    val_dataset = None
    if args.val_dir:
        val_dataset = GameShardDataset(args.val_dir, game=args.game,
                                       max_len=args.max_len)
        if (val_dataset.converter_version != dataset.converter_version
                or val_dataset.shard_meta != dataset.shard_meta):
            raise ValueError(
                f"val shards (converter_version="
                f"{val_dataset.converter_version}, meta={val_dataset.shard_meta}) "
                f"were converted differently from train shards "
                f"(converter_version={dataset.converter_version}, "
                f"meta={dataset.shard_meta}); re-convert both together"
            )
    min_cv = MIN_CONVERTER_VERSION.get(args.game)
    if min_cv is not None and (dataset.converter_version or 1) < min_cv:
        print(f"[WARN] {args.game} shards have converter_version "
              f"{dataset.converter_version} < {min_cv}: they predate a "
              f"semantic converter fix — re-convert with the current "
              f"converter, and never mix generations in one directory")

    # A resume checkpoint defines the architecture (same invariant as
    # train_il/train_rl): rebuilding from CLI overrides would silently
    # change the model shape and crash — or worse, drift — on load.
    if resume_ckpt is not None:
        arch = resume_arch_overrides(resume_ckpt, overrides)
    else:
        arch = dict(overrides)
        for meta_key, cfg_keys in META_ARCH_KEYS.get(args.game, {}).items():
            if meta_key in dataset.shard_meta:
                for ck in cfg_keys:
                    arch.setdefault(ck, dataset.shard_meta[meta_key])
    net, dropped = build_net_tolerant(args.game, arch)
    if dropped:
        print(f"[warn] checkpoint config fields unknown to current code, "
              f"ignored: {dropped}")
    net = net.to(device)
    cfg = net.cfg
    check_meta_arch(args.game, cfg, dataset.shard_meta)
    overrides = arch
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[init] game={args.game} device={device.type} params={n_params:,}")

    # Shard-locality shuffling: plain shuffle=True would reload a whole
    # shard from disk for nearly every sample (the dataset caches one
    # shard); this sampler shuffles shard order + within-shard order.
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=ShardLocalitySampler(dataset),
                        num_workers=args.num_workers,
                        collate_fn=collate_game_sequences, drop_last=False)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
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
    if resume_ckpt is not None:
        load_model_state(net, resume_ckpt["model_state"], ctx=args.resume)
        if "optimizer_state" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        if "scheduler_state" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch = resume_ckpt.get("epoch", -1) + 1
        print(f"[resume] {args.resume} → epoch {start_epoch}")
        resume_ckpt = None  # release checkpoint tensors

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
