"""Joint imitation learning of one shared core ("game brain") on many games,
and adaptation of a trained brain to a new game.

    # joint pretraining on several games
    uv run python -m training.train_multi_game \\
        --data othello=data/othello/train --data chess=data/chess/train \\
        --val othello=data/othello/val --val chess=data/chess/val \\
        --steps 20000 --override moe_experts=8

    # new game, frozen brain: only its encoder/decoder (+ core norms) train
    uv run python -m training.train_multi_game --data go=data/go/train \\
        --init_core checkpoints/multi/multi_final.pt --freeze_core --train_core_norms

Objective.  Every optimizer step averages K micro-batches (``--games_per_step``),
each from a game drawn i.i.d. from the mixture

    w_g ∝ N_g^α          (N_g = frames of game g, α = --sample_alpha)

so the expected step gradient is ∇ Σ_g w_g L_g, where L_g is game g's mean
per-frame loss (decoder IL loss + value MSE).  α = 1 weighs every frame
equally (concatenated data), α = 0 weighs games equally, α ≈ 0.5 is the
usual temperature compromise (HPT uses √N, π0 N^0.43).

Everything else is the IL recipe of training/train_il_game.py: AdamATan2,
warmup then constant LR, EMA weights for evaluation/checkpoints, deep
supervision only if the core is the legacy TRM.  With a MoE core the
aux-loss-free balance bias is updated once per optimizer step from the
loads of ALL K micro-batches, i.e. balanced on the game mixture.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from data_processing.games.dataset import (
    GameShardDataset,
    ShardLocalitySampler,
    collate_game_sequences,
)
from model.components.core import update_moe_balance
from model.config import upgrade_config_dict
from model.multi_game import (
    MultiGameNetwork,
    core_config_overrides,
    freeze_core,
    is_core_key,
    load_core,
)
from model.optim import (
    TRM_BETAS,
    TRM_EMA_RATE,
    TRM_LR,
    TRM_LR_MIN_RATIO,
    TRM_WARMUP_STEPS,
    TRM_WEIGHT_DECAY,
    AdamATan2,
    ModelEMA,
    trm_lr_lambda,
)
from model.rl_utils import qk_norm_param_ids
from training.deep_supervision import Q_HALT_WEIGHT, BatchCarry
from training.train_il_game import (
    LABEL_KEYS,
    META_ARCH_KEYS,
    UNCROPPABLE,
    batch_loss,
    build_net_tolerant,
    mb_from_merged,
    parse_overrides,
    pick_device,
    prepare_batch,
    prepare_rows,
)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", action="append", required=True, metavar="GAME=DIR",
                   help="training shards of one game (repeat per game)")
    p.add_argument("--val", action="append", default=[], metavar="GAME=DIR")
    p.add_argument("--steps", type=int, default=10000, help="optimizer steps")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--game_batch", action="append", default=[], metavar="GAME=N",
                   help="per-game batch size override (losses are per-frame means, so "
                        "this changes gradient noise, not the mixture objective)")
    p.add_argument("--games_per_step", type=int, default=0,
                   help="micro-batches per optimizer step (0 = number of games)")
    p.add_argument("--sample_alpha", type=float, default=0.5,
                   help="mixture temperature: w_g ∝ frames_g^alpha")
    p.add_argument("--max_len", type=int, default=None, help="random-crop training sequences")
    p.add_argument("--val_max_len", type=int, default=None,
                   help="validation crop (default: full sequences, so every evaluation "
                        "sees exactly the same frames)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="core/config overrides shared by all games")
    p.add_argument("--tf_prob", type=float, default=1.0)
    p.add_argument("--value_weight", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=TRM_LR)
    p.add_argument("--weight_decay", type=float, default=TRM_WEIGHT_DECAY)
    p.add_argument("--warmup_steps", type=int, default=TRM_WARMUP_STEPS)
    p.add_argument("--lr_min_ratio", type=float, default=TRM_LR_MIN_RATIO)
    p.add_argument("--beta1", type=float, default=TRM_BETAS[0])
    p.add_argument("--beta2", type=float, default=TRM_BETAS[1])
    p.add_argument("--clip_grad", type=float, default=0.0)
    p.add_argument("--ema_rate", type=float, default=TRM_EMA_RATE)
    p.add_argument("--moe_balance_rate", type=float, default=1e-3)
    p.add_argument("--init_core", default=None,
                   help="checkpoint whose shared core initialises (and fixes the "
                        "architecture of) this run")
    p.add_argument("--freeze_core", action="store_true",
                   help="train only the per-game encoder/decoder/value head")
    p.add_argument("--train_core_norms", action="store_true",
                   help="with --freeze_core: keep the core's norm parameters "
                        "trainable (Lu et al. 2021)")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=0, help="0 = full val set")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_dir", default="checkpoints/multi")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _pairs(items: List[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        g, sep, d = it.partition("=")
        if not sep or g not in LABEL_KEYS:
            raise SystemExit(f"expected GAME=DIR with GAME in {sorted(LABEL_KEYS)}, got {it!r}")
        out[g] = d
    return out


def mixture_weights(frames: Dict[str, int], alpha: float) -> Dict[str, float]:
    """w_g ∝ N_g^α, normalised."""
    raw = {g: float(max(n, 1)) ** alpha for g, n in frames.items()}
    z = sum(raw.values())
    return {g: v / z for g, v in raw.items()}


def multi_game_param_groups(model: torch.nn.Module, weight_decay: float) -> List[Dict]:
    """Every trainable parameter once (shared core tensors de-duplicated),
    decayed — except the QK-norm scales (wd 0, see
    ``model.rl_utils.qk_norm_param_ids``).  Not ``optim_param_groups``: this
    trainer decays the recurrence memory parameters, and a qk_norm=False
    model keeps exactly its historical single group."""
    qk = qk_norm_param_ids(model)
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if id(p) not in qk],
               "weight_decay": weight_decay}]
    no_decay = [p for p in params if id(p) in qk]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def load_core_checkpoint(path: str):
    """(core_state, core config overrides) from a multi-game checkpoint
    (``core_state`` + ``core_config``) or a single-game one (``model_state``
    + ``config``; its core keys are extracted)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "core_state" in ck:
        return ck["core_state"], core_config_overrides(ck["core_config"])
    state = {k: v for k, v in ck["model_state"].items() if is_core_key(k)}
    return state, core_config_overrides(upgrade_config_dict(ck["config"]))


@torch.no_grad()
def evaluate(model: MultiGameNetwork, loaders: Dict[str, DataLoader], device,
             value_weight: float, bf16: bool, max_batches: int = 0) -> Dict[str, Dict[str, float]]:
    """Per-game validation at the default inference depth (EMA weights are
    passed in by the caller)."""
    model.eval()
    res = {}
    for g, loader in loaders.items():
        net = model[g]
        agg: Dict[str, float] = {}
        n = 0
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            mb = prepare_batch(g, batch, device, tf_prob=1.0)
            _, stats, _ = batch_loss(net, mb, device, value_weight, Q_HALT_WEIGHT, bf16,
                                     act_steps=net.default_depth)
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        res[g] = {k: v / max(n, 1) for k, v in agg.items()}
    model.train()
    return res


def main(argv=None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else pick_device()
    data_dirs, val_dirs = _pairs(args.data), _pairs(args.val)
    games = list(data_dirs)
    game_batch = {g: int(n) for g, n in (x.split("=", 1) for x in args.game_batch)}

    def crop(g: str, max_len):
        return None if g in UNCROPPABLE else max_len

    datasets = {g: GameShardDataset(d, game=g, max_len=crop(g, args.max_len))
                for g, d in data_dirs.items()}
    frames = {g: ds.total_frames() for g, ds in datasets.items()}
    weights = mixture_weights(frames, args.sample_alpha)

    core_kw = parse_overrides(args.override)
    core_state = None
    if args.init_core:
        core_state, fixed = load_core_checkpoint(args.init_core)
        clash = {k: v for k, v in core_kw.items() if k in fixed and fixed[k] != v}
        if clash:
            raise SystemExit(f"--override {clash} contradicts the core in {args.init_core}")
        core_kw = {**core_kw, **fixed}

    nets = {}
    for g in games:
        kw = dict(core_kw)
        for meta_key, cfg_keys in META_ARCH_KEYS.get(g, {}).items():
            if meta_key in datasets[g].shard_meta:
                for ck in cfg_keys:
                    kw.setdefault(ck, datasets[g].shard_meta[meta_key])
        net, dropped = build_net_tolerant(g, kw)
        if dropped:
            print(f"[warn] {g}: config fields unknown to {g}'s config, ignored: {dropped}")
        nets[g] = net
    model = MultiGameNetwork(nets).to(device)
    first = model[games[0]]
    if core_state is not None:
        load_core(first, {k: v.to(device) for k, v in core_state.items()})
    if args.freeze_core:
        freeze_core(first, train_norms=args.train_core_norms)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[init] games={games} core={first.cfg.core_type} moe={first.cfg.moe_experts} "
          f"params={n_total:,} trainable={n_train:,} device={device.type}")
    print("[mixture] " + "  ".join(f"{g}: frames={frames[g]:,} w={weights[g]:.3f}" for g in games))

    loaders = {g: _cycle(DataLoader(ds, batch_size=game_batch.get(g, args.batch_size),
                                    sampler=ShardLocalitySampler(ds),
                                    num_workers=args.num_workers,
                                    collate_fn=collate_game_sequences))
               for g, ds in datasets.items()}
    val_loaders = {g: DataLoader(GameShardDataset(d, game=g, max_len=crop(g, args.val_max_len)),
                                 batch_size=game_batch.get(g, args.batch_size),
                                 collate_fn=collate_game_sequences)
                   for g, d in val_dirs.items() if g in games}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamATan2(multi_game_param_groups(model, args.weight_decay), lr=args.lr,
                          betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, trm_lr_lambda(args.warmup_steps, args.steps, args.lr_min_ratio))
    ema = ModelEMA(model, mu=args.ema_rate) if args.ema_rate > 0 else None
    carries = {g: BatchCarry(model[g].supervision_steps, model[g].cfg.halt_exploration_prob)
               for g in games}
    K = args.games_per_step or len(games)
    names, probs = list(weights), [weights[g] for g in weights]
    os.makedirs(args.save_dir, exist_ok=True)

    def save(tag: str) -> str:
        state = ema.ema_model_state(model) if ema is not None else model.state_dict()
        pre = f"nets.{games[0]}."
        core = {k[len(pre):]: v for k, v in state.items()
                if k.startswith(pre) and is_core_key(k[len(pre):])}
        path = os.path.join(args.save_dir, f"multi_{tag}.pt")
        torch.save({
            "games": games,
            "configs": {g: model[g].cfg.__dict__ for g in games},
            "core_config": core_config_overrides(model[games[0]].cfg),
            "core_state": core,
            "model_state": state,
            "train_model_state": model.state_dict(),
            "mixture": weights,
            "args": vars(args),
        }, path)
        return path

    model.train()
    t0 = time.time()
    window: Dict[str, List[float]] = {}
    for step in range(1, args.steps + 1):
        drawn = random.choices(names, weights=probs, k=K)
        for g in drawn:
            net = model[g]

            def forward_backward(merged, carry, active, reset, weight, g=g, net=net):
                loss, stats, pred = batch_loss(net, mb_from_merged(g, merged), device,
                                               args.value_weight, Q_HALT_WEIGHT, args.bf16,
                                               carry=carry, active=active, reset=reset)
                if torch.isfinite(loss):
                    (loss * (weight / K)).backward()
                return pred, stats

            st = carries[g].step(prepare_rows(g, next(loaders[g]), device, args.tf_prob),
                                 forward_backward)
            if st is not None and "loss" in st:
                window.setdefault(f"{g}/loss", []).append(st["loss"])
        grad_norm = torch.nn.utils.clip_grad_norm_(
            params, args.clip_grad if args.clip_grad > 0 else float("inf"))
        if torch.isfinite(grad_norm):
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            loads = update_moe_balance(model, args.moe_balance_rate)
        else:
            loads = None
            print(f"[step {step}] non-finite grad norm, step dropped")
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0:
            msg = "  ".join(f"{k}={sum(v) / len(v):.4f}" for k, v in sorted(window.items()))
            extra = ""
            if loads is not None:
                extra = f"  moe_load_cv={float(loads.std(-1).mean()):.3f}"
            print(f"[step {step}/{args.steps}] lr={scheduler.get_last_lr()[0]:.2e} "
                  f"gnorm={float(grad_norm):.3f} {msg}{extra}  ({time.time() - t0:.0f}s)",
                  flush=True)
            window = {}
        if val_loaders and (step % args.eval_every == 0 or step == args.steps):
            ev_model = ema.ema_copy(model) if ema is not None else model
            res = evaluate(ev_model, val_loaders, device, args.value_weight, args.bf16,
                           args.eval_batches)
            for g, m in res.items():
                keys = [k for k in sorted(m) if k.endswith("loss") or k.endswith("acc")]
                print(f"[val {step}] {g}: " + " ".join(f"{k}={m[k]:.4f}" for k in keys),
                      flush=True)
            del ev_model
    print(f"[done] saved {save('final')}")


if __name__ == "__main__":
    main()
