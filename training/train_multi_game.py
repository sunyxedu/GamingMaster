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

    # the same on 4 GPUs of one node (data parallel, see below)
    uv run torchrun --standalone --nproc_per_node=4 -m training.train_multi_game ...

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

Data parallelism (torchrun; NCCL on CUDA, gloo on CPU).  Every rank draws its
own K micro-batches per optimizer step with its own RNG streams (seed + rank:
game draws, data order, crops), so a step averages world·K i.i.d.
micro-batches: each rank scales its loss by 1/K and the gradient sync
averages over ranks — the single-GPU objective with K' = world·K (to
reproduce a single-GPU run with K micro-batches per step, pass
``--games_per_step K/world``).  The sync (model.rl_utils.sync_gradients)
posts the same collectives on every rank even though a game's
encoder/decoder has no gradient on ranks that did not draw it (zeros are
contributed), and a parameter without gradient on EVERY rank keeps grad None
so AdamATan2 skips it exactly as single-process.  The step-drop decision on a
non-finite gradient norm is collective, MoE load tallies are summed over
ranks before the balance-bias update, the encoders' BatchNorm running
statistics are merged every step (training.distributed.BatchNormStatSync),
and the initial weights and buffers are broadcast from rank 0 — so
parameters, buffers (balance biases included) and EMA shadows stay
bit-identical across ranks.  Rank 0 alone logs (losses aggregated over all
ranks), validates and writes checkpoints.  ``--steps`` counts optimizer steps.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data_processing.games.dataset import (
    GameShardDataset,
    ShardLocalitySampler,
    collate_game_sequences,
)
from model.components.core import allreduce_moe_loads, update_moe_balance
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
from model.rl_utils import sync_gradients
from training.deep_supervision import Q_HALT_WEIGHT, BatchCarry
from training.distributed import (
    BatchNormStatSync,
    broadcast_module_state,
    collective_clip_grad_norm_,
    init_distributed,
)
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
    p.add_argument("--seed", type=int, default=0,
                   help="data-parallel rank r uses seed + r for its game draws and data order")
    p.add_argument("--bucket_mb", type=float, default=25.0,
                   help="data parallel: gradient all-reduce bucket size in MiB (0 = per tensor)")
    p.add_argument("--dist_timeout", type=float, default=60.0,
                   help="data parallel: collective timeout in minutes (the other ranks "
                        "wait at a barrier while rank 0 validates / saves)")
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


def _global_window(window: Dict[str, List[float]], games: List[str],
                   device: torch.device) -> Dict[str, List[float]]:
    """Per-game mean loss over EVERY rank's micro-batches of the log window
    (one fixed-size SUM all-reduce of (loss sum, count) per game)."""
    t = torch.tensor([[sum(window.get(f"{g}/loss", [])), len(window.get(f"{g}/loss", []))]
                      for g in games], dtype=torch.float64).to(device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t.cpu()
    return {f"{g}/loss": [float(t[i, 0] / t[i, 1])]
            for i, g in enumerate(games) if t[i, 1] > 0}


def main(argv=None) -> Dict[str, object]:
    """Train; returns the live objects (``model``, ``ema``, ``optimizer``,
    ``dist``) for callers that drive the trainer in-process (tests)."""
    args = parse_args(argv)
    device = torch.device(args.device) if args.device else pick_device()
    dist_env, device = init_distributed(device, args.dist_timeout)
    rank0 = dist_env.is_main
    # rank-specific streams (game draws, data order, crops); rank 0 == single-process
    random.seed(args.seed + dist_env.rank)
    torch.manual_seed(args.seed + dist_env.rank)

    def log(*a, **kw) -> None:
        if rank0:
            print(*a, **kw)

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
            log(f"[warn] {g}: config fields unknown to {g}'s config, ignored: {dropped}")
        nets[g] = net
    model = MultiGameNetwork(nets).to(device)
    first = model[games[0]]
    if core_state is not None:
        load_core(first, {k: v.to(device) for k, v in core_state.items()})
    if args.freeze_core:
        freeze_core(first, train_norms=args.train_core_norms)
    bn_sync = None
    if dist_env.enabled:
        # every rank initialised its per-game adapters from its own seed
        broadcast_module_state(model, src=0)
        bn_sync = BatchNormStatSync(model)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    world = f" world={dist_env.world}" if dist_env.enabled else ""
    log(f"[init] games={games} core={first.cfg.core_type} moe={first.cfg.moe_experts} "
        f"params={n_total:,} trainable={n_train:,} device={device.type}{world}")
    log("[mixture] " + "  ".join(f"{g}: frames={frames[g]:,} w={weights[g]:.3f}" for g in games))

    loaders = {g: _cycle(DataLoader(ds, batch_size=game_batch.get(g, args.batch_size),
                                    sampler=ShardLocalitySampler(ds),
                                    num_workers=args.num_workers,
                                    collate_fn=collate_game_sequences))
               for g, ds in datasets.items()}
    has_val = any(g in games for g in val_dirs)
    val_loaders = {g: DataLoader(GameShardDataset(d, game=g, max_len=crop(g, args.val_max_len)),
                                 batch_size=game_batch.get(g, args.batch_size),
                                 collate_fn=collate_game_sequences)
                   for g, d in val_dirs.items() if g in games} if rank0 else {}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamATan2(params, lr=args.lr, betas=(args.beta1, args.beta2),
                          weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, trm_lr_lambda(args.warmup_steps, args.steps, args.lr_min_ratio))
    ema = ModelEMA(model, mu=args.ema_rate) if args.ema_rate > 0 else None
    carries = {g: BatchCarry(model[g].supervision_steps, model[g].cfg.halt_exploration_prob)
               for g in games}
    K = args.games_per_step or len(games)
    names, probs = list(weights), [weights[g] for g in weights]
    max_norm = args.clip_grad if args.clip_grad > 0 else float("inf")
    if rank0:
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
            "world_size": dist_env.world,
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
        if dist_env.enabled:
            # mean over all world·K micro-batches; a game's adapter that no
            # rank drew keeps grad None (AdamATan2 skips it, as single-process)
            sync_gradients(model, dist_env.world, warn_partial=False, bucket_mb=args.bucket_mb)
            grad_norm = collective_clip_grad_norm_(params, max_norm, device)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if torch.isfinite(grad_norm):
            optimizer.step()
            scheduler.step()
            if ema is not None:
                ema.update(model)
            if dist_env.enabled:
                allreduce_moe_loads(model)      # balance on the global load
            loads = update_moe_balance(model, args.moe_balance_rate)
        else:
            loads = None
            log(f"[step {step}] non-finite grad norm, step dropped")
        optimizer.zero_grad(set_to_none=True)
        if bn_sync is not None:
            bn_sync()       # forward-side state: merged on dropped steps too

        if step % args.log_every == 0:
            if dist_env.enabled:
                window = _global_window(window, games, device)
            msg = "  ".join(f"{k}={sum(v) / len(v):.4f}" for k, v in sorted(window.items()))
            extra = ""
            if loads is not None:
                extra = f"  moe_load_cv={float(loads.std(-1).mean()):.3f}"
            log(f"[step {step}/{args.steps}] lr={scheduler.get_last_lr()[0]:.2e} "
                f"gnorm={float(grad_norm):.3f} {msg}{extra}  ({time.time() - t0:.0f}s)",
                flush=True)
            window = {}
        if has_val and (step % args.eval_every == 0 or step == args.steps):
            if rank0:
                ev_model = ema.ema_copy(model) if ema is not None else model
                res = evaluate(ev_model, val_loaders, device, args.value_weight, args.bf16,
                               args.eval_batches)
                for g, m in res.items():
                    keys = [k for k in sorted(m) if k.endswith("loss") or k.endswith("acc")]
                    print(f"[val {step}] {g}: " + " ".join(f"{k}={m[k]:.4f}" for k in keys),
                          flush=True)
                del ev_model
            dist_env.barrier()
    if rank0:
        print(f"[done] saved {save('final')}")
    dist_env.barrier()
    dist_env.destroy()
    return {"model": model, "ema": ema, "optimizer": optimizer, "dist": dist_env}


if __name__ == "__main__":
    main()
