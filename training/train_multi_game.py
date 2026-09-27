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

    # the same on 4 GPUs of one node (data parallel, see below) — or on 4 MIG
    # slices (CUDA_VISIBLE_DEVICES=MIG-...,MIG-...: every rank pins its own
    # slice, --dist_backend auto picks gloo)
    uv run torchrun --standalone --nproc_per_node=4 -m training.train_multi_game ...

    # generals next to the others: its `gmdata generals shard` gm-shards
    # (recommended), or a directory of gio records (the output of
    # `gmdata generals ladder|pipeline`, decoded like SeqReplayDataset: slow,
    # small corpora only).  Teacher forcing defaults to 0 for generals
    # (train_il.py's default; --game_tf generals=0.4:0 anneals like the
    # production IL jobs); its crop must be given per game.
    uv run python -m training.train_multi_game \\
        --data generals=data/generals/shards --data othello=data/othello/train \\
        --game_batch generals=2 --game_max_len generals=512 --num_workers 4 ...

Games and data.  Every registered game except generals reads GameShardDataset
shards (.pt / gm-shard directories); generals reads gio record directories or
gm-shards (data_processing/games/generals.py).  All games go through the same
GameNetwork interface and the decoder's own ``il_loss`` (generals: train_il.py's
ImitationLoss).  Crops: ``--max_len`` (per game ``--game_max_len``) takes a
random window, except for generals, whose adapter reads the absolute frame
index (pass head, turn_globals, fog-of-war memory): it keeps the FIRST frames,
like train_il.py's --max_seq_len, so a run with generals and a global
``--max_len`` must bound generals explicitly (``--game_max_len generals=N``),
and generals validates on its training crop unless told otherwise;
clash_royale is never cropped.

Objective.  Every optimizer step averages K micro-batches (``--games_per_step``),
each from a game drawn i.i.d. from the mixture

    w_g ∝ N_g^α          (N_g = frames of game g, α = --sample_alpha)

so the expected step gradient is ∇ Σ_g w_g λ_g L_g, where L_g is game g's
mean per-frame loss (decoder IL loss + value loss) and λ_g its
``--game_loss_weight`` (default 1).  α = 1 weighs every frame equally
(concatenated data), α = 0 weighs games equally, α ≈ 0.5 is the usual
temperature compromise (HPT uses √N, π0 N^0.43).  The decoders' losses are not
on one scale: generals' is train_il.py's weighted ImitationLoss (source 3.0
focal, direction 1.5, split 0.3, pass 1.0), the others sum unit-weight
cross-entropies, so a generals micro-batch pulls the shared core ~2-3x harder
(λ_generals ≈ 0.4 roughly equalises the initial loss scales).

Value heads are per game.  The value loss is MSE for the scalar head, or the
HL-Gauss cross-entropy with ``--override hl_gauss_value=True`` (every game
gets a categorical head over [-hl_gauss_vmax, hl_gauss_vmax]; IL targets are
final outcomes in [-1, 1], generals' an outcome / army-ratio blend in [-1, 1]).
Logs report ``value_loss`` (the trained loss) and ``value_mse`` (MSE of the
scalar prediction, comparable across heads).

Engine supervision, per game wherever its shards carry it (train_il_game's
batch_loss): soft policy targets ``tgt["<head>_probs"]`` replace the one-hot
label on teacher frames (--soft_target_weight α; validation logs
``<head>_teacher_kl``), ``tgt["engine_value"]`` blends into the value target
(--engine_value_weight β); --il_legal_mask feeds the shards' ``legal_<key>``
masks to the decoders.  Games whose shards predate the frozen pre-training
schema are warned about (--require_frozen_schema: refused).

Everything else is the IL recipe of training/train_il_game.py: AdamATan2,
warmup then constant LR, EMA weights for evaluation/checkpoints, deep
supervision only if the core is the legacy TRM.  With a MoE core the
aux-loss-free balance bias is updated once per optimizer step from the
loads of ALL K micro-batches, i.e. balanced on the game mixture.

Conv encoders (generals, board games, shogi, lux, clash_royale, starcraft)
normalise every frame on its own by default (``encoder_norm="group"``,
model/components/encoder.py): a training forward over B·T frames gives frame
t no statistics of later frames or other samples, and train and eval mode
compute the same function.  ``--override encoder_norm=batch`` builds the
legacy BatchNorm encoders, which every checkpoint predating the field gets
automatically (resume, evaluation); it is not a core field, so the fresh
encoder of a game adapted to an ``--init_core`` brain takes the default.

Data parallelism (torchrun; NCCL on CUDA, gloo on CPU and on MIG slices —
``--dist_backend``, training.distributed: on a MIG allocation every rank pins
itself to its own slice and runs on cuda:0).  Every rank draws its
own micro-batches with its own RNG streams (training.distributed.rank_seed:
game draws, data order, crops), so a step averages i.i.d. micro-batches of
all ranks — the single-GPU objective with K' = the global count.  How many:

    --games_per_step K          K per rank, K' = world·K: the global batch
                                grows with the GPU count at the same --lr and
                                --steps (default K = number of games)
    --global_games_per_step K'  K' in total, split ⌈K'/world⌉ / ⌊K'/world⌋
                                over the ranks: the same objective as a
                                single-GPU run with --games_per_step K', only
                                faster (the step waits for the ranks with ⌈⌉)

Either way each micro-batch's loss is scaled so that the step gradient is
the mean over the K' micro-batches.  The sync (model.rl_utils.sync_gradients)
posts the same collectives on every rank even though a game's
encoder/decoder has no gradient on ranks that did not draw it (zeros are
contributed), and a parameter without gradient on EVERY rank keeps grad None
so AdamATan2 skips it exactly as single-process.  The step-drop decision on a
non-finite gradient norm is collective, MoE load tallies are summed over
ranks before the balance-bias update, the BatchNorm running statistics of
legacy encoders (``encoder_norm="batch"``; the default per-frame GroupNorm has
none) are merged every step (training.distributed.BatchNormStatSync),
and the initial weights and buffers are broadcast from rank 0 — so
parameters, buffers (balance biases included) and EMA shadows stay
bit-identical across ranks.  Rank 0 alone logs (losses aggregated over all
ranks) and writes checkpoints; validation batches are split over the ranks
and their (sum, count) statistics gathered.  ``--steps`` counts optimizer
steps.

Sharded optimizer state (``--shard_optimizer``, data parallel only; ZeRO-1,
training/zero.py).  Without it every rank holds the weights, gradients,
AdamATan2 moments and EMA shadows of the whole model: 20 bytes per
parameter.  With it the trainable parameters are partitioned over the ranks
(whole tensors, balanced by size, a function of the architecture and the
world size only) and each rank keeps the moments, step counters and EMA
shadows of its partition: 8 + 12/world bytes per parameter.  Each rank steps
its own parameters (the same AdamATan2 kernel on the same tensors; grad-None
parameters skipped as before) and the owners broadcast the updated weights
(bucketed by --bucket_mb), so the run is bit-identical to the unsharded one.
Validation swaps the gathered EMA into the live model instead of copying it;
checkpoints are gathered onto rank 0 (every rank takes part) in the unsharded
format, so either kind of run resumes the other, at any world size.

Resuming.  ``--ckpt_every N`` makes rank 0 write ``multi_latest.pt`` every N
steps and at the last one (weights, EMA shadows, optimizer, LR schedule,
step; written atomically).  ``--resume <that file>`` continues from it with
the same arguments, any world size: weights, optimizer moments, EMA, balance
biases, (legacy) BatchNorm statistics and the LR schedule continue exactly; game
draws and data order restart from fresh RNG streams, and a TRM core's deep-
supervision carries restart empty.  Each game's network is rebuilt from the
config saved in the checkpoint (``--override`` values on top), so a config
default that changed since it was written cannot reshape it.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from data_processing.games.dataset import collate_game_sequences, training_sampler
from data_processing.games.generals import GeneralsGioDataset, gio_dataset_advice
from model.components.core import allreduce_moe_loads, update_moe_balance
from model.components.encoder import describe_encoder_norm
from model.config import checkpoint_config_dict, legacy_memory_vision, upgrade_config_dict
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
from model.rl_utils import qk_norm_param_ids, sync_gradients
from training.deep_supervision import Q_HALT_WEIGHT, BatchCarry
from training.diagnostics import attention_probe, format_attention, logit_lse_absmax, qk_caps
from training.distributed import (
    DIST_BACKENDS,
    BatchNormStatSync,
    broadcast_module_state,
    collective_clip_grad_norm_,
    init_distributed,
    pin_rank_device,
    rank_seed,
)
from training.train_il_game import (
    LABEL_KEYS,
    PREFIX_CROP,
    UNCROPPABLE,
    check_crop_policy,
    apply_meta_arch_defaults,
    batch_loss,
    build_net_tolerant,
    check_meta_arch,
    default_tf,
    describe_value_head,
    greedy_decoding,
    mb_from_merged,
    open_dataset,
    parse_overrides,
    pick_device,
    prepare_batch,
    prepare_rows,
    schema_freeze_warning,
)
from training.zero import Zero1


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
                   help="micro-batches per optimizer step (0 = number of games); "
                        "data parallel: PER RANK")
    p.add_argument("--global_games_per_step", type=int, default=0,
                   help="data parallel: micro-batches per optimizer step summed over "
                        "all ranks (the single-GPU --games_per_step objective); "
                        "exclusive with --games_per_step")
    p.add_argument("--sample_alpha", type=float, default=0.5,
                   help="mixture temperature: w_g ∝ frames_g^alpha")
    p.add_argument("--max_len", type=int, default=None,
                   help="crop training sequences to a random window (not generals: "
                        "with generals in the run, bound it via --game_max_len)")
    p.add_argument("--val_max_len", type=int, default=None,
                   help="validation crop (default: full sequences, so every evaluation "
                        "sees exactly the same frames); generals ignores it and validates "
                        "on its training crop unless given --game_val_max_len")
    p.add_argument("--game_max_len", action="append", default=[], metavar="GAME=N",
                   help="per-game --max_len (sequence lengths differ by orders of "
                        "magnitude across games); generals keeps its first N frames")
    p.add_argument("--game_val_max_len", action="append", default=[], metavar="GAME=N",
                   help="per-game --val_max_len (e.g. bound generals' ~1000-frame "
                        "games; a prefix crop is deterministic)")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VALUE",
                   help="core/config overrides shared by all games")
    p.add_argument("--tf_prob", type=float, default=None,
                   help="teacher-forcing probability of the conditioning labels, all "
                        "games (default: per game, train_il_game.DEFAULT_TF: generals 0 "
                        "like train_il.py, the others 1.0)")
    p.add_argument("--game_tf", action="append", default=[], metavar="GAME=P[:P_END]",
                   help="per-game teacher forcing: a constant P, or P:P_END annealed "
                        "linearly over --steps (e.g. generals=0.4:0, the production "
                        "generals IL schedule)")
    p.add_argument("--game_loss_weight", action="append", default=[], metavar="GAME=W",
                   help="per-game loss multiplier λ_g (default 1; generals' weighted "
                        "ImitationLoss is ~2-3x the scale of the other decoders' losses)")
    p.add_argument("--value_weight", type=float, default=0.2)
    p.add_argument("--soft_target_weight", type=float, default=1.0,
                   help="alpha of the engine soft targets (tgt <head>_probs, where the "
                        "shards have them): 1 = train the teacher distribution instead "
                        "of the one-hot label on teacher frames, 0 = hard labels only")
    p.add_argument("--engine_value_weight", type=float, default=1.0,
                   help="beta of the engine value targets (tgt engine_value, where the "
                        "shards have them): value target = beta*engine + (1-beta)*outcome")
    p.add_argument("--il_legal_mask", action="store_true",
                   help="feed the shards' tgt legal_<key> masks to the decoders as "
                        "legal[<key>] (legal-masked IL loss / metrics where supported)")
    p.add_argument("--require_frozen_schema", action="store_true",
                   help="refuse shards older than the frozen pre-training schema "
                        "(train_il_game.SCHEMA_FREEZE_CONVERTER_VERSION) instead of warning")
    p.add_argument("--logit_z_loss", type=float, default=1e-4,
                   help="z-loss weight on every categorical action head's log-partition "
                        "(PaLM; prevents output-logit divergence, seen on go at lr 3e-4); "
                        "0 = off")
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
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="write the resumable multi_latest.pt every N steps and at the "
                        "last step (0 = only multi_final.pt)")
    p.add_argument("--resume", default=None,
                   help="continue from a multi_latest.pt (same arguments otherwise)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="data-parallel rank r uses rank_seed(seed, r) for its game draws "
                        "and data order (rank 0: seed)")
    p.add_argument("--bucket_mb", type=float, default=25.0,
                   help="data parallel: gradient all-reduce bucket size in MiB (0 = per tensor); "
                        "25 suits gloo / MIG slices too (sync time flat from 1 to 400 MiB, "
                        "scripts/bench_grad_sync.py)")
    p.add_argument("--dist_timeout", type=float, default=10.0,
                   help="data parallel: collective timeout in minutes (the other ranks "
                        "wait while rank 0 writes a checkpoint)")
    p.add_argument("--dist_backend", choices=DIST_BACKENDS, default="auto",
                   help="data parallel: process-group backend; auto = NCCL on full GPUs, "
                        "gloo on MIG slices (NCCL refuses two ranks on one physical GPU; "
                        "every rank is pinned to its own slice) and on the CPU")
    p.add_argument("--shard_optimizer", action="store_true",
                   help="data parallel (world > 1 only): ZeRO-1 — every rank keeps the "
                        "AdamATan2 moments and the EMA of its 1/world partition of the "
                        "parameters only (training/zero.py); the same training, bit for "
                        "bit, and the same checkpoint format (resumable either way, any "
                        "world size), for 12/world instead of 12 bytes per parameter")
    return p.parse_args(argv)


def _pairs(items: List[str], what: str = "DIR") -> Dict[str, str]:
    out = {}
    for it in items:
        g, sep, d = it.partition("=")
        if not sep or g not in LABEL_KEYS:
            raise SystemExit(f"expected GAME={what} with GAME in {sorted(LABEL_KEYS)}, "
                             f"got {it!r}")
        out[g] = d
    return out


def tf_schedule(spec: str) -> Tuple[float, float]:
    """``--game_tf`` value → (start, end): ``"P"`` or ``"P:P_END"``."""
    a, sep, b = spec.partition(":")
    lo, hi = float(a), float(b) if sep else float(a)
    if not (0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0):
        raise SystemExit(f"--game_tf probabilities must be in [0, 1], got {spec!r}")
    return lo, hi


def tf_at(sched: Tuple[float, float], step: int, steps: int) -> float:
    """Linear teacher-forcing schedule: start at step 1, end at ``steps``."""
    lo, hi = sched
    return lo + (hi - lo) * (step - 1) / max(1, steps - 1)


def mixture_weights(frames: Dict[str, int], alpha: float) -> Dict[str, float]:
    """w_g ∝ N_g^α, normalised."""
    raw = {g: float(max(n, 1)) ** alpha for g, n in frames.items()}
    z = sum(raw.values())
    return {g: v / z for g, v in raw.items()}


def micro_batch_plan(games_per_step: int, global_games_per_step: int, n_games: int,
                     rank: int, world: int) -> Tuple[int, int, float]:
    """(K, k_glob, k_div): K micro-batches on this rank, k_glob per optimizer
    step over all ranks.  Every micro-batch loss is scaled by weight / k_div
    and the sync averages over the ranks, so each weighs 1 / (world·k_div) =
    1 / k_glob in the step gradient: the mean over the k_glob micro-batches.
    Single-process (world 1): K = k_glob = k_div, the historical scaling."""
    if global_games_per_step:
        if games_per_step:
            raise SystemExit("pass --games_per_step (per rank) or --global_games_per_step "
                             "(all ranks), not both")
        k_glob = global_games_per_step
        if k_glob < world:
            raise SystemExit(f"--global_games_per_step {k_glob} < {world} ranks: "
                             f"{world - k_glob} GPU(s) would idle; use fewer GPUs")
        return k_glob // world + int(rank < k_glob % world), k_glob, k_glob / world
    K = games_per_step or n_games
    return K, K * world, K


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


def game_checkpoint(ck: Dict, game: str) -> Dict:
    """One game's network out of a multi-game checkpoint ``ck``, in the
    single-game format (``game``, ``config``, ``model_state`` = EMA weights,
    ``train_model_state`` = raw weights): what training/train_il.py
    ``--finetune_from``, training/train_rl.py ``--il_checkpoint`` (generals)
    and ``build_game_network(game, **config)`` + ``load_state_dict`` consume
    — the per-game post-training entry point of a jointly pretrained brain."""
    if game not in ck["games"]:
        raise KeyError(f"{game!r} not in the checkpoint's games {ck['games']}")
    pre = f"nets.{game}."

    def part(state: Dict) -> Dict:
        return {k[len(pre):]: v for k, v in state.items() if k.startswith(pre)}

    config = dict(ck["configs"][game])
    if game == "generals" and "memory_vision" not in config:
        # the observation it was trained on, which only the multi-game
        # checkpoint as a whole tells (a train_rl_joint --env_kw)
        config["memory_vision"] = legacy_memory_vision(ck)
    out = {"game": game, "config": config,
           "model_state": part(ck["model_state"]),
           "source": {"multi_game_games": list(ck["games"]), "step": ck.get("step")}}
    if "train_model_state" in ck:
        out["train_model_state"] = part(ck["train_model_state"])
    return out


EvalSums = Dict[str, Tuple[Dict[str, float], int]]

#: eval_sums' batch_loss / prepare_batch keyword defaults (= the CLI defaults)
EVAL_LOSS_DEFAULTS = {"il_legal_mask": False, "soft_target_weight": 1.0,
                      "engine_value_weight": 1.0}


@torch.no_grad()
def eval_sums(model: MultiGameNetwork, loaders: Dict[str, DataLoader], device,
              value_weight: float, bf16: bool, max_batches: int = 0,
              diag: Optional[Dict[str, str]] = None, il_legal_mask: bool = False,
              soft_target_weight: float = 1.0,
              engine_value_weight: float = 1.0) -> EvalSums:
    """Per game: (sum over batches of each batch-mean statistic, batch count)
    at the default inference depth, with deterministic conditioning
    (greedy_decoding; EMA weights are passed in by the caller).
    With ``diag`` (a dict) the first batch of each game also runs under
    training.diagnostics.attention_probe, and ``diag[g]`` gets its
    format_attention line.  The remaining keywords are the training
    batch_loss / prepare_batch settings."""
    model.eval()
    res = {}
    for g, loader in loaders.items():
        net = model[g]
        agg: Dict[str, float] = {}
        n = 0
        for i, batch in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            mb = prepare_batch(g, batch, device, tf_prob=1.0, cfg=net.cfg,
                               il_legal_mask=il_legal_mask)
            probe = (attention_probe(net) if diag is not None and i == 0
                     else contextlib.nullcontext())
            with probe as attn, greedy_decoding(net):
                _, stats, pred = batch_loss(net, mb, device, value_weight, Q_HALT_WEIGHT,
                                            bf16, act_steps=net.default_depth,
                                            soft_target_weight=soft_target_weight,
                                            engine_value_weight=engine_value_weight)
            if attn is not None:
                diag[g] = format_attention(attn, logit_lse_absmax(pred, mb["frame_mask"]))
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
        res[g] = (agg, n)
    model.train()
    return res


def _normalise(sums: EvalSums) -> Dict[str, Dict[str, float]]:
    return {g: {k: v / max(n, 1) for k, v in agg.items()} for g, (agg, n) in sums.items()}


def evaluate(model: MultiGameNetwork, loaders: Dict[str, DataLoader], device,
             value_weight: float, bf16: bool, max_batches: int = 0,
             diag: Optional[Dict[str, str]] = None, il_legal_mask: bool = False,
             soft_target_weight: float = 1.0,
             engine_value_weight: float = 1.0) -> Dict[str, Dict[str, float]]:
    """Per-game validation: the mean over batches of every statistic."""
    return _normalise(eval_sums(model, loaders, device, value_weight, bf16, max_batches,
                                diag=diag, il_legal_mask=il_legal_mask,
                                soft_target_weight=soft_target_weight,
                                engine_value_weight=engine_value_weight))


def evaluate_sharded(model: MultiGameNetwork, loaders: Dict[str, DataLoader], device,
                     value_weight: float, bf16: bool, il_legal_mask: bool = False,
                     soft_target_weight: float = 1.0,
                     engine_value_weight: float = 1.0) -> Dict[str, Dict[str, float]]:
    """:func:`evaluate` with the batches split over the ranks (``loaders``
    from :func:`_sharded_val_loader`): every rank evaluates its share, the
    per-game (sums, count) are all-gathered (one small object collective,
    same on every rank), and every rank returns the global mean."""
    local = eval_sums(model, loaders, device, value_weight, bf16,
                      il_legal_mask=il_legal_mask, soft_target_weight=soft_target_weight,
                      engine_value_weight=engine_value_weight)
    everyone: List[EvalSums] = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, local)
    merged: EvalSums = {}
    for g in local:
        agg: Dict[str, float] = {}
        n = 0
        for sums in everyone:
            a, m = sums[g]
            for k, v in a.items():
                agg[k] = agg.get(k, 0.0) + v
            n += m
        merged[g] = (agg, n)
    return _normalise(merged)


def _sharded_val_loader(ds, batch_size: int, rank: int, world: int,
                        max_batches: int = 0) -> DataLoader:
    """Rank ``rank``'s share of the single-process validation batches
    (sequential, the first ``max_batches`` if > 0): batches rank, rank+world, ..."""
    batches = list(BatchSampler(SequentialSampler(ds), batch_size, drop_last=False))
    if max_batches:
        batches = batches[:max_batches]
    return DataLoader(ds, batch_sampler=batches[rank::world], collate_fn=collate_game_sequences)


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
    # MIG slices: this rank's own instance only — before pick_device() touches CUDA
    pin_rank_device()
    device = torch.device(args.device) if args.device else pick_device()
    dist_env, device = init_distributed(device, args.dist_timeout, args.dist_backend)
    rank0, rank, world = dist_env.is_main, dist_env.rank, dist_env.world
    if args.shard_optimizer and not dist_env.enabled:
        raise SystemExit("--shard_optimizer partitions the optimizer state and the EMA over "
                         "data-parallel ranks: it needs torchrun with more than one rank "
                         "(this run has world size 1; drop the flag)")
    # rank-specific streams (game draws, data order, crops); rank 0 == single-process
    random.seed(rank_seed(args.seed, rank))
    torch.manual_seed(rank_seed(args.seed, rank))

    def log(*a, **kw) -> None:
        if rank0:
            print(*a, **kw)

    data_dirs, val_dirs = _pairs(args.data), _pairs(args.val)
    games = list(data_dirs)
    game_batch = {g: int(n) for g, n in _pairs(args.game_batch, "N").items()}
    game_max_len = {g: int(n) for g, n in _pairs(args.game_max_len, "N").items()}
    game_val_max_len = {g: int(n) for g, n in _pairs(args.game_val_max_len, "N").items()}
    game_tf = {g: tf_schedule(v) for g, v in _pairs(args.game_tf, "P[:P_END]").items()}
    for g in games:
        if g not in game_tf:
            p0 = args.tf_prob if args.tf_prob is not None else default_tf(g)
            game_tf[g] = (p0, p0)
    loss_weight = {g: float(v) for g, v in _pairs(args.game_loss_weight, "W").items()}
    for g in games:
        # a generals-style prefix crop from a GLOBAL --max_len is a footgun:
        # --max_len 64 would train only on every game's first 32 turns
        if g in PREFIX_CROP and args.max_len is not None and g not in game_max_len:
            raise SystemExit(
                f"--max_len {args.max_len} would keep only the FIRST {args.max_len} "
                f"frames of every {g} game (its adapter reads the absolute frame index, "
                f"so it is prefix-cropped, never windowed): bound {g} explicitly with "
                f"--game_max_len {g}=N")

    K, k_glob, k_div = micro_batch_plan(args.games_per_step, args.global_games_per_step,
                                        len(games), rank, world)

    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state" not in resume:
            raise SystemExit(f"{args.resume} is not resumable (only --ckpt_every writes "
                             f"multi_latest.pt with the optimizer state)")
        if resume["games"] != games:
            raise SystemExit(f"{args.resume} trained games {resume['games']}, "
                             f"this run has {games} (same --data order required)")

    def crop(g: str, max_len, per_game: Dict[str, int]):
        # UNCROPPABLE games ignore the global max_len; only an explicit
        # per-game value crops them (checked against the net, below)
        return per_game.get(g) if g in UNCROPPABLE else per_game.get(g, max_len)

    def val_crop(g: str):
        if g in PREFIX_CROP:
            # validated on the training crop unless given their own: a full
            # ~1000-frame generals game costs far more than a training batch,
            # and a global --val_max_len window means nothing for a prefix crop
            return game_val_max_len.get(g, crop(g, args.max_len, game_max_len))
        return crop(g, args.val_max_len, game_val_max_len)

    datasets = {g: open_dataset(g, d, crop(g, args.max_len, game_max_len))
                for g, d in data_dirs.items()}
    for g, ds in datasets.items():
        if isinstance(ds, GeneralsGioDataset):
            if dist_env.enabled:
                # the first-run length scan reads every record: split it over
                # the ranks (every rank's share, then one all-gather) instead
                # of each rank reading the whole corpus
                part = ds.scan_lengths(rank, world)
                parts: List[Optional[Dict[int, list]]] = [None] * world
                dist.all_gather_object(parts, part)
                ds.adopt_lengths({i: n for p in parts for i, n in p.items()},
                                 write_cache=rank0)
            advice = gio_dataset_advice(len(ds), args.num_workers)
            if advice:
                log(f"[warn] {g}: {advice}")
    for g, ds in datasets.items():
        frozen = schema_freeze_warning(g, ds)
        if frozen is not None:
            if args.require_frozen_schema:
                # every rank sees the same shards: all of them exit here
                raise SystemExit(f"--require_frozen_schema: {frozen}")
            log(f"[WARN] {frozen}")
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
        saved = (resume.get("configs") or {}).get(g) if resume is not None else None
        if saved is not None:
            # the checkpoint defines each game's architecture (train_il_game's
            # resume invariant): a config default changed since it was written
            # (e.g. lux_sap_head) must not reshape the net under its weights;
            # fields newer than the checkpoint take their legacy values
            kw = checkpoint_config_dict(resume, g)
            changed = {k: v for k, v in core_kw.items() if k in kw and kw[k] != v}
            if changed:
                log(f"[warn] {g}: --override replaces checkpoint arch values: {changed}")
            kw.update(core_kw)
        else:
            kw = apply_meta_arch_defaults(g, dict(core_kw), datasets[g].shard_meta)
        net, dropped = build_net_tolerant(g, kw)
        if dropped:
            log(f"[warn] {g}: config fields unknown to {g}'s config, ignored: {dropped}")
        check_meta_arch(g, net.cfg, datasets[g].shard_meta)
        check_crop_policy(g, net.cfg, crop(g, args.max_len, game_max_len))
        if g in val_dirs:
            check_crop_policy(g, net.cfg, val_crop(g))
        nets[g] = net
    model = MultiGameNetwork(nets).to(device)
    first = model[games[0]]
    if core_state is not None:
        load_core(first, {k: v.to(device) for k, v in core_state.items()})
    if args.freeze_core:
        freeze_core(first, train_norms=args.train_core_norms)
    if resume is not None:
        model.load_state_dict(resume["train_model_state"])
    bn_sync = None
    if dist_env.enabled:
        # every rank initialised its per-game adapters from its own seed
        broadcast_module_state(model, src=0)
        # running statistics exist only in legacy BatchNorm encoders
        # (encoder_norm="batch"); the default GroupNorm keeps none
        bn_sync = BatchNormStatSync(model)
        if not bn_sync.bns:
            bn_sync = None

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wtag = f" world={world}" if dist_env.enabled else ""
    enc_norms = {g: describe_encoder_norm(model[g].cfg) for g in games}
    enc_norm = (next(iter(enc_norms.values())) if len(set(enc_norms.values())) == 1
                else ",".join(f"{g}:{v}" for g, v in enc_norms.items()))
    log(f"[init] games={games} core={first.cfg.core_type} moe={first.cfg.moe_experts} "
        f"qk_norm={first.cfg.qk_norm} encoder_norm={enc_norm} params={n_total:,} "
        f"trainable={n_train:,} device={device.type}{wtag} "
        f"value_head={describe_value_head(first.cfg)}")
    if args.init_core and not first.cfg.qk_norm:
        log(f"[note] qk_norm=False comes from the --init_core checkpoint "
            f"({args.init_core}): a pre-QK-norm core cannot switch it on")
    log("[mixture] " + "  ".join(f"{g}: frames={frames[g]:,} w={weights[g]:.3f}" for g in games))
    for g in games:
        if g in PREFIX_CROP:
            ml = crop(g, args.max_len, game_max_len)
            log(f"[data] {g}: {type(datasets[g]).__name__}, sequences cut to their first "
                f"{ml} frames" if ml is not None else
                f"[data] {g}: {type(datasets[g]).__name__}, FULL games (no --game_max_len "
                f"{g}=N): bound them if memory is short")
    log("[objective] " + "  ".join(
        f"{g}: tf={game_tf[g][0]:g}" + (f"→{game_tf[g][1]:g}" if game_tf[g][1] != game_tf[g][0]
                                         else "")
        + (f" λ={loss_weight[g]:g}" if g in loss_weight else "") for g in games))
    if dist_env.enabled:
        lo, hi = k_glob // world, -(-k_glob // world)
        log(f"[dist] micro-batches per optimizer step: {k_glob} = {world} ranks x "
            f"{lo if lo == hi else f'{lo}-{hi}'} (single-GPU equivalent: "
            f"--games_per_step {k_glob})")

    loaders = {g: _cycle(DataLoader(ds, batch_size=game_batch.get(g, args.batch_size),
                                    sampler=training_sampler(ds),
                                    num_workers=args.num_workers,
                                    collate_fn=collate_game_sequences))
               for g, ds in datasets.items()}
    has_val = any(g in games for g in val_dirs)

    def val_loader(g: str, d: str) -> DataLoader:
        ds = open_dataset(g, d, val_crop(g))
        bs = game_batch.get(g, args.batch_size)
        if dist_env.enabled:
            return _sharded_val_loader(ds, bs, rank, world, args.eval_batches)
        return DataLoader(ds, batch_size=bs, collate_fn=collate_game_sequences)

    val_loaders = {g: val_loader(g, d) for g, d in val_dirs.items() if g in games}

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamATan2(multi_game_param_groups(model, args.weight_decay), lr=args.lr,
                          betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, trm_lr_lambda(args.warmup_steps, args.steps, args.lr_min_ratio))
    # --shard_optimizer: this rank's partition of the optimizer state and EMA
    zero = (Zero1(model, optimizer, rank, world, bucket_mb=args.bucket_mb)
            if args.shard_optimizer else None)
    ema = None
    if args.ema_rate > 0:
        ema = (zero.make_ema(model, args.ema_rate) if zero is not None
               else ModelEMA(model, mu=args.ema_rate))
    if zero is not None:
        log(zero.describe(ema is not None))
    carries = {g: BatchCarry(model[g].supervision_steps, model[g].cfg.halt_exploration_prob)
               for g in games}
    start = 1
    if resume is not None:
        if zero is not None:
            zero.load_optimizer_state(resume["optimizer_state"])    # own partition only
        else:
            optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        if ema is not None and resume.get("ema_state") is not None:
            ema.load_state_dict(resume["ema_state"])
        start = resume["step"] + 1
        # fresh streams: the resumed run does not replay the first steps' draws
        random.seed(rank_seed(args.seed, rank, start))
        torch.manual_seed(rank_seed(args.seed, rank, start))
        log(f"[resume] {args.resume}: step {resume['step']} "
            f"(saved with world={resume.get('world_size', 1)})")
        resume = None                       # release the checkpoint tensors
    names, probs = list(weights), [weights[g] for g in weights]
    max_norm = args.clip_grad if args.clip_grad > 0 else float("inf")
    if rank0:
        os.makedirs(args.save_dir, exist_ok=True)

    def save(tag: str, step: int = 0, resumable: bool = False) -> Optional[str]:
        """Rank 0 writes ``multi_<tag>.pt``.  With --shard_optimizer EVERY rank
        calls this: the EMA and optimizer partitions are gathered onto rank 0
        (on the CPU, bucket by bucket) in the unsharded layout; the other
        ranks return None."""
        if zero is not None:
            ema_state = zero.gather_ema(ema) if ema is not None else None
            optimizer_state = zero.gather_optimizer_state() if resumable else None
            if not rank0:
                return None
            state = (zero.ema_model_state(model, ema_state) if ema is not None
                     else model.state_dict())
        else:
            ema_state = ema.state_dict() if ema is not None else None
            optimizer_state = optimizer.state_dict() if resumable else None
            state = ema.ema_model_state(model) if ema is not None else model.state_dict()
        pre = f"nets.{games[0]}."
        core = {k[len(pre):]: v for k, v in state.items()
                if k.startswith(pre) and is_core_key(k[len(pre):])}
        path = os.path.join(args.save_dir, f"multi_{tag}.pt")
        ck = {
            "games": games,
            "configs": {g: model[g].cfg.__dict__ for g in games},
            "core_config": core_config_overrides(model[games[0]].cfg),
            "core_state": core,
            "model_state": state,
            "train_model_state": model.state_dict(),
            "mixture": weights,
            "args": vars(args),
            "world_size": dist_env.world,
        }
        if resumable:
            ck.update(step=step, optimizer_state=optimizer_state,
                      scheduler_state=scheduler.state_dict(), ema_state=ema_state)
        # atomic: a job killed mid-write never leaves a truncated checkpoint
        torch.save(ck, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)
        return path

    model.train()
    t0 = time.time()
    window: Dict[str, List[float]] = {}
    for step in range(start, args.steps + 1):
        drawn = random.choices(names, weights=probs, k=K)
        for g in drawn:
            net = model[g]

            def forward_backward(merged, carry, active, reset, weight, g=g, net=net):
                mb = mb_from_merged(g, merged, args.il_legal_mask)
                loss, stats, pred = batch_loss(net, mb, device,
                                               args.value_weight, Q_HALT_WEIGHT, args.bf16,
                                               carry=carry, active=active, reset=reset,
                                               z_loss_weight=args.logit_z_loss,
                                               soft_target_weight=args.soft_target_weight,
                                               engine_value_weight=args.engine_value_weight)
                if torch.isfinite(loss):
                    # logs report the unweighted loss; λ_g scales the gradient
                    (loss * (loss_weight.get(g, 1.0) * weight / k_div)).backward()
                return pred, stats

            st = carries[g].step(prepare_rows(g, next(loaders[g]), device,
                                              tf_at(game_tf[g], step, args.steps),
                                              cfg=net.cfg, il_legal_mask=args.il_legal_mask),
                                 forward_backward)
            if st is not None and "loss" in st:
                window.setdefault(f"{g}/loss", []).append(st["loss"])
        if dist_env.enabled:
            # mean over the k_glob micro-batches of all ranks; a game's adapter
            # that no rank drew keeps grad None (AdamATan2 skips it, as
            # single-process)
            sync_gradients(model, world, warn_partial=False, bucket_mb=args.bucket_mb)
            grad_norm = collective_clip_grad_norm_(params, max_norm, device)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
        if torch.isfinite(grad_norm):
            if zero is not None:
                zero.step()     # own partition, then every owner broadcasts its update
            else:
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
            caps = qk_caps(first)
            if caps:
                extra += f"  qk_cap_max={max(caps.values()):.2f}"
            if device.type == "cuda":
                # peak allocated GiB since the last log line, max over ranks
                # (every rank takes part in the all-reduce)
                peak = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                                     torch.cuda.max_memory_reserved(device) / 2**30],
                                    device=device)
                if dist_env.enabled:
                    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                torch.cuda.reset_peak_memory_stats(device)
                extra += f"  mem_peak={peak[0].item():.1f}/{peak[1].item():.1f}G"
            log(f"[step {step}/{args.steps}] lr={scheduler.get_last_lr()[0]:.2e} "
                f"gnorm={float(grad_norm):.3f} {msg}{extra}  ({time.time() - t0:.0f}s)",
                flush=True)
            window = {}
        if has_val and (step % args.eval_every == 0 or step == args.steps):
            if zero is not None and ema is not None:
                # the gathered EMA swapped into the live model (no model copy)
                ev_ctx = zero.ema_weights(model, ema)
            else:
                ev_ctx = contextlib.nullcontext(ema.ema_copy(model) if ema is not None
                                                else model)
            diag: Dict[str, str] = {}
            # only the flags that differ from eval_sums' defaults: an unused
            # flag leaves the historical evaluate(...) call unchanged
            loss_kw = {k: v for k, v in (("il_legal_mask", args.il_legal_mask),
                                         ("soft_target_weight", args.soft_target_weight),
                                         ("engine_value_weight", args.engine_value_weight))
                       if v != EVAL_LOSS_DEFAULTS[k]}
            with ev_ctx as ev_model:
                if dist_env.enabled:
                    res = evaluate_sharded(ev_model, val_loaders, device, args.value_weight,
                                           args.bf16, **loss_kw)
                else:
                    res = evaluate(ev_model, val_loaders, device, args.value_weight,
                                   args.bf16, args.eval_batches, diag=diag, **loss_kw)
            del ev_model, ev_ctx
            for g, m in res.items():
                keys = [k for k in sorted(m) if k.endswith(("loss", "acc", "mse", "kl"))]
                log(f"[val {step}] {g}: " + " ".join(f"{k}={m[k]:.4f}" for k in keys),
                    flush=True)
                if diag.get(g):
                    log(f"[attn {step}] {g}: {diag[g]}", flush=True)
        # sharded: every rank takes part in the gather, rank 0 writes
        writes = rank0 or zero is not None
        if args.ckpt_every and (step % args.ckpt_every == 0 or step == args.steps) and writes:
            save("latest", step, resumable=True)
    final = save("final") if rank0 or zero is not None else None
    if rank0:
        print(f"[done] saved {final}")
    dist_env.barrier()
    dist_env.destroy()
    return {"model": model, "ema": ema, "optimizer": optimizer, "dist": dist_env,
            "zero": zero}


if __name__ == "__main__":
    main()
