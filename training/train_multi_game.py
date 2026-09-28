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

    w_g ∝ m_g · min(N_g, C_g)^α     (N_g = frames of game g, α = --sample_alpha)

so the expected step gradient is ∇ Σ_g w_g λ_g L_g, where L_g is game g's
mean per-frame loss (decoder IL loss + value loss) and λ_g its
``--game_loss_weight`` (default 1).  α = 1 weighs every frame equally
(concatenated data), α = 0 weighs games equally, α ≈ 0.5 is the usual
temperature compromise (HPT uses √N, π0 N^0.43).  New games bring corpora of
1e8-1e10 frames: C_g (``--mixture_frame_cap N``, per game ``--game_frame_cap
GAME=N``; default none) caps the frames a game counts with, so one giant corpus
cannot dominate however large it grows (e.g. cap 1e8 with α = 0.5: a 1e10-frame
game weighs like a 1e8 one, 10x a 1e6-frame game), and m_g (``--game_weight
GAME=W``, default 1) multiplies a game's draw probability explicitly.
Small corpora are repeated: over the run game g is expected to pass over its
frames E_g = steps · K · w_g · B_g · F̄_g / R_g times (B_g its batch size, F̄_g
its mean frames per sample after the crop, R_g its uncapped frames) — logged
per game as ``epochs``.  ``--max_epochs E`` bounds every E_g by E: the
KL-closest mixture to the temperature one under w_g ≤ u_g (u_g the w_g of
E_g = E) is w'_g = min(u_g, c·w_g) with c normalising (water-filling; the
games under their cap keep their ratios), so the steps a small corpus cannot
use without memorising go to the games that still have unseen data
(repeating data beyond ~4-16 epochs is worth little: Muennighoff et al. 2023,
"Scaling Data-Constrained Language Models").  The chosen weights are logged
("[mixture]") and saved in the checkpoint.  The decoders' losses are not
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
step, the optimizer's parameter names; written atomically).  ``--resume
<that file>`` continues from it with the same arguments, any world size,
sharded or not: weights, optimizer moments, EMA, balance biases, (legacy)
BatchNorm statistics and the LR schedule continue exactly; game draws and data order
restart from fresh RNG streams, and a TRM core's deep-supervision carries
restart empty.  Each game's network is rebuilt from the config saved in the
checkpoint (``--override`` values on top), so a config default that changed
since it was written cannot reshape it.

Games can JOIN a running pretraining: ``--resume`` with ``--data`` naming a
superset of the checkpoint's games, in any order (removing a game is
refused).  Nothing is matched by position — games by name, tensors by
parameter name (a shared core tensor has one name per game; its canonical
one, the EMA's and the optimizer's, lies under the run's first game):

    # chess joins the othello + go run: the same command, one --data more
    uv run torchrun --standalone --nproc_per_node=4 -m training.train_multi_game \\
        --data othello=... --data chess=data/chess/train --data go=... \\
        --resume checkpoints/multi/multi_latest.pt <the run's other arguments>

- an added game is built from the command line and its shard meta, every
  shared-core field (CORE_CONFIG_FIELDS) forced to the checkpoint's core as
  with --init_core: an --override contradicting that core is refused;
- the checkpoint's weights and buffers load exactly; it may lack exactly the
  added games' keys, which keep their fresh initialisation;
- the optimizer state moves by parameter name to the new layout (names saved
  in the checkpoint, rebuilt on the meta device from its configs for older
  checkpoints and checked against them): moments and step counters of every
  old parameter, the param groups (decay / QK-norm no-decay) with the
  checkpoint's hyper-parameters; the EMA shadows likewise.  The added games'
  parameters start without optimizer state (step counter 1, fresh bias
  correction) and with their initial weights as EMA shadows;
- --shard_optimizer: the partition is the new model's, every rank loads its
  share of the remapped state (from sharded or unsharded checkpoints);
- step and LR schedule continue (no new warmup: the added games train at the
  current LR), --freeze_core / --train_core_norms must be the checkpoint's;
  the mixture weights are recomputed for the new game set, the log shows the
  checkpoint's and the new ones.  The default --games_per_step is the number
  of games: pass it (or --global_games_per_step) to keep the batch size.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from data_processing.games.dataset import (WholeRecordBatchSampler, collate_game_sequences,
                                           training_sampler)
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
    wsd_lr_lambda,
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
    chunked_batch_loss,
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
from training.zero import MOMENT_KEYS, Zero1

#: resumable checkpoints: the canonical names of the optimizer's parameters,
#: per param group in ``optimizer.state_dict()`` index order — what a resume
#: matches the saved optimizer state by (reconstructed for older checkpoints,
#: :func:`checkpoint_param_names`)
OPT_NAMES_KEY = "optimizer_param_names"


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
    p.add_argument("--mixture_frame_cap", type=frame_count, default=None, metavar="N",
                   help="frames counted per game in the mixture weights: w_g ∝ "
                        "min(frames_g, N)^alpha, so a giant corpus (1e9-1e10 frames) cannot "
                        "dominate the mixture (default: no cap)")
    p.add_argument("--game_frame_cap", action="append", default=[], metavar="GAME=N",
                   help="per-game --mixture_frame_cap (N may be written 1e8)")
    p.add_argument("--game_weight", action="append", default=[], metavar="GAME=W",
                   help="multiplier of a game's mixture weight (draw probability), "
                        "w_g ∝ W_g * min(frames_g, cap)^alpha (default 1); unlike "
                        "--game_loss_weight it changes how often the game is drawn, not the "
                        "scale of its gradient")
    p.add_argument("--max_epochs", type=float, default=None, metavar="E",
                   help="cap every game's expected passes over its frames during the run "
                        "(steps x games per step x draw probability x batch x frames per "
                        "sample / corpus frames) at E: capped games give their excess draw "
                        "probability to the others, in proportion (default: no cap)")
    p.add_argument("--max_len", type=int, default=None,
                   help="crop training sequences to a random window (not generals: "
                        "with generals in the run, bound it via --game_max_len); for a "
                        "game with a frame budget (--frames_per_batch / --game_frames) "
                        "a cap: records keep their first max_len frames")
    p.add_argument("--frames_per_batch", type=int, default=0, metavar="N",
                   help="whole-record training: every game's micro-batches hold WHOLE "
                        "records from their first frame (each trained frame carries its "
                        "record's full history, as in play), as many similar-length "
                        "records as keep records x the longest within N frames "
                        "(WholeRecordBatchSampler; --batch_size is then ignored). 0 = off")
    p.add_argument("--game_frames", action="append", default=[], metavar="GAME=N",
                   help="per-game --frames_per_batch (0 = that game keeps --batch_size "
                        "windows)")
    p.add_argument("--chunk_len", type=int, default=0, metavar="N",
                   help="a micro-batch longer than N frames (a whole record longer than "
                        "its game's frame budget, kept up to --max_len) is trained in "
                        "chunks of N frames, in order, each continuing from the memory "
                        "state the previous one ended in (truncated backpropagation "
                        "through time: one chunk's activations, the gradient cut at chunk "
                        "boundaries; train_il_game.chunked_batch_loss). 0 = whole")
    p.add_argument("--game_chunk_len", action="append", default=[], metavar="GAME=N",
                   help="per-game --chunk_len")
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
    p.add_argument("--lr_schedule", choices=("cosine", "wsd"), default="cosine",
                   help="cosine: the reference warmup + cosine to --lr_min_ratio over --steps "
                        "(constant with the default ratio 1); wsd: warmup, constant, then a "
                        "decay to --lr_min_ratio over the last --lr_decay_frac of --steps "
                        "(extend a run by resuming with a larger --steps)")
    p.add_argument("--lr_decay_frac", type=float, default=0.1,
                   help="wsd: fraction of --steps the final decay spans")
    p.add_argument("--lr_decay_shape", choices=("cosine", "linear"), default="cosine")
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
    p.add_argument("--stop_file", default=None,
                   help="a file holding a step S (checked every 10 steps; empty = now): the run "
                        "writes its resumable multi_latest.pt at step S — at the first check after "
                        "the file appears when S has passed — and exits without the final save; "
                        "scripts/slurm/multi/multi_game_ddp_job.sh then continues in the same "
                        "allocation with the arguments of $OUT/handover.env (a data change without "
                        "giving the GPUs back to the queue)")
    p.add_argument("--resume", default=None,
                   help="continue from a multi_latest.pt (same arguments otherwise; --data "
                        "may add games, in any order: see Resuming in the module docstring)")
    p.add_argument("--reinit_game", action="append", default=[], metavar="GAME[:TAG][,...]",
                   help="with --resume: rebuild these games' own modules (encoder, decoder, "
                        "value head) from the command line and their shard meta, fresh, around "
                        "the checkpoint's core — as if they joined the run now (a game whose "
                        "adapter changed).  Once per GAME:TAG: the checkpoints record it "
                        "(reinit_log), so resuming the same command again keeps the weights; "
                        "a new TAG reinitialises again.  Repeatable")
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


def frame_count(s: str) -> int:
    """A frame count on the command line; scientific notation allowed (``1e8``)."""
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a frame count: {s!r}") from None
    if not (v >= 0 and v == int(v)):
        raise argparse.ArgumentTypeError(f"frame count must be a whole number >= 0: {s!r}")
    return int(v)


def mixture_weights(frames: Dict[str, int], alpha: float, *,
                    frame_cap: Optional[int] = None,
                    game_frame_cap: Optional[Dict[str, int]] = None,
                    game_weight: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """w_g ∝ m_g · min(N_g, C_g)^α, normalised.

    ``N_g`` = frames of game g; ``C_g`` = its cap (``game_frame_cap[g]``, else
    ``frame_cap``, else none): a corpus of 1e10 frames counts as C_g, so it cannot
    swamp the mixture however large it grows.  ``m_g`` = ``game_weight[g]``
    (default 1): an explicit multiplier of the game's draw probability.  Without
    caps / multipliers this is the plain temperature mixture w_g ∝ N_g^α."""
    caps = dict(game_frame_cap or {})
    mult = dict(game_weight or {})
    raw = {}
    for g, n in frames.items():
        cap = caps.get(g, frame_cap)
        eff = n if cap is None else min(n, cap)
        raw[g] = float(mult.get(g, 1.0)) * float(max(eff, 1)) ** alpha
    z = sum(raw.values())
    if not z > 0:
        raise SystemExit(f"mixture weights sum to {z}: every --game_weight is 0?")
    return {g: v / z for g, v in raw.items()}


def expected_epochs(weights: Dict[str, float], stats: Dict[str, Dict[str, int]],
                    batch: Dict[str, int], draws: float) -> Dict[str, float]:
    """E_g = draws · w_g · B_g · (frames_g / samples_g) / raw_frames_g: the
    expected passes over game g's (uncapped) frames when ``draws`` micro-batches
    (steps × global games per step) are drawn from ``weights``; ``stats`` are
    the datasets' ``frame_stats()`` (``frames`` capped at the crop)."""
    out = {}
    for g, w in weights.items():
        st = stats[g]
        per_draw = batch[g] * st["frames"] / max(st["samples"], 1)
        out[g] = draws * w * per_draw / max(st["raw_frames"], 1)
    return out


def cap_mixture(weights: Dict[str, float], upper: Dict[str, float]
                ) -> Tuple[Dict[str, float], List[str]]:
    """The KL-closest distribution to ``weights`` with w_g ≤ ``upper[g]``:
    w'_g = min(upper_g, c·w_g), c normalising (found by water-filling; the
    games below their bound keep their ratios).  Returns (weights, the games
    held at their bound).  If the bounds sum to less than 1 (every game
    capped) they are scaled up together: w'_g ∝ upper_g."""
    capped: Dict[str, float] = {}
    while True:
        free = {g: v for g, v in weights.items() if g not in capped}
        mass = 1.0 - sum(capped.values())
        z = sum(free.values())
        if not free or not z > 0 or not mass > 0:
            break
        scaled = {g: v * mass / z for g, v in free.items()}
        over = [g for g, v in scaled.items() if v > upper.get(g, math.inf)]
        if not over:
            return {**capped, **scaled}, sorted(capped)
        for g in over:
            capped[g] = upper[g]
    total = sum(upper[g] for g in weights)
    return {g: upper[g] / total for g in weights}, sorted(weights)


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


def multi_checkpoint_core(ck: Dict) -> Dict:
    """The shared-core fields (core config overrides) of a multi-game
    checkpoint: its ``core_config`` over the full saved config of its first
    game — the network whose modules the core was — so that core fields newer
    than the code that wrote ``core_config`` (e.g. moe_bias_clip) still come
    from the checkpoint.  {} for a checkpoint with neither."""
    games = ck.get("games") or []
    first = (ck.get("configs") or {}).get(games[0]) if games else None
    src = {**(first or {}), **(ck.get("core_config") or {})}
    return core_config_overrides(src) if src else {}


def load_core_checkpoint(path: str):
    """(core_state, core config overrides) from a multi-game checkpoint
    (``core_state`` + its core fields, :func:`multi_checkpoint_core`) or a
    single-game one (``model_state`` + ``config``; its core keys are
    extracted)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "core_state" in ck:
        return ck["core_state"], multi_checkpoint_core(ck)
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


# ----------------------------------------------------------------------
# resume, possibly with games added: everything matched by game and
# parameter NAME, never by position
# ----------------------------------------------------------------------

def _split_name(name: str) -> Tuple[str, str]:
    """``nets.<game>.<rest>`` (a MultiGameNetwork parameter / buffer name)
    → (game, rest); ("", name) for anything else."""
    parts = name.split(".", 2)
    if len(parts) < 3 or parts[0] != "nets":
        return "", name
    return parts[1], parts[2]


def _is_core_memory(name: str) -> bool:
    """A parameter / buffer of the core's episode-memory sub-layers
    (model.components.memory) — zero-output modules a resume may add to a core."""
    return _split_name(name)[1].startswith("core.memory.")


def _is_own(name: str, games) -> bool:
    """True for a parameter / buffer of one of ``games``' own modules
    (encoder, decoder, value head), not of the shared core."""
    game, rest = _split_name(name)
    return game in games and not is_core_key(rest)


def canonical_names(model: torch.nn.Module) -> Dict[str, str]:
    """Every name of every parameter of ``model`` → its canonical name, the
    first one in ``named_parameters()`` (the name ModelEMA keys its shadow
    and :func:`optimizer_param_names` records).  A shared core tensor has one
    name per game and its canonical one lies under the FIRST game of the
    run, so a checkpoint's names resolve to this model's tensors whatever
    the --data order and whichever games were added."""
    canon = {id(p): n for n, p in model.named_parameters()}
    return {n: canon[id(p)] for n, p in model.named_parameters(remove_duplicate=False)}


def optimizer_param_names(model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer) -> List[List[str]]:
    """Canonical names of ``optimizer``'s parameters per param group, in
    ``optimizer.state_dict()`` index order (saved as :data:`OPT_NAMES_KEY`)."""
    canon = {id(p): n for n, p in model.named_parameters()}
    return [[canon[id(p)] for p in g["params"]] for g in optimizer.param_groups]


def _layout_mismatch(ck: Dict, groups: List[List[torch.Tensor]],
                     trainable: Optional[List[str]] = None) -> Optional[str]:
    """Why ``groups`` (per param group, the parameters in index order) cannot
    be the layout of the checkpoint's optimizer, or None: the group sizes and
    the moment shapes must agree and, given ``trainable`` (the trainable
    parameters' names in model order), the names of the checkpoint's EMA."""
    opt = ck["optimizer_state"]
    sizes = [len(g["params"]) for g in opt["param_groups"]]
    if [len(g) for g in groups] != sizes:
        return f"{[len(g) for g in groups]} parameters per group, the optimizer state has {sizes}"
    shape = dict(zip((int(i) for g in opt["param_groups"] for i in g["params"]),
                     (tuple(p.shape) for g in groups for p in g)))
    bad = [i for i, st in opt["state"].items()
           if any(tuple(st[k].shape) != shape[int(i)] for k in MOMENT_KEYS if k in st)]
    if bad:
        return f"moment shapes differ at optimizer indices {bad[:5]}"
    ema = ck.get("ema_state")
    if trainable is not None and ema is not None and list(ema) != trainable:
        return "the EMA's parameter names differ from the trainable parameters'"
    return None


def checkpoint_param_names(ck: Dict, where: str = "checkpoint",
                           fallback_configs: Optional[Dict[str, Dict]] = None
                           ) -> List[List[str]]:
    """The names of a resumable checkpoint's optimizer parameters, per param
    group in the index order of its ``optimizer_state``.

    Saved in the checkpoint (:data:`OPT_NAMES_KEY`), or — for checkpoints
    written before it was — reconstructed the way that run built its
    optimizer: the checkpoint's MultiGameNetwork rebuilt on the meta device
    from its saved configs (its games, in its order; ``fallback_configs``
    for a checkpoint without configs), frozen as its run was (saved
    ``args``: --freeze_core / --train_core_norms) and grouped by
    :func:`multi_game_param_groups`.  The reconstruction must reproduce the
    saved group sizes and moment shapes, and with an EMA its names in model
    order (every trainable parameter), else the checkpoint is refused."""
    names = ck.get(OPT_NAMES_KEY)
    if names is not None:
        names = [list(n) for n in names]
        sizes = [len(g["params"]) for g in ck["optimizer_state"]["param_groups"]]
        if [len(n) for n in names] != sizes:
            raise SystemExit(f"{where}: {OPT_NAMES_KEY} ({[len(n) for n in names]} names per "
                             f"group) does not match its optimizer state ({sizes})")
        return names
    games = list(ck["games"])
    configs = {**(fallback_configs or {}), **(ck.get("configs") or {})}
    run = ck.get("args") or {}
    # no RNG side effects (a meta build draws nothing; stay safe anyway)
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        old = MultiGameNetwork({g: build_net_tolerant(g, upgrade_config_dict(dict(configs[g])))[0]
                                for g in games})
    if run.get("freeze_core"):
        freeze_core(old[games[0]], train_norms=bool(run.get("train_core_norms")))
    groups = [g["params"] for g in multi_game_param_groups(old, 0.0)]
    why = _layout_mismatch(ck, groups, [n for n, p in old.named_parameters() if p.requires_grad])
    if why is not None:
        raise SystemExit(f"{where} predates saved optimizer parameter names ({OPT_NAMES_KEY}) "
                         f"and rebuilding its model from its saved configs does not reproduce "
                         f"its optimizer ({why}); resume it with its own games and code first")
    canon = {id(p): n for n, p in old.named_parameters()}
    return [[canon[id(p)] for p in g] for g in groups]


def forget_games(ck: Dict, games: List[str], where: str = "checkpoint") -> Dict:
    """``ck`` as if it had never trained ``games``' own modules (--reinit_game): the
    resume then treats them as games joining the run — built from the command line
    and their shard meta around the checkpoint's core, their own parameters fresh
    (no optimizer state, EMA = their initial weights).  Dropped: their configs, every
    weight and buffer under their names (their aliases of the shared core too — the
    core loads through the other games' names), the optimizer state and EMA shadows
    of their own parameters.  The core's optimizer state and shadows stay, also where
    their canonical names are one of these games' (the run's first game)."""
    unknown = [g for g in games if g not in ck["games"]]
    if unknown:
        raise SystemExit(f"--reinit_game {unknown}: not trained by {where} ({ck['games']})")
    keep = [g for g in ck["games"] if g not in games]
    if not keep:
        raise SystemExit("--reinit_game: at least one game must keep its weights (the core "
                         "loads through them)")
    names = ck.get(OPT_NAMES_KEY)
    if names is None:
        raise SystemExit(f"--reinit_game: {where} predates saved optimizer parameter names "
                         f"({OPT_NAMES_KEY}); resume it once without --reinit_game first")
    out = dict(ck)
    out["games"] = keep
    out["configs"] = {g: c for g, c in (ck.get("configs") or {}).items() if g not in games}
    out["train_model_state"] = {k: v for k, v in ck["train_model_state"].items()
                                if _split_name(k)[0] not in games}
    opt = ck["optimizer_state"]
    saved_state = {int(i): st for i, st in opt["state"].items()}
    state: Dict[int, Dict] = {}
    groups, group_names_out, n = [], [], 0          # n: the new flat parameter index
    for group_names, sg in zip(names, opt["param_groups"]):
        idx, kept = [], []
        for name, i in zip(group_names, sg["params"]):
            if _is_own(name, games):
                continue
            if int(i) in saved_state:
                state[n] = saved_state[int(i)]
            idx.append(n)
            kept.append(name)
            n += 1
        groups.append({**sg, "params": idx})
        group_names_out.append(kept)
    out["optimizer_state"] = {"state": state, "param_groups": groups}
    out[OPT_NAMES_KEY] = group_names_out
    if ck.get("ema_state") is not None:
        out["ema_state"] = {n: v for n, v in ck["ema_state"].items() if not _is_own(n, games)}
    return out


def load_resume_weights(model: MultiGameNetwork, state: Dict[str, torch.Tensor],
                        added: List[str], where: str = "checkpoint") -> None:
    """Load a checkpoint's ``train_model_state`` into ``model``, whose games
    are the checkpoint's plus ``added``: every parameter and buffer of the
    checkpoint's games and of the shared core (MoE balance biases, BatchNorm
    statistics included) loads exactly.  The keys the checkpoint may lack
    are exactly the added games' — their own modules, and their names of the
    shared core, which is loaded through the checkpoint games' names — and
    nothing may be left over."""
    new = tuple(f"nets.{g}." for g in added)
    keys = set(model.state_dict())
    # decided on the key sets, not on load_state_dict's report: a module the
    # checkpoint lacks has no version metadata there, and _BatchNorm then
    # fills its num_batches_tracked from the module itself (unreported)
    missing, unexpected = keys - set(state), set(state) - keys
    want = {k for k in keys if k.startswith(new)}
    # core memory sub-layers joining the run: absent from a checkpoint that has none
    # (their zero outputs leave the loaded network's function unchanged)
    if not any(_is_core_memory(k) for k in state):
        want |= {k for k in keys if _is_core_memory(k)}
    core = {_split_name(k)[1] for k in want
            if is_core_key(_split_name(k)[1]) and not _is_core_memory(k)}
    loaded_core = {_split_name(k)[1] for k in state if is_core_key(_split_name(k)[1])}
    if missing != want or unexpected or not core <= loaded_core:
        raise SystemExit(
            f"{where}: its weights do not fit this run's networks beyond the added games "
            f"{added}: missing {sorted(missing - want)[:5]}, unexpected {sorted(unexpected)[:5]}, "
            f"added games' keys present {sorted(want - missing)[:5]}, core keys without "
            f"weights {sorted(core - loaded_core)[:5]}")
    res = model.load_state_dict(state, strict=False)      # shapes: always strict
    assert not res.unexpected_keys and set(res.missing_keys) <= want, res


def resume_training_state(ck: Dict, model: MultiGameNetwork, optimizer: torch.optim.Optimizer,
                          where: str = "checkpoint"
                          ) -> Tuple[Dict, Optional[Dict[str, torch.Tensor]], List[str]]:
    """The checkpoint's optimizer state and EMA shadows in THIS run's layout,
    matched by parameter NAME (the run's games are the checkpoint's plus any
    added ones, in any --data order): ``(optimizer state dict, EMA state,
    fresh)``.  The checkpoint's names are its saved ones, or rebuilt
    (:func:`checkpoint_param_names`) — except for an older checkpoint of
    exactly this run's games in this order, whose layout is this run's own.

    The optimizer state dict has the unsharded ``optimizer.state_dict()``
    format for ``optimizer.load_state_dict`` or ``Zero1.load_optimizer_state``
    (which keeps a rank's partition of the NEW model): each checkpoint
    parameter's moments and step counter move to its index here, its param
    group keeping its role (the decayed parameters, the QK-norm scales' no-
    decay group) and the checkpoint's group hyper-parameters (lr, initial_lr,
    betas, weight_decay).  The EMA state is keyed by this model's canonical
    names (``ModelEMA.load_state_dict``; a partial EMA takes its own), None
    without a checkpoint EMA.  ``fresh`` names the trainable parameters
    without checkpoint state — which must be exactly the added games' own (and core
    memory sub-layers the checkpoint lacks):
    no optimizer state (their step counters start at 1, with the matching
    bias correction), and an EMA initialised from their fresh weights.
    Refused: a checkpoint parameter frozen here or in another group, a
    trainable parameter of an old game the checkpoint did not train (changed
    --freeze_core / --train_core_norms), mismatched shapes."""
    saved = ck["optimizer_state"]
    groups = optimizer.param_groups
    if OPT_NAMES_KEY not in ck and list(ck["games"]) == model.games:
        # an older checkpoint of exactly these games in this order: its
        # optimizer layout is this run's own (as the resume assumed before the
        # names were saved), no rebuild needed — sizes and shapes still checked
        why = _layout_mismatch(ck, [g["params"] for g in groups])
        if why is not None:
            raise SystemExit(f"{where}: its optimizer state does not fit this run's parameters "
                             f"({why}; resume with the checkpoint's --freeze_core / "
                             f"--train_core_norms)")
        names = optimizer_param_names(model, optimizer)
    else:
        names = checkpoint_param_names(
            ck, where, fallback_configs={g: model[g].cfg.__dict__ for g in model.games})
    if len(groups) != len(saved["param_groups"]):
        raise SystemExit(f"{where}: {len(saved['param_groups'])} optimizer param group(s), this "
                         f"run has {len(groups)} (weight-decay grouping changed)")
    saved_state = {int(i): st for i, st in saved["state"].items()}
    alias = dict(model.named_parameters(remove_duplicate=False))
    canon = canonical_names(model)
    slot: Dict[int, Tuple[int, int]] = {}            # id(p) -> (group, index here)
    for j, g in enumerate(groups):
        for p in g["params"]:
            slot[id(p)] = (j, len(slot))
    state: Dict[int, Dict] = {}
    trained: Dict[int, str] = {}                     # id(p) -> checkpoint name
    for j, (group_names, sg) in enumerate(zip(names, saved["param_groups"])):
        for name, i in zip(group_names, sg["params"]):
            p = alias.get(name)
            if p is None:
                raise SystemExit(f"{where}: optimizer parameter {name} is not a parameter of "
                                 f"this run's networks")
            if id(p) in trained:
                raise SystemExit(f"{where}: optimizer parameters {trained[id(p)]} and {name} "
                                 f"are the same tensor here")
            trained[id(p)] = name
            if id(p) not in slot:
                raise SystemExit(f"{where}: {name} is trained in the checkpoint but frozen in "
                                 f"this run (resume with the checkpoint's --freeze_core / "
                                 f"--train_core_norms)")
            jj, k = slot[id(p)]
            if jj != j:
                raise SystemExit(f"{where}: {name} is in optimizer param group {j} there, "
                                 f"{jj} here (weight-decay grouping changed)")
            st = saved_state.get(int(i))
            if st is not None:
                bad = [m for m in MOMENT_KEYS if tuple(st[m].shape) != tuple(p.shape)]
                if bad:
                    raise SystemExit(f"{where}: {name} {bad} has shape {tuple(st[bad[0]].shape)}, "
                                     f"the parameter {tuple(p.shape)}")
                state[k] = st
    added = [g for g in model.games if g not in ck["games"]]
    own_name = {id(p): n for n, p in model.named_parameters()}
    fresh = [own_name[id(p)] for g in groups for p in g["params"] if id(p) not in trained]
    stray = [n for n in fresh if not _is_own(n, added) and not _is_core_memory(n)]
    if stray:
        raise SystemExit(f"{where}: {len(stray)} trainable parameter(s) of the checkpoint's "
                         f"games are not in its optimizer, e.g. {stray[:3]} (resume with the "
                         f"checkpoint's --freeze_core / --train_core_norms)")
    out_groups, start = [], 0
    for sg, g in zip(saved["param_groups"], groups):
        hyper = {k: v for k, v in sg.items() if k not in ("params", "param_names")}
        out_groups.append({**hyper, "params": list(range(start, start + len(g["params"])))})
        start += len(g["params"])
    ema = None
    if ck.get("ema_state") is not None:
        ema = {}
        for n, v in ck["ema_state"].items():
            c = canon.get(n)
            if c is None or c in ema or tuple(v.shape) != tuple(alias[c].shape):
                raise SystemExit(f"{where}: EMA shadow {n} does not fit this run's networks "
                                 f"(unknown, duplicate or reshaped)")
            ema[c] = v
        want = {canon[n] for n in trained.values()}
        if set(ema) != want:
            raise SystemExit(f"{where}: its EMA shadows {len(ema)} parameters, its optimizer "
                             f"{len(want)}: e.g. {sorted(set(ema) ^ want)[:3]}")
    return {"state": state, "param_groups": out_groups}, ema, fresh


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


def stop_step(path: Optional[str]) -> int:
    """The step a ``--stop_file`` asks the run to stop at: its integer content, 1 (= now) when
    it is empty or unreadable, 0 without the file."""
    if not path or not os.path.exists(path):
        return 0
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return 1
    try:
        return max(int(text), 1) if text else 1
    except ValueError:
        return 1


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
    game_frames = {g: int(n) for g, n in _pairs(args.game_frames, "N").items()}
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
    ck_games: List[str] = []
    added: List[str] = []            # games this run adds to the resumed checkpoint's
    reinit_log: List[str] = []       # GAME:TAG entries --reinit_game applied in this line
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer_state" not in resume:
            raise SystemExit(f"{args.resume} is not resumable (only --ckpt_every writes "
                             f"multi_latest.pt with the optimizer state)")
        requested = [e for arg in args.reinit_game for e in arg.split(",") if e]
        done_before = list(resume.get("reinit_log") or [])
        todo = [e for e in requested if e not in done_before]
        if requested and not todo:
            log(f"[resume] --reinit_game {requested}: done by an earlier run of this "
                f"checkpoint's line (reinit_log {done_before}); weights kept")
        if todo:
            resume = forget_games(resume, [e.split(":", 1)[0] for e in todo], args.resume)
            log(f"[resume] --reinit_game {todo}: their own modules start fresh around the "
                f"checkpoint's core")
        reinit_log = done_before + todo
        ck_games = list(resume["games"])
        removed = [g for g in ck_games if g not in games]
        if removed:
            raise SystemExit(f"{args.resume} trained games {ck_games}, this run (--data) lacks "
                             f"{removed}: a resume may add games, not remove them")
        added = [g for g in games if g not in ck_games]

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

    def frames_of(g: str) -> int:
        """g's whole-record frame budget (0 = --batch_size windows)."""
        return game_frames.get(g, args.frames_per_batch)

    game_chunk = {g: int(n) for g, n in _pairs(args.game_chunk_len, "N").items()}

    def chunk_of(g: str) -> int:
        """g's truncated-BPTT chunk length (0 = micro-batches trained whole)."""
        return game_chunk.get(g, args.chunk_len)

    datasets = {g: open_dataset(g, d, crop(g, args.max_len, game_max_len),
                                whole_records=frames_of(g) > 0)
                for g, d in data_dirs.items()}
    whole = {g: WholeRecordBatchSampler(ds, frames_of(g))
             for g, ds in datasets.items() if frames_of(g) > 0}
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
    frame_stats = {g: ds.frame_stats() for g, ds in datasets.items()}
    frames = {g: st["frames"] for g, st in frame_stats.items()}
    try:
        game_frame_cap = {g: frame_count(n) for g, n in _pairs(args.game_frame_cap, "N").items()}
    except argparse.ArgumentTypeError as e:
        raise SystemExit(f"--game_frame_cap: {e}") from None
    game_weight = {g: float(w) for g, w in _pairs(args.game_weight, "W").items()}
    for flag, vals in (("--game_frame_cap", game_frame_cap), ("--game_weight", game_weight)):
        unknown = sorted(set(vals) - set(games))
        if unknown:
            raise SystemExit(f"{flag} for games without --data: {unknown}")
        if any(v < 0 for v in vals.values()):
            raise SystemExit(f"{flag} values must be >= 0")
    weights = mixture_weights(frames, args.sample_alpha, frame_cap=args.mixture_frame_cap,
                              game_frame_cap=game_frame_cap, game_weight=game_weight)
    batch_of = {g: (whole[g].records_per_batch() if g in whole
                    else game_batch.get(g, args.batch_size)) for g in games}
    draws = float(args.steps) * k_glob
    epoch_capped: List[str] = []
    if args.max_epochs is not None:
        if not args.max_epochs > 0:
            raise SystemExit("--max_epochs must be > 0")
        unit = expected_epochs({g: 1.0 for g in games}, frame_stats, batch_of, draws)
        weights, epoch_capped = cap_mixture(
            weights, {g: args.max_epochs / unit[g] for g in games})
        if len(epoch_capped) == len(games):
            log(f"[warn] --max_epochs {args.max_epochs:g}: every game is capped (the run "
                f"draws more than its corpora hold at this cap): weights ∝ the caps")
    epochs = expected_epochs(weights, frame_stats, batch_of, draws)

    core_kw = parse_overrides(args.override)
    core_state = None
    if args.init_core:
        core_state, fixed = load_core_checkpoint(args.init_core)
        clash = {k: v for k, v in core_kw.items() if k in fixed and fixed[k] != v}
        if clash:
            raise SystemExit(f"--override {clash} contradicts the core in {args.init_core}")
        core_kw = {**core_kw, **fixed}
    ck_core: Dict = {}
    if added:
        # games joining a resumed run are built around ITS core, like
        # --init_core: every core field is the checkpoint's, and an
        # --override that contradicts it would reshape the core under the
        # old games' weights or split it from the new games'
        ck_core = multi_checkpoint_core(resume)
        clash = {k: v for k, v in core_kw.items() if k in ck_core and ck_core[k] != v}
        if clash:
            have = {k: ck_core[k] for k in clash}
            raise SystemExit(f"--override {clash} contradicts the core in {args.resume} "
                             f"({have}): the games added on resume {added} share the "
                             f"checkpoint's core")

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
            # fresh runs and games added on resume: command line + shard meta
            kw = apply_meta_arch_defaults(g, {**core_kw, **ck_core} if g in added
                                          else dict(core_kw), datasets[g].shard_meta)
        net, dropped = build_net_tolerant(g, kw)
        if dropped:
            log(f"[warn] {g}: config fields unknown to {g}'s config, ignored: {dropped}")
        if g in added:
            own = core_config_overrides(net.cfg)
            off = {k: (own.get(k), v) for k, v in ck_core.items() if own.get(k) != v}
            if off:
                raise SystemExit(f"{g} (added on resume): its network's core differs from the "
                                 f"checkpoint's, field: (this net, checkpoint) {off}")
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
        # by name: the added games' own modules keep their fresh weights
        load_resume_weights(model, resume["train_model_state"], added, args.resume)
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
    def _mix_note(g: str) -> str:
        cap = game_frame_cap.get(g, args.mixture_frame_cap)
        note = f" (capped {cap:,})" if cap is not None and frames[g] > cap else ""
        note += f" x{game_weight[g]:g}" if g in game_weight else ""
        return note + f" epochs={epochs[g]:.1f}" + ("(max)" if g in epoch_capped else "")

    log("[mixture] " + "  ".join(f"{g}: frames={frames[g]:,}{_mix_note(g)} w={weights[g]:.3f}"
                                 for g in games))
    for g in games:
        if g in whole:
            ml = crop(g, args.max_len, game_max_len)
            log(f"[data] {g}: whole records from their first frame"
                + (f" (capped at {ml})" if ml is not None else "")
                + f", batches of <= {whole[g].frames} frames: {whole[g].records_per_batch():.1f} "
                f"records per batch on average"
                + (f"; longer batches trained in chunks of {chunk_of(g)} frames"
                   if chunk_of(g) > 0 else ""))
        elif g in PREFIX_CROP:
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

    loaders = {g: _cycle(DataLoader(ds, batch_sampler=whole[g], num_workers=args.num_workers,
                                    collate_fn=collate_game_sequences) if g in whole
                         else DataLoader(ds, batch_size=game_batch.get(g, args.batch_size),
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
    if args.lr_schedule == "wsd":
        if args.lr_min_ratio >= 1.0:
            log("[warn] --lr_schedule wsd with --lr_min_ratio >= 1 never decays")
        lr_lambda = wsd_lr_lambda(args.warmup_steps, args.steps, args.lr_decay_frac,
                                  args.lr_min_ratio, args.lr_decay_shape)
    else:
        lr_lambda = trm_lr_lambda(args.warmup_steps, args.steps, args.lr_min_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        # matched by parameter name: the checkpoint's layout may differ
        # (games added, another --data order); the added games' own
        # parameters start without state, their EMA at their fresh weights
        ck_opt, ck_ema, fresh = resume_training_state(resume, model, optimizer, args.resume)
        if zero is not None:
            zero.load_optimizer_state(ck_opt)       # own partition (of this model) only
        else:
            optimizer.load_state_dict(ck_opt)
        scheduler.load_state_dict(resume["scheduler_state"])
        if ema is not None and ck_ema is not None:
            ema.load_state_dict(ck_ema)             # a partial EMA takes its own shadows
        start = resume["step"] + 1
        # fresh streams: the resumed run does not replay the first steps' draws
        random.seed(rank_seed(args.seed, rank, start))
        torch.manual_seed(rank_seed(args.seed, rank, start))
        log(f"[resume] {args.resume}: step {resume['step']} "
            f"(saved with world={resume.get('world_size', 1)})")
        if added:
            old_w = resume.get("mixture") or {}
            log(f"[resume] games added: {added} (checkpoint: {ck_games}); their "
                f"{len(fresh)} parameter tensors start fresh (no optimizer state"
                + (", EMA = their initial weights)" if ema is not None else ")"))
            log("[resume] mixture (checkpoint → this run): " + "  ".join(
                f"{g}: {old_w[g]:.3f}→{weights[g]:.3f}" if g in old_w
                else f"{g}: new→{weights[g]:.3f}" for g in games))
        elif games != ck_games:
            log(f"[resume] --data order differs from the checkpoint's {ck_games}: "
                f"matched by name")
        resume = ck_opt = ck_ema = None             # release the checkpoint tensors
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
            "reinit_log": reinit_log,
        }
        if resumable:
            ck.update(step=step, optimizer_state=optimizer_state,
                      scheduler_state=scheduler.state_dict(), ema_state=ema_state)
            # what a resume matches the optimizer state by (games may be added)
            ck[OPT_NAMES_KEY] = optimizer_param_names(model, optimizer)
        # atomic: a job killed mid-write never leaves a truncated checkpoint
        torch.save(ck, f"{path}.tmp")
        os.replace(f"{path}.tmp", path)
        return path

    model.train()
    t0 = time.time()
    window: Dict[str, List[float]] = {}
    # Σ over the log window of the steps' relative MoE loads [L, E]: the per-step
    # moe_load_cv mixes in which games the step drew (experts specialise); the
    # window sum shows whether the balance bias evens the load out over time
    load_window: Optional[torch.Tensor] = None
    stopped: Optional[int] = None
    for step in range(start, args.steps + 1):
        drawn = random.choices(names, weights=probs, k=K)
        for g in drawn:
            net = model[g]

            def forward_backward(merged, carry, active, reset, weight, g=g, net=net):
                mb = mb_from_merged(g, merged, args.il_legal_mask)
                scale = loss_weight.get(g, 1.0) * weight / k_div

                def backward(loss):
                    if torch.isfinite(loss):
                        # logs report the unweighted loss; λ_g scales the gradient
                        (loss * scale).backward()
                stats, pred = chunked_batch_loss(
                    net, mb, device, chunk_of(g), backward,
                    args.value_weight, Q_HALT_WEIGHT, args.bf16,
                    carry=carry, active=active, reset=reset,
                    z_loss_weight=args.logit_z_loss,
                    soft_target_weight=args.soft_target_weight,
                    engine_value_weight=args.engine_value_weight)
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
            if loads is not None:
                load_window = loads.clone() if load_window is None else load_window + loads
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
            if load_window is not None:
                rel = load_window / load_window.mean(-1, keepdim=True)
                extra += (f"  moe_win_cv={float(rel.std(-1).mean()):.3f}"
                          f"  moe_win_min={float(rel.min()):.3f}")
                load_window = None
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
        # the checkpoint first: a validation that fails (a batch out of memory) costs the
        # evaluation, not the steps since the last checkpoint
        # sharded: every rank takes part in the gather, rank 0 writes
        writes = rank0 or zero is not None
        saved_now = False
        if args.ckpt_every and (step % args.ckpt_every == 0 or step == args.steps) and writes:
            save("latest", step, resumable=True)
            saved_now = True
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
        if args.stop_file and step % 10 == 0:
            # rank 0 reads the file (one view of it for every rank), all ranks agree
            at = stop_step(args.stop_file) if rank0 else 0
            if dist_env.enabled:
                t = torch.tensor([at], dtype=torch.int64, device=device)
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                at = int(t.item())
            if at and step >= at:
                if writes and not saved_now:
                    save("latest", step, resumable=True)
                stopped = step
                if rank0:
                    late = f" (asked for step {at})" if at > 1 and step > at else ""
                    print(f"[stop] step {step}{late}: {os.path.join(args.save_dir, 'multi_latest.pt')} written, "
                          f"exiting for the handover ({args.stop_file})", flush=True)
                break
    if stopped is not None:
        dist_env.barrier()
        dist_env.destroy()
        return {"model": model, "ema": ema, "optimizer": optimizer, "dist": dist_env,
                "zero": zero, "stopped": stopped}
    final = save("final") if rank0 or zero is not None else None
    if rank0:
        print(f"[done] saved {final}")
    dist_env.barrier()
    dist_env.destroy()
    return {"model": model, "ema": ema, "optimizer": optimizer, "dist": dist_env,
            "zero": zero}


if __name__ == "__main__":
    main()
