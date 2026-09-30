"""Stage 2: joint mixed-game RL on the WHOLE MultiGameNetwork.

One model, many games: the Stage-1 checkpoint (training/train_multi_game.py,
its EMA weights) is both the initial policy and the FROZEN KL reference, and
every optimizer step mixes the RL data of every simulator game with an IL/KL
replay stream of every configured game (simulator-less games included) — there
is no per-game post-training.

    uv run python -m training.train_rl_joint --init checkpoints/multi/multi_final.pt \\
        --rl generals --rl othello \\
        --data generals=data/generals/shards --data othello=data/othello/train \\
        --data chess=data/chess/train --tokens_per_iter 65536 --iterations 2000

    # data parallel (NCCL; gloo on CPU and on MIG slices, --dist_backend)
    uv run torchrun --standalone --nproc_per_node=4 -m training.train_rl_joint ...

Iteration (synchronous; weights AND the MoE routing bias frozen from the first
rollout step to the last minibatch of the iteration):

  1. quotas      TokenQuotaScheduler: per-game learner-token quotas (static
                 --share, floor/cap, debt carrying; optional learning-progress
                 bandit via --bandit_mix).  Split over the ranks.
  2. rollouts    per game, lockstep waves of --envs VecGameEnv envs (reset
                 together, played until all end — the batch thins out toward
                 the longest episode) until the quota is met.  The
                 learner steps eval-mode, fixed depth, one batched
                 GameNetwork.step per env step (per-stream Stage-1 state;
                 --actor fixed: one fixed-shape batch, capturable with
                 --cuda_graph); it stores the behaviour log-prob μ of its
                 sampled action, the value, masks and rewards — and, for MoE
                 cores, every token's selected experts (--routing_replay).
                 League: mirror
                 self-play (--mirror_prob) or learner vs a snapshot-pool
                 opponent (PoolPolicy; one snapshot every --snapshot_every
                 iterations, the Stage-1 policy first).  Board games are ONE
                 interleaved stream per game with both movers; the loss goes
                 on learner seats (League.learner_policy_mask; forced frames
                 keep only the value loss).
  3. π_old       the learner recomputes log π_old and V on every stored frame
                 with the SAME weights and frozen routing (sparse MoE path,
                 exactly the training forward), before epoch 1.  With routing
                 replay every learner forward of the iteration (π_old and the
                 epochs) takes the actor's recorded experts, the gates from
                 the current router: a top-k near-tie cannot select different
                 experts in the actor's batched step and the learner's
                 sequence forward (π_old = μ up to float), and π stays smooth
                 in θ over the epochs (model.components.core, routing replay).
  4. targets     per-game γ/λ seat_gae (interleaved streams: every seat its own
                 chain) on V = popart.denormalize(v); PopArt statistics update
                 (heads + EMA shadow rescaled, outputs preserved) or HL-Gauss
                 passthrough; per-game running advantage normalisation (never
                 per mixed minibatch); optional top-q |A| filter per stream.
  5. epochs      --epochs × --minibatches optimizer steps.  Every minibatch
                 holds a stratified slice of EVERY active game's streams plus an
                 IL/KL replay batch per configured game (train_multi_game data
                 loading + train_il_game.batch_loss; --replay_weight split of
                 --replay_weight_total, the replay : RL ratio):
                   decoupled PPO   −min(1, π_old/μ)·min(rA, clip(r)A), r = π/π_old
                   value           PopArt-normalised MSE (or HL-Gauss CE)
                   entropy         per-game adaptive coefficient
                   KL              full-head KL to the frozen reference
                                   (every decoder head, il_head_terms), per-game
                                   adaptive beta toward --kl_budget
                 Games are combined as a weighted token mean with GLOBAL
                 per-game counts (combine_game_losses), so every rank's
                 averaged gradient is the global mean; each game's tokens
                 share the weight min(quota, realised tokens) (_loss_weights),
                 so the QUOTAS — not the overshoot of the last wave — set each
                 game's pull on the shared core.  (The full-head KL term is a
                 frame-weighted mean of per-chunk KLs, see _rl_chunk_loss.)
  6. between     MoE loads (tallied over every training forward of the
                 iteration — RL streams and replay batches, all epochs, all
                 ranks; the eval-mode forwards count through
                 model.components.core.moe_load_counting) → ONE balance-bias
                 update; EMA; entropy / KL controllers from the all-reduced
                 measurements; quota ledger; snapshot pool; logs; checkpoint.

Canaries (logged every iteration, per game): first-minibatch |log π_old/μ|
(actor/learner mismatch, ~0) and |log π/π_old| (0 up to float), KL to the
reference, entropy, value explained variance, expert-load entropy, learner
score vs the pool, the realised RL loss share next to the quota, the
first-minibatch shared-core gradient norm of each RL game and each replay
batch (the replay : RL balance), and the replay IL loss with its drift since
the first iteration (the forgetting canary of simulator-less games).

Multi-GPU: torchrun, training.distributed (manual rank-invariant gradient
sync, collective clipping; on MIG slices every rank pins itself to its own
slice and the ranks use gloo).  Every collective is posted identically on every
rank: per-epoch count all-reduce, per-step gradient sync + norm, per-iteration
statistics / diagnostics / MoE-load all-reduces.  Rollouts are rank-local.

Resume: ``rl_latest.pt`` (every --ckpt_minutes minutes, every --ckpt_every
iterations, and at the end) holds weights, EMA, optimizer, PopArt, advantage
statistics, entropy / KL controllers, the quota ledger, the snapshot pool,
every rank's simulator state (lux waves continue the envs' 5-match games) and
a fingerprint of the frozen reference (a resume against a different --init is
refused, and so is one that would switch the generals observation: an
``--env_kw generals:memory_vision=`` is adopted into the network's recorded
cfg.memory_vision, the setting a resume must keep).  Randomness is re-derived
from (seed, rank, iteration) at every iteration start (rollouts, league,
opponent draws, replay data order), so
``--resume rl_latest.pt`` reproduces the next iteration exactly on the same
world size: evicted pool files are deleted only after the next rl_latest.pt is
written (older rl_iter<N>.pt files may refer to deleted snapshots, which a
resume from them drops with a warning), and metrics.jsonl is cut back to the
resumed iteration.  The checkpoint is also a train_multi_game checkpoint:
``model_state`` (EMA), ``train_model_state`` and ``ema_state`` hold the
weights with the PopArt statistics folded into the value heads (reward
units), loadable by eval_multi_game / export_game_checkpoint / a next
``--init``; the normalised heads for the RL resume are in ``rl_state``.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from data_processing.games.dataset import (WholeRecordBatchSampler, collate_game_sequences,
                                           training_sampler)
from env.league import (
    League,
    NetworkPolicy,
    PoolPolicy,
    SnapshotPool,
    StreamStepper,
    restart_streams,
    scatter_rows,
)
from env.vec_game import VecGameEnv, VecObs, has_simulator, make_vec_env
from model.components.core import (
    MoEFFN,
    allreduce_moe_loads,
    moe_layers,
    moe_load_counting,
    update_moe_balance,
)
from model.components.heads import HLGaussValueHead
from model.config import checkpoint_config_dict
from model.games.base import categorical_entropy, masked_log_prob
from model.games.gmrules import resolve, rules_games
from model.multi_game import MultiGameNetwork, core_config_overrides, is_core_key
from model.optim import TRM_BETAS, AdamATan2, ModelEMA
from model.rl_utils import sync_gradients
from training.distributed import (
    DIST_BACKENDS,
    broadcast_module_state,
    collective_clip_grad_norm_,
    init_distributed,
    pin_rank_device,
)
from training.eval_multi_game import load_multi
from training.rl_joint import (
    PASSTHROUGH,
    AdaptiveEntropyCoef,
    AdaptiveKLBeta,
    AdvantageNormalizer,
    LearningProgressBandit,
    PopArt,
    TokenQuotaScheduler,
    combine_game_losses,
    decoupled_ppo_loss,
    entropy_loss,
    expert_load_entropy,
    head_terms_kl,
    hl_gauss_value_loss,
    kl_penalty,
    seat_gae,
    top_q_mask,
    value_loss,
)
from training.rl_joint.stats import collective_device, dist_active
from training.train_il_game import (
    LABEL_KEYS,
    UNCROPPABLE,
    batch_loss,
    default_tf,
    open_dataset,
    pick_device,
    prepare_batch,
    value_losses,
    value_target,
)
from training.train_multi_game import multi_game_param_groups

CKPT_LATEST = "rl_latest.pt"
CKPT_FINAL = "rl_final.pt"
SNAPSHOT_CORE = "core."     # the shared core's key prefix in pool snapshots
TERM_KINDS = ("kept", "policy", "value")   # token kinds of the loss terms' denominators


# ======================================================================
# CLI
# ======================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    # --- model / data ---
    a("--init", required=True,
      help="train_multi_game checkpoint: its EMA weights are the initial policy AND the "
           "frozen KL reference (pass it again with --resume)")
    a("--rl", action="append", default=[], metavar="GAME",
      help="game trained by RL (needs a simulator, env/vec_game.py); repeat per game")
    a("--data", action="append", default=[], metavar="GAME=DIR",
      help="IL replay shards of a game of the checkpoint (repeat; simulator-less games "
           "are kept from forgetting by this stream alone)")
    a("--iterations", type=int, default=1000)
    # --- quotas / rollouts ---
    a("--tokens_per_iter", type=int, default=16384,
      help="learner tokens per iteration over all games and ranks (quota budget)")
    a("--share", action="append", default=[], metavar="GAME=W",
      help="static learner-token share of an RL game (default equal)")
    a("--share_floor", type=float, default=0.0)
    a("--share_cap", type=float, default=1.0)
    a("--debt_decay", type=float, default=0.9)
    a("--bandit_mix", type=float, default=0.0,
      help="mix of the learning-progress bandit (signal: learner score vs the pool) into "
           "the static shares; 0 = static shares only")
    a("--envs", action="append", default=[], metavar="GAME=N",
      help="lockstep envs per wave per rank (default 8).  A wave runs until its LONGEST "
           "episode ends (no autoreset inside a wave), so the actor batch shrinks as "
           "episodes finish: size N for the tail of the episode-length distribution, "
           "not its mean")
    a("--env_kw", action="append", default=[], metavar="GAME:KEY=VALUE",
      help="adapter option, e.g. generals:max_turns=500 (Python literal values). "
           "generals:memory_vision=... sets the observation the network trains on: "
           "its cfg.memory_vision, recorded in the checkpoints (default: the --init "
           "network's own setting)")
    a("--rl_placement", choices=("spread", "split"), default="spread",
      help="multi-GPU rollouts: 'spread' = every rank plays every RL game (quota / world "
           "tokens each; a rank's rollouts take as long as one GPU's); 'split' = each game's "
           "whole quota is played by one rank, the games assigned by cost — the previous "
           "iteration's rollout + learner seconds, the quotas at first — so the ranks roll "
           "out different games in parallel (size --envs for a game's whole quota)")
    a("--max_wave_steps", type=int, default=100000,
      help="safety cap on env steps per lockstep wave")
    a("--actor", choices=("stream", "fixed"), default="stream",
      help="rollout stepping: 'stream' = StreamStepper (any temporal mixer); 'fixed' = "
           "all rows in one fixed-shape batch (recurrent temporal mixer; --cuda_graph)")
    a("--cuda_graph", action="store_true",
      help="with --actor fixed on CUDA: replay the step from a CUDA graph (GraphedStep)")
    a("--cache_ref_kl", type=int, default=1,
      help="compute the frozen reference's KL inputs once per iteration, with π_old, and "
           "reuse them in every epoch (1, default) instead of a reference forward per "
           "learner forward (0)")
    a("--routing_replay", type=int, default=1,
      help="MoE cores: the actor records every token's experts and every learner "
           "forward of the iteration replays them (1, default); 0 = the learner routes "
           "on its own (a top-k near-tie can then flip experts between actor and learner)")
    # --- league ---
    a("--mirror_prob", action="append", default=[], metavar="GAME=P",
      help="probability of a mirror self-play episode (default 1.0 = pure mirror); "
           "otherwise the learner plays a snapshot-pool opponent")
    a("--learner_seat", type=int, default=None,
      help="pin the learner's seat in non-mirror episodes (default uniform)")
    a("--snapshot_every", type=int, default=10, help="iterations between pool snapshots")
    a("--pool_size", type=int, default=16)
    a("--pool_max_nets", type=int, default=2,
      help="snapshot networks loaded per game and iteration (PoolPolicy active set)")
    a("--snapshot_weights", choices=("policy", "ema"), default="policy",
      help="weights a pool snapshot holds: the live policy (default) or its EMA")
    a("--snapshot_dir", default=None,
      help="file-backed snapshot pool directory (default <save_dir>/snapshots); 'memory' "
           "keeps snapshots in host RAM and inline in checkpoints")
    # --- advantage / value ---
    a("--gamma", action="append", default=[], metavar="GAME=G", help="discount (default 1.0)")
    a("--lam", action="append", default=[], metavar="GAME=L", help="GAE lambda (default 0.95)")
    a("--popart_decay", type=float, default=0.99)
    a("--no_popart", action="store_true", help="identity value normalisation")
    a("--adv_decay", type=float, default=0.99, help="per-game advantage moment EMA decay")
    a("--adv_filter_q", type=float, default=0.0,
      help="top-q |advantage| filter per stream (e.g. 0.25); 0 = off")
    # --- optimisation ---
    a("--epochs", type=int, default=2)
    a("--minibatches", type=int, default=4, help="optimizer steps per epoch")
    a("--rows_per_forward", action="append", default=[], metavar="GAME=N",
      help="streams per learner forward (gradient accumulation within a minibatch; "
           "default 8)")
    a("--frames_per_forward", action="append", default=[], metavar="GAME=N",
      help="frame budget of a learner forward instead of --rows_per_forward: as many "
           "streams (longest first) as keep rows x padded length <= N — short-episode "
           "games then fill a forward instead of running 8 short rows at a time")
    a("--frames_per_forward_default", type=int, default=0, metavar="N",
      help="--frames_per_forward of every RL game without its own (0 = off)")
    a("--clip", type=float, default=0.2)
    a("--clip_high", type=float, default=None)
    a("--is_clip", type=float, default=1.0, help="truncation of π_old/μ (0 = untruncated)")
    a("--dual_clip", type=float, default=None)
    a("--value_coef", type=float, default=0.5)
    a("--value_clip", type=float, default=None)
    a("--value_loss", choices=("mse", "huber"), default="mse")
    a("--game_weight", action="append", default=[], metavar="GAME=W",
      help="per-game multiplier of the RL loss terms (token-mean weights, default 1)")
    a("--entropy_coef", action="append", default=[], metavar="GAME=C",
      help="initial entropy coefficient (default 0.01)")
    a("--entropy_target", action="append", default=[], metavar="GAME=H",
      help="target policy entropy (nats); games without one keep a fixed coefficient")
    a("--timeout_reward", action="append", default=[], metavar="GAME=R",
      help="every seat's final reward when an episode hits the game's time limit (the env's "
           "truncation; Tetris: its turn-limit draw).  Default -1 for "
           + ", ".join(sorted(TIMEOUT_REWARD)) + ": running out the clock never beats "
           "losing, so a losing side gains nothing by stalling ('none' = the env's own score)")
    a("--shaping", action="append", default=[], metavar="GAME=S",
      help="initial weight of an env's shaped reward (its `shaping` attribute; default "
           + ", ".join(f"{g}={v}" for g, v in sorted(SHAPING.items())) + ")")
    a("--shaping_anneal", action="append", default=[], metavar="GAME=ITERS",
      help="iterations over which the shaping weight falls linearly to 0, so the game's "
           "outcome is all that is left (default " + ", ".join(
               f"{g}={v}" for g, v in sorted(SHAPING_ANNEAL.items())) + "; 0 = constant)")
    a("--alarm_timeout", type=float, default=0.05,
      help="reward-hacking alarm: a game's time-out rate above this ... (0 = off)")
    a("--alarm_len_x", type=float, default=2.0,
      help="... or its mean episode length above this multiple of its first iteration's ...")
    a("--alarm_kl_x", type=float, default=3.0,
      help="... or its KL to the IL policy above this multiple of its --kl_budget, for "
           "--alarm_patience iterations in a row, stops the game's RL (quota 0 for the rest "
           "of the run; logged, kept on resume)")
    a("--alarm_vs_init", type=float, default=0.35,
      help="... or (every --eval_every) its score against the frozen --init network below this")
    a("--alarm_patience", type=int, default=2)
    a("--alarm_min_episodes", type=int, default=8,
      help="iterations with fewer episodes of the game are not judged")
    a("--eval_every", type=int, default=20,
      help="every N iterations each 2+-seat RL game plays the frozen --init network (the "
           "learner against it, seats drawn per episode; not trained on): vs_init in "
           "metrics.jsonl, the referee self-play reward cannot fool (0 = off)")
    a("--eval_episodes", type=int, default=32, help="episodes per game and evaluation (whole waves)")
    a("--entropy_lr", type=float, default=0.05)
    a("--entropy_mode", choices=("log", "linear"), default="log")
    a("--entropy_min", type=float, default=1e-5)
    a("--entropy_max", type=float, default=0.1)
    a("--kl_beta", action="append", default=[], metavar="GAME=B",
      help="initial KL beta (default 0.05; 0 = no KL anchor for the game)")
    a("--kl_budget", action="append", default=[], metavar="GAME=K",
      help="per-game KL budget the beta controller steers toward (default 0.03)")
    a("--kl_rate", type=float, default=0.5)
    a("--kl_adapt", type=int, default=1, help="0 = fixed betas")
    a("--kl_min", type=float, default=1e-4)
    a("--kl_max", type=float, default=10.0)
    a("--lr", type=float, default=1e-5)
    a("--weight_decay", type=float, default=0.0)
    a("--beta1", type=float, default=TRM_BETAS[0])
    a("--beta2", type=float, default=TRM_BETAS[1])
    a("--clip_grad", type=float, default=1.0)
    a("--ema_rate", type=float, default=0.998, help="EMA decay per iteration (0 = off)")
    a("--moe_balance_rate", type=float, default=1e-3,
      help="balance-bias step, ONCE per iteration (loads of the whole iteration)")
    a("--act_steps", type=int, default=0, help="fixed core depth (0 = the model's default)")
    a("--bf16", action="store_true")
    # --- replay stream ---
    a("--replay_weight", action="append", default=[], metavar="GAME=W",
      help="relative weight of a game's IL replay loss (default --replay_weight_default); "
           "normalised to sum to --replay_weight_total")
    a("--replay_weight_default", type=float, default=1.0)
    a("--replay_weight_total", type=float, default=1.0,
      help="total weight of the replay stream, split over the --data games by their "
           "--replay_weight; the RL loss of all --rl games together is one unit (a "
           "token mean), so this is the replay : RL ratio whatever the number of games.  "
           "0 = no normalisation (every --replay_weight is absolute)")
    a("--replay_games_per_mb", type=int, default=0,
      help="replay games drawn per minibatch by their weights (0 = every game every "
           "minibatch)")
    a("--replay_split", type=int, default=0,
      help="multi-GPU: deal each minibatch's replay games (every game once) to the ranks, "
           "balanced by their measured cost, at world x their weight (the same objective "
           "as every rank replaying every game, at 1/world of the replay work); 0 = every "
           "rank replays every game (or its --replay_games_per_mb draws)")
    a("--replay_batch_size", type=int, default=4)
    a("--replay_batch", action="append", default=[], metavar="GAME=N")
    a("--replay_max_len", action="append", default=[], metavar="GAME=N",
      help="crop replay sequences (generals: its first N frames, as in train_multi_game)")
    a("--replay_frames", action="append", default=[], metavar="GAME=N",
      help="whole-record replay: the game's replay batches hold whole records from their "
           "first frame (its --replay_max_len a cap), as many similar-length records as "
           "keep records x the longest within N frames (0 = --replay_batch windows)")
    a("--replay_frames_default", type=int, default=0, metavar="N",
      help="--replay_frames of every replay game without its own (0 = off)")
    a("--replay_max_len_default", type=int, default=None, metavar="N",
      help="crop of every replay game without its own --replay_max_len (random windows as "
           "in pretraining, e.g. 64; games that are never cropped keep their full sequences; "
           "default: full sequences)")
    a("--replay_tf", action="append", default=[], metavar="GAME=P",
      help="teacher forcing of the replay batches (default train_il_game.default_tf)")
    a("--replay_value_weight", action="append", default=[], metavar="GAME=W",
      help="replay value-loss weight (PopArt-normalised targets; default 0.2 for "
           "simulator-less games, 0 for RL games whose critic RL trains)")
    a("--replay_kl_weight", action="append", default=[], metavar="GAME=W",
      help="full-head KL to the frozen reference on the replay batches (default 0)")
    a("--il_legal_mask", action="store_true")
    a("--soft_target_weight", type=float, default=1.0)
    a("--engine_value_weight", type=float, default=1.0)
    a("--logit_z_loss", type=float, default=1e-4)
    a("--num_workers", type=int, default=0)
    # --- run ---
    a("--save_dir", default="checkpoints/rl_joint")
    a("--ckpt_minutes", type=float, default=30.0, help="write rl_latest.pt this often")
    a("--ckpt_every", type=int, default=0, help="also every N iterations (0 = off)")
    a("--save_every", type=int, default=0, help="keep rl_iter<N>.pt every N iterations")
    a("--resume", default=None, help="rl_latest.pt to continue from (same arguments)")
    a("--log_every", type=int, default=1,
      help="print every N iterations (rank 0 appends every record to <save_dir>/metrics.jsonl)")
    a("--seed", type=int, default=0)
    a("--device", default=None)
    a("--bucket_mb", type=float, default=25.0,
      help="gradient all-reduce bucket size in MiB (0 = per tensor); 25 suits gloo / MIG "
           "slices too (scripts/bench_grad_sync.py)")
    a("--dist_timeout", type=float, default=30.0)
    a("--dist_backend", choices=DIST_BACKENDS, default="auto",
      help="process-group backend; auto = NCCL on full GPUs, gloo on MIG slices (every "
           "rank pinned to its own slice) and on the CPU")
    return p.parse_args(argv)


def _pairs(items: Sequence[str], cast: Callable = str, what: str = "VALUE") -> Dict[str, Any]:
    out = {}
    for it in items:
        g, sep, v = it.partition("=")
        if not sep or not g:
            raise SystemExit(f"expected GAME={what}, got {it!r}")
        out[g] = cast(v)
    return out


def _check_games(flag: str, values: Mapping[str, Any], allowed: Sequence[str],
                 what: str) -> None:
    unknown = sorted(set(values) - set(allowed))
    if unknown:
        raise SystemExit(f"{flag} for {unknown}: not {what} ({list(allowed)})")


def _env_kwargs(items: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for it in items:
        g, sep, kv = it.partition(":")
        k, sep2, v = kv.partition("=")
        if not (sep and sep2 and g and k):
            raise SystemExit(f"expected GAME:KEY=VALUE, got {it!r}")
        try:
            val = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            val = v
        out.setdefault(g, {})[k] = val
    return out


def derive_seed(*parts: int) -> int:
    """A 32-bit seed from non-negative integer parts (seed, rank, iteration,
    tags): independent streams for every combination."""
    return int(np.random.SeedSequence([int(x) for x in parts]).generate_state(1)[0])


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)


def module_fingerprint(module: torch.nn.Module) -> str:
    """SHA-1 of a module's state (names and exact bytes; tensors shared
    under several names — the multi-game core — hashed once)."""
    h = hashlib.sha1()
    seen = set()
    for k, v in module.state_dict().items():
        key = (v.data_ptr(), tuple(v.shape), v.dtype)
        if v.numel() and key in seen:
            continue
        seen.add(key)
        h.update(k.encode())
        t = v.detach().to("cpu").contiguous().reshape(-1)
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ======================================================================
# Per-game RL specs: how the trainer drives a simulator game's decoder
# ======================================================================

class RLGameSpec:
    """Generic spec (only for decoders without conditioning labels — see
    :func:`game_spec`): the decoder's ``sample()`` log-prob is μ, its
    ``log_prob_entropy`` the policy, no teacher-forced conditioning, and the
    KL anchor is the sampled k3 estimate on the taken actions (games with a
    full-head spec override :meth:`kl_targets` / :meth:`kl_terms`)."""

    full_kl = False

    def adopt_env_kw(self, net, overrides: Mapping[str, Any]) -> Optional[str]:
        """Apply the ``--env_kw`` ``overrides`` that are a property of the
        network's observation to its config (so the env built from
        :meth:`env_kwargs`, the IL replay batches and the checkpoints follow);
        returns a log line when that changes the config."""
        return None

    def env_kwargs(self, cfg) -> Dict[str, Any]:
        return {}

    def check_env(self, env: VecGameEnv, net) -> None:
        pass

    def behaviour_logp(self, out: Mapping[str, torch.Tensor],
                       legal: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return out["action_log_prob"].float()

    def labels(self, actions: Mapping[str, torch.Tensor], cfg) -> Optional[Dict[str, torch.Tensor]]:
        return None

    def logp_entropy(self, net, pred, actions, legal) -> Tuple[torch.Tensor, torch.Tensor]:
        lp, ent, _ = net.action_decoder.log_prob_entropy(pred, actions, legal)
        return lp.float(), ent.float()

    def kl_targets(self, actions, forced, cfg) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def kl_terms(self, net, pred, tgt, mask, legal) -> Dict[str, Dict]:
        return net.action_decoder.il_head_terms(pred, tgt, mask)

    def rollout_legal(self, legal: Dict[str, torch.Tensor],
                      out: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """The ``legal`` dict stored with the learner's frames for the replay: the
        env's (default), or a compact equivalent derived from the actor's outputs
        ``out`` (its sample() keys) that replays the rollout exactly."""
        return legal


class ForcedPassPlacementSpec(RLGameSpec):
    """Placement games whose passes are rules-forced (othello): the env
    applies the pass itself on ``forced`` frames and ignores the decoder's
    pass Bernoulli elsewhere, so the RL action is the placement alone — a
    placement-only masked log-prob / entropy (train_rl_game.py's rule), the
    pass head anchored by the KL only.  Forced frames carry no policy loss."""

    full_kl = True

    def behaviour_logp(self, out, legal):
        return masked_log_prob(out["place_logits"].float(), out["action_place"],
                               legal["place"].bool())

    def logp_entropy(self, net, pred, actions, legal):
        logits = pred["place_logits"].float()
        mask = legal["place"].bool()
        lp = masked_log_prob(logits, actions["place_idx"].long(), mask)
        safe = mask | ~mask.any(-1, keepdim=True)     # forced rows: uniform, no NaN
        ent = categorical_entropy(logits.masked_fill(~safe, float("-inf")))
        return lp, ent

    def kl_targets(self, actions, forced, cfg):
        return {"place_idx": actions["place_idx"].long(), "passed": forced.bool()}

    def kl_terms(self, net, pred, tgt, mask, legal):
        terms = dict(net.action_decoder.il_head_terms(pred, tgt, mask))
        terms["place"] = {**terms["place"], "mask": legal["place"].bool()}
        return terms


class GomokuSpec(ForcedPassPlacementSpec):
    """gomoku: pure placement, the othello spec without forced frames (the
    env never passes, ``forced`` is all False).  LABEL_KEYS["gomoku"] is
    empty and the registry builds the decoder without a pass head
    (``allow_pass=False``), so the generic spec would already be exact; this
    one adds (1) the env built on the network's board (``board_h`` /
    ``board_w``: a 15x15 net on the env's 20x20 default would fail) and
    checked against it, (2) the full-head KL (exact over every legal point,
    not the sampled k3 estimate of the taken one), and (3) a placement-only
    μ / π when a checkpoint does carry a pass head: the env applies the
    placement whatever pass flag is sampled, so the pass Bernoulli is not
    part of the action (the full-head KL keeps it near the reference's pass
    head, which IL trained on all-False targets)."""

    def env_kwargs(self, cfg):
        return {"board_size": (int(cfg.board_h), int(cfg.board_w))}

    def check_env(self, env, net):
        have = tuple(env.obs_schema()["board_seq"][0])
        cfg = net.cfg
        want = (cfg.piece_channels, cfg.board_h, cfg.board_w)
        if have != want:
            raise SystemExit(f"gomoku env emits board_seq {have}, the network expects {want} "
                             f"(piece_channels, board_h, board_w)")
        if getattr(cfg, "board_global_features", 0):
            raise SystemExit(f"gomoku env emits no globals_seq, the network expects "
                             f"{cfg.board_global_features} board_global_features")


class GoSpec(RLGameSpec):
    """go (model/games/board.py PlacementDecoder with its pass head) on the gmrules go
    env (legal ``place`` [N·N], the pass always legal; action keys place_idx / passed):
    μ is the decoder's pass-gated log-prob (a pass frame by the pass head alone, a move
    by pass + placement), π its ``log_prob_entropy`` of the same action — no
    conditioning labels (LABEL_KEYS["go"] is empty), so the replay is exact without
    teacher forcing.  Unlike othello's forced passes a Go pass is a real choice: it is
    part of the action.  The KL covers both heads (``il_head_terms``; the placement
    under the env's mask).  The env is built on the network's board (``board_size`` =
    board_h) and checked against it (board_seq [piece_channels, N, N], globals)."""

    full_kl = True

    def env_kwargs(self, cfg):
        return {"board_size": int(cfg.board_h)}

    def check_env(self, env, net):
        cfg = net.cfg
        schema = env.obs_schema()
        have = tuple(schema["board_seq"][0])
        want = (cfg.piece_channels, cfg.board_h, cfg.board_w)
        g = schema["globals_seq"][0][0] if "globals_seq" in schema else 0
        if have != want or g != getattr(cfg, "board_global_features", 0):
            raise SystemExit(f"go env emits board_seq {have} / {g} globals, the network expects "
                             f"{want} / {getattr(cfg, 'board_global_features', 0)} "
                             f"(piece_channels, board_h, board_w / board_global_features)")

    def kl_targets(self, actions, forced, cfg):
        return {"place_idx": actions["place_idx"].long(), "passed": actions["passed"].bool()}

    def kl_terms(self, net, pred, tgt, mask, legal):
        terms = dict(net.action_decoder.il_head_terms(pred, tgt, mask))
        if "place" in terms and "place" in legal:
            terms["place"] = {**terms["place"], "mask": legal["place"].bool()}
        return terms


class Arc3Spec(RLGameSpec):
    """ARC-AGI-3 (model/games/arc3.py Arc3Decoder on env/vec_arc3.py): the rollout
    samples type → (ACTION6) patch → cell under the env's tables; the replay
    teacher-forces the stored click (the cell head conditions on its patch) under the
    masks the rollout sampled with (``arc3_replay_legal``: type [8], patch [P²], the
    stored patch's cells [p²]), so log π_old = μ; full-head KL (type, patch, cell).
    The env is checked against the network (grid size, globals width)."""

    full_kl = True

    def check_env(self, env, net):
        cfg = net.cfg
        schema = env.obs_schema()
        have = (tuple(schema["grid"][0]), tuple(schema["globals"][0]))
        want = ((cfg.arc_grid, cfg.arc_grid), (cfg.arc_global_features,))
        if have != want:
            raise SystemExit(f"arc3 env emits grid / globals {have}, the network expects {want}")

    def rollout_legal(self, legal, out):
        from model.games.arc3 import arc3_replay_legal

        stored = arc3_replay_legal(out)
        return stored if {"type", "patch", "sub_of"} <= set(stored) else legal

    def labels(self, actions, cfg):
        return {"click_idx": actions["click_idx"].long()}

    def kl_targets(self, actions, forced, cfg):
        return {"type_idx": actions["type_idx"].long(), "click_idx": actions["click_idx"].long()}


class HokSpec(RLGameSpec):
    """Honor of Kings 1v1 (model/games/hok.py HokDecoder on env/vec_hok.py): the rollout
    samples the button, then its argument heads under the env's tables; the replay
    teacher-forces the stored button under the masks the rollout sampled with
    (``hok_replay_legal``: the button / direction masks, the button's target and sub
    rows), so log π_old = μ; full-head KL (the button and the heads it uses).  The env
    is checked against the network (feature width)."""

    full_kl = True
    ACTIONS = ("button_idx", "move_x_idx", "move_z_idx", "skill_x_idx", "skill_z_idx",
               "target_idx")

    def check_env(self, env, net):
        have = tuple(env.obs_schema()["features"][0])
        want = (getattr(net.cfg, "hok_features", 725),)
        if have != want:
            raise SystemExit(f"hok env emits {have} features, the network expects {want}")

    def rollout_legal(self, legal, out):
        from model.games.hok import hok_replay_legal

        stored = hok_replay_legal(out)
        return stored if {"button", "target_of", "sub_of"} <= set(stored) else legal

    def labels(self, actions, cfg):
        return {"button_idx": actions["button_idx"].long()}

    def kl_targets(self, actions, forced, cfg):
        return {k: actions[k].long() for k in self.ACTIONS}


class GeneralsSpec(RLGameSpec):
    """generals: the decoder's hierarchical log-prob (μ = ``action_log_prob``),
    the replay teacher-forces the stored source (direction / split condition
    on it exactly as in the rollout), and the KL covers every head
    (ImitationLoss.head_terms weights; the source restricted to legal cells,
    the direction logits carry their own hard mask)."""

    full_kl = True

    def adopt_env_kw(self, net, overrides):
        # --env_kw generals:memory_vision= is the run's observation, i.e. the
        # network's cfg.memory_vision (as training/train_rl.py --memory_vision):
        # a run launched with it before the field existed recorded it only in
        # its args (model.config.legacy_memory_vision) and resumes with them.
        # Parsed as GeneralsVecEnv reads it (bool of the literal).
        if "memory_vision" not in overrides:
            return None
        want = bool(overrides["memory_vision"])
        have = bool(getattr(net.cfg, "memory_vision", False))
        if want == have:
            return None
        net.cfg.memory_vision = want
        return (f"generals: --env_kw memory_vision={want} switches the observation "
                f"(cfg.memory_vision {have} -> {want}; not what the --init network "
                f"trained on — recorded in this run's checkpoints, the replay batches "
                f"follow)")

    def env_kwargs(self, cfg):
        return {"grid_h": cfg.grid_h, "grid_w": cfg.grid_w,
                "turn_globals": bool(getattr(cfg, "turn_globals", False)),
                "memory_vision": bool(getattr(cfg, "memory_vision", False))}

    def check_env(self, env, net):
        schema = env.obs_schema()
        cfg = net.cfg
        g = schema["globals_seq"][0][0]
        c = schema["grid_seq"][0][0]
        if g != cfg.global_features or c != cfg.grid_channels:
            raise SystemExit(f"generals env emits {c} grid channels / {g} globals, the network "
                             f"expects {cfg.grid_channels} / {cfg.global_features} "
                             f"(turn_globals={getattr(cfg, 'turn_globals', False)})")
        # same shapes either way, so the schema check cannot see it: the
        # sighted-mountain memory must be the network's (adopt_env_kw has
        # already applied an --env_kw generals:memory_vision to its config)
        mv = bool(getattr(cfg, "memory_vision", False))
        if bool(getattr(env, "memory_vision", False)) != mv:
            raise SystemExit(f"generals env memory_vision={getattr(env, 'memory_vision', None)}, "
                             f"but the network's cfg.memory_vision (the observation it "
                             f"trains on) is {mv}")

    def labels(self, actions, cfg):
        src = actions["source_idx"].long()
        return {"source_y": src // cfg.grid_w, "source_x": src % cfg.grid_w}

    def kl_targets(self, actions, forced, cfg):
        src = actions["source_idx"].long()
        return {"count": (~forced.bool()).long(),
                "src_y": (src // cfg.grid_w).unsqueeze(-1),
                "src_x": (src % cfg.grid_w).unsqueeze(-1),
                "dir_id": actions["direction"].long().unsqueeze(-1),
                "ratio_id": actions["split"].long().unsqueeze(-1)}

    def kl_terms(self, net, pred, tgt, mask, legal):
        terms = dict(net.action_decoder.il_head_terms(pred, tgt, mask))
        src = legal["source"].bool()
        terms["source"] = {**terms["source"], "mask": src.reshape(*src.shape[:2], -1)}
        return terms


class LuxSpec(RLGameSpec):
    """lux (env/vec_lux.py): the factored per-unit decoder conditions on
    nothing (``LABEL_KEYS["lux"] == ()``), so μ is ``sample()``'s
    ``action_log_prob`` and the policy its ``log_prob_entropy`` — the sum over
    the ``legal["unit"]`` units of the action log-prob plus, for sapping
    units, the sap target's — as in the generic spec.  The KL covers every
    head: each unit's action distribution (legal-masked) and the sap target
    distribution of the sapping units.  The env's sap window is the
    decoder's ``lux_sap_range``."""

    full_kl = True

    def env_kwargs(self, cfg):
        return {"sap_window": int(getattr(cfg, "lux_sap_range", 0))}

    def check_env(self, env, net):
        cfg = net.cfg
        got = {k: tuple(v[0]) for k, v in env.obs_schema().items()}
        want = {"map_seq": (cfg.map_channels, cfg.map_h, cfg.map_w),
                "units_seq": (cfg.max_units, cfg.unit_feature_dim),
                "unit_mask": (cfg.max_units,), "globals_seq": (cfg.lux_global_features,)}
        if got != want:
            raise SystemExit(f"lux env emits {got}, the network expects {want} (the env "
                             f"speaks the converter-v2 schema: lux_global_features=11)")
        R = int(getattr(cfg, "lux_sap_range", 0))
        if getattr(env, "sap_window", R) != R or cfg.n_unit_actions != 6:
            raise SystemExit(f"lux env sap window {getattr(env, 'sap_window', None)} / 6 unit "
                             f"actions vs the network's lux_sap_range={R} / "
                             f"n_unit_actions={cfg.n_unit_actions}")

    def kl_targets(self, actions, forced, cfg):
        ua = actions["unit_actions"].long()
        tgt = {"unit_actions": ua}
        if "sap_offsets" in actions:
            sap = ua == int(getattr(cfg, "lux_sap_action", 5))
            tgt["sap_offset"] = torch.where(sap, actions["sap_offsets"].long(),
                                            torch.full_like(ua, -1))
        return tgt

    def kl_terms(self, net, pred, tgt, mask, legal):
        terms = dict(net.action_decoder.il_head_terms(pred, tgt, mask))
        terms["unit_action"] = {**terms["unit_action"], "mask": legal["unit_action"].bool()}
        return terms


class CandidateSpec(RLGameSpec):
    """Candidate-action games (gmrules: env/vec_gmsim.py +
    model/games/candidate.py) — the GENERIC spec of every such game.  The action
    is one categorical over the frame's legal candidates: μ = ``sample()``'s
    ``action_log_prob`` and the policy ``log_prob_entropy``, both masked by the
    frame's candidate mask (the decoder's decode context) ∧ ``legal["action"]``;
    nothing to teacher-force; the KL is exact over every legal candidate (the
    decoder's ``action`` head).  Forced frames (one candidate) carry no policy
    loss.  The env is configured by ``--env_kw <game>:key=value`` (the gmrules
    game config, e.g. ``kuhn_poker:players=3``) and checked against the network's
    frame dimensions.  Per-seat games are not lockstep: use ``--actor stream``."""

    full_kl = True

    def check_env(self, env, net):
        from model.games.gmrules import CANDIDATE_META_ARCH

        spec = getattr(env, "spec", None)
        # the candidate schema, or a custom one with its layout (e.g. an abstracted
        # action space: exact bets in the rules, discretised sizes as candidates)
        need = {"entities", "entity_mask", "globals", "actions", "action_mask"}
        if (spec is None or not need <= {t["name"] for t in spec["obs"]}
                or spec["action_keys"] != ["action_idx"]
                or not set(CANDIDATE_META_ARCH) <= set(spec["meta"])):
            raise SystemExit(f"{env.game}: not a candidate-action gmsim env (env/vec_gmsim.py)")
        meta = spec["meta"]
        bad = {k: (meta[k], getattr(net.cfg, f[0], None)) for k, f in CANDIDATE_META_ARCH.items()
               if meta[k] != getattr(net.cfg, f[0], None)}
        if bad:
            raise SystemExit(f"{env.game} env {env.config} emits frames {{dim: (env, network)}} "
                             f"{bad}: the network was trained on another variant")

    def kl_targets(self, actions, forced, cfg):
        return {"action_idx": actions["action_idx"].long()}


class FromToSpec(RLGameSpec):
    """From-to decoder games (model/games/board.py FromToDecoder on a gmrules custom
    schema: env/vec_gmsim.py legal ``from`` / ``to_cond`` (/ ``promotion_cond``),
    action keys ``from_idx`` / ``to_idx`` (/ ``promotion``) — breakthrough, chess
    variants, xiangqi, ...) — the GENERIC spec of such games.  The rollout commits
    from → to (→ promotion) under the env's exact tables; the replay teacher-forces
    the stored from / to (``labels``), so the to / promotion heads condition exactly
    as in the rollout, under the masks the rollout sampled with (FromToDecoder.sample's
    ``legal_from`` / ``legal_to`` / ``legal_promotion``, stored as ``from`` / ``to_of``
    / ``promotion_of`` — [N] per frame instead of the env's [N, N] tables): log π_old
    = μ.  The KL covers every head (``il_head_terms``: from, to | stored from,
    promotion | stored from / to) under those masks.  The env is checked against the
    network's board (``board_seq`` [piece_channels, board_h, board_w], globals)."""

    full_kl = True

    def check_env(self, env, net):
        cfg = net.cfg
        schema = env.obs_schema()
        if "board_seq" not in schema:
            raise SystemExit(f"{env.game}: FromToSpec needs a board_seq observation")
        have = tuple(schema["board_seq"][0])
        want = (cfg.piece_channels, cfg.board_h, cfg.board_w)
        g = schema["globals_seq"][0][0] if "globals_seq" in schema else 0
        if have != want or g != getattr(cfg, "board_global_features", 0):
            raise SystemExit(f"{env.game} env emits board_seq {have} / {g} globals, the network "
                             f"expects {want} / {getattr(cfg, 'board_global_features', 0)} "
                             f"(piece_channels, board_h, board_w / board_global_features)")

    def rollout_legal(self, legal, out):
        from model.games.board import fromto_replay_legal

        stored = fromto_replay_legal(out)
        return stored if "from" in stored and "to_of" in stored else legal

    def labels(self, actions, cfg):
        return {"from_idx": actions["from_idx"].long(), "to_idx": actions["to_idx"].long()}

    def kl_targets(self, actions, forced, cfg):
        tgt = {"from_idx": actions["from_idx"].long(), "to_idx": actions["to_idx"].long()}
        if "promotion" in actions:
            tgt["promotion"] = actions["promotion"].long()
        return tgt


class MahjongSpec(RLGameSpec):
    """Riichi mahjong (model/games/mahjong.py MahjongDecoder on env/vec_mahjong.py): the
    rollout samples type → tile under the env's tables (legal ``action_type`` [8],
    ``tile_cond`` [8, 34]); the replay teacher-forces the stored type (the tile head
    conditions on it) under the masks the rollout sampled with (the decoder's
    ``legal_action_type`` / ``legal_tile``, stored as ``action_type`` / ``tile_of``), so
    log π_old = μ.  The KL covers both heads (``il_head_terms``: type, tile | stored type).
    The env is checked against the network's tile planes."""

    full_kl = True

    def check_env(self, env, net):
        have = tuple(env.obs_schema()["tiles_seq"][0])
        want = (34, net.cfg.mahjong_tile_feature_dim)
        if have != want:
            raise SystemExit(f"mahjong env emits tiles_seq {have}, the network expects {want} "
                             f"(mahjong_tile_feature_dim: converter v4 shards have 37)")

    def rollout_legal(self, legal, out):
        if "legal_action_type" in out and "legal_tile" in out:
            return {"action_type": out["legal_action_type"], "tile_of": out["legal_tile"]}
        return legal

    def labels(self, actions, cfg):
        return {"action_type": actions["action_type"].long()}

    def kl_targets(self, actions, forced, cfg):
        return {"action_type": actions["action_type"].long(), "tile_idx": actions["tile_idx"].long()}


class PokemonSpec(RLGameSpec):
    """Pokémon Showdown (model/games/pokemon.py PokemonDecoder on env/vec_pokemon.py): the
    rollout samples type → (move + tera | switch) under the request's legality (usable moves,
    switchable slots, tera, the active slot and revealed moves as state); the replay
    teacher-forces the stored type (the move / switch heads condition on it) under the same
    stored legal dict, so log π_old = μ; the KL covers every head (``il_head_terms``: type,
    move | move type, switch | switch type, tera | move type)."""

    full_kl = True

    def labels(self, actions, cfg):
        return {"action_type": actions["action_type"].long()}

    def kl_targets(self, actions, forced, cfg):
        return {"action_type": actions["action_type"].long(), "move_id": actions["move_id"].long(),
                "switch_idx": actions["switch_idx"].long(), "tera": actions["tera"].long()}


class Sc2Spec(RLGameSpec):
    """StarCraft II (model/games/sc2.py Sc2Decoder on env/vec_sc2.py): the rollout commits
    type → actors → the coarse target cell in decode and samples the rest; the replay
    teacher-forces the stored type, actors and target point (the argument and target heads
    condition on them), the masks come from the obs (own units, occupied slots) and the
    vocabulary's target kinds, so log π_old = μ; full-head KL (``il_head_terms``: type, kind,
    actors, queued, unit, coarse, fine)."""

    full_kl = True

    def check_env(self, env, net):
        have = env.obs_schema()
        cfg = net.cfg
        want = {"units": (cfg.sc2_units, cfg.sc2_unit_features), "upgrades": (cfg.sc2_upgrades,)}
        for k, shape in want.items():
            if tuple(have[k][0]) != shape:
                raise SystemExit(f"sc2 env emits {k} {tuple(have[k][0])}, the network expects {shape}")

    def labels(self, actions, cfg):
        return {"action_type": actions["action_type"].long(), "actors": actions["actors"].float(),
                "target_point": actions["target_point"].long()}

    def kl_targets(self, actions, forced, cfg):
        return {"action_type": actions["action_type"].long(), "actors": actions["actors"].float(),
                "target_kind": actions["target_kind"].long(), "target_unit": actions["target_unit"].long(),
                "target_point": actions["target_point"].long(), "queued": actions["queued"].bool()}


class ClashRoyaleSpec(RLGameSpec):
    """Clash Royale (model/games/clash_royale.py CardPlacementDecoder on env/vec_cr.py): the
    rollout commits the card (and the placement patch) in decode and samples wait / cell;
    the replay teacher-forces the stored card and cell under the stored legal dict (the
    env's ``card`` / ``placement``), so log π_old = μ; full-head KL (``il_head_terms``)."""

    full_kl = True

    def check_env(self, env, net):
        have = env.obs_schema()
        cfg = net.cfg
        if tuple(have["deck_ids"][0]) != (getattr(cfg, "deck_size", 8),):
            raise SystemExit(f"clash_royale env emits {tuple(have['deck_ids'][0])} deck tokens, the network "
                             f"expects {getattr(cfg, 'deck_size', 8)}")

    def labels(self, actions, cfg):
        return {"card_slot": actions["card_slot"].long(), "place_idx": actions["place_idx"].long()}

    def kl_targets(self, actions, forced, cfg):
        return {"waited": actions["waited"].bool(), "card_slot": actions["card_slot"].long(),
                "place_idx": actions["place_idx"].long()}


class ShogiSpec(RLGameSpec):
    """The shogi move-or-drop decoder (model/games/shogi.py ShogiDecoder) on the gmrules
    shogi env (legal ``mode`` / ``from`` / ``to_cond`` / ``promo_cond`` / ``drop_piece`` /
    ``drop_to_cond``, action keys dropped / from_idx / to_idx / promoted / piece_idx /
    drop_to_idx).  The rollout commits mode, from → to → promotion and piece → drop
    square under the env's exact tables; the replay teacher-forces the stored dropped /
    from_idx / to_idx / piece_idx (the decoder's LABEL_KEYS) under the masks the rollout
    sampled with (``shogi_replay_legal``: every ``_of`` row at the stored conditioning),
    so the leaves condition exactly as in the rollout: log π_old = μ.  Both branches
    are decoded on every frame, the mode gates which one's log-prob counts.  The KL
    covers every head (``il_head_terms``: mode, from, to | from, promotion | from / to,
    piece, drop square | piece) under those masks."""

    full_kl = True
    LABELS = ("dropped", "from_idx", "to_idx", "piece_idx")

    def check_env(self, env, net):
        cfg = net.cfg
        schema = env.obs_schema()
        want = {"board_seq": (cfg.piece_channels, cfg.board_h, cfg.board_w),
                "hand_seq": (2 * cfg.n_hand_piece_types, cfg.hand_feature_dim),
                "globals_seq": (cfg.shogi_global_features,)}
        have = {k: tuple(schema[k][0]) if k in schema else None for k in want}
        if have != want:
            raise SystemExit(f"{env.game} env emits {have}, the network expects {want} "
                             f"(piece_channels / board dims, hand slots x hand_feature_dim, "
                             f"shogi_global_features)")

    def rollout_legal(self, legal, out):
        from model.games.shogi import shogi_replay_legal

        stored = shogi_replay_legal(out)
        return stored if {"mode", "from", "to_of", "promo_of", "drop_piece",
                          "drop_to_of"} <= set(stored) else legal

    def labels(self, actions, cfg):
        return {k: actions[k].long() for k in self.LABELS}

    def kl_targets(self, actions, forced, cfg):
        tgt = {k: actions[k].long() for k in self.LABELS + ("drop_to_idx",)}
        tgt["promoted"] = actions["promoted"].bool()
        return tgt


GAME_SPECS: Dict[str, RLGameSpec] = {
    "generals": GeneralsSpec(),
    "othello": ForcedPassPlacementSpec(),
    "gomoku": GomokuSpec(),
    "lux": LuxSpec(),
    "arc3": Arc3Spec(),
    "hok": HokSpec(),
    "mahjong": MahjongSpec(),
    "pokemon": PokemonSpec(),
    "sc2": Sc2Spec(),
    "clash_royale": ClashRoyaleSpec(),
}


def _spec_factory(ref) -> Callable[[], RLGameSpec]:
    """A RulesGame ``rl_spec`` reference; ``"training.train_rl_joint:X"`` resolves
    here (no second import of this module when it runs as ``__main__``)."""
    if isinstance(ref, str) and ref.startswith("training.train_rl_joint:"):
        return globals()[ref.split(":", 1)[1]]
    return resolve(ref)


# every gmrules game (model/games/gmrules/): its own spec, else the generic
# candidate spec (custom schemas without one fall back to game_spec's rules)
for _game, _entry in rules_games().items():
    if _entry.rl_spec is not None:
        GAME_SPECS.setdefault(_game, _spec_factory(_entry.rl_spec)())
    elif _entry.is_candidate:
        GAME_SPECS.setdefault(_game, CandidateSpec())


def game_spec(game: str) -> RLGameSpec:
    """The game's RL spec: a dedicated one from :data:`GAME_SPECS`, else the
    generic :class:`RLGameSpec` — refused for autoregressive decoders (a
    non-empty ``train_il_game.LABEL_KEYS`` entry: chess from→to, shogi, ...):
    the generic spec passes no teacher-forcing labels, so the learner's
    ``decode`` would condition the later heads on its OWN sampled choices
    and ``log_prob_entropy`` would score the stored action under the wrong
    conditioning (log π_old ≠ μ, a random ratio, a wrong KL)."""
    spec = GAME_SPECS.get(game)
    if spec is not None:
        return spec
    if LABEL_KEYS.get(game):
        raise SystemExit(f"--rl {game}: no RL spec in GAME_SPECS, and its decoder conditions on "
                         f"{list(LABEL_KEYS[game])} — the generic spec cannot teacher-force the "
                         f"stored actions; add a dedicated spec (labels / kl_targets)")
    return RLGameSpec()


# ======================================================================
# Rollouts
# ======================================================================

def _autocast(device: torch.device, bf16: bool):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)


class LearnerActor:
    """The learner network of one game as its rollout policy: eval mode,
    fixed depth, no grad; per call it returns (on the CPU) the sampled
    actions (decoder ``sample()`` keys), ``mu`` = the spec's behaviour
    log-prob of them and the ``value`` (head units).

    ``mode="stream"``: :class:`env.league.StreamStepper` — per-stream Stage-1
    state, one batched call per distinct frame index (one per env step in
    lockstep), any temporal mixer.  ``mode="fixed"``: every row of the env in
    ONE fixed-shape batch from :meth:`GameNetwork.initial_st_cache` with a
    tensor step index (recurrent temporal mixer; rows the learner does not
    play are computed and discarded), optionally replayed from a CUDA graph
    (``cuda_graph``, :class:`model.cuda_graph.GraphedStep`).  Must be called
    once per env step of a wave in fixed mode (also with no rows).
    ``record_routes`` (MoE cores): the output also holds ``moe_routes``
    [rows, C, S, k], the experts the step routed every token to."""

    def __init__(self, net, spec: RLGameSpec, act_steps: int, device: torch.device, *,
                 bf16: bool = False, mode: str = "stream", cuda_graph: bool = False,
                 record_routes: bool = False):
        self.net, self.spec, self.act_steps, self.device = net, spec, act_steps, device
        self.bf16, self.mode = bf16, mode
        self.cuda_graph = bool(cuda_graph)
        #: also return ``moe_routes`` (routing replay; MoE cores only)
        self.record_routes = bool(record_routes) and net.moe_route_calls(act_steps) > 0
        if mode not in ("stream", "fixed"):
            raise ValueError(f"actor mode must be 'stream' or 'fixed', got {mode!r}")
        if mode == "fixed" and net.cfg.temporal_mixer != "recurrent":
            raise SystemExit("--actor fixed needs temporal_mixer='recurrent' (the attention "
                             "mixer's KV cache grows every frame)")
        if self.cuda_graph and (mode != "fixed" or device.type != "cuda"):
            raise SystemExit("--cuda_graph needs --actor fixed on a CUDA device")
        self._graph = None
        self.begin_wave()

    def begin_wave(self) -> None:
        self.stepper = StreamStepper(self.net, act_steps=self.act_steps, device=self.device,
                                     record_routes=self.record_routes)
        self._cache: Optional[List[torch.Tensor]] = None
        self._t = 0

    def _fixed_step(self, obs: VecObs) -> Dict[str, torch.Tensor]:
        o = {k: v.to(self.device) for k, v in obs.obs.items()}
        lg = {k: v.to(self.device) for k, v in obs.legal.items()}
        if self.cuda_graph:
            if self._graph is None or self._graph.batch != obs.n_rows:
                from model.cuda_graph import GraphedStep
                # the capture's eager warm-up samples actions: keep it off the RNG
                # streams, so the rollout draws do not depend on the iteration in
                # which the graph was built (a resumed run builds it later)
                cpu_rng = torch.get_rng_state()
                cuda_rng = torch.cuda.get_rng_state(self.device)
                self._graph = GraphedStep(self.net, o, lg, act_steps=self.act_steps,
                                          autocast_dtype=torch.bfloat16 if self.bf16 else None,
                                          record_routes=self.record_routes)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, self.device)
            if self._t == 0:
                self._graph.reset()
            out = dict(self._graph(o, self._t, lg))
        else:
            if self._t == 0:
                self._cache = self.net.initial_st_cache(obs.n_rows)
            step_idx = torch.tensor([self._t], dtype=torch.long, device=self.device)
            with _autocast(self.device, self.bf16):
                out = self.net.step(o, legal=lg, st_cache=self._cache, step_idx=step_idx,
                                    act_steps=self.act_steps, record_routes=self.record_routes)
            self._cache = out.pop("st_cache")
        self._t += 1
        return out

    @torch.no_grad()
    def __call__(self, obs: VecObs, rows: torch.Tensor) -> Dict[str, torch.Tensor]:
        rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
        was_training = self.net.training
        self.net.eval()
        try:
            if self.mode == "fixed":
                if rows.numel() and not bool((obs.step_idx[rows] == self._t).all()):
                    raise RuntimeError("fixed actor: rows are not in lockstep "
                                       f"(frame {self._t} vs {obs.step_idx[rows].tolist()[:4]})")
                full = self._fixed_step(obs)
                idx = rows.to(self.device)
                out = {k: v.index_select(0, idx) for k, v in full.items()
                       if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == obs.n_rows}
            else:
                if rows.numel() == 0:
                    return {}
                streams = obs.stream_id[rows].tolist()
                restart = restart_streams(obs, rows)
                if restart:
                    self.stepper.reset(restart)
                with _autocast(self.device, self.bf16):
                    out = self.stepper.step({k: v[rows] for k, v in obs.obs.items()},
                                            {k: v[rows] for k, v in obs.legal.items()},
                                            streams, obs.step_idx[rows].tolist())
            if rows.numel() == 0:
                return {}
            legal = {k: v[rows].to(self.device) for k, v in obs.legal.items()}
            out["mu"] = self.spec.behaviour_logp(out, legal)
            out["value"] = out["value"].float()
        finally:
            self.net.train(was_training)
        return {k: v.detach().cpu() for k, v in out.items()}


@dataclass
class Stream:
    """One learner stream of one wave: frames 0..T-1 of one episode row."""

    obs: Dict[str, torch.Tensor]
    legal: Dict[str, torch.Tensor]
    actions: Dict[str, torch.Tensor]
    mu: torch.Tensor          # [T] behaviour log-prob
    value: torch.Tensor       # [T] actor value (head units)
    seat: torch.Tensor        # [T] acting seat of the frame
    vmask: torch.Tensor       # [T] learner acting (value / KL frames)
    pmask: torch.Tensor       # [T] policy-loss frames (acting, not forced)
    forced: torch.Tensor      # [T]
    rewards: torch.Tensor     # [T, S] reward of every seat after the frame
    done: torch.Tensor        # [T] episode ended after the frame
    routes: Optional[torch.Tensor] = None   # [T, C, S_tok, k] actor's MoE experts (replay)
    # learner-side, filled before the epochs
    lp_old: Optional[torch.Tensor] = None
    v_old: Optional[torch.Tensor] = None
    adv: Optional[torch.Tensor] = None
    ret: Optional[torch.Tensor] = None
    keep: Optional[torch.Tensor] = None
    old_pred: Optional[torch.Tensor] = None
    # the frozen reference's KL inputs (--cache_ref_kl): per-head logits [T, ...] of
    # its il_head_terms (full-head KL) or its log-prob of the taken action [T] (k3),
    # on the learner device
    ref_logits: Optional[Dict[str, torch.Tensor]] = None
    ref_lp: Optional[torch.Tensor] = None

    @property
    def length(self) -> int:
        return int(self.mu.shape[0])


@dataclass
class WaveStats:
    episodes: int = 0
    steps: int = 0
    tokens: int = 0
    ep_len_sum: float = 0.0
    score_sum: float = 0.0
    score_n: int = 0
    mirror: int = 0
    timeouts: int = 0


def _assemble_streams(recs: List[Dict[str, Any]]) -> List[Stream]:
    """Per-step learner records → one Stream per row (records are in step
    order; a stable sort by row keeps each row's frames in time order)."""
    if not recs:
        return []
    rows = torch.cat([r["rows"] for r in recs])
    order = torch.sort(rows, stable=True).indices
    _, counts = torch.unique_consecutive(rows[order], return_counts=True)
    counts = counts.tolist()

    def cat(get) -> List[torch.Tensor]:
        return list(torch.cat([get(r) for r in recs])[order].split(counts))

    def cat_dict(key) -> List[Dict[str, torch.Tensor]]:
        keys = list(recs[0][key])
        parts = {k: cat(lambda r, k=k: r[key][k]) for k in keys}
        return [{k: parts[k][i] for k in keys} for i in range(len(counts))]

    obs, legal, actions = cat_dict("obs"), cat_dict("legal"), cat_dict("actions")
    keys = ["mu", "value", "seat", "vmask", "pmask", "forced", "rewards", "done"]
    if "routes" in recs[0]:
        keys.append("routes")
    fields_ = {k: cat(lambda r, k=k: r[k]) for k in keys}
    return [Stream(obs=obs[i], legal=legal[i], actions=actions[i],
                   **{k: fields_[k][i] for k in fields_}) for i in range(len(counts))]


#: --timeout_reward defaults: the games whose time limit otherwise scores better than a loss
#: (SC2 / Pokémon / HoK: 0 on truncation; Tetris: a draw at max_turns).  Generals scores its
#: turn limit by material minus a penalty (env/generals_env.py), Clash Royale's overtime ends
#: in the game's own tower-hp tiebreak, ARC's reward is the levels cleared.
TIMEOUT_REWARD: Dict[str, float] = {"sc2": -1.0, "pokemon": -1.0, "hok": -1.0, "tetris": -1.0}
#: --shaping / --shaping_anneal defaults.  HoK's SDK reward (gold, experience, hp, towers,
#: kills, last hits) sums to ~22 per expert episode at weight 1 against the ±1 of the crystal:
#: at 0.02 a typical episode's shaping (~0.45) stays below the outcome, then fades to 0.
SHAPING: Dict[str, float] = {"hok": 0.02}
SHAPING_ANNEAL: Dict[str, int] = {"hok": 300}


def shaping_weight(s0: float, anneal: int, iteration: int) -> float:
    """The shaping weight at ``iteration``: ``s0`` falling linearly to 0 over ``anneal``."""
    if anneal <= 0:
        return s0
    return s0 * max(0.0, 1.0 - iteration / anneal)


#: games whose time limit is a rules-level draw rather than an env truncation
TIMEOUT_IS_DRAW = frozenset({"tetris"})


def apply_timeout(rew: torch.Tensor, done: torch.Tensor, info: Mapping[str, Any],
                  value: Optional[float], draw: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rewards [E, S] with every seat of a timed-out episode set to ``value``, and the
    timed-out rows [E] bool.  Timed out = done and truncated (or, ``draw``: a done
    episode without a winner and all-zero rewards)."""
    done = torch.as_tensor(done).bool().reshape(-1)
    hit = torch.zeros_like(done)
    if value is None:
        return rew, hit
    if "truncated" in info:
        hit |= done & torch.as_tensor(info["truncated"]).bool().reshape(-1)
    if draw:
        no_winner = torch.as_tensor(info["winner"]).reshape(-1) < 0 if "winner" in info else True
        hit |= done & no_winner & (rew.reshape(done.shape[0], -1) == 0).all(-1)
    if bool(hit.any()):
        rew = rew.clone()
        rew[hit] = value
    return rew, hit


def alarm_reasons(row: Mapping[str, float], base_len: Optional[float], kl_budget: float, *,
                  timeout: float, len_x: float, kl_x: float) -> List[str]:
    """What looks like reward hacking in one iteration's record of a game (its row of
    metrics.jsonl): time-outs, episodes growing long, the policy leaving the IL one."""
    out = []
    tr, ln, kl = row.get("timeout_rate", float("nan")), row.get("ep_len", float("nan")), \
        row.get("kl", float("nan"))
    if timeout > 0 and math.isfinite(tr) and tr > timeout:
        out.append(f"time-out rate {tr:.3f} > {timeout}")
    if len_x > 0 and base_len and math.isfinite(ln) and ln > len_x * base_len:
        out.append(f"episode length {ln:.0f} > {len_x} x {base_len:.0f} (its first iteration)")
    if kl_x > 0 and math.isfinite(kl) and kl > kl_x * kl_budget:
        out.append(f"KL {kl:.4f} > {kl_x} x budget {kl_budget}")
    return out


def collect_wave(env: VecGameEnv, league: League, actor: LearnerActor, *, seed: int,
                 max_steps: int, timeout_reward: Optional[float] = None,
                 timeout_draw: bool = False) -> Tuple[List[Stream], WaveStats]:
    """Reset every env and play one lockstep wave to the end: the learner on
    its rows, the league's opponents on theirs.  Returns the learner's
    streams (episode rows it plays on) and episode statistics."""
    obs = env.reset(seed=seed)
    league.update(obs)
    actor.begin_wave()
    R = obs.n_rows
    recs: List[Dict[str, Any]] = []
    st = WaveStats(mirror=int(league.is_mirror().sum()))
    winner = torch.full((env.num_envs,), -1, dtype=torch.long)
    final_ret: Optional[torch.Tensor] = None    # per-seat payoffs at the end (gmsim envs)
    ep_len = torch.zeros(env.num_envs, dtype=torch.long)
    last: Dict[int, Tuple[int, int]] = {}    # row → (record, position) of its latest frame
    try:
        while bool(obs.active.any()):
            if st.steps >= max_steps:
                raise RuntimeError(f"{env.game}: wave exceeded --max_wave_steps {max_steps}")
            rows = league.learner_rows(obs).nonzero().squeeze(1)
            out = actor(obs, rows)
            if rows.numel():
                acts = env.replay_actions(out)
                la = scatter_rows(acts, rows, R)
            else:
                acts, la = {}, {}
            acting = league.learner_acting(obs)[rows]
            pmask = league.learner_policy_mask(obs)[rows]
            nxt, rew, done, info = env.step(league.act(obs, la))
            rew, timed_out = apply_timeout(rew, done, info, timeout_reward, timeout_draw)
            st.timeouts += int(timed_out.sum())
            if rows.numel():
                env_rows = obs.env_id[rows]
                recs.append({
                    "rows": rows.clone(),
                    "obs": {k: v[rows] for k, v in obs.obs.items()},
                    "legal": actor.spec.rollout_legal({k: v[rows] for k, v in obs.legal.items()},
                                                      out),
                    "actions": {k: v.cpu() for k, v in acts.items()},
                    "mu": out["mu"], "value": out["value"],
                    "seat": obs.to_play[rows].long(), "vmask": acting.bool(),
                    "pmask": pmask.bool(), "forced": obs.forced[rows].bool(),
                    "rewards": rew[env_rows].float(), "done": done[env_rows].bool()})
                if "moe_routes" in out:
                    recs[-1]["routes"] = out["moe_routes"]
                for pos, r in enumerate(rows.tolist()):
                    last[r] = (len(recs) - 1, pos)
            # per-seat streams of alternating games (env/vec_gmsim.py): a learner
            # row idle on this step (another seat moved) still earns its seat's reward
            # and sees its episode end — credited to its latest frame.  A no-op for
            # layouts whose learner rows act on every step of their episode (skipped
            # for lockstep envs: every env but those per-seat gmrules games).
            if not getattr(env, "lockstep", True):
                earned = ((rew != 0).any(-1) | done)[obs.env_id] & league.learner_streams(obs)
                if rows.numel():
                    earned[rows] = False
                for r in earned.nonzero().squeeze(1).tolist():
                    if r in last:
                        k, pos = last[r]
                        e = int(obs.env_id[r])
                        recs[k]["rewards"][pos] += rew[e].float()
                        recs[k]["done"][pos] |= bool(done[e])
            winner = torch.where(done, info["winner"].long(), winner)
            if "returns" in info:           # shared wins (teammates) in learner_score
                ret = info["returns"].double()
                final_ret = torch.where(done.unsqueeze(-1), ret,
                                        torch.zeros_like(ret) if final_ret is None else final_ret)
            ep_len = torch.where(done, info["episode_len"].long(), ep_len)
            league.update(nxt)
            obs = nxt
            st.steps += 1
    finally:
        league.release_all()
    streams = _assemble_streams(recs)
    st.episodes = env.num_envs
    st.ep_len_sum = float(ep_len.sum())
    score = league.learner_score(winner, returns=final_ret)
    ok = ~torch.isnan(score)
    st.score_sum, st.score_n = float(score[ok].sum()), int(ok.sum())
    st.tokens = int(sum(int(s.vmask.sum()) for s in streams))
    return streams, st


# ======================================================================
# Learner batches
# ======================================================================

def _pad(ts: Sequence[torch.Tensor], T: int) -> torch.Tensor:
    out = ts[0].new_zeros((len(ts), T) + tuple(ts[0].shape[1:]))
    for i, t in enumerate(ts):
        out[i, :t.shape[0]] = t
    return out


def build_batch(streams: Sequence[Stream], device: torch.device,
                extra: Sequence[str] = ()) -> Dict[str, Any]:
    """Tail-padded [B, T, ...] tensors of ``streams`` on ``device``: obs,
    legal, actions, frame_mask, the recorded MoE ``routes`` (when the
    streams carry them) and the per-frame fields in ``extra``."""
    T = max(s.length for s in streams)

    def stack(get) -> torch.Tensor:
        return _pad([get(s) for s in streams], T).to(device)

    b: Dict[str, Any] = {
        "obs": {k: stack(lambda s, k=k: s.obs[k]) for k in streams[0].obs},
        "legal": {k: stack(lambda s, k=k: s.legal[k]) for k in streams[0].legal},
        "actions": {k: stack(lambda s, k=k: s.actions[k]) for k in streams[0].actions},
        "frame_mask": stack(lambda s: torch.ones(s.length, dtype=torch.bool)),
    }
    if streams[0].routes is not None:
        b["routes"] = stack(lambda s: s.routes)
    for k in extra:
        b[k] = stack(lambda s, k=k: getattr(s, k))
    return b


def _chunks(idx: Sequence[int], n: int) -> List[List[int]]:
    idx = list(idx)
    return [idx[i:i + n] for i in range(0, len(idx), max(1, n))]


def _length_chunks(idx: Sequence[int], lengths: Sequence[int], rows: int,
                   frames: int = 0) -> List[List[int]]:
    """Learner-forward chunks of the stream indices ``idx`` (``lengths`` indexed by
    stream): longest first, so a chunk's rows pad to similar lengths.  A chunk
    holds ``rows`` streams, or with a ``frames`` budget as many as keep rows x its
    longest (first) stream's length <= frames — at least one."""
    out: List[List[int]] = []
    cur: List[int] = []
    for i in sorted(idx, key=lambda j: -lengths[j]):
        full = (len(cur) + 1) * lengths[cur[0]] > frames if frames > 0 and cur \
            else len(cur) >= max(1, rows)
        if cur and full:
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def learner_forward(net, spec: RLGameSpec, b: Dict[str, Any], depth: int,
                    device: torch.device, bf16: bool, replay_routes: bool = False
                    ) -> Dict[str, torch.Tensor]:
    """``replay_routes``: take the batch's recorded actor routing (the
    learner network only — another network, e.g. the frozen reference,
    routes on its own)."""
    routes = b.get("routes") if replay_routes else None
    with _autocast(device, bf16):
        return net(b["obs"], legal=b["legal"], labels=spec.labels(b["actions"], net.cfg),
                   act_steps=depth, frame_mask=b["frame_mask"], routes=routes)


# ======================================================================
# IL / KL replay stream
# ======================================================================

class ReplayStream:
    """One game's IL replay batches (train_multi_game's datasets, crops,
    sampler and collate).  :meth:`restart` builds a fresh loader — called at
    every iteration start after the per-iteration reseed, so the data order
    is a function of (seed, rank, iteration) and a resume reproduces it."""

    def __init__(self, game: str, data_dir: str, batch_size: int, max_len: Optional[int],
                 num_workers: int, frames: int = 0):
        self.game = game
        # frames > 0: whole records from their first frame (max_len = a cap), batches of
        # similar-length records within the frame budget (WholeRecordBatchSampler)
        self.dataset = open_dataset(game, data_dir, None if game in UNCROPPABLE and not frames
                                    else max_len, whole_records=frames > 0)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames = frames
        self._it = None

    def restart(self) -> None:
        if self.frames > 0:
            self._loader = DataLoader(self.dataset,
                                      batch_sampler=WholeRecordBatchSampler(self.dataset,
                                                                            self.frames),
                                      num_workers=self.num_workers,
                                      collate_fn=collate_game_sequences)
        else:
            self._loader = DataLoader(self.dataset, batch_size=self.batch_size,
                                      sampler=training_sampler(self.dataset),
                                      num_workers=self.num_workers,
                                      collate_fn=collate_game_sequences)
        self._it = iter(self._loader)

    def next(self) -> Dict:
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self._loader)
            return next(self._it)


# ======================================================================
# Diagnostics: one fixed-size all-reduce per iteration
# ======================================================================

DIAG_KEYS = (
    "tokens_v", "tokens_p", "streams", "episodes", "ep_len_sum", "score_sum", "score_n",
    "mirror", "timeouts", "vs_init_sum", "vs_init_n", "ret_sum",
    "ent_sum", "ent_n", "kl_sum", "kl_n", "pol_n", "approx_kl_sum", "clip_frac_sum",
    "is_weight_sum", "is_trunc_sum", "vloss_sum", "vloss_n",
    "canary_oldmu_sum", "canary_ratio_sum", "canary_n",
    "ev_n", "ev_g", "ev_g2", "ev_e", "ev_e2",
    "il_sum", "il_n", "rkl_sum", "rkl_n", "rvl_sum",
    "gn_rl", "gn_rl_n", "gn_replay", "gn_replay_n",
    "learn_s", "learn_fwd", "replay_s", "rollout_s", "rollout_steps",
)


class Diag:
    def __init__(self, games: Sequence[str], n_moe: Tuple[int, int]):
        self.games = list(games)
        self.v = {g: {k: 0.0 for k in DIAG_KEYS} for g in self.games}
        L, E = n_moe
        self.loads = torch.zeros(len(self.games), L * E, dtype=torch.float64)
        self.n_moe = n_moe

    def add(self, g: str, **kw: float) -> None:
        d = self.v[g]
        for k, x in kw.items():
            d[k] += float(x)

    def add_loads(self, g: str, delta: torch.Tensor) -> None:
        self.loads[self.games.index(g)] += delta.detach().double().cpu().reshape(-1)

    def reduce(self) -> None:
        t = torch.tensor([[self.v[g][k] for k in DIAG_KEYS] for g in self.games],
                         dtype=torch.float64)
        t = torch.cat([t, self.loads], dim=1)
        if dist_active():
            dt = t.to(collective_device())
            dist.all_reduce(dt, op=dist.ReduceOp.SUM)
            t = dt.cpu()
        K = len(DIAG_KEYS)
        for i, g in enumerate(self.games):
            self.v[g] = {k: float(t[i, j]) for j, k in enumerate(DIAG_KEYS)}
        self.loads = t[:, K:]

    def mean(self, g: str, num: str, den: str) -> float:
        d = self.v[g]
        return d[num] / d[den] if d[den] > 0 else float("nan")

    def explained_variance(self, g: str) -> float:
        d = self.v[g]
        n = d["ev_n"]
        if n <= 1:
            return float("nan")
        var_g = d["ev_g2"] / n - (d["ev_g"] / n) ** 2
        var_e = d["ev_e2"] / n - (d["ev_e"] / n) ** 2
        return 1.0 - var_e / var_g if var_g > 1e-12 else float("nan")

    def load_entropy(self, g: str) -> float:
        L, E = self.n_moe
        if L == 0:
            return float("nan")
        x = self.loads[self.games.index(g)].reshape(L, E)
        if float(x.sum()) <= 0:
            return float("nan")
        return float(expert_load_entropy(x).nanmean())


class _GradProbe:
    """Norm of the gradient each loss component adds to ``params`` (the
    shared core): :meth:`delta_norm` after each component's backward.  Keeps
    one copy of the accumulated gradients (used on one minibatch per
    iteration: the replay-vs-RL balance canary)."""

    def __init__(self, params: Sequence[torch.nn.Parameter]):
        self.params = list(params)
        self.prev = [None if p.grad is None else p.grad.detach().clone() for p in self.params]

    @torch.no_grad()
    def delta_norm(self) -> float:
        norms = []
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.detach()
            if self.prev[i] is None:
                norms.append(g.float().norm())
                self.prev[i] = g.clone()
            else:
                norms.append((g - self.prev[i]).float().norm())
                self.prev[i].copy_(g)
        return float(torch.stack(norms).norm()) if norms else 0.0


def _moe_load_vector(model: torch.nn.Module) -> Optional[torch.Tensor]:
    layers = moe_layers(model)
    if not layers:
        return None
    return torch.stack([m.load.detach().clone() for m in layers])


# ======================================================================
# Trainer
# ======================================================================

@dataclass
class GameRL:
    """Per simulator game: env, actor, league pieces, hyper-parameters."""

    game: str
    spec: RLGameSpec
    env: VecGameEnv
    actor: LearnerActor
    pool_policy: Optional[PoolPolicy]
    mirror_prob: float
    gamma: float
    lam: float
    rows_per_forward: int
    hl_gauss: bool
    frames_per_forward: int = 0
    timeout_reward: Optional[float] = None
    shaping: Optional[Tuple[float, int]] = None      # (initial weight, anneal iterations)
    streams: List[Stream] = field(default_factory=list)


class JointRLTrainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # MIG slices: this rank's own instance only — before pick_device() touches CUDA
        pin_rank_device()
        device = torch.device(args.device) if args.device else pick_device()
        self.dist_env, self.device = init_distributed(device, args.dist_timeout,
                                                      args.dist_backend)
        self.rank, self.world = self.dist_env.rank, self.dist_env.world
        self.rank0 = self.dist_env.is_main
        seed_all(derive_seed(args.seed, self.rank, 0, 0))
        self._setup()

    def log(self, *a, **kw) -> None:
        if self.rank0:
            print(*a, **kw, flush=True)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        a, device = self.args, self.device
        model = load_multi(a.init, device)
        self.model: MultiGameNetwork = model
        self.games = model.games
        self.ref = load_multi(a.init, device).requires_grad_(False).eval()
        self.init_hash = module_fingerprint(self.ref)
        if self.dist_env.enabled:
            broadcast_module_state(self.model, src=0)
        self.rl_games = list(dict.fromkeys(a.rl))
        replay_dirs = _pairs(a.data, str, "DIR")
        for g in self.rl_games:
            if g not in self.games:
                raise SystemExit(f"--rl {g}: not a game of {a.init} ({self.games})")
            if not has_simulator(g):
                raise SystemExit(f"--rl {g}: no simulator (env/vec_game.py VEC_ENV_REGISTRY)")
        for g in replay_dirs:
            if g not in self.games:
                raise SystemExit(f"--data {g}: not a game of {a.init} ({self.games})")
        if not self.rl_games and not replay_dirs:
            raise SystemExit("nothing to train: give --rl games and/or --data replay shards")
        if a.seed < 0 or a.epochs < 1 or a.minibatches < 1:
            raise SystemExit("--seed must be >= 0, --epochs and --minibatches >= 1")
        first = model[self.games[0]]
        self.depth = a.act_steps or first.default_depth
        self.moe = moe_layers(model)
        self.n_moe = (len(self.moe), self.moe[0].n_experts if self.moe else 0)
        # the shared core's parameters (each once): the per-component gradient canary
        self.core_params = [p for n, p in model.named_parameters()
                            if p.requires_grad and is_core_key(n.split(".", 2)[2])]

        # --- per-game hyper-parameters -----------------------------------
        envs_n = _pairs(a.envs, int, "N")
        env_kw = _env_kwargs(a.env_kw)
        mirror = _pairs(a.mirror_prob, float, "P")
        gamma = _pairs(a.gamma, float, "G")
        lam = _pairs(a.lam, float, "L")
        rpf = _pairs(a.rows_per_forward, int, "N")
        fpf = _pairs(a.frames_per_forward, int, "N")
        self.game_weight = _pairs(a.game_weight, float, "W")
        shaping = {**SHAPING, **_pairs(a.shaping, float, "S")}
        anneal = {**SHAPING_ANNEAL, **_pairs(a.shaping_anneal, int, "ITERS")}
        timeout = {**TIMEOUT_REWARD, **_pairs(a.timeout_reward,
                                             lambda v: None if v == "none" else float(v), "R")}
        for flag, vals in (("--envs", envs_n), ("--env_kw", env_kw), ("--mirror_prob", mirror),
                           ("--gamma", gamma), ("--lam", lam), ("--rows_per_forward", rpf),
                           ("--frames_per_forward", fpf),
                           ("--game_weight", self.game_weight),
                           ("--timeout_reward", _pairs(a.timeout_reward, str, "R")),
                           ("--shaping", _pairs(a.shaping, str, "S")),
                           ("--shaping_anneal", _pairs(a.shaping_anneal, str, "ITERS")),
                           ("--share", _pairs(a.share, float, "W")),
                           ("--entropy_coef", _pairs(a.entropy_coef, float, "C")),
                           ("--entropy_target", _pairs(a.entropy_target, float, "H")),
                           ("--kl_beta", _pairs(a.kl_beta, float, "B")),
                           ("--kl_budget", _pairs(a.kl_budget, float, "K"))):
            _check_games(flag, vals, self.rl_games, "an --rl game")
        store = a.snapshot_dir or os.path.join(a.save_dir, "snapshots")
        use_pool = any(mirror.get(g, 1.0) < 1.0 for g in self.rl_games)
        # a resume keeps the store even with opponents off: its checkpoint's pool
        # may be file-backed (and is carried on for a later resume that turns
        # them back on).  Evicted files are deleted only once rl_latest.pt no
        # longer refers to them (flushed in save), so a requeue restores the pool
        self.pool = SnapshotPool(
            max_size=a.pool_size, seed=a.seed,
            store_dir=None if store == "memory" or not (use_pool or a.resume) else store,
            defer_delete=a.ckpt_minutes > 0 or a.ckpt_every > 0)
        self.templates: Dict[str, torch.nn.Module] = {}
        self.rl: Dict[str, GameRL] = {}
        for gi, g in enumerate(self.rl_games):
            net = model[g]
            spec = game_spec(g)
            note = spec.adopt_env_kw(net, env_kw.get(g, {}))
            if note:
                self.log(f"[rl] {note}")
            kw = {**spec.env_kwargs(net.cfg), **env_kw.get(g, {})}
            env = make_vec_env(g, envs_n.get(g, 8), seed=derive_seed(a.seed, self.rank, gi),
                               **kw)
            if env.autoreset:
                raise SystemExit(f"{g}: the trainer's lockstep waves need autoreset off")
            if a.actor == "fixed" and not getattr(env, "lockstep", True):
                raise SystemExit(f"{g}: --actor fixed steps every row at one frame index, but "
                                 f"this env's per-seat streams advance separately (seats act "
                                 f"on different turns); use --actor stream")
            spec.check_env(env, net)
            if g in shaping and not hasattr(env, "shaping"):
                raise SystemExit(f"--shaping {g}: its env has no shaped reward")
            actor = LearnerActor(net, spec, self.depth, device, bf16=a.bf16, mode=a.actor,
                                 cuda_graph=a.cuda_graph, record_routes=bool(a.routing_replay))
            p_mirror = mirror.get(g, 1.0)
            pool_policy = None
            if p_mirror < 1.0:
                # host-side template of the game's network (materialised on the
                # device per loaded snapshot)
                self.templates[g] = copy.deepcopy(net).cpu().requires_grad_(False).eval()
                pool_policy = PoolPolicy(self.pool, self._snapshot_loader(g),
                                         act_steps=self.depth, device=device,
                                         max_nets=a.pool_max_nets)
            self.rl[g] = GameRL(
                game=g, spec=spec, env=env, actor=actor, pool_policy=pool_policy,
                mirror_prob=p_mirror, gamma=gamma.get(g, 1.0), lam=lam.get(g, 0.95),
                rows_per_forward=rpf.get(g, 8),
                hl_gauss=isinstance(net.value_head, HLGaussValueHead),
                frames_per_forward=fpf.get(g, a.frames_per_forward_default),
                timeout_reward=timeout.get(g),
                shaping=(shaping[g], anneal.get(g, 0)) if g in shaping else None)

        # --- replay stream ------------------------------------------------
        rw = _pairs(a.replay_weight, float, "W")
        rb = _pairs(a.replay_batch, int, "N")
        rml = _pairs(a.replay_max_len, int, "N")
        rfr = _pairs(a.replay_frames, int, "N")
        rtf = _pairs(a.replay_tf, float, "P")
        rvw = _pairs(a.replay_value_weight, float, "W")
        rkw = _pairs(a.replay_kl_weight, float, "W")
        for flag, vals in (("--replay_weight", rw), ("--replay_batch", rb),
                           ("--replay_max_len", rml), ("--replay_frames", rfr),
                           ("--replay_tf", rtf),
                           ("--replay_value_weight", rvw), ("--replay_kl_weight", rkw)):
            _check_games(flag, vals, list(replay_dirs), "a --data game")
        self.replay: Dict[str, ReplayStream] = {}
        self.replay_weight: Dict[str, float] = {}
        self.replay_tf: Dict[str, float] = {}
        self.replay_value_weight: Dict[str, float] = {}
        self.replay_kl_weight: Dict[str, float] = {}
        rel = {g: rw.get(g, a.replay_weight_default) for g in replay_dirs}
        if any(v < 0 for v in rel.values()) or a.replay_weight_total < 0:
            raise SystemExit("replay weights must be >= 0")
        rel_sum = sum(rel.values())
        for g, d in replay_dirs.items():
            if g == "generals" and g not in rml:
                raise SystemExit("--data generals: bound its prefix crop with "
                                 "--replay_max_len generals=N (as train_multi_game)")
            self.replay[g] = ReplayStream(g, d, rb.get(g, a.replay_batch_size),
                                          rml.get(g, a.replay_max_len_default), a.num_workers,
                                          frames=rfr.get(g, a.replay_frames_default))
            if a.replay_weight_total > 0:
                self.replay_weight[g] = (a.replay_weight_total * rel[g] / rel_sum
                                         if rel_sum > 0 else 0.0)
            else:
                self.replay_weight[g] = rel[g]
            self.replay_tf[g] = rtf.get(g, default_tf(g))
            self.replay_value_weight[g] = rvw.get(g, 0.0 if g in self.rl else 0.2)
            self.replay_kl_weight[g] = rkw.get(g, 0.0)
            if self.replay_kl_weight[g] > 0 and LABEL_KEYS.get(g) and self.replay_tf[g] < 1:
                self.log(f"[warn] {g}: replay KL with teacher forcing {self.replay_tf[g]} < 1 — "
                         f"self-sampled conditioning differs between policy and reference")

        # --- algorithm state ---------------------------------------------
        rl = self.rl_games or [self.games[0]]     # controllers need a non-empty game list
        modes = {g: PASSTHROUGH for g in rl} if a.no_popart else None
        self.popart = PopArt.from_model(model, games=rl, decay=a.popart_decay, modes=modes)
        self.advnorm = AdvantageNormalizer(rl, decay=a.adv_decay)
        ent_init = _pairs(a.entropy_coef, float, "C")
        self.ent_target = _pairs(a.entropy_target, float, "H")
        self.entc = AdaptiveEntropyCoef(
            rl, target={g: self.ent_target.get(g, 0.0) for g in rl},
            init={g: ent_init.get(g, 0.01) for g in rl}, lr=a.entropy_lr,
            min_coef=a.entropy_min, max_coef=a.entropy_max, mode=a.entropy_mode)
        kl_init = _pairs(a.kl_beta, float, "B")
        self.kl_off = {g: kl_init.get(g, 0.05) <= 0 for g in rl}
        budget = _pairs(a.kl_budget, float, "K")
        self.klb = AdaptiveKLBeta(
            rl, budget={g: budget.get(g, 0.03) for g in rl},
            init={g: max(kl_init.get(g, 0.05), a.kl_min) for g in rl}, rate=a.kl_rate,
            min_beta=a.kl_min, max_beta=a.kl_max)
        shares = _pairs(a.share, float, "W")
        self.bandit = LearningProgressBandit(rl) if a.bandit_mix > 0 else None
        self.sched = TokenQuotaScheduler(
            {g: shares.get(g, 1.0) for g in rl}, floor=a.share_floor, cap=a.share_cap,
            bandit=self.bandit, bandit_mix=a.bandit_mix, debt_decay=a.debt_decay)

        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamATan2(multi_game_param_groups(model, a.weight_decay), lr=a.lr,
                                   betas=(a.beta1, a.beta2), weight_decay=a.weight_decay)
        self.ema = ModelEMA(model, mu=a.ema_rate) if a.ema_rate > 0 else None
        self.iteration = 0
        self.il_baseline: Dict[str, float] = {}
        self.kl_budget = {g: budget.get(g, 0.03) for g in rl}
        #: reward-hacking alarms: each game's first-iteration episode length, the
        #: consecutive iterations it has looked wrong, and the games stopped {game: why}
        self.alarm_base_len: Dict[str, float] = {}
        self.alarm_streak: Dict[str, int] = {}
        self.vs_init_streak: Dict[str, int] = {}
        self.stopped: Dict[str, str] = {}
        #: per-game cost of the latest iteration that played it (rollout + learner
        #: seconds, rank-identical): the --rl_placement split assignment
        self.game_cost: Dict[str, float] = {}
        #: per-game seconds of one replay batch (the latest measurement, rank-identical):
        #: the --replay_split assignment
        self.replay_cost: Dict[str, float] = {}
        #: per-game learner seconds of an iteration (on its rank, rank-identical): the
        #: RL load a rank carries into every minibatch, which --replay_split tops up
        self.learn_cost: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []
        if a.resume:
            self._load_resume(a.resume)
        if self.rank0:
            os.makedirs(a.save_dir, exist_ok=True)
            if a.resume:
                self._truncate_metrics()
        if any(g.pool_policy is not None for g in self.rl.values()) and len(self.pool) == 0:
            # the Stage-1 policy opens the pool (a resume that turns opponents
            # on for the first time starts it from the current policy)
            self._add_snapshot(tag="resume" if a.resume else "init")
        n_total = sum(p.numel() for p in model.parameters())
        self.log(f"[init] games={self.games} rl={self.rl_games} replay={list(self.replay)} "
                 f"core={first.cfg.core_type} moe={first.cfg.moe_experts} depth={self.depth} "
                 f"params={n_total:,} device={device.type} world={self.world} "
                 f"actor={a.actor}{'+graph' if a.cuda_graph else ''} "
                 f"routing_replay={'on' if any(r.actor.record_routes for r in self.rl.values()) else 'off'}")
        for g, r in self.rl.items():
            self.log(f"[rl] {g}: envs={r.env.num_envs} gamma={r.gamma} lam={r.lam} "
                     f"mirror={r.mirror_prob} value={'hl_gauss' if r.hl_gauss else 'popart'} "
                     f"kl={'off' if self.kl_off[g] else ('full' if r.spec.full_kl else 'k3')}")
        if self.replay:
            ws = " ".join(f"{g}={w:.4g}" for g, w in self.replay_weight.items())
            self.log(f"[replay] weights {ws} (total {sum(self.replay_weight.values()):.4g}; "
                     f"the RL loss of all --rl games is 1)")

    def _snapshot_loader(self, g: str) -> Callable[[Dict[str, torch.Tensor]], torch.nn.Module]:
        pre = f"nets.{g}."

        def load(sd: Dict[str, torch.Tensor]) -> torch.nn.Module:
            net = copy.deepcopy(self.templates[g])
            own = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            # the shared core is stored once per snapshot (_policy_state)
            own.update({k[len(SNAPSHOT_CORE):]: v for k, v in sd.items()
                        if k.startswith(SNAPSHOT_CORE)})
            net.load_state_dict(own)
            return net.to(self.device).eval().requires_grad_(False)
        return load

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------

    def _policy_state(self) -> Dict[str, torch.Tensor]:
        """The weights published to the pool (--snapshot_weights: the live
        policy or its EMA): every simulator game's own modules under
        ``nets.<game>.`` (so a resume may turn on opponents for any of them)
        and the shared core ONCE under ``core.``.  Uncopied views: the pool
        copies them to the host."""
        ema = self.ema if self.args.snapshot_weights == "ema" else None
        out: Dict[str, torch.Tensor] = {}
        for k, v in self.model.state_dict().items():
            if ema is not None and k in ema.alias:
                v = ema.shadow[ema.alias[k]]
            for g in self.rl_games:
                pre = f"nets.{g}."
                if k.startswith(pre):
                    rest = k[len(pre):]
                    if not is_core_key(rest):
                        out[k] = v
                    elif g == self.rl_games[0]:
                        out[SNAPSHOT_CORE + rest] = v
                    break
        return out

    def _add_snapshot(self, tag: str) -> None:
        """Every rank's pool gets the same entry: in memory each rank adds the
        (rank-identical) weights; file-backed rank 0 writes the file and the
        pool state is broadcast (one file per snapshot for all ranks)."""
        meta = {"iteration": self.iteration, "tag": tag}
        if self.pool.store_dir is None or not self.dist_env.enabled:
            if self.pool.store_dir is None or self.rank0:
                self.pool.add(self._policy_state(), meta)
            return
        if self.rank0:
            self.pool.add(self._policy_state(), meta)
        state = [self.pool.state_dict() if self.rank0 else None]
        dist.broadcast_object_list(state, src=0)
        if not self.rank0:
            self.pool.load_state_dict(state[0])

    # ------------------------------------------------------------------
    # rollouts
    # ------------------------------------------------------------------

    def _rollouts(self, it: int, quotas: Dict[str, int], diag: Diag) -> Dict[str, WaveStats]:
        a = self.args
        stats: Dict[str, WaveStats] = {}
        for r in self.rl.values():      # free the last iteration's buffers first
            r.streams = []
        for gi, g in enumerate(self.rl_games):
            r = self.rl[g]
            q = quotas.get(g, 0)
            if q <= 0:
                continue
            if r.shaping is not None:
                r.env.shaping = shaping_weight(*r.shaping, it)
            if r.pool_policy is not None:
                self.pool.reseed(derive_seed(a.seed, self.rank, it, gi, 1))
                r.pool_policy.refresh()
            total = WaveStats()
            wave = 0
            t_game = time.time()
            while total.tokens < q:
                opps = {"pool": r.pool_policy} if r.pool_policy is not None else {}
                league = League(r.env, opps, mirror_prob=r.mirror_prob,
                                learner_seat=a.learner_seat,
                                seed=derive_seed(a.seed, self.rank, it, gi, 2, wave))
                streams, st = collect_wave(r.env, league, r.actor,
                                           seed=derive_seed(a.seed, self.rank, it, gi, 3, wave),
                                           max_steps=a.max_wave_steps,
                                           timeout_reward=r.timeout_reward,
                                           timeout_draw=g in TIMEOUT_IS_DRAW)
                r.streams.extend(streams)
                for k in ("episodes", "steps", "tokens", "ep_len_sum", "score_sum", "score_n",
                          "mirror", "timeouts"):
                    setattr(total, k, getattr(total, k) + getattr(st, k))
                wave += 1
                if st.tokens == 0:
                    raise RuntimeError(f"{g}: a wave of {r.env.num_envs} envs produced no "
                                       f"learner tokens")
            stats[g] = total
            if a.eval_every > 0 and it > 0 and it % a.eval_every == 0 and r.env.n_seats >= 2:
                ev = self._eval_vs_init(g, r, it, gi)
                diag.add(g, vs_init_sum=ev.score_sum, vs_init_n=ev.score_n)
            diag.add(g, episodes=total.episodes, ep_len_sum=total.ep_len_sum,
                     score_sum=total.score_sum, score_n=total.score_n, mirror=total.mirror,
                     timeouts=total.timeouts,
                     streams=len(r.streams), rollout_s=time.time() - t_game,
                     rollout_steps=total.steps)
        return stats

    # ------------------------------------------------------------------
    # π_old, targets, statistics
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _recompute(self, g: str) -> None:
        """log π_old and V of every stored frame: the iteration's weights and
        routing bias (or the actor's replayed routing), the same (sparse-MoE,
        eval) forward the epochs use."""
        r = self.rl[g]
        net = self.model[g]
        ref = self.ref[g] if self.args.cache_ref_kl and not self.kl_off[g] else None
        with MoEFFN.inference_path("sparse"):
            for chunk in self._forward_chunks(g, range(len(r.streams))):
                ss = [r.streams[i] for i in chunk]
                b = build_batch(ss, self.device, ("forced",))
                pred = learner_forward(net, r.spec, b, self.depth, self.device, self.args.bf16,
                                       replay_routes=True)
                lp, _ = r.spec.logp_entropy(net, pred, b["actions"], b["legal"])
                v = pred["value"].float()
                for j, s in enumerate(ss):
                    s.lp_old = lp[j, :s.length].detach().cpu()
                    s.v_old = v[j, :s.length].detach().cpu()
                if ref is not None:
                    # the reference routes on its own (no replay), as in _rl_chunk_loss
                    ref_pred = learner_forward(ref, r.spec, b, self.depth, self.device,
                                               self.args.bf16)
                    if r.spec.full_kl:
                        tgt = r.spec.kl_targets(b["actions"], b["forced"], net.cfg)
                        ref_t = r.spec.kl_terms(ref, ref_pred, tgt, b["frame_mask"], b["legal"])
                        for j, s in enumerate(ss):
                            s.ref_logits = {h: t["logits"][j, :s.length].detach()
                                            for h, t in ref_t.items()}
                    else:
                        ref_lp, _ = r.spec.logp_entropy(ref, ref_pred, b["actions"], b["legal"])
                        for j, s in enumerate(ss):
                            s.ref_lp = ref_lp[j, :s.length].detach()

    def _targets(self, diag: Diag) -> None:
        """GAE per game (reward units), PopArt + advantage statistics (one
        all-reduce each, every rank), normalised advantages, top-q filter."""
        a = self.args
        batches: Dict[str, Dict[str, torch.Tensor]] = {}
        for g in self.rl_games:
            r = self.rl[g]
            if not r.streams:
                continue
            ss = r.streams
            T = max(s.length for s in ss)
            S = r.env.n_seats
            valid = _pad([torch.ones(s.length, dtype=torch.bool) for s in ss], T)
            V = self.popart.denormalize(g, _pad([s.v_old for s in ss], T))
            adv, ret = seat_gae(_pad([s.rewards for s in ss], T), V, _pad([s.seat for s in ss], T),
                                _pad([s.done for s in ss], T) & valid, gamma=r.gamma, lam=r.lam,
                                n_seats=S, valid=valid)
            vmask = _pad([s.vmask for s in ss], T) & valid
            pmask = _pad([s.pmask for s in ss], T) & valid
            batches[g] = dict(V=V, adv=adv, ret=ret, vmask=vmask, pmask=pmask, T=T)
            e = (ret - V)[vmask].double()
            gv = ret[vmask].double()
            diag.add(g, tokens_v=int(vmask.sum()), tokens_p=int(pmask.sum()),
                     ret_sum=float(gv.sum()), ev_n=gv.numel(), ev_g=float(gv.sum()),
                     ev_g2=float(gv.square().sum()), ev_e=float(e.sum()),
                     ev_e2=float(e.square().sum()))
        mirrors = [self.popart.mirror_tensors(self.model, self.ema.shadow)] \
            if self.ema is not None else []
        self.popart.update({g: (b["ret"], b["vmask"]) for g, b in batches.items()},
                           mirrors=mirrors)
        self.advnorm.update({g: (b["adv"], b["pmask"]) for g, b in batches.items()})
        for g, b in batches.items():
            adv_n = self.advnorm.normalize(g, b["adv"], b["pmask"])
            if 0.0 < a.adv_filter_q < 1.0:
                keep = top_q_mask(adv_n, b["pmask"], q=a.adv_filter_q, per_row=True)
            else:
                keep = b["pmask"].clone()
            old_pred = self.popart.normalize(g, b["V"])
            for i, s in enumerate(self.rl[g].streams):
                n = s.length
                s.adv, s.ret = adv_n[i, :n].float(), b["ret"][i, :n].float()
                s.keep, s.old_pred = keep[i, :n], old_pred[i, :n].float()

    # ------------------------------------------------------------------
    # epochs
    # ------------------------------------------------------------------

    def _forward_chunks(self, g: str, idx: Sequence[int]) -> List[List[int]]:
        """The learner forwards over game ``g``'s streams ``idx``: longest first,
        --rows_per_forward streams each or a --frames_per_forward budget."""
        r = self.rl[g]
        return _length_chunks(idx, [s.length for s in r.streams], r.rows_per_forward,
                              r.frames_per_forward)

    def _minibatch_plan(self, it: int, epoch: int) -> Tuple[List[Dict[str, List[int]]],
                                                              torch.Tensor]:
        """Per minibatch, per game: the stream indices (a stratified random
        slice of every game's streams), and the GLOBAL per-minibatch token
        counts [M, G, 3] (policy-kept, policy, value) divided by the world
        size — one all-reduce per epoch, the same on every rank."""
        a = self.args
        M = max(1, a.minibatches)
        plan: List[Dict[str, List[int]]] = [{} for _ in range(M)]
        G = len(self.rl_games)
        counts = torch.zeros(M, max(G, 1), 3, dtype=torch.float64)
        for gi, g in enumerate(self.rl_games):
            ss = self.rl[g].streams
            if not ss:
                continue
            gen = torch.Generator().manual_seed(derive_seed(a.seed, self.rank, it, gi, 4, epoch))
            perm = torch.randperm(len(ss), generator=gen).tolist()
            for m, part in enumerate(np.array_split(np.asarray(perm, dtype=np.int64), M)):
                idx = part.tolist()
                plan[m][g] = idx
                counts[m, gi, 0] = sum(int((ss[i].keep & ss[i].pmask).sum()) for i in idx)
                counts[m, gi, 1] = sum(int(ss[i].pmask.sum()) for i in idx)
                counts[m, gi, 2] = sum(int(ss[i].vmask.sum()) for i in idx)
        if dist_active():
            dc = counts.to(collective_device())
            dist.all_reduce(dc, op=dist.ReduceOp.SUM)
            counts = dc.cpu() / self.world
        return plan, counts

    def _loss_weights(self, counts: torch.Tensor, quotas: Mapping[str, int]
                      ) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """Per term kind (kept / policy / value): game → per-token weight
        ``w_g · e_g / N_g``, with N_g the game's GLOBAL token count of that
        kind this iteration (the plan's counts summed over the minibatches)
        and e_g = min(q_g, N^value_g) its effective tokens.

        Rollouts are whole lockstep waves, so a game's realised tokens
        overshoot its quota by up to a wave; in a plain token mean that
        overshoot — not the quota — would set its pull on the shared core.
        With these weights every term gives game g the total weight
        ``w_g · e_g`` (its quota under overshoot, its realised tokens under
        a shortfall, which the scheduler's debt then tops up), spread
        uniformly over its tokens.  Rank-identical (global counts).
        Returns (weights, e)."""
        tot = counts.sum(0) * self.world            # [G, 3] global per-iteration counts
        weights: Dict[str, Dict[str, float]] = {k: {} for k in TERM_KINDS}
        eff: Dict[str, float] = {}
        for gi, g in enumerate(self.rl_games):
            e = min(float(quotas.get(g, 0)), float(tot[gi, 2]))
            eff[g] = e
            base = self.game_weight.get(g, 1.0) * e
            for j, kind in enumerate(TERM_KINDS):
                n = float(tot[gi, j])
                weights[kind][g] = base / n if n > 0 else 0.0
        return weights, eff

    def _rl_chunk_loss(self, g: str, ss: List[Stream], counts: Dict[str, Dict[str, float]],
                       diag: Diag, canary: bool,
                       weights: Optional[Mapping[str, Mapping[str, float]]] = None
                       ) -> torch.Tensor:
        """The RL loss of streams ``ss`` of game ``g``, one term of the
        minibatch's combined loss: ``counts`` = the minibatch's global
        per-game token counts / world per kind, ``weights`` = per-kind
        per-token game weights (:meth:`_loss_weights`; default the plain
        ``--game_weight`` token mean)."""
        a = self.args
        if weights is None:
            unit = {x: self.game_weight.get(x, 1.0) for x in self.rl_games}
            weights = {k: unit for k in TERM_KINDS}
        r = self.rl[g]
        net = self.model[g]
        extra = ("mu", "lp_old", "adv", "ret", "keep", "old_pred", "vmask", "pmask", "forced")
        b = build_batch(ss, self.device, extra)
        pred = learner_forward(net, r.spec, b, self.depth, self.device, a.bf16,
                               replay_routes=True)
        lp, ent = r.spec.logp_entropy(net, pred, b["actions"], b["legal"])
        pm, vm = b["pmask"], b["vmask"]
        pol = decoupled_ppo_loss(lp, b["lp_old"], b["mu"], b["adv"], pm, policy_mask=b["keep"],
                                 clip=a.clip, clip_high=a.clip_high,
                                 is_clip=a.is_clip if a.is_clip > 0 else None,
                                 dual_clip=a.dual_clip, reduction="sum")
        if r.hl_gauss:
            val = hl_gauss_value_loss(pred["value_logits"], b["ret"], vm,
                                      vmax=float(net.value_head.vmax),
                                      sigma=float(net.cfg.hl_gauss_sigma), reduction="sum")
        else:
            val = value_loss(pred["value"], self.popart.normalize(g, b["ret"]), vm,
                             old_pred=b["old_pred"] if a.value_clip else None,
                             clip=a.value_clip, kind=a.value_loss, reduction="sum")
        ent_t = entropy_loss(ent, pm, coef=self.entc[g], reduction="sum")
        wk, wp, wv = weights["kept"], weights["policy"], weights["value"]
        loss = (combine_game_losses({g: pol}, wk, counts=counts["kept"])
                + a.value_coef * combine_game_losses({g: val}, wv, counts=counts["value"])
                + combine_game_losses({g: ent_t}, wp, counts=counts["policy"]))
        n_p, n_v = float(pm.sum()), float(vm.sum())
        if not self.kl_off[g]:
            ref = self.ref[g]
            T = b["frame_mask"].shape[1]
            cached = (all(s.ref_logits is not None for s in ss) if r.spec.full_kl
                      else all(s.ref_lp is not None for s in ss))
            if not cached:
                with torch.no_grad():
                    ref_pred = learner_forward(ref, r.spec, b, self.depth, self.device, a.bf16)
            if r.spec.full_kl:
                tgt = r.spec.kl_targets(b["actions"], b["forced"], net.cfg)
                cur_t = r.spec.kl_terms(net, pred, tgt, vm, b["legal"])
                if cached:          # the iteration's reference logits (_recompute)
                    ref_t = {h: {**cur_t[h], "logits": _pad([s.ref_logits[h] for s in ss], T)}
                             for h in cur_t}
                else:
                    with torch.no_grad():
                        ref_t = r.spec.kl_terms(ref, ref_pred, tgt, vm, b["legal"])
                # sum reduction = β·(this chunk's head-normalised KL)·frames: the
                # heads' denominators (valid / conditioned / pass frames) are the
                # chunk's own, so across chunks this is a frame-weighted mean of
                # per-chunk KLs, not an exact per-head global token mean (the
                # policy / value / entropy terms are exact; measured < 1% apart)
                kl = head_terms_kl(cur_t, ref_t, beta=self.klb[g], reduction="sum")
                loss = loss + combine_game_losses({g: kl}, wv, counts=counts["value"])
                diag.add(g, kl_sum=float(kl["kl"]) * n_v, kl_n=n_v)
            else:
                if cached:
                    ref_lp = _pad([s.ref_lp for s in ss], T)
                else:
                    with torch.no_grad():
                        ref_lp, _ = r.spec.logp_entropy(ref, ref_pred, b["actions"], b["legal"])
                kl = kl_penalty(lp, ref_lp, pm, beta=self.klb[g], estimator="k3",
                                reduction="sum")
                loss = loss + combine_game_losses({g: kl}, wp, counts=counts["policy"])
                diag.add(g, kl_sum=float(kl["kl_k3"]) * n_p, kl_n=n_p)
        diag.add(g, ent_sum=float(ent_t["entropy"]) * n_p, ent_n=n_p, pol_n=n_p,
                 approx_kl_sum=float(pol["approx_kl"]) * n_p,
                 clip_frac_sum=float(pol["clip_frac"]) * n_p,
                 is_weight_sum=float(pol["is_weight"]) * n_p,
                 is_trunc_sum=float(pol["is_trunc_frac"]) * n_p,
                 vloss_sum=float(val["loss"].detach()), vloss_n=n_v)
        if canary:
            diag.add(g, canary_oldmu_sum=float(pol["abs_log_old_mu"]) * n_p,
                     canary_ratio_sum=float(pol["abs_log_ratio"]) * n_p, canary_n=n_p)
        return loss

    def _replay_loss(self, g: str, diag: Diag) -> torch.Tensor:
        a = self.args
        net = self.model[g]
        mb = prepare_batch(g, self.replay[g].next(), self.device, self.replay_tf[g],
                           cfg=net.cfg, il_legal_mask=a.il_legal_mask)
        loss, stats, pred = batch_loss(net, mb, self.device, 0.0, 0.0, a.bf16,
                                       act_steps=self.depth, z_loss_weight=a.logit_z_loss,
                                       soft_target_weight=a.soft_target_weight,
                                       engine_value_weight=a.engine_value_weight)
        diag.add(g, il_sum=stats["loss"], il_n=1)
        vw = self.replay_value_weight[g]
        if vw > 0:
            v_tgt, v_mask, _, _ = value_target(mb["tgt"], mb["frame_mask"], a.engine_value_weight)
            if g in self.popart.games:
                v_tgt = self.popart.normalize(g, v_tgt)
            vl, _ = value_losses(net, pred, v_tgt, v_mask)
            loss = loss + vw * vl
            diag.add(g, rvl_sum=float(vl.detach()))
        kw = self.replay_kl_weight[g]
        if kw > 0:
            ref = self.ref[g]
            fm = mb["frame_mask"]
            with torch.no_grad(), _autocast(self.device, a.bf16):
                ref_pred = ref(mb["obs"], legal=mb["legal"], labels=mb["labels"],
                               act_steps=self.depth, frame_mask=fm)
                ref_t = ref.action_decoder.il_head_terms(ref_pred, mb["tgt"], fm)
            kl = head_terms_kl(net.action_decoder.il_head_terms(pred, mb["tgt"], fm), ref_t,
                               beta=kw)
            loss = loss + kl["loss"]
            diag.add(g, rkl_sum=float(kl["kl"]), rkl_n=1)
        return loss

    def _replay_draws(self) -> List[Tuple[str, float]]:
        """(game, loss scale) of this minibatch's replay batches: every game
        at its weight, or K i.i.d. draws ∝ weight at scale Σw / K (the same
        expected objective)."""
        games = [g for g in self.replay if self.replay_weight[g] > 0]
        if not games:
            return []
        K = self.args.replay_games_per_mb
        if self.args.replay_split and self.world > 1:
            # every game once per minibatch over the ranks, longest first to the least
            # loaded rank (rank-identical), at world x weight: the ranks' averaged
            # gradient is the every-game objective
            cost = {g: self.replay_cost.get(g, 1.0) for g in games}
            load = [0.0] * self.world
            owner = getattr(self, "_owner", None)
            if owner and self.replay_cost:
                # the ranks' RL work of one minibatch (--rl_placement split): the replay
                # games fill the lightly loaded ranks first
                n_mb = max(1, self.args.epochs * max(1, self.args.minibatches))
                for g, r in owner.items():
                    load[r] += self.learn_cost.get(g, 0.0) / n_mb
            mine = []
            for g in sorted(games, key=lambda x: (-cost[x], x)):
                r = min(range(self.world), key=lambda i: (load[i], i))
                load[r] += cost[g]
                if r == self.rank:
                    mine.append((g, self.replay_weight[g] * self.world))
            return mine
        if K <= 0:
            return [(g, self.replay_weight[g]) for g in games]
        w = [self.replay_weight[g] for g in games]
        total = sum(w)
        return [(g, total / K) for g in random.choices(games, weights=w, k=K)]

    def _step(self) -> Tuple[bool, float]:
        a = self.args
        max_norm = a.clip_grad if a.clip_grad > 0 else float("inf")
        if self.dist_env.enabled:
            sync_gradients(self.model, self.world, warn_partial=False, bucket_mb=a.bucket_mb)
            norm = collective_clip_grad_norm_(self.params, max_norm, self.device)
        else:
            norm = torch.nn.utils.clip_grad_norm_(self.params, max_norm)
        ok = bool(torch.isfinite(norm))
        if ok:
            self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return ok, float(norm)

    def _forward_tracked(self, g: str, diag: Diag, fn: Callable[[], torch.Tensor]
                         ) -> torch.Tensor:
        before = _moe_load_vector(self.model)
        out = fn()
        if before is not None:
            diag.add_loads(g, _moe_load_vector(self.model) - before)
        return out

    def _epochs(self, it: int, diag: Diag, quotas: Mapping[str, int]) -> Dict[str, Any]:
        """--epochs × --minibatches optimizer steps.  Returns step statistics
        and ``loss_share``: per RL game, the mean over the steps of its share
        of the value term's total weight (the realised per-game RL loss
        weight; rank-identical)."""
        a = self.args
        dropped, skipped, norms = 0, 0, []
        G = self.rl_games
        share_sum = {g: 0.0 for g in G}
        n_steps = 0
        t = self._phase_t
        with moe_load_counting(self.model):
            for epoch in range(a.epochs):
                plan, counts = self._minibatch_plan(it, epoch)
                weights, _ = self._loss_weights(counts, quotas)
                for m in range(len(plan)):
                    cnt = {name: {g: float(counts[m, gi, j]) for gi, g in enumerate(G)}
                           for j, name in enumerate(TERM_KINDS)}
                    wn = {g: weights["value"][g] * cnt["value"][g] for g in G}
                    tot = sum(wn.values())
                    for g in G:
                        share_sum[g] += wn[g] / tot if tot > 0 else 0.0
                    n_steps += 1
                    canary = epoch == 0 and m == 0
                    # first minibatch: each component's shared-core gradient norm
                    probe = _GradProbe(self.core_params) if canary and self.core_params else None
                    t0 = self._clock()
                    marks = []                  # per-game (key, game, start, end) marks
                    for g in G:
                        idx = plan[m].get(g, [])
                        ss = self.rl[g].streams
                        mg = self._mark()
                        chunks = self._forward_chunks(g, idx)
                        for chunk in chunks:
                            loss = self._forward_tracked(
                                g, diag, lambda chunk=chunk, g=g, canary=canary:
                                self._rl_chunk_loss(g, [ss[i] for i in chunk], cnt, diag,
                                                    canary, weights))
                            if torch.isfinite(loss):
                                loss.backward()
                            else:
                                skipped += 1
                        if probe is not None:
                            diag.add(g, gn_rl=probe.delta_norm(), gn_rl_n=1)
                        marks.append(("learn_s", g, mg, self._mark()))
                        diag.add(g, learn_fwd=len(chunks))
                    t1 = self._clock()
                    t["rl"] += t1 - t0
                    for g, scale in self._replay_draws():
                        mg = self._mark()
                        loss = self._forward_tracked(g, diag,
                                                     lambda g=g: self._replay_loss(g, diag))
                        if torch.isfinite(loss):
                            (loss * scale).backward()
                        else:
                            skipped += 1
                        if probe is not None:
                            diag.add(g, gn_replay=probe.delta_norm(), gn_replay_n=1)
                        marks.append(("replay_s", g, mg, self._mark()))
                    del probe
                    t2 = self._clock()
                    t["replay"] += t2 - t1
                    for key, g, a, b in marks:          # complete after the sync
                        diag.add(g, **{key: self._seconds(a, b)})
                    ok, norm = self._step()
                    t["step"] += self._clock() - t2
                    dropped += int(not ok)
                    norms.append(norm)
        return {"dropped": dropped, "nonfinite": skipped,
                "gnorm": float(np.mean(norms)) if norms else float("nan"),
                "loss_share": {g: share_sum[g] / max(n_steps, 1) for g in G}}

    # ------------------------------------------------------------------
    # iteration
    # ------------------------------------------------------------------

    def _global_quotas(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        if not self.rl_games:
            return {}, {}
        q = self.sched.quotas(self.args.tokens_per_iter)
        q = {g: (0 if g in self.stopped else v) for g, v in q.items()}
        if self.args.rl_placement == "split" and self.world > 1:
            owner = self.placement(q)
            self._owner = owner
            per_rank = {g: (v if owner[g] == self.rank else 0) for g, v in q.items()}
        else:
            per_rank = {g: -(-v // self.world) for g, v in q.items()}
        return q, per_rank

    def placement(self, quotas: Mapping[str, int]) -> Dict[str, int]:
        """game → the rank that plays its whole quota (--rl_placement split):
        longest-processing-time first over :attr:`game_cost` (rank-identical), the
        quota for a game not measured yet; ties by name and rank (deterministic)."""
        cost = {g: self.game_cost.get(g, float(quotas.get(g, 0))) for g in self.rl_games}
        load = [0.0] * self.world
        owner: Dict[str, int] = {}
        for g in sorted(self.rl_games, key=lambda x: (-cost[x], x)):
            r = min(range(self.world), key=lambda i: (load[i], i))
            owner[g] = r
            load[r] += cost[g]
        return owner

    def run_iteration(self) -> Dict[str, Any]:
        a = self.args
        it = self.iteration
        seed_all(derive_seed(a.seed, self.rank, it, 5))
        for rs in self.replay.values():
            rs.restart()
        t0 = time.time()
        diag = Diag(self.games, self.n_moe)
        bias0 = [m.balance_bias.clone() for m in self.moe]
        self.model.eval()
        q_glob, q_rank = self._global_quotas()
        self._phase_t = collections.defaultdict(float)
        self._rollouts(it, q_rank, diag)
        t_roll = time.time() - t0
        t1 = self._clock()
        for g in self.rl_games:
            if self.rl[g].streams:
                self._recompute(g)
        t2 = self._clock()
        self._phase_t["recompute"] += t2 - t1
        self._targets(diag)
        self._phase_t["targets"] += self._clock() - t2
        epoch_stats = self._epochs(it, diag, q_glob)
        for m, b in zip(self.moe, bias0):
            if not torch.equal(m.balance_bias, b):
                raise RuntimeError("MoE balance bias moved inside the iteration")
        diag.reduce()
        for g in self.rl_games:             # the next --rl_placement split assignment
            c = diag.v[g]["rollout_s"] + diag.v[g]["learn_s"]
            if c > 0:
                self.game_cost[g] = c
        for g in self.rl_games:
            if diag.v[g]["learn_s"] > 0:
                self.learn_cost[g] = diag.v[g]["learn_s"]
        for g in self.replay:               # the next --replay_split assignment
            if diag.v[g]["il_n"] > 0 and diag.v[g]["replay_s"] > 0:
                self.replay_cost[g] = diag.v[g]["replay_s"] / diag.v[g]["il_n"]
        # ---- between iterations ----------------------------------------
        if self.dist_env.enabled:
            allreduce_moe_loads(self.model)
        rel = update_moe_balance(self.model, a.moe_balance_rate)
        if self.ema is not None:
            self.ema.update(self.model)
        ent = {g: diag.mean(g, "ent_sum", "ent_n") for g in self.rl_games}
        kl = {g: diag.mean(g, "kl_sum", "kl_n") for g in self.rl_games}
        upd_e = {g: v for g, v in ent.items() if g in self.ent_target and math.isfinite(v)}
        if upd_e:
            self.entc.update(upd_e)
        upd_k = {g: v for g, v in kl.items() if not self.kl_off[g] and math.isfinite(v)}
        if a.kl_adapt and upd_k:
            self.klb.update(upd_k)
        if self.rl_games:
            # the tokens that carried loss weight (_loss_weights): the quota under
            # overshoot, so a wave's overshoot is not booked as a lead that
            # starves the game next iteration
            self.sched.record({g: min(int(diag.v[g]["tokens_v"]), int(q_glob.get(g, 0)))
                               for g in self.rl_games})
        if self.bandit is not None:
            sig = {g: diag.mean(g, "score_sum", "score_n") for g in self.rl_games}
            sig = {g: v for g, v in sig.items() if math.isfinite(v)}
            if sig:
                self.bandit.update(sig)
        for g in self.replay:
            m_il = diag.mean(g, "il_sum", "il_n")
            if g not in self.il_baseline and math.isfinite(m_il):
                self.il_baseline[g] = m_il
        self.iteration += 1
        if (a.snapshot_every > 0 and self.iteration % a.snapshot_every == 0
                and any(r.pool_policy is not None for r in self.rl.values())):
            self._add_snapshot(tag="iter")
        rec = self._record(diag, epoch_stats, q_glob, rel, time.time() - t0, t_roll)
        self._alarms(rec)
        if getattr(self, "_owner", None):
            rec["placement"] = dict(self._owner)
        self.history.append(rec)
        if self.rank0:
            with open(os.path.join(a.save_dir, "metrics.jsonl"), "a") as f:
                f.write(json.dumps(rec) + "\n")
        if self.iteration % a.log_every == 0:
            self._log(rec)
        return rec

    def _eval_vs_init(self, g: str, r: GameRL, it: int, gi: int) -> WaveStats:
        """The learner against the frozen --init network on the game's env (no mirror
        episodes, seats drawn per episode), whole waves until --eval_episodes; the
        streams are dropped.  Self-play reward can rise from an exploit of the learner's
        own play; a policy that got worse against where it started cannot hide it here."""
        a = self.args
        ref = NetworkPolicy(self.ref[g], act_steps=self.depth, device=self.device)
        total, wave = WaveStats(), 0
        while total.episodes < a.eval_episodes:
            league = League(r.env, {"init": ref}, mirror_prob=0.0,
                            seed=derive_seed(a.seed, self.rank, it, gi, 4, wave))
            _, st = collect_wave(r.env, league, r.actor,
                                 seed=derive_seed(a.seed, self.rank, it, gi, 5, wave),
                                 max_steps=a.max_wave_steps, timeout_reward=r.timeout_reward,
                                 timeout_draw=g in TIMEOUT_IS_DRAW)
            for k in ("episodes", "score_sum", "score_n"):
                setattr(total, k, getattr(total, k) + getattr(st, k))
            wave += 1
        return total

    def _alarms(self, rec: Dict[str, Any]) -> None:
        """Stop a game's RL when its record looks like reward hacking (alarm_reasons)
        --alarm_patience iterations in a row.  The record is all-reduced, so every
        rank takes the same decision."""
        a = self.args
        for g in self.rl_games:
            row = rec["games"].get(g, {})
            if g in self.stopped or row.get("episodes", 0) < a.alarm_min_episodes:
                continue
            if g not in self.alarm_base_len and math.isfinite(row.get("ep_len", float("nan"))):
                self.alarm_base_len[g] = float(row["ep_len"])
                continue
            why = alarm_reasons(row, self.alarm_base_len.get(g), self.kl_budget[g],
                                timeout=a.alarm_timeout, len_x=a.alarm_len_x, kl_x=a.alarm_kl_x)
            self.alarm_streak[g] = self.alarm_streak.get(g, 0) + 1 if why else 0
            # the referee (every --eval_every): its own streak over evaluations
            vi = row.get("vs_init", float("nan"))
            if a.alarm_vs_init > 0 and math.isfinite(vi) \
                    and row.get("vs_init_n", 0) >= a.alarm_min_episodes:
                low = vi < a.alarm_vs_init
                self.vs_init_streak[g] = self.vs_init_streak.get(g, 0) + 1 if low else 0
                if low:
                    why.append(f"score vs the --init network {vi:.3f} < {a.alarm_vs_init}")
            if why:
                row["alarm"] = "; ".join(why)
            if why and max(self.alarm_streak[g], self.vs_init_streak.get(g, 0)) >= a.alarm_patience:
                self.stopped[g] = f"iteration {self.iteration}: " + "; ".join(why)
                self.log(f"[alarm] {g}: RL stopped ({self.stopped[g]}) — its quota is 0 from "
                         f"now on; inspect its replays / fixed-opponent results before resuming "
                         f"it from a checkpoint before iteration {self.iteration}")
        if self.stopped:
            rec["stopped"] = dict(self.stopped)

    def _mark(self):
        """A per-game timing mark without a host sync: a recorded CUDA event (read
        with :meth:`_seconds` after the phase's :meth:`_clock` sync), or the wall
        clock on other devices.  Syncing per game would stall the host between
        replay draws, serialising their data loading with the device's work."""
        if self.device.type == "cuda":
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e
        return time.time()

    @staticmethod
    def _seconds(a, b) -> float:
        """Seconds between two :meth:`_mark` marks (CUDA events: after a sync)."""
        if isinstance(a, float):
            return b - a
        return a.elapsed_time(b) / 1000.0

    def _clock(self) -> float:
        """Wall time after the device's queued work (phase timings: the CUDA
        kernels of a phase are asynchronous; a few syncs per minibatch)."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.time()

    def _record(self, diag: Diag, ep: Dict[str, Any], quotas: Dict[str, int],
                rel: Optional[torch.Tensor], dt: float, t_roll: float) -> Dict[str, Any]:
        rec: Dict[str, Any] = {"iteration": self.iteration, "time": dt, "rollout_time": t_roll,
                               "phase_time": {k: round(v, 2) for k, v in
                                              getattr(self, "_phase_t", {}).items()},
                               "gnorm": ep["gnorm"], "dropped": ep["dropped"],
                               "nonfinite": ep["nonfinite"],
                               "moe_load_cv": float(rel.std(-1).mean()) if rel is not None
                               else float("nan"), "games": {}}
        for g in self.games:
            d = diag.v[g]
            row: Dict[str, float] = {}
            if g in self.rl:
                row.update(
                    quota=quotas.get(g, 0), tokens=d["tokens_v"],
                    loss_share=ep["loss_share"].get(g, 0.0), episodes=d["episodes"],
                    ep_len=diag.mean(g, "ep_len_sum", "episodes"),
                    timeout_rate=diag.mean(g, "timeouts", "episodes"),
                    vs_init=diag.mean(g, "vs_init_sum", "vs_init_n"), vs_init_n=d["vs_init_n"],
                    shaping=float(getattr(self.rl[g].env, "shaping", 0.0)),
                    score=diag.mean(g, "score_sum", "score_n"),
                    ret=diag.mean(g, "ret_sum", "tokens_v"),
                    entropy=diag.mean(g, "ent_sum", "ent_n"),
                    kl=diag.mean(g, "kl_sum", "kl_n"),
                    approx_kl=diag.mean(g, "approx_kl_sum", "pol_n"),
                    clip_frac=diag.mean(g, "clip_frac_sum", "pol_n"),
                    is_weight=diag.mean(g, "is_weight_sum", "pol_n"),
                    is_trunc=diag.mean(g, "is_trunc_sum", "pol_n"),
                    value_loss=diag.mean(g, "vloss_sum", "vloss_n"),
                    ev=diag.explained_variance(g),
                    canary_log_old_mu=diag.mean(g, "canary_oldmu_sum", "canary_n"),
                    canary_log_ratio=diag.mean(g, "canary_ratio_sum", "canary_n"),
                    ent_coef=self.entc[g], kl_beta=self.klb[g],
                    popart_mu=self.popart.mean(g), popart_sigma=self.popart.std(g),
                    adv_std=self.advnorm.std(g))
            if g in self.rl:
                row["core_gnorm_rl"] = diag.mean(g, "gn_rl", "gn_rl_n")
                # per-rank rollout / learner seconds, env steps and forwards (summed
                # over ranks)
                row.update(rollout_s=d["rollout_s"] / self.world,
                           rollout_steps=d["rollout_steps"] / self.world,
                           learn_s=d["learn_s"] / self.world,
                           learn_fwd=d["learn_fwd"] / self.world)
            if g in self.replay:
                il = diag.mean(g, "il_sum", "il_n")
                row.update(il_loss=il, il_drift=il - self.il_baseline.get(g, il))
                if d["gn_replay_n"] > 0:
                    row["core_gnorm_replay"] = diag.mean(g, "gn_replay", "gn_replay_n")
                row["replay_s"] = d["replay_s"] / self.world
                if self.replay_kl_weight[g] > 0:
                    row["replay_kl"] = diag.mean(g, "rkl_sum", "rkl_n")
            if self.n_moe[0]:
                row["load_entropy"] = diag.load_entropy(g)
            if row:
                rec["games"][g] = row
        return rec

    def _log(self, rec: Dict[str, Any]) -> None:
        def f(x) -> str:
            return f"{x:.4g}" if isinstance(x, float) else str(x)
        self.log(f"[it {rec['iteration']}/{self.args.iterations}] gnorm={f(rec['gnorm'])} "
                 f"dropped={rec['dropped']} nonfinite={rec['nonfinite']} "
                 f"moe_load_cv={f(rec['moe_load_cv'])} "
                 f"({rec['time']:.1f}s, rollouts {rec['rollout_time']:.1f}s"
                 + "".join(f", {k} {v:.1f}s" for k, v in rec.get("phase_time", {}).items())
                 + ")")
        if rec.get("placement"):
            by_rank: Dict[int, List[str]] = {}
            for g, r in rec["placement"].items():
                by_rank.setdefault(r, []).append(g)
            self.log("  placement: " + "  ".join(f"rank{r}={','.join(sorted(gs))}"
                                                 for r, gs in sorted(by_rank.items())))
        for g, row in rec["games"].items():
            self.log(f"  {g}: " + " ".join(f"{k}={f(v)}" for k, v in row.items()))

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------

    def _rl_state(self) -> Dict[str, Any]:
        return {"iteration": self.iteration, "popart": self.popart.state_dict(),
                "advnorm": self.advnorm.state_dict(), "entc": self.entc.state_dict(),
                "klb": self.klb.state_dict(), "sched": self.sched.state_dict(),
                "pool": self.pool.state_dict(), "il_baseline": dict(self.il_baseline),
                "game_cost": dict(self.game_cost), "replay_cost": dict(self.replay_cost),
                "learn_cost": dict(self.learn_cost),
                "alarm_base_len": dict(self.alarm_base_len),
                "alarm_streak": dict(self.alarm_streak), "stopped": dict(self.stopped),
                "vs_init_streak": dict(self.vs_init_streak),
                "rl_games": list(self.rl_games), "replay_games": list(self.replay),
                "init": self.args.init, "init_hash": self.init_hash}

    def _env_states(self) -> Optional[List[Dict[str, Any]]]:
        """Every rank's ``{game: {"num_envs", "state": env.state_dict()}}``
        (gathered on every rank: one collective, posted by every rank's
        :meth:`save`).  Most games restart their episodes from the wave seed,
        but a lux wave CONTINUES the envs' 5-match games (env/vec_lux.py), so
        their state is part of what a resume must restore."""
        mine = {g: {"num_envs": r.env.num_envs, "state": r.env.state_dict()}
                for g, r in self.rl.items() if callable(getattr(r.env, "state_dict", None))}
        if not self.dist_env.enabled:
            return [mine]
        out: List[Optional[Dict[str, Any]]] = [None] * self.world
        dist.all_gather_object(out, mine)
        return out if self.rank0 else None

    def _load_env_states(self, states: Optional[Sequence[Mapping[str, Any]]],
                         saved_world: int) -> None:
        """Restore this rank's env states from a checkpoint (same world size
        and env counts only: otherwise the games restart, with a note)."""
        if not states:
            return
        if saved_world != self.world or len(states) != self.world:
            self.log(f"[resume] note: env states saved with world={saved_world}, running with "
                     f"{self.world}: the simulator games restart")
            return
        for g, r in self.rl.items():
            ent = states[self.rank].get(g)
            if ent is None or not callable(getattr(r.env, "load_state_dict", None)):
                continue
            if int(ent["num_envs"]) != r.env.num_envs:
                self.log(f"[resume] note: {g} saved with {ent['num_envs']} envs, running with "
                         f"{r.env.num_envs}: its games restart")
                continue
            r.env.load_state_dict(ent["state"])

    def save(self, name: str) -> Optional[str]:
        """rank 0 writes ``name`` atomically (every rank must call: barrier).

        Every train_multi_game field is in reward units — ``model_state``
        (EMA), ``train_model_state`` (live weights) and ``ema_state`` (the
        shadow) all have the PopArt statistics folded into the value heads —
        so eval_multi_game (``--raw`` too), game_checkpoint /
        export_game_checkpoint and a train_multi_game / train_il resume read
        the right value scale.  The normalised value-head tensors the RL
        state needs are kept in ``rl_state["raw_heads"]`` (a few small
        tensors) and restored by :meth:`_load_resume`, so a resume stays
        bit-exact.  After ``rl_latest.pt`` the pool deletes the snapshot
        files evicted since the last one (no checkpoint a resume starts
        from refers to them any more).  ``rl_state["envs"]`` holds every
        rank's simulator state (:meth:`_env_states`)."""
        path = None
        envs = self._env_states()
        if self.rank0:
            live = self.model.state_dict()
            shadow = self.ema.state_dict() if self.ema is not None else None
            ema_state = self.ema.ema_model_state(self.model) if self.ema is not None else live
            raw_heads: Dict[str, Dict[str, torch.Tensor]] = {"train": {}, "ema": {}}
            if self.rl_games:
                fold = self.popart.denormalized_state_dict
                exported = fold(self.model, ema_state)
                live_f = fold(self.model, live)
                raw_heads["train"] = {k: live[k] for k, v in live_f.items() if v is not live[k]}
                if shadow is not None:
                    shadow_f = fold(self.model, shadow)
                    raw_heads["ema"] = {k: shadow[k] for k, v in shadow_f.items()
                                        if v is not shadow[k]}
                    shadow = shadow_f
            else:
                exported, live_f = ema_state, live
            g0 = self.games[0]
            pre = f"nets.{g0}."
            core = {k[len(pre):]: v for k, v in exported.items()
                    if k.startswith(pre) and is_core_key(k[len(pre):])}
            ck = {
                "games": self.games,
                "configs": {g: self.model[g].cfg.__dict__ for g in self.games},
                "core_config": core_config_overrides(self.model[g0].cfg),
                "core_state": core,
                "model_state": exported,
                "train_model_state": live_f,
                "mixture": self.sched.shares() if self.rl_games else {},
                "args": vars(self.args),
                "world_size": self.world,
                "step": self.iteration,
                "stage": "rl_joint",
                "optimizer_state": self.optimizer.state_dict(),
                "ema_state": shadow,
                "rl_state": {**self._rl_state(), "raw_heads": raw_heads, "envs": envs},
            }
            path = os.path.join(self.args.save_dir, name)
            torch.save(ck, f"{path}.tmp")
            os.replace(f"{path}.tmp", path)
            if name == CKPT_LATEST:
                self.pool.flush_deletes()
        self.dist_env.barrier()
        return path

    def _load_resume(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("stage") != "rl_joint" or "rl_state" not in ck:
            raise SystemExit(f"{path} is not a train_rl_joint checkpoint")
        st = ck["rl_state"]
        if st["rl_games"] != self.rl_games or ck["games"] != self.games:
            raise SystemExit(f"{path}: games {ck['games']} / rl {st['rl_games']} differ from "
                             f"this run's {self.games} / {self.rl_games}")
        saved_hash, saved_init = st.get("init_hash"), st.get("init")
        if saved_hash is not None and saved_hash != self.init_hash:
            raise SystemExit(f"{path}: trained against the frozen reference {saved_init} "
                             f"(fingerprint {saved_hash[:12]}), but --init {self.args.init} has "
                             f"different weights ({self.init_hash[:12]}) — the KL anchor and "
                             f"--kl_budget would silently change; pass the original --init")
        if saved_init != self.args.init:
            self.log(f"[resume] note: --init {self.args.init} != the checkpoint's {saved_init}"
                     + (" (same weights)" if saved_hash is not None else
                        " (no fingerprint in this checkpoint: make sure it is the same file)"))
        if "generals" in ck["games"]:
            # the observation is part of the run (--env_kw generals:memory_vision,
            # adopted into the config): a resume must not switch it mid-run.
            # Checkpoints written before cfg.memory_vision carry it in their args.
            saved_mv = bool(checkpoint_config_dict(ck, "generals").get("memory_vision", False))
            mv = bool(getattr(self.model["generals"].cfg, "memory_vision", False))
            if saved_mv != mv:
                raise SystemExit(f"{path}: its generals network trained with memory_vision="
                                 f"{saved_mv} (the sighted-mountain memory), this run's is {mv} "
                                 f"— resume with the run's own arguments (--env_kw "
                                 f"generals:memory_vision={saved_mv})")
        # the train_multi_game fields are PopArt-folded; the normalised value
        # heads the RL state assumes are in rl_state["raw_heads"]
        raw = st.get("raw_heads") or {}
        self.model.load_state_dict({**ck["train_model_state"], **raw.get("train", {})})
        if self.ema is not None and ck.get("ema_state") is not None:
            self.ema.load_state_dict({**ck["ema_state"], **raw.get("ema", {})})
        self.optimizer.load_state_dict(ck["optimizer_state"])
        mirrors = [self.popart.mirror_tensors(self.model, self.ema.shadow)] \
            if self.ema is not None else []
        self.popart.load_state_dict(st["popart"], mirrors=mirrors)
        self.advnorm.load_state_dict(st["advnorm"])
        self.entc.load_state_dict(st["entc"])
        self.klb.load_state_dict(st["klb"])
        self.sched.load_state_dict(st["sched"])
        self.pool.load_state_dict(st["pool"])
        self.il_baseline = dict(st.get("il_baseline", {}))
        self.game_cost = dict(st.get("game_cost", {}))
        self.replay_cost = dict(st.get("replay_cost", {}))
        self.learn_cost = dict(st.get("learn_cost", {}))
        self.alarm_base_len = dict(st.get("alarm_base_len", {}))
        self.alarm_streak = dict(st.get("alarm_streak", {}))
        self.vs_init_streak = dict(st.get("vs_init_streak", {}))
        self.stopped = dict(st.get("stopped", {}))
        for g, why in self.stopped.items():
            self.log(f"[resume] {g}: RL stays stopped ({why})")
        self._load_env_states(st.get("envs"), int(ck.get("world_size", 1)))
        self.iteration = int(st["iteration"])
        self.log(f"[resume] {path}: iteration {self.iteration} (saved with world="
                 f"{ck.get('world_size', 1)}, pool={len(self.pool)})")

    def _truncate_metrics(self) -> None:
        """Drop the metrics.jsonl records of iterations past the resumed
        checkpoint (run after it by the interrupted job; they are re-run
        now), so a requeued run leaves one record per iteration."""
        path = os.path.join(self.args.save_dir, "metrics.jsonl")
        if not os.path.exists(path):
            return
        keep = []
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:          # a record cut short by the crash
                    continue
                if int(rec.get("iteration", 0)) <= self.iteration:
                    keep.append(line if line.endswith("\n") else line + "\n")
        with open(f"{path}.tmp", "w") as f:
            f.writelines(keep)
        os.replace(f"{path}.tmp", path)

    def _want_checkpoint(self, last_save: float) -> bool:
        a = self.args
        due = (self.iteration >= a.iterations
               or (a.ckpt_every and self.iteration % a.ckpt_every == 0)
               or (a.ckpt_minutes > 0 and time.time() - last_save >= 60.0 * a.ckpt_minutes))
        if not self.dist_env.enabled:
            return bool(due)
        flag = torch.tensor([1.0 if due else 0.0], device=collective_device())
        dist.broadcast(flag, src=0)          # rank 0's clock decides for everyone
        return bool(flag.item() > 0)

    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        a = self.args
        last_save = time.time()
        while self.iteration < a.iterations:
            self.run_iteration()
            if self.rl_games and len(self.stopped) == len(self.rl_games):
                self.log("[alarm] every RL game is stopped: ending the run")
                break
            if self._want_checkpoint(last_save):
                self.save(CKPT_LATEST)
                last_save = time.time()
            if a.save_every and self.iteration % a.save_every == 0:
                self.save(f"rl_iter{self.iteration}.pt")
        final = self.save(CKPT_FINAL)
        if final:
            self.log(f"[done] saved {final}")
        return {"model": self.model, "ema": self.ema, "optimizer": self.optimizer,
                "history": self.history, "trainer": self}


def main(argv=None) -> Dict[str, Any]:
    """Train; returns the live objects (``model``, ``ema``, ``optimizer``,
    ``history`` of per-iteration records, ``trainer``) for in-process callers."""
    trainer = JointRLTrainer(parse_args(argv))
    out = trainer.train()
    trainer.dist_env.barrier()
    trainer.dist_env.destroy()
    return out


if __name__ == "__main__":
    main()
