"""
Reinforcement Learning training script for the generals.io bot.

Implements distributed PPO self-play with:
  - Dense reward (heuristic delta + event rewards + terminal ±5.0)
  - Hard legal masking of sources and directions in rollout AND replay
    (--rl_legal_mask, default on; 0 restores the legacy learn-legality-
    through-reward mode)
  - IL auxiliary regularisation (AlphaStar-style)
  - KL regularisation against frozen reference policy
  - Opponent pool (league training)
  - Encoder freeze / unfreeze schedule

Usage:
    torchrun --nproc_per_node=8 --nnodes=4 ... train_rl.py \\
        --il_checkpoint checkpoints/finetune/checkpoint_epoch67.pt \\
        --il_data_dir data-1k/training/Human.exe \\
        --iterations 10000
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_processing.dataset_seq import SeqReplayDataset, collate_variable_length_sequences
from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.losses import ImitationLoss, RLLoss
from model.network import GeneralsNetwork
from model.rl_utils import (
    RolloutBuffer,
    compute_policy_kl_heads,
    freeze_encoder,
    is_legacy_state_dict,
    load_model_state,
    optim_param_groups,
    optim_state_mismatch,
    sync_gradients,
    unfreeze_all,
)


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RL training for generals.io bot")
    # Checkpoint & data
    p.add_argument("--il_checkpoint", type=str, default=None,
                   help="Path to IL-pretrained checkpoint (.pt)")
    p.add_argument("--il_data_dir", type=str, nargs="+", default=None,
                   help="Directory(s) with .json IL training data (for joint training)")
    p.add_argument("--output_dir", type=str, default="checkpoints_rl")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from RL checkpoint")

    # RL hyper-parameters
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--rollout_steps", type=int, default=1000,
                   help="Max steps per self-play episode")
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--accum_episodes", type=int, default=0,
                   help="Optimizer step every N local episode buffers "
                        "within an epoch instead of one step per epoch "
                        "(0 = legacy single step). More Adam steps per "
                        "iteration at smaller effective batch.")
    p.add_argument("--target_kl", type=float, default=0.0,
                   help="Skip remaining PPO epochs once the all-reduced "
                        "approx_kl exceeds this (0 = off). Guards the "
                        "second epoch from destructively large updates.")
    p.add_argument("--clip_range", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--act_steps", type=int, default=1,
                   help="Fixed TRM ACT depth used for BOTH rollout and PPO update "
                        "(the two must match for the PPO ratio to be valid)")

    # Loss weights
    p.add_argument("--value_weight", type=float, default=0.5)
    p.add_argument("--entropy_weight", type=float, default=0.01)
    p.add_argument("--entropy_weight_decay", type=float, default=1.0,
                   help="Per-iteration decay for entropy weight (1.0 = no decay)")
    p.add_argument("--entropy_weight_min", type=float, default=0.008,
                   help="Minimum entropy weight (floor for decay)")
    p.add_argument("--il_weight", type=float, default=0.5,
                   help="Weight of IL auxiliary loss")
    p.add_argument("--il_decay", type=float, default=0.9999,
                   help="Per-iteration decay for il_weight")
    p.add_argument("--kl_weight", type=float, default=0.01,
                   help="KL regularisation against reference policy")
    p.add_argument("--kl_weight_pow", type=float, default=0.0,
                   help="Anchor-weight decay exponent (AverageJoe's magnet "
                        "schedule): effective weight = kl_weight / "
                        "(1 + t)^pow with t counted from THIS run's start "
                        "iteration. 0 = constant (legacy). The anchor is "
                        "strong while the sparse-reward critic finds its "
                        "feet and fades as self-play takes over.")
    p.add_argument("--kl_weight_min", type=float, default=0.0,
                   help="Floor for the decayed anchor weight (0 = none)")
    p.add_argument("--il_weight_min", type=float, default=0.0,
                   help="Floor for the decayed IL auxiliary weight — keeps "
                        "a permanent human-data pull (opening quality "
                        "erodes to a measurably worse equilibrium when the "
                        "IL term decays to zero; land@100 37 vs anchor "
                        "45.6 measured at phase-5 end). 0 = legacy decay "
                        "to nothing.")
    p.add_argument("--entropy_ceiling", type=float, default=0.0,
                   help="If the previous update's mean policy entropy "
                        "exceeds this, the entropy bonus is switched off "
                        "for the current iteration (0 = no ceiling). "
                        "Breaks the despair-exploration feedback loop "
                        "observed in the phase-5 collapse (losing streak → "
                        "entropy 3.0 → noise-dominated updates).")
    p.add_argument("--entropy_penalty", type=float, default=0.0,
                   help="When the --entropy_ceiling gate fires, use this as "
                        "a NEGATIVE entropy weight instead of 0, actively "
                        "pushing entropy back down. The phase-6 collapse "
                        "showed a zeroing gate only stops adding fuel: "
                        "entropy kept self-amplifying 2.3 -> 3.5 through "
                        "long-game/large-territory feedback with the gate "
                        "firing every iteration.")
    p.add_argument("--trunc_material", type=float, default=0.35,
                   help="Material scale of the truncation soft terminal.")
    p.add_argument("--trunc_penalty", type=float, default=0.0,
                   help="Flat penalty charged to BOTH players when a game "
                        "hits the turn limit. Anti-stalling: the phase-6 "
                        "collapse drifted into mutual clock-farming "
                        "(ep_len 400 -> 1400 vs cap 1536) because a "
                        "truncated lead scored nearly as well as winning.")
    p.add_argument("--adv_filter_frac", type=float, default=0.0,
                   help="Train the policy only on the top-FRAC samples of "
                        "each episode by |advantage| (AverageJoe's top-25%% "
                        "filter at 0.25). The critic still trains on every "
                        "sample. 0 = no filtering.")
    p.add_argument("--sparse_reward", type=int, default=0,
                   help="1 = pure outcome reward: city and dense shaping "
                        "scales are pinned to 0 (the ±1 terminal and the "
                        "±0.35 material-scored soft terminal for "
                        "truncations remain). Overrides the decay "
                        "schedules and any scales restored from a resume "
                        "checkpoint.")
    p.add_argument("--hl_gauss_value", type=int, default=0,
                   help="1 = HL-Gauss distributional critic (128 bins on "
                        "[-1.5, 1.5], CE against Gaussian-smeared "
                        "targets); scalar consumers read the expectation. "
                        "Enabling on a scalar-critic checkpoint "
                        "reinitialises the value head fresh (the rest of "
                        "the net is untouched). Sticky via checkpoint "
                        "config. 0 = legacy Huber scalar critic.")
    p.add_argument("--kl_ref", type=str, default=None,
                   help="Checkpoint whose weights define the frozen KL "
                        "reference policy. Default: this run's starting "
                        "weights — on --resume that is the RESUMED policy, "
                        "so pass the IL checkpoint here to keep the human "
                        "anchor across restarts.")
    p.add_argument("--invalidity_weight", type=float, default=0.1,
                   help="Weight for source/direction invalidity auxiliary "
                        "loss (ignored when --rl_legal_mask is on: masked "
                        "policies place no mass on invalid actions)")
    p.add_argument("--derive_army_channels", type=int, default=0,
                   help="1 = derive 3 LINEAR army channels (own/enemy/"
                        "neutral, army/50) inside the encoder — exact army "
                        "comparisons are nearly invisible in the single "
                        "log-compressed channel. Function-preserving "
                        "zero-pad migration; sticky via checkpoint config.")
    p.add_argument("--turn_globals", type=int, default=0,
                   help="1 = append production-cycle features "
                        "((turn%%50)/50, turn%%2, turn/500) to the globals "
                        "(global_features -> 7). The temporal attention has "
                        "no positional encoding, so the policy otherwise "
                        "cannot see the production clock. Sticky via "
                        "checkpoint config.")
    p.add_argument("--mirror_prob", type=float, default=0.5,
                   help="Probability an episode is MIRROR self-play (live "
                        "model vs itself, BOTH seats' trajectories train — "
                        "the opponent forward pass is computed every step "
                        "anyway). Mirror games are excluded from the pool-"
                        "gate winrate (they are ~0.5 by construction). "
                        "Non-mirror games draw a fresh pool opponent per "
                        "episode.")
    p.add_argument("--sparring_prob", type=float, default=0.0,
                   help="Probability an episode is played against the "
                        "frozen --sparring_ckpt opponent instead of the "
                        "mirror/pool draw. Pure mirror self-play cannot "
                        "punish style drift (rush vs rush is symmetric): a "
                        "frozen sparring partner re-introduces the counter-"
                        "style into the training distribution WITHOUT the "
                        "pool/gate machinery. Sparring games use a random "
                        "learner seat, collect one trajectory, and show up "
                        "in the w/l/d counters (mirror games don't).")
    p.add_argument("--sparring_ckpt", type=str, default=None,
                   help="Checkpoint for the frozen sparring partner. Built "
                        "from its OWN saved config, so a scalar-critic "
                        "phase-5 EMA can spar a HL-Gauss phase-6 learner "
                        "(its value head is never read).")
    p.add_argument("--randomize_seat", type=int, default=1,
                   help="1 = sample the learner's seat per episode (the env "
                        "move-priority tiebreak favors P0; a learner pinned "
                        "to P0 trains on a rules advantage it won't have on "
                        "the real server); 0 = legacy always-P0")
    p.add_argument("--rl_legal_mask", type=int, default=1,
                   help="1 = hard-mask illegal sources and directions in "
                        "BOTH rollout and PPO replay (entropy, log-probs and "
                        "the KL anchor all see the masked distributions); "
                        "0 = legacy unmasked RL where legality was learned "
                        "through the -0.05 penalty")
    p.add_argument("--capture_aux_weight", type=float, default=0.05,
                   help="Weight of the auxiliary capture-feasibility BCE "
                        "head (0 = disabled). Supervises 'would a full move "
                        "from tile X in direction d capture the target' so "
                        "the encoder learns exact army comparisons.")
    p.add_argument("--city_budget", type=float, default=0.8,
                   help="Total neutral-city reward budget per map; the "
                        "per-city payout unit is budget/n_cities (capped "
                        "at 0.25), keeping aggregate city shaping below "
                        "the terminal win/loss reward")
    p.add_argument("--city_reward_decay", type=float, default=0.9999,
                   help="Per-iteration decay for city capture reward scale")
    p.add_argument("--dense_reward_decay", type=float, default=0.9999,
                   help="Per-iteration decay for dense heuristic reward scale")
    p.add_argument("--reset_reward_decay", action="store_true",
                   help="On --resume, restart the city/dense reward decay "
                        "schedules at 1.0 instead of decay^done_iters. Use "
                        "when a phase boundary changes the reward structure "
                        "and the new shaping should not enter pre-annealed.")

    # Encoder freeze schedule
    p.add_argument("--freeze_encoder_iters", type=int, default=1000,
                   help="Freeze encoder for the first N iterations (0 = never freeze)")

    # Multi-episode collection
    p.add_argument("--episodes_per_gpu", type=int, default=1,
                   help="Episodes to collect per GPU before PPO update (gradient accumulation)")

    # Opponent pool
    p.add_argument("--pool_size", type=int, default=10,
                   help="Max opponents in pool (excluding IL baseline)")
    p.add_argument("--pool_update_interval", type=int, default=100,
                   help="Check pool update eligibility every N iterations")
    p.add_argument("--pool_update_winrate", type=float, default=0.0,
                   help="Min winrate over recent iterations to update pool (0 = always update)")
    p.add_argument("--pool_update_window", type=int, default=50,
                   help="Number of recent iterations to compute winrate for pool update")

    # Map pool (use real replay maps instead of random)
    p.add_argument("--replay_dir", type=str, default=None,
                   help="Directory with raw replay JSON files")
    p.add_argument("--filtered_file", type=str, default=None,
                   help="Path to filtered replay list JSON")

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_decay", type=float, default=0.9997,
                   help="Per-iteration multiplicative LR decay")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # Model (used when no checkpoint is provided)
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--n_heads", type=int, default=12)
    p.add_argument("--trm_H_cycles", type=int, default=3)
    p.add_argument("--trm_L_cycles", type=int, default=6)
    p.add_argument("--trm_block_layers", type=int, default=3)
    p.add_argument("--temporal_mixer", type=str, default="attention",
                   choices=["attention", "recurrent"],
                   help="Temporal mixer for from-scratch models (checkpoints "
                        "carry their own setting in the saved config)")

    # EMA policy weights — smoother snapshots for evaluation, the opponent
    # pool and deployment (AverageJoe's #1 ladder agent IS its EMA copy).
    # Never fed back into training.
    p.add_argument("--ema_decay", type=float, default=0.998,
                   help="Per-iteration EMA decay for a CPU shadow copy of "
                        "the weights; used for eval, pool snapshots and the "
                        "exported bot (0 = disabled). 0.998 ~ 500-iter "
                        "time constant.")

    # Fixed-opponent evaluation (progress signal independent of shaping
    # rewards and of the moving opponent pool)
    p.add_argument("--eval_interval", type=int, default=50,
                   help="Every N iterations play eval games vs the frozen "
                        "KL reference policy (0 = never; requires --kl_weight"
                        " > 0 so the reference model exists)")
    p.add_argument("--eval_games", type=int, default=4,
                   help="Eval SEED PAIRS per GPU per round — each seed is "
                        "played twice, greedy, with seats swapped")

    # Logging
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--save_replay_interval", type=int, default=50,
                   help="Save a game replay every N iterations (0 = never)")
    p.add_argument("--device", type=str, default="auto")

    args = p.parse_args()
    # An empty path (e.g. a shell glob that matched nothing) is falsy at
    # every `if args.resume:` check and would silently start from scratch.
    for flag in ("resume", "il_checkpoint", "kl_ref"):
        val = getattr(args, flag)
        if val is not None and not val.strip():
            p.error(f"--{flag} given an empty path")
    return args


# ======================================================================
# Opponent pool
# ======================================================================

class OpponentPool:
    """Maintains a pool of past model checkpoints for league training.

    Index 0 is always the IL baseline (never removed).
    """

    @staticmethod
    def _to_cpu(state_dict: dict) -> dict:
        """Detached CPU copy — keeps up to max_size+1 models off the GPU."""
        return {k: v.detach().to("cpu", copy=True) for k, v in state_dict.items()}

    def __init__(self, initial_state: dict, max_size: int = 10):
        self.pool: List[dict] = [self._to_cpu(initial_state)]
        self.max_size = max_size

    def add(self, state_dict: dict) -> None:
        self.pool.append(self._to_cpu(state_dict))
        if len(self.pool) > self.max_size + 1:  # +1 for IL baseline
            self.pool.pop(1)  # keep index 0 (IL baseline)

    def sample(self) -> dict:
        """50% from latest 3, 50% random from pool."""
        if len(self.pool) > 1 and random.random() < 0.5:
            return random.choice(self.pool)
        # Pick from the latest 3 (excluding IL baseline at index 0)
        recent = self.pool[-min(3, len(self.pool) - 1):]
        return random.choice(recent) if recent else self.pool[-1]

    def __len__(self) -> int:
        return len(self.pool)


def init_ema(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Detached fp32 CPU copy of the full state dict (params AND buffers —
    BatchNorm running stats must shadow too or the EMA net sees stale
    normalization)."""
    return {
        k: v.detach().to("cpu", torch.float32, copy=True)
        if v.dtype.is_floating_point else v.detach().to("cpu", copy=True)
        for k, v in model.state_dict().items()
    }


def update_ema(ema_state: Dict[str, torch.Tensor], model: torch.nn.Module,
               decay: float) -> None:
    """ema = decay*ema + (1-decay)*current for float tensors; integer
    buffers (e.g. num_batches_tracked) are copied through."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            e = ema_state[k]
            if v.dtype.is_floating_point:
                e.mul_(decay).add_(v.detach().to("cpu", torch.float32),
                                   alpha=1.0 - decay)
            else:
                e.copy_(v.detach().to("cpu"))


# Relative weights of the per-head KL-anchor terms (source dominates the
# action space; direction/split are conditional micro; pass guards against
# the entropy bonus dragging the pass Bernoulli toward 50%). The composite
# is scaled by --kl_weight.
KL_HEAD_WEIGHTS = {"kl_src": 1.0, "kl_dir": 0.5, "kl_spl": 0.25, "kl_pass": 0.5}


# ======================================================================
# Auxiliary capture-feasibility head
# ======================================================================

class CaptureHead(torch.nn.Module):
    """Per-tile, per-direction capture-feasibility predictor.

    Reads the Stage-1 belief tokens and predicts, for each owned tile and
    direction onto a defended target (neutral city or enemy tile), whether
    a full-strength move would capture it — labels come exactly from the
    env (``capture_label``/``capture_mask`` observation keys).

    The policy's army input is a single log-compressed channel, so the
    "src.army - 1 > dst.army" threshold is nearly invisible in input space;
    this head's BCE backpropagates into the encoder and forces it to
    represent the comparison. Kept OUTSIDE ``GeneralsNetwork`` so model
    checkpoints, the opponent pool and the KL reference stay byte-
    compatible; it is trained by its own small optimizer.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model // 2, 4),
        )

    def forward(self, belief_tokens: torch.Tensor) -> torch.Tensor:
        """[B, T, N, d] → capture logits [B, T, N, 4]."""
        return self.net(belief_tokens)


# ======================================================================
# Self-play rollout collection
# ======================================================================

def collect_selfplay_episode(
    model: GeneralsNetwork,
    opponent: GeneralsNetwork,
    env: GeneralsEnv,
    device: torch.device,
    max_steps: int = 1000,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    record: bool = False,
    act_steps: int = 1,
    legal_mask: bool = True,
    learner_pid: int = 0,
    reset_seed: Optional[int] = None,
    collect_both: bool = False,
    collect_buffers: bool = True,
) -> Tuple[List[RolloutBuffer], Optional[List[Dict]], Dict[str, float]]:
    """Run one self-play episode: model as ``learner_pid``, opponent as the
    other seat.

    Collects the learner's trajectory — or BOTH seats' trajectories with
    ``collect_both`` (mirror self-play: the opponent's forward pass happens
    every step anyway; a mirror game turns it into a second training
    trajectory for free). Returned buffers are ordered [learner, (other)].
    The learner's seat matters: the env's move-priority tiebreak favors P0,
    so training/eval pinned to P0 would learn (and measure) a rules
    advantage that doesn't transfer.
    With ``legal_mask`` both players sample from source/direction
    distributions hard-masked to legal actions (the PPO update must replay
    with the same masks); without it, the legacy unmasked RL mode.
    ``reset_seed`` reproduces a map (paired seat-swapped evaluation).
    ``collect_buffers=False`` plays the game without storing transitions or
    computing GAE (evaluation only reads env.winner — a 2048-step buffer is
    ~90MB of immediately-discarded allocations per game).

    If ``record=True``, returns a list of per-step dicts for replay viewing.
    """
    model.eval()
    opponent.eval()
    env.breakdown_pid = learner_pid
    obs = env.reset(seed=reset_seed)
    nets = {learner_pid: model, 1 - learner_pid: opponent}
    seats = (learner_pid, 1 - learner_pid) if collect_both else (learner_pid,)
    buffers = {pid: RolloutBuffer() for pid in seats}
    caches = {0: None, 1: None}    # spatiotemporal KV caches
    replay_log: Optional[List[Dict]] = [] if record else None
    reward_totals: Dict[str, float] = {
        "r_invalid": 0.0, "r_city": 0.0, "r_dense": 0.0, "r_winlose": 0.0,
        "c_cap": 0.0, "c_bash": 0.0, "c_steal": 0.0, "c_loss": 0.0,
    }

    # Record initial state before any moves
    if replay_log is not None:
        snapshot = env.get_state_snapshot()
        replay_log.append({
            "turn": -1,
            "p0_action": None, "p1_action": None,
            "p0_reward": 0.0, "p1_reward": 0.0,
            "illegal": [False, False],
            "p0_stats": env._player_stats(0),
            "p1_stats": env._player_stats(1),
            "state": snapshot["state"],
            "army": snapshot["army"],
            "done": False, "winner": None,
        })

    for _step in range(max_steps):
        step_out = {}

        for pid in (0, 1):
            net = nets[pid]
            grid = obs[pid]["grid"].unsqueeze(0).to(device)
            glb = obs[pid]["globals"].unsqueeze(0).to(device)

            with torch.no_grad():
                if legal_mask:
                    out = net.step(
                        grid, glb,
                        legal_mask=obs[pid]["legal_mask"].unsqueeze(0).to(device),
                        direction_mask=obs[pid]["dir_mask"].unsqueeze(0).to(device),
                        st_cache=caches[pid], step_idx=_step,
                        act_steps=act_steps)
                else:
                    out = net.step_rl(grid, glb,
                                      st_cache=caches[pid], step_idx=_step,
                                      act_steps=act_steps)

            caches[pid] = out.get("st_cache")
            step_out[pid] = out

        # Build env actions. With masking, a player with NO legal source
        # (e.g. turn 0, all armies == 1) sampled through the all-masked
        # fallback — send a pass instead: the env treats any illegal move
        # as a no-op anyway, this just skips the unavoidable -0.05 noise.
        actions_for_env = {}
        for pid in [0, 1]:
            did_pass = step_out[pid]["action_pass"].item()
            no_moves = legal_mask and not bool(obs[pid]["legal_mask"].any())
            if not did_pass and not no_moves:
                actions_for_env[pid] = {
                    "source_idx": step_out[pid]["action_source"].item(),
                    "direction": step_out[pid]["action_direction"].item(),
                    "split": step_out[pid]["action_split"].item(),
                }
            else:
                actions_for_env[pid] = None

        # Environment step
        new_obs, rewards, done, info = env.step(
            actions_for_env.get(0), actions_for_env.get(1),
        )

        # Record step for replay (snapshot AFTER env.step)
        if replay_log is not None:
            snapshot = env.get_state_snapshot()
            replay_log.append({
                "turn": info["turn"],
                "p0_action": actions_for_env.get(0),
                "p1_action": actions_for_env.get(1),
                "p0_reward": rewards[0],
                "p1_reward": rewards[1],
                "illegal": [info["illegal"][0], info["illegal"][1]],
                "p0_stats": env._player_stats(0),
                "p1_stats": env._player_stats(1),
                "state": snapshot["state"],
                "army": snapshot["army"],
                "done": done,
                "winner": info.get("winner"),
            })

        # Accumulate reward breakdown
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                reward_totals[k] += v

        # Store the collected seats' transitions
        for pid in seats if collect_buffers else ():
            action_dict = {
                "source_idx": step_out[pid]["action_source"].item(),
                "direction": step_out[pid]["action_direction"].item(),
                "split": step_out[pid]["action_split"].item(),
                "passed": step_out[pid]["action_pass"].item(),
            }
            # Compute valid directions for chosen source
            valid_dirs = env.get_valid_directions(action_dict["source_idx"])
            buffers[pid].add(
                obs=obs[pid],
                action=action_dict,
                log_prob=step_out[pid]["action_log_prob"].item(),
                value=step_out[pid]["value"].item(),
                reward=rewards[pid],
                # Truncation (env turn limit) is a SOFT terminal: the env
                # already scored it by material — GAE must not bootstrap.
                done=done,
                valid_directions=valid_dirs,
            )

        obs = new_obs
        if done:
            break

    # Rollout budget exhausted mid-game: the env never saw a terminal, so
    # apply the same material-based truncation score here and close the
    # episode — mirrors the env's own turn-limit handling.
    for pid in seats if collect_buffers else ():
        buf = buffers[pid]
        if not done and len(buf) > 0:
            buf.rewards[-1] += env.truncation_score(pid)
            buf.dones[-1] = True
            if pid == learner_pid:
                reward_totals["r_winlose"] += env.truncation_score(pid)
        buf.compute_returns_and_advantages(
            gamma=gamma, lam=gae_lambda, last_value=0.0,
        )

    return [buffers[pid] for pid in seats], replay_log, reward_totals


# ======================================================================
# PPO update
# ======================================================================

def anchor_weight(base: float, t: int, pow_: float, floor: float) -> float:
    """Decayed KL-anchor weight: base / (1 + t)^pow_, floored.

    ``t`` counts iterations since the current REGIME started (see
    ``resolve_phase_start``) — a phase boundary resets the clock, a
    within-phase crash recovery continues it. ``pow_ = 0`` reproduces the
    legacy constant weight.
    """
    w = base if pow_ <= 0 else base / ((1.0 + max(t, 0)) ** pow_)
    return max(w, floor)


def load_sparring_net(path: str, device: torch.device,
                      expected_globals: int) -> GeneralsNetwork:
    """Load the frozen sparring partner from ITS OWN saved config.

    The learner's architecture may have moved on (e.g. HL-Gauss critic);
    the sparring net only needs a compatible OBSERVATION interface — its
    policy heads drive its moves, its value head is never read. Globals
    width must match what the env emits for the learner (the env has one
    setting per run), hence the explicit check instead of a silent slice.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sp_cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                            if k in ModelConfig.__dataclass_fields__})
    if sp_cfg.global_features != expected_globals:
        raise ValueError(
            f"sparring checkpoint expects {sp_cfg.global_features} globals "
            f"but the run's env emits {expected_globals}; pick a sparring "
            "checkpoint from the same input-feature era")
    net = GeneralsNetwork(sp_cfg).to(device)
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    load_model_state(net, ckpt[state_key], ctx="sparring checkpoint")
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


def resolve_phase_start(rl_ckpt: Optional[Dict], start_iter: int,
                        cross_phase: bool) -> int:
    """First iteration of the current regime, for the anchor-weight clock.

    Within-phase crash recovery (same output dir) continues the clock
    persisted in the checkpoint, so a restart cannot snap a decayed anchor
    back to full strength — late in a phase that jump grows arbitrarily
    large and shocks the policy. Any cross-phase resume, or a checkpoint
    predating this key, restarts the clock at this run's first iteration
    (the phase-boundary reset the schedule was designed around).
    """
    if rl_ckpt is not None and not cross_phase:
        return int(rl_ckpt.get("phase_start_iter", start_iter))
    return start_iter


def ppo_update(
    model: GeneralsNetwork,
    ref_model: Optional[GeneralsNetwork],
    buffers: List[RolloutBuffer],
    rl_criterion: RLLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cfg: ModelConfig,
    world_size: int = 1,
    kl_weight: float = 0.01,
    invalidity_weight: float = 0.1,
    grad_clip: float = 1.0,
    act_steps: int = 1,
    capture_head: Optional[CaptureHead] = None,
    capture_optimizer: Optional[torch.optim.Optimizer] = None,
    capture_weight: float = 0.0,
    legal_masked: bool = True,
    accum_episodes: int = 0,
    n_chunks: int = 1,
    adv_filter_frac: float = 0.0,
) -> Dict[str, float]:
    """Run one PPO update epoch with gradient accumulation over multiple
    trajectories.

    ``accum_episodes > 0`` splits the epoch into ``n_chunks`` optimizer
    steps of that many local buffers each (2 Adam steps/iteration is a far
    outlier vs PPO practice — AverageJoe takes 128). ``n_chunks`` MUST be
    the all-reduced MAX across ranks: every rank has to post the same
    number of sync_gradients collectives even when its own chunk is empty
    (buffer counts vary per rank under mirror self-play), or NCCL pairs
    collectives across different steps and silently corrupts them.
    """
    # eval mode: rollout log-probs were collected in eval mode, so the update
    # must disable dropout and use BatchNorm running stats — otherwise
    # old/new log-probs come from different policies and the ratio is invalid.
    # Gradients still flow in eval mode.
    model.eval()
    optimizer.zero_grad()
    use_capture = (capture_head is not None and capture_optimizer is not None
                   and capture_weight > 0)
    if use_capture:
        capture_optimizer.zero_grad()

    n_buffers = len(buffers)
    metrics_accum: Dict[str, float] = {}

    def _flush_step() -> None:
        """Sync, clip and apply one optimizer step (both optimizers).

        An empty local chunk needs NO explicit zero grads: sync_gradients'
        presence mask is MAX-reduced, so the data-holding ranks define
        participation and this rank contributes zeros with an identical
        collective count. Pre-filling zeros here would instead mark heads
        untouched by the objective as present everywhere, silently breaking
        the never-allreduced/never-stepped semantics for skipped modules.
        """
        if world_size > 1:
            sync_gradients(model, world_size)
            if use_capture:
                sync_gradients(capture_head, world_size)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if use_capture:
                torch.nn.utils.clip_grad_norm_(
                    capture_head.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        if use_capture:
            capture_optimizer.step()
            capture_optimizer.zero_grad()

    if accum_episodes and accum_episodes > 0:
        chunks = [buffers[i * accum_episodes:(i + 1) * accum_episodes]
                  for i in range(n_chunks)]
    else:
        chunks = [buffers]

    def _process_buffer(buf: RolloutBuffer, denom: int) -> None:
        batch = buf.to_tensor_batch(device)

        grid_seq = batch["grid_seq"]        # [1, T, C, H, W]
        globals_seq = batch["globals_seq"]  # [1, T, G]
        actions = batch["actions"]
        old_lp = batch["old_log_probs"]     # [1, T]
        advantages = batch["advantages"]    # [1, T]
        returns = batch["returns"]          # [1, T]

        # Forward with stored source positions. Masks must match the rollout
        # exactly (masked there => masked here) or the PPO ratio is invalid.
        source_y = actions["source_idx"] // cfg.grid_w
        source_x = actions["source_idx"] % cfg.grid_w
        use_mask = legal_masked and "dir_mask" in batch
        src_mask = batch["legal_mask"] if use_mask else None
        dir_mask = batch["dir_mask"] if use_mask else None

        pred = model(
            grid_seq, globals_seq,
            legal_mask=src_mask,
            source_y=source_y,
            source_x=source_x,
            act_steps=act_steps,
            direction_mask=dir_mask,
        )

        if use_mask:
            log_probs, entropy, head_entropy = model.compute_log_prob(
                pred, actions, legal={"source": src_mask})
        else:
            log_probs, entropy, head_entropy = \
                model.compute_action_log_prob_rl(pred, actions)

        # Advantage filter (AverageJoe's top-25%): the policy trains only
        # on this episode's highest-|advantage| samples — the decisive
        # moves — which also mutes the noise-dominated gradient of long
        # losing streaks. Per-episode (not per-batch) so every episode
        # contributes its own most-informative steps regardless of how
        # extreme its neighbours in the batch are. Advantages are already
        # normalized across the iteration at this point. kthvalue keeps
        # ties, so the kept count is >= the target fraction.
        policy_mask = None
        adv_kept = 1.0
        n_samples = advantages.numel()
        if 0.0 < adv_filter_frac < 1.0 and n_samples >= 8:
            k = max(1, int(round(n_samples * adv_filter_frac)))
            thresh = advantages.abs().flatten().kthvalue(
                n_samples - k + 1).values
            policy_mask = advantages.abs() >= thresh
            adv_kept = policy_mask.float().mean().item()

        # RL loss
        rl_losses = rl_criterion(
            log_probs=log_probs,
            entropy=entropy,
            values=pred["value"],
            returns=returns,
            advantages=advantages,
            old_log_probs=old_lp,
            value_logits=pred.get("value_logits"),
            policy_mask=policy_mask,
        )

        total = rl_losses["loss"]

        # KL regularisation against reference policy, over ALL four heads.
        # Source-only anchoring left pass-rate/direction/split free to
        # drift (and the entropy bonus pushes pass toward 50%); the ref
        # forward computes those logits anyway. Under masking the KL
        # compares the MASKED distributions — the IL reference was trained
        # with the mask, so its illegal-cell logits never received gradient
        # and are noise; the unmasked KL was anchoring to that noise.
        kl_loss = torch.tensor(0.0, device=device)
        kl_heads: Dict[str, float] = {}
        if ref_model is not None and kl_weight > 0:
            with torch.no_grad():
                ref_pred = ref_model(
                    grid_seq, globals_seq,
                    legal_mask=src_mask,
                    source_y=source_y,
                    source_x=source_x,
                    act_steps=act_steps,
                    direction_mask=dir_mask,
                )
            kls = compute_policy_kl_heads(
                pred, ref_pred,
                legal_mask=src_mask,
                passed=actions["passed"],
            )
            kl_loss = sum(KL_HEAD_WEIGHTS[k] * v for k, v in kls.items())
            kl_heads = {k: v.item() for k, v in kls.items()}
            total = total + kl_weight * kl_loss

        # Invalidity loss: push probability toward legal sources & valid
        # directions. Redundant under hard masking (the masked policy places
        # zero mass on invalid actions by construction) — skip it there.
        inv_loss = torch.tensor(0.0, device=device)
        if invalidity_weight > 0 and not use_mask:
            T = grid_seq.size(1)
            not_passed = ~actions["passed"]  # [1, T]

            # Source validity: encourage prob mass on legal source positions
            legal_mask_flat = batch["legal_mask"].reshape(1, T, -1).float()
            source_probs = F.softmax(pred["source_logits"], dim=-1)
            prob_on_valid_src = (source_probs * legal_mask_flat).sum(-1)
            src_val_loss = -torch.log(prob_on_valid_src.clamp(min=1e-8))
            src_mask = not_passed.float()
            n_src = src_mask.sum().clamp(min=1)
            src_val_loss = (src_val_loss * src_mask).sum() / n_src

            # Direction validity: encourage prob mass on valid directions
            valid_dirs = batch["valid_directions"].float()
            dir_probs = F.softmax(pred["direction_logits"], dim=-1)
            prob_on_valid_dir = (dir_probs * valid_dirs).sum(-1)
            dir_val_loss = -torch.log(prob_on_valid_dir.clamp(min=1e-8))
            source_was_valid = legal_mask_flat.gather(
                -1, actions["source_idx"].unsqueeze(-1),
            ).squeeze(-1)
            dir_mask = source_was_valid * not_passed.float()
            n_dir = dir_mask.sum().clamp(min=1)
            dir_val_loss = (dir_val_loss * dir_mask).sum() / n_dir

            inv_loss = src_val_loss + dir_val_loss
            total = total + invalidity_weight * inv_loss

        # Auxiliary capture-feasibility BCE: exact env labels, masked to
        # (owned movable tile, defended neighbour) pairs. Backpropagates
        # through the belief tokens into the encoder.
        cap_loss = torch.tensor(0.0, device=device)
        cap_acc = 0.0
        if use_capture and "capture_label" in batch:
            T = grid_seq.size(1)
            cap_logits = capture_head(pred["belief_tokens"])  # [1, T, N, 4]
            # [1, T, 4, H, W] → [1, T, N, 4] matching the token layout
            cap_labels = (batch["capture_label"]
                          .reshape(1, T, 4, -1).permute(0, 1, 3, 2))
            cap_mask = (batch["capture_mask"]
                        .reshape(1, T, 4, -1).permute(0, 1, 3, 2)).float()
            n_lab = cap_mask.sum().clamp(min=1)
            bce = F.binary_cross_entropy_with_logits(
                cap_logits, cap_labels, reduction="none")
            cap_loss = (bce * cap_mask).sum() / n_lab
            total = total + capture_weight * cap_loss
            with torch.no_grad():
                cap_acc = ((((cap_logits > 0).float() == cap_labels).float()
                            * cap_mask).sum() / n_lab).item()

        # Scale loss for gradient accumulation within this chunk
        (total / max(denom, 1)).backward()

        # Accumulate metrics
        step_metrics = {
            "total_loss": total.item(),
            "policy_loss": rl_losses["policy_loss"].item(),
            "value_loss": rl_losses["value_loss"].item(),
            "entropy": rl_losses["entropy"].item(),
            "approx_kl": rl_losses["approx_kl"].item(),
            "clip_frac": rl_losses["clip_fraction"].item(),
            "kl_reg": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "inv_loss": inv_loss.item() if isinstance(inv_loss, torch.Tensor) else inv_loss,
            "ent_pass": head_entropy["ent_pass"].item(),
            "ent_src": head_entropy["ent_src"].item(),
            "ent_dir": head_entropy["ent_dir"].item(),
            "ent_spl": head_entropy["ent_spl"].item(),
            "cap_loss": cap_loss.item() if isinstance(cap_loss, torch.Tensor) else cap_loss,
            "cap_acc": cap_acc,
            "adv_kept": adv_kept,
        }
        step_metrics.update(kl_heads)
        for k, v in step_metrics.items():
            metrics_accum[k] = metrics_accum.get(k, 0.0) + v

    for chunk in chunks:
        for buf in chunk:
            _process_buffer(buf, denom=len(chunk))
        _flush_step()

    return {k: v / max(n_buffers, 1) for k, v in metrics_accum.items()}


def il_auxiliary_update(
    model: GeneralsNetwork,
    il_batch: Dict,
    il_criterion: ImitationLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    il_weight: float,
    world_size: int = 1,
    grad_clip: float = 1.0,
    act_steps: int = 1,
) -> Dict[str, float]:
    """One gradient step on IL expert data (auxiliary regularisation).

    IL path keeps legal_mask (model learns from expert data with mask).
    Runs at the same ACT depth as PPO so the regularizer constrains the
    policy that is actually being trained.
    """
    model.train()

    grid_seq = il_batch["obs"]["grid_seq"].to(device)
    globals_seq = il_batch["obs"]["globals_seq"].to(device)
    # IL shards carry 4 globals; under cfg.turn_globals synthesize the
    # production-cycle features from the frame index — the dataset
    # truncates tail-only, so frame i IS turn i.
    if (model.cfg.turn_globals
            and globals_seq.shape[-1] < model.cfg.global_features):
        B_il, T_il, _ = globals_seq.shape
        t = torch.arange(T_il, device=device, dtype=torch.float32)
        t = t.view(1, T_il, 1).expand(B_il, -1, -1)
        extra = torch.cat([(t % 50) / 50.0, t % 2, t / 500.0], dim=-1)
        globals_seq = torch.cat([globals_seq, extra], dim=-1)
    legal_mask = il_batch["obs"]["legal_source_mask_seq"].to(device)
    tgt = {k: v.to(device) for k, v in il_batch["tgt"].items()}
    lengths = il_batch["meta"]["lengths"]

    pred = model(
        grid_seq, globals_seq,
        legal_mask=legal_mask,
        source_y=None,
        source_x=None,
        act_steps=act_steps,
    )

    il_losses = il_criterion(pred, tgt, lengths)

    loss = il_weight * il_losses["loss"]
    optimizer.zero_grad()
    loss.backward()
    if world_size > 1:
        sync_gradients(model, world_size)
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {
        "il_loss": il_losses["loss"].item(),
        "il_source": il_losses["source_loss"].item(),
    }


# ======================================================================
# Main training loop
# ======================================================================

def main():
    args = parse_args()

    # Distributed setup
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
    world_size = dist.get_world_size() if distributed else 1
    if rank0:
        suffix = f" (world_size={world_size})" if distributed else ""
        print(f"Device: {device}{suffix}")

    # ---- Build or load model -------------------------------------------
    # A resume checkpoint defines the architecture: rebuilding from CLI
    # defaults would silently change the run's lineage (drop the world-model
    # head, switch temporal mixer) or crash at opponent-pool load.
    rl_ckpt = None
    if args.resume:
        # Load once, on CPU: the checkpoint embeds the whole opponent pool,
        # and load_state_dict copies to the right devices on its own.
        rl_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

    il_ckpt = None
    if rl_ckpt is not None and "config" in rl_ckpt:
        cfg = ModelConfig(**{k: v for k, v in rl_ckpt["config"].items()
                             if k in ModelConfig.__dataclass_fields__})
        src_msg = f"Architecture from resume checkpoint {args.resume}"
    elif args.il_checkpoint:
        il_ckpt = torch.load(args.il_checkpoint, map_location=device,
                             weights_only=False)
        cfg = ModelConfig(**il_ckpt["config"])
        src_msg = f"Loaded IL checkpoint from {args.il_checkpoint}"
    else:
        cfg = ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            trm_H_cycles=args.trm_H_cycles,
            trm_L_cycles=args.trm_L_cycles,
            trm_block_layers=args.trm_block_layers,
            temporal_mixer=args.temporal_mixer,
        )
        src_msg = "No IL checkpoint — training from scratch"

    # Input-feature upgrades (one-way; checkpoints saved after this carry
    # the flags in their config, so plain resumes keep them without the
    # CLI). Enabling widens conv1 / global_proj — the shape-adaptive loader
    # zero-pads every old state dict (resume weights, IL weights, KL ref,
    # each pool snapshot), which is exactly function-preserving.
    if args.derive_army_channels:
        cfg.derive_army_channels = True
    if args.turn_globals:
        cfg.turn_globals = True
    if cfg.turn_globals:
        cfg.global_features = max(cfg.global_features, 7)

    # HL-Gauss critic is sticky like the input features, but NOT
    # function-preserving: freshly enabling it swaps the value head for a
    # different architecture, so the head is reinitialised at every load
    # of a scalar-critic state dict (policy weights are untouched; GAE
    # runs off an untrained critic for the first iterations, which under
    # sparse ±1 rewards degrades to REINFORCE — acceptable at a phase
    # boundary, which is the only place this flag should be flipped).
    hl_gauss_migrated = False
    if args.hl_gauss_value and not cfg.hl_gauss_value:
        cfg.hl_gauss_value = True
        hl_gauss_migrated = True
    value_reinit = ("value_head",) if hl_gauss_migrated else ()

    model = GeneralsNetwork(cfg).to(device)
    if il_ckpt is not None:
        load_model_state(model, il_ckpt["model_state"], ctx="IL checkpoint",
                         reinit_prefixes=value_reinit)
        del il_ckpt
    if rank0:
        print(src_msg)
        if rl_ckpt is not None and args.il_checkpoint:
            print("  [note] --il_checkpoint ignored on resume (weights and "
                  "architecture come from the resume checkpoint)")
        if cfg.derive_army_channels or cfg.turn_globals:
            print(f"  Input features: derive_army_channels="
                  f"{cfg.derive_army_channels} turn_globals={cfg.turn_globals}"
                  f" (global_features={cfg.global_features})")
        if cfg.hl_gauss_value:
            migrated = " (fresh head: migrated from scalar critic)" \
                if hl_gauss_migrated else ""
            print(f"  HL-Gauss critic: {cfg.hl_gauss_bins} bins on "
                  f"[-{cfg.hl_gauss_vmax}, {cfg.hl_gauss_vmax}], "
                  f"sigma={cfg.hl_gauss_sigma}{migrated}")

    n_params = sum(p.numel() for p in model.parameters())
    if rank0:
        print(f"Model parameters: {n_params:,}")

    # ---- Resume: load weights & counters FIRST -------------------------
    # Must happen before the KL reference model and opponent pool are
    # captured, otherwise a resume without --il_checkpoint regularizes
    # toward (and plays against) a randomly initialized network.
    start_iter = 1
    il_weight = args.il_weight
    wins = {0: 0, 1: 0, "draw": 0}
    if args.resume:
        # (checkpoint already loaded above, where it defined the architecture)
        # Support both RL checkpoints (model_state) and IL checkpoints
        state_key = "model_state" if "model_state" in rl_ckpt else "model_state_dict"
        load_model_state(model, rl_ckpt[state_key], ctx="resume checkpoint",
                         reinit_prefixes=value_reinit)
        start_iter = rl_ckpt.get("iteration", 0) + 1
        il_weight = rl_ckpt.get("il_weight", args.il_weight)
        wins = rl_ckpt.get("wins", wins)
        if rank0:
            print(f"Resumed weights from {args.resume} (starting iteration {start_iter})")

    # Regime clock for the anchor-weight schedule (must be resolved before
    # the training loop; persisted into every checkpoint below).
    cross_phase = bool(args.resume) and (
        os.path.dirname(os.path.abspath(args.resume))
        != os.path.abspath(args.output_dir))
    phase_start_iter = resolve_phase_start(rl_ckpt, start_iter, cross_phase)
    if rank0 and args.kl_weight_pow > 0:
        print(f"  Anchor clock: phase start iter {phase_start_iter} "
              f"(t={start_iter - phase_start_iter} at resume)")

    # ---- Auxiliary capture head (own optimizer, outside the model) ------
    # Separate module + optimizer so model checkpoints, the opponent pool
    # and the KL reference stay byte-compatible with head-less runs.
    capture_head: Optional[CaptureHead] = None
    capture_optimizer: Optional[torch.optim.Optimizer] = None
    if args.capture_aux_weight > 0:
        capture_head = CaptureHead(cfg.d_model).to(device)
        capture_optimizer = torch.optim.AdamW(
            capture_head.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
        if rl_ckpt is not None and "capture_head_state" in rl_ckpt:
            capture_head.load_state_dict(rl_ckpt["capture_head_state"])
            if "capture_optim_state" in rl_ckpt:
                capture_optimizer.load_state_dict(rl_ckpt["capture_optim_state"])
            if rank0:
                print("  Capture head resumed from checkpoint")
        elif rank0:
            print(f"Capture head enabled (weight={args.capture_aux_weight})")

    # ---- Sync initial state across ranks --------------------------------
    # Gradients are manually all-reduced, but nothing else aligns the ranks:
    # a from-scratch multi-GPU run would average gradients over different
    # random initializations. Broadcast rank 0's parameters and buffers
    # (DDP does this at construction; the manual setup must too).
    if distributed:
        for t in model.state_dict().values():
            if isinstance(t, torch.Tensor):
                dist.broadcast(t, src=0)
        if capture_head is not None:
            for t in capture_head.state_dict().values():
                if isinstance(t, torch.Tensor):
                    dist.broadcast(t, src=0)

    # ---- EMA shadow weights (post-broadcast so all ranks agree) --------
    ema_state: Optional[Dict[str, torch.Tensor]] = None
    if args.ema_decay > 0:
        saved_ema = rl_ckpt.get("ema_state") if rl_ckpt is not None else None
        if saved_ema is not None and any(
                k not in saved_ema or saved_ema[k].shape != v.shape
                for k, v in model.state_dict().items()):
            # Input-feature migration widened conv1/global_proj: restart the
            # EMA from the (function-preserved) current weights.
            saved_ema = None
            if rank0:
                print("  EMA restarted: checkpoint EMA shapes predate the "
                      "input-feature migration")
        if saved_ema is not None:
            ema_state = {k: v.clone() for k, v in saved_ema.items()}
            if rank0:
                print("  EMA weights resumed from checkpoint")
        else:
            ema_state = init_ema(model)
            if rank0:
                print(f"EMA enabled (decay={args.ema_decay}, "
                      f"initialized from current weights)")

    # ---- Reference model (frozen copy for KL regularisation) -----------
    ref_model: Optional[GeneralsNetwork] = None
    if args.kl_weight > 0:
        ref_model = copy.deepcopy(model)
        if args.kl_ref:
            ref_ckpt = torch.load(args.kl_ref, map_location="cpu", weights_only=False)
            # The reference anchors the POLICY heads only; its value head is
            # never read, so when the anchor checkpoint predates the
            # HL-Gauss critic the fresh-init head is simply dead weight.
            ref_reinit = ("value_head",) if (
                cfg.hl_gauss_value
                and not ref_ckpt.get("config", {}).get("hl_gauss_value")
            ) else ()
            load_model_state(ref_model, ref_ckpt["model_state"],
                             ctx="KL reference checkpoint",
                             reinit_prefixes=ref_reinit)
            del ref_ckpt
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        if rank0:
            src = args.kl_ref if args.kl_ref else "this run's starting weights"
            print(f"Created frozen reference model for KL regularisation ({src})")

    # ---- Opponent pool -------------------------------------------------
    opponent_pool = OpponentPool(
        initial_state=model.state_dict(),
        max_size=args.pool_size,
    )
    if rl_ckpt is not None and "opponent_pool" in rl_ckpt:
        if hl_gauss_migrated:
            # Pool snapshots carry the scalar critic and can no longer be
            # loaded into opponent_model. Dropping them is safe: pool
            # opponents' value heads are never queried, but load_state_dict
            # is strict about shapes — and a phase that flips this flag is
            # a regime change that wants a fresh pool anyway.
            if rank0:
                print("  [note] resume opponent pool dropped "
                      "(HL-Gauss migration changed the critic shapes)")
        else:
            opponent_pool.pool = [
                OpponentPool._to_cpu(sd) for sd in rl_ckpt["opponent_pool"]
            ]
    # Opponent model (loaded with pool checkpoint each iteration)
    opponent_model = copy.deepcopy(model)
    opponent_model.eval()
    for p in opponent_model.parameters():
        p.requires_grad = False
    if rank0:
        print(f"Opponent pool initialised (max_size={args.pool_size})")

    # ---- Frozen sparring partner (style counterweight) -----------------
    sparring_net: Optional[GeneralsNetwork] = None
    if args.sparring_ckpt and args.sparring_prob > 0:
        sparring_net = load_sparring_net(
            args.sparring_ckpt, device, cfg.global_features)
        if rank0:
            print(f"Sparring partner: {args.sparring_ckpt} "
                  f"(prob={args.sparring_prob}, frozen)")

    # ---- Encoder freeze ------------------------------------------------
    # Phase decided by start_iter so a resume past the boundary builds the
    # optimizer over the full parameter set from the start.
    frozen_phase = args.freeze_encoder_iters > 0 and start_iter <= args.freeze_encoder_iters
    if frozen_phase:
        freeze_encoder(model)
        if rank0:
            print(f"Encoder frozen for the first {args.freeze_encoder_iters} iterations")

    # ---- DDP -----------------------------------------------------------
    # No DDP wrapper — use manual gradient all-reduce instead.
    # DDP conflicts with gradient checkpointing + varying computation graphs
    # (PPO vs IL forward paths). Manual sync is simpler and reliable.

    # ---- Losses --------------------------------------------------------
    rl_criterion = RLLoss(
        value_weight=args.value_weight,
        entropy_weight=args.entropy_weight,
        clip_range=args.clip_range,
        hl_gauss=bool(cfg.hl_gauss_value),
        hl_gauss_vmax=cfg.hl_gauss_vmax,
        hl_gauss_sigma=cfg.hl_gauss_sigma,
    )
    # AlphaStar-style auxiliary regularization targets the POLICY heads only:
    # the replay-outcome value labels live on a different scale and state
    # distribution than the PPO returns and would fight the critic every step.
    il_criterion = ImitationLoss(grid_w=cfg.grid_w, value_weight=0.0)

    # ---- Optimiser -----------------------------------------------------
    # optim_param_groups keeps weight decay off the recurrence memory
    # parameters (retention_bias / h_init) and respects requires_grad.
    optimizer = torch.optim.AdamW(
        optim_param_groups(model, args.weight_decay),
        lr=args.lr,
    )

    # ---- IL data loader (optional) -------------------------------------
    il_loader = None
    il_iter = None
    if args.il_data_dir:
        if len(args.il_data_dir) == 1:
            il_dataset = SeqReplayDataset(args.il_data_dir[0], max_sequence_length=200)
        else:
            from torch.utils.data import ConcatDataset
            il_dataset = ConcatDataset([
                SeqReplayDataset(d, max_sequence_length=200)
                for d in args.il_data_dir
            ])
        il_sampler = DistributedSampler(il_dataset, shuffle=True) if distributed else None
        il_loader = DataLoader(
            il_dataset, batch_size=1, shuffle=(il_sampler is None),
            sampler=il_sampler,
            collate_fn=collate_variable_length_sequences,
        )
        il_iter = iter(il_loader)
        if rank0:
            print(f"IL dataset: {len(il_dataset)} replays from {args.il_data_dir}")

    # ---- Environment (each rank gets its own) --------------------------
    # max_turns == rollout_steps: truncation is decided in ONE place (the
    # env's material-scored soft terminal); the collect-loop timeout patch
    # becomes a backstop instead of the common path.
    env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w,
                      max_turns=args.rollout_steps)
    env.city_budget_total = args.city_budget
    env.turn_globals = cfg.turn_globals
    # Truncation knobs are plain env attrs — never stored in checkpoints,
    # so setting them once here covers resumes too.
    env.trunc_material_scale = args.trunc_material
    env.trunc_penalty = args.trunc_penalty
    if rank0 and (args.trunc_penalty > 0 or args.trunc_material != 0.35):
        print(f"Truncation terminal: ±{args.trunc_material} material "
              f"− {args.trunc_penalty} stalling penalty (both players)")
    if args.sparse_reward:
        # Pure outcome reward: shaping scales pinned to zero here and after
        # every place that could re-set them (resume restore below, the
        # per-iteration decay in the training loop).
        env.city_reward_scale = 0.0
        env.dense_reward_scale = 0.0
        if rank0:
            print("Sparse reward: city/dense shaping OFF "
                  "(±1 terminal and ±0.35 soft terminal only)")
    if args.replay_dir and args.filtered_file:
        n_maps = env.load_map_pool(args.replay_dir, args.filtered_file)
        if rank0:
            print(f"Loaded {n_maps} replay maps for environment")

    # ---- Resume: optimizer state & schedules (weights loaded earlier) ---
    if args.resume:
        if "iteration" in rl_ckpt and "optimizer_state" in rl_ckpt:
            # Skip states saved for a different trainable set (e.g. a
            # checkpoint from the frozen phase resumed past the boundary) —
            # loading them raises inside torch.optim. Also skip states saved
            # before the game-adapter refactor: identical counts but a
            # different parameter order, which a bare count check misses.
            saved = rl_ckpt["optimizer_state"]
            saved_model = rl_ckpt.get(
                "model_state", rl_ckpt.get("model_state_dict", {}))
            if is_legacy_state_dict(saved_model, model):
                reason = "pre-refactor checkpoint (parameter order changed)"
            else:
                reason = optim_state_mismatch(optimizer, saved)
            if reason is None:
                optimizer.load_state_dict(saved)
            elif rank0:
                print(f"  [warn] optimizer state skipped ({reason}); "
                      f"starting with a fresh optimizer")

        # Restore the LR schedule: base LR decayed for the iterations already
        # done (like every other decayed quantity below), halved once the
        # unfreeze boundary has been crossed.
        done_iters = start_iter - 1
        resume_lr = args.lr * (args.lr_decay ** done_iters)
        if args.freeze_encoder_iters > 0 and start_iter > args.freeze_encoder_iters:
            resume_lr *= 0.5
        for pg in optimizer.param_groups:
            pg["lr"] = resume_lr
        if capture_optimizer is not None:
            for pg in capture_optimizer.param_groups:
                pg["lr"] = resume_lr
        if rank0:
            print(f"  LR restored to {resume_lr:.3e} (decay^{done_iters})")

        # Restore reward scales. A checkpoint from THIS phase (same
        # output_dir) continues its schedule exactly — --reset_reward_decay
        # must NOT re-zero the clock on every within-phase crash recovery.
        # A checkpoint from a DIFFERENT directory is a phase boundary: the
        # reset flag wins there, otherwise it is a silent no-op and the new
        # phase's shaping enters pre-annealed by however long the previous
        # phase ran.
        if args.reset_reward_decay and cross_phase:
            env.city_reward_scale = 1.0
            env.dense_reward_scale = 1.0
            if rank0:
                print("  Reward decay clocks reset (cross-phase resume)")
        elif "city_reward_scale" in rl_ckpt:
            env.city_reward_scale = rl_ckpt["city_reward_scale"]
            env.dense_reward_scale = rl_ckpt["dense_reward_scale"]
        elif args.reset_reward_decay:
            env.city_reward_scale = 1.0
            env.dense_reward_scale = 1.0
        else:
            env.city_reward_scale = args.city_reward_decay ** done_iters
            env.dense_reward_scale = args.dense_reward_decay ** done_iters
        if args.sparse_reward:
            env.city_reward_scale = 0.0
            env.dense_reward_scale = 0.0
        rl_criterion.entropy_weight = max(
            args.entropy_weight * (args.entropy_weight_decay ** done_iters),
            args.entropy_weight_min,
        )

        if rank0:
            print(f"Resumed from {args.resume}, iteration {start_iter}, "
                  f"pool={len(opponent_pool)}")
        rl_ckpt = None  # release checkpoint tensors

    os.makedirs(args.output_dir, exist_ok=True)

    # Win rate history for conditional pool updates
    from collections import deque
    winrate_history: deque = deque(maxlen=args.pool_update_window)
    # Fresh runs may add to the pool immediately; a resume waits a full
    # interval so a 1-sample winrate window can't evict a historical
    # opponent on the first iteration after every restart.
    last_pool_update_iter = start_iter if args.resume \
        else start_iter - args.pool_update_interval

    # Previous update's mean policy entropy, for the --entropy_ceiling gate
    last_policy_entropy: Optional[float] = None

    # ---- Training loop -------------------------------------------------
    for iteration in range(start_iter, args.iterations + 1):
        t0 = time.time()

        # Unfreeze encoder after schedule. Only fires when THIS process
        # started in the frozen phase — a resume landing exactly past the
        # boundary already applied the 0.5x factor in the resume block and
        # would otherwise be halved a second time.
        if frozen_phase and iteration == args.freeze_encoder_iters + 1:
            raw = model
            unfreeze_all(raw)
            current_lr = optimizer.param_groups[0]["lr"]
            optimizer = torch.optim.AdamW(
                optim_param_groups(model, args.weight_decay),
                lr=current_lr * 0.5,
            )
            if rank0:
                print(f"Iteration {iteration}: encoder unfrozen, LR reduced to {current_lr * 0.5:.3e}")

        # ---- 1/2. Collect self-play episodes (opponent drawn per episode:
        # a load is ~10-30ms against a multi-second episode, and per-episode
        # draws expose the learner to up to episodes_per_gpu distinct
        # opponents per gate window instead of one) ----
        should_record = (
            rank0
            and args.save_replay_interval > 0
            and iteration % args.save_replay_interval == 0
        )
        ep_buffers: List[RolloutBuffer] = []
        replay_log = None
        reward_totals: Dict[str, float] = {
            "r_invalid": 0.0, "r_city": 0.0, "r_dense": 0.0, "r_winlose": 0.0,
            "c_cap": 0.0, "c_bash": 0.0, "c_steal": 0.0, "c_loss": 0.0,
        }
        local_ep_len = 0.0
        local_ep_reward = 0.0
        local_p0 = 0
        local_p1 = 0
        local_draw = 0
        local_mirror = 0

        replay_env_snapshot = None  # env state for the recorded episode
        for ep_i in range(args.episodes_per_gpu):
            record_this = should_record and ep_i == 0
            # Sparring draw first: a frozen counter-style opponent breaks
            # pure mirror's blind spot (a drifting style is never punished
            # when both seats share it). Remaining probability mass splits
            # mirror-vs-pool exactly as before.
            sparring = (sparring_net is not None
                        and random.random() < args.sparring_prob)
            mirror = (not sparring and args.mirror_prob > 0
                      and random.random() < args.mirror_prob)
            if sparring:
                opp_net = sparring_net
                learner_pid = random.randint(0, 1) if args.randomize_seat else 0
            elif mirror:
                opp_net = model
                learner_pid = 0  # both seats collected; seat is symmetric
            else:
                opponent_model.load_state_dict(opponent_pool.sample())
                opp_net = opponent_model
                learner_pid = random.randint(0, 1) if args.randomize_seat else 0
            bufs, ep_replay, ep_rw = collect_selfplay_episode(
                model, opp_net, env, device,
                max_steps=args.rollout_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                record=record_this,
                act_steps=args.act_steps,
                legal_mask=bool(args.rl_legal_mask),
                learner_pid=learner_pid,
                collect_both=mirror,
            )
            if ep_replay is not None:
                replay_log = ep_replay
                replay_env_snapshot = {
                    "winner": env.winner,
                    "map_pool_idx": env.current_map_idx,
                    "replay_id": env.map_pool[env.current_map_idx]["replay_id"] if env.map_pool and env.current_map_idx is not None else None,
                    "real_h": env.real_h, "real_w": env.real_w,
                    "generals": env.generals, "cities": env.cities,
                    "mountains": env.mountains,
                }
            ep_buffers.extend(bufs)
            local_ep_len += len(bufs[0])
            local_ep_reward += sum(bufs[0].rewards)
            for k in reward_totals:
                reward_totals[k] += ep_rw[k]

            # Win counters are LEARNER-relative (seat is randomized) and
            # exclude mirror games — a mirror winrate is ~0.5 by
            # construction and would jam the pool gate.
            if mirror:
                local_mirror += 1
            elif env.winner == learner_pid:
                wins[0] += 1
                local_p0 += 1
            elif env.winner == 1 - learner_pid:
                wins[1] += 1
                local_p1 += 1
            else:
                wins["draw"] += 1
                local_draw += 1

        n_local_eps = args.episodes_per_gpu
        local_ep_len /= n_local_eps
        local_ep_reward /= n_local_eps
        for k in reward_totals:
            reward_totals[k] /= n_local_eps

        # ---- 3. PPO update with gradient accumulation ----
        # Normalize advantages across all local buffers + all GPUs
        all_advs = []
        for buf in ep_buffers:
            if len(buf) > 0:
                all_advs.append(buf.advantages)
        if all_advs:
            cat_adv = torch.cat(all_advs)
            if dist.is_initialized():
                local_sum = cat_adv.sum()
                local_sq_sum = (cat_adv ** 2).sum()
                local_count = torch.tensor(cat_adv.numel(), dtype=torch.float32, device=device)
                stats = torch.stack([local_sum.to(device), local_sq_sum.to(device), local_count])
                dist.all_reduce(stats)
                g_mean = stats[0] / stats[2]
                g_var = (stats[1] / stats[2] - g_mean ** 2).clamp(min=1e-8)
                for buf in ep_buffers:
                    if len(buf) > 0:
                        buf.advantages = (buf.advantages - g_mean.cpu()) / g_var.sqrt().cpu()
            else:
                adv_mean = cat_adv.mean()
                adv_std = cat_adv.std().clamp(min=1e-8)
                for buf in ep_buffers:
                    if len(buf) > 0:
                        buf.advantages = (buf.advantages - adv_mean) / adv_std
            del cat_adv, all_advs
            torch.cuda.empty_cache()

        valid_buffers = [buf for buf in ep_buffers if len(buf) > 0]
        # Chunked stepping: the chunk COUNT must be rank-invariant (buffer
        # counts differ per rank under mirror self-play) — every rank has
        # to post the same number of grad-sync collectives.
        if args.accum_episodes > 0:
            local_chunks = max(
                1, -(-len(valid_buffers) // args.accum_episodes))
            if dist.is_initialized():
                nc = torch.tensor([local_chunks], device=device)
                dist.all_reduce(nc, op=dist.ReduceOp.MAX)
                n_chunks = int(nc.item())
            else:
                n_chunks = local_chunks
        else:
            n_chunks = 1

        # Anchor weight schedule: strong at the phase start, fading as
        # self-play takes over (clock = iterations since THIS run began).
        kl_weight_t = anchor_weight(
            args.kl_weight, iteration - phase_start_iter,
            args.kl_weight_pow, args.kl_weight_min)

        # Entropy ceiling: the phase-5 collapse rode a feedback loop of
        # losing → exploding entropy (3.0) → noise updates → losing. When
        # the PREVIOUS update's mean policy entropy is already above the
        # ceiling, spend no bonus pushing it higher this iteration — and
        # with --entropy_penalty, actively push it back down. Phase-6
        # showed the zeroing gate alone can't stop a runaway: long-game /
        # large-territory feedback kept raising entropy 2.3 → 3.5 with the
        # gate firing every single iteration.
        base_entropy_weight = rl_criterion.entropy_weight
        entropy_gated = (args.entropy_ceiling > 0
                         and last_policy_entropy is not None
                         and last_policy_entropy > args.entropy_ceiling)
        if entropy_gated:
            rl_criterion.entropy_weight = -args.entropy_penalty
            if rank0:
                mode = (f"pushing down at {args.entropy_penalty}"
                        if args.entropy_penalty > 0 else "bonus off")
                print(f"  Entropy gate ({mode}): last ent "
                      f"{last_policy_entropy:.2f} > {args.entropy_ceiling}")

        ppo_metrics_agg = {}
        epochs_run = 0
        if valid_buffers or (dist.is_initialized() and args.ppo_epochs > 0):
            for _epoch in range(args.ppo_epochs):
                metrics = ppo_update(
                    model, ref_model, valid_buffers, rl_criterion,
                    optimizer, device, cfg,
                    world_size=world_size,
                    kl_weight=kl_weight_t,
                    invalidity_weight=args.invalidity_weight,
                    grad_clip=args.grad_clip,
                    act_steps=args.act_steps,
                    capture_head=capture_head,
                    capture_optimizer=capture_optimizer,
                    capture_weight=args.capture_aux_weight,
                    legal_masked=bool(args.rl_legal_mask),
                    accum_episodes=args.accum_episodes,
                    n_chunks=n_chunks,
                    adv_filter_frac=args.adv_filter_frac,
                )
                epochs_run += 1
                for k, v in metrics.items():
                    ppo_metrics_agg[k] = ppo_metrics_agg.get(k, 0.0) + v

                # Live target-KL guard: the decision must be identical on
                # every rank, so it uses the all-reduced mean approx_kl.
                if args.target_kl > 0 and _epoch < args.ppo_epochs - 1:
                    kl_t = torch.tensor(
                        [metrics.get("approx_kl", 0.0)], device=device)
                    if dist.is_initialized():
                        dist.all_reduce(kl_t)
                        kl_t /= world_size
                    if kl_t.item() > args.target_kl:
                        if rank0:
                            print(f"  PPO early stop after epoch "
                                  f"{_epoch + 1}: approx_kl={kl_t.item():.4f}"
                                  f" > {args.target_kl}")
                        break

        rl_criterion.entropy_weight = base_entropy_weight

        n_updates = max(epochs_run, 1)
        ppo_avg = {k: v / n_updates for k, v in ppo_metrics_agg.items()}
        if "entropy" in ppo_avg:
            last_policy_entropy = ppo_avg["entropy"]

        # ---- 4. IL auxiliary update ----
        il_metrics = {}
        if il_loader is not None and il_weight > 1e-6:
            try:
                il_batch = next(il_iter)
            except StopIteration:
                if il_sampler is not None:
                    il_sampler.set_epoch(iteration)
                il_iter = iter(il_loader)
                il_batch = next(il_iter)

            il_metrics = il_auxiliary_update(
                model, il_batch, il_criterion, optimizer, device,
                il_weight=il_weight,
                world_size=world_size,
                grad_clip=args.grad_clip,
                act_steps=args.act_steps,
            )
            il_weight = max(il_weight * args.il_decay, args.il_weight_min)

            # The IL forward runs in train mode: BatchNorm running stats
            # chase each rank's own IL batch, and buffers are NOT covered
            # by gradient sync. Realign so EMA/eval/pool snapshots stay
            # rank-identical (mirrors the startup broadcast).
            if dist.is_initialized():
                for b in model.buffers():
                    dist.broadcast(b, src=0)

        # ---- 4b. EMA update (once per iteration, after all optimizer
        # steps; ranks hold identical weights so their EMAs stay in sync) --
        if ema_state is not None:
            update_ema(ema_state, model, args.ema_decay)

        # Decay shaping rewards
        env.city_reward_scale *= args.city_reward_decay
        env.dense_reward_scale *= args.dense_reward_decay
        if args.sparse_reward:
            env.city_reward_scale = 0.0
            env.dense_reward_scale = 0.0

        # Decay learning rate
        for pg in optimizer.param_groups:
            pg["lr"] *= args.lr_decay
        if capture_optimizer is not None:
            for pg in capture_optimizer.param_groups:
                pg["lr"] *= args.lr_decay

        # Decay entropy weight
        rl_criterion.entropy_weight = max(
            rl_criterion.entropy_weight * args.entropy_weight_decay,
            args.entropy_weight_min,
        )

        elapsed = time.time() - t0

        # ---- 5. Aggregate stats across all GPUs ----
        total_games = args.episodes_per_gpu * world_size
        if dist.is_initialized():
            rw_keys = ["r_invalid", "r_city", "r_dense", "r_winlose",
                       "c_cap", "c_bash", "c_steal", "c_loss"]
            local_stats = torch.tensor([
                float(local_p0),
                float(local_p1),
                float(local_draw),
                float(local_mirror),
                local_ep_reward,
                local_ep_len,
            ] + [reward_totals[k] for k in rw_keys], device=device)
            dist.all_reduce(local_stats)
            global_p0 = int(local_stats[0].item())
            global_p1 = int(local_stats[1].item())
            global_draw = int(local_stats[2].item())
            global_mirror = int(local_stats[3].item())
            global_avg_r = local_stats[4].item() / world_size
            global_avg_len = local_stats[5].item() / world_size
            for i, k in enumerate(rw_keys):
                reward_totals[k] = local_stats[6 + i].item() / world_size
        else:
            global_p0 = local_p0
            global_p1 = local_p1
            global_draw = local_draw
            global_mirror = local_mirror
            global_avg_r = local_ep_reward
            global_avg_len = local_ep_len

        # ---- 6. Update opponent pool (conditional on winrate + min interval) ----
        # Pure mirror self-play (mirror_prob >= 1) never plays the pool:
        # there are no pool games, so the gate winrate is undefined and the
        # whole mechanism — including the very losing-streak-vs-frozen-
        # snapshots feedback that drove the phase-5 collapse — is off.
        if args.mirror_prob < 1.0:
            winrate_history.append(global_p0 / max(global_p0 + global_p1 + global_draw, 1))
            iters_since_update = iteration - last_pool_update_iter
            if iters_since_update >= args.pool_update_interval:
                recent_wr = sum(winrate_history) / len(winrate_history) if winrate_history else 0
                if recent_wr >= args.pool_update_winrate:
                    # Pool snapshots use the EMA when available — smoother
                    # opponents damp self-play cycling vs high-variance
                    # single-iteration snapshots.
                    opponent_pool.add(
                        ema_state if ema_state is not None else model.state_dict())
                    last_pool_update_iter = iteration
                    if rank0:
                        print(f"  Pool updated: {len(opponent_pool)} opponents "
                              f"(winrate={recent_wr:.2f}"
                              f"{', ema' if ema_state is not None else ''})")
                elif rank0:
                    print(f"  Pool update skipped: winrate={recent_wr:.2f} < {args.pool_update_winrate}")

        # ---- 6b. Fixed-opponent evaluation vs the KL anchor ----
        # Paired, seat-swapped, GREEDY: each seed is played twice with seats
        # exchanged so the P0 priority advantage cancels, actions are argmax
        # so the pair is reproducible, and won_both/lost_both separates
        # dominance from map luck. Progress signal independent of shaping
        # rewards and of the moving opponent pool.
        if (args.eval_interval > 0 and ref_model is not None
                and iteration % args.eval_interval == 0):
            rng_state = random.getstate()  # reset(seed) reseeds global RNG
            # Evaluate the EMA policy when available — it is what gets
            # deployed and pooled. opponent_model is reloaded from the pool
            # at the top of the next iteration, so borrowing it is free.
            eval_net = model
            if ema_state is not None:
                opponent_model.load_state_dict(ema_state)
                eval_net = opponent_model
            eval_net.action_decoder.greedy = True
            ref_model.action_decoder.greedy = True
            rank = dist.get_rank() if dist.is_initialized() else 0
            w = l = d = won_both = lost_both = 0
            try:
                for i in range(args.eval_games):
                    seed = 9_000_000 + iteration * 10_000 + rank * 100 + i
                    outcome = []
                    for lp in (0, 1):
                        collect_selfplay_episode(
                            eval_net, ref_model, env, device,
                            max_steps=args.rollout_steps,
                            gamma=args.gamma, gae_lambda=args.gae_lambda,
                            act_steps=args.act_steps,
                            legal_mask=bool(args.rl_legal_mask),
                            learner_pid=lp, reset_seed=seed,
                            collect_buffers=False,
                        )
                        outcome.append(
                            1 if env.winner == lp
                            else (-1 if env.winner == 1 - lp else 0))
                    w += sum(1 for o in outcome if o == 1)
                    l += sum(1 for o in outcome if o == -1)
                    d += sum(1 for o in outcome if o == 0)
                    won_both += int(outcome[0] == 1 and outcome[1] == 1)
                    lost_both += int(outcome[0] == -1 and outcome[1] == -1)
            finally:
                eval_net.action_decoder.greedy = False
                ref_model.action_decoder.greedy = False
                random.setstate(rng_state)
            if dist.is_initialized():
                ev_t = torch.tensor([w, l, d, won_both, lost_both],
                                    dtype=torch.float32, device=device)
                dist.all_reduce(ev_t)
                w, l, d, won_both, lost_both = (int(x) for x in ev_t)
            if rank0:
                n_ev = max(w + l + d, 1)
                n_pairs = n_ev // 2
                tag = "ema" if ema_state is not None else "raw"
                print(f"  Eval vs anchor (paired greedy, {tag}): {w}/{l}/{d} "
                      f"(W/L/D of {n_ev})  wr={w / n_ev:.2f}  "
                      f"pairs: won_both={won_both}/{n_pairs} "
                      f"lost_both={lost_both}/{n_pairs}")

        # ---- 7. Logging ----
        if rank0 and iteration % args.log_interval == 0:
            print(
                f"Iter {iteration:>6d} | "
                f"ep_len={global_avg_len:>6.0f}  "
                f"r={global_avg_r:+.3f}  "
                f"w/l/d={global_p0}/{global_p1}/{global_draw}"
                f"+mir{global_mirror}({total_games}) | "
                f"pol={ppo_avg.get('policy_loss', 0):.4f}  "
                f"val={ppo_avg.get('value_loss', 0):.4f}  "
                f"ent={ppo_avg.get('entropy', 0):.4f}  "
                f"kl={ppo_avg.get('approx_kl', 0):.4f}  "
                f"clip={ppo_avg.get('clip_frac', 0):.3f}  "
                f"keep={ppo_avg.get('adv_kept', 1):.2f}  "
                f"kl_reg={ppo_avg.get('kl_reg', 0):.4f}  "
                f"klw={kl_weight_t:.3f}  "
                f"klh={ppo_avg.get('kl_src', 0):.3f}/{ppo_avg.get('kl_dir', 0):.3f}/"
                f"{ppo_avg.get('kl_spl', 0):.3f}/{ppo_avg.get('kl_pass', 0):.3f}  "
                f"inv={ppo_avg.get('inv_loss', 0):.4f}  "
                f"e_p={ppo_avg.get('ent_pass', 0):.3f}  "
                f"e_s={ppo_avg.get('ent_src', 0):.3f}  "
                f"e_d={ppo_avg.get('ent_dir', 0):.3f}  "
                f"e_sp={ppo_avg.get('ent_spl', 0):.3f} | "
                f"cap={ppo_avg.get('cap_loss', 0):.3f}/"
                f"{ppo_avg.get('cap_acc', 0):.2f} | "
                f"il={il_metrics.get('il_loss', 0):.4f}  "
                f"il_w={il_weight:.4f} | "
                f"pool={len(opponent_pool)}  "
                f"t={elapsed:.1f}s\n"
                f"    rw: inv={reward_totals['r_invalid']:.3f}  "
                f"city={reward_totals['r_city']:.3f}  "
                f"dense={reward_totals['r_dense']:.3f}  "
                f"wl={reward_totals['r_winlose']:.2f} | "
                f"city/game: cap={reward_totals['c_cap']:.2f}  "
                f"bash={reward_totals['c_bash']:.2f}  "
                # Extra hits per capture — direct measure of garrison-number
                # judgement (0 = every city taken in one blow).
                f"fail={reward_totals['c_bash'] / max(reward_totals['c_bash'] + reward_totals['c_cap'], 1e-6):.2f}  "
                f"steal={reward_totals['c_steal']:.2f}  "
                f"loss={reward_totals['c_loss']:.2f}"
            )

        # ---- 7. Save game replay ----
        if replay_log is not None:
            import json as _json
            replay_path = os.path.join(
                args.output_dir, f"replay_iter{iteration}.json",
            )
            snap = replay_env_snapshot
            # Compact dump: at 2048-step episodes an indented file is
            # ~50MB written synchronously on rank0 while other ranks idle.
            with open(replay_path, "w") as fh:
                _json.dump({
                    "iteration": iteration,
                    "ep_len": len(ep_buffers[0]),
                    "total_reward_p0": sum(ep_buffers[0].rewards),
                    "winner": snap["winner"],
                    "map_pool_idx": snap["map_pool_idx"],
                    "replay_id": snap["replay_id"],
                    "map": {
                        "real_h": snap["real_h"],
                        "real_w": snap["real_w"],
                        "generals": snap["generals"],
                        "cities": snap["cities"],
                        "mountains": snap["mountains"],
                    },
                    "steps": replay_log,
                }, fh)
            print(f"  Saved replay {replay_path}")

        # ---- 8. Save checkpoint ----
        if rank0 and iteration % args.save_interval == 0:
            raw = model
            ckpt_path = os.path.join(
                args.output_dir, f"rl_checkpoint_iter{iteration}.pt",
            )
            ckpt_data = {
                "iteration": iteration,
                "model_state": raw.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg.__dict__,
                "phase_start_iter": phase_start_iter,
                "il_weight": il_weight,
                "wins": wins,
                "opponent_pool": opponent_pool.pool,
                # Shaping decay state — lets a resume continue the schedule
                # exactly instead of recomputing from the iteration counter.
                "city_reward_scale": env.city_reward_scale,
                "dense_reward_scale": env.dense_reward_scale,
            }
            if capture_head is not None:
                ckpt_data["capture_head_state"] = capture_head.state_dict()
                ckpt_data["capture_optim_state"] = capture_optimizer.state_dict()
            if ema_state is not None:
                ckpt_data["ema_state"] = ema_state
            torch.save(ckpt_data, ckpt_path + ".tmp")
            # Atomic publish: a reader (or a resume glob) must never see a
            # half-written checkpoint.
            os.replace(ckpt_path + ".tmp", ckpt_path)
            print(f"  Saved {ckpt_path}")
            if ema_state is not None:
                # Standalone strict-loadable EMA checkpoint — this is the
                # deployment/eval artifact (no optimizer state, no pool).
                # Distinct prefix so the crash-recovery resume glob
                # (rl_checkpoint_iter*.pt) can never pick it up.
                ema_path = os.path.join(
                    args.output_dir, f"rl_ema_iter{iteration}.pt")
                torch.save({
                    "iteration": iteration,
                    "model_state": ema_state,
                    "config": cfg.__dict__,
                    "ema": True,
                }, ema_path + ".tmp")
                os.replace(ema_path + ".tmp", ema_path)

    if rank0:
        print("RL training complete.")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()