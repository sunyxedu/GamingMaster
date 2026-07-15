"""Frozen-reference Elo ladder — cross-phase progress measurement.

Winrate against a single anchor saturates toward 1.0 as training
progresses and is blind to regressions against earlier milestones. This
script plays paired, seat-swapped, GREEDY matches (the same protocol as
train_rl's in-loop eval) between candidate checkpoints and a directory of
frozen references, caches every pairing's head-to-head record in a JSON
file, and fits Elo over the merged record (repeated low-K online updates
over shuffled games — converges to the same ordering as the MLE fit).

Usage:
    python scripts/elo_ladder.py \
        --refs_dir /scratch/.../refs \
        --candidates ckpt_a.pt ckpt_b.pt \
        --h2h /scratch/.../refs/h2h.json \
        --games 32 [--play_refs] [--device cuda]

    --play_refs also (re)plays missing ref-vs-ref pairings into the cache
    (run once when the ref set changes). Elo numbers are comparable only
    while the reference set stays frozen.

References: name each file refs/<name>.pt; the EMA deployment artifacts
(rl_ema_iterN.pt) are ideal refs. Everything strict-loads through
load_model_state.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import zlib

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.generals_env import GeneralsEnv                     # noqa: E402
from model.config import ModelConfig                         # noqa: E402
from model.network import GeneralsNetwork                    # noqa: E402
from model.rl_utils import load_model_state                  # noqa: E402
from train_rl import collect_selfplay_episode                # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-reference Elo ladder")
    p.add_argument("--refs_dir", type=str, required=True,
                   help="Directory of frozen reference checkpoints (*.pt)")
    p.add_argument("--candidates", type=str, nargs="*", default=[],
                   help="Candidate checkpoints to rate against the refs")
    p.add_argument("--h2h", type=str, default=None,
                   help="Head-to-head cache JSON (default: refs_dir/h2h.json)")
    p.add_argument("--games", type=int, default=32,
                   help="Games per pairing (played as games/2 seat-swapped "
                        "seed pairs)")
    p.add_argument("--play_refs", action="store_true",
                   help="Also play missing ref-vs-ref pairings (round robin)")
    p.add_argument("--rollout_steps", type=int, default=2048)
    p.add_argument("--act_steps", type=int, default=1)
    p.add_argument("--seed_base", type=int, default=77_000_000)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--replay_dir", type=str, default=None)
    p.add_argument("--filtered_file", type=str, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# Elo fit
# ----------------------------------------------------------------------

def fit_elo(records, init=1500.0, k=4.0, passes=200, seed=0):
    """Batch Elo from head-to-head records.

    records: list of {"a", "b", "wins_a", "wins_b", "draws"}.
    Expands to individual games (draw = 0.5), shuffles, and runs many
    low-K online passes — a simple, deterministic approximation of the
    maximum-likelihood Elo fit.
    """
    games = []
    players = set()
    for r in records:
        players.update((r["a"], r["b"]))
        games += [(r["a"], r["b"], 1.0)] * int(r["wins_a"])
        games += [(r["a"], r["b"], 0.0)] * int(r["wins_b"])
        games += [(r["a"], r["b"], 0.5)] * int(r["draws"])
    elo = {p: init for p in players}
    rng = random.Random(seed)
    for _ in range(passes):
        rng.shuffle(games)
        for a, b, s in games:
            ea = 1.0 / (1.0 + 10 ** ((elo[b] - elo[a]) / 400.0))
            elo[a] += k * (s - ea)
            elo[b] -= k * (s - ea)
    return elo


# ----------------------------------------------------------------------
# Match harness
# ----------------------------------------------------------------------

def load_net(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**{k: v for k, v in ckpt["config"].items()
                         if k in ModelConfig.__dataclass_fields__})
    net = GeneralsNetwork(cfg).to(device)
    state_key = "model_state" if "model_state" in ckpt else "model_state_dict"
    load_model_state(net, ckpt[state_key], ctx=path)
    net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net, cfg


def play_pairing(net_a, net_b, env, device, games, seed_base, rollout_steps,
                 act_steps):
    """Paired seat-swapped greedy games; returns (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0
    net_a.action_decoder.greedy = True
    net_b.action_decoder.greedy = True
    try:
        for i in range(max(1, games // 2)):
            seed = seed_base + i
            for lp in (0, 1):
                collect_selfplay_episode(
                    net_a, net_b, env, device,
                    max_steps=rollout_steps,
                    record=False, act_steps=act_steps,
                    legal_mask=True, learner_pid=lp, reset_seed=seed,
                )
                if env.winner == lp:
                    wins_a += 1
                elif env.winner == 1 - lp:
                    wins_b += 1
                else:
                    draws += 1
    finally:
        net_a.action_decoder.greedy = False
        net_b.action_decoder.greedy = False
    return wins_a, wins_b, draws


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    h2h_path = args.h2h or os.path.join(args.refs_dir, "h2h.json")
    records = []
    if os.path.exists(h2h_path):
        records = json.load(open(h2h_path))
    have = {tuple(sorted((r["a"], r["b"]))) for r in records}

    refs = sorted(glob.glob(os.path.join(args.refs_dir, "*.pt")))
    if not refs:
        raise SystemExit(f"no reference checkpoints in {args.refs_dir}")
    name = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa: E731

    paths = {name(p): p for p in refs}
    ref_names = list(paths)
    cand_names = []
    for c in args.candidates:
        paths[name(c)] = c
        cand_names.append(name(c))

    pairings = []
    if args.play_refs:
        for i, a in enumerate(ref_names):
            for b in ref_names[i + 1:]:
                if tuple(sorted((a, b))) not in have:
                    pairings.append((a, b))
    for c in cand_names:
        for r in ref_names:
            if tuple(sorted((c, r))) not in have:
                pairings.append((c, r))

    env = None
    nets = {}

    def get_net(n):
        if n not in nets:
            nets[n] = load_net(paths[n], device)
        return nets[n]

    for a, b in pairings:
        net_a, cfg = get_net(a)
        net_b, _ = get_net(b)
        if env is None:
            env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w,
                              max_turns=args.rollout_steps)
            if args.replay_dir and args.filtered_file:
                n_maps = env.load_map_pool(args.replay_dir,
                                           args.filtered_file)
                print(f"Loaded {n_maps} replay maps")
        # crc32, not hash(): per-pair seeds must be stable across runs
        # (PYTHONHASHSEED randomizes str hashing).
        pair_id = zlib.crc32(f"{a}|{b}".encode()) % 1_000_000
        seed_base = args.seed_base + pair_id * 10_000
        wa, wb, dr = play_pairing(net_a, net_b, env, device, args.games,
                                  seed_base, args.rollout_steps,
                                  args.act_steps)
        rec = {"a": a, "b": b, "wins_a": wa, "wins_b": wb, "draws": dr}
        records.append(rec)
        print(f"{a} vs {b}: {wa}-{wb}-{dr}")
        json.dump(records, open(h2h_path + ".tmp", "w"), indent=1)
        os.replace(h2h_path + ".tmp", h2h_path)

    if not records:
        raise SystemExit("no head-to-head records to fit")
    elo = fit_elo(records)
    best_ref = max((elo[r] for r in ref_names if r in elo), default=1500.0)
    print(f"\n{'model':40s} {'elo':>7s} {'vs_best_ref':>12s}")
    for n, e in sorted(elo.items(), key=lambda kv: -kv[1]):
        tag = " (ref)" if n in ref_names else ""
        print(f"{n + tag:40s} {e:7.1f} {e - best_ref:+12.1f}")


if __name__ == "__main__":
    main()
