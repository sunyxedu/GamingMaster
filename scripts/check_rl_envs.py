"""Check every game's RL environment with a trained model before RL: the model plays itself
(every seat, sampled actions, per-stream Stage-1 state — how train_rl_joint's actor plays)
in N lockstep envs per game until every episode ends, and each game gets a report:

    episodes finished / truncated, episode length (env steps; min / median / max) next to
    the IL data's record length when --data is given, outcomes (seat-0 score, draws, the
    winner per seat: a strong first-mover bias shows), illegal actions the env reported,
    value predictions (range, NaN / inf), step time.

    uv run python scripts/check_rl_envs.py --ckpt main-all-ema-bf16.pt --games chess,go \\
        --envs 8 --out report.json [--env_kw generals:map_dir=REPLAYS] [--data chess=DIR]

--ckpt is a train_multi_game checkpoint (every game's network, EMA weights) or a single-game
one (--games must then be that game). Anything odd is listed under "flags".
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import statistics
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.league import NetworkPolicy  # noqa: E402
from env.vec_game import make_vec_env  # noqa: E402
from training.arena import pick_device  # noqa: E402


def _kw(pairs):
    out = {}
    for p in pairs:
        g, _, kv = p.partition(":")
        k, _, v = kv.partition("=")
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            v = os.path.expanduser(v)
        out.setdefault(g, {})[k] = v
    return out


def _il_lengths(path, game):
    """Median / p90 record length (frames) of an IL shard directory."""
    import glob
    lens = []
    for ix in glob.glob(os.path.join(path, "**", "index.json"), recursive=True)[:50]:
        with open(ix) as f:
            lens += [s["length"] for s in json.load(f)["samples"]]
    if not lens:
        return None
    lens.sort()
    return {"records": len(lens), "median": lens[len(lens) // 2], "p90": lens[int(len(lens) * 0.9)]}


def check(game, net, device, n_envs, max_steps, env_kw, seed):
    env = make_vec_env(game, n_envs=n_envs, seed=seed, **env_kw)
    pol = NetworkPolicy(net, device=device)
    obs = env.reset(seed=seed)
    n, S = env.num_envs, env.n_seats
    ep_len = [0] * n
    done_env = [False] * n
    lengths, outcomes, winners, truncated, illegal = [], [], [], 0, 0
    values, nonfinite, step_times = [], 0, []
    steps = 0
    while not all(done_env) and steps < max_steps:
        rows = torch.nonzero(obs.active).reshape(-1)
        if len(rows) == 0:
            act = {}
        t0 = time.perf_counter()
        with torch.no_grad():
            out = pol(obs, rows) if len(rows) else {}
        step_times.append(time.perf_counter() - t0)
        full = {}
        for k, v in out.items():
            if not torch.is_tensor(v) or v.shape[:1] != rows.shape:
                continue
            buf = torch.zeros((obs.n_rows,) + tuple(v.shape[1:]), dtype=v.dtype)
            buf[rows] = v.cpu()
            full[k] = buf
        if "value" in out:
            v = out["value"].float().reshape(-1)
            nonfinite += int((~torch.isfinite(v)).sum())
            values += v[torch.isfinite(v)].tolist()[:64]
        obs, reward, done, info = env.step(full)
        steps += 1
        ill = info.get("illegal")
        if ill is not None:
            illegal += int(torch.as_tensor(ill).sum())
        dn = torch.as_tensor(done).reshape(-1)
        for i in range(n):
            if done_env[i]:
                continue
            ep_len[i] += 1
            if bool(dn[i]):
                done_env[i] = True
                lengths.append(ep_len[i])
                r = torch.as_tensor(reward, dtype=torch.float32).reshape(n, -1)[i].tolist()
                outcomes.append(r)
                w = info.get("winner")
                winners.append(int(torch.as_tensor(w).reshape(-1)[i]) if w is not None else None)
                tr = info.get("truncated")
                truncated += bool(tr is not None and bool(torch.as_tensor(tr).reshape(-1)[i]))
    rep = {"game": game, "envs": n, "seats": S, "env_steps": steps, "finished": len(lengths),
           "truncated": truncated, "illegal_actions": illegal,
           "episode_len": {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)} if lengths else None,
           "seat0_return_mean": round(statistics.mean(o[0] for o in outcomes), 3) if outcomes else None,
           "winner_counts": {str(w): winners.count(w) for w in sorted(set(winners), key=lambda x: (x is None, x))},
           "value": {"min": round(min(values), 3), "max": round(max(values), 3)} if values else None,
           "value_nonfinite": nonfinite,
           "ms_per_step": round(1000 * statistics.median(step_times), 1) if step_times else None}
    flags = []
    if len(lengths) < n:
        flags.append(f"only {len(lengths)}/{n} episodes ended within {max_steps} steps")
    if truncated:
        flags.append(f"{truncated} episodes truncated")
    if illegal:
        flags.append(f"{illegal} illegal actions")
    if nonfinite:
        flags.append(f"{nonfinite} non-finite values")
    if len(winners) >= 4 and S == 2:
        c0, c1 = winners.count(0), winners.count(1)
        if max(c0, c1) >= 0.85 * len(winners) and min(c0, c1) <= 1:
            flags.append(f"one seat wins almost everything ({c0} vs {c1})")
    rep["flags"] = flags
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", required=True)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--env_kw", action="append", default=[])
    p.add_argument("--data", action="append", default=[], help="GAME=DIR: the IL records' length for comparison")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="rl_env_check.json")
    a = p.parse_args(argv)
    device = pick_device(a.device)
    kw = _kw(a.env_kw)
    data = dict(d.split("=", 1) for d in a.data)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False, mmap=True)
    if "games" in ck:
        from training.eval_multi_game import load_multi
        del ck
        model = load_multi(a.ckpt, device)
        get = lambda g: model[g]          # noqa: E731
    else:
        from training.arena import load_policy_network
        del ck
        get = lambda g: load_policy_network(a.ckpt, g, device)[0]   # noqa: E731
    reports = []
    for g in a.games.split(","):
        t0 = time.time()
        try:
            net = get(g).float().eval()
            rep = check(g, net, device, a.envs, a.max_steps, kw.get(g, {}), a.seed)
        except Exception as e:           # noqa: BLE001 — report every game
            rep = {"game": g, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:],
                   "flags": ["ERROR"]}
        if g in data:
            rep["il_record_len"] = _il_lengths(data[g], g)
        rep["seconds"] = round(time.time() - t0, 1)
        reports.append(rep)
        print(json.dumps({k: v for k, v in rep.items() if k != "trace"}), flush=True)
        with open(a.out, "w") as f:
            json.dump(reports, f, indent=1)


if __name__ == "__main__":
    main()
