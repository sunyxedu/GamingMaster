"""Smoke run of every game's RL environment: N envs, uniformly random legal actions
(the env's own ``random_actions``), until E episodes finished or a step budget ran out.
Prints per game: episodes finished, steps, frames/s, reward range — the evidence that
each simulator runs end to end in the RL interface (env/vec_game.py).

    uv run python scripts/smoke_all_envs.py [--games a,b] [--envs 4] [--episodes 4]

Games needing an external game binary (sc2: the StarCraft II client; hok: the
Honor of Kings gamecore) are skipped where it is missing (reported as such).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from env.vec_game import VEC_ENV_REGISTRY, make_vec_env

TRAINING_GAMES = ["generals", "chess", "go", "shogi", "othello", "mahjong", "lux", "clash_royale", "gomoku",
                  "xiangqi", "chess_variants", "bridge", "skat", "doudizhu", "hunl", "backgammon", "pokemon",
                  "sc2", "arc3", "hok", "tetris"]


def smoke(game: str, n_envs: int, episodes: int, max_steps: int):
    env = make_vec_env(game, n_envs=n_envs, autoreset=True) if "autoreset" in make_vec_env.__code__.co_varnames \
        else make_vec_env(game, n_envs=n_envs)
    g = torch.Generator().manual_seed(0)
    obs = env.reset()
    done_eps, steps, frames, rmin, rmax = 0, 0, 0, 0.0, 0.0
    t0 = time.time()
    while done_eps < episodes and steps < max_steps:
        act = env.random_actions(obs, generator=g)
        obs, reward, done, info = env.step(act)
        steps += 1
        frames += int(obs.active.sum()) if getattr(obs, "active", None) is not None else n_envs
        r = torch.as_tensor(reward, dtype=torch.float32)
        rmin, rmax = min(rmin, float(r.min())), max(rmax, float(r.max()))
        done_eps += int(torch.as_tensor(done).sum())
        if bool(torch.as_tensor(done).all()) and not getattr(env, "autoreset", True):
            obs = env.reset()
    return done_eps, steps, frames / max(time.time() - t0, 1e-9), rmin, rmax


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--games", default=",".join(TRAINING_GAMES))
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=20000)
    a = p.parse_args(argv)
    bad = 0
    for game in a.games.split(","):
        if game not in VEC_ENV_REGISTRY:
            print(f"{game:15s} NO ENVIRONMENT")
            bad += 1
            continue
        try:
            n, s, fps, lo, hi = smoke(game, a.envs, a.episodes, a.max_steps)
            ok = n > 0
            bad += not ok
            print(f"{game:15s} {'ok ' if ok else 'NO EPISODE END'} episodes={n} steps={s} frames/s={fps:,.0f} reward=[{lo:.2f}, {hi:.2f}]",
                  flush=True)
        except Exception as e:                      # noqa: BLE001 — report every game
            msg = str(e).splitlines()[0][:160]
            missing = any(k in msg.lower() for k in ("sc2path", "starcraft", "gamecore", "license", "no such file", "not found"))
            print(f"{game:15s} {'SKIPPED (external binary missing)' if missing else 'ERROR'}: {type(e).__name__}: {msg}", flush=True)
            bad += not missing
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
