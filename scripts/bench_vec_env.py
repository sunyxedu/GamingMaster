"""Throughput of a VecGameEnv adapter (env/vec_game.py) under random play.

    uv run python scripts/bench_vec_env.py --game gomoku --envs 1024 --steps 2000
    uv run python scripts/bench_vec_env.py --game gomoku --env_kw board_size=15
    uv run python scripts/bench_vec_env.py --game othello --envs 4096

Plays uniformly random legal moves with ``autoreset`` (the batch never thins
out) and reports env steps/s (one step = one move of one env = one frame of
its stream) for ``env.step`` alone — observation building included, since
``step`` returns the next frame — and for the whole loop with the env's own
``random_actions`` (uniform over the legal mask: othello a torch
multinomial, gomoku an inverse CDF).
"""

from __future__ import annotations

import argparse
import ast
import time

import torch

from env.vec_game import make_vec_env


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--game", default="gomoku")
    p.add_argument("--envs", type=int, nargs="+", default=[64, 256, 1024, 4096])
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    p.add_argument("--env_kw", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    kw = {}
    for it in a.env_kw:
        k, _, v = it.partition("=")
        try:
            kw[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            kw[k] = v
    for n in a.envs:
        env = make_vec_env(a.game, n, autoreset=True, seed=0, **kw)
        gen = torch.Generator().manual_seed(0)
        obs = env.reset()
        for _ in range(a.warmup):
            obs, *_ = env.step(env.random_actions(obs, generator=gen))
        t_env = t_act = 0.0
        episodes = 0
        for _ in range(a.steps):
            t0 = time.perf_counter()
            act = env.random_actions(obs, generator=gen)
            t1 = time.perf_counter()
            obs, _, done, _ = env.step(act)
            t2 = time.perf_counter()
            t_act += t1 - t0
            t_env += t2 - t1
            episodes += int(done.sum())
        frames = n * a.steps
        print(f"{a.game} envs={n:5d}  step: {frames / t_env / 1e6:6.3f} M steps/s  "
              f"step+random_actions: {frames / (t_env + t_act) / 1e6:6.3f} M steps/s  "
              f"({a.steps} steps, {episodes} episodes, {t_env / a.steps * 1e3:.3f} ms/step)")


if __name__ == "__main__":
    main()
