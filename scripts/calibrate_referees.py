"""Calibrate the RL engine referees against a policy: the score of ``--init`` (the IL network,
greedy=False as in the RL evaluations) against each engine setting, through the trainer's own
evaluation (JointRLTrainer._eval_against: seats drawn per episode, whole waves).  A referee
is useful when the policy scores well inside (0, 1) against it — 0.00 against every setting
of the 2026-10-03 bench (Stockfish / Pikafish / YaneuraOu at 2000 nodes, Egaroucid level 5)
measures nothing.

    uv run python scripts/calibrate_referees.py --init CKPT --games chess,othello \\
        [--episodes 32] [--device cpu] [--threads 32] [--engines ~/gm-engines]

Prints one line per (game, setting): score, episodes, seconds.  The settings are the
SETTINGS table below (engine command, go limit, options per game) — weakest first."""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def settings(E: str):
    """game -> [(label, factory(env) -> policy)], weakest first."""
    from env.engine_policy import BackgammonEnginePolicy, EnginePolicy, OthelloEnginePolicy

    def uci(cmd, go, **opt):
        return lambda env: EnginePolicy(env, cmd, go=go, options=opt or None)

    kg = (f"{E}/bin/katago gtp -model {E}/nets/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz "
          f"-human-model {E}/nets/katago/b18c384nbt-humanv0.bin.gz "
          f"-config {E}/katago/v1.18.1-eigenavx2/gtp_human5k_example.cfg -override-config ")
    kg_opt = "maxVisits=1,logDir=,delayMoveScale=0,delayMoveMax=0,allowResignation=false,numSearchThreads=1"
    sf, pk, yo = f"{E}/bin/stockfish", f"{E}/bin/pikafish", f"{E}/bin/yaneuraou"
    yo_opt = dict(Threads="1", USI_Hash="64")
    return {
        "chess": [(f"stockfish elo {e}", uci(sf, "go nodes 20000", UCI_LimitStrength="true",
                                              UCI_Elo=str(e), Threads="1", Hash="16"))
                  for e in (1320, 1600, 2000, 2400)],
        "chess_variants": [(f"stockfish elo {e}", uci(sf, "go nodes 20000", UCI_LimitStrength="true",
                                                       UCI_Elo=str(e), Threads="1", Hash="16"))
                           for e in (1320, 1600, 2000)],
        "xiangqi": [(f"pikafish nodes {n}", uci(pk, f"go nodes {n}", Threads="1", Hash="16"))
                    for n in (1, 10, 100, 1000)],
        "shogi": [(f"yaneuraou nodes {n}", uci(yo, f"go nodes {n}", **yo_opt))
                  for n in (1, 10, 100, 1000)],
        "othello": [(f"egaroucid level {lv}", lambda env, lv=lv: OthelloEnginePolicy(env, f"{E}/bin/egaroucid", level=lv))
                    for lv in (0, 1, 2, 3)],
        "go": [(f"katago human {p}", lambda env, p=p: EnginePolicy(env, kg + f"humanSLProfile={p}," + kg_opt))
               for p in ("rank_10k", "rank_5k", "rank_1d", "rank_5d")],
        "backgammon": [(f"gnubg plies {p}", lambda env, p=p: BackgammonEnginePolicy(env, None, plies=p))
                       for p in (0,)],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--init", required=True)
    ap.add_argument("--games", required=True, help="comma-separated")
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--engines", default=os.path.expanduser("~/gm-engines"))
    ap.add_argument("--only", default="", help="substring filter on the setting labels")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from training.train_rl_joint import JointRLTrainer, parse_args

    games = [g for g in a.games.split(",") if g]
    argv = ["--init", a.init, "--device", a.device, "--eval_episodes", str(a.episodes),
            "--seed", str(a.seed), "--save_dir", os.path.join("/tmp", f"calib-{os.getpid()}")]
    for g in games:
        argv += ["--rl", g, "--envs", f"{g}={a.envs}"]
    t0 = time.time()
    tr = JointRLTrainer(parse_args(argv))
    print(f"trainer up in {time.time() - t0:.0f} s ({a.init})", flush=True)
    table = settings(a.engines)
    for gi, g in enumerate(tr.rl_games):
        for label, make in table.get(g, []):
            if a.only and a.only not in label:
                continue
            t = time.time()
            try:
                pol = make(tr.rl[g].env)
            except Exception as e:  # noqa: BLE001
                print(f"{g:10s} {label:28s} FAILED to start: {type(e).__name__}: {e}", flush=True)
                continue
            try:
                st = tr._eval_against(g, tr.rl[g], 0, gi, "engine", pol)
                score = st.score_sum / max(st.score_n, 1)
                print(f"{g:10s} {label:28s} score {score:.3f}  n {st.score_n:3d}  "
                      f"{time.time() - t:6.0f} s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{g:10s} {label:28s} FAILED: {type(e).__name__}: {e}", flush=True)
            finally:
                try:
                    pol.close()
                except Exception:  # noqa: BLE001
                    pass


if __name__ == "__main__":
    main()
