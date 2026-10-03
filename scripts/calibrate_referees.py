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


def settings(E: str, nodes=(), workers: int = 1):
    """game -> [(label, factory(env) -> policy)], weakest first."""
    from env.engine_policy import BackgammonEnginePolicy, EnginePolicy, OthelloEnginePolicy

    def uci(cmd, go, **opt):
        return lambda env: EnginePolicy(env, cmd, go=go, options=opt or None, workers=workers)

    kg = (f"{E}/bin/katago gtp -model {E}/nets/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz "
          f"-human-model {E}/nets/katago/b18c384nbt-humanv0.bin.gz "
          f"-config {E}/katago/v1.18.1-eigenavx2/gtp_human5k_example.cfg -override-config ")
    kg_opt = "maxVisits=1,logDir=,delayMoveScale=0,delayMoveMax=0,allowResignation=false,numSearchThreads=1"
    small = {"b10": "kata1-b10c128-s1141046784-d204142634.txt.gz", "b6": "kata1-b6c96-s95425792-d14091095.txt.gz"}

    def kgs(n):
        return (f"{E}/bin/katago gtp -model {E}/nets/katago/{small[n]} "
                f"-config {E}/katago/v1.18.1-eigenavx2/default_gtp.cfg -override-config "
                "maxVisits=1,logDir=,numSearchThreads=1,allowResignation=false,ponderingEnabled=false")
    sf, pk, yo = f"{E}/bin/stockfish", f"{E}/bin/pikafish", f"{E}/bin/yaneuraou"
    yo_opt = dict(Threads="1", USI_Hash="64")
    def noisy(make, eps):
        if eps <= 0:
            return make
        from env.engine_policy import NoisyReferee
        return lambda env: NoisyReferee(make(env), env, eps, seed=7)

    def sfe(e):
        return uci(sf, "go nodes 20000", UCI_LimitStrength="true", UCI_Elo=str(e), Threads="1", Hash="16")

    ot = lambda lv: (lambda env: OthelloEnginePolicy(env, f"{E}/bin/egaroucid", level=lv, workers=workers))
    if nodes:       # --nodes N[@eps],...: a finer grid for the node-limited engines only
        return {"xiangqi": [(f"pikafish nodes {n} eps {p}",
                             noisy(uci(pk, f"go nodes {n}", Threads="1", Hash="16"), p)) for n, p in nodes],
                "shogi": [(f"yaneuraou nodes {n} eps {p}", noisy(uci(yo, f"go nodes {n}", **yo_opt), p))
                          for n, p in nodes]}
    return {
        "chess": [(f"stockfish elo 1320 eps {p}", noisy(sfe(1320), p)) for p in (0.5, 0.25)]
                 + [(f"stockfish elo {e}", sfe(e)) for e in (1320, 1800)],
        "xiangqi": [(f"pikafish nodes 1 eps {p}", noisy(uci(pk, "go nodes 1", Threads="1", Hash="16"), p))
                    for p in (0.5, 0.25)]
                   + [(f"pikafish nodes {n}", uci(pk, f"go nodes {n}", Threads="1", Hash="16"))
                      for n in (1, 100)],
        "shogi": [(f"yaneuraou nodes 1 eps {p}", noisy(uci(yo, "go nodes 1", **yo_opt), p))
                  for p in (0.5, 0.25)]
                 + [(f"yaneuraou nodes {n}", uci(yo, f"go nodes {n}", **yo_opt)) for n in (1, 100)],
        "othello": [(f"egaroucid level 1 eps {p}", noisy(ot(1), p)) for p in (0.5, 0.25)]
                   + [("egaroucid level 1", ot(1))],
        # KataGo's raw policy (1 visit) of a small network: ~0.06 s a move on one CPU thread
        # (the b18 human net: ~1.7 s, a 32-game evaluation ~45 min)
        "go": [(f"katago {n} v1 eps {p}", noisy(
                    lambda env, n=n: EnginePolicy(env, kgs(n), workers=workers), p))
               for n, p in (("b10", 0.0), ("b10", 0.25), ("b10", 0.5), ("b6", 0.0))],
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
    ap.add_argument("--workers", type=int, default=4, help="engine processes per referee")
    ap.add_argument("--nodes", default="", help="N[@eps],...: node limits (and noise) for xiangqi / shogi only")
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
    grid = [(int(x.split("@")[0]), float(x.split("@")[1]) if "@" in x else 0.0)
            for x in a.nodes.split(",") if x]
    table = settings(a.engines, grid, a.workers)
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
