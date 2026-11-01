"""Self-play showcase: a few games per game with the current weights on both seats, written as
readable records (move strings and board / state text from the rules engine, the model's own
value estimate per decision, the Showdown log for Pokémon, generals' map snapshots).

    cd ~/gm-check/<snap> && .venv/bin/python ~/showcase.py --ckpt ~/gm-runs/rl/final-mem/rl_latest.pt \\
        --out ~/gm-runs/rl/final-mem/showcase --n 2 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402

GAMES = ["chess", "chess_variants", "xiangqi", "shogi", "go", "othello", "gomoku", "backgammon",
         "hunl", "bridge", "skat", "doudizhu", "mahjong", "generals", "lux", "clash_royale",
         "pokemon", "tetris", "sc2", "arc3"]
ENV_KW = ["generals:map_dir=/home/ys2924/gm-data/generals-ladder2/replays", "generals:max_maps=400",
          "arc3:max_actions=2048", "sc2:difficulty=very_easy"]


def plain(x):
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x


def replay_gmrules(game, traj, keep_states):
    import gmsim._native as gm
    st = gm.State(game)
    moves, states, chance = [], [], 0
    for seat, a in np.asarray(traj).reshape(-1, 2).tolist():
        if seat < 0:
            st.apply_chance(int(a))
            chance += 1
            continue
        moves.append([int(seat), st.action_string(int(seat), int(a))])
        st.apply_action(int(seat), int(a))
        if keep_states:
            states.append(str(st))
    return {"moves": moves, "chance_events": chance, "final_state": str(st),
            "states": states if keep_states else None,
            "returns": plain(st.returns()) if hasattr(st, "returns") else None}


def one_game(game, env, actor, k, max_steps):
    from env.league import League
    from training.train_rl_joint import TIMEOUT_IS_DRAW, TIMEOUT_REWARD, collect_wave
    league = League(env, {}, mirror_prob=1.0, seed=100 + k)
    trace = []
    t0 = time.time()
    streams, st = collect_wave(env, league, actor, seed=1000 + k, max_steps=max_steps,
                               timeout_reward=TIMEOUT_REWARD.get(game), timeout_draw=game in TIMEOUT_IS_DRAW,
                               trace=trace)
    rec = {"game": game, "index": k, "seconds": round(time.time() - t0, 1), "steps": len(trace),
           "winner": trace[-1].get("winner") if trace else None,
           "truncated": trace[-1].get("truncated") if trace else None,
           "final_rewards": trace[-1].get("rewards") if trace else None,
           "values": [{"seat": plain(s.seat), "value": plain(s.value)} for s in streams]}
    if callable(getattr(env, "trajectory", None)):
        traj = np.asarray(env.trajectory(0))
        rec["trajectory"] = traj.tolist()
        rec.update(replay_gmrules(game, traj, keep_states=(traj.shape[0] <= 400)))
    if game == "pokemon":
        rec["log"] = list(env.logs[0])
    if game == "generals":
        rec["snapshots"] = [s.get("state") for s in trace[::10]] + [trace[-1].get("state")]
    if game == "othello":
        rec["board"] = {"black": int(env.black[0]), "white": int(env.white[0])}
    if game == "mahjong" and hasattr(env, "scores"):
        try:
            rec["scores"] = plain(env.scores(0))
        except Exception:
            pass
    rec["trace"] = [{k_: v for k_, v in s.items() if k_ != "state"} for s in trace]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max_steps", type=int, default=20000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from env.vec_game import make_vec_env
    from training.eval_multi_game import load_multi
    from training.train_rl_joint import GAME_SPECS, LearnerActor, _env_kwargs
    device = torch.device(a.device)
    model = load_multi(a.ckpt, device)
    kws = _env_kwargs(ENV_KW)
    summary = []
    for game in a.games.split(","):
        try:
            net = model[game].to(device).eval()
            spec = GAME_SPECS[game]
            env = make_vec_env(game, 1, seed=7, **kws.get(game, {}))
            if hasattr(spec, "check_env"):
                spec.check_env(env, net)
            actor = LearnerActor(net, spec, net.default_depth, device, bf16=True, mode="stream")
            for k in range(a.n):
                rec = one_game(game, env, actor, k, a.max_steps)
                with open(os.path.join(a.out, f"{game}_{k}.json"), "w") as f:
                    json.dump(rec, f, default=str)
                summary.append({k_: rec[k_] for k_ in ("game", "index", "seconds", "steps", "winner", "truncated",
                                                       "final_rewards")})
                print(f"[{game} {k}] steps={rec['steps']} winner={rec['winner']} rewards={rec['final_rewards']} "
                      f"{rec['seconds']}s", flush=True)
            try:
                env.close()
            except Exception:
                pass
        except Exception as e:   # noqa: BLE001
            print(f"[{game}] FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            summary.append({"game": game, "error": f"{type(e).__name__}: {e}"})
        with open(os.path.join(a.out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)


if __name__ == "__main__":
    main()
