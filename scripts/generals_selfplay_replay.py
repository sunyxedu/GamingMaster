"""Watch a generals.io network play itself: one 1v1 game on the RL environment
(env/vec_generals.py, the same rules and observations as training), both seats driven
by the network, until one general is captured (or ``--max_turns``), written as the
replay JSON of scripts/view_rl_replay.py and, with ``--html``, as its HTML viewer.

    uv run python scripts/generals_selfplay_replay.py --ckpt generals.pt --map_dir REPLAYS --html game.html
    # --ckpt: a train_multi_game checkpoint (its generals network, EMA weights) or a
    # single-game one (scripts/export_game_checkpoint.py)

Each seat is its own temporal stream (its own fog-of-war view and Stage-1 state), the
actions sampled from the policy (``--greedy``: its argmax), the environment's
observation options taken from the network (cfg.memory_vision). The replay records the
fog-free board after every turn, both seats' moves, land and army.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.league import NetworkPolicy  # noqa: E402
from env.vec_game import make_vec_env  # noqa: E402
from training.arena import load_policy_network, pick_device  # noqa: E402


def _action_of(applied, row):
    """A seat's applied action in the viewer's format (None = pass)."""
    if applied is None:
        return None
    if bool(applied["passed"][row]):
        return None
    return {"source_idx": int(applied["source_idx"][row]), "direction": int(applied["direction"][row]),
            "split": int(applied["split"][row])}


def play(net, *, seed: int, max_turns: int, greedy: bool, device, map_dir=None) -> dict:
    mv = bool(getattr(net.cfg, "memory_vision", False))
    env = make_vec_env("generals", n_envs=1, max_turns=max_turns, memory_vision=mv, seed=seed, map_dir=map_dir)
    policy = NetworkPolicy(net, device=device, greedy=greedy)
    obs = env.reset(seed=seed)
    g = env.envs[0]

    def snap(turn, a0=None, a1=None, illegal=(False, False), done=False, winner=None):
        s = g.get_state_snapshot()
        return {"turn": turn, "p0_action": a0, "p1_action": a1, "p0_reward": 0.0, "p1_reward": 0.0,
                "illegal": list(illegal), "p0_stats": g._player_stats(0), "p1_stats": g._player_stats(1),
                "state": s["state"], "army": s["army"], "done": done, "winner": winner}

    steps = [snap(-1)]
    rows = torch.arange(obs.n_rows)
    turn = 0
    t0 = time.time()
    while True:
        with torch.no_grad():
            act = policy(obs, rows)
        obs, reward, done, info = env.step(act)
        applied = info.get("applied")
        winner = int(info["winner"][0]) if bool(done[0]) else None
        ill = info.get("illegal")
        illegal = (bool(ill[0]), bool(ill[1])) if ill is not None else (False, False)
        steps.append(snap(turn, _action_of(applied, 0), _action_of(applied, 1), illegal, bool(done[0]),
                          None if winner is None or winner < 0 else winner))
        turn += 1
        if turn % 50 == 0:
            st = steps[-1]
            print(f"turn {turn}: land {st['p0_stats']['land']} vs {st['p1_stats']['land']}, army "
                  f"{st['p0_stats']['army']} vs {st['p1_stats']['army']} ({time.time() - t0:.0f} s)", flush=True)
        if bool(done[0]):
            break
    w = steps[-1]["winner"]
    rid = g.map_pool[g.current_map_idx]["replay_id"] if g.map_pool and g.current_map_idx is not None else None
    return {"map": {"real_h": g.real_h, "real_w": g.real_w, "replay_id": rid}, "steps": steps, "ep_len": turn,
            "winner": w, "total_reward_p0": 0.0 if w is None else (1.0 if w == 0 else -1.0),
            "seed": seed, "greedy": greedy}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="generals_selfplay.json")
    p.add_argument("--html", help="also write the HTML viewer here")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max_turns", type=int, default=2048)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--map_dir", help="real maps: a directory of generals.io replay JSONs (one is drawn "
                                    "by --seed); default: a generated map")
    p.add_argument("--device", default=None, help="cpu / cuda / mps (default: the best available)")
    a = p.parse_args(argv)
    device = pick_device(a.device)
    net, info = load_policy_network(a.ckpt, "generals", device)
    net = net.float()
    print(f"loaded {a.ckpt} ({info}) on {device}", flush=True)
    rep = play(net, seed=a.seed, max_turns=a.max_turns, greedy=a.greedy, device=device, map_dir=a.map_dir)
    with open(a.out, "w") as f:
        json.dump(rep, f)
    print(f"map {rep['map']} — {rep['ep_len']} turns, winner: {rep['winner']} -> {a.out}")
    if a.html:
        from scripts.view_rl_replay import generate_html
        generate_html(rep, a.html)
        print(f"viewer: {a.html}")


if __name__ == "__main__":
    main()
