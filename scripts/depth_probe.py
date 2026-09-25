"""Depth probe: does the SAME checkpoint play stronger with more TRM
pondering steps?

Pits one set of weights against itself with different act_steps per
seat, paired (seat-swapped) on shared map seeds so map/seat luck cancels.
The policy has only ever been RL-trained at depth 1 — this measures
whether extra inference-time pondering helps or hurts BEFORE committing
production training to a deeper fixed depth.

    python scripts/depth_probe.py --ckpt path.pt --depth_a 1 --depth_b 2 \
        --games 16 --replay_dir ... --filtered_file ...

Reports A-wins/B-wins/draws over 2*games games (each seed played twice,
seats swapped).
"""

import argparse
import sys

import torch

from env.generals_env import GeneralsEnv
from scripts.elo_ladder import load_net
from training.train_rl import collect_selfplay_episode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--depth_a", type=int, default=1,
                   help="0 = full depth (--halt_max)")
    p.add_argument("--depth_b", type=int, default=2,
                   help="0 = full depth (--halt_max)")
    p.add_argument("--halt_max", type=int, default=0,
                   help="Full TRM depth for depth-0 seats (0 = checkpoint config)")
    p.add_argument("--games", type=int, default=16,
                   help="map seeds; each played twice (seats swapped)")
    p.add_argument("--rollout_steps", type=int, default=1536)
    p.add_argument("--replay_dir", type=str, default=None)
    p.add_argument("--filtered_file", type=str, default=None)
    p.add_argument("--greedy", type=int, default=1,
                   help="1 = greedy actions (deployment/eval convention)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _cfg = load_net(args.ckpt, device)
    if args.greedy:
        net.action_decoder.greedy = True
    env = GeneralsEnv(grid_h=_cfg.grid_h, grid_w=_cfg.grid_w,
                      max_turns=args.rollout_steps)
    # turn_globals nets expect 7 global features; bare env emits 4
    env.turn_globals = bool(getattr(_cfg, "turn_globals", False))
    if args.replay_dir and args.filtered_file:
        n = env.load_map_pool(args.replay_dir, args.filtered_file)
        print(f"Loaded {n} maps")
    if args.halt_max > 0:
        _cfg.halt_max_steps = args.halt_max  # net holds this cfg object
    halt_max = _cfg.halt_max_steps

    # depth 0 = full depth (None → cfg.halt_max_steps)
    da = args.depth_a if args.depth_a > 0 else None
    db = args.depth_b if args.depth_b > 0 else None

    wins = {"a": 0, "b": 0, "draw": 0}
    for seed in range(args.games):
        # game 1: seat0 thinks depth_a, seat1 depth_b; game 2 swapped
        for depths, a_seat in (
            ({0: da, 1: db}, 0),
            ({0: db, 1: da}, 1),
        ):
            collect_selfplay_episode(
                net, net, env, device,
                max_steps=args.rollout_steps,
                record=False,
                act_steps=depths,
                legal_mask=True,
                learner_pid=0,
                reset_seed=seed,
                collect_both=False,
                collect_buffers=False,
            )
            if env.winner is None:
                wins["draw"] += 1
            elif env.winner == a_seat:
                wins["a"] += 1
            else:
                wins["b"] += 1
            print(f"seed {seed} a_seat={a_seat}: winner={env.winner} "
                  f"(a={wins['a']} b={wins['b']} d={wins['draw']})",
                  flush=True)

    total = wins["a"] + wins["b"] + wins["draw"]
    name_a = f"d{halt_max}" if da is None else f"d{da}"
    name_b = f"d{halt_max}" if db is None else f"d{db}"
    print(f"\nDEPTH PROBE  {name_a} vs {name_b} "
          f"({total} games, paired greedy={bool(args.greedy)}, "
          f"halt_max={halt_max})")
    print(f"  a={wins['a']}  b={wins['b']}  draw={wins['draw']}  "
          f"wr_b={(wins['b'] + 0.5 * wins['draw']) / max(total, 1):.3f}")


if __name__ == "__main__":
    sys.exit(main())
