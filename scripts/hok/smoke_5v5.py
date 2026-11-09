#!/usr/bin/env python3
"""Run the real 5v5 interface for a few random-policy decisions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Import the low-level module without importing env/__init__.py.  This keeps the
# command runnable in hok_env's small Python 3.8 venv, which intentionally has
# no torch installation.
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "env"))

from hok5v5 import Hok5v5Env  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamecore", help="running gamecore server as host:port")
    parser.add_argument("--hok-root", default=os.environ.get("HOK_ROOT", "~/hok"))
    parser.add_argument("--sdk-python")
    parser.add_argument("--sdk-config")
    parser.add_argument("--port-base", type=int, default=36350)
    parser.add_argument("--gamecore-port", type=int, default=23432)
    parser.add_argument("--gamecore-local")
    parser.add_argument("--runtime-id", default="gm-hok5v5-smoke")
    parser.add_argument("--decisions", type=int, default=8)
    parser.add_argument(
        "--require-gameover",
        action="store_true",
        help="fail unless a terminal frame arrives before the decision limit",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--both-camps", action="store_true", help="control both camps and test lockstep self-play"
    )
    parser.add_argument("--max-frame-num", type=int, default=600)
    parser.add_argument("--workdir")
    parser.add_argument("--log")
    args = parser.parse_args(argv)

    with Hok5v5Env(
        gamecore=args.gamecore,
        hok_root=args.hok_root,
        sdk_python=args.sdk_python,
        sdk_config=args.sdk_config,
        ports=(args.port_base, args.port_base + 1),
        gamecore_port=args.gamecore_port,
        gamecore_local=args.gamecore_local,
        runtime_id=args.runtime_id,
        max_frame_num=args.max_frame_num,
        workdir=args.workdir,
        log=args.log,
    ) as env:
        print(
            json.dumps(
                {
                    "event": "ready",
                    "worker": env.metadata,
                    "config": env.config_report.as_dict(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        env.reset(
            eval_mode=True,
            use_common_ai=(False, False) if args.both_camps else (True, False),
        )
        completed = 0
        receives = 0
        saw_gameover = False
        while completed < args.decisions:
            if args.both_camps:
                stepped = env.step_round(
                    {
                        agent_id: (
                            lambda _frame, seed=args.seed + completed * 2 + agent_id: (
                                env.random_probabilities(seed)
                            )
                        )
                        for agent_id in env.controlled_agents
                    }
                )
                receives += len(stepped)
                print(
                    json.dumps(
                        {
                            "event": "frame_all",
                            "frames": {
                                str(agent_id): {
                                    "frame_no": frame.frame_no,
                                    "gameover": frame.gameover,
                                    "hero_rows": len(frame.heroes),
                                    "observation_shape": list(frame.observations.shape),
                                    "frame_heroes": len(frame.state.get("heroes", [])),
                                    "first_reward": (
                                        frame.heroes[0].reward if frame.heroes else None
                                    ),
                                    "hero_reward_entries": (
                                        len(frame.heroes[0].hero_rid_reward) if frame.heroes else 0
                                    ),
                                }
                                for agent_id, (frame, _results) in stepped.items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                terminal = [frame.gameover for frame, _results in stepped.values()]
                if any(terminal):
                    if not all(terminal):
                        raise RuntimeError("controlled camps disagreed on terminal state")
                    saw_gameover = True
                    break
                results = {
                    agent_id: rows
                    for agent_id, (_frame, rows) in stepped.items()
                    if rows is not None
                }
                if not results:
                    continue
                print(
                    json.dumps(
                        {
                            "event": "action_all",
                            "hero_rows": {
                                str(agent_id): len(rows) for agent_id, rows in results.items()
                            },
                            "legal_head_sizes": {
                                str(agent_id): [
                                    [int(head.size) for head in row.legal_action] for row in rows
                                ]
                                for agent_id, rows in results.items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                completed += 1
                continue

            frame = env.step_feature(1)
            receives += 1
            print(
                json.dumps(
                    {
                        "event": "frame",
                        "frame_no": frame.frame_no,
                        "continue_process": frame.continue_process,
                        "gameover": frame.gameover,
                        "hero_rows": len(frame.heroes),
                        "observation_shape": list(frame.observations.shape),
                        "frame_heroes": len(frame.state.get("heroes", [])),
                        "first_reward": frame.heroes[0].reward if frame.heroes else None,
                        "hero_reward_entries": (
                            len(frame.heroes[0].hero_rid_reward) if frame.heroes else 0
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if frame.gameover:
                saw_gameover = True
                break
            if not frame.continue_process:
                continue
            result = env.step_action(1, env.random_probabilities(args.seed + completed), frame)
            print(
                json.dumps(
                    {
                        "event": "action",
                        "hero_rows": len(result),
                        "actions": [list(row.actions) for row in result],
                        "legal_head_sizes": [
                            [int(head.size) for head in row.legal_action] for row in result
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            completed += 1
        if args.require_gameover and not saw_gameover:
            raise RuntimeError(
                f"no terminal frame after {completed} decisions and {receives} receives"
            )
        env.close_game(force=True)
        print(
            json.dumps(
                {
                    "event": "ok",
                    "decisions": completed,
                    "receives": receives,
                    "gameover": saw_gameover,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
