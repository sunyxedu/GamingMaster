"""External demo evaluator: real cadence by default; optional offline replay checks."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from game import PROFILES, make_game


def run(seed=0, profile="demo-v1", *, fast=False, max_ticks=None):
    game = make_game(profile=profile)
    game.reset(seed)
    policy_rng = random.Random(seed)  # Independent random player, never engine RNG.
    cap = game.recommended_tick_cap
    if max_ticks is not None:
        if max_ticks <= 0:
            raise ValueError("max_ticks must be positive")
        cap = min(cap, max_ticks)
    origin = time.monotonic()
    receipts = []
    while not game.done and game.ticks < cap:
        deadline = origin + (game.ticks + 1) * game.tick_s
        action = policy_rng.choice(game.action_spec.names)
        receipt = time.monotonic() - origin
        if not fast:
            time.sleep(max(0, deadline - time.monotonic()))
            # Severe host lag is an infrastructure truncation, never a hidden pause.
            if time.monotonic() - deadline > 5 * game.tick_s:
                game.truncate("infrastructure_failure")
                break
        game.tick(action)
        receipts.append({"tick": game.ticks, "requested": action,
                         "received_elapsed_s": receipt,
                         "applied_elapsed_s": time.monotonic() - origin})
    if not game.done:
        game.truncate()
    return {"clock": "offline-fast" if fast else "wall-monotonic",
            "wall_s": time.monotonic() - origin,
            "tick_cap": cap, "receipts": receipts,
            "replay": game.replay(), "metrics": game.info()}


def replay_file(path):
    log = json.loads(Path(path).read_text())
    replay = log["replay"]
    game = make_game(profile=replay["profile"])
    game.reset(replay["seed"])
    for record in replay["actions"]:
        game.tick(record["requested"])
        assert game.ticks == record["tick"]
        assert game.observe()["last_applied_action"] == record["applied"]
    if replay["final"]["truncated"]:
        game.truncate(replay["final"]["end_reason"])
    if game.observe() != replay["final"]:
        raise ValueError("Replay differs from recorded final state")
    return game.info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", choices=PROFILES, default="demo-v1")
    parser.add_argument("--fast", action="store_true", help="Offline diagnostics, not real-time scoring")
    parser.add_argument("--max-ticks", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(replay_file(args.replay), indent=2))
    else:
        result = run(args.seed, args.profile, fast=args.fast, max_ticks=args.max_ticks)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2))
