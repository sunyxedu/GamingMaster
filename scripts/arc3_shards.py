"""ARC-AGI-3 imitation shards from solver solutions (scripts/arc3_solve.py): every game's
solved levels replayed through the RL env (env/vec_arc3.py), so an IL frame is exactly
the frame the network sees when it plays — one sample per game playthrough.

Per frame: obs grid / delta / globals (the env's), tgt type_idx / click_idx (the
solution's action), legal_type [8] / legal_click [4096] (the env's legal tables, the
decoder's legal dict under ``--il_legal_mask``) and value = the return still to come
(levels this playthrough completes after the frame / the game's levels: the RL reward
units, a whole game = 1).  Shard meta {"global_features": 22, "grid": 64}.

    ENVIRONMENTS_DIR=... uv run python scripts/arc3_shards.py --solutions sol.json \\
        --out data/arc3/train [--games cd82 lf52 ls20 sb26]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_processing.games.dataset import save_shard  # noqa: E402
from env.vec_arc3 import GRID, N_GLOBALS, Arc3VecEnv  # noqa: E402

CONVERTER_VERSION = 1


def playthrough(game: str, levels: List[Dict], environments_dir=None) -> Dict:
    """Replay a game's solved levels; the IL sample of that playthrough."""
    actions = [a for lv in levels if lv.get("actions") for a in lv["actions"]]
    if not actions:
        raise ValueError(f"{game}: no solved level")
    env = Arc3VecEnv(1, games=(game,), max_actions=len(actions) + 1,
                     environments_dir=environments_dir)
    obs = env.observe()
    frames: Dict[str, List[torch.Tensor]] = {k: [] for k in
                                             ("grid", "delta", "globals", "legal_type",
                                              "legal_click", "type_idx", "click_idx")}
    rewards = []
    for t, (typ, x, y) in enumerate(actions):
        if not bool(obs.active[0]):
            raise RuntimeError(f"{game}: the episode ended at action {t} of {len(actions)}")
        if bool(obs.forced[0]):
            raise RuntimeError(f"{game}: a solution reached GAME_OVER at action {t}")
        for k in ("grid", "delta", "globals"):
            frames[k].append(obs.obs[k][0])
        frames["legal_type"].append(obs.legal["type"][0])
        frames["legal_click"].append(obs.legal["click"][0])
        click = int(y) * GRID + int(x)
        frames["type_idx"].append(torch.tensor(int(typ)))
        frames["click_idx"].append(torch.tensor(click if int(typ) == 6 else 0))
        obs, rew, done, info = env.step({"type_idx": torch.tensor([int(typ)]),
                                         "click_idx": torch.tensor([click])})
        rewards.append(float(rew[0, 0]))
    solved = sum(1 for lv in levels if lv.get("actions"))
    if int(info["levels"][0]) != solved:
        raise RuntimeError(f"{game}: replay completed {int(info['levels'][0])} levels, the "
                           f"solutions {solved}")
    togo = torch.tensor(rewards).flip(0).cumsum(0).flip(0)          # return still to come
    env.close()
    return {"obs": {k: torch.stack(frames[k]) for k in ("grid", "delta", "globals")},
            "tgt": {"type_idx": torch.stack(frames["type_idx"]).long(),
                    "click_idx": torch.stack(frames["click_idx"]).long(),
                    "legal_type": torch.stack(frames["legal_type"]),
                    "legal_click": torch.stack(frames["legal_click"]),
                    "value": togo.float()},
            "meta": {"game": game, "levels": solved, "actions": len(actions)}}


def main(argv=None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--solutions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", nargs="*", default=None, help="default: every solved game")
    p.add_argument("--environments_dir", default=os.environ.get("ENVIRONMENTS_DIR"))
    a = p.parse_args(argv)
    with open(a.solutions) as fh:
        sol = json.load(fh)
    games = a.games or sorted(sol)
    samples, stats = [], {}
    for g in games:
        levels = sol[g]["levels"]
        if not any(lv.get("actions") for lv in levels):
            stats[g] = 0
            continue
        s = playthrough(g, levels, a.environments_dir)
        samples.append(s)
        stats[g] = s["meta"]["levels"]
    if not samples:
        raise SystemExit("no solved level in the given games")
    os.makedirs(a.out, exist_ok=True)
    save_shard(os.path.join(a.out, "arc3_00000.pt"), "arc3", samples,
               converter_version=CONVERTER_VERSION,
               meta={"global_features": N_GLOBALS, "grid": GRID})
    frames = sum(int(s["obs"]["grid"].shape[0]) for s in samples)
    out = {"samples": len(samples), "frames": frames, "levels": stats}
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    main()
