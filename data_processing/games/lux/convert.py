"""Convert Lux AI Season 3 Kaggle episode replays (JSON) into training
shards for the "lux" adapter (LuxEncoder + LuxDecoder).

    uv run python -m data_processing.games.lux.convert \
        --input replays/ --output_dir data/lux/train \
        --shard_size 64 --max_episodes 1000

One sample = one (episode, agent, match): a Lux S3 episode is 5 matches
of 101 recorded frames each (``match_steps`` 0..100, resetting at every
match start), plus one trailing frame after the last match.  Memory
resets between matches — units, points and the fog of war all restart —
so each match becomes its own ~101-step sequence and the recurrent core
never has to carry state across a reset it cannot observe.  Both agents'
perspectives are emitted (observations in the replay are already
per-agent, fog of war included), so one full episode yields
5 matches × 2 agents = 10 samples of 101 frames.

Replay step/action alignment (verified against unit movement in real
replays): ``steps[t][agent]["action"]`` is the action the agent produced
AFTER seeing ``steps[t-1]``'s observation, i.e. observation ``t`` pairs
with the action stored at ``steps[t+1]``.  The trailing frame (the very
last step, ``match_steps == 0`` again) has no following action and is
dropped.  ``steps[0]``'s action is empty (env reset) and is never used.

Coordinates: the JSON 2-D map arrays (``sensor_mask``, ``map_features``)
are indexed ``[x][y]`` (Lux-Design-S3 convention, verified against unit
positions); we transpose to ``[y][x]`` so planes are [C, H, W] row-major
with ``plane[c, y, x]``.

MAP PLANES (12 channels, must equal LuxConfig.map_channels):
     0  tile empty     (tile_type == 0; unseen tiles (−1) are 0 in all three)
     1  tile nebula    (tile_type == 1)
     2  tile asteroid  (tile_type == 2)
     3  map energy / 20, zeroed outside the sensor mask
     4  sensor mask (1 = visible)
     5  relic node one-hot (visible relic_nodes only)
     6  own  unit count / 4 (units stack in Lux S3)
     7  opp  unit count / 4 (visible opponents only)
     8  own  unit energy sum / 400
     9  opp  unit energy sum / 400
    10  match progress: constant match_steps / 100
    11  ones (bias plane)

UNIT FEATURES (16 dims, must equal LuxConfig.unit_feature_dim):
    [x/24, y/24, energy/400, 1.0 present flag, 12 zeros padding]
    Absent slots are all-zero; the encoder additionally zeroes them and
    substitutes its learned empty embedding via obs["unit_mask"].

GLOBALS (6 dims, must equal LuxConfig.lux_global_features):
    [own points/500, opp points/500, own wins/5, opp wins/5,
     match_steps/101, steps/506]

TARGETS:
    unit_actions [T, 16] long — action_type only (0=center/noop,
        1=up, 2=right, 3=down, 4=left, 5=sap).  The sap dx/dy target
        offsets are DROPPED in v1: the adapter's LuxDecoder is an
        action-type-only factored categorical (its documented v1
        simplification); a per-unit target pointer can be added later.
    unit_mask [T, 16] bool — echoed in BOTH obs (encoder input, and the
        source of legal["unit"] → pred["unit_present"] via the trainer's
        LEGAL_FROM_OBS hook) and tgt (self-describing shards).
    value [T] float — the MATCH result from this agent's perspective,
        +1 win / −1 loss, constant across the match's frames.  Winner =
        the team whose team_wins increments at the next match boundary
        (authoritative, ties already resolved by the engine); if the
        boundary frame is missing (truncated replay) we fall back to
        sign(own − opp final team_points), 0 on a tie.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from data_processing.games.dataset import save_shard

# Must match LuxConfig defaults (model/games/lux.py); a pipeline test
# asserts this so the two cannot drift silently.
MAP_H = 24
MAP_W = 24
MAP_CHANNELS = 12
MAX_UNITS = 16
UNIT_FEATURE_DIM = 16
GLOBAL_FEATURES = 6
N_UNIT_ACTIONS = 6

# Normalisation constants (Lux S3 ranges: tile energy ∈ [−20, 20], unit
# energy ≤ 400, 5 match wins per episode, points a few hundred).
TILE_ENERGY_NORM = 20.0
UNIT_ENERGY_NORM = 400.0
POINTS_NORM = 500.0
WINS_NORM = 5.0
UNIT_COUNT_NORM = 4.0
MATCH_LEN = 100.0        # max_steps_in_match
MATCH_FRAMES = 101.0     # frames per match (steps 0..100)
EPISODE_FRAMES = 506.0   # 5 × 101 + trailing frame


def parse_obs(step_entry: Dict) -> Optional[Dict]:
    """The per-agent observation dict from one ``steps[t][agent]`` entry.

    Kaggle stores ``observation.obs`` as a JSON string; some tooling
    re-saves it as a dict — accept both."""
    raw = (step_entry.get("observation") or {}).get("obs")
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return raw if isinstance(raw, dict) else None


def _map_array(rows: List, dtype) -> np.ndarray:
    """JSON ``[x][y]`` map array → ``[y][x]`` (row-major H×W)."""
    return np.asarray(rows, dtype=dtype).T


def build_frame(
    o: Dict, team: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One observation dict → (map [12,H,W], units [U,16], mask [U], globals [6])."""
    opp = 1 - team

    planes = np.zeros((MAP_CHANNELS, MAP_H, MAP_W), dtype=np.float32)
    tt = _map_array(o["map_features"]["tile_type"], np.int64)
    planes[0] = tt == 0
    planes[1] = tt == 1
    planes[2] = tt == 2
    sm = _map_array(o["sensor_mask"], bool)
    energy = _map_array(o["map_features"]["energy"], np.float32)
    planes[3] = np.where(sm, energy / TILE_ENERGY_NORM, 0.0)
    planes[4] = sm

    for (x, y), visible in zip(o["relic_nodes"], o["relic_nodes_mask"]):
        if visible and 0 <= x < MAP_W and 0 <= y < MAP_H:
            planes[5, y, x] = 1.0

    positions = o["units"]["position"]
    energies = o["units"]["energy"]
    masks = o["units_mask"]
    for side, count_c, energy_c in ((team, 6, 8), (opp, 7, 9)):
        for (x, y), e, m in zip(positions[side], energies[side], masks[side]):
            if m and 0 <= x < MAP_W and 0 <= y < MAP_H:
                planes[count_c, y, x] += 1.0 / UNIT_COUNT_NORM
                planes[energy_c, y, x] += max(float(e), 0.0) / UNIT_ENERGY_NORM

    planes[10] = float(o["match_steps"]) / MATCH_LEN
    planes[11] = 1.0

    units = np.zeros((MAX_UNITS, UNIT_FEATURE_DIM), dtype=np.float32)
    unit_mask = np.zeros(MAX_UNITS, dtype=bool)
    for u in range(min(MAX_UNITS, len(masks[team]))):
        if not masks[team][u]:
            continue
        x, y = positions[team][u]
        unit_mask[u] = True
        units[u, 0] = float(x) / MAP_W
        units[u, 1] = float(y) / MAP_H
        units[u, 2] = float(energies[team][u]) / UNIT_ENERGY_NORM
        units[u, 3] = 1.0
        # dims 4..15 reserved (zero padding)

    pts, wins = o["team_points"], o["team_wins"]
    globals_v = np.array([
        float(pts[team]) / POINTS_NORM,
        float(pts[opp]) / POINTS_NORM,
        float(wins[team]) / WINS_NORM,
        float(wins[opp]) / WINS_NORM,
        float(o["match_steps"]) / MATCH_FRAMES,
        float(o["steps"]) / EPISODE_FRAMES,
    ], dtype=np.float32)
    return planes, units, unit_mask, globals_v


def _actions_at(steps: List, t: int, agent: int) -> Optional[List]:
    """The [max_units, 3] action list stored at ``steps[t][agent]``."""
    if not (0 <= t < len(steps)):
        return None
    action = steps[t][agent].get("action")
    if not action:                      # {} at step 0, None on agent error
        return None
    return action


def _match_value(
    parsed: List[Optional[Dict]], a: int, b: int, team: int,
) -> float:
    """Match result in [a, b) from ``team``'s perspective (+1 / −1 / 0)."""
    o_start, o_end = parsed[a], parsed[b] if b < len(parsed) else None
    if o_start is not None and o_end is not None:
        dw_own = o_end["team_wins"][team] - o_start["team_wins"][team]
        dw_opp = o_end["team_wins"][1 - team] - o_start["team_wins"][1 - team]
        if dw_own != dw_opp:
            return 1.0 if dw_own > dw_opp else -1.0
    # Truncated replay fallback: final points of the last parsed frame.
    for t in range(b - 1, a - 1, -1):
        if parsed[t] is not None:
            diff = (parsed[t]["team_points"][team]
                    - parsed[t]["team_points"][1 - team])
            return float(np.sign(diff))
    return 0.0


def episode_to_samples(
    episode: Dict, source: str,
) -> Tuple[List[Dict], Counter, Dict[str, int]]:
    """One replay dict → (samples, present-unit action histogram, counters)."""
    steps = episode["steps"]
    episode_id = episode.get("id", source)
    samples: List[Dict] = []
    hist: Counter = Counter()
    counters = {"matches": 0, "skipped_agents": 0, "empty_matches": 0}

    for agent in range(min(2, len(steps[0]) if steps else 0)):
        parsed = [parse_obs(s[agent]) for s in steps]
        if all(o is None for o in parsed):
            counters["skipped_agents"] += 1
            continue
        player = (steps[0][agent].get("observation") or {}).get("player", "")
        team = int(player.rsplit("_", 1)[-1]) if player.startswith("player_") \
            else agent

        # Match boundaries: every parsed frame with match_steps == 0.
        bounds = [t for t, o in enumerate(parsed)
                  if o is not None and o["match_steps"] == 0]
        ranges = list(zip(bounds, bounds[1:] + [len(steps)]))

        for match_idx, (a, b) in enumerate(ranges):
            maps, units, masks, globs, acts = [], [], [], [], []
            for t in range(a, b):
                o = parsed[t]
                action = _actions_at(steps, t + 1, agent)
                if o is None or action is None:
                    break               # trailing frame / truncated tail
                m, u, um, g = build_frame(o, team)
                unit_actions = np.zeros(MAX_UNITS, dtype=np.int64)
                for slot in range(min(MAX_UNITS, len(action))):
                    unit_actions[slot] = int(
                        np.clip(action[slot][0], 0, N_UNIT_ACTIONS - 1))
                maps.append(m); units.append(u); masks.append(um)
                globs.append(g); acts.append(unit_actions)
                hist.update(unit_actions[um].tolist())
            if not maps:
                counters["empty_matches"] += 1
                continue
            counters["matches"] += 1
            T = len(maps)
            value = _match_value(parsed, a, b, team)
            unit_mask_t = torch.from_numpy(np.stack(masks))
            samples.append({
                "obs": {
                    "map_seq": torch.from_numpy(np.stack(maps)),
                    "units_seq": torch.from_numpy(np.stack(units)),
                    "unit_mask": unit_mask_t,
                    "globals_seq": torch.from_numpy(np.stack(globs)),
                },
                "tgt": {
                    "unit_actions": torch.from_numpy(np.stack(acts)),
                    "unit_mask": unit_mask_t.clone(),
                    "value": torch.full((T,), value, dtype=torch.float32),
                },
                "meta": {
                    "source": source,
                    "episode_id": episode_id,
                    "agent": agent,
                    "team": team,
                    "match_index": match_idx,
                    "step_range": (a, a + T),
                    "match_value": value,
                },
            })
    return samples, hist, counters


def input_files(path: str) -> List[str]:
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "**", "*.json"),
                                 recursive=True))
        if not found:
            raise FileNotFoundError(f"no .json replays under {path}")
        return found
    return [path]


def convert(
    input_path: str,
    output_dir: str,
    shard_size: int = 64,
    max_episodes: Optional[int] = None,
) -> Dict:
    stats = {
        "episodes": 0, "bad_files": 0, "matches": 0, "empty_matches": 0,
        "skipped_agents": 0, "samples": 0, "frames": 0, "shards": 0,
        "action_hist": Counter(),
    }
    pending: List[Dict] = []

    def flush() -> None:
        if not pending:
            return
        path = os.path.join(output_dir, f"lux_{stats['shards']:05d}.pt")
        save_shard(path, "lux", list(pending))
        stats["shards"] += 1
        pending.clear()

    for path in input_files(input_path):
        if max_episodes is not None and stats["episodes"] >= max_episodes:
            break
        try:
            with open(path) as fh:
                episode = json.load(fh)
            if not isinstance(episode.get("steps"), list) or not episode["steps"]:
                raise ValueError("no steps[] — not an episode replay")
        except (ValueError, OSError) as e:
            stats["bad_files"] += 1
            print(f"[warn] {path}: skipped — {e}")
            continue
        stats["episodes"] += 1
        samples, hist, counters = episode_to_samples(
            episode, os.path.basename(path))
        for k in ("matches", "empty_matches", "skipped_agents"):
            stats[k] += counters[k]
        stats["action_hist"].update(hist)
        for s in samples:
            stats["samples"] += 1
            stats["frames"] += s["tgt"]["value"].shape[0]
            pending.append(s)
            if len(pending) >= shard_size:
                flush()
    flush()
    return stats


ACTION_NAMES = ["noop", "up", "right", "down", "left", "sap"]


def main(argv: Optional[List[str]] = None) -> Dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="replay .json, or directory scanned recursively")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--shard_size", type=int, default=64)
    p.add_argument("--max_episodes", type=int, default=None)
    args = p.parse_args(argv)

    stats = convert(args.input, args.output_dir,
                    shard_size=args.shard_size, max_episodes=args.max_episodes)
    hist = stats["action_hist"]
    total = sum(hist.values()) or 1
    hist_str = " ".join(
        f"{ACTION_NAMES[a]}={hist.get(a, 0)}({100 * hist.get(a, 0) / total:.1f}%)"
        for a in range(N_UNIT_ACTIONS)
    )
    print(
        f"[done] episodes={stats['episodes']} bad_files={stats['bad_files']} "
        f"matches={stats['matches']} samples={stats['samples']} "
        f"frames={stats['frames']} shards={stats['shards']} "
        f"→ {args.output_dir}\n[actions] {hist_str}"
    )
    return stats


if __name__ == "__main__":
    main()
