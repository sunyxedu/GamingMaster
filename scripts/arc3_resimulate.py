"""The human ARC-AGI-3 sessions re-simulated on the current game files, recorded as the env's
own responses — IL frames that are exactly what env/vec_arc3.py shows RL.

The official games were patched after the human testing without a new version id: ft09 now
offers ACTION6 alone (the recordings list ACTION1-6, and players pressed keys), s5i5 draws two
highlight cells differently.  Converting the recordings as they are would train on legal
masks and pixels the env never produces.  So each session's actions (the converter's labels;
a session whose recorder missed the starting RESET gets its first action back from the meta)
are replayed through the env, and every response is written in the recording format the
converter reads (tests/test_arc3_recordings.py: a recording of the env converts to exactly the
env's frames).

A session stops where it no longer follows its recording: the level count differs, or the
grids differ in more than --max-diff cells for --patience frames in a row.  An action the
current game refuses is skipped (the recorded game ignored it as well if the grids keep
matching).

    uv run python scripts/arc3_resimulate.py --env-dir data/arc3/official --out data/arc3/resim \\
        --input ~/Documents/arc3-wm/data/replays/arc_agi_3_public_demo_human_testing.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CLICK = 6


def convert(gmdata: str, inputs, out: str, max_actions: int, games):
    cmd = [gmdata, "arc3", "convert", *sum((["--input", str(i)] for i in inputs), []), "--out", out,
           "--max-actions", str(max_actions)]
    if games:
        cmd += ["--games", ",".join(games)]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    from data_processing.games.dataset import load_gm_samples

    return json.loads(r.stdout.strip().splitlines()[-1]), load_gm_samples(out)


def resimulate(sample, env_dir: str, max_actions: int, max_diff: int, patience: int):
    """(rows of the env's responses with the actions taken, stats)."""
    import torch
    from env.vec_arc3 import Arc3VecEnv

    meta = sample["meta"]
    game = meta["game"]
    env = Arc3VecEnv(1, games=(game,), max_actions=max_actions, environments_dir=env_dir)
    rows = []

    def record(i, fd, _load=env._load):
        rows.append({"frame": [[list(map(int, r)) for r in fd.frame[-1]]], "state": fd.state.name,
                     "levels_completed": int(fd.levels_completed), "win_levels": int(fd.win_levels),
                     "available_actions": [int(a) for a in fd.available_actions]})
        return _load(i, fd)

    env._load = record
    actions = [(0, None)]                     # the reset that starts the game
    obs = env.reset()
    st = collections.Counter()

    def act(typ, click):
        nonlocal obs
        obs, _, done, _ = env.step({"type_idx": torch.tensor([typ]), "click_idx": torch.tensor([click])})
        actions.append((typ, (click % 64, click // 64) if typ == CLICK else None))
        return bool(done[0])

    done = False
    if meta.get("missing_start"):
        typ, x, y = meta["first_action"]
        click = y * 64 + x if typ == CLICK else 0
        done = act(typ, click)
    grid, glob = sample["obs"]["grid"], sample["obs"]["globals"]
    types, clicks = sample["tgt"]["type_idx"], sample["tgt"]["click_idx"]
    T = int(types.shape[0])
    bad = 0
    stop = "end"
    for t in range(T):
        if done:
            stop = "env_done"
            break
        diff = int((obs.obs["grid"][0] != grid[t]).sum())
        st["max_cells_differing"] = max(st["max_cells_differing"], diff)
        if diff:
            st["frames_differing"] += 1
        # globals[8] = levels / win_levels on both sides
        if abs(float(obs.obs["globals"][0][8]) - float(glob[t][8])) > 1e-6:
            stop = f"levels differ at {t}"
            break
        bad = bad + 1 if diff > max_diff else 0
        if bad >= patience:
            stop = f"grids differ at {t - patience + 1}"
            break
        typ, click = int(types[t]), int(clicks[t])
        if not bool(obs.legal["type"][0][typ]):
            st["refused"] += 1
            continue
        done = act(typ, click)
        st["steps"] += 1
    env.close()
    st["recorded_frames"] = T
    return rows, actions, stop, st


def write_recording(path: str, game_id: str, guid: str, rows, actions):
    lines = []
    for (typ, xy), r in zip(actions, rows):
        data = {"game_id": game_id}
        if xy is not None:
            data.update(x=xy[0], y=xy[1])
        lines.append(json.dumps({"timestamp": "", "data": {"game_id": game_id, "guid": guid, "full_reset": False,
                                                            "action_input": {"id": typ, "data": data}, **r}}))
    won = int(bool(rows) and rows[-1]["state"] == "WIN")
    lines.append(json.dumps({"timestamp": "", "data": {"won": won}}))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    os.replace(path + ".part", path)
    return won


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", action="append", required=True, help="recordings (zip / directory), repeatable")
    p.add_argument("--env-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--games", default="", help="comma-separated (default: every game in --env-dir)")
    p.add_argument("--max-actions", type=int, default=2048)
    p.add_argument("--max-diff", type=int, default=16)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--gmdata", default=os.path.join(ROOT, "data_processing", "target", "release", "gmdata"))
    a = p.parse_args(argv)
    os.environ["ENVIRONMENTS_DIR"] = a.env_dir
    games = [g for g in a.games.split(",") if g] or sorted(
        g for g in os.listdir(a.env_dir) if os.path.isdir(os.path.join(a.env_dir, g)))
    with tempfile.TemporaryDirectory() as tmp:
        st, samples = convert(a.gmdata, a.input, os.path.join(tmp, "shards"), a.max_actions, games)
    per_game = collections.defaultdict(collections.Counter)
    stops = collections.Counter()
    for s in samples:
        m = s["meta"]
        rows, actions, stop, sst = resimulate(s, a.env_dir, a.max_actions, a.max_diff, a.patience)
        won = write_recording(os.path.join(a.out, m["game"], f"{m['guid']}.recording.jsonl"), m["game_id"],
                              m["guid"], rows, actions)
        g = per_game[m["game"]]
        g["sessions"] += 1
        g["won"] += won
        g["recorded_won"] += int(bool(m.get("won")))
        g["frames"] += len(actions) - 1
        g["recorded_frames"] += sst["recorded_frames"]
        g["refused"] += sst["refused"]
        g["frames_differing"] += sst["frames_differing"]
        g["cut"] += int(stop not in ("end", "env_done"))
        g["max_cells_differing"] = max(g["max_cells_differing"], sst["max_cells_differing"])
        stops[stop.split(" at ")[0]] += 1
    print(json.dumps({"converted": {k: st.get(k) for k in ("kept", "errors", "noise_rows", "missing_start")},
                      "stops": stops, "per_game": per_game}, indent=1))


if __name__ == "__main__":
    main()
