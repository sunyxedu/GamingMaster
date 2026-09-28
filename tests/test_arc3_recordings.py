"""ARC-AGI-3 recordings → shards (`gmdata arc3 convert`, data_processing/rs/gmdata/src/arc3.rs)
give the frames env/vec_arc3.py gives the network for the same actions.

* A recording written in the toolkit's format from the env's own SDK responses (random
  legal play with clicks, RESETs and game overs) converts to exactly the env's obs /
  legal tensors and the return still to come.
* The human baseline recordings (``GM_ARC3_HUMAN_ZIP``, default the arc3-wm checkout's
  copy): sessions of the locally cached games replayed through the env reproduce the
  converter's frames — the game versions match and the row semantics (the starting
  RESET, forced RESETs after a GAME_OVER, bookkeeping rows) are the env's.

* Retrodict's winning runs (``GM_ARC3_RETRODICT``, default data/arc3/retrodict/release-runs:
  its release-runs archive) of the cached games replay through the env, reproduce every
  logged board and win.

All need the ARC-AGI-3 SDK and the cached games (ENVIRONMENTS_DIR); the last two also their
data.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
#: the official games' files: all 25 from the ARC API (scripts/arc3_fetch_games.py), else the
#: arc3-wm checkout's 8
OFFICIAL_DIR = os.path.join(ROOT, "data", "arc3", "official")
ENV_DIR = os.environ.get("ENVIRONMENTS_DIR") or (
    OFFICIAL_DIR if os.path.isdir(OFFICIAL_DIR) else os.path.expanduser("~/Documents/arc3-wm/environment_files"))
#: GM_ARC3_FULL=1: every human session to its end (else 2 per game, 300 frames each)
FULL = os.environ.get("GM_ARC3_FULL") == "1"
HUMAN_ZIP = os.environ.get("GM_ARC3_HUMAN_ZIP") or os.path.expanduser(
    "~/Documents/arc3-wm/data/replays/arc_agi_3_public_demo_human_testing.zip")
RETRODICT = os.environ.get("GM_ARC3_RETRODICT") or os.path.join(ROOT, "data", "arc3", "retrodict", "release-runs")
MAX_ACTIONS = 2048


def _sdk_ok() -> bool:
    try:
        import arc_agi  # noqa: F401
        import arcengine  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(os.path.join(ENV_DIR, "vc33"))


needs_sdk = pytest.mark.skipif(not _sdk_ok(), reason="ARC-AGI-3 SDK or the cached games not available")


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS, capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


def _convert(gmdata, inputs, out, *extra):
    r = subprocess.run([gmdata, "arc3", "convert", *sum((["--input", str(i)] for i in inputs), []), "--out", str(out),
                        "--max-actions", str(MAX_ACTIONS), *extra], capture_output=True, text=True, check=True)
    from data_processing.games.dataset import load_gm_samples

    return json.loads(r.stdout.strip().splitlines()[-1]), load_gm_samples(str(out))


def _env(game):
    os.environ.setdefault("ENVIRONMENTS_DIR", ENV_DIR)
    from env.vec_arc3 import Arc3VecEnv

    return Arc3VecEnv(1, games=(game,), max_actions=MAX_ACTIONS, environments_dir=ENV_DIR)


def _frame(obs):
    return {"grid": obs.obs["grid"][0].clone(), "delta": obs.obs["delta"][0].clone(),
            "globals": obs.obs["globals"][0].clone(), "legal_type": obs.legal["type"][0].clone(),
            "legal_click": obs.legal["click"][0].clone(), "forced": bool(obs.forced[0])}


def _check(sample, frames, actions, first_delta=True):
    obs, tgt = sample["obs"], sample["tgt"]
    T = len(frames)
    assert obs["grid"].shape[0] == T, (obs["grid"].shape, T)
    for t, f in enumerate(frames):
        assert torch.equal(obs["grid"][t], f["grid"]), t
        if t or first_delta:
            assert torch.equal(obs["delta"][t].bool(), f["delta"]), t
        torch.testing.assert_close(obs["globals"][t], f["globals"], rtol=0, atol=1e-6, msg=f"globals {t}")
        assert torch.equal(tgt["legal_type"][t].bool(), f["legal_type"]), t
        assert torch.equal(tgt["legal_click"][t].bool(), f["legal_click"]), t
        typ, click = actions[t]
        assert int(tgt["type_idx"][t]) == typ, t
        if typ == 6:
            assert int(tgt["click_idx"][t]) == click, t


@needs_sdk
@pytest.mark.parametrize("game", ["vc33", "ls20"])
def test_a_recording_of_the_env_converts_to_the_envs_frames(gmdata, tmp_path, game):
    env = _env(game)
    rows = []

    def record(i, fd, _load=env._load):
        rows.append({"frame": [[list(map(int, r)) for r in fd.frame[-1]]], "state": fd.state.name,
                     "levels_completed": int(fd.levels_completed), "win_levels": int(fd.win_levels),
                     "available_actions": [int(a) for a in fd.available_actions]})
        return _load(i, fd)

    env._load = record
    obs = env.reset()                       # the SDK's reset response: the starting RESET row
    ids = [("RESET", None)]
    rng = random.Random(0)
    frames, actions, rewards = [], [], []
    for _ in range(80):
        f = _frame(obs)
        legal = [a for a in range(8) if f["legal_type"][a]]
        typ = 0 if f["forced"] else rng.choice(legal + [a for a in legal if a == 6] * 3)
        click = rng.randrange(4096)
        frames.append(f)
        actions.append((typ, click))
        obs, rew, done, info = env.step({"type_idx": torch.tensor([typ]), "click_idx": torch.tensor([click])})
        rewards.append(float(rew[0, 0]))
        ids.append((typ, (click % 64, click // 64) if typ == 6 else None))
        if bool(done[0]):
            break
    lines = []
    for (typ, xy), r in zip(ids, rows):
        data = {"game_id": f"{game}-x"}
        if xy is not None:
            data.update(x=xy[0], y=xy[1])
        lines.append(json.dumps({"timestamp": "", "data": {"game_id": f"{game}-x", "guid": "g", "full_reset": False,
                                                            "action_input": {"id": typ, "data": data}, **r}}))
    lines.append(json.dumps({"timestamp": "", "data": {"won": 0}}))
    (tmp_path / "rec").mkdir()
    (tmp_path / "rec" / f"{game}.recording.jsonl").write_text("\n".join(lines) + "\n")
    st, samples = _convert(gmdata, [tmp_path / "rec"], tmp_path / "out")
    assert st["kept"] == 1 and st["errors"] == 0 and st["missing_start"] == 0
    (s,) = samples
    _check(s, frames, actions)
    togo = torch.tensor(rewards).flip(0).cumsum(0).flip(0)
    torch.testing.assert_close(s["tgt"]["value"], togo.float(), rtol=0, atol=1e-6)


OFFICIAL = ("ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52", "lp85", "ls20", "m0r0", "r11l",
            "re86", "s5i5", "sb26", "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30")
LOCAL = tuple(g for g in OFFICIAL if os.path.isdir(os.path.join(ENV_DIR, g)))
#: games patched after the human testing without a new version id: their recordings do not
#: replay exactly (scripts/arc3_resimulate.py re-records them on the current files)
PATCHED = {"ft09": "ACTION1-5 no longer available (the players pressed keys)",
           "s5i5": "up to 22 cells drawn differently, the same levels and wins"}


@needs_sdk
@pytest.mark.skipif(not os.path.exists(HUMAN_ZIP), reason="the human baseline recordings are not available")
def test_human_sessions_replay_through_the_env(gmdata, tmp_path):
    games = [g for g in LOCAL if g not in PATCHED]
    st, samples = _convert(gmdata, [HUMAN_ZIP], tmp_path / "out", "--games", ",".join(games))
    assert st["errors"] == 0
    checked = 0
    per_game = {}
    for s in samples:
        g = s["meta"]["game"]
        if per_game.get(g, 0) >= 2 and not FULL:
            continue
        per_game[g] = per_game.get(g, 0) + 1
        env = _env(g)
        obs = env.reset()
        if s["meta"]["missing_start"]:
            # the recorder missed the starting RESET: the session's first frame is the state
            # after its first action
            typ, x, y = s["meta"]["first_action"]
            obs, _, _, _ = env.step({"type_idx": torch.tensor([typ]), "click_idx": torch.tensor([y * 64 + x])})
        T = int(s["tgt"]["type_idx"].shape[0]) if FULL else min(int(s["tgt"]["type_idx"].shape[0]), 300)
        frames, actions = [], []
        for t in range(T):
            frames.append(_frame(obs))
            typ, click = int(s["tgt"]["type_idx"][t]), int(s["tgt"]["click_idx"][t])
            actions.append((typ, click))
            obs, _, done, _ = env.step({"type_idx": torch.tensor([typ]), "click_idx": torch.tensor([click])})
            if bool(done[0]):
                break
        cut = {"obs": {k: v[:len(frames)] for k, v in s["obs"].items()},
               "tgt": {k: v[:len(frames)] for k, v in s["tgt"].items()}}
        # a missing-start session's first frame has no delta (the converter never saw the
        # frame before the first action)
        _check(cut, frames, actions, first_delta=not s["meta"]["missing_start"])
        checked += len(frames)
        env.close()
    assert len(per_game) == len(games) and checked > 2000, (per_game, checked)


@needs_sdk
@pytest.mark.skipif(not os.path.exists(HUMAN_ZIP), reason="the human baseline recordings are not available")
def test_resimulated_sessions_are_the_envs_frames(gmdata, tmp_path):
    """scripts/arc3_resimulate.py on a patched game (s5i5, ft09) and a clean one: the
    re-recorded sessions convert to exactly the env's frames, and a clean game's sessions come
    out as recorded (all of their frames, the same wins)."""
    games = [g for g in ("s5i5", "ft09", "vc33") if g in LOCAL]
    if not games:
        pytest.skip("no official game files")
    from scripts.arc3_resimulate import convert, resimulate, write_recording

    _, samples = convert(gmdata, [HUMAN_ZIP], str(tmp_path / "rec_shards"), MAX_ACTIONS, games)
    picked = {}
    for s in samples:
        picked.setdefault(s["meta"]["game"], []).append(s)
    out = tmp_path / "resim"
    for g, ss in picked.items():
        for s in ss[:2]:
            rows, actions, stop, st = resimulate(s, ENV_DIR, MAX_ACTIONS, 32, 3)
            write_recording(str(out / g / f"{s['meta']['guid']}.recording.jsonl"), s["meta"]["game_id"],
                            s["meta"]["guid"], rows, actions)
            if g == "vc33":
                assert stop in ("end", "env_done") and st["frames_differing"] == 0 and len(actions) - 1 == \
                    st["recorded_frames"], (stop, st)
    st, resim = _convert(gmdata, [out], tmp_path / "out")
    assert st["errors"] == 0 and st["noise_rows"] == 0 and st["kept"] == len(resim) > 0
    for s in resim:
        env = _env(s["meta"]["game"])
        obs = env.reset()
        frames, actions = [], []
        for t in range(int(s["tgt"]["type_idx"].shape[0])):
            frames.append(_frame(obs))
            typ, click = int(s["tgt"]["type_idx"][t]), int(s["tgt"]["click_idx"][t])
            actions.append((typ, click))
            obs, _, done, _ = env.step({"type_idx": torch.tensor([typ]), "click_idx": torch.tensor([click])})
            if bool(done[0]):
                break
        _check(s, frames, actions)
        env.close()


@needs_sdk
@pytest.mark.skipif(not os.path.isdir(RETRODICT), reason="the Retrodict release runs are not available")
def test_retrodict_runs_replay_through_the_env_and_win(gmdata, tmp_path):
    """Retrodict's winning runs (github.com/ryanbbrown/Retrodict release-runs) of the cached
    games: their actions replayed through the env reproduce every logged settled board and win
    the game — the logs and the env are the same game versions."""
    games = [g for g in LOCAL if os.path.isdir(os.path.join(RETRODICT, g))]
    st, samples = _convert(gmdata, [RETRODICT], tmp_path / "out", "--games", ",".join(games))
    assert st["errors"] == 0
    won = set()
    for s in samples:
        g = s["meta"]["game"]
        env = _env(g)
        obs = env.reset()
        frames, actions = [], []
        for t in range(int(s["tgt"]["type_idx"].shape[0])):
            frames.append(_frame(obs))
            typ, click = int(s["tgt"]["type_idx"][t]), int(s["tgt"]["click_idx"][t])
            actions.append((typ, click))
            obs, _, done, info = env.step({"type_idx": torch.tensor([typ]), "click_idx": torch.tensor([click])})
            if bool(done[0]):
                break
        _check(s, frames, actions)
        if bool(info["terminated"][0]):
            won.add(g)
        env.close()
    assert won == set(games), (sorted(won), games)
