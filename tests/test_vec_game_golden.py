"""Golden tests: real recorded games replayed through the VecGameEnv adapters
(env/vec_othello.py, env/vec_generals.py, env/vec_gomoku.py) reproduce, frame by frame and bit
for bit, the IL data of the converters — observations (the gm-shard obs
schema: keys, dtypes, shapes, values), the actions the rules applied (as IL
targets) and the outcome.  So the IL-pretrained encoder sees in-distribution
inputs in RL.

othello   the WTHOR fixture replayed as ONE interleaved stream per game (both
          movers, side-to-move planes, forced pass frames) vs the pinned
          digests of ``gmdata othello convert`` (tests/fixtures/golden/
          othello.json, always) and vs its gm-shards frame by frame (cargo).
generals  the .gior fixture replayed on its own map, one stream per seat, vs
          the gio records → SeqReplayDataset decoding of the Python reference
          simulator / converter (always) and vs ``gmdata generals
          decode | convert | shard`` gm-shards (cargo).  Full move-priority
          ties follow the replay's move order (``tiebreak_first``) as in the
          simulator; the env's own rule (seat 0 first) gives the same boards
          but flips executed / dropped on those ticks (a head-on swap of
          equal armies: about one game in ten of real 1v1 replays).
gomoku    Gomocup .psq records replayed as ONE interleaved stream per game
          (side-to-move planes, no pass) vs the pinned digests of ``gmdata
          gomoku convert`` on the 20x20 fixture games (always), vs the checked-in
          gm-shards of hand-written games (tests/fixtures/gomoku_env: edge /
          corner fives in all four directions, a gap fill, an overline, the
          9-move win, row-wrap traps, a 15x15 board; always), vs fresh gmdata
          conversions (cargo) and vs the real Gomocup shards
          ($GOMOKU_SHARDS, default ~/gm-data/gomoku/shards20; skipped when
          absent; a spread of shards, all with GOMOKU_GOLDEN_ALL=1).
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    GmShard,
    find_gm_shards,
    load_gm_samples,
)
from data_processing.games.generals import SAMPLE_OBS_KEYS, SAMPLE_TGT_KEYS, GeneralsGioDataset
from env.vec_generals import (
    GeneralsVecEnv,
    map_from_replay,
    replay_actions_by_step,
    replay_tiebreak_by_step,
)
from env.vec_gomoku import GomokuVecEnv
from env.vec_othello import OthelloVecEnv
from model.games.generals import append_turn_globals

sys.path.insert(0, os.path.dirname(__file__))
from golden_digests import sample_digests  # noqa: E402
from test_dial_probe import parse_wtb  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
FIX = os.path.join(ROOT, "tests", "fixtures")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
WTHOR = os.path.join(FIX, "othello", "WTH_2021.wtb")
GIOR = os.path.join(FIX, "generals", "TyvMRVMys.gior")
OTHELLO_META = ("game_index", "replayed_black", "n_pass_frames")


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


def _assert_frames_equal(ours: Dict[str, torch.Tensor], il: Dict[str, torch.Tensor], what: str):
    assert set(ours) == set(il), (what, sorted(ours), sorted(il))
    for k in il:
        a, b = ours[k], il[k]
        assert a.dtype == b.dtype and a.shape == b.shape, (what, k, a.dtype, b.dtype,
                                                           a.shape, b.shape)
        if not torch.equal(a, b):
            bad = next(t for t in range(a.shape[0]) if not torch.equal(a[t], b[t]))
            raise AssertionError(f"{what}: {k} differs first at frame {bad}")


# ---------------------------------------------------------------------------
# othello
# ---------------------------------------------------------------------------

def replay_othello(games: List[List[int]]) -> List[Dict]:
    """Replay WTHOR move lists (passes implicit) through ONE batched
    OthelloVecEnv, feeding each game's recorded square on its decision frames
    (the env inserts the forced pass frames).  Records the converter rejects
    (an illegal move, moves after the end, or a game not over after its last
    move) are dropped like it drops them."""
    G = len(games)
    env = OthelloVecEnv(G)
    obs = env.reset()
    frames: List[List[torch.Tensor]] = [[] for _ in range(G)]
    movers: List[List[int]] = [[] for _ in range(G)]
    place: List[List[int]] = [[] for _ in range(G)]
    passed: List[List[bool]] = [[] for _ in range(G)]
    ptr, bad, final = [0] * G, [False] * G, [None] * G
    while bool(obs.active.any()):
        send = torch.zeros(G, dtype=torch.long)
        for i in torch.nonzero(obs.active).squeeze(1).tolist():
            frames[i].append(obs.obs["board_seq"][i])
            movers[i].append(int(obs.to_play[i]))
            if obs.forced[i]:
                continue
            legal = obs.legal["place"][i]
            if not bad[i] and ptr[i] < len(games[i]) and legal[games[i][ptr[i]]]:
                send[i] = games[i][ptr[i]]
                ptr[i] += 1
            else:                  # rejected record: keep its env stepping, drop it later
                bad[i] = True
                send[i] = int(legal.nonzero()[0])
        obs, rew, done, info = env.step({"action_place": send})
        for i in torch.nonzero(info["acted"]).squeeze(1).tolist():
            place[i].append(int(info["applied"]["place_idx"][i]))
            passed[i].append(bool(info["applied"]["passed"][i]))
            if done[i]:
                final[i] = (rew[i].clone(), int(info["score"][i, 0]), int(info["n_pass"][i]),
                            int(info["episode_len"][i]))
    out = []
    for i in range(G):
        if bad[i] or ptr[i] != len(games[i]):
            continue
        rew, black, n_pass, ep_len = final[i]
        assert ep_len == len(frames[i]) == len(place[i])
        out.append({
            "obs": {"board_seq": torch.stack(frames[i])},
            "tgt": {"place_idx": torch.tensor(place[i], dtype=torch.int64),
                    "passed": torch.tensor(passed[i], dtype=torch.bool),
                    # the seat's final reward is the IL value target of its frames
                    "value": torch.stack([rew[m] for m in movers[i]])},
            "meta": {"game_index": i, "replayed_black": black, "n_pass_frames": n_pass},
        })
    return out


@pytest.fixture(scope="module")
def othello_replayed():
    with open(WTHOR, "rb") as fh:
        return replay_othello(parse_wtb(fh.read()))


def test_othello_replay_matches_golden_digests(othello_replayed):
    """Bit-identical to `gmdata othello convert` (its pinned per-tensor
    digests): obs.board_seq, tgt.place_idx / passed / value, per game."""
    with open(os.path.join(FIX, "golden", "othello.json")) as fh:
        golden = json.load(fh)
    got = sample_digests(othello_replayed, OTHELLO_META)
    assert len(got) == len(golden["samples"]) > 0
    for i, (g, w) in enumerate(zip(got, golden["samples"])):
        assert g == w, f"game {i} differs from the converter"
    assert sum(s["meta"]["n_pass_frames"] for s in othello_replayed) > 0     # passes exercised


def test_othello_replay_matches_gmdata_frame_by_frame(gmdata, tmp_path, othello_replayed):
    out = tmp_path / "shards"
    subprocess.run([gmdata, "othello", "convert", "--input", os.path.dirname(WTHOR),
                    "--out", str(out)], check=True, capture_output=True)
    il = load_gm_samples(str(out))
    assert len(il) == len(othello_replayed) > 0
    schema = OthelloVecEnv(1).obs_schema()
    for ours, ref in zip(othello_replayed, il):
        assert ours["meta"]["game_index"] == ref["meta"]["game_index"]
        for k, (shape, dtype) in schema.items():
            assert ref["obs"][k].shape[1:] == shape and ref["obs"][k].dtype == dtype
        _assert_frames_equal(ours["obs"], ref["obs"], f"othello game {ref['meta']['game_index']}")
        _assert_frames_equal(ours["tgt"], ref["tgt"], f"othello game {ref['meta']['game_index']}")


# ---------------------------------------------------------------------------
# generals
# ---------------------------------------------------------------------------

def replay_generals(replay: Dict, T: int, **env_kw) -> Dict:
    """Replay a recorded 1v1 game through GeneralsVecEnv on its own map (seat
    i = usernames[i]) for the IL record's ``T`` frames: per seat the frames'
    obs and the applied actions as IL targets (count / src / dir / ratio in
    slot 0 of K = 4), plus the winner if the game ended right after."""
    env_kw.setdefault("max_turns", 10**6)          # a recorded game is never truncated
    env = GeneralsVecEnv(1, **env_kw)
    obs = env.reset(maps=[map_from_replay(replay)])
    by_step = replay_actions_by_step(replay, env.grid_w)
    tiebreak = replay_tiebreak_by_step(replay)
    seats = {p: {"obs": {k: [] for k in obs.obs}, "tgt": {k: [] for k in SAMPLE_TGT_KEYS
                                                          if k != "value"}}
             for p in (0, 1)}
    K = 4
    winner = None
    for t in range(T):
        assert bool(obs.active.all()), f"game over before frame {t}"
        assert (obs.step_idx == t).all()
        acts = {"source_idx": torch.zeros(2, dtype=torch.long),
                "direction": torch.zeros(2, dtype=torch.long),
                "split": torch.zeros(2, dtype=torch.long),
                "passed": torch.ones(2, dtype=torch.bool)}
        for p, a in by_step.get(t + 1, {}).items():
            for k, v in a.items():
                acts[k][p] = v
        for p in (0, 1):
            for k in obs.obs:
                seats[p]["obs"][k].append(obs.obs[k][p])
        obs, _, done, info = env.step(acts, tiebreak_first=[tiebreak.get(t + 1, 0)])
        for p in (0, 1):
            c = int(info["count"][p])
            move = c == 1
            sy, sx = divmod(int(info["applied"]["source_idx"][p]), env.grid_w)
            slot = lambda v: [v if move else 0] + [0] * (K - 1)  # noqa: E731
            tg = seats[p]["tgt"]
            tg["count"].append(c)
            tg["src_y"].append(slot(sy))
            tg["src_x"].append(slot(sx))
            tg["dir_id"].append(slot(int(info["applied"]["direction"][p])))
            tg["ratio_id"].append(slot(int(info["applied"]["split"][p])))
        if bool(done[0]):
            winner = int(info["winner"][0])
    out = {}
    for p in (0, 1):
        out[p] = {"obs": {k: torch.stack(v) for k, v in seats[p]["obs"].items()},
                  "tgt": {k: torch.tensor(v, dtype=torch.int64)
                          for k, v in seats[p]["tgt"].items()}}
    out["winner"] = winner
    return out


def _assert_generals_equal(ours: Dict, il: List[Dict], what: str, turn_globals: bool = False):
    for p in (0, 1):
        ref = il[p]
        obs = dict(ref["obs"])
        if turn_globals:        # what the IL trainers feed a turn_globals net
            obs["globals_seq"] = append_turn_globals(obs["globals_seq"][None])[0]
        assert set(ref["obs"]) == set(SAMPLE_OBS_KEYS)
        _assert_frames_equal(ours[p]["obs"], obs, f"{what} seat {p}")
        tgt = {k: v for k, v in ref["tgt"].items() if k != "value"}
        _assert_frames_equal(ours[p]["tgt"], tgt, f"{what} seat {p}")


def _winner_seat(replay: Dict, il_first: Dict) -> int:
    """The replay's winner (the IL value's outcome sign at the last frame)."""
    v = float(il_first["tgt"]["value"][-1])        # w_T = 1: pure outcome for seat 0
    assert abs(v) == 1.0
    return 0 if v > 0 else 1


@pytest.fixture(scope="module")
def generals_python_reference(tmp_path_factory):
    """gio records of both seats from the Python reference simulator and
    converter, decoded by SeqReplayDataset (GeneralsGioDataset)."""
    from data_processing.download.deserializer import deserialize_replay
    from data_processing.parsers.generals_json import convert_simulator_to_gio
    from data_processing.processing.simulator import ReplaySimulator
    with open(GIOR, "rb") as fh:
        replay = deserialize_replay(fh.read())
    root = tmp_path_factory.mktemp("gio")
    for p, name in enumerate(replay["usernames"]):
        gio = convert_simulator_to_gio(ReplaySimulator(replay, name).simulate())
        with open(root / f"p{p}.json", "w") as fh:
            json.dump(gio, fh)
    ds = GeneralsGioDataset(str(root))
    samples = [ds[0], ds[1]]
    assert [s["meta"]["file"] for s in samples] == ["p0.json", "p1.json"]
    return replay, samples


def test_generals_replay_matches_seq_replay_dataset(generals_python_reference):
    """Both seats, every frame: obs (grid / globals / source mask) and the
    applied actions as IL targets, incl. passes and dropped moves (count)."""
    replay, il = generals_python_reference
    T = il[0]["obs"]["grid_seq"].shape[0]
    assert T == il[1]["obs"]["grid_seq"].shape[0] > 100
    ours = replay_generals(replay, T)
    _assert_generals_equal(ours, il, "generals")
    # the capture follows the last IL frame; the env names the same winner
    assert ours["winner"] == _winner_seat(replay, il[0])
    counts = torch.cat([ours[p]["tgt"]["count"] for p in (0, 1)])
    assert (counts == 1).any() and (counts == 0).any()
    schema = GeneralsVecEnv(1).obs_schema()
    for k, (shape, dtype) in schema.items():
        assert il[0]["obs"][k].shape[1:] == shape and il[0]["obs"][k].dtype == dtype


def test_generals_turn_globals_and_memory_vision_options(generals_python_reference):
    replay, il = generals_python_reference
    T = il[0]["obs"]["grid_seq"].shape[0]
    # turn_globals: the env's 7 globals == the IL trainers' append_turn_globals
    _assert_generals_equal(replay_generals(replay, T, turn_globals=True), il,
                           "generals turn_globals", turn_globals=True)
    # memory_vision (off by default: the IL data has no such memory) only
    # re-encodes sighted mountains out of vision: state 1 → 4, nothing else
    mem = replay_generals(replay, T, memory_vision=True)
    n_diff = 0
    for p in (0, 1):
        a, b = mem[p]["obs"]["grid_seq"], il[p]["obs"]["grid_seq"]
        diff = (a != b).any(-1).any(-1).any(0)                  # per channel
        assert set(torch.nonzero(diff).flatten().tolist()) <= {1, 4}
        remembered = (a[:, 4] == 1) & (b[:, 1] == 1)
        assert torch.equal((a != b).any(1), remembered)
        n_diff += int(remembered.sum())
        for k in ("globals_seq", "legal_source_mask_seq"):
            assert torch.equal(mem[p]["obs"][k], il[p]["obs"][k])
    assert n_diff > 0


def test_generals_replay_matches_gmdata_shards(gmdata, tmp_path):
    """The Rust pipeline end to end: `gmdata generals decode` → `convert`
    (both seats) → `shard`; the adapter replays the Rust-decoded game."""
    dec = subprocess.run([gmdata, "generals", "decode", GIOR], capture_output=True, text=True,
                         check=True)
    replay = json.loads(dec.stdout)
    rp = tmp_path / "replay.json"
    rp.write_text(json.dumps(replay))
    gio = tmp_path / "gio"
    gio.mkdir()
    for p, name in enumerate(replay["usernames"]):
        subprocess.run([gmdata, "generals", "convert", "--replay", str(rp), "--player", name,
                        "--out", str(gio / f"p{p}.json")], check=True, capture_output=True)
    subprocess.run([gmdata, "generals", "shard", "--input", str(gio), "--out",
                    str(tmp_path / "shards"), "--keep-order"], check=True, capture_output=True)
    ds = GameShardDataset(str(tmp_path / "shards"), game="generals")
    by_file = {ds[i]["meta"]["file"]: ds[i] for i in range(len(ds))}
    il = [by_file["p0.json"], by_file["p1.json"]]
    T = il[0]["obs"]["grid_seq"].shape[0]
    ours = replay_generals(replay, T)
    _assert_generals_equal(ours, il, "generals (gmdata)")
    assert ours["winner"] == _winner_seat(replay, il[0])


# ---------------------------------------------------------------------------
# gomoku
# ---------------------------------------------------------------------------

GOMOKU_FIX = os.path.join(FIX, "gomoku")               # real Gomocup games, 20x20
GOMOKU_ENV_FIX = os.path.join(FIX, "gomoku_env")       # hand-written games + their shards
GOMOKU_PSQ = os.path.join(GOMOKU_ENV_FIX, "psq")
#   regenerate the checked-in shards (N = 20, 15):
#   gmdata gomoku convert --input tests/fixtures/gomoku_env/psq \
#       --out tests/fixtures/gomoku_env/shards<N> --board-size <N> --rule freestyle
GOMOKU_ENV_SHARDS = {20: os.path.join(GOMOKU_ENV_FIX, "shards20"),
                     15: os.path.join(GOMOKU_ENV_FIX, "shards15")}
GOMOKU_SHARDS = os.path.expanduser(os.environ.get("GOMOKU_SHARDS", "~/gm-data/gomoku/shards20"))
GOMOKU_META = ("source", "plies", "winner")
GOMOKU_MIN_MOVES = 9                                   # gmdata gomoku convert --min-moves default
_PSQ_MOVE = re.compile(r"^\s*(\d+),(\d+)(?:,(-?\d+))?\s*$", re.ASCII)
_PSQ_DIMS = re.compile(r"(?<![0-9])([0-9]+)x([0-9]+)")


def parse_psq(text: str) -> Tuple[int, int, List[Tuple[int, int]]]:
    """(width, height, 0-based (row, col) moves) of a Piskvorky record,
    as gomoku.rs ``parse_psq``: the header's first ``<W>x<H>``, then the
    ``x,y[,time]`` lines (1-based column, row) up to the first other line."""
    lines = [ln[:-1] if ln.endswith("\r") else ln for ln in text.split("\n")]
    dims = _PSQ_DIMS.search(lines[0])
    assert dims, f"unrecognized header {lines[0]!r}"
    moves: List[Tuple[int, int]] = []
    for ln in lines[1:]:
        m = _PSQ_MOVE.match(ln.strip())
        if m:
            moves.append((int(m.group(2)) - 1, int(m.group(1)) - 1))
        elif moves:
            break
    return int(dims.group(1)), int(dims.group(2)), moves


def replay_gomoku(games: Sequence[Sequence[int]], board_size: int, *,
                  on_frame: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None
                  ) -> List[Optional[Dict]]:
    """Replay recorded move lists (place_idx = row · N + col, −1 = off the
    board) through ONE batched GomokuVecEnv, one interleaved stream per game,
    feeding each game's recorded move on its frames.  Per game: the IL sample
    (obs.board_seq, tgt.place_idx / passed = the applied actions, tgt.value =
    the seat's final reward, meta.plies / winner) or None where the converter
    drops the record (fewer than 9 moves, an illegal move, moves after the
    win, no five after the last move).  A dropped record's env keeps stepping
    on its first legal point until it ends.  ``on_frame(t, rows, board_seq)``
    takes the frames of the replayed rows instead of storing them."""
    G = len(games)
    env = GomokuVecEnv(G, board_size=board_size)
    P = env.n_points
    lens = torch.tensor([len(g) for g in games], dtype=torch.long)
    moves = torch.full((G, int(lens.max()) + 1), -1, dtype=torch.long)
    for i, g in enumerate(games):
        moves[i, :len(g)] = torch.from_numpy(np.array(g, dtype=np.int64))
    bad = lens < GOMOKU_MIN_MOVES
    frames: List[List[torch.Tensor]] = [[] for _ in range(G)]
    applied: List[List[int]] = [[] for _ in range(G)]
    ended_at = torch.full((G,), -1, dtype=torch.long)
    final = torch.zeros(G, 2)
    winner = torch.full((G,), -1, dtype=torch.long)
    obs = env.reset()
    t = 0
    while bool(obs.active.any()):
        act = obs.active
        assert (obs.step_idx[act] == t).all() and (obs.to_play[act] == t % 2).all()
        assert not obs.forced.any() and not obs.legal["pass"].any()
        legal = obs.legal["place"]
        assert torch.equal(legal, act[:, None] & (obs.obs["board_seq"][:, 2] > 0).flatten(1))
        want = moves[:, min(t, moves.shape[1] - 1)]
        ok = (want >= 0) & (want < P)
        ok &= legal.gather(1, want.clamp(0, P - 1)[:, None]).squeeze(1)
        bad |= act & ~ok
        rows = torch.nonzero(act & ~bad).squeeze(1)
        if on_frame is not None:
            on_frame(t, rows, obs.obs["board_seq"][rows])
        else:
            for i in rows.tolist():
                frames[i].append(obs.obs["board_seq"][i])
        send = torch.where(bad, legal.to(torch.uint8).argmax(-1), want.clamp(0, P - 1))
        obs, rew, done, info = env.step({"action_place": send})
        assert not info["applied"]["passed"].any() and not info["truncated"].any()
        for i in rows.tolist():
            applied[i].append(int(info["applied"]["place_idx"][i]))
        ended_at = torch.where(done, torch.full_like(ended_at, t), ended_at)
        final[done] = rew[done]
        winner = torch.where(done, info["winner"], winner)
        t += 1
    out: List[Optional[Dict]] = []
    for i in range(G):
        T = int(lens[i])
        if bool(bad[i]) or int(ended_at[i]) != T - 1 or int(winner[i]) < 0:
            out.append(None)
            continue
        assert applied[i] == [int(x) for x in games[i]]
        seat = torch.arange(T) % 2
        s = {"tgt": {"place_idx": torch.tensor(applied[i], dtype=torch.int64),
                     "passed": torch.zeros(T, dtype=torch.bool),
                     # the seat's final reward is the IL value target of its frames
                     "value": final[i][seat]},
             "meta": {"plies": T, "winner": ("black", "white")[int(winner[i])]}}
        if on_frame is None:
            s["obs"] = {"board_seq": torch.stack(frames[i])}
        out.append(s)
    return out


def replay_psq_dir(directory: str, board_size: int) -> List[Dict]:
    """Every ``board_size`` record under ``directory`` (gmdata's file order)
    replayed; the samples the converter keeps, meta.source = the file name."""
    names, games = [], []
    for f in sorted(glob.glob(os.path.join(directory, "**", "*.psq"), recursive=True)):
        with open(f, "rb") as fh:
            w, h, moves = parse_psq(fh.read().decode("utf-8", errors="replace"))
        if (w, h) != (board_size, board_size) or not moves:
            continue
        names.append(os.path.basename(f))
        games.append([r * board_size + c if 0 <= r < board_size and 0 <= c < board_size else -1
                      for r, c in moves])
    out = []
    for name, s in zip(names, replay_gomoku(games, board_size)):
        if s is not None:
            s["meta"]["source"] = name
            out.append(s)
    return out


def _assert_gomoku_samples_equal(ours: List[Dict], il: List[Dict], board_size: int) -> None:
    assert [s["meta"]["source"] for s in ours] == [s["meta"]["source"] for s in il]
    schema = GomokuVecEnv(1, board_size=board_size).obs_schema()
    for a, b in zip(ours, il):
        what = f"gomoku {b['meta']['source']}"
        for k, (shape, dtype) in schema.items():
            assert b["obs"][k].shape[1:] == shape and b["obs"][k].dtype == dtype, what
        _assert_frames_equal(a["obs"], b["obs"], what)
        _assert_frames_equal(a["tgt"], b["tgt"], what)
        assert a["meta"]["winner"] == b["meta"]["winner"], what


def test_gomoku_replay_matches_golden_digests():
    """The 20x20 Gomocup fixture games: bit-identical to `gmdata gomoku
    convert --board-size 20` (its pinned per-tensor digests)."""
    with open(os.path.join(FIX, "golden", "gomoku_freestyle.json")) as fh:
        golden = json.load(fh)
    assert golden["shard_meta"] == {"board_size": 20}
    ours = replay_psq_dir(GOMOKU_FIX, 20)
    got = sample_digests(ours, GOMOKU_META)
    assert len(got) == len(golden["samples"]) > 30
    for g, w in zip(got, golden["samples"]):
        assert g == w, f"{w['meta']['source']} differs from the converter"
    assert {s["meta"]["winner"] for s in ours} == {"black", "white"}


@pytest.mark.parametrize("board_size", [20, 15])
def test_gomoku_replay_matches_fixture_shards(board_size):
    """The hand-written records vs their checked-in gm-shards, frame by
    frame: every direction, board edges and corners, a gap fill, a freestyle
    overline, the fastest (9-move) win, and row-wrap traps (five consecutive
    point indices across a row end, which must not win); the undecided and
    moves-after-win records are dropped like the converter drops them."""
    il = load_gm_samples(GOMOKU_ENV_SHARDS[board_size])
    ours = replay_psq_dir(GOMOKU_PSQ, board_size)
    assert len(ours) == len(il) == {20: 5, 15: 2}[board_size]
    _assert_gomoku_samples_equal(ours, il, board_size)


def test_gomoku_replay_matches_gmdata_frame_by_frame(gmdata, tmp_path):
    """Fresh `gmdata gomoku convert` output: the Gomocup fixtures frame by
    frame, and the hand-written records == their checked-in shards."""
    out = tmp_path / "gomocup"
    subprocess.run([gmdata, "gomoku", "convert", "--input", GOMOKU_FIX, "--out", str(out),
                    "--board-size", "20", "--rule", "freestyle"], check=True, capture_output=True)
    _assert_gomoku_samples_equal(replay_psq_dir(GOMOKU_FIX, 20), load_gm_samples(str(out)), 20)
    for n, checked_in in GOMOKU_ENV_SHARDS.items():
        fresh = tmp_path / f"env{n}"
        subprocess.run([gmdata, "gomoku", "convert", "--input", GOMOKU_PSQ, "--out", str(fresh),
                        "--board-size", str(n), "--rule", "freestyle"],
                       check=True, capture_output=True)
        a, b = load_gm_samples(str(fresh)), load_gm_samples(checked_in)
        assert len(a) == len(b) > 0
        for x, y in zip(a, b):
            assert x["meta"] == y["meta"]
            _assert_frames_equal(x["obs"], y["obs"], f"fixture {y['meta']['source']}")
            _assert_frames_equal(x["tgt"], y["tgt"], f"fixture {y['meta']['source']}")


def _replay_real_shard(path: str) -> Tuple[int, int]:
    """Replay every game of one real gm-shard (its recorded place_idx) and
    compare every frame's planes with the shard's packed bits; returns
    (games, frames)."""
    sh = GmShard(path)
    n = int(sh.index["meta"]["board_size"])
    arr = sh.arrays()
    bits = arr["obs.board_seq"]
    info = sh.index["tensors"]["obs.board_seq"]
    assert info["storage"] == "packbits" and info["frame_shape"] == [3, n, n]
    assert info["dtype"] == "float32"
    place = np.asarray(arr["tgt.place_idx"])
    value = np.asarray(arr["tgt.value"])
    assert not np.asarray(arr["tgt.passed"]).any()
    samples = sh.index["samples"]
    starts = torch.tensor([s["start"] for s in samples], dtype=torch.long)
    games = [place[s["start"]:s["start"] + s["length"]] for s in samples]
    per = 3 * n * n

    def on_frame(t: int, rows: torch.Tensor, board_seq: torch.Tensor) -> None:
        ours = board_seq.reshape(rows.numel(), per)
        assert ((ours == 0) | (ours == 1)).all()
        il = np.unpackbits(np.asarray(bits[(starts[rows] + t).numpy()]), axis=1, count=per)
        diff = (torch.from_numpy(il).bool() != ours.bool()).any(1)
        if diff.any():
            src = samples[int(rows[diff][0])]["meta"]["source"]
            raise AssertionError(f"{path}: {src} obs.board_seq differs at frame {t}")

    out = replay_gomoku(games, n, on_frame=on_frame)
    for s, o in zip(samples, out):
        where = f"{path}: {s['meta'].get('source')}"
        assert o is not None, f"{where}: the env does not end the game on its last move"
        assert o["meta"]["winner"] == s["meta"]["winner"] and o["meta"]["plies"] == s["length"]
        v = torch.from_numpy(value[s["start"]:s["start"] + s["length"]].copy())
        assert torch.equal(o["tgt"]["value"], v), where
    return len(samples), sum(s["length"] for s in samples)


@pytest.mark.skipif(not os.path.isdir(GOMOKU_SHARDS),
                    reason=f"no real gomoku shards at {GOMOKU_SHARDS} (set GOMOKU_SHARDS)")
def test_gomoku_replay_matches_real_gomocup_shards():
    """The real Gomocup gm-shards (read-only): every game of a spread of
    shards (all of them with GOMOKU_GOLDEN_ALL=1) replayed through the env —
    every frame's planes equal, the game ends exactly on its last move with
    the recorded winner, and the seats' rewards are the value targets."""
    shards = find_gm_shards(GOMOKU_SHARDS)
    if not shards:
        pytest.skip(f"no gm-shards under {GOMOKU_SHARDS}")
    if os.environ.get("GOMOKU_GOLDEN_ALL") != "1":
        shards = shards[::max(1, len(shards) // 8)]
    games = frames = 0
    for d in shards:
        g, f = _replay_real_shard(d)
        games += g
        frames += f
    assert games > 0 and frames > games
