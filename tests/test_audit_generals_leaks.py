"""Information-leak audit of the generals.io observation pipeline.

Everything the network reads at decision time must be information the acting
player really has at that point: the gio / gm-shard obs (``grid_seq``,
``globals_seq``, ``legal_source_mask_seq``), the RL env's per-seat observation
and the decoder's legal masks.  Two mathematically clean properties are pinned
here, on the Rust converter (``gmdata generals convert`` → gio records → gm-shards)
and on the RL simulator (env/generals_env.py via env/vec_generals.py):

FUTURE (causality)
    frame t of a record is a function of the replay up to decision t only:
    cutting the replay's moves after turn t and reconverting reproduces frames
    0..t bit for bit (obs and action targets; ``tgt.value`` is hindsight by
    design and excluded).  That cut keeps the move of turn t itself, so a
    second check pins the off-by-one direction: frame t's target is the
    player's move of turn t, and deleting it (or the opponent's move of the
    same tick) leaves frames 0..t unchanged.  The network is causal too, both
    against replaced later frames and against a shorter sequence (no feature
    normalised by the sequence length).

PRIVACY (fog of war)
    player p's frames are invariant to everything p cannot see:
      * terrain under fog — generals.io draws never-seen cities and mountains
        with the same obstacle icon (TILE_FOG_OBSTACLE), so swapping a
        never-seen city for a mountain (and changing garrisons) must not change
        p's view;
      * the opponent's activity in fog — where it expands, where its general
        stands — as long as its total army and land (the public leaderboard)
        are the same;
      * in the RL env, any opponent-private state: army and ownership of fogged
        tiles (totals kept), the fogged enemy general, fogged garrisons, never
        seen obstacles.
    The opponent totals p's globals carry are exactly the opponent's own
    army/land (the leaderboard columns; no city count), and the decoder's
    direction mask is a function of p's view at every cell that can be a legal
    source (a move's destination is always inside the 3x3 vision of an owned
    tile), while its entries at other cells never reach the policy.

Known issues pinned as strict xfails (they flip to XPASS, i.e. fail, once
fixed; drop the marker then): the direction mask off a seat's own cells reads
never-seen terrain, and memory_vision is not one recorded setting across the
IL data, GeneralsEnv (train_rl.py) and GeneralsVecEnv (rl_joint).

Data: the committed .gior fixture always; the Rust tests need cargo (skipped
otherwise); the real-replay variants read a few ladder replays from
``$GENERALS_REPLAYS`` (default ``~/gm-data/generals/ladder/replays``) and skip
when it is absent (``$GENERALS_AUDIT_N`` replays, default 2).
"""
from __future__ import annotations

import copy
import glob
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "generals", "TyvMRVMys.gior")
REPLAY_DIR = os.path.expanduser(os.environ.get("GENERALS_REPLAYS",
                                               "~/gm-data/generals/ladder/replays"))
N_REAL = int(os.environ.get("GENERALS_AUDIT_N", "2"))

#: every tensor the network reads from a gio record (the gm-shard obs after expand)
OBS_KEYS = ("grid_seq", "globals_seq", "legal_source_mask_seq")
#: the per-frame action targets (decided AT frame t, so causal too)
ACTION_KEYS = ("count", "src_y", "src_x", "dir_id", "ratio_id")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fixture_replay() -> Dict:
    from data_processing.download.deserializer import deserialize_replay
    with open(FIXTURE, "rb") as fh:
        return deserialize_replay(fh.read())


def _real_replays(n: int) -> List[Dict]:
    """``n`` ladder replays spread over the sorted directory listing (1v1,
    no map features the strict simulator rejects)."""
    files = sorted(glob.glob(os.path.join(REPLAY_DIR, "*.json")))
    if not files:
        pytest.skip(f"no generals replays under {REPLAY_DIR}")
    out = []
    step = max(1, len(files) // (4 * max(n, 1)))
    for f in files[::step]:
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, ValueError):
            continue
        if (len(r.get("generals", [])) == 2 and r["mapWidth"] <= 25 and r["mapHeight"] <= 25
                and not any(r.get(k) for k in ("swamps", "deserts", "lookouts", "observatories",
                                               "lights", "neutrals", "modifiers"))
                and len(r.get("moves", [])) > 100):
            out.append(r)
        if len(out) >= n:
            break
    if not out:
        pytest.skip(f"no usable 1v1 replays under {REPLAY_DIR}")
    return out


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    build = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                           capture_output=True, text=True)
    if build.returncode != 0:
        pytest.skip(f"cargo build failed: {build.stderr[-500:]}")
    return os.path.join(WS, "target", "release", "gmdata")


class _Gm:
    """``gmdata generals`` on in-memory replays (JSON through a temp dir)."""

    def __init__(self, exe: str, tmp: Path):
        self.exe, self.tmp, self.n = exe, tmp, 0

    def _run(self, cmd: str, replay: Dict, player: str, extra=()) -> Dict:
        self.n += 1
        rp = self.tmp / f"r{self.n}.json"
        out = self.tmp / f"o{self.n}.json"
        rp.write_text(json.dumps(replay))
        # --player=NAME: ladder usernames may start with '-'
        res = subprocess.run([self.exe, "generals", cmd, "--replay", str(rp), f"--player={player}",
                              "--out", str(out), *extra], capture_output=True, text=True)
        rp.unlink()
        if res.returncode != 0:
            raise RuntimeError(f"gmdata generals {cmd} failed: {res.stderr[-400:]}")
        with open(out) as fh:
            data = json.load(fh)
        out.unlink()
        return data

    def convert(self, replay: Dict, player: str) -> Dict:
        return self._run("convert", replay, player)

    def simulate(self, replay: Dict, player: str) -> Dict:
        return self._run("simulate", replay, player)


def _first_diff(a: list, b: list) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _assert_frames_equal(a: Dict, b: Dict, n: Optional[int], what: str,
                         keys=OBS_KEYS + ACTION_KEYS) -> None:
    """gio records ``a`` and ``b`` agree bit for bit on frames [0, n) (all
    frames when ``n`` is None) — JSON floats compare exactly."""
    for k in keys:
        x = a[k] if n is None else a[k][:n]
        y = b[k] if n is None else b[k][:n]
        i = _first_diff(x, y)
        assert i is None, f"{what}: {k} differs at frame {i} (of {len(x)} / {len(y)})"


def _truncated(replay: Dict, t: int) -> Dict:
    """The replay as it stood right after decision turn ``t``: moves (and
    surrenders) of turns > t never happened."""
    r = dict(replay)
    r["moves"] = [m for m in replay["moves"] if m["turn"] <= t]
    r["afks"] = [a for a in replay.get("afks", []) if a["turn"] <= t]
    return r


def _truncation_points(replay: Dict, n_frames: int, k: int) -> List[int]:
    """``k`` move turns spread over the record (a cut right after a move turn
    t leaves a record of exactly t + 1 frames)."""
    turns = sorted({m["turn"] for m in replay["moves"] if m["turn"] < n_frames - 1})
    if len(turns) <= k:
        return turns
    idx = np.linspace(0, len(turns) - 1, k).round().astype(int)
    return sorted({turns[i] for i in idx})


# ---------------------------------------------------------------------------
# FUTURE: gio records are causal
# ---------------------------------------------------------------------------

def _check_causal(gm: _Gm, replay: Dict, k: int) -> int:
    checked = 0
    for player in replay["usernames"]:
        full = gm.convert(replay, player)
        T = len(full["count"])
        assert T > 20
        for t in _truncation_points(replay, T, k):
            part = gm.convert(_truncated(replay, t), player)
            assert len(part["count"]) == t + 1, (player, t, len(part["count"]))
            _assert_frames_equal(full, part, t + 1,
                                 f"{replay['id']} {player}: cut after turn {t}")
            checked += 1
    return checked


def test_gio_record_is_causal_fixture(gmdata, tmp_path):
    """Cutting the fixture replay after decision t (several t, both players)
    reproduces frames 0..t of the full record bit for bit — no feature is
    computed over the whole game (normalisers, game length, final score)."""
    assert _check_causal(_Gm(gmdata, tmp_path), _fixture_replay(), k=8) >= 12


def test_gio_record_is_causal_real_replays(gmdata, tmp_path):
    gm = _Gm(gmdata, tmp_path)
    for replay in _real_replays(N_REAL):
        assert _check_causal(gm, replay, k=4) >= 4


def _move_target(rec: Dict, t: int, w: int):
    """Frame ``t``'s action target as (start cell, direction, is50) in the
    replay's flat indexing (None on a frame without an executed move)."""
    if rec["count"][t] != 1:
        return None
    return rec["src_y"][t][0] * w + rec["src_x"][t][0], rec["dir_id"][t][0], rec["ratio_id"][t][0]


def _replay_move_target(m: Dict, w: int):
    from env.vec_generals import direction_of
    sr, sc = divmod(m["start"], w)
    er, ec = divmod(m["end"], w)
    return m["start"], direction_of(er - sr, ec - sc), int(bool(m["is50"]))


def _check_label_and_simultaneous_move(gm: _Gm, replay: Dict, k: int) -> int:
    """The truncation check keeps the move of turn t, so it cannot see the
    classic off-by-one leak where frame t already shows the effect of its own
    labelled move.  Here, for ``k`` frames t per player where the player
    moved: (1) frame t's target IS the player's replay move of turn t;
    (2) deleting that move leaves frames 0..t bit-identical, turns frame t's
    target into a pass and changes frame t+1 (positive control); (3)
    deleting the OPPONENT's move of the same turn (executed in the same tick,
    so unknowable at decision t) leaves the player's frames 0..t
    bit-identical while the opponent's own frame t+1 changes."""
    w = replay["mapWidth"]
    last = max(m["turn"] for m in replay["moves"])
    checked = 0
    for p, player in enumerate(replay["usernames"]):
        other = replay["usernames"][1 - p]
        full = gm.convert(replay, player)
        mine = [m for m in replay["moves"]
                if m["index"] == p and m["turn"] < last and full["count"][m["turn"]] == 1]
        assert mine, f"{replay['id']} {player}: no executed move"
        idx = np.linspace(0, len(mine) - 1, min(k, len(mine))).round().astype(int)
        for m in (mine[i] for i in sorted(set(idx))):
            t = m["turn"]
            what = f"{replay['id']} {player} turn {t}"
            assert _move_target(full, t, w) == _replay_move_target(m, w), what
            # (the game may now end differently: only frames 0..t+1 matter)
            cut = gm.convert(dict(replay, moves=[x for x in replay["moves"] if x is not m]), player)
            assert len(cut["count"]) >= t + 1, what
            _assert_frames_equal(full, cut, t + 1, f"{what}: own move deleted", keys=OBS_KEYS)
            _assert_frames_equal(full, cut, t, f"{what}: own move deleted", keys=ACTION_KEYS)
            assert cut["count"][t] == 0, what
            if len(cut["count"]) > t + 1:
                assert cut["grid_seq"][t + 1] != full["grid_seq"][t + 1], what
            theirs = [x for x in replay["moves"] if x["turn"] == t and x["index"] == 1 - p]
            if theirs:
                r2 = dict(replay, moves=[x for x in replay["moves"] if x is not theirs[0]])
                mine2 = gm.convert(r2, player)
                assert len(mine2["count"]) >= t + 1, what
                _assert_frames_equal(full, mine2, t + 1,
                                     f"{what}: opponent's simultaneous move deleted",
                                     keys=OBS_KEYS)
                a, b = gm.convert(replay, other), gm.convert(r2, other)
                if min(len(a["count"]), len(b["count"])) > t + 1:
                    assert a["grid_seq"][t + 1] != b["grid_seq"][t + 1], what
            checked += 1
    return checked


def test_gio_frame_excludes_its_own_and_the_simultaneous_move_fixture(gmdata, tmp_path):
    """Frame t pairs the state after tick t with the move executed in tick
    t + 1: neither that move (the label) nor the opponent's move of the same
    tick can show in frames 0..t (fixture, both players, 8 frames each)."""
    assert _check_label_and_simultaneous_move(_Gm(gmdata, tmp_path), _fixture_replay(), k=8) >= 12


def test_gio_frame_excludes_its_own_and_the_simultaneous_move_real_replays(gmdata, tmp_path):
    gm = _Gm(gmdata, tmp_path)
    for replay in _real_replays(N_REAL):
        assert _check_label_and_simultaneous_move(gm, replay, k=3) >= 4


def test_gm_shard_obs_is_causal(gmdata, tmp_path):
    """The same through the training format: `gmdata generals shard` +
    GameShardDataset's load-time expand (data_processing/games/generals.py)
    give the truncated record's frames exactly as the full record's prefix."""
    from data_processing.games.dataset import GameShardDataset
    replay = _fixture_replay()
    player = replay["usernames"][1]
    gm = _Gm(gmdata, tmp_path)
    full = gm.convert(replay, player)
    t = _truncation_points(replay, len(full["count"]), 3)[1]
    samples = []
    for tag, rec in (("full", full), ("cut", gm.convert(_truncated(replay, t), player))):
        d = tmp_path / tag
        d.mkdir()
        (d / "g.json").write_text(json.dumps(rec))
        subprocess.run([gmdata, "generals", "shard", "--input", str(d), "--out",
                        str(tmp_path / f"{tag}_shards")], check=True, capture_output=True)
        samples.append(GameShardDataset(str(tmp_path / f"{tag}_shards"), game="generals")[0])
    a, b = samples
    assert b["obs"]["grid_seq"].shape[0] == t + 1
    for k in OBS_KEYS:
        assert torch.equal(a["obs"][k][:t + 1], b["obs"][k]), k
    for k in ACTION_KEYS:
        assert torch.equal(a["tgt"][k][:t + 1], b["tgt"][k]), k


# ---------------------------------------------------------------------------
# PRIVACY: terrain under fog (real replays)
# ---------------------------------------------------------------------------

def _never_seen(sim: Dict, h: int, w: int) -> np.ndarray:
    """[h*w] tiles the simulated player never had in vision (states 0 / 1 =
    fog blank / fog obstacle in every frame)."""
    s = np.asarray(sim["map"], dtype=np.int16)[:, :h, :w]
    return np.all((s == 0) | (s == 1), axis=0).reshape(-1)


def _fog_terrain_perturbation(replay: Dict, never: np.ndarray):
    """The replay with every untouched never-seen obstacle altered: cities
    alternately turned into mountains or given another garrison, mountains
    alternately turned into cities.  No move starts or ends on these tiles, so
    the game plays out identically.  Returns (replay, altered tiles)."""
    touched = {m["start"] for m in replay["moves"]} | {m["end"] for m in replay["moves"]}
    touched |= set(replay["generals"])
    cand_c = [c for c in replay["cities"] if never[c] and c not in touched]
    cand_m = [m for m in replay["mountains"] if never[m] and m not in touched]
    to_mtn = set(cand_c[::2])
    regarrison = set(cand_c[1::2])
    to_city = cand_m[::2]
    cities, armies = [], []
    for c, a in zip(replay["cities"], replay["cityArmies"]):
        if c in to_mtn:
            continue
        cities.append(c)
        armies.append(a + 7 if c in regarrison else a)
    for i, m in enumerate(to_city):
        cities.append(m)
        armies.append(40 + i % 11)
    mountains = [m for m in replay["mountains"] if m not in set(to_city)] + sorted(to_mtn)
    r = dict(replay, cities=cities, cityArmies=armies, mountains=mountains)
    return r, sorted(to_mtn | regarrison | set(to_city))


def _check_fog_terrain(gm: _Gm, replay: Dict) -> int:
    h, w = replay["mapHeight"], replay["mapWidth"]
    n_altered = 0
    for p, player in enumerate(replay["usernames"]):
        other = replay["usernames"][1 - p]
        never = _never_seen(gm.simulate(replay, player), h, w)
        pert, altered = _fog_terrain_perturbation(replay, never)
        if not altered:
            continue
        n_altered += len(altered)
        _assert_frames_equal(gm.convert(replay, player), gm.convert(pert, player), None,
                             f"{replay['id']} {player}: never-seen terrain altered at {altered}")
        # positive control: the opponent did see (some of) these tiles
        never_q = _never_seen(gm.simulate(replay, other), h, w)
        if not all(never_q[i] for i in altered):
            a, b = gm.convert(replay, other), gm.convert(pert, other)
            assert a["grid_seq"] != b["grid_seq"], "perturbation invisible to the opponent too"
    return n_altered


def test_gio_invariant_to_fogged_terrain_fixture(gmdata, tmp_path):
    """generals.io shows never-seen cities and mountains as the same
    obstacle: p's record must not change when a never-seen city becomes a
    mountain (or vice versa) or a never-seen garrison changes."""
    assert _check_fog_terrain(_Gm(gmdata, tmp_path), _fixture_replay()) > 0


def test_gio_invariant_to_fogged_terrain_real_replays(gmdata, tmp_path):
    gm = _Gm(gmdata, tmp_path)
    assert sum(_check_fog_terrain(gm, r) for r in _real_replays(N_REAL)) > 0


# ---------------------------------------------------------------------------
# PRIVACY: the opponent's play in fog (synthetic replays)
# ---------------------------------------------------------------------------

_W = _H = 12
# the opponent's moves as (turn, from, to) offsets from its general
_Q_MOVES = [(10, (0, 0), (0, -1)), (11, (0, -1), (0, -2)), (12, (0, -2), (0, -3)),
            (13, (0, -3), (0, -4)), (24, (0, 0), (-1, 0)), (25, (-1, 0), (-2, 0)),
            (26, (-2, 0), (-3, 0)), (40, (0, 0), (1, 0)), (52, (0, 0), (0, -1)),
            (53, (0, -1), (0, -2)), (56, (-3, 0), (-4, 0))]
# player 0's own moves (absolute cells), all within rows 0..2 → vision rows 0..3
_P_MOVES = [(10, (1, 1), (1, 2)), (11, (1, 2), (1, 3)), (12, (1, 3), (1, 4)),
            (20, (1, 1), (2, 1)), (21, (2, 1), (2, 2)), (40, (1, 1), (0, 1)),
            (55, (1, 1), (1, 0)), (60, (1, 2), (0, 2)), (61, (2, 2), (2, 3))]


def _synthetic_replay(variant: str) -> Dict:
    """A 12x12 1v1 whose opponent (index 1) plays far outside player 0's
    vision.  Variants give the opponent the same army / land totals at every
    tick with different private details:

      A  general (10, 10); expands left then up; a mountain at (10, 5) and
         a 40-army city at (5, 10), both never seen by player 0
      B  the opponent's whole half transposed about its general (expands up
         then left; the fogged mountain and city trade places)
      C  like A with the general on (10, 11) and every move shifted with it
         (fog obstacles stay put: their positions are public in generals.io)
    Player 0's map corner (a mountain, a 44-army city) and moves are shared."""
    idx = lambda r, c: r * _W + c              # noqa: E731
    g1 = (10, 11) if variant == "C" else (10, 10)

    def q(off):
        dr, dc = (off[1], off[0]) if variant == "B" else off
        return idx(g1[0] + dr, g1[1] + dc)

    moves = [{"index": 0, "start": idx(*a), "end": idx(*b), "is50": 0, "turn": t}
             for t, a, b in _P_MOVES]
    moves += [{"index": 1, "start": q(a), "end": q(b), "is50": 0, "turn": t}
              for t, a, b in _Q_MOVES]
    moves.sort(key=lambda m: (m["turn"], m["index"]))
    return {
        "version": 19, "id": f"synthetic{variant}", "mapWidth": _W, "mapHeight": _H,
        "usernames": ["P0", "P1"], "stars": [60, 60],
        "cities": [idx(3, 0), idx(10, 5) if variant == "B" else idx(5, 10)],
        "cityArmies": [44, 40], "generals": [idx(1, 1), idx(*g1)],
        "mountains": [idx(0, 4), idx(5, 10) if variant == "B" else idx(10, 5)],
        "moves": moves, "afks": [], "teams": None, "map": None, "neutrals": [],
        "neutralArmies": [], "swamps": [], "chat": [], "playerColors": [0, 1], "lights": [],
        "speed": 1, "city_density": 0.5, "mountain_density": 0.5, "swamp_density": 0,
        "modifiers": [], "observatories": [], "lookouts": [], "deserts": [],
        "city_fairness": 0.5, "spawn_fairness": 0.5, "desert_density": 0,
        "lookout_density": 0, "observatory_density": 0,
    }


def test_gio_invariant_to_opponent_play_in_fog(gmdata, tmp_path):
    """Player 0's record is identical whether the opponent expanded left or
    up, stood its general on another fogged tile, or had its fogged city and
    mountain swapped — its totals (the leaderboard) being equal; the
    opponent's own records do differ (the variants are not trivial)."""
    gm = _Gm(gmdata, tmp_path)
    recs = {v: {p: gm.convert(_synthetic_replay(v), p) for p in ("P0", "P1")} for v in "ABC"}
    T = len(recs["A"]["P0"]["count"])
    assert T == 62
    for v in "BC":
        _assert_frames_equal(recs["A"]["P0"], recs[v]["P0"], None, f"P0 view, variant A vs {v}")
        assert recs["A"]["P1"]["grid_seq"] != recs[v]["P1"]["grid_seq"]
        assert recs["A"]["P1"]["globals_seq"] == recs[v]["P1"]["globals_seq"]
    # the opponent's moves were real (executed), not dropped no-ops
    assert sum(c == 1 for c in recs["A"]["P1"]["count"]) == len(_Q_MOVES)
    assert sum(c == 1 for c in recs["A"]["P0"]["count"]) == len(_P_MOVES)


def test_gio_fog_encoding_semantics(gmdata, tmp_path):
    """What player 0 sees of the synthetic opponent at every frame: never-seen
    mountain and city → the one fog-obstacle plane (1) with no army; the
    enemy general and every enemy tile in fog → fog-blank (0), no army; the
    source mask = own tiles with army > 1."""
    rec = _Gm(gmdata, tmp_path).convert(_synthetic_replay("A"), "P0")
    grid = np.asarray(rec["grid_seq"], dtype=np.float32)          # [T, 12, 25, 25]
    legal = np.asarray(rec["legal_source_mask_seq"], dtype=bool)[:, 0]
    rc = lambda i: divmod(i, _W)                                   # noqa: E731
    mtn, city, gen = rc(10 * _W + 5), rc(5 * _W + 10), rc(10 * _W + 10)
    for (r, c), plane in ((mtn, 1), (city, 1), (gen, 0)):
        assert (grid[:, plane, r, c] == 1).all() and (grid[:, 11, r, c] == 0).all()
    q_cells = {(10, 10 - k) for k in range(5)} | {(10 - k, 10) for k in range(5)} | {(11, 10)}
    for r, c in q_cells:
        assert (grid[:, 0, r, c] == 1).all() and (grid[:, 11, r, c] == 0).all(), (r, c)
    own = grid[:, 5:8].sum(1) > 0
    army = np.expm1(grid[:, 11].astype(np.float64) * 10.0)
    assert np.array_equal(legal, own & (army > 1.5))
    # everything outside the 12x12 map is padding (all-zero one-hot)
    assert (grid[:, :11, _H:, :] == 0).all() and (grid[:, :11, :, _W:] == 0).all()


def test_gio_opponent_globals_are_the_leaderboard(gmdata, tmp_path):
    """globals = log1p(own army, own land, opp army, opp land) / 10: the
    opponent entries of one player's record are exactly the other player's
    own totals (generals.io's leaderboard shows every player's army and land;
    no city count or other hidden aggregate is fed)."""
    gm = _Gm(gmdata, tmp_path)
    replay = _fixture_replay()
    a, b = (gm.convert(replay, p) for p in replay["usernames"])
    ga, gb = np.asarray(a["globals_seq"]), np.asarray(b["globals_seq"])
    assert ga.shape == gb.shape and ga.shape[1] == 4
    assert np.array_equal(ga[:, 2:], gb[:, :2]) and np.array_equal(gb[:, 2:], ga[:, :2])
    # own army / land recomputed from the player's own grid
    grid = np.asarray(a["grid_seq"], dtype=np.float64)
    own = grid[:, 5:8].sum(1) > 0
    army = np.rint(np.expm1(grid[:, 11] * 10.0))
    assert np.allclose(np.expm1(ga[:, 1] * 10.0), own.sum((1, 2)))
    assert np.allclose(np.expm1(ga[:, 0] * 10.0), (army * own).sum((1, 2)), rtol=1e-4)


def test_gm_shard_schema_is_the_audited_one(gmdata, tmp_path):
    """The shard carries exactly the audited inputs (a new obs.* / raw.*
    tensor needs its own leak audit before it lands here)."""
    replay = _fixture_replay()
    d = tmp_path / "gio"
    d.mkdir()
    rec = _Gm(gmdata, tmp_path).convert(replay, replay["usernames"][0])
    (d / "g.json").write_text(json.dumps(rec))
    subprocess.run([gmdata, "generals", "shard", "--input", str(d), "--out", str(tmp_path / "s")],
                   check=True, capture_output=True)
    with open(tmp_path / "s" / "generals_00000" / "index.json") as fh:
        idx = json.load(fh)
    inputs = {k for k in idx["tensors"] if not k.startswith("tgt.")}
    assert inputs == {"obs.globals_seq", "raw.grid_onehot", "raw.army", "raw.legal_source_mask"}
    from data_processing.games.generals import OBS_KEYS as EXPANDED
    from data_processing.games.generals import SAMPLE_OBS_KEYS
    assert set(SAMPLE_OBS_KEYS) == set(OBS_KEYS) and set(EXPANDED) <= set(OBS_KEYS)


# ---------------------------------------------------------------------------
# PRIVACY: the RL simulator's per-seat observation (env/generals_env.py)
# ---------------------------------------------------------------------------

def _vision(env, p: int):
    """(own, in-vision) [H, W] bool of seat p: owned tiles and the 3x3
    neighbourhood of every owned tile, inside the real map."""
    H, W = env.grid_h, env.grid_w
    own = np.zeros(H * W, dtype=bool)
    own[env.player_land[p]] = True
    own = own.reshape(H, W)
    pad = np.zeros((H + 2, W + 2), dtype=bool)
    pad[1:-1, 1:-1] = own
    near = np.zeros((H, W), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= pad[dr:dr + H, dc:dc + W]
    inb = np.zeros((H, W), dtype=bool)
    inb[:env.real_h, :env.real_w] = True
    return own, near & inb


def _perturb_private(env, p: int, never: np.ndarray, rng: random.Random):
    """A copy of ``env`` that differs only in what seat p cannot observe now
    (and, for static terrain, has never observed): the opponent's fogged
    armies shuffled / transferred and fogged tiles relocated (its total army
    and land unchanged), a fogged city of its own made neutral (its city
    count changes, army and land do not), its fogged never-seen general
    moved, fogged neutral garrisons changed, never-seen neutral cities /
    mountains swapped."""
    from env.generals_env import _BLANK, _CITY, _GENERAL, _MOUNTAIN
    e = copy.deepcopy(env)
    q = 1 - p
    H, W = e.grid_h, e.grid_w
    _, vis = _vision(e, p)
    inb = np.zeros((H, W), dtype=bool)
    inb[:e.real_h, :e.real_w] = True
    fog = (inb & ~vis).reshape(-1)
    never = never.reshape(-1) & fog
    tile = lambda i: e.map[i // W][i % W]                          # noqa: E731

    fq = [i for i in e.player_land[q] if fog[i]]
    armies = [tile(i).army for i in fq]
    rng.shuffle(armies)
    for i, a in zip(fq, armies):
        tile(i).army = a
    rich = [i for i in fq if tile(i).army >= 3]
    if rich and len(fq) > 1:
        a, b = rng.choice(rich), rng.choice([i for i in fq if i not in rich] or fq)
        if a != b:
            tile(a).army -= 1
            tile(b).army += 1
    free = [i for i in range(H * W)
            if fog[i] and tile(i).terrain == _BLANK and tile(i).occupied == -1]
    blank_q = [i for i in fq if tile(i).terrain == _BLANK]
    rng.shuffle(free)
    rng.shuffle(blank_q)
    for src, dst in list(zip(blank_q, free))[:max(1, len(blank_q) // 2)]:
        tile(dst).occupied, tile(dst).army = q, tile(src).army
        tile(src).occupied, tile(src).army = -1, 0
        e.player_land[q].remove(src)
        e.player_land[q].append(dst)
    # a fogged city of the opponent's changes hands (an obstacle icon either
    # way): it goes neutral and the opponent holds a fogged blank tile with
    # that army instead — army and land totals kept, its city count is not
    owned_c = [i for i in e.player_land[q] if fog[i] and tile(i).terrain == _CITY]
    free = [i for i in free if tile(i).occupied == -1]
    if owned_c and free:
        c, dst = rng.choice(owned_c), free[-1]
        tile(dst).occupied, tile(dst).army = q, tile(c).army
        tile(c).occupied = -1
        e.player_land[q].remove(c)
        e.player_land[q].append(dst)
    gq = e.generals[q]
    spots = [i for i in e.player_land[q] if never[i] and tile(i).terrain == _BLANK]
    if never[gq] and spots:
        new = rng.choice(spots)
        tile(gq).terrain, tile(new).terrain = _BLANK, _GENERAL
        e.generals[q] = new
    for c in e.cities:
        if fog[c] and tile(c).occupied == -1:
            tile(c).army += rng.randint(1, 20)
    for i in map(int, np.flatnonzero(never)):
        t = tile(i)
        if t.occupied != -1 or rng.random() < 0.5:
            continue
        if t.terrain == _CITY:
            t.terrain, t.army = _MOUNTAIN, 0
            e.cities.remove(i)
            e.mountains.append(i)
        elif t.terrain == _MOUNTAIN:
            t.terrain, t.army = _CITY, rng.randint(40, 50)
            e.mountains.remove(i)
            e.cities.append(i)
    return e


_OBS_FIELDS = ("grid", "globals", "legal_mask", "capture_label", "capture_mask")


def _replay_frames(replay: Dict, **vec_kw):
    """Replay a recorded 1v1 through GeneralsVecEnv on its own map (seat i =
    usernames[i]), yielding ``(t, vec, obs)`` at every frame before stepping
    the recorded moves of turn t (with the replay's move-order tiebreak) —
    the simulator's frames: up to the capture, or up to the frame after the
    last recorded move (a game ended by surrender has no capture)."""
    from env.vec_generals import (
        GeneralsVecEnv,
        map_from_replay,
        replay_actions_by_step,
        replay_tiebreak_by_step,
    )
    vec = GeneralsVecEnv(1, max_turns=10 ** 6, **vec_kw)
    obs = vec.reset(maps=[map_from_replay(replay)])
    by_step = replay_actions_by_step(replay, vec.grid_w)
    tiebreak = replay_tiebreak_by_step(replay)
    end = max(m["turn"] for m in replay["moves"]) + 2
    t = 0
    while bool(obs.active.all()) and t < end:
        yield t, vec, obs
        acts = {"source_idx": torch.zeros(2, dtype=torch.long),
                "direction": torch.zeros(2, dtype=torch.long),
                "split": torch.zeros(2, dtype=torch.long),
                "passed": torch.ones(2, dtype=torch.bool)}
        for p, a in by_step.get(t + 1, {}).items():
            for k, v in a.items():
                acts[k][p] = v
        obs, _, _, _ = vec.step(acts, tiebreak_first=[tiebreak.get(t + 1, 0)])
        t += 1


def _replay_env_privacy(replay: Dict, memory_vision: bool, every: int = 1, seed: int = 0):
    """At every ``every``-th frame of the replayed game and for each seat,
    compare its observation (what the vec env hands the learner's row) with
    the observation of an opponent-private perturbation of the state.
    Returns (#frames checked, #frames where the perturbation changed the
    opponent's own view)."""
    rng = random.Random(seed)
    seen: Optional[List[np.ndarray]] = None
    checked = changed = 0
    for t, vec, obs in _replay_frames(replay, memory_vision=memory_vision, turn_globals=True):
        env = vec.envs[0]
        if seen is None:
            seen = [np.zeros((env.grid_h, env.grid_w), dtype=bool) for _ in (0, 1)]
        for p in (0, 1):
            seen[p] |= _vision(env, p)[1]
        if t % every:
            continue
        for p in (0, 1):
            mine = vec._obs[0][p]
            assert torch.equal(obs.obs["grid_seq"][p], mine["grid"])
            assert torch.equal(obs.obs["globals_seq"][p], mine["globals"])
            assert torch.equal(obs.legal["source"][p], mine["legal_mask"])
            pert = _perturb_private(env, p, ~seen[p], rng)
            theirs = pert._get_observation(p)
            for k in _OBS_FIELDS:
                assert torch.equal(mine[k], theirs[k]), \
                    f"{replay['id']} frame {t} seat {p}: {k} depends on opponent-private state"
            changed += not torch.equal(pert._get_observation(1 - p)["grid"],
                                       vec._obs[0][1 - p]["grid"])
            checked += 1
    return checked, changed


@pytest.mark.parametrize("memory_vision", [False, True])
def test_env_obs_invariant_to_opponent_private_state(memory_vision):
    """Every frame of the fixture, both seats: grid, globals (incl.
    turn_globals), source mask and the capture aux labels are unchanged by
    any opponent-private perturbation of the simulator state — with and
    without the sighted-mountain memory."""
    checked, changed = _replay_env_privacy(_fixture_replay(), memory_vision)
    assert checked > 200
    assert changed > checked // 4        # the perturbations are real for the other seat


def test_env_obs_invariant_to_fogged_city_ownership():
    """The opponent's city count is not on generals.io's leaderboard: on the
    synthetic map, with the opponent holding the fogged (5, 10) city, seat 0's
    view must not change when that city goes neutral and the opponent holds a
    fogged blank tile with the same army instead (totals kept)."""
    from env.generals_env import _CITY
    from env.vec_generals import GeneralsVecEnv, map_from_replay
    vec = GeneralsVecEnv(1, max_turns=10 ** 6, memory_vision=True, turn_globals=True)
    vec.reset(maps=[map_from_replay(_synthetic_replay("A"))])
    env = vec.envs[0]
    W = env.grid_w
    city = 5 * W + 10
    assert env.map[5][10].terrain == _CITY
    env.map[5][10].occupied, env.map[5][10].army = 1, 9
    env.player_land[1].append(city)
    base = env._observations()
    never = ~_vision(env, 0)[1]
    for seed in range(8):
        pert = _perturb_private(env, 0, never, random.Random(seed))
        assert pert.map[5][10].occupied == -1           # the ownership branch fired
        assert pert._player_stats(1)["cities"] < env._player_stats(1)["cities"]
        assert pert._player_stats(1)["army"] == env._player_stats(1)["army"]
        assert pert._player_stats(1)["land"] == env._player_stats(1)["land"]
        theirs = pert._get_observation(0)
        for k in _OBS_FIELDS:
            assert torch.equal(base[0][k], theirs[k]), k


def test_env_obs_invariant_to_opponent_private_state_real_replays():
    for i, replay in enumerate(_real_replays(N_REAL)):
        checked, changed = _replay_env_privacy(replay, memory_vision=True, every=3, seed=i)
        assert checked > 20 and changed > 0


def _dir_ok_from_view(grid: torch.Tensor) -> torch.Tensor:
    """[4, H, W] directions whose destination is on the map and not a
    mountain, as far as the player's own grid tells (off-map = all-zero one-hot)."""
    from env.generals_env import _DIR_DC, _DIR_DR
    onehot = grid[:11]
    blocked = (onehot.sum(0) == 0) | (onehot[4] == 1)
    H, W = blocked.shape
    pad = torch.ones(H + 2, W + 2, dtype=torch.bool)
    pad[1:-1, 1:-1] = blocked
    return torch.stack([~pad[1 + _DIR_DR[d]:1 + _DIR_DR[d] + H, 1 + _DIR_DC[d]:1 + _DIR_DC[d] + W]
                        for d in range(4)])


def test_env_direction_mask_is_observable_at_own_cells():
    """legal["direction"] is built from the true map (static, player
    independent), but at every cell a seat owns — the only cells that can be
    legal sources — it equals what the seat's own grid shows (a destination
    is always inside the 3x3 vision of the owned source)."""
    n = 0
    for t, _vec, obs in _replay_frames(_fixture_replay()):
        for p in (0, 1):
            grid = obs.obs["grid_seq"][p]
            own = grid[5:8].sum(0) > 0
            dm = obs.legal["direction"][p]
            assert torch.equal(dm[:, own], _dir_ok_from_view(grid)[:, own]), (t, p)
            n += int(own.sum())
    assert n > 1000


def _env_rows(variant: str) -> List[Dict[str, torch.Tensor]]:
    """Every frame of the synthetic replay ``variant`` played through
    GeneralsVecEnv (memory_vision and turn_globals on): per frame, both seats'
    rows of everything the learner is handed."""
    out = []
    for _t, _vec, obs in _replay_frames(_synthetic_replay(variant), memory_vision=True,
                                        turn_globals=True):
        out.append({"grid": obs.obs["grid_seq"].clone(), "globals": obs.obs["globals_seq"].clone(),
                    "source": obs.legal["source"].clone(),
                    "direction": obs.legal["direction"].clone(),
                    "capture_label": obs.aux["capture_label"].clone(),
                    "capture_mask": obs.aux["capture_mask"].clone(),
                    "forced": obs.forced.clone()})
    return out


def test_env_obs_invariant_to_opponent_play_in_fog():
    """The RL env's version of test_gio_invariant_to_opponent_play_in_fog:
    seat 0's grid, globals, source mask, forced flag and capture aux labels
    are identical at every frame whether the opponent expanded left or up,
    stood on another fogged general tile, or had its never-seen city and
    mountain swapped (equal leaderboard totals); seat 1's own rows differ."""
    rows = {v: _env_rows(v) for v in "ABC"}
    assert len(rows["A"]) > 60
    keys = ("grid", "globals", "source", "capture_label", "capture_mask", "forced")
    for v in "BC":
        assert len(rows[v]) == len(rows["A"])
        for t, (a, b) in enumerate(zip(rows["A"], rows[v])):
            for k in keys:
                assert torch.equal(a[k][0], b[k][0]), f"variant {v} frame {t}: seat 0 {k}"
        assert any(not torch.equal(a["grid"][1], b["grid"][1])
                   for a, b in zip(rows["A"], rows[v]))


@pytest.mark.xfail(strict=True, reason=(
    "confirmed leak (audit): GeneralsEnv.dir_valid is built from the true map, so "
    "legal['direction'] off a seat's own cells tells a never-seen mountain from a "
    "never-seen city; no action depends on it today (decoders read it only at legal "
    "sources), fix = build it per seat from the seat's view"))
def test_env_direction_mask_invariant_to_opponent_private_terrain():
    """The decoder's direction mask is decision-time input, so it must obey
    the same privacy invariance as the observation: seat 0's
    legal['direction'] must not change when never-seen fog obstacles trade
    places (variants A vs B: the fogged mountain (10, 5) and city (5, 10)
    swap) or the opponent's general moves (C)."""
    rows = {v: _env_rows(v) for v in "ABC"}
    for v in "BC":
        for t, (a, b) in enumerate(zip(rows["A"], rows[v])):
            assert torch.equal(a["direction"][0], b["direction"][0]), \
                f"variant {v} frame {t}: seat 0 direction mask reads fogged terrain"


@pytest.mark.xfail(strict=True, reason=(
    "confirmed deploy mismatch (audit): GeneralsEnv defaults memory_vision=True "
    "(training/train_rl.py never sets it), GeneralsVecEnv / rl_joint default False, "
    "the IL records have no memory and ModelConfig does not record the setting"))
def test_memory_vision_is_one_recorded_setting():
    """Which observation a generals checkpoint expects (pure fog as in the IL
    records, or fog + sighted-mountain memory) must be one setting: the same
    default in both env entry points and recorded in the model config (like
    turn_globals), so a live client can reproduce it."""
    from env.generals_env import GeneralsEnv
    from env.vec_generals import GeneralsVecEnv
    from model.config import ModelConfig
    assert GeneralsEnv().memory_vision == GeneralsVecEnv(1).memory_vision
    assert "memory_vision" in ModelConfig().__dict__


@pytest.mark.parametrize("norm", ["group", "batch"])
@pytest.mark.parametrize("mixer,joint,act_steps", [("attention", False, 1),
                                                    ("recurrent", True, 1),
                                                    ("attention", True, 3)])
def test_network_prefix_run_equals_full_run(mixer, joint, act_steps, norm):
    """Complement to test_network_decision_at_t_ignores_later_frames, which
    keeps the sequence length T: a feature normalised by T (the game length
    in IL, where a record is the whole game) would pass it.  Running only the
    prefix [0, t] must give frames 0..t exactly the outputs of the full run
    (eval mode, teacher-forced sources, TRM pondering depth 1 and 3; both
    encoder layouts: the default GroupNorm and legacy BatchNorm checkpoints).
    "Exactly" up to fp32 rounding: the CPU conv kernels block B·(t+1) and B·T
    frames differently (~1e-6 relative, up to ~1e-5 on the O(5) belief
    tokens), hence rtol 1e-5 — a T-normalised feature moves outputs by O(1/T)."""
    from model.config import ModelConfig
    from model.network import GeneralsNetwork
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
                      trm_block_layers=1, temporal_mixer=mixer, joint_decoder=joint,
                      predict_next_obs=True, turn_globals=True, global_features=7,
                      encoder_norm=norm)
    net = GeneralsNetwork(cfg).eval()
    B, T = 2, 9
    g = torch.Generator().manual_seed(3)
    grid = torch.randn(B, T, 12, 25, 25, generator=g)
    glb = torch.randn(B, T, cfg.global_features, generator=g)
    lm = torch.rand(B, T, 1, 25, 25, generator=g) > 0.8
    lm.view(B, T, -1)[..., 0] = True
    src = lm.view(B, T, -1).float().reshape(B * T, -1).multinomial(1, generator=g).view(B, T)
    dm = torch.rand(B, T, 4, 25, 25, generator=g) > 0.3

    def run(n):
        return net(grid[:, :n], glb[:, :n], legal_mask=lm[:, :n], direction_mask=dm[:, :n],
                   source_y=src[:, :n] // 25, source_x=src[:, :n] % 25, act_steps=act_steps)

    with torch.no_grad():
        full = run(T)
        checked = 0
        for t in (0, 3, 6):
            part = run(t + 1)
            for k, v in full.items():
                if torch.is_tensor(v) and v.dim() >= 2 and v.shape[:2] == (B, T):
                    assert part[k].shape[1] == t + 1, k
                    assert torch.allclose(v[:, :t + 1].float(), part[k].float(),
                                          atol=1e-5, rtol=1e-5), (k, t)
                    checked += 1
    assert checked >= 15


@pytest.mark.parametrize("mixer,joint", [("attention", False), ("recurrent", True)])
def test_network_decision_at_t_ignores_later_frames(mixer, joint):
    """The Stage-1 temporal state and the decode context (absolute frame
    index for the pass head) are causal: replacing every frame after t by
    other frames leaves every output of frames 0..t unchanged (eval mode, as
    in play; teacher-forced sources, so no sampling is involved)."""
    from model.config import ModelConfig
    from model.network import GeneralsNetwork
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
                      trm_block_layers=1, temporal_mixer=mixer, joint_decoder=joint,
                      predict_next_obs=True)
    net = GeneralsNetwork(cfg).eval()
    B, T, t = 2, 7, 3
    g = torch.Generator().manual_seed(2)
    grid = torch.randn(B, T, 12, 25, 25, generator=g)
    glb = torch.randn(B, T, cfg.global_features, generator=g)
    lm = torch.rand(B, T, 1, 25, 25, generator=g) > 0.8
    lm.view(B, T, -1)[..., 0] = True
    src = lm.view(B, T, -1).float().reshape(B * T, -1).multinomial(1, generator=g).view(B, T)
    grid2, glb2, lm2, src2 = grid.clone(), glb.clone(), lm.clone(), src.clone()
    grid2[:, t + 1:] = torch.randn(B, T - t - 1, 12, 25, 25, generator=g)
    glb2[:, t + 1:] = torch.randn(B, T - t - 1, cfg.global_features, generator=g)
    lm2[:, t + 1:] = torch.rand(B, T - t - 1, 1, 25, 25, generator=g) > 0.5
    lm2.view(B, T, -1)[..., 0] = True
    src2[:, t + 1:] = 0
    with torch.no_grad():
        a = net(grid, glb, legal_mask=lm, source_y=src // 25, source_x=src % 25, act_steps=1)
        b = net(grid2, glb2, legal_mask=lm2, source_y=src2 // 25, source_x=src2 % 25, act_steps=1)
    n = 0
    for k, v in a.items():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[:2] == (B, T):
            assert torch.allclose(v[:, :t + 1].float(), b[k][:, :t + 1].float(),
                                  atol=1e-5, rtol=0), k
            n += 1
    assert n >= 5


@pytest.mark.parametrize("joint", [False, True])
def test_decoder_ignores_direction_mask_outside_legal_sources(joint):
    """The direction-mask entries at cells that are not legal sources (where
    the env's static mask does encode fogged terrain) never reach the policy:
    rollout step and teacher-forced replay give bit-identical outputs when
    only those entries change."""
    from model.config import ModelConfig
    from model.network import GeneralsNetwork
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=64, n_heads=4, core_type="trm", trm_H_cycles=1, trm_L_cycles=2,
                      trm_block_layers=1, temporal_mixer="recurrent", joint_decoder=joint)
    net = GeneralsNetwork(cfg).eval()
    B = 4
    g = torch.Generator().manual_seed(1)
    grid = torch.randn(B, 12, 25, 25, generator=g)
    glb = torch.randn(B, cfg.global_features, generator=g)
    lm = torch.rand(B, 1, 25, 25, generator=g) > 0.9
    lm.view(B, -1)[:, 0] = True                     # every row has a legal source
    dm1 = torch.rand(B, 4, 25, 25, generator=g) > 0.3
    dm2 = dm1.clone()
    off = ~lm.expand(-1, 4, -1, -1)
    dm2[off] = (torch.rand(B, 4, 25, 25, generator=g) > 0.5)[off]
    assert (dm1 != dm2).any()

    def same(a, b, what):
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k], b[k]), f"{what}: {k}"

    with torch.no_grad():
        for seed in range(3):
            torch.manual_seed(seed)
            o1 = net.step(grid, glb, legal_mask=lm, direction_mask=dm1, act_steps=1)
            torch.manual_seed(seed)
            o2 = net.step(grid, glb, legal_mask=lm, direction_mask=dm2, act_steps=1)
            same(o1, o2, "step")
        legal_src = lm.view(B, -1).float().multinomial(3, replacement=True, generator=g)  # [B, 3]
        kw = dict(legal_mask=lm.unsqueeze(1).expand(-1, 3, -1, -1, -1),
                  source_y=legal_src // 25, source_x=legal_src % 25, act_steps=1)
        gs, gl = grid.unsqueeze(1).expand(-1, 3, -1, -1, -1), glb.unsqueeze(1).expand(-1, 3, -1)
        f1 = net(gs, gl, direction_mask=dm1.unsqueeze(1).expand(-1, 3, -1, -1, -1), **kw)
        f2 = net(gs, gl, direction_mask=dm2.unsqueeze(1).expand(-1, 3, -1, -1, -1), **kw)
        same(f1, f2, "forward")
