"""Differential tests of the gmrules xiangqi engine against an independent reference
implementation, Fairy-Stockfish (pyffish 0.0.90, its ``xiangqi`` variant: Asian rules
with AXF chasing).

- committed fixture (tests/fixtures/xiangqi/ffish_reference.json.gz, written by
  scripts/reference/ffish_xiangqi_dump.py from Fairy-Stockfish's own random playouts
  — with repetition / check biases, from the initial position, endgames and the AXF
  examples of Fairy-Stockfish's test suite): at every step gmsim.State must agree on
  the FEN (the transition, the side to move, the no-capture clock), the legal move
  set, whether the side to move is in check (the frame's globals), and at the end on
  the game end and its result (repetition with perpetual check / chase, move limit,
  checkmate, stalemate, insufficient material);
- live (``GM_REFERENCE=1``, needs uv + network the first time): fresh Fairy-Stockfish
  playouts, AND gmrules' own random playouts replayed through Fairy-Stockfish — the
  randomized differential test in both directions.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "xiangqi" / "ffish_reference.json.gz"
DUMP = ROOT / "scripts" / "reference" / "ffish_xiangqi_dump.py"
START = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
#: gmrules end reasons → the reference dump's labels
REASON = {"checkmate": "checkmate", "stalemate": "stalemate", "material": "material",
          "move_limit": "move_limit", "repetition": "repetition",
          "perpetual_check": "repetition", "perpetual_chase": "repetition"}


def config_of(fen: str) -> dict:
    return {"fen": "" if fen == START else fen}


def compare_trajectory(traj) -> int:
    """Step gmsim.State through a reference trajectory, comparing every step; returns
    the number of positions compared."""
    st = gmsim.State("xiangqi", config_of(traj["fen"]))
    n = len(traj["steps"])
    for i, step in enumerate(traj["steps"]):
        lines = str(st).splitlines()
        assert lines[0] == step["fen"], (i, lines[0], step["fen"])
        kind, seats = st.turn()
        if i == n - 1 and traj["result"] is not None:
            assert kind == "terminal", (i, step["fen"], traj["reason"])
            assert st.returns() == [traj["result"], -traj["result"]], (lines, traj["reason"])
            ours = REASON[lines[1].split()[1]]
            if {ours, traj["reason"]} - {"repetition", "move_limit"}:
                assert ours == traj["reason"], (lines, traj["reason"])
            break
        assert kind == "act", (i, lines)
        seat = seats[0]
        ids = st.legal_actions(seat)
        names = {st.action_string(seat, a): a for a in ids}
        assert " ".join(sorted(names)) == step["legal"], (i, step["fen"])
        glob = st.frame(seat)["obs"]["globals_seq"]
        assert bool(glob[3] == 1.0) == step["check"], (i, step["fen"])
        if i < len(traj["moves"]):
            st.apply_action(seat, names[traj["moves"][i]])
    return n


with gzip.open(FIXTURE, "rt") as _fh:
    _FIXTURE = json.load(_fh)


def test_fixture_matches_fairy_stockfish():
    n = sum(compare_trajectory(t) for t in _FIXTURE["trajectories"])
    assert n > 5000


def test_fixture_covers_the_rules():
    """The fixture exercises every end rule (else it proves little)."""
    trajs = _FIXTURE["trajectories"]
    reasons = Counter((t["reason"], t["result"]) for t in trajs)
    for r in ("checkmate", "stalemate", "material", "move_limit"):
        assert any(k[0] == r for k in reasons), r
    assert reasons[("repetition", 0)] > 0                        # plain repetition draws
    assert reasons[("repetition", 1)] > 0 and reasons[("repetition", -1)] > 0   # perpetuals
    assert max(len(t["moves"]) for t in trajs if t["fen"] == START) >= 100
    assert sum(len(t["moves"]) for t in trajs) > 5000


def _uv():
    return shutil.which("uv")


live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1" or _uv() is None,
                          reason="live reference tests: set GM_REFERENCE=1 (needs uv + network)")


def _run_reference(tmp_path, *args):
    out = tmp_path / "ref.json"
    subprocess.run([_uv(), "run", "--no-project", "--python", "3.12", "--with", "pyffish==0.0.90",
                    "python", str(DUMP), "--out", str(out), *args],
                   check=True, capture_output=True, text=True, timeout=1800)
    with open(out) as fh:
        return json.load(fh)


@live
def test_live_fairy_stockfish_playouts(tmp_path):
    data = _run_reference(tmp_path, "--episodes", "6", "--seed", "11")
    for traj in data["trajectories"]:
        compare_trajectory(traj)


@live
def test_live_gmrules_playouts_replayed_in_fairy_stockfish(tmp_path):
    """Our random playouts through Fairy-Stockfish: every move accepted, the same FENs,
    legal sets, checks and game ends."""
    fens = [START, "3akab2/9/4b4/4n4/2p1c1p2/9/2P1C1P2/4B4/4A4/2BAK4 w - - 0 1",
            "4ka3/9/4b4/9/2p6/6P2/9/4B4/4A4/1NC1K4 w - - 0 1",
            "4k4/4a4/9/2p1p1p2/9/9/2P1P1P2/9/4A4/4K4 b - - 0 1",
            "2bakabnr/9/r1n1c4/2p1p1p1p/PP7/9/4P1P1P/2C3NC1/9/1NBAKAB1R w - - 0 1"]
    todo = []
    for k, fen in enumerate(fens):
        for seed in range(8 if fen == START else 6):
            ev = gmsim.random_trajectory("xiangqi", config_of(fen), 1000 * k + seed)
            st = gmsim.State("xiangqi", config_of(fen))
            moves = []
            for seat, a in ev.tolist():
                moves.append(st.action_string(seat, a))
                st.apply_action(seat, a)
            todo.append({"fen": fen, "moves": moves})
    (tmp_path / "ours.json").write_text(json.dumps({"trajectories": todo}))
    data = _run_reference(tmp_path, "--replay", str(tmp_path / "ours.json"))
    for mine, traj in zip(todo, data["trajectories"]):
        assert not traj["reason"].startswith("illegal"), traj["reason"]
        assert traj["moves"] == mine["moves"]
        if traj["result"] is None:          # truncated by max_decisions: running for both
            assert traj["reason"] == "running"
        compare_trajectory(traj)
