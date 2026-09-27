"""Differential tests of the gmrules bridge against an independent reference
implementation: OpenSpiel's ``bridge(use_double_dummy_result=false, dealer=D,
dealer_vul=., non_dealer_vul=.)`` (dump script: scripts/reference/openspiel_bridge_dump.py).
Ids are identical (cards 0..51, calls 52..89); a gmrules config ``{"board": b}`` has
no board chance event, so its trajectories are OpenSpiel histories verbatim.

- committed fixture (tests/fixtures/bridge/openspiel_reference.json): OpenSpiel's own
  random playouts on boards 1, 6, 7, 12 (four dealers, every vulnerability) — at every
  step gmsim.State agrees on the turn, the acting seat (declarer on dummy's plays), the
  legal actions (= the frame's candidates), the chance outcomes and probabilities, and
  the final returns; plus the returns OpenSpiel gives the real WBridge5 records of
  tests/fixtures/bridge/openspiel_test_sample.txt (passed-out, doubled and redoubled
  deals included);
- live (``GM_REFERENCE=1``, needs uv + network the first time): fresh OpenSpiel playouts
  on all 16 boards, gmrules' own random playouts replayed through OpenSpiel (every
  event accepted, same legal sets and returns), and the real records through both
  engines (``BRIDGE_OPENSPIEL_DATA=<test.txt>`` adds a whole dataset file).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "bridge"
REFERENCE = FIX / "openspiel_reference.json"
RECORDS = FIX / "openspiel_test_sample.txt"
DUMP = ROOT / "scripts" / "reference" / "openspiel_bridge_dump.py"

with open(REFERENCE) as _fh:
    _REF = json.load(_fh)


def compare_trajectory(board: int, traj) -> int:
    """Step gmsim.State("bridge", {"board": board}) through an OpenSpiel trajectory
    (compact chance steps: outcome bit mask + common probability), comparing every
    step; returns the number of steps."""
    st = gmsim.State("bridge", {"board": board})
    for i, step in enumerate(traj["steps"]):
        ref = step["turn"]
        kind, seats = st.turn()
        assert kind == ref["kind"], (i, kind, ref)
        if kind == "chance":
            outs = st.chance_outcomes()
            mask = 0
            for o, p in outs:
                mask |= 1 << o
                assert p == pytest.approx(ref["p"], abs=1e-12)
            assert mask == int(ref["mask"]), i
        else:
            assert seats == [ref["seat"]], (i, seats, ref["seat"])
            legal = st.legal_actions(ref["seat"])
            assert legal == ref["legal"], (i, legal, ref["legal"])
            assert st.frame(ref["seat"])["legal_ids"] == legal
        st.apply(step["event"])
    kind, _ = st.turn()
    assert kind == traj["final"]
    if kind == "terminal":
        assert st.returns() == pytest.approx(traj["returns"], abs=1e-9)
    return len(traj["steps"])


def record_events(line: str):
    """A WBridge5 dataset line (an OpenSpiel history of board 1) as gmrules events
    (the acting seats from the gmrules state)."""
    st = gmsim.State("bridge", {"board": 1})
    events = []
    for tok in line.split():
        a = int(tok)
        kind, seats = st.turn()
        ev = [-1, a] if kind == "chance" else [seats[0], a]
        st.apply(ev)
        events.append(ev)
    return events, st


def test_fixture_playouts_match_openspiel():
    n = 0
    for v in _REF["random"]:
        for traj in v["trajectories"]:
            n += compare_trajectory(v["board"], traj)
            assert traj["final"] == "terminal"
    assert n > 16 * 100


def test_fixture_covers_the_rules():
    """The fixture exercises the interesting branches (else it proves little)."""
    boards = {v["board"] for v in _REF["random"]}
    assert boards == {1, 6, 7, 12}                      # dealers N, E, S, W; every vulnerability
    events = [s["event"][1] for v in _REF["random"] for t in v["trajectories"] for s in t["steps"]
              if s["turn"]["kind"] == "act"]
    assert 53 in events and 54 in events                # doubles and redoubles
    returns = {t["returns"][0] for v in _REF["random"] for t in v["trajectories"]}
    assert len(returns) >= 8
    lines = RECORDS.read_text().splitlines()
    assert any(len(ln.split()) == 56 for ln in lines)   # a passed-out deal
    assert any(54 in [int(x) for x in ln.split()[52:-52]] for ln in lines)


def test_real_records_match_openspiel():
    rec = _REF["records"]
    lines = RECORDS.read_text().splitlines()
    assert len(lines) == len(rec["returns"]) >= 50
    for ln, want in zip(lines, rec["returns"]):
        events, st = record_events(ln)
        assert st.turn()[0] == "terminal"
        r = gmsim.replay("bridge", {"board": 1}, events)
        assert r["terminal"] and r["returns"] == pytest.approx(want, abs=1e-9)
        # the declarer's stream carries dummy's plays: 26 of the 52 cards
        if len(ln.split()) > 56:
            seats = r["seat"][r["action"] < 52].tolist()
            assert sorted(seats.count(s) for s in range(4) if seats.count(s)) == [13, 13, 26]


def _uv():
    return shutil.which("uv")


live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1" or _uv() is None,
                          reason="live reference tests: set GM_REFERENCE=1 (needs uv + network)")


def _run_reference(tmp_path, *args):
    out = tmp_path / "ref.json"
    subprocess.run([_uv(), "run", "--no-project", "--python", "3.12", "--with", "open_spiel",
                    "python", str(DUMP), "--out", str(out), *args],
                   check=True, capture_output=True, text=True, timeout=1800)
    with open(out) as fh:
        return json.load(fh)


@live
def test_live_openspiel_playouts(tmp_path):
    data = _run_reference(tmp_path, "--episodes", "40", "--seed", "7",
                          "--boards", *[str(b) for b in range(1, 17)])
    n = 0
    for v in data["variants"]:
        for traj in v["trajectories"]:
            compare_trajectory(v["board"], traj)
            n += 1
    assert n == 16 * 40


@live
def test_live_gmrules_playouts_replayed_in_openspiel(tmp_path):
    todo = {"variants": [{"board": b, "trajectories": [
        gmsim.random_trajectory("bridge", {"board": b}, 100 * b + s).tolist() for s in range(25)]}
        for b in range(1, 17)]}
    (tmp_path / "ours.json").write_text(json.dumps(todo))
    data = _run_reference(tmp_path, "--replay", str(tmp_path / "ours.json"))
    for v, mine in zip(data["variants"], todo["variants"]):
        for traj, events in zip(v["trajectories"], mine["trajectories"]):
            assert traj["ok"] and traj["final"] == "terminal"      # OpenSpiel accepted every event
            assert len(traj["steps"]) == len(events)
            r = gmsim.replay("bridge", {"board": v["board"]}, events)
            assert r["returns"] == pytest.approx(traj["returns"], abs=1e-9)
            compare_trajectory(v["board"], traj)


@live
def test_live_real_records_through_both_engines(tmp_path):
    files = [str(RECORDS)]
    if os.environ.get("BRIDGE_OPENSPIEL_DATA"):
        files.append(os.environ["BRIDGE_OPENSPIEL_DATA"])
    for f in files:
        data = _run_reference(tmp_path, "--records", f, "--summary")
        for traj in data["variants"][0]["trajectories"]:
            assert traj["ok"] and traj["final"] == "terminal"
            r = gmsim.replay("bridge", {"board": 1}, traj["events"])
            assert r["returns"] == pytest.approx(traj["returns"], abs=1e-9)
