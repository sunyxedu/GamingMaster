"""Differential tests of the gmrules games against an independent reference
implementation (OpenSpiel): kuhn_poker (2-4 players) and goofspiel (simultaneous,
2-4 players, random / fixed prize order).

- committed fixture (tests/fixtures/candidate/openspiel_reference.json, written by
  scripts/reference/openspiel_dump.py from OpenSpiel's own random playouts): at every
  step gmsim.State must agree on the turn, the acting seats, every legal action set,
  the chance outcomes and their probabilities, and the final returns;
- live (``GM_REFERENCE=1``, needs uv + network the first time): fresh OpenSpiel
  playouts, AND gmrules' own random playouts replayed through OpenSpiel — the
  randomized differential test in both directions.

The pattern for a new game: map the reference's ids to the canonical ids, add its
variants to openspiel_dump.py (or a dump script for another reference), commit a
small fixture, reuse :func:`compare_trajectory`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "candidate" / "openspiel_reference.json"
DUMP = ROOT / "scripts" / "reference" / "openspiel_dump.py"


def compare_trajectory(game: str, config, traj) -> int:
    """Step gmsim.State through a reference trajectory, comparing every step;
    returns the number of steps compared."""
    st = gmsim.State(game, config)
    for i, step in enumerate(traj["steps"]):
        ref = step["turn"]
        kind, seats = st.turn()
        assert kind == ref["kind"], (i, kind, ref)
        if kind == "chance":
            ours = st.chance_outcomes()
            assert [o for o, _ in ours] == [o for o, _ in ref["outcomes"]], (i, ours, ref)
            for (_, p), (_, q) in zip(ours, ref["outcomes"]):
                assert p == pytest.approx(q, abs=1e-12)
        else:
            assert seats == ref["seats"], (i, seats, ref["seats"])
            for s in seats:
                assert st.legal_actions(s) == ref["legal"][str(s)], (i, s)
                # the candidates of the seat's frame ARE its legal actions
                assert st.frame(s)["legal_ids"] == ref["legal"][str(s)]
        for ev in step["events"]:
            st.apply(ev)
    kind, _ = st.turn()
    assert kind == traj["final"]["kind"]
    if kind == "terminal":
        assert st.returns() == pytest.approx(traj["returns"], abs=1e-9)
    return len(traj["steps"])


def _variants(data):
    return [(v["key"], v) for v in data["variants"]]


with open(FIXTURE) as _fh:
    _FIXTURE = json.load(_fh)


@pytest.mark.parametrize("key,variant", _variants(_FIXTURE),
                         ids=[k for k, _ in _variants(_FIXTURE)])
def test_fixture_trajectories_match_openspiel(key, variant):
    n = 0
    for traj in variant["trajectories"]:
        n += compare_trajectory(variant["gm_game"], variant["gm_config"], traj)
        assert traj["final"]["kind"] == "terminal"
    assert n > 0


def test_fixture_covers_the_rules():
    """The fixture exercises the interesting branches (else it proves little)."""
    kuhn3 = next(v for v in _FIXTURE["variants"] if v["key"] == "kuhn3")
    lengths = {len(t["steps"]) for t in kuhn3["trajectories"]}
    assert min(lengths) == 3 + 3 and max(lengths) >= 3 + 4        # all-pass and bet/call lines
    goof = next(v for v in _FIXTURE["variants"] if v["key"] == "goof5x3")
    returns = {tuple(t["returns"]) for t in goof["trajectories"]}
    assert len(returns) >= 3                                       # wins, split wins, draws


def _uv():
    return shutil.which("uv")


live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1" or _uv() is None,
                          reason="live reference tests: set GM_REFERENCE=1 (needs uv + network)")


def _run_reference(tmp_path, *args):
    out = tmp_path / "ref.json"
    subprocess.run([_uv(), "run", "--no-project", "--python", "3.12", "--with", "open_spiel",
                    "python", str(DUMP), "--out", str(out), *args],
                   check=True, capture_output=True, text=True, timeout=900)
    with open(out) as fh:
        return json.load(fh)


@live
def test_live_openspiel_playouts(tmp_path):
    data = _run_reference(tmp_path, "--episodes", "200", "--seed", "7")
    for key, v in _variants(data):
        for traj in v["trajectories"]:
            compare_trajectory(v["gm_game"], v["gm_config"], traj)


@live
def test_live_gmrules_playouts_replayed_in_openspiel(tmp_path):
    """Our random playouts through OpenSpiel: same turns, legal sets, returns."""
    todo = {"variants": []}
    for key, v in _variants(_FIXTURE):
        trajs = [gmsim.random_trajectory(v["gm_game"], v["gm_config"], seed).tolist()
                 for seed in range(150)]
        todo["variants"].append({"key": key, "trajectories": trajs})
    (tmp_path / "ours.json").write_text(json.dumps(todo))
    data = _run_reference(tmp_path, "--replay", str(tmp_path / "ours.json"))
    for (key, v), mine in zip(_variants(data), todo["variants"]):
        for traj, events in zip(v["trajectories"], mine["trajectories"]):
            n_events = sum(len(s["events"]) for s in traj["steps"])
            assert n_events == len(events), key             # OpenSpiel accepted every event
            assert traj["final"]["kind"] == "terminal", key
            r = gmsim.replay(v["gm_game"], v["gm_config"], events)
            assert r["returns"] == pytest.approx(traj["returns"], abs=1e-9), key
            compare_trajectory(v["gm_game"], v["gm_config"], traj)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
