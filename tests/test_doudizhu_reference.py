"""Differential tests of the gmrules Dou Dizhu game (data_processing/rs/gmrules/src/
games/doudizhu) against OpenSpiel's ``dou_dizhu``.

The gmrules game plays a combination with kickers as several decisions (main, then
one kicker per unit); :func:`full_plays` walks those decisions to list every legal
FULL play by its OpenSpiel string (the action string of a completed play), so each
legal set is compared with OpenSpiel's play for play.  Chance outcomes (the face-up
position, every card dealt) with their probabilities, the auction's legal ids, the
transitions and the final returns are compared at every step.

- ``rules="openspiel"`` must equal OpenSpiel exactly;
- the standard rules may only add four-with-two plays and chains whose lowest rank
  the player holds four of (OpenSpiel's ``SearchChainOnlyActions`` skips such heads),
  and only lack OpenSpiel's bombs after the rocket (its ``SearchForLegalActions``
  lets any bomb follow the rocket; nothing beats the rocket in the real game);
- committed fixture (tests/fixtures/doudizhu/openspiel_reference.json.gz, written by
  scripts/reference/doudizhu_openspiel_dump.py --games 40 --seed 2: OpenSpiel random playouts);
- live (``GM_REFERENCE=1``, uv + network the first time): fresh OpenSpiel playouts, and
  gmrules' own random playouts replayed through OpenSpiel (both directions).
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "doudizhu" / "openspiel_reference.json.gz"
DUMP = ROOT / "scripts" / "reference" / "doudizhu_openspiel_dump.py"
GAME = "doudizhu"
OPENSPIEL = {"rules": "openspiel"}
STANDARD: Dict = {}
RANKS = "3456789TJQKA2"


def full_plays(st: gmsim.State, seat: int) -> Dict[str, List[int]]:
    """Every legal full play of ``seat`` now: its string → its decision path (a
    combination with kickers: main, then kickers; '*' marks an incomplete one)."""
    out: Dict[str, List[int]] = {}

    def walk(s: gmsim.State, path: List[int]) -> None:
        for a in s.legal_actions(seat):
            name = s.action_string(seat, a)
            if "*" in name:
                nxt = s.clone()
                nxt.apply_action(seat, a)
                walk(nxt, path + [a])
            else:
                assert name not in out, f"two paths to {name}"
                out[name] = path + [a]

    walk(st, [])
    return out


def in_auction(st: gmsim.State, seat: int) -> bool:
    """The auction always offers a bid (ids 1..3, never legal in the play phase)."""
    return any(1 <= a <= 3 for a in st.legal_actions(seat))


def counts(cards: str) -> collections.Counter:
    """Rank counts of an OpenSpiel card string ('(BWJ)' / '(CJ)' for the jokers)."""
    c: collections.Counter = collections.Counter()
    s = cards.replace("(BWJ)", "B").replace("(CJ)", "R")
    for ch in s:
        c[ch] += 1
    return c


def known_extra(play: str, hand: str) -> bool:
    """A play of the standard rules that OpenSpiel lacks: four-with-two, or a chain
    (solo / pair / kicker-free airplane) starting at a rank the player holds four of."""
    if "-" in play:
        main = counts(play.split("-")[0])
        return len(main) == 1 and set(main.values()) == {4}
    c = counts(play)
    if len(c) < 2 or len(set(c.values())) != 1 or not set(c) <= set(RANKS[:12]):
        return False
    idx = sorted(RANKS.index(r) for r in c)
    if idx != list(range(idx[0], idx[0] + len(idx))):
        return False
    return counts(hand)[RANKS[idx[0]]] == 4


def compare_trajectory(config, traj) -> Dict[str, int]:
    """Step gmsim.State through a reference trajectory, comparing every step; returns
    counters (steps, extra plays of the standard rules)."""
    st = gmsim.State(GAME, config)
    stats = collections.Counter()
    exact = config.get("rules") == "openspiel"
    to_beat, passes = None, 0            # the play to beat (None: a free lead)
    for i, step in enumerate(traj["steps"]):
        kind, seats = st.turn()
        assert kind == step["kind"], (i, kind, step["kind"])
        if kind == "chance":
            outs = st.chance_outcomes()
            assert hex(sum(1 << o for o, _ in outs)) == step["outcomes"], i
            ps = [p for _, p in outs]
            assert min(ps) == pytest.approx(step["p"][0], abs=1e-12)
            assert max(ps) == pytest.approx(step["p"][1], abs=1e-12)
            st.apply_chance(step["event"][1])
        else:
            seat = step["seat"]
            assert seats == [seat], (i, seats, seat)
            assert in_auction(st, seat) == (step["phase"] == "auction"), i
            if step["phase"] == "auction":
                assert st.legal_actions(seat) == step["legal"], i
                st.apply_action(seat, step["event"][1])
            else:
                plays = full_plays(st, seat)
                ours, theirs = set(plays), set(step["legal"])
                missing, extra = theirs - ours, ours - theirs
                if exact:
                    assert not missing and not extra, (i, sorted(missing)[:5], sorted(extra)[:5])
                else:
                    # OpenSpiel's bombs after the rocket; the standard rules' additions
                    bomb = lambda p: len(counts(p)) == 1 and 4 in counts(p).values()  # noqa: E731
                    assert all(bomb(p) and to_beat == "(BWJ)(CJ)" for p in missing), (i, missing)
                    bad = [p for p in extra if not known_extra(p, step["hand"])]
                    assert not bad, (i, step["hand"], bad[:5])
                    stats["extra"] += len(extra)
                    stats["missing"] += len(missing)
                if "rejected" in step:
                    stats["rejected"] += 1
                    break
                played = step["event"][1]
                if played not in plays:           # an OpenSpiel-only play: stop here
                    stats["diverged"] += 1
                    return stats
                for a in plays[played]:
                    st.apply_action(seat, a)
                if played == "Pass":
                    passes += 1
                    if passes == 2:
                        to_beat, passes = None, 0
                else:
                    to_beat, passes = played, 0
        stats["steps"] += 1
    kind, _ = st.turn()
    if traj["terminal"] and not stats["rejected"]:
        assert kind == "terminal"
        assert st.returns() == pytest.approx(traj["returns"], abs=1e-9)
    return stats


with gzip.open(FIXTURE, "rt") as _fh:
    _FIXTURE = json.load(_fh)


@pytest.mark.parametrize("config", [OPENSPIEL, STANDARD], ids=["openspiel", "standard"])
def test_fixture_trajectories_match_openspiel(config):
    total = collections.Counter()
    for traj in _FIXTURE["trajectories"]:
        total.update(compare_trajectory(config, traj))
    assert total["steps"] > 1000
    if not config:
        assert total["extra"] > 0          # the standard rules do add plays


def test_fixture_covers_the_rules():
    """The fixture exercises the interesting branches (else it proves little)."""
    trajs = _FIXTURE["trajectories"]
    assert _FIXTURE["num_distinct_actions"] == 26057 and _FIXTURE["max_chance_outcomes"] == 105
    legal = [p for t in trajs for s in t["steps"] if s.get("phase") == "play" for p in s["legal"]]
    played = [s["event"][1] for t in trajs for s in t["steps"] if s.get("phase") == "play"]
    assert any("-" in p and len(p.split("-")[1]) >= 2 for p in legal)       # airplane + kickers
    assert any(len(counts(p)) == 1 and 4 in counts(p).values() for p in played)  # a bomb
    assert "(BWJ)(CJ)" in played                                            # the rocket
    rets = [tuple(t["returns"]) for t in trajs]
    assert (0.0, 0.0, 0.0) in rets                                          # nobody bid
    assert len({max(abs(x) for x in r) for r in rets}) >= 3                 # bids x doublings


def test_full_plays_follow_the_decision_paths():
    """Every legal full play is one path of decisions; paths are prefix-free."""
    st = gmsim.State(GAME, STANDARD)
    ev = gmsim.random_trajectory(GAME, STANDARD, 3)
    n = 0
    for seat, a in ev.tolist():
        kind, seats = st.turn()
        if kind == "act" and not in_auction(st, seats[0]):
            plays = full_plays(st, seats[0])
            paths = [tuple(p) for p in plays.values()]
            assert len(set(paths)) == len(paths)
            firsts = collections.Counter(p[0] for p in paths)
            assert set(firsts) == set(st.legal_actions(seats[0]))
            n += 1
        st.apply((seat, a))
    assert n > 10


def _uv():
    return shutil.which("uv")


live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1" or _uv() is None,
                          reason="live reference tests: set GM_REFERENCE=1 (needs uv + network)")


def _run_reference(tmp_path, *args):
    out = tmp_path / "ref.json.gz"
    subprocess.run([_uv(), "run", "--no-project", "--python", "3.12", "--with", "open_spiel",
                    "python", str(DUMP), "--out", str(out), *args],
                   check=True, capture_output=True, text=True, timeout=1800)
    with gzip.open(out, "rt") as fh:
        return json.load(fh)


def reference_events(config, events) -> list:
    """A gmrules trajectory in the dump script's replay format: chance / auction
    events as ids, every play as its full OpenSpiel string (kicker decisions folded)."""
    st = gmsim.State(GAME, config)
    out, i, events = [], 0, [tuple(e) for e in gmsim.events_array(events).tolist()]
    while i < len(events):
        seat, a = events[i]
        if seat < 0:
            st.apply_chance(a)
            out.append([-1, a])
            i += 1
            continue
        auction = in_auction(st, seat)
        name = st.action_string(seat, a)
        st.apply_action(seat, a)
        i += 1
        while "*" in name:
            s2, a2 = events[i]
            assert s2 == seat
            name = st.action_string(seat, a2)
            st.apply_action(seat, a2)
            i += 1
        out.append([seat, a] if auction else [seat, name])
    return out


@live
def test_live_openspiel_playouts(tmp_path):
    data = _run_reference(tmp_path, "--games", "150", "--seed", "7")
    for config in (OPENSPIEL, STANDARD):
        for traj in data["trajectories"]:
            compare_trajectory(config, traj)


@live
def test_live_gmrules_playouts_replayed_in_openspiel(tmp_path):
    """Our random playouts (rules=openspiel) through OpenSpiel: every event accepted,
    the same states, legal sets and returns at every step."""
    ours = [gmsim.random_trajectory(GAME, OPENSPIEL, seed) for seed in range(150)]
    todo = {"trajectories": [reference_events(OPENSPIEL, ev) for ev in ours]}
    (tmp_path / "ours.json").write_text(json.dumps(todo))
    data = _run_reference(tmp_path, "--replay", str(tmp_path / "ours.json"))
    for traj, ev, ref in zip(data["trajectories"], ours, todo["trajectories"]):
        assert traj["terminal"] and not any("rejected" in s for s in traj["steps"])
        assert len(traj["steps"]) == len(ref)                # OpenSpiel took every event
        compare_trajectory(OPENSPIEL, traj)
        r = gmsim.replay(GAME, OPENSPIEL, ev)
        assert r["returns"] == pytest.approx(traj["returns"], abs=1e-9)
