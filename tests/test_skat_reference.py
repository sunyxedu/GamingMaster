"""Differential tests of the gmrules skat rules against the International Skat Server's
own rule engine — the SimpleState / GameDeclaration / GameResult classes of the ISS
client package (GPLv3, not in this repository), driven by
scripts/reference/IssSkatRef.java through scripts/reference/iss_skat_dump.py:

- committed fixture (tests/fixtures/skat/iss_reference.jsonl.gz): random playouts of
  the ISS engine (every move chosen among the moves it accepts) AND gmrules random
  playouts (gmsim.random_trajectory → `gmdata skat render`) replayed through the ISS
  engine. At every decision gmsim.State must agree on the seat to move and its legal
  set — the ISS-accepted moves (probed black-box: pass + every bid-list value, y / p,
  pickup + every hand declaration, every game after the pickup x every card pair,
  every card) expanded into the gmrules sub-decisions (bid window + MORE, hand game
  type + modifier, game + one discard at a time) — the game must end at the same move,
  and the result (declarer, win / loss, value, matadors, overbid, card points, tricks,
  schneider / schwarz flags) and the money scores must be identical;
- the one documented difference: LOST games with schneider / schwarz announced, where
  the ISS client code (2012) counts only the announcement bonuses and the ISS server
  counts the announced levels too — the rules follow the server (every such game of the
  9.1M-record archive, `gmdata skat verify`), so value / overbid / flags / money are not
  compared there (everything else is);
- live (``GM_REFERENCE=1``, needs a JDK and, the first time, network): fresh ISS
  playouts, fresh gmrules playouts replayed through the ISS engine, and the real
  fixture records replayed through it (the engine reproduces every recorded result but
  those of the documented client / server differences).

Regenerate the fixture: ``uv run python tests/test_skat_reference.py --write-fixture``.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
FIXTURE = ROOT / "tests" / "fixtures" / "skat" / "iss_reference.jsonl.gz"
REAL = ROOT / "tests" / "fixtures" / "skat" / "iss_sample.sgf"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
sys.path.insert(0, str(ROOT / "scripts" / "reference"))
import iss_skat_dump  # noqa: E402

# gmrules skat canonical actions (data_processing/rs/gmrules/src/games/skat.rs)
BIDS = [18, 20, 22, 23, 24, 27, 30, 33, 35, 36, 40, 44, 45, 46, 48, 50, 54, 55, 59, 60, 63, 66,
        70, 72, 77, 80, 81, 84, 88, 90, 96, 99, 100, 108, 110, 117, 120, 121, 126, 130, 132, 135,
        140, 143, 144, 150, 153, 154, 156, 160, 162, 165, 168, 170, 171, 176, 180, 187, 189, 190,
        192, 198, 204, 216, 240, 264]
PLAY0, DISCARD0, PASS, YES, BID0, MORE = 0, 32, 64, 65, 66, 132
PICKUP, HAND0, MOD0, DECLARE0 = 133, 134, 140, 144
GAMES = "DHSCGN"
HAND_DECLS = {"H": 0, "HS": 1, "HZ": 2, "O": 3}          # trump hand modifiers
PICKUP_GAMES = {"D": 0, "H": 1, "S": 2, "C": 3, "G": 4, "N": 5, "NO": 6}


def card(s: str) -> int:
    return "DHSC".index(s[0]) * 8 + "789TJQKA".index(s[1])


def next_bid_index(max_bid: int) -> int:
    return next((i for i, b in enumerate(BIDS) if b > max_bid), len(BIDS))


def expand(phase: str, move: str, max_bid: int) -> List[List[int]]:
    """An ISS move as its gmrules action sequences (the two discard orders)."""
    if phase == "bid":
        if move == "p":
            return [[PASS]]
        i, n = BIDS.index(int(move)), next_bid_index(max_bid)
        assert i >= n, (move, max_bid)
        return [[MORE] * ((i - n) // 10) + [BID0 + i]]
    if phase == "answer":
        assert move in ("y", "p"), move
        return [[YES if move == "y" else PASS]]
    if phase == "skat_or_hand":
        if move == "s":
            return [[PICKUP]]
        d = move.split(".")[0]
        ty = GAMES.index(d[0])
        if ty == 5:
            assert d in ("NH", "NOH"), d
            return [[HAND0 + 5, MOD0 + (3 if "O" in d else 0)]]
        return [[HAND0 + ty, MOD0 + HAND_DECLS[d[1:]]]]
    if phase == "discard_and_decl":
        parts = move.split(".")
        d = DECLARE0 + PICKUP_GAMES[parts[0]]
        a, b = card(parts[1]), card(parts[2])
        return [[d, DISCARD0 + a, DISCARD0 + b], [d, DISCARD0 + b, DISCARD0 + a]]
    if phase == "cardplay":
        return [[PLAY0 + card(move)]]
    raise AssertionError(f"no gmrules decision for ISS phase {phase!r}")


def legal_expansions(step: Dict) -> List[List[int]]:
    phase, legal, mb = step["phase"], step["legal"], step["max_bid"]
    if phase == "discard_and_decl":
        # no hand game after the pickup
        assert set(legal["games"]) <= set(PICKUP_GAMES), legal["games"]
        out = []
        for g in legal["games"]:
            for a, b in legal["discards"]:
                assert a != b
                out += [[DECLARE0 + PICKUP_GAMES[g], DISCARD0 + a, DISCARD0 + b],
                        [DECLARE0 + PICKUP_GAMES[g], DISCARD0 + b, DISCARD0 + a]]
        return out
    return [e for m in legal for e in expand(phase, m, mb)]


def moves_of(record: str) -> List[Tuple[str, str]]:
    toks = re.search(r"MV\[(.*?)\]", record).group(1).split()
    return list(zip(toks[0::2], toks[1::2]))


def lost_announced(result: str, events: Sequence[Sequence[int]]) -> bool:
    """A lost trump game with schneider / schwarz announced (incl. ouvert)."""
    hand_mod = [a for s, a in events if s >= 0 and MOD0 <= a < DECLARE0]
    hand_ty = [a for s, a in events if s >= 0 and HAND0 <= a < MOD0]
    return (" loss " in result and bool(hand_mod) and hand_ty[0] != HAND0 + 5
            and hand_mod[0] - MOD0 >= 1)


def compare_game(g: Dict) -> List[List[int]]:
    """Step gmsim.State through the ISS game ``g``, comparing every decision; returns the
    gmrules trajectory ([seat, id] events)."""
    assert g["error"] is None, g["error"]
    moves = moves_of(g["record"])
    assert moves[0][0] == "w"
    st = gmsim.State("skat", {})
    events = []
    for c in moves[0][1].split("."):
        st.apply([-1, card(c)])
        events.append([-1, card(c)])
    steps = iter(g["steps"])
    for who, move in moves[1:]:
        if who == "w":
            continue                                     # the server shows the skat
        step = next(steps)
        seat = int(who)
        assert (step["seat"], step["move"]) == (seat, move)
        exps = legal_expansions(step)
        chosen = expand(step["phase"], move, step["max_bid"])[0]
        for k, a in enumerate(chosen):
            kind, seats = st.turn()
            assert (kind, seats) == ("act", [seat]), (g["id"], move, kind, seats)
            want = sorted({e[k] for e in exps if len(e) > k and e[:k] == chosen[:k]})
            ours = st.legal_actions(seat)
            assert ours == want, (g["id"], step["phase"], move, k, ours, want)
            assert st.frame(seat)["legal_ids"] == want
            st.apply([seat, a])
            events.append([seat, a])
    assert next(steps, None) is None
    kind, _ = st.turn()
    assert (kind == "terminal") == g["finished"], (g["id"], kind)
    return events


def result_fields(r: str) -> Dict[str, str]:
    if r == "passed":
        return {"passed": "1"}
    toks = r.split()
    out = dict(t.split(":", 1) for t in toks if ":" in t)
    out["kind"], out["overbid"] = toks[1], toks[4]
    return out


def compare_results(games: List[Dict], trajs: List[List[List[int]]],
                    ours: List[str]) -> Dict[str, int]:
    """The ISS engine's results vs the rules' (``ours``: `gmdata skat render` R[...])."""
    n = {"exact": 0, "lost_announced": 0}
    for g, ev, r in zip(games, trajs, ours):
        if not g["finished"]:
            continue
        st = gmsim.State("skat", {})
        for e in ev:
            st.apply(e)
        if lost_announced(g["result"], ev):
            a, b = result_fields(g["result"]), result_fields(r)
            for k in ("v", "overbid", "s", "z"):
                a.pop(k), b.pop(k)
            assert a == b, (g["id"], g["result"], r)
            n["lost_announced"] += 1
            continue
        assert r == g["result"], (g["id"], g["result"], r)
        assert st.returns() == pytest.approx([float(x) for x in g["money"]], abs=1e-9), g["id"]
        n["exact"] += 1
    return n


# ---------------------------------------------------------------------------
# the converter binary (renders gmrules trajectories as ISS records with our R[...])
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gmdata():
    return _gmdata()


def _gmdata() -> str:
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return str(WS / "target" / "release" / "gmdata")


def render(gmdata: str, tmp: Path, trajs: Dict[str, List[List[int]]]) -> List[str]:
    """gmrules trajectories → ISS record lines (R[...] = the rules' result)."""
    tmp.mkdir(parents=True, exist_ok=True)
    with open(tmp / "t.jsonl", "w") as fh:
        for k, ev in trajs.items():
            fh.write(json.dumps({"id": k, "events": ev}) + "\n")
    subprocess.run([gmdata, "skat", "render", "--input", str(tmp / "t.jsonl"),
                    "--out", str(tmp / "g.sgf")], check=True, capture_output=True)
    return (tmp / "g.sgf").read_text().splitlines()


def check_all(gmdata: str, tmp: Path, games: List[Dict]) -> Dict[str, int]:
    trajs = [compare_game(g) for g in games]
    lines = render(gmdata, tmp, {g["id"]: ev for g, ev in zip(games, trajs)})
    ours = [re.search(r"\]R\[(.*?)\]", line).group(1) for line in lines]
    n = compare_results(games, trajs, ours)
    n["games"] = len(games)
    n["decisions"] = sum(len(g["steps"]) for g in games)
    return n


def _load_fixture() -> List[Dict]:
    with gzip.open(FIXTURE, "rt") as fh:
        return [json.loads(line) for line in fh]


# ---------------------------------------------------------------------------
# committed fixture
# ---------------------------------------------------------------------------

def test_fixture_iss_playouts(gmdata, tmp_path):
    games = [g for g in _load_fixture() if g["kind"] == "iss"]
    n = check_all(gmdata, tmp_path, games)
    assert n["games"] >= 150 and n["exact"] >= 80 and n["lost_announced"] > 0, n


def test_fixture_gmrules_playouts_in_iss(gmdata, tmp_path):
    """Our random playouts, replayed by the ISS engine: it accepts every move, agrees on
    every legal set and on the results."""
    games = [g for g in _load_fixture() if g["kind"] == "gmrules"]
    n = check_all(gmdata, tmp_path, games)
    assert n["games"] >= 80 and n["exact"] >= 40, n
    # the fixture games ARE gmsim's playouts of their seeds
    for g in games:
        want = gmsim.random_trajectory("skat", {}, g["seed"]).tolist()
        assert compare_game(g) == want, g["id"]


def test_fixture_covers_the_rules():
    """The reference games exercise every phase and contract (else they prove little)."""
    games = _load_fixture()
    phases, contracts, bids = set(), set(), 0
    for g in games:
        for s in g["steps"]:
            phases.add(s["phase"])
            if s["phase"] == "skat_or_hand" and s["move"] != "s":
                contracts.add(s["move"].split(".")[0])
            if s["phase"] == "discard_and_decl":
                contracts.add(s["move"].split(".")[0])
            if s["phase"] == "bid" and s["move"] != "p":
                i = BIDS.index(int(s["move"]))
                bids += i > next_bid_index(s["max_bid"])            # jump bids (MORE)
    assert phases == {"bid", "answer", "skat_or_hand", "discard_and_decl", "cardplay"}
    assert {"D", "H", "S", "C", "G", "N", "NO", "GH", "NH", "NOH"} <= contracts, sorted(contracts)
    for mod in ("HS", "HZ", "O"):                                  # announcements, ouvert
        assert any(c[0] != "N" and c[1:] == mod for c in contracts), (mod, sorted(contracts))
    assert len(contracts) >= 20 and bids >= 10, (sorted(contracts), bids)
    results = [g["result"] for g in games]
    assert any(" win " in r for r in results) and any(" loss " in r for r in results)
    assert any("passed" == r for r in results) and any(" overbid " in r for r in results)
    assert any(" s:1 " in r and " win " in r for r in results)


# ---------------------------------------------------------------------------
# live reference
# ---------------------------------------------------------------------------

live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1" or not iss_skat_dump.have_jdk(),
                          reason="live ISS reference: set GM_REFERENCE=1 (needs a JDK + network)")


def _dump(tmp: Path, args: List[str]) -> List[Dict]:
    out = tmp / "ref.jsonl"
    iss_skat_dump.main(args + ["--out", str(out)])
    return [json.loads(line) for line in out.read_text().splitlines()]


@live
def test_live_iss_playouts(gmdata, tmp_path):
    games = _dump(tmp_path, ["--play", "3000", "--seed", "2024"])
    n = check_all(gmdata, tmp_path, games)
    assert n["exact"] > 1500, n


@live
def test_live_gmrules_playouts_replayed_in_iss(gmdata, tmp_path):
    trajs = {f"g{s}": gmsim.random_trajectory("skat", {}, s).tolist() for s in range(1000, 2000)}
    (tmp_path / "ours.sgf").write_text("\n".join(render(gmdata, tmp_path / "r", trajs)) + "\n")
    games = _dump(tmp_path, ["--replay", str(tmp_path / "ours.sgf")])
    assert [g["id"] for g in games] == list(trajs)
    for g in games:
        assert compare_game(g) == trajs[g["id"]], g["id"]      # ISS accepted every move
    n = check_all(gmdata, tmp_path, games)
    assert n["exact"] > 400, n


@live
def test_live_real_records_in_iss(tmp_path):
    """The ISS engine replays every real fixture record; its results equal the recorded
    ones but in the documented client / server differences (lost games with announced
    schneider / schwarz, the declarer's own schneider / schwarz in games ended early)."""
    games = _dump(tmp_path, ["--replay", str(REAL)])
    recorded = {re.search(r"ID\[(.*?)\]", line).group(1): re.search(r"\]R\[(.*?)\]", line).group(1)
                for line in REAL.read_text().splitlines() if line.strip()}
    same = differ = 0
    for g in games:
        assert g["error"] is None, (g["id"], g["error"])
        if g["result"] == recorded[g["id"]]:
            same += 1
            continue
        differ += 1
        moves = moves_of(g["record"])
        early = any(m in ("RE",) or m.startswith(("TI.", "LE.")) for _, m in moves)
        announced = any(re.fullmatch(r"[DHSCG](HS|HZ|O)(\..*)?", m) for w, m in moves if w != "w")
        assert " loss " in recorded[g["id"]] and (early or announced), \
            (g["id"], g["result"], recorded[g["id"]])
    assert same > 0.9 * len(games), (same, differ)


# ---------------------------------------------------------------------------
# fixture generation
# ---------------------------------------------------------------------------

def write_fixture(n_iss: int = 200, n_ours: int = 100, seed: int = 11) -> None:
    import tempfile

    gm = _gmdata()
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        iss = _dump(tmp, ["--play", str(n_iss), "--seed", str(seed)])
        trajs = {f"s{s}": gmsim.random_trajectory("skat", {}, s).tolist() for s in range(n_ours)}
        (tmp / "ours.sgf").write_text("\n".join(render(gm, tmp / "r", trajs)) + "\n")
        ours = _dump(tmp, ["--replay", str(tmp / "ours.sgf")])
    with gzip.open(FIXTURE, "wt") as fh:
        for g in iss:
            fh.write(json.dumps({"kind": "iss", **g}, separators=(",", ":")) + "\n")
        for g in ours:
            row = {"kind": "gmrules", "seed": int(g["id"][1:]), **g}
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"{FIXTURE}: {len(iss)} ISS playouts, {len(ours)} gmrules playouts")


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-fixture"]:
        write_fixture()
    else:
        sys.exit(pytest.main([__file__, "-q"]))
