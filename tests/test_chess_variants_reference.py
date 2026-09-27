"""Differential tests of the gmrules game ``chess_variants`` against an independent
implementation of the lichess variants: Fairy-Stockfish (pyffish).

At every position of a trajectory both engines must agree on the legal move set
(canonical UCI: castling king -> own rook, drops ``P@e4``, promotions incl. antichess
``e7e8k``), and at its end on the result — mate, stalemate and every variant end
(explosion, hill, third check, race, horde captured, antichess). Lichess' automatic
draws that are no Fairy-Stockfish game end are checked for consistency instead: our
fifty-move / fivefold draws where Fairy-Stockfish has a claimable draw, our chess960
insufficient-material draws where it says neither side can win; lichess' dead-position
draws of atomic / antichess / horde (scalachess rules, pinned by the gmrules unit tests
and by real games) end our trajectory while Fairy-Stockfish plays on. Known rule
difference: Fairy-Stockfish applies the fifty-move rule to crazyhouse, lichess does not.

- committed fixture (tests/fixtures/gmrules/fairy_stockfish_reference.json, written by
  ``python -m tests.test_chess_variants_reference --regenerate``): Fairy-Stockfish's
  random playouts of every variant, gmrules' random playouts replayed through
  Fairy-Stockfish and the 49 real lichess games of tests/fixtures/chess_variants —
  legal move sets as hashes, compared with ``gmsim.State`` step by step;
- live (``GM_REFERENCE=1``, needs uv + network the first time): the same in both
  directions with fresh seeds and full move lists.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import gmsim
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "gmrules" / "fairy_stockfish_reference.json"
DUMP = ROOT / "scripts" / "reference" / "fairy_stockfish_dump.py"
REAL = ROOT / "tests" / "fixtures" / "chess_variants" / "lichess_variants_2026-08_sample.pgn"
WS = ROOT / "data_processing"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")

sys.path.insert(0, str(DUMP.parent))
from fairy_stockfish_dump import chess960_back_rank, legal_hash, parse_step  # noqa: E402

#: the variant chance outcomes (gmrules ChessVariant order)
VARIANTS = ["chess960", "crazyhouse", "atomic", "antichess", "threecheck", "kingofthehill",
            "horde", "racingkings"]
#: our automatic ends that are no Fairy-Stockfish game end
LICHESS_DRAWS = ("InsufficientMaterial", "FiftyMoves", "Fivefold", "PlyCap")


class Ours:
    """``gmsim.State`` of one chess_variants game, driven by UCI moves."""

    def __init__(self, variant: str, start: Optional[int], config=None):
        self.st = gmsim.State("chess_variants", config)
        self.st.apply([-1, VARIANTS.index(variant)])
        if variant == "chess960":
            self.st.apply([-1, int(start)])
        self.plies = 0

    def terminal(self) -> bool:
        return self.st.turn()[0] == "terminal"

    def legal(self) -> Dict[str, int]:
        kind, seats = self.st.turn()
        if kind != "act":
            return {}
        s = seats[0]
        return {self.st.action_string(s, a): a for a in self.st.legal_actions(s)}

    def play(self, uci: str) -> None:
        legal = self.legal()
        assert uci in legal, f"{uci} is not legal here: {str(self.st)}"
        self.st.apply([self.st.turn()[1][0], legal[uci]])
        self.plies += 1

    def reason(self) -> str:
        """Why the game is over (gmrules End), from the state string."""
        s = str(self.st)
        return next((r for r in ("InsufficientMaterial", "FiftyMoves", "Fivefold", "Threefold",
                                 "PlyCap", "Rules") if r in s), "")

    def fen(self) -> str:
        return str(self.st).split(" | ")[0].split(" ", 1)[1]


def _outcome(value: float, stm_white: bool) -> List[float]:
    """Fairy-Stockfish's side-to-move result as a [white, black] outcome."""
    if value == 0:
        return [0.0, 0.0]
    win = value > 0
    white_wins = win if stm_white else not win
    return [1.0, -1.0] if white_wins else [-1.0, 1.0]


def _same_legal(ours: Dict[str, int], step: Dict, where: str) -> None:
    mine = sorted(ours)
    if "legal" in step:
        assert mine == step["legal"], (where, sorted(set(mine) ^ set(step["legal"])))
    else:
        assert (len(mine), legal_hash(mine)) == (step["n"], step["h"]), \
            (where, len(mine), step["n"])


def compare(t: Dict, stats: Counter) -> None:
    """Step ``t`` (a dump trajectory) through our engine: same legal moves at every
    position, a consistent end (see the module docs)."""
    v = t["variant"]
    g = Ours(v, t.get("start"))
    if v == "chess960":
        assert g.fen().split(" ")[0].split("/")[-1] == chess960_back_rank(t["start"]), t["key"]
    moves, steps = t["moves"], [parse_step(s) for s in t["steps"]]
    for i, step in enumerate(steps):
        where = f"{t['key']} ply {i} ({g.fen()})"
        if g.terminal():
            # our engine ended with an automatic lichess draw, Fairy-Stockfish plays on
            r = g.reason()
            assert r in LICHESS_DRAWS, (where, r)
            _check_lichess_draw(v, r, step.get("o", 0), None, where)
            stats[f"lichess_draw:{v}:{r}"] += 1
            return
        _same_legal(g.legal(), step, where)
        if step.get("ok", 1) == 0:
            raise AssertionError(f"{where}: Fairy-Stockfish rejects {moves[i]}")
        if i < len(moves):
            g.play(moves[i])
        stats["positions"] += 1
    f = t["final"]
    where = f"{t['key']} final ({g.fen()})"
    stm_white = g.plies % 2 == 0
    fsf_over = f["immediate"][0] or f["result"] is not None
    if fsf_over:
        value = f["immediate"][1] if f["immediate"][0] else f["result"]
        assert g.terminal() and g.reason() == "Rules", (where, g.reason(), f)
        assert g.st.outcome() == _outcome(value, stm_white), (where, g.st.outcome(), value)
        stats[f"end:{v}:{'draw' if value == 0 else 'decisive'}"] += 1
        return
    if not g.terminal():
        _same_legal(g.legal(), f, where)
        if f["optional"][0] and f["halfmoves"] >= 100:
            # Fairy-Stockfish's fifty-move claim: lichess' automatic draw, not in crazyhouse
            assert v == "crazyhouse", where
            stats["crazyhouse_fifty_moves"] += 1
        stats["open"] += 1
        return
    r = g.reason()
    assert r in LICHESS_DRAWS, (where, r)
    _check_lichess_draw(v, r, f["optional"][0], f["insufficient"], where)
    stats[f"lichess_draw:{v}:{r}"] += 1


def _check_lichess_draw(v: str, reason: str, claimable, insufficient, where: str) -> None:
    if reason in ("FiftyMoves", "Fivefold"):
        assert claimable, (where, reason, "Fairy-Stockfish has no claimable draw here")
    if reason == "InsufficientMaterial" and v == "chess960" and insufficient is not None:
        assert insufficient == [True, True], (where, insufficient)


_FIXTURE: Dict = {}


def _fixture() -> Dict:
    if not _FIXTURE:
        with open(FIXTURE) as fh:
            _FIXTURE.update(json.load(fh))
    return _FIXTURE


def _by_kind(kind: str) -> List[Dict]:
    return [t for t in _fixture()["trajectories"] if t["key"].startswith(kind)]


@pytest.mark.parametrize("kind", ["fsf:", "ours:", "lichess:"])
def test_fixture_trajectories_match_fairy_stockfish(kind):
    stats: Counter = Counter()
    trajs = _by_kind(kind)
    assert trajs
    for t in trajs:
        compare(t, stats)
    assert stats["positions"] > 100, stats


def test_fixture_covers_the_rules():
    """The fixture exercises every variant's end and the special moves."""
    stats: Counter = Counter()
    trajs = _fixture()["trajectories"]
    for t in trajs:
        compare(t, stats)
    variants = {t["variant"] for t in trajs}
    assert variants == set(VARIANTS)
    decisive = {k.split(":")[1] for k in stats if k.startswith("end:") and k.endswith("decisive")}
    assert decisive == set(VARIANTS), decisive
    moves = [m for t in trajs for m in t["moves"]]
    assert any("@" in m for m in moves), "crazyhouse drops"
    assert any(m.endswith("k") and len(m) == 5 for m in moves), "antichess king promotions"
    assert any(m[-1] in "nbr" and len(m) == 5 for m in moves), "under-promotions"
    lichess = _by_kind("lichess:")
    assert len(lichess) == 49 and all(parse_step(t["steps"][-1])["ok"] for t in lichess)
    assert any(k.startswith("lichess_draw:") for k in stats), stats


def test_chess960_numbering_is_scharnagl():
    """Our 960 start positions are the standard Scharnagl numbering (518 = RNBQKBNR)."""
    assert chess960_back_rank(518) == "RNBQKBNR"
    seen = set()
    for sp in range(960):
        g = Ours("chess960", sp)
        rank = g.fen().split(" ")[0].split("/")[-1]
        assert rank == chess960_back_rank(sp), sp
        seen.add(rank)
    assert len(seen) == 960


# ---------------------------------------------------------------------------
# fixture generation / live runs (Fairy-Stockfish in a throw-away uv env)
# ---------------------------------------------------------------------------

def _uv():
    return shutil.which("uv")


def run_dump(tmp: Path, *args) -> Dict:
    out = tmp / "fsf.json"
    subprocess.run([_uv(), "run", "--no-project", "--python", "3.12", "--with", "pyffish",
                    "python", str(DUMP), "--out", str(out), *args],
                   check=True, capture_output=True, text=True, timeout=1800)
    with open(out) as fh:
        return json.load(fh)


def our_playouts(seeds, max_plies: int = 200) -> List[Dict]:
    """gmrules' random playouts of every variant as UCI move lists."""
    out = []
    for v in VARIANTS:
        cfg = {"variants": [v], "max_plies": max_plies}
        for seed in seeds:
            ev = gmsim.random_trajectory("chess_variants", cfg, seed).tolist()
            out.append(_uci_record(f"ours:{v}:{seed}", ev, cfg))
    return out


def _uci_record(key: str, events, cfg=None) -> Dict:
    st = gmsim.State("chess_variants", cfg)
    moves, start, variant = [], None, None
    for seat, a in events:
        if seat < 0:
            if variant is None:
                variant = VARIANTS[a]
            else:
                start = a
        else:
            moves.append(st.action_string(seat, a))
        st.apply([seat, a])
    return {"key": key, "variant": variant, "start": start, "moves": moves}


def real_games(tmp: Path) -> List[Dict]:
    """The committed real lichess games (gmdata chess-variants convert --store-events)."""
    subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS, check=True,
                   capture_output=True)
    out = tmp / "real"
    subprocess.run([str(WS / "target" / "release" / "gmdata"), "chess-variants", "convert",
                    "--input", str(REAL), "--out", str(out), "--store-events"], check=True,
                   capture_output=True)
    from data_processing.games.dataset import load_gm_samples

    return [_uci_record(s["meta"]["record"], s["meta"]["events"])
            for s in load_gm_samples(str(out))]


def regenerate() -> None:
    """Write the committed fixture (compact: legal-move hashes)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (tmp / "ours.json").write_text(json.dumps(
            {"trajectories": our_playouts(range(3), max_plies=150) + real_games(tmp)}))
        data = run_dump(tmp, "--playouts", "3", "--seed", "0", "--max-plies", "150",
                        "--replay", str(tmp / "ours.json"), "--compact")
    FIXTURE.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(f"{len(data['trajectories'])} trajectories -> {FIXTURE}")


live = pytest.mark.skipif(os.environ.get("GM_REFERENCE") != "1" or _uv() is None,
                          reason="live reference tests: set GM_REFERENCE=1 (needs uv + network)")


@live
def test_live_fairy_stockfish_playouts(tmp_path):
    stats: Counter = Counter()
    for t in run_dump(tmp_path, "--playouts", "25", "--seed", "11")["trajectories"]:
        compare(t, stats)
    assert stats["positions"] > 5000, stats


@live
def test_live_gmrules_playouts_and_real_games_replayed_in_fairy_stockfish(tmp_path):
    ours = our_playouts(range(100, 125)) + real_games(tmp_path)
    (tmp_path / "ours.json").write_text(json.dumps({"trajectories": ours}))
    stats: Counter = Counter()
    data = run_dump(tmp_path, "--replay", str(tmp_path / "ours.json"))
    assert len(data["trajectories"]) == len(ours)
    for t, mine in zip(data["trajectories"], ours):
        assert t["moves"] == mine["moves"], t["key"]           # Fairy-Stockfish accepted every move
        compare(t, stats)
    assert stats["positions"] > 5000, stats


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        sys.exit(pytest.main([__file__, "-q"]))
