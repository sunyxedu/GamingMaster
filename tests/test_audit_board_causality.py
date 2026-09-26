"""Future-leak (causality) audit of the perfect-information board-game
converters: chess, go, shogi, othello, gomoku (``gmdata <game> convert``).

The contract: every decoder INPUT of frame t — ``obs.*``, the teacher-forcing
labels and the ``tgt.legal_*`` masks (fed to the decoder under
``--il_legal_mask``) — is a function of the record up to decision t only.
Only ``tgt.value`` (the final outcome) may use hindsight.

Two record surgeries on the committed real fixtures, reconverted with the
real binary and compared tensor by tensor (bit-identical):

  truncation  the record cut right after decision k (k at several points
              of every game, the first and the last included): frames
              0..k-1 must equal the full conversion's in EVERY tensor except
              ``tgt.value``.  Rules out anything computed from the rest of
              the game (normalisers over the game, final position / result,
              game length, repetition counts looking ahead).
  divergence  the record with a different decision k (another legal move,
              a pass for go) and a different continuation: frames 0..k-1
              equal in every tensor but ``tgt.value``, and frame k's OBS
              equal too (its label differs by construction).  Also rules
              out a frame's observation depending on its own move (e.g. the
              position AFTER the move being encoded).

and a header surgery: the result tag / line flipped (``tgt.value`` must
change, nothing else may).  Headers that are legitimately known before the
game (komi, ruleset, handicap) are kept.

Skips cleanly without cargo; the fixtures are checked in.
"""
from __future__ import annotations

import json
import os
import random
import re
import struct
import subprocess
from typing import Dict, List, Sequence, Tuple

import pytest
import torch

from data_processing.games.dataset import load_gm_samples
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture)

HINDSIGHT = frozenset({"tgt.value"})


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------

def _flat(sample: Dict) -> Dict[str, torch.Tensor]:
    return {f"{g}.{k}": v for g in ("obs", "tgt") for k, v in sample[g].items()}


def _first_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    return next(t for t in range(a.shape[0]) if not torch.equal(a[t], b[t]))


def assert_prefix_identical(full: Dict, part: Dict, n: int, what: str,
                            obs_extra_frame: bool = False) -> None:
    """Frames [0, n) identical in every tensor but the hindsight ones; with
    ``obs_extra_frame`` also frame n of every ``obs.*`` tensor."""
    a, b = _flat(full), _flat(part)
    assert set(a) == set(b), (what, sorted(set(a) ^ set(b)))
    for k in sorted(a):
        if k in HINDSIGHT:
            continue
        m = n + 1 if (obs_extra_frame and k.startswith("obs.")) else n
        assert a[k].shape[0] >= m and b[k].shape[0] >= m, (what, k, a[k].shape, b[k].shape, m)
        if not torch.equal(a[k][:m], b[k][:m]):
            t = _first_diff(a[k][:m], b[k][:m])
            raise AssertionError(f"{what}: {k} frame {t} depends on the record after "
                                 f"decision {n - 1 if m == n else n} (future leak)")


def assert_only_value_differs(a_s: Dict, b_s: Dict, what: str) -> None:
    a, b = _flat(a_s), _flat(b_s)
    assert set(a) == set(b)
    for k in sorted(a):
        if k in HINDSIGHT:
            continue
        assert torch.equal(a[k], b[k]), f"{what}: {k} depends on the result header"
    assert not torch.equal(a["tgt.value"], b["tgt.value"]), f"{what}: value ignores the result"


def _run(binary: str, args: Sequence[str]) -> Dict:
    r = subprocess.run([binary, *args], check=True, capture_output=True, text=True)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def _cuts(n: int) -> List[int]:
    """Decision counts to keep: first, second, a third, half, last."""
    return sorted({k for k in (1, 2, n // 3, n // 2, n - 1) if 1 <= k < n})


# ---------------------------------------------------------------------------
# chess (Lichess PGN)
# ---------------------------------------------------------------------------

PGN = os.path.join(FIX, "chess", "lichess_2013-01_head.pgn")
FLIP = {"1-0": "0-1", "0-1": "1-0", "1/2-1/2": "1-0"}


def _chess_games(limit: int = 24):
    import chess.pgn
    out = []
    with open(PGN) as fh:
        while len(out) < limit and (g := chess.pgn.read_game(fh)) is not None:
            h = g.headers
            if g.errors or h.get("Variant", "Standard") != "Standard" or "FEN" in h:
                continue
            if h.get("Result") not in FLIP:
                continue
            moves = list(g.mainline_moves())
            if len(moves) >= 12:
                out.append((dict(h), moves))
    return out


def _pgn(headers: Dict[str, str], moves, result: str = None) -> str:
    import chess.pgn
    g = chess.pgn.Game()
    for k, v in headers.items():
        g.headers[k] = v
    if result is not None:
        g.headers["Result"] = result
    node = g
    for m in moves:
        node = node.add_variation(m)
    return str(g) + "\n\n"


def _chess_diverge(moves, k: int, rng: random.Random, extra: int = 12):
    """moves[:k] + a legal move other than moves[k] + a random continuation."""
    import chess
    b = chess.Board()
    for m in moves[:k]:
        b.push(m)
    alts = [m for m in b.legal_moves if m != moves[k]]
    if not alts:
        return None
    out = list(moves[:k]) + [rng.choice(alts)]
    b.push(out[-1])
    for _ in range(extra):
        legal = list(b.legal_moves)
        if not legal or b.is_game_over():
            break
        out.append(rng.choice(legal))
        b.push(out[-1])
    return out


@pytest.fixture(scope="module")
def chess_cases(gmdata, tmp_path_factory):  # noqa: F811
    rng = random.Random(0)
    games = _chess_games()
    assert len(games) >= 12
    records: List[Tuple[str, int, int, str]] = []   # (kind, game, k, pgn)
    for gi, (h, moves) in enumerate(games):
        records.append(("full", gi, len(moves), _pgn(h, moves)))
        records.append(("flip", gi, len(moves), _pgn(h, moves, FLIP[h["Result"]])))
        for k in _cuts(len(moves)):
            records.append(("trunc", gi, k, _pgn(h, moves[:k])))
            d = _chess_diverge(moves, k, rng)
            if d is not None:
                records.append(("div", gi, k, _pgn(h, d)))
    d = tmp_path_factory.mktemp("audit_chess")
    (d / "all.pgn").write_text("".join(r[3] for r in records))
    stats = _run(gmdata, ["chess", "convert", "--input", str(d / "all.pgn"),
                          "--out", str(d / "out"), "--min-plies", "1"])
    samples = load_gm_samples(str(d / "out"))
    assert stats.get("mask_mismatch") == 0 and len(samples) == len(records), stats
    return records, samples


def test_chess_truncation_and_divergence(chess_cases):
    records, samples = chess_cases
    full = {gi: s for (kind, gi, _, _), s in zip(records, samples) if kind == "full"}
    n_trunc = n_div = 0
    for (kind, gi, k, _), s in zip(records, samples):
        if kind == "trunc":
            assert s["tgt"]["from_idx"].shape[0] == k
            assert_prefix_identical(full[gi], s, k, f"chess game {gi} cut after {k}")
            n_trunc += 1
        elif kind == "div":
            lab = ("from_idx", "to_idx", "promotion")
            assert any(int(s["tgt"][x][k]) != int(full[gi]["tgt"][x][k]) for x in lab)
            assert_prefix_identical(full[gi], s, k, f"chess game {gi} diverging at {k}",
                                    obs_extra_frame=True)
            n_div += 1
    assert n_trunc >= 50 and n_div >= 50


def test_chess_result_header_only_moves_the_value(chess_cases):
    records, samples = chess_cases
    full = {gi: s for (kind, gi, _, _), s in zip(records, samples) if kind == "full"}
    for (kind, gi, _, _), s in zip(records, samples):
        if kind == "flip":
            assert_only_value_differs(full[gi], s, f"chess game {gi}")


def test_chess_repetition_flags_exercised(chess_cases):
    """The truncation cuts must hit games whose repetition flags fire, or the
    look-ahead check of globals 8/9 is vacuous."""
    records, samples = chess_cases
    reps = sum(int(s["obs"]["globals_seq"][:, 8].sum()) for (kind, *_), s in zip(records, samples)
               if kind == "full")
    assert reps > 0


# ---------------------------------------------------------------------------
# go (KGS SGF)
# ---------------------------------------------------------------------------

GO_MOVE = re.compile(r";\s*([BW])\[([a-z]{0,2})\]")


def _go_variants(text: str) -> List[Tuple[str, int, str]]:
    """(kind, k, sgf) variants of one single-game main-line SGF."""
    moves = list(GO_MOVE.finditer(text))
    n = len(moves)
    out = []
    re_m = re.search(r"RE\[([^\]]*)\]", text)
    if re_m:
        v = re_m.group(1)
        flipped = {"B": "W" + v[1:], "W": "B" + v[1:]}.get(v[:1], "B+R")
        out.append(("flip", n, text[:re_m.start(1)] + flipped + text[re_m.end(1):]))
    for k in _cuts(n):
        cut = moves[k - 1].end()
        out.append(("trunc", k, text[:cut] + ")\n"))
        colour = moves[k].group(1)
        if moves[k].group(2) not in ("", "tt"):     # diverge: a pass instead of the stone
            out.append(("div", k, text[:cut] + f";{colour}[])\n"))
    return out


@pytest.fixture(scope="module")
def go_cases(gmdata, tmp_path_factory):  # noqa: F811
    src = os.path.join(FIX, "go")
    d = tmp_path_factory.mktemp("audit_go")
    files = sorted(f for f in os.listdir(src) if f.endswith(".sgf"))
    expect = []
    for f in files:
        text = open(os.path.join(src, f), encoding="utf-8", errors="replace").read()
        if text.count("(") != 1:              # main line only, no variations
            continue
        stem = f[:-4]
        (d / "in").mkdir(exist_ok=True)
        (d / "in" / f).write_text(text)
        expect.append((stem, "full", len(GO_MOVE.findall(text))))
        for kind, k, sgf in _go_variants(text):
            name = f"{stem}__{kind}_{k:04d}.sgf"
            (d / "in" / name).write_text(sgf)
            expect.append((stem, kind, k))
    args = ["go", "convert", "--input", str(d / "in"), "--out", str(d / "out"), "--min-moves", "1",
            "--allow-handicap", "--allow-no-result"]
    _run(gmdata, args)
    by_src = {os.path.basename(s["meta"]["source"])[:-4]: s
              for s in load_gm_samples(str(d / "out"))}
    return expect, by_src


def test_go_truncation_and_divergence(go_cases):
    expect, by_src = go_cases
    n_trunc = n_div = n_flip = 0
    for stem, kind, k in expect:
        if kind == "full":
            continue
        full = by_src.get(stem)
        s = by_src.get(f"{stem}__{kind}_{k:04d}")
        if full is None:
            continue                          # the converter rejects the game itself
        assert s is not None, f"go {stem}: the {kind} variant at {k} was rejected"
        if kind == "flip":
            assert_only_value_differs(full, s, f"go {stem}")
            n_flip += 1
        elif kind == "trunc":
            assert s["tgt"]["place_idx"].shape[0] == k
            assert_prefix_identical(full, s, k, f"go {stem} cut after {k}")
            n_trunc += 1
        else:
            assert bool(s["tgt"]["passed"][k]) and not bool(full["tgt"]["passed"][k])
            assert_prefix_identical(full, s, k, f"go {stem} diverging at {k}", obs_extra_frame=True)
            n_div += 1
    assert n_trunc >= 100 and n_div >= 100 and n_flip >= 20


# ---------------------------------------------------------------------------
# shogi (floodgate CSA)
# ---------------------------------------------------------------------------

CSA_MOVE = re.compile(r"^([+-])(\d\d)(\d\d)([A-Z]{2})")
CSA_PIECES = ["* ", "FU", "KY", "KE", "GI", "KI", "KA", "HI", "OU",
              "TO", "NY", "NK", "NG", "UM", "RY"]
USI_DROP = {"FU": "P", "KY": "L", "KE": "N", "GI": "S", "KI": "G", "KA": "B", "HI": "R"}
HIRATE = ["P1-KY-KE-GI-KI-OU-KI-GI-KE-KY", "P2 * -HI *  *  *  *  * -KA * ",
          "P3-FU-FU-FU-FU-FU-FU-FU-FU-FU", "P4 *  *  *  *  *  *  *  *  * ",
          "P5 *  *  *  *  *  *  *  *  * ", "P6 *  *  *  *  *  *  *  *  * ",
          "P7+FU+FU+FU+FU+FU+FU+FU+FU+FU", "P8 * +KA *  *  *  *  * +HI * ",
          "P9+KY+KE+GI+KI+OU+KI+GI+KE+KY"]


def _usi_sq(csa: str) -> str:
    return csa[0] + "abcdefghi"[int(csa[1]) - 1]


def _csa_to_usi(board, line: str) -> str:
    import shogi
    m = CSA_MOVE.match(line)
    frm, to, pc = m.group(2), m.group(3), m.group(4)
    if frm == "00":
        return f"{USI_DROP[pc]}*{_usi_sq(to)}"
    moved = board.piece_at(shogi.SQUARE_NAMES.index(_usi_sq(frm))).piece_type
    return _usi_sq(frm) + _usi_sq(to) + ("+" if CSA_PIECES.index(pc) != moved else "")


def _usi_to_csa(board, usi: str) -> str:
    import shogi
    sign = "+" if board.turn == shogi.BLACK else "-"

    def csa_sq(s):
        return s[0] + str("abcdefghi".index(s[1]) + 1)
    if usi[1] == "*":
        pc = {v: k for k, v in USI_DROP.items()}[usi[0]]
        return f"{sign}00{csa_sq(usi[2:4])}{pc}"
    pt = board.piece_at(shogi.SQUARE_NAMES.index(usi[:2])).piece_type
    if usi.endswith("+"):
        pt += 8 if pt <= 4 else 7      # P L N S → +8, B R → +7 (UM 13, RY 14)
    return f"{sign}{csa_sq(usi[:2])}{csa_sq(usi[2:4])}{CSA_PIECES[pt]}"


def _shogi_variants(text: str, rng: random.Random) -> List[Tuple[str, int, str]]:
    import shogi
    lines = text.split("\n")
    if [ln for ln in lines if ln.startswith("P")] != HIRATE:
        return []
    mv_idx = [i for i, ln in enumerate(lines) if CSA_MOVE.match(ln)]
    res_idx = next(i for i, ln in enumerate(lines) if ln.startswith("%"))
    n = len(mv_idx)
    out = []
    flip = {"%TORYO": "%KACHI"}                      # resign ↔ declared win: the winner flips
    tok = lines[res_idx].split()[0]
    if tok in flip:
        out.append(("flip", n, "\n".join(lines[:res_idx] + [flip[tok]] + lines[res_idx + 1:])))
    board = shogi.Board()
    played = 0
    for k in _cuts(n):
        while played < k:
            board.push_usi(_csa_to_usi(board, lines[mv_idx[played]]))
            played += 1
        head = lines[:mv_idx[k - 1] + 1]
        out.append(("trunc", k, "\n".join(head + ["%TORYO", ""])))
        rec = _csa_to_usi(board, lines[mv_idx[k]])
        alts = sorted(m.usi() for m in board.legal_moves if m.usi() != rec)
        if alts:
            alt = _usi_to_csa(board, rng.choice(alts))
            out.append(("div", k, "\n".join(head + [alt, "%TORYO", ""])))
    return out


def _shogi_audit(binary: str, paths: Sequence[str], d) -> Tuple[Dict[str, int], Dict[str, Dict]]:
    """Convert every record of ``paths`` with its variants; check them all.
    Records the converter rejects by themselves (no result, illegal) are
    skipped; a variant of an accepted record must convert.  Returns the
    checked-variant counts and the full conversions."""
    rng = random.Random(0)
    (d / "in").mkdir()
    expect = []
    for gi, path in enumerate(paths):
        text = open(path, encoding="utf-8", errors="replace").read()
        (d / "in" / f"g{gi:05d}.csa").write_text(text)
        try:
            variants = _shogi_variants(text, rng)
        except (ValueError, KeyError, StopIteration, AttributeError):
            continue                     # a record python-shogi cannot replay: skip
        for kind, k, csa in variants:
            (d / "in" / f"g{gi:05d}__{kind}_{k:04d}.csa").write_text(csa)
            expect.append((gi, kind, k))
    stats = _run(binary, ["shogi", "convert", "--input", str(d / "in"), "--out", str(d / "out"),
                          "--min-moves", "1"])
    assert stats["mask_mismatch"] == 0, stats
    by_src = {s["meta"]["source"][:-4]: s for s in load_gm_samples(str(d / "out"))}
    n = {"trunc": 0, "div": 0, "flip": 0}
    for gi, kind, k in expect:
        full, s = by_src.get(f"g{gi:05d}"), by_src.get(f"g{gi:05d}__{kind}_{k:04d}")
        if full is None:
            continue
        assert s is not None, f"shogi game {gi}: its {kind} variant at {k} was rejected"
        if kind == "flip":
            assert_only_value_differs(full, s, f"shogi game {gi}")
        elif kind == "trunc":
            assert s["tgt"]["dropped"].shape[0] == k
            assert_prefix_identical(full, s, k, f"shogi game {gi} cut after {k}")
        else:
            lab = ("dropped", "from_idx", "to_idx", "promoted", "piece_idx", "drop_to_idx")
            assert any(int(s["tgt"][x][k]) != int(full["tgt"][x][k]) for x in lab)
            assert_prefix_identical(full, s, k, f"shogi game {gi} diverging at {k}",
                                    obs_extra_frame=True)
        n[kind] += 1
    fulls = {k: v for k, v in by_src.items() if "__" not in k}
    return n, fulls


@pytest.fixture(scope="module")
def shogi_cases(gmdata, tmp_path_factory):  # noqa: F811
    pytest.importorskip("shogi")
    src = os.path.join(FIX, "shogi")
    paths = [os.path.join(src, x) for x in sorted(os.listdir(src)) if x.endswith(".csa")]
    return _shogi_audit(gmdata, paths, tmp_path_factory.mktemp("audit_shogi"))


def test_shogi_truncation_and_divergence(shogi_cases):
    n, fulls = shogi_cases
    assert len(fulls) == 40
    assert n["trunc"] >= 100 and n["div"] >= 100 and n["flip"] >= 20, n


SHOGI_RAW = os.environ.get("GM_AUDIT_SHOGI_RAW", os.path.expanduser("~/gm-data/v3/raw/shogi"))


def test_shogi_truncation_and_divergence_local_raw(gmdata, tmp_path):  # noqa: F811
    """The same audit on a spread of local floodgate records
    ($GM_AUDIT_SHOGI_RAW, default ~/gm-data/v3/raw/shogi; skipped when absent;
    $GM_AUDIT_SHOGI_N records, default 300)."""
    pytest.importorskip("shogi")
    if not os.path.isdir(SHOGI_RAW):
        pytest.skip(f"no local shogi records at {SHOGI_RAW}")
    import glob
    paths = sorted(glob.glob(os.path.join(SHOGI_RAW, "**", "*.csa"), recursive=True))
    if not paths:
        pytest.skip(f"no .csa under {SHOGI_RAW}")
    want = int(os.environ.get("GM_AUDIT_SHOGI_N", "300"))
    paths = paths[:: max(1, len(paths) // want)][:want]
    n, fulls = _shogi_audit(gmdata, paths, tmp_path)
    assert n["trunc"] >= len(fulls) and n["div"] >= len(fulls), n


def test_shogi_repetition_counts_exercised(shogi_cases):
    _, fulls = shogi_cases
    reps = sum(int(s["obs"]["globals_seq"][:, 4].sum()) for s in fulls.values())
    assert reps > 0


# ---------------------------------------------------------------------------
# othello (WTHOR): records must end the game, so only divergence applies
# ---------------------------------------------------------------------------

WTHOR = os.path.join(FIX, "othello", "WTH_2021.wtb")


def _othello_playout(prefix: List[int], k: int, rng: random.Random):
    """prefix[:k] + another legal move at k + a random legal continuation to
    the end (WTHOR stores no passes; the converter re-inserts them)."""
    from data_processing.games.othello import rules
    b, player = rules.initial_board(), rules.BLACK
    out = []
    for sq in prefix[:k]:
        if not rules.has_legal_move(b, player):
            player = rules.opponent(player)
        rules.apply_move(b, player, sq // 8, sq % 8)
        out.append(sq)
        player = rules.opponent(player)
    first = True
    while not rules.is_game_over(b):
        if not rules.has_legal_move(b, player):
            player = rules.opponent(player)
        legal = [r * 8 + c for r, c in rules.legal_moves(b, player)]
        if first:
            legal = [sq for sq in legal if sq != prefix[k]]
            if not legal:
                return None
            first = False
        sq = rng.choice(legal)
        rules.apply_move(b, player, sq // 8, sq % 8)
        out.append(sq)
        player = rules.opponent(player)
    return out


def _wtb(games: List[List[int]]) -> bytes:
    head = bytearray(16)
    struct.pack_into("<I", head, 4, len(games))
    head[12] = 8
    body = bytearray()
    for mv in games:
        rec = bytearray(68)
        for i, sq in enumerate(mv):
            rec[8 + i] = 10 * (sq // 8 + 1) + sq % 8 + 1
        body += rec
    return bytes(head + body)


def _frame_of_move(passed: torch.Tensor, k: int) -> int:
    """Frame index of the k-th recorded (non-pass) move."""
    idx = torch.nonzero(~passed).squeeze(1)
    return int(idx[k])


def _parse_wtb(data: bytes) -> List[List[int]]:
    """WTHOR records → move lists (square = row * 8 + col; 0 = end of record)."""
    n1 = struct.unpack_from("<I", data, 4)[0]
    assert len(data) == 16 + 68 * n1
    games = []
    for i in range(n1):
        rec = data[16 + 68 * i + 8: 16 + 68 * (i + 1)]
        if 0 in rec:
            rec = rec[:rec.index(0)]
        games.append([(b // 10 - 1) * 8 + b % 10 - 1 for b in rec])
    return games


def test_othello_divergence(gmdata, tmp_path):  # noqa: F811
    rng = random.Random(0)
    with open(WTHOR, "rb") as fh:
        games = _parse_wtb(fh.read())[:40]
    recs, meta = [], []
    for gi, mv in enumerate(games):
        recs.append(mv)
        meta.append((gi, "full", 0))
        for k in _cuts(len(mv)):
            alt = _othello_playout(mv, k, rng)
            if alt is not None:
                recs.append(alt)
                meta.append((gi, "div", k))
    (tmp_path / "in").mkdir()
    (tmp_path / "in" / "audit.wtb").write_bytes(_wtb(recs))
    _run(gmdata, ["othello", "convert", "--input", str(tmp_path / "in"),
                  "--out", str(tmp_path / "out")])
    by_idx = {s["meta"]["game_index"]: s for s in load_gm_samples(str(tmp_path / "out"))}
    assert len(by_idx) == len(recs), "the converter rejected a legal playout"
    full = {gi: by_idx[i] for i, (gi, kind, _) in enumerate(meta) if kind == "full"}
    n_div = 0
    for i, (gi, kind, k) in enumerate(meta):
        if kind != "div":
            continue
        s, f = by_idx[i], full[gi]
        t = _frame_of_move(f["tgt"]["passed"], k)
        assert t == _frame_of_move(s["tgt"]["passed"], k)
        assert int(s["tgt"]["place_idx"][t]) != int(f["tgt"]["place_idx"][t])
        assert_prefix_identical(f, s, t, f"othello game {gi} diverging at move {k}",
                                obs_extra_frame=True)
        n_div += 1
    assert n_div >= 100


# ---------------------------------------------------------------------------
# gomoku (Gomocup psq)
# ---------------------------------------------------------------------------

PSQ_MOVE = re.compile(r"^\s*\d+,\d+(,-?\d+)?\s*$")


def _gomoku_variants(text: str, rng: random.Random) -> List[Tuple[str, int, str]]:
    lines = text.split("\n")
    mv_idx = [i for i, ln in enumerate(lines) if PSQ_MOVE.match(ln)]
    # the move block is the first contiguous run of move lines
    run = [mv_idx[0]]
    for i in mv_idx[1:]:
        if i != run[-1] + 1:
            break
        run.append(i)
    n = len(run)
    size = int(re.search(r"(\d+)x(\d+)", lines[0]).group(1))
    occupied = [tuple(int(x) for x in lines[i].split(",")[:2]) for i in run]
    tail = lines[run[-1] + 1:]
    out = []
    for k in _cuts(n):
        out.append(("trunc", k, "\n".join(lines[:run[k - 1] + 1] + tail)))
        free = [(x, y) for x in range(1, size + 1) for y in range(1, size + 1)
                if (x, y) not in occupied[:k] and (x, y) != occupied[k]]
        x, y = rng.choice(free)
        out.append(("div", k, "\n".join(lines[:run[k - 1] + 1] + [f"{x},{y},0"] + tail)))
    return out


def test_gomoku_truncation_and_divergence(gmdata, tmp_path):  # noqa: F811
    rng = random.Random(0)
    src = os.path.join(FIX, "gomoku")
    (tmp_path / "in").mkdir()
    expect = []
    for f in sorted(x for x in os.listdir(src) if x.endswith(".psq")):
        text = open(os.path.join(src, f), encoding="utf-8", errors="replace").read()
        stem = f[:-4]
        (tmp_path / "in" / f).write_text(text)
        for kind, k, psq in _gomoku_variants(text, rng):
            (tmp_path / "in" / f"{stem}__{kind}_{k:04d}.psq").write_text(psq)
            expect.append((stem, kind, k))
    _run(gmdata, ["gomoku", "convert", "--input", str(tmp_path / "in"),
                  "--out", str(tmp_path / "out"),
                  "--board-size", "20", "--min-moves", "1", "--keep-undecided"])
    by_src = {s["meta"]["source"][:-4]: s for s in load_gm_samples(str(tmp_path / "out"))}
    n = {"trunc": 0, "div": 0}
    for stem, kind, k in expect:
        full, s = by_src.get(stem), by_src.get(f"{stem}__{kind}_{k:04d}")
        if full is None:
            continue
        assert s is not None, f"gomoku {stem}: {kind} at {k} rejected"
        if kind == "trunc":
            assert s["tgt"]["place_idx"].shape[0] == k
            assert_prefix_identical(full, s, k, f"gomoku {stem} cut after {k}")
        else:
            assert int(s["tgt"]["place_idx"][k]) != int(full["tgt"]["place_idx"][k])
            assert_prefix_identical(full, s, k, f"gomoku {stem} diverging at {k}",
                                    obs_extra_frame=True)
        n[kind] += 1
    assert n["trunc"] >= 100 and n["div"] >= 100, n
