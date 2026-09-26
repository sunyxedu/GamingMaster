"""Chess converter v3 (`gmdata chess convert`): legality masks and repetition
flags on the committed Lichess fixture, cross-checked against python-chess.

    obs.globals_seq [T, 10]: 0..7 as v2, 8/9 = position seen ≥ 1 / ≥ 2 times before
    tgt.legal_from / legal_to_of [T, 64], tgt.legal_promotion_of [T, 5]  {0, 1}

python-chess replays every converted game; its legal moves are mapped with
the converter's convention (rank mirror for black, castling = king → g/c
file, promotion 0 none / 1 N / 2 B / 3 R / 4 Q) and ``board.is_repetition``
defines the repetition flags.
"""
from __future__ import annotations

import subprocess

import chess
import chess.pgn
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, load_gm_samples
from tests.test_rust_converters_parity import CASES, FIX, gmdata  # noqa: F401

PGN = f"{FIX}/chess/lichess_2013-01_head.pgn"
PROMO = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}


@pytest.fixture(scope="module")
def converted(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("chess_v3") / "rs"
    args, _ = CASES["chess"]
    r = subprocess.run([gmdata, *args, "--out", str(out)], check=True,
                       capture_output=True, text=True)
    return out, load_gm_samples(str(out)), r.stdout


def pov(sq: int, black: bool) -> int:
    rank, file = chess.square_rank(sq), chess.square_file(sq)
    return rank * 8 + file if black else (7 - rank) * 8 + file


def target(move: chess.Move, black: bool):
    return pov(move.from_square, black), pov(move.to_square, black), PROMO[move.promotion]


def replay_games(samples):
    """python-chess mainline moves of each converted sample (file order; the
    converter skips filtered games, so match on players/result/length)."""
    games = []
    with open(PGN) as fh:
        while (g := chess.pgn.read_game(fh)) is not None:
            games.append(g)
    out, gi = [], 0
    for s in samples:
        meta, T = s["meta"], s["tgt"]["from_idx"].shape[0]
        while True:
            g = games[gi]
            gi += 1
            moves = list(g.mainline_moves())
            h = g.headers
            if (h.get("White"), h.get("Black"), h.get("Result"), len(moves)) == (
                    meta["white"], meta["black"], meta["result"], T):
                out.append(moves)
                break
    return out


def ones(row: torch.Tensor):
    return set(row.nonzero().flatten().tolist())


def test_shard_schema(converted):
    out, samples, stdout = converted
    assert '"mask_mismatch":0' in stdout
    ds = GameShardDataset(str(out))
    assert ds.converter_version == 3 and ds.shard_meta == {"global_features": 10}
    for s in samples:
        T = s["tgt"]["from_idx"].shape[0]
        g = s["obs"]["globals_seq"]
        assert g.shape == (T, 10) and g.dtype == torch.float32
        flags = g[:, 8:]
        assert ((flags == 0) | (flags == 1)).all()
        assert (flags[:, 1] <= flags[:, 0]).all()          # seen twice ⇒ seen once
        for k, n in (("legal_from", 64), ("legal_to_of", 64), ("legal_promotion_of", 5)):
            m = s["tgt"][k]
            assert m.shape == (T, n) and m.dtype == torch.float32
            assert ((m == 0) | (m == 1)).all()


def test_gt_move_inside_masks(converted):
    _, samples, _ = converted
    for s in samples:
        t = s["tgt"]
        ar = torch.arange(t["from_idx"].shape[0])
        assert (t["legal_from"][ar, t["from_idx"]] == 1).all()
        assert (t["legal_to_of"][ar, t["to_idx"]] == 1).all()
        assert (t["legal_promotion_of"][ar, t["promotion"]] == 1).all()


def test_masks_and_repetition_match_python_chess(converted):
    _, samples, _ = converted
    n_rep = n_rep3 = n_promo = n_castle = 0
    for s, moves in zip(samples, replay_games(samples), strict=True):
        o, t = s["obs"], s["tgt"]
        board = chess.Board()
        for k, move in enumerate(moves):
            black = board.turn == chess.BLACK
            legal = [target(m, black) for m in board.legal_moves]
            gf, gt, gp = target(move, black)
            assert (int(t["from_idx"][k]), int(t["to_idx"][k]), int(t["promotion"][k])) \
                == (gf, gt, gp)
            assert ones(t["legal_from"][k]) == {f for f, _, _ in legal}
            assert ones(t["legal_to_of"][k]) == {d for f, d, _ in legal if f == gf}
            assert ones(t["legal_promotion_of"][k]) == {
                p for f, d, p in legal if (f, d) == (gf, gt)}
            g = o["globals_seq"][k]
            own = board.turn
            want = torch.tensor([
                board.has_kingside_castling_rights(own),
                board.has_queenside_castling_rights(own),
                board.has_kingside_castling_rights(not own),
                board.has_queenside_castling_rights(not own),
                own == chess.WHITE,
                min(board.fullmove_number / 100, 1.0),
                min(board.halfmove_clock / 100, 1.0),
                board.is_check(),
                board.is_repetition(2),
                board.is_repetition(3),
            ], dtype=torch.float32)
            assert torch.allclose(g, want, atol=1e-6), (k, g, want)
            n_rep += bool(g[8])
            n_rep3 += bool(g[9])
            n_promo += gp > 0
            n_castle += board.is_castling(move)
            board.push(move)
    # the fixture exercises every branch
    assert n_rep > 0 and n_rep3 > 0 and n_promo > 0 and n_castle > 0, (
        n_rep, n_rep3, n_promo, n_castle)
