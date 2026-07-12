"""End-to-end tests for the chess data pipeline (Lichess PGN → shards).

Covers:
  1. conversion produces a valid shard (counts, T per game, dtypes, shapes)
  2. POV mapping correctness for 1.e4 (white frame) and 1...e5 (black frame)
  3. label legality round-trip: un-flipped labels replayed through
     python-chess reproduce every game move-for-move (all games, all frames)
  4. promotion and en-passant frames encoded as specced
  5. value is from the mover's perspective (+1 win / 0 draw / −1 loss)
  6. shard → GameShardDataset → collate → chess network forward →
     il_loss finite, backward produces finite grads
"""

import io
import os

import chess
import chess.pgn
import pytest
import torch

from data_processing.games.chess.convert import (
    EP_PLANE,
    IDX_TO_PROMO,
    ONES_PLANE,
    convert,
    encode_game,
    pov_idx_to_square,
    square_to_pov_idx,
)
from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
)
from model.games import build_game_network

# Three short real games (validated legal with python-chess):
#   A: both sides castle, white wins    (10 plies)
#   B: en passant capture, draw         (8 plies)
#   C: black promotes on h1, black wins (10 plies)
PGN = """\
[Event "Test Castling"]
[White "Alice"]
[Black "Bob"]
[Result "1-0"]
[WhiteElo "1800"]
[BlackElo "1750"]
[TimeControl "300+3"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O Bc5 5. c3 O-O 1-0

[Event "Test En Passant"]
[White "Carol"]
[Black "Dave"]
[Result "1/2-1/2"]
[WhiteElo "2100"]
[BlackElo "2050"]
[TimeControl "60+0"]

1. e4 Nf6 2. e5 d5 3. exd6 exd6 4. d4 d5 1/2-1/2

[Event "Test Promotion"]
[White "Eve"]
[Black "Frank"]
[Result "0-1"]
[WhiteElo "1500"]
[BlackElo "1600"]
[TimeControl "180+2"]

1. Na3 h5 2. Nb1 h4 3. Na3 h3 4. Nb1 hxg2 5. Na3 gxh1=Q 0-1
"""

PLIES = {"1-0": 10, "1/2-1/2": 8, "0-1": 10}  # keyed by result, all unique


def read_games():
    handle = io.StringIO(PGN)
    games = []
    while True:
        g = chess.pgn.read_game(handle)
        if g is None:
            break
        games.append(g)
    assert len(games) == 3
    return games


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("chess_shards")
    pgn_path = d / "games.pgn"
    pgn_path.write_text(PGN)
    out = d / "out"
    stats = convert(str(pgn_path), str(out), shard_size=256, min_elo=0,
                    min_plies=8)
    assert stats == {"seen": 3, "kept": 3, "shards": 1}
    return str(out)


def load_samples(shard_dir):
    paths = sorted(
        os.path.join(shard_dir, p) for p in os.listdir(shard_dir)
        if p.endswith(".pt")
    )
    assert len(paths) == 1
    shard = torch.load(paths[0], map_location="cpu", weights_only=False)
    assert shard["game"] == "chess"
    return shard["samples"]


# ======================================================================
# 0. POV mapping helper is exact and invertible
# ======================================================================

def test_pov_mapping_invertible_and_known_squares():
    for flip in (False, True):
        for sq in chess.SQUARES:
            idx = square_to_pov_idx(sq, flip)
            assert 0 <= idx < 64
            assert pov_idx_to_square(idx, flip) == sq
    # White POV: e2 → row 6 (rank 1), col 4 (file e) → 52; e4 → 36.
    assert square_to_pov_idx(chess.E2, flip=False) == 6 * 8 + 4
    assert square_to_pov_idx(chess.E4, flip=False) == 4 * 8 + 4
    # Black POV (rank mirror, files kept): e7 lands on the SAME index as
    # white's e2 — that is the color symmetry the mapping exists for.
    assert square_to_pov_idx(chess.E7, flip=True) == 6 * 8 + 4
    assert square_to_pov_idx(chess.A8, flip=True) == 7 * 8 + 0
    # Kingside castling encodes identically for both colors: 60 → 62.
    assert square_to_pov_idx(chess.E1, flip=False) == 60
    assert square_to_pov_idx(chess.G1, flip=False) == 62
    assert square_to_pov_idx(chess.E8, flip=True) == 60
    assert square_to_pov_idx(chess.G8, flip=True) == 62


# ======================================================================
# 1. Shard structure: counts, lengths, dtypes, shapes
# ======================================================================

def test_shard_counts_shapes_dtypes(shard_dir):
    samples = load_samples(shard_dir)
    assert len(samples) == 3
    seen_plies = {}
    for s in samples:
        T = s["obs"]["board_seq"].shape[0]
        seen_plies[s["meta"]["result"]] = T
        assert s["obs"]["board_seq"].shape == (T, 14, 8, 8)
        assert s["obs"]["board_seq"].dtype == torch.float32
        assert s["obs"]["globals_seq"].shape == (T, 8)
        assert s["obs"]["globals_seq"].dtype == torch.float32
        for k in ("from_idx", "to_idx", "promotion"):
            assert s["tgt"][k].shape == (T,)
            assert s["tgt"][k].dtype == torch.long
        assert s["tgt"]["value"].shape == (T,)
        assert s["tgt"]["value"].dtype == torch.float32
        assert (s["obs"]["board_seq"][:, ONES_PLANE] == 1.0).all()
        assert s["meta"]["elo"][0] is not None
    assert seen_plies == PLIES


def test_filters_min_elo_and_min_plies(tmp_path):
    pgn_path = tmp_path / "games.pgn"
    pgn_path.write_text(PGN)
    # min_elo=1900 keeps only game B (2100/2050); min_plies=9 then kills
    # its 8 plies too.
    stats = convert(str(pgn_path), str(tmp_path / "o1"), min_elo=1900,
                    min_plies=8)
    assert stats["kept"] == 1
    stats = convert(str(pgn_path), str(tmp_path / "o2"), min_elo=1900,
                    min_plies=9)
    assert stats["kept"] == 0
    # skip_bullet drops game B (60+0 → 60s < 180s).
    stats = convert(str(pgn_path), str(tmp_path / "o3"), min_plies=8,
                    skip_bullet=True)
    assert stats["kept"] == 2


NULL_MOVE_PGN = """\
[Event "Null Move"]
[White "Gwen"]
[Black "Hal"]
[Result "1-0"]

1. e4 e5 2. -- Nc6 3. Nf3 Nf6 4. Nc3 Bb4 5. a3 Ba5 1-0
"""

EMPTY_GAME_PGN = """\
[Event "Empty"]
[White "Ivy"]
[Black "Jon"]
[Result "1/2-1/2"]

1/2-1/2
"""


def test_null_move_game_is_dropped(tmp_path):
    """'--' parses into Move.null() with NO game.errors entry — the game
    must be dropped, not emitted with an unreplayable from==to label."""
    pgn_path = tmp_path / "null.pgn"
    pgn_path.write_text(NULL_MOVE_PGN)
    stats = convert(str(pgn_path), str(tmp_path / "out"), min_plies=1)
    assert stats["kept"] == 0
    assert encode_game(chess.pgn.read_game(io.StringIO(NULL_MOVE_PGN))) is None


def test_zero_ply_game_never_crashes(tmp_path):
    """A result-tagged game with an empty mainline must be filtered even
    with --min_plies 0 (encode_game would torch.stack empty lists)."""
    pgn_path = tmp_path / "empty.pgn"
    pgn_path.write_text(EMPTY_GAME_PGN)
    stats = convert(str(pgn_path), str(tmp_path / "out"), min_plies=0)
    assert stats == {"seen": 1, "kept": 0, "shards": 0}


# ======================================================================
# 2. POV correctness on known opening frames
# ======================================================================

def test_pov_frames_e4_e5(shard_dir):
    samples = load_samples(shard_dir)
    s = next(x for x in samples if x["meta"]["white"] == "Alice")

    # Frame 0: white to move, 1. e4. Own pawn plane (0) has e2 occupied.
    e2, e4 = 6 * 8 + 4, 4 * 8 + 4
    board0 = s["obs"]["board_seq"][0]
    assert board0[0].reshape(64)[e2] == 1.0
    assert s["tgt"]["from_idx"][0] == e2
    assert s["tgt"]["to_idx"][0] == e4
    # White starting position under white POV: own back rank is row 7.
    assert board0[0, 6].sum() == 8          # own pawns on row 6
    assert board0[5, 7, 4] == 1.0           # own king e1 → row 7, col 4
    assert board0[6, 1].sum() == 8          # opp pawns on row 1

    # Frame 1: black to move, 1... e5 — rank-mirrored: e7 → 52, e5 → 36
    # (same POV indices as white's e2/e4 — color symmetry).
    e7, e5 = 6 * 8 + 4, 4 * 8 + 4
    board1 = s["obs"]["board_seq"][1]
    assert board1[0].reshape(64)[e7] == 1.0
    assert s["tgt"]["from_idx"][1] == e7
    assert s["tgt"]["to_idx"][1] == e5
    assert board1[0, 6].sum() == 8          # own (black) pawns also row 6
    assert board1[5, 7, 4] == 1.0           # own king e8 → row 7, col 4
    # White's e4 pawn is now an opponent pawn at row 3, col 4 (mirrored e4).
    assert board1[6, 3, 4] == 1.0

    # Globals: side_to_move_is_white flag alternates; castling rights on.
    assert s["obs"]["globals_seq"][0][4] == 1.0
    assert s["obs"]["globals_seq"][1][4] == 0.0
    assert (s["obs"]["globals_seq"][0][:4] == 1.0).all()


# ======================================================================
# 3. Label legality round-trip (all games, all frames)
# ======================================================================

def test_labels_replay_all_games(shard_dir):
    samples = load_samples(shard_dir)
    games = read_games()
    by_result = {s["meta"]["result"]: s for s in samples}
    for game in games:
        s = by_result[game.headers["Result"]]
        board = game.board()
        original = list(game.mainline_moves())
        T = s["obs"]["board_seq"].shape[0]
        assert T == len(original)
        for t in range(T):
            flip = board.turn == chess.BLACK
            move = chess.Move(
                pov_idx_to_square(int(s["tgt"]["from_idx"][t]), flip),
                pov_idx_to_square(int(s["tgt"]["to_idx"][t]), flip),
                promotion=IDX_TO_PROMO[int(s["tgt"]["promotion"][t])],
            )
            assert move in board.legal_moves, (
                f"{game.headers['Event']} frame {t}: {move.uci()} illegal"
            )
            assert move == original[t], (
                f"{game.headers['Event']} frame {t}: "
                f"{move.uci()} != {original[t].uci()}"
            )
            # from-square holds an own piece in the emitted planes
            row, col = divmod(int(s["tgt"]["from_idx"][t]), 8)
            assert s["obs"]["board_seq"][t, 0:6, row, col].sum() > 0
            board.push(move)


# ======================================================================
# 4. Promotion and en-passant frames
# ======================================================================

def test_en_passant_frame(shard_dir):
    samples = load_samples(shard_dir)
    s = next(x for x in samples if x["meta"]["white"] == "Carol")
    # Frame 4: white plays exd6 e.p. after 3...d5; ep target = d6.
    # White POV: d6 → row 2, col 3.  from e5 → 28, to d6 → 19.
    d6 = 2 * 8 + 3
    ep_plane = s["obs"]["board_seq"][4, EP_PLANE].reshape(64)
    assert ep_plane[d6] == 1.0 and ep_plane.sum() == 1.0
    assert s["tgt"]["from_idx"][4] == 3 * 8 + 4
    assert s["tgt"]["to_idx"][4] == d6
    assert s["tgt"]["promotion"][4] == 0
    # Frame 1 (black to move after 1. e4): ep target e3, rank-mirrored →
    # rank 2 → row 2, file e → col 4: the POV mapping applies to plane 12.
    ep1 = s["obs"]["board_seq"][1, EP_PLANE].reshape(64)
    assert ep1[2 * 8 + 4] == 1.0 and ep1.sum() == 1.0
    # Frame 0 (initial position): no ep square anywhere.
    assert s["obs"]["board_seq"][0, EP_PLANE].sum() == 0.0


def test_promotion_frame(shard_dir):
    samples = load_samples(shard_dir)
    s = next(x for x in samples if x["meta"]["black"] == "Frank")
    promo = s["tgt"]["promotion"]
    # Only the final frame (black's gxh1=Q) promotes; label 4 = queen.
    assert promo[-1] == 4
    assert (promo[:-1] == 0).all()
    # Black POV (rank mirror): g2 → row 1, col 6 → 14; h1 → row 0, col 7 → 7.
    assert s["tgt"]["from_idx"][-1] == 1 * 8 + 6
    assert s["tgt"]["to_idx"][-1] == 0 * 8 + 7
    # The capturing pawn sits on g2 in the own-pawn plane.
    assert s["obs"]["board_seq"][-1, 0, 1, 6] == 1.0
    # The captured rook (h1) is an opponent rook at POV (0, 7).
    assert s["obs"]["board_seq"][-1, 6 + chess.ROOK - 1, 0, 7] == 1.0


# ======================================================================
# 5. Value is mover-perspective
# ======================================================================

def test_value_perspective(shard_dir):
    samples = load_samples(shard_dir)
    for s in samples:
        v = s["tgt"]["value"]
        white_frames, black_frames = v[0::2], v[1::2]
        if s["meta"]["result"] == "1-0":
            assert (white_frames == 1.0).all() and (black_frames == -1.0).all()
        elif s["meta"]["result"] == "0-1":
            assert (white_frames == -1.0).all() and (black_frames == 1.0).all()
        else:
            assert (v == 0.0).all()


# ======================================================================
# 6. E2E: shard → dataset → collate → network → il_loss → backward
# ======================================================================

def test_e2e_dataset_to_gradients(shard_dir):
    torch.manual_seed(0)
    ds = GameShardDataset(shard_dir, game="chess")
    assert len(ds) == 3
    batch = collate_game_sequences([ds[i] for i in range(3)])
    frame_mask = batch["meta"]["frame_mask"]
    assert batch["obs"]["board_seq"].shape == (3, 10, 14, 8, 8)
    assert frame_mask.shape == (3, 10)
    assert frame_mask.sum() == 10 + 8 + 10

    net = build_game_network(
        "chess", d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, dropout=0.0, encoder_channels=16,
    )
    net.train()
    labels = {k: batch["tgt"][k] for k in ("from_idx", "to_idx")}
    pred = net(batch["obs"], labels=labels, frame_mask=frame_mask)
    losses = net.action_decoder.il_loss(pred, batch["tgt"], frame_mask)
    mask_f = frame_mask.float()
    v_loss = (torch.nn.functional.mse_loss(
        pred["value"], batch["tgt"]["value"], reduction="none",
    ) * mask_f).sum() / mask_f.sum()
    loss = losses["loss"] + 0.2 * v_loss

    assert torch.isfinite(loss), losses
    for k in ("from_loss", "to_loss", "promo_loss"):
        assert torch.isfinite(losses[k]), (k, losses[k])
    loss.backward()

    got_grad = False
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
            got_grad = got_grad or p.grad.abs().sum() > 0
    assert got_grad
