"""Gomoku data pipeline tests (Gomocup .psq → shards).

Covers:
  1. .psq parsing (header size, 1-based x=col/y=row moves, metadata trailer)
  2. rules: five/overline detection in all four directions, occupied-cell
     illegality
  3. POV planes (channel swap, no flip) + label/value correctness on a
     hand-written decisive game
  4. skips: undecided (default) vs --keep_undecided, moves after a win,
     illegal move, wrong board size, too short
  5. e2e: convert → GameShardDataset → collate → build_game_network("gomoku")
     forward → il_loss (pure placement, allow_pass=False) → backward finite
  6. --board_size 20 (Gomocup freestyle league size) + shard meta,
     --rule standard (exact five; overlines are not wins) vs freestyle
"""

import os

import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
)
from data_processing.games.gomoku.convert import (
    BOARD_SIZE,
    SkipGame,
    convert,
    game_to_sample,
    parse_psq,
)
from data_processing.games.gomoku.rules import (
    BLACK,
    WHITE,
    IllegalMove,
    apply_move,
    initial_board,
    wins,
)
from model.games import build_game_network

# Black plays a horizontal five on row 7 (cols 3..7); white answers on
# row 8.  9 plies, black wins on the last move.
BLACK_WIN_MOVES = [
    (7, 3), (8, 3),
    (7, 4), (8, 4),
    (7, 5), (8, 5),
    (7, 6), (8, 6),
    (7, 7),
]

# 20x20 variant: black's winning five ends in the very last cell (19,19)
# so place_idx pins row*20+col indexing; white answers on row 18.
BLACK_WIN_MOVES_20 = [
    (19, 15), (18, 15),
    (19, 16), (18, 16),
    (19, 17), (18, 17),
    (19, 18), (18, 18),
    (19, 19),
]

# Move 11 joins black's row-0 stones (cols 0..3 and 5) into a 6-run
# overline; move 12 gives white an exact five on row 5.  Freestyle: the
# overline wins at move 11 → move 12 is moves_after_win.  Standard: the
# overline is not a win, white's exact five decides.
OVERLINE_THEN_FIVE_MOVES = [
    (0, 0), (5, 0),
    (0, 1), (5, 1),
    (0, 2), (5, 2),
    (0, 3), (5, 3),
    (0, 5), (9, 9),
    (0, 4), (5, 4),
]


def psq_text(moves, header="Piskvorky 15x15, 11:11, 0", trailer=True):
    lines = [header]
    lines += [f"{col + 1},{row + 1},1000" for row, col in moves]
    if trailer:
        lines += ["some_engine_0.0", "other_engine_1.2", "-1"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# 1. parsing
# ----------------------------------------------------------------------

def test_parse_psq_roundtrip():
    w, h, moves = parse_psq(psq_text(BLACK_WIN_MOVES))
    assert (w, h) == (15, 15)
    assert moves == BLACK_WIN_MOVES     # trailer ignored, coords 0-based


def test_parse_psq_bad_header():
    with pytest.raises(SkipGame, match="header"):
        parse_psq("not a psq file\n1,1,0\n")


# ----------------------------------------------------------------------
# 2. rules
# ----------------------------------------------------------------------

@pytest.mark.parametrize("stones,last", [
    ([(3, 3), (3, 4), (3, 5), (3, 6), (3, 7)], (3, 7)),          # horizontal
    ([(2, 9), (3, 9), (4, 9), (5, 9), (6, 9)], (4, 9)),          # vertical
    ([(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], (1, 1)),          # diagonal
    ([(10, 4), (9, 5), (8, 6), (7, 7), (6, 8)], (8, 6)),         # anti-diag
    ([(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], (0, 3)),  # overline
])
def test_wins_all_directions(stones, last):
    board = initial_board(BOARD_SIZE)
    for r, c in stones:
        apply_move(board, BOARD_SIZE, BLACK, r, c)
    assert wins(board, BOARD_SIZE, BLACK, *last)
    assert not wins(board, BOARD_SIZE, WHITE, *last)


def test_four_is_not_a_win():
    board = initial_board(BOARD_SIZE)
    for c in range(4):
        apply_move(board, BOARD_SIZE, BLACK, 7, c)
    assert not wins(board, BOARD_SIZE, BLACK, 7, 3)


def test_exact_five_rejects_overline_accepts_five():
    board = initial_board(BOARD_SIZE)
    for c in range(6):                              # overline, cols 0..5
        apply_move(board, BOARD_SIZE, BLACK, 0, c)
    assert wins(board, BOARD_SIZE, BLACK, 0, 3)     # freestyle: 6 wins
    assert not wins(board, BOARD_SIZE, BLACK, 0, 3, exact_five=True)

    board = initial_board(BOARD_SIZE)
    for c in range(5):                              # exactly five
        apply_move(board, BOARD_SIZE, BLACK, 3, c)
    assert wins(board, BOARD_SIZE, BLACK, 3, 4, exact_five=True)


def test_occupied_cell_is_illegal():
    board = initial_board(BOARD_SIZE)
    apply_move(board, BOARD_SIZE, BLACK, 7, 7)
    with pytest.raises(IllegalMove):
        apply_move(board, BOARD_SIZE, WHITE, 7, 7)


# ----------------------------------------------------------------------
# 3. sample correctness on the hand-written game
# ----------------------------------------------------------------------

def test_black_win_sample_labels_and_pov():
    s = game_to_sample(BLACK_WIN_MOVES)
    T = len(BLACK_WIN_MOVES)
    board_seq = s["obs"]["board_seq"]
    assert board_seq.shape == (T, 3, 15, 15)
    assert s["tgt"]["place_idx"].tolist() == [
        r * 15 + c for r, c in BLACK_WIN_MOVES
    ]
    assert not s["tgt"]["passed"].any()
    assert s["meta"]["winner"] == "black"

    # frame 0: empty board, black to move → planes 0/1 empty, plane 2 full
    assert board_seq[0, :2].sum() == 0
    assert board_seq[0, 2].sum() == 15 * 15

    # frame 2 (black to move, after 7,3 and 8,3): own plane holds black's
    # stone, opp plane white's — the channel swap on frame 1 showed the
    # reverse to white.
    assert board_seq[2, 0, 7, 3] == 1.0 and board_seq[2, 1, 8, 3] == 1.0
    assert board_seq[1, 0, 8, 3] == 0.0          # white's own stone not yet placed
    assert board_seq[1, 1, 7, 3] == 1.0          # black's stone is white's opp

    # value is mover-POV: black frames (even t) +1, white frames −1
    v = s["tgt"]["value"]
    assert (v[0::2] == 1.0).all() and (v[1::2] == -1.0).all()


# ----------------------------------------------------------------------
# 4. skips
# ----------------------------------------------------------------------

def test_undecided_skipped_by_default():
    undecided = BLACK_WIN_MOVES[:-1] + [(12, 12)]   # no five anywhere
    with pytest.raises(SkipGame, match="undecided"):
        game_to_sample(undecided)


def test_keep_undecided_last_mover_wins():
    undecided = BLACK_WIN_MOVES[:-1] + [(12, 12)]   # 9 plies, black last
    s = game_to_sample(undecided, keep_undecided=True)
    assert s["meta"]["winner"] == "black"
    assert (s["tgt"]["value"][0::2] == 1.0).all()


def test_moves_after_win_skipped():
    with pytest.raises(SkipGame, match="moves_after_win"):
        game_to_sample(BLACK_WIN_MOVES + [(0, 0)], min_moves=1)


def test_illegal_move_skipped():
    with pytest.raises(SkipGame, match="illegal_move"):
        game_to_sample([(7, 7), (7, 7)] + BLACK_WIN_MOVES, min_moves=1)


def test_too_short_skipped():
    with pytest.raises(SkipGame, match="too_short"):
        game_to_sample(BLACK_WIN_MOVES[:4])


def test_overline_game_freestyle_vs_standard():
    # Same game: freestyle treats the mid-game overline as a win (so the
    # remaining moves make it unusable), standard plays on and white's
    # exact five on the last move decides.
    with pytest.raises(SkipGame, match="moves_after_win"):
        game_to_sample(OVERLINE_THEN_FIVE_MOVES)
    s = game_to_sample(OVERLINE_THEN_FIVE_MOVES, exact_five=True)
    assert s["meta"]["winner"] == "white"
    v = s["tgt"]["value"]
    assert (v[1::2] == 1.0).all() and (v[0::2] == -1.0).all()


# ----------------------------------------------------------------------
# 5. --board_size / --rule / shard meta
# ----------------------------------------------------------------------

def test_convert_board_size_20(tmp_path):
    src = tmp_path / "psq"
    src.mkdir()
    (src / "win20.psq").write_text(
        psq_text(BLACK_WIN_MOVES_20, header="Piskvorky 20x20, 11:11, 0"))
    (src / "win15.psq").write_text(psq_text(BLACK_WIN_MOVES))
    out = tmp_path / "shards"

    stats = convert(str(src), str(out), board_size=20)
    assert stats["kept"] == 1
    assert stats["wrong_board_size"] == 1       # 15x15 file skipped in turn

    ds = GameShardDataset(str(out), game="gomoku")
    assert len(ds) == 1
    s = ds[0]
    T = len(BLACK_WIN_MOVES_20)
    assert s["obs"]["board_seq"].shape == (T, 3, 20, 20)
    assert s["tgt"]["place_idx"].tolist() == [
        r * 20 + c for r, c in BLACK_WIN_MOVES_20
    ]
    assert s["tgt"]["place_idx"][-1] == 19 * 20 + 19
    assert s["meta"]["winner"] == "black"
    v = s["tgt"]["value"]
    assert (v[0::2] == 1.0).all() and (v[1::2] == -1.0).all()


def test_convert_rejects_bad_board_size(tmp_path):
    for bad in (4, 26):
        with pytest.raises(ValueError, match="board_size"):
            convert(str(tmp_path), str(tmp_path / "out"), board_size=bad)


def test_convert_rule_standard_vs_freestyle(tmp_path):
    src = tmp_path / "psq"
    src.mkdir()
    (src / "overline.psq").write_text(psq_text(OVERLINE_THEN_FIVE_MOVES))

    stats = convert(str(src), str(tmp_path / "std"), rule="standard")
    assert stats["kept"] == 1
    ds = GameShardDataset(str(tmp_path / "std"), game="gomoku")
    assert ds[0]["meta"]["winner"] == "white"

    stats = convert(str(src), str(tmp_path / "free"), rule="freestyle")
    assert stats["kept"] == 0
    assert stats["moves_after_win"] == 1


def test_shard_meta_records_board_size(tmp_path):
    for size, moves, header in (
        (15, BLACK_WIN_MOVES, "Piskvorky 15x15, 11:11, 0"),
        (20, BLACK_WIN_MOVES_20, "Piskvorky 20x20, 11:11, 0"),
    ):
        src = tmp_path / f"psq{size}"
        src.mkdir()
        (src / "win.psq").write_text(psq_text(moves, header=header))
        out = tmp_path / f"shards{size}"
        stats = convert(str(src), str(out), board_size=size)
        assert stats["kept"] == 1
        shard = torch.load(out / "gomoku_00000.pt", map_location="cpu",
                           weights_only=False)
        assert shard["meta"]["board_size"] == size


# ----------------------------------------------------------------------
# 6. e2e: convert → dataset → network → il_loss → backward
# ----------------------------------------------------------------------

def test_e2e_convert_dataset_network(tmp_path):
    src = tmp_path / "psq"
    src.mkdir()
    (src / "win.psq").write_text(psq_text(BLACK_WIN_MOVES))
    (src / "undecided.psq").write_text(
        psq_text(BLACK_WIN_MOVES[:-1] + [(12, 12)]))
    (src / "bad_size.psq").write_text(
        psq_text(BLACK_WIN_MOVES, header="Piskvorky 20x20, 11:11, 0"))
    out = tmp_path / "shards"

    stats = convert(str(src), str(out))
    assert stats["kept"] == 1
    assert stats["undecided"] == 1
    assert stats["wrong_board_size"] == 1

    ds = GameShardDataset(str(out), game="gomoku")
    assert len(ds) == 1
    batch = collate_game_sequences([ds[0]])
    frame_mask = batch["meta"]["frame_mask"]

    torch.manual_seed(0)
    net = build_game_network(
        "gomoku", d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, dropout=0.0, encoder_channels=16,
    )
    assert net.action_decoder.pass_head is None     # allow_pass=False preset
    net.train()
    pred = net(batch["obs"], frame_mask=frame_mask)
    assert "pass_logits" not in pred
    losses = net.action_decoder.il_loss(pred, batch["tgt"], frame_mask)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
