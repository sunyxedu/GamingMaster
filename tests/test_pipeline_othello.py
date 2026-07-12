"""Tests for the Othello data pipeline (rules → WTHOR parser → converter).

Byte-convention ground truth: the move encoding was verified against the
FFO spec and two independent parsers (LimeEng/wthor, Edax game.c), and
empirically by replaying two real WTHOR databases (WTH_1977/1980.wtb:
172/172 games legal, 172/172 recorded scores reproduced).
"""

import random

import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
    save_shard,
)
from data_processing.games.othello.convert import (
    IllegalGameError,
    adjusted_black_score,
    board_planes,
    convert,
    game_to_sample,
    replay_frames,
)
from data_processing.games.othello.rules import (
    BLACK,
    WHITE,
    apply_move,
    disc_counts,
    flips,
    has_legal_move,
    initial_board,
    is_game_over,
    legal_moves,
    opponent,
)
from data_processing.games.othello.wthor import (
    WthorGame,
    build_wtb,
    decode_move,
    encode_move,
    parse_wtb,
)
from model.games import build_game_network

SMALL = dict(
    d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
    trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
    halt_max_steps=2, dropout=0.0, encoder_channels=16,
)


def random_playout(seed: int):
    """Play random legal moves to the end.  Returns (recorded moves,
    mover per recorded move, whether any forced pass occurred, board)."""
    rng = random.Random(seed)
    board = initial_board()
    player = BLACK
    moves, movers = [], []
    had_pass = False
    while not is_game_over(board):
        if not has_legal_move(board, player):
            had_pass = True          # forced pass — never recorded in WTHOR
            player = opponent(player)
            continue
        r, c = rng.choice(legal_moves(board, player))
        apply_move(board, player, r, c)
        moves.append((r, c))
        movers.append(player)
        player = opponent(player)
    return moves, movers, had_pass, board


def find_playout(seed0: int, want):
    for seed in range(seed0, seed0 + 500):
        out = random_playout(seed)
        if want(out):
            return out
    raise AssertionError("no random playout matched the predicate")


# ---------------------------------------------------------------- rules


def test_opening_position_and_first_moves():
    board = initial_board()
    assert disc_counts(board) == (2, 2)
    # Black's four legal openings: d3, c4, f5, e6.
    assert set(legal_moves(board, BLACK)) == {(2, 3), (3, 2), (4, 5), (5, 4)}
    assert set(legal_moves(board, WHITE)) == {(2, 4), (4, 2), (3, 5), (5, 3)}


def test_flip_in_every_direction():
    board = bytearray(64)
    ring = [(2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 2), (4, 3), (4, 4)]
    anchors = [(1, 1), (1, 3), (1, 5), (3, 1), (3, 5), (5, 1), (5, 3), (5, 5)]
    for r, c in ring:
        board[r * 8 + c] = WHITE
    for r, c in anchors:
        board[r * 8 + c] = BLACK
    flipped = flips(board, BLACK, 3, 3)
    assert sorted(flipped) == sorted(r * 8 + c for r, c in ring)  # all 8 dirs
    apply_move(board, BLACK, 3, 3)
    assert all(board[r * 8 + c] == BLACK for r, c in ring)
    assert board[3 * 8 + 3] == BLACK


def test_forced_pass_position():
    board = bytearray(64)
    board[0] = BLACK          # a1
    board[1] = WHITE          # b1
    # White cannot flank the lone corner disc; black can (plays (0,2)).
    assert not has_legal_move(board, WHITE)
    assert legal_moves(board, WHITE) == []
    assert (0, 2) in legal_moves(board, BLACK)
    assert not is_game_over(board)


def test_double_pass_termination_and_counts():
    board = bytearray(64)
    board[0] = BLACK          # wipeout-like: nothing can be flanked
    assert is_game_over(board)
    assert disc_counts(board) == (1, 0)

    # A full random game must terminate with neither side able to move.
    _, _, _, board = random_playout(seed=0)
    assert is_game_over(board)
    b, w = disc_counts(board)
    assert 0 < b + w <= 64


def test_apply_illegal_move_raises():
    with pytest.raises(Exception):
        apply_move(initial_board(), BLACK, 0, 0)


# --------------------------------------------------------------- wthor


def test_move_byte_convention():
    # Verified: byte = 10*(row+1) + (col+1); a1=11, h8=88, f5=56.
    assert decode_move(11) == (0, 0)
    assert decode_move(88) == (7, 7)
    assert decode_move(56) == (4, 5)          # f5, black's canonical opening
    assert encode_move(4, 5) == 56
    for byte in range(11, 89):
        r, c = divmod(byte, 10)
        if 1 <= (byte % 10) <= 8:
            assert encode_move(*decode_move(byte)) == byte
    for bad in (1, 9, 10, 19, 20, 89, 90, 100, 255):
        with pytest.raises(ValueError):
            decode_move(bad)


def _wthor_game_from_playout(seed: int) -> WthorGame:
    moves, _, _, board = random_playout(seed)
    b, w = disc_counts(board)
    return WthorGame(
        tournament_id=7, black_player_id=101, white_player_id=202,
        black_real_score=adjusted_black_score(b, w),
        black_theoretical_score=32, moves=moves,
    )


def test_wtb_roundtrip_synthetic_blob():
    games = [_wthor_game_from_playout(s) for s in (1, 2, 3)]
    blob = build_wtb(games, game_year=1999, depth=24)
    assert len(blob) == 16 + 68 * 3

    header, parsed = parse_wtb(blob)
    assert header.n_games == 3
    assert header.game_year == 1999
    assert header.board_size == 8 and header.game_type == 0
    assert len(parsed) == 3
    for orig, back in zip(games, parsed):
        assert back.tournament_id == orig.tournament_id
        assert back.black_player_id == orig.black_player_id
        assert back.white_player_id == orig.white_player_id
        assert back.black_real_score == orig.black_real_score
        assert back.black_theoretical_score == orig.black_theoretical_score
        assert back.moves == orig.moves


def test_wtb_early_end_trailing_zeros():
    # Wipeout / early double-pass games record < 60 moves; the rest of
    # the 60 move bytes is zero filler and must parse cleanly.
    moves, _, _, _ = find_playout(0, lambda out: len(out[0]) < 60)
    game = WthorGame(0, 1, 2, 40, 40, moves=moves)
    blob = build_wtb([game])
    record = blob[16:]
    assert len(record) == 68
    assert set(record[8 + len(moves):]) == {0}   # trailing filler bytes
    _, parsed = parse_wtb(blob)
    assert parsed[0].moves == moves
    assert len(parsed[0].moves) < 60


def test_wtb_size_mismatch_rejected():
    blob = build_wtb([_wthor_game_from_playout(4)])
    with pytest.raises(ValueError, match="size mismatch"):
        parse_wtb(blob[:-1])


# ------------------------------------------------------------ converter


def test_pass_insertion():
    moves, movers, had_pass, _ = find_playout(0, lambda out: out[2])
    assert had_pass
    frames, _ = replay_frames(moves)
    passes = [i for i, f in enumerate(frames) if f[3]]
    assert passes, "replay must reinsert the unstored forced pass"
    move_frames = [f for f in frames if not f[3]]
    assert len(move_frames) == len(moves)
    for i in passes:
        planes, mover, place_idx, passed = frames[i]
        assert passed and place_idx == 0
        # The frame after a pass belongs to the OTHER player — the one
        # who actually recorded the next move in the WTHOR file.
        assert i + 1 < len(frames), "a pass is only inserted before a move"
        assert frames[i + 1][1] == opponent(mover)
    # Every recorded move is attributed to the player who really made it.
    assert [f[1] for f in move_frames] == movers


def test_perspective_and_value():
    moves, movers, _, board = find_playout(0, lambda out: out[2])
    game = WthorGame(0, 1, 2, 0, 0, moves=moves)
    sample, _ = game_to_sample(game, "synthetic", 0)
    obs, tgt = sample["obs"], sample["tgt"]
    seq = obs["board_seq"]
    T = seq.shape[0]
    assert seq.shape == (T, 3, 8, 8) and seq.dtype == torch.float32
    assert tgt["place_idx"].dtype == torch.long
    assert tgt["passed"].dtype == torch.bool

    # Exactly one plane active per square.
    assert torch.equal(seq.sum(dim=1), torch.ones(T, 8, 8))

    # Frame 0 = opening from black's perspective.
    assert torch.equal(seq[0], board_planes(initial_board(), BLACK))

    frames, final_board = replay_frames(moves)
    frame_movers = [f[1] for f in frames]
    # Movers strictly alternate (a pass hands the turn over too)…
    for t in range(T - 1):
        assert frame_movers[t + 1] == opponent(frame_movers[t])
    # …and across a pass frame the BOARD is unchanged, so the planes are
    # exactly the previous frame's own/opp channel swap.
    for t in range(T - 1):
        if tgt["passed"][t]:
            assert torch.equal(seq[t + 1, 0], seq[t, 1])
            assert torch.equal(seq[t + 1, 1], seq[t, 0])
            assert torch.equal(seq[t + 1, 2], seq[t, 2])

    # Value: sign of the REPLAYED final disc lead, from each mover's side.
    b, w = disc_counts(final_board)
    black_sign = float(b > w) - float(b < w)
    expect = torch.tensor(
        [black_sign if m == BLACK else -black_sign for m in frame_movers]
    )
    assert torch.equal(tgt["value"], expect)
    if black_sign != 0.0:
        assert torch.equal(tgt["value"][1:], -tgt["value"][:-1])


def test_illegal_recorded_move_skips_game(tmp_path):
    good = _wthor_game_from_playout(5)
    bad = WthorGame(0, 1, 2, 33, 33, moves=[(0, 0), (2, 3)])  # a1: illegal
    with pytest.raises(IllegalGameError):
        game_to_sample(bad, "synthetic", 0)

    wtb = tmp_path / "SYN_2024.wtb"
    wtb.write_bytes(build_wtb([good, bad]))
    out_dir = tmp_path / "shards"
    stats = convert(str(wtb), str(out_dir), shard_size=512)
    assert stats["games"] == 2
    assert stats["converted"] == 1
    assert stats["illegal"] == 1
    assert stats["shards"] == 1
    assert stats["score_mismatch"] == 0      # scores were self-consistent

    ds = GameShardDataset(str(out_dir), game="othello")
    assert len(ds) == 1
    assert ds[0]["meta"]["n_recorded_moves"] == len(good.moves)


def test_truncated_record_rejected():
    # A record that stops while either side can still move must not
    # replay: its value labels would come from a non-terminal disc count.
    moves, _, _, _ = random_playout(seed=0)
    truncated = moves[: len(moves) // 2]
    with pytest.raises(IllegalGameError, match="truncated"):
        replay_frames(truncated)
    game = WthorGame(0, 1, 2, 33, 33, moves=truncated)
    with pytest.raises(IllegalGameError, match="truncated"):
        game_to_sample(game, "synthetic", 0)


# Shortest possible game: Black wipes White out in 9 moves
# (e6 f4 e3 f6 g5 d6 e7 f5 c5) — (row, col) 0-based.
WIPEOUT_MOVES = [
    (5, 4), (3, 5), (2, 4), (5, 5), (4, 6), (5, 3), (6, 4), (4, 5), (4, 2),
]


def test_early_wipeout_accepted():
    # A legitimate early finish well before 60 moves is game-over (no side
    # can move) and must convert despite being far shorter than a full game.
    frames, board = replay_frames(WIPEOUT_MOVES)
    b, w = disc_counts(board)
    assert w == 0 and b == 13                    # true wipeout
    assert len(frames) == 9 and not any(f[3] for f in frames)

    game = WthorGame(0, 1, 2, adjusted_black_score(b, w), 64,
                     moves=WIPEOUT_MOVES)
    sample, score_ok = game_to_sample(game, "wipeout", 0)
    assert score_ok                              # 13 + 51 empties = 64
    assert sample["meta"]["replayed_white"] == 0
    assert sample["tgt"]["value"].tolist() == [1, -1] * 4 + [1]


def test_e2e_shard_to_gradients(tmp_path):
    torch.manual_seed(0)
    games = [_wthor_game_from_playout(s) for s in (6, 7)]
    samples = [game_to_sample(g, "synthetic", i)[0]
               for i, g in enumerate(games)]
    save_shard(str(tmp_path / "othello_00000.pt"), "othello", samples)

    ds = GameShardDataset(str(tmp_path), game="othello", max_len=24)
    assert len(ds) == 2
    batch = collate_game_sequences([ds[0], ds[1]])
    obs, tgt = batch["obs"], batch["tgt"]
    frame_mask = batch["meta"]["frame_mask"]
    assert obs["board_seq"].shape[2:] == (3, 8, 8)

    net = build_game_network("othello", **SMALL)
    pred = net(obs, frame_mask=frame_mask)
    assert pred["place_logits"].shape[-1] == 64
    losses = net.action_decoder.il_loss(pred, tgt, frame_mask)
    mask_f = frame_mask.float()
    v_loss = (torch.nn.functional.mse_loss(
        pred["value"], tgt["value"], reduction="none",
    ) * mask_f).sum() / mask_f.sum()
    loss = losses["loss"] + 0.2 * v_loss
    assert torch.isfinite(loss), f"non-finite loss: {losses}"

    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads, "no gradients flowed"
    for g in grads:
        assert torch.isfinite(g).all()
