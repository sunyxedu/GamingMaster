"""Tests for the shogi data pipeline (floodgate CSA → shards).

Covers:
  1. conversion shapes / dtypes; T == ply count
  2. POV rotation: known sente + gote moves land at expected indices
  3. capture → hand count increments; drop labels + count decrement
  4. promotion flag + promoted plane afterwards
  5. label replay round-trip: rebuilt moves reproduce the original game
  6. value perspective alternates ±1; draw → 0
  7. skips: no result / illegal move / too short
  8. e2e: CLI → shard → GameShardDataset → collate → network il_loss

The CSA fixtures are real legal games: two hand-written (one with a
capture-to-hand, a drop and two promotions; one draw) and one generated
in-test by replaying legal moves with python-shogi and serializing back
to CSA.
"""

import pytest
import shogi
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
)
from data_processing.games.shogi.convert import (
    SkipGame,
    convert_game,
    encode_frame,
    main as convert_main,
    pov_square,
)
from model.games.registry import build_game_network

# ----------------------------------------------------------------------
# CSA fixtures
# ----------------------------------------------------------------------

CSA_PIECE = {1: "FU", 2: "KY", 3: "KE", 4: "GI", 5: "KI", 6: "KA", 7: "HI",
             8: "OU", 9: "TO", 10: "NY", 11: "NK", 12: "NG", 13: "UM", 14: "RY"}
PROMOTE = {1: 9, 2: 10, 3: 11, 4: 12, 6: 13, 7: 14}


def sq_csa(sq: int) -> str:
    """python-shogi square index → CSA file+rank digits (e.g. 56 → '77')."""
    return f"{9 - sq % 9}{sq // 9 + 1}"


def to_csa(usi_moves, end_token="%TORYO") -> str:
    """Serialize a legal game to a one-game CSA record (asserts legality)."""
    board = shogi.Board()
    lines = ["V2.2", "N+black_test", "N-white_test", "PI", "+"]
    for usi in usi_moves:
        move = shogi.Move.from_usi(usi)
        assert board.is_legal(move), f"fixture move {usi} is illegal"
        color = "+" if board.turn == shogi.BLACK else "-"
        if move.drop_piece_type is not None:
            frm, piece_type = "00", move.drop_piece_type
        else:
            frm = sq_csa(move.from_square)
            piece_type = board.piece_at(move.from_square).piece_type
            if move.promotion:
                piece_type = PROMOTE[piece_type]
        lines.append(f"{color}{frm}{sq_csa(move.to_square)}{CSA_PIECE[piece_type]}")
        board.push(move)
    lines.append(end_token)
    return "\n".join(lines) + "\n"


# Classic opening: Bx2b+ (capture to hand + promotion), Sx2b, B*4e (drop),
# Bx6c+ (second capture + promotion).  %TORYO with white to move → black wins.
MAIN_MOVES = ["7g7f", "3c3d", "8h2b+", "3a2b", "B*4e", "9c9d", "4e6c+"]
DRAW_MOVES = ["2g2f", "8c8d", "2f2e", "8d8e"]


def gen_moves(n_plies: int = 12):
    """Deterministic legal game: always the lexicographically first move."""
    board = shogi.Board()
    moves = []
    for _ in range(n_plies):
        usi = sorted(m.usi() for m in board.legal_moves)[0]
        moves.append(usi)
        board.push_usi(usi)
    return moves


@pytest.fixture(scope="module")
def main_sample():
    return convert_game(to_csa(MAIN_MOVES), min_moves=0)


# ----------------------------------------------------------------------
# 1. shapes / dtypes
# ----------------------------------------------------------------------

def test_conversion_shapes_and_dtypes(main_sample):
    T = len(MAIN_MOVES)
    obs, tgt = main_sample["obs"], main_sample["tgt"]
    assert obs["board_seq"].shape == (T, 28, 9, 9)
    assert obs["hand_seq"].shape == (T, 14, 4)
    assert obs["globals_seq"].shape == (T, 4)
    for k in ("board_seq", "hand_seq", "globals_seq"):
        assert obs[k].dtype == torch.float32
    for k in ("dropped", "from_idx", "to_idx", "piece_idx", "drop_to_idx"):
        assert tgt[k].shape == (T,) and tgt[k].dtype == torch.long, k
    assert tgt["promoted"].shape == (T,) and tgt["promoted"].dtype == torch.bool
    assert tgt["value"].shape == (T,) and tgt["value"].dtype == torch.float32
    assert main_sample["meta"]["plies"] == T
    assert main_sample["meta"]["players"] == ["black_test", "white_test"]
    assert main_sample["meta"]["result"] == "b"
    # board planes are one-hot; constant features present
    assert set(obs["board_seq"].unique().tolist()) <= {0.0, 1.0}
    assert (obs["hand_seq"][:, :, 3] == 1.0).all()
    assert (obs["globals_seq"][:, 3] == 1.0).all()


# ----------------------------------------------------------------------
# 2. POV rotation
# ----------------------------------------------------------------------

def test_pov_square_roundtrip():
    for sq in range(81):
        assert pov_square(sq, True) == sq
        assert pov_square(pov_square(sq, False), False) == sq
    assert pov_square(56, False) == 24        # 7g ↔ 3c under 180° rotation


def test_pov_frames_and_labels(main_sample):
    obs, tgt = main_sample["obs"], main_sample["tgt"]
    # Frame 0: sente plays P-7f (+7776FU).  7g = sq 56 = (6, 2), 7f = 47.
    assert tgt["dropped"][0] == 0
    assert tgt["from_idx"][0] == 56 and tgt["to_idx"][0] == 47
    assert obs["board_seq"][0, 0, 6, 2] == 1        # own pawn on 7g
    assert obs["board_seq"][0, 7, 8, 4] == 1        # own king on 5i = (8, 4)
    assert obs["board_seq"][0, 14 + 7, 0, 4] == 1   # opp king on 5a → POV (0, 4)
    assert obs["globals_seq"][0, 1] == 1.0          # mover is black

    # Frame 1: gote plays P-3d (-3334FU).  3c = sq 24 → POV 80-24 = 56;
    # 3d = sq 33 → POV 47.  Rotated gote frame mirrors the sente frame.
    assert tgt["from_idx"][1] == 56 and tgt["to_idx"][1] == 47
    assert obs["board_seq"][1, 0, 6, 2] == 1        # own (white) pawn on 3c
    assert obs["globals_seq"][1, 1] == 0.0          # mover is white
    # black's pawn now on 7f (sq 47) → opponent plane at POV 80-47=33=(3,6)
    assert obs["board_seq"][1, 14, 3, 6] == 1


# ----------------------------------------------------------------------
# 3. capture → hand, drop labels, count decrement
# ----------------------------------------------------------------------

def test_capture_drop_hand_counts(main_sample):
    obs, tgt = main_sample["obs"], main_sample["tgt"]
    hand = obs["hand_seq"]
    B_OWN, B_OPP = 5, 12                        # bishop slot: own / opponent

    assert hand[2, :, 0].sum() == 0             # before Bx2b+: all hands empty
    # frame 3 (white to move): black just captured a bishop → opponent slot
    assert hand[3, B_OPP, 0] == 1 and hand[3, B_OPP, 1] == 1
    assert hand[3, :7, 0].sum() == 0
    # frame 4 (black to move): white recaptured the horse → both hold a bishop
    assert hand[4, B_OWN, 0] == 1               # black's own bishop (next own frame)
    assert hand[4, B_OPP, 0] == 1               # white's bishop
    # frame 4 is the drop B*4e: piece_idx slot has count ≥ 1
    assert tgt["dropped"][4] == 1
    assert tgt["piece_idx"][4] == B_OWN
    assert tgt["drop_to_idx"][4] == 41          # 4e = sq 41, black POV
    assert tgt["from_idx"][4] == 0 and tgt["to_idx"][4] == 0
    assert not tgt["promoted"][4]
    assert hand[4, tgt["piece_idx"][4], 0] >= 1
    # frame 5 (white to move): black's bishop left the hand, white still holds
    assert hand[5, B_OPP, 0] == 0
    assert hand[5, B_OWN, 0] == 1


# ----------------------------------------------------------------------
# 4. promotion
# ----------------------------------------------------------------------

def test_promotion_flag_and_plane(main_sample):
    obs, tgt = main_sample["obs"], main_sample["tgt"]
    # frame 2: Bx2b+ from 8h (sq 64) to 2b (sq 16), promoting
    assert tgt["dropped"][2] == 0 and bool(tgt["promoted"][2])
    assert tgt["from_idx"][2] == 64 and tgt["to_idx"][2] == 16
    assert torch.equal(tgt["promoted"],
                       torch.tensor([0, 0, 1, 0, 0, 0, 1], dtype=torch.bool))
    # frame 3 (white POV): black's horse on 2b → opp +B plane 14+12=26 at
    # POV 80-16 = 64 = (7, 1)
    assert obs["board_seq"][3, 26, 7, 1] == 1
    # frame 6 (black POV): the dropped bishop sits unpromoted on 4e = (4, 5)
    assert obs["board_seq"][6, 5, 4, 5] == 1
    assert bool(tgt["promoted"][6])             # Bx6c+


# ----------------------------------------------------------------------
# 5. label replay round-trip
# ----------------------------------------------------------------------

def replay_from_labels(sample, expected_usi):
    obs, tgt = sample["obs"], sample["tgt"]
    board = shogi.Board()
    for t, usi in enumerate(expected_usi):
        mover_is_black = bool(obs["globals_seq"][t, 1] > 0.5)
        if tgt["dropped"][t]:
            to_sq = pov_square(int(tgt["drop_to_idx"][t]), mover_is_black)
            move = shogi.Move(None, to_sq,
                              drop_piece_type=int(tgt["piece_idx"][t]) + 1)
        else:
            move = shogi.Move(
                pov_square(int(tgt["from_idx"][t]), mover_is_black),
                pov_square(int(tgt["to_idx"][t]), mover_is_black),
                promotion=bool(tgt["promoted"][t]),
            )
        assert move.usi() == usi, f"frame {t}: {move.usi()} != {usi}"
        assert board.is_legal(move)
        board.push(move)


def test_label_replay_roundtrip(main_sample):
    replay_from_labels(main_sample, MAIN_MOVES)


def test_label_replay_roundtrip_generated():
    moves = gen_moves(12)
    sample = convert_game(to_csa(moves), min_moves=0)
    assert sample["obs"]["board_seq"].shape[0] == 12
    replay_from_labels(sample, moves)


# ----------------------------------------------------------------------
# 6. value perspective
# ----------------------------------------------------------------------

def test_value_alternates_mover_perspective(main_sample):
    # black wins: black-to-move frames (even) +1, gote frames −1
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    assert torch.equal(main_sample["tgt"]["value"], expected)

    # white wins (%TORYO with black to move after 12 plies)
    sample_w = convert_game(to_csa(gen_moves(12)), min_moves=0)
    assert sample_w["meta"]["result"] == "w"
    assert torch.equal(sample_w["tgt"]["value"][:4],
                       torch.tensor([-1.0, 1.0, -1.0, 1.0]))

    # draw → 0 everywhere
    sample_d = convert_game(to_csa(DRAW_MOVES, end_token="%SENNICHITE"),
                            min_moves=0)
    assert sample_d["meta"]["result"] == "draw"
    assert (sample_d["tgt"]["value"] == 0).all()


# ----------------------------------------------------------------------
# 7. skips
# ----------------------------------------------------------------------

def test_skip_reasons():
    no_result = to_csa(DRAW_MOVES).replace("%TORYO\n", "")
    with pytest.raises(SkipGame, match="no_result"):
        convert_game(no_result, min_moves=0)
    sample = convert_game(no_result, min_moves=0, allow_no_result=True)
    assert (sample["tgt"]["value"] == 0).all()

    with pytest.raises(SkipGame, match="too_short"):
        convert_game(to_csa(DRAW_MOVES), min_moves=30)

    # parseable but illegal: rook 2h→2c jumps over the 2g pawn
    illegal = to_csa(DRAW_MOVES[:2]).replace("%TORYO", "+2823HI\n%TORYO")
    with pytest.raises(SkipGame, match="illegal_move"):
        convert_game(illegal, min_moves=0)


def test_self_check_passes_on_real_game():
    convert_game(to_csa(MAIN_MOVES), min_moves=0, self_check=True)


def test_encode_frame_check_flag():
    # white to move, white king on 9a checked by the black bishop on 7c
    board = shogi.Board("k8/9/2B6/9/9/9/9/9/8K w - 1")
    assert board.is_check()
    _, _, glb = encode_frame(board)
    assert glb[2] == 1.0                        # mover in check
    assert glb[1] == 0.0                        # mover is white
    board_start = shogi.Board()
    _, _, glb0 = encode_frame(board_start)
    assert glb0[2] == 0.0 and glb0[1] == 1.0


# ----------------------------------------------------------------------
# 8. e2e: CLI → shards → dataset → collate → network
# ----------------------------------------------------------------------

def test_e2e_cli_dataset_network(tmp_path):
    csa_dir = tmp_path / "csa"
    csa_dir.mkdir()
    (csa_dir / "g1.csa").write_text(to_csa(MAIN_MOVES))
    (csa_dir / "g2.csa").write_text(to_csa(DRAW_MOVES, end_token="%SENNICHITE"))
    (csa_dir / "g3.csa").write_text(to_csa(gen_moves(12)))
    out_dir = tmp_path / "shards"

    stats = convert_main([
        "--input", str(csa_dir), "--output_dir", str(out_dir),
        "--shard_size", "2", "--min_moves", "0",
    ])
    assert stats["converted"] == 3 and stats["shards"] == 2

    ds = GameShardDataset(str(out_dir), game="shogi")
    assert len(ds) == 3
    batch = collate_game_sequences([ds[i] for i in range(3)])
    obs, tgt = batch["obs"], batch["tgt"]
    frame_mask = batch["meta"]["frame_mask"]
    assert obs["board_seq"].shape == (3, 12, 28, 9, 9)
    assert sorted(batch["meta"]["lengths"]) == [4, 7, 12]

    torch.manual_seed(0)
    net = build_game_network(
        "shogi", d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, dropout=0.0, encoder_channels=16,
    )
    net.train()
    labels = {k: tgt[k] for k in ("dropped", "from_idx", "to_idx", "piece_idx")}
    pred = net(obs, labels=labels, frame_mask=frame_mask)
    losses = net.action_decoder.il_loss(pred, tgt, frame_mask)
    value_loss = ((pred["value"] - tgt["value"]) ** 2 * frame_mask).sum() / frame_mask.sum()
    loss = losses["loss"] + 0.2 * value_loss
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
