"""Go pipeline tests: SGF parsing → rules replay → shards → network.

Covers the converter spec end to end:
  1. SGF parser: properties, moves, passes, variations, collections
  2. rules: captures (single + multi-stone), suicide, simple ko
  3. converter shapes/dtypes, POV channel swap, empty-plane identity
  4. value perspective and pass frames
  5. illegal-record handling (corrupt SGF skipped, no crash)
  6. e2e: CLI → shard → GameShardDataset → collate → go network IL step
"""

import os

import numpy as np
import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
)
from data_processing.games.go import convert as goconvert
from data_processing.games.go.rules import (
    BLACK,
    EMPTY,
    WHITE,
    GoBoard,
    IllegalMove,
)
from data_processing.games.go.sgf import parse_sgf

# ---------------------------------------------------------------------------
# Inline SGF fixtures (SGF coords: [xy] = column, row; 'a' = 0)
# ---------------------------------------------------------------------------

# 9x9: move 6 (B de) captures the white stone at dd=(3,3); two passes end it.
CAPTURE_SGF = (
    "(;GM[1]FF[4]SZ[9]KM[5.5]RE[B+2.5]PB[Alice]PW[Bob]BR[3d]WR[2d]"
    "C[escaped bracket \\] and paren ) inside a comment]"
    ";B[dc];W[dd];B[cd];W[gg];B[ed];W[gf];B[de];W[];B[])"
)

# 9x9: move 4 (B ac) captures the 2-stone white group {aa=(0,0), ab=(1,0)}.
MULTI_CAPTURE_SGF = (
    "(;GM[1]FF[4]SZ[9]RE[B+R];B[ba];W[aa];B[bb];W[ab];B[ac];W[])"
)

# 9x9 ko: B db (move 8) captures W cb in ko shape → (1,2) forbidden;
# W plays a ko threat (ff), B answers (gf), then W legally recaptures cb.
KO_SGF = (
    "(;GM[1]FF[4]SZ[9]KM[6.5]RE[W+R]"
    ";B[ca];W[da];B[bb];W[eb];B[cc];W[dc];B[gg];W[cb];B[db]"
    ";W[ff];B[gf];W[cb])"
)

# 9x9 handicap: 2 AB stones, White moves first.
HANDICAP_SGF = (
    "(;GM[1]FF[4]SZ[9]HA[2]KM[0.5]RE[W+R]AB[cc][gg]"
    ";W[ee];B[ce];W[cg];B[eg])"
)

# Corrupt record: White plays onto an occupied point.
CORRUPT_SGF = "(;GM[1]FF[4]SZ[9]RE[B+R];B[ee];W[ee];B[cc])"

# HA declared but only one AB stone recorded: the server implied a second
# placement the record does not carry.
HA_MISMATCH_SGF = (
    "(;GM[1]FF[4]SZ[9]HA[2]KM[0.5]RE[W+R]AB[cc];W[ee];B[ce];W[cg];B[eg])"
)

# Free placement: HA[2] with NO AB — the handicap stones are carried as
# consecutive Black moves before White's first move.
FREE_PLACEMENT_SGF = (
    "(;GM[1]FF[4]SZ[9]HA[2]KM[0.5]RE[W+R];B[cc];B[gg];W[ee];B[ce];W[cg])"
)

# HA[2] with no AB but strictly alternating colors from move 0: the
# placements are genuinely missing from the record.
FREE_PLACEMENT_MISSING_SGF = (
    "(;GM[1]FF[4]SZ[9]HA[2]KM[0.5]RE[W+R];B[cc];W[ee];B[ce];W[cg])"
)

# Teaching-style record: an AB setup stone injected after move 2.
MIDGAME_AB_SGF = (
    "(;GM[1]FF[4]SZ[9]RE[B+R];B[cc];W[gg];B[ee]AB[dd];W[ge];B[ec];W[gc])"
)

# Root-node AE: a pre-move setup node clears one of the two AB stones.
ROOT_AE_SGF = (
    "(;GM[1]FF[4]SZ[9]RE[B+R]AB[cc][gg];AE[cc];B[ee];W[ge];B[ec];W[gc])"
)

# Mid-game AE: the exact corruption record from the audit finding — before
# AE was interpreted this converted "ok" with the removed stone still
# present on every frame after the AE node.
MIDGAME_AE_SGF = (
    "(;GM[1]SZ[9]RE[B+R];B[cc];W[gg];B[ee]AE[cc];W[ge];B[ec];W[gc])"
)

# Tiny 19x19 game with [tt] passes.
TINY19_SGF = (
    "(;GM[1]FF[4]SZ[19]KM[6.5]RE[W+3.5]PB[P1]PW[P2]"
    ";B[pd];W[dd];B[pq];W[dp];B[tt];W[tt])"
)

NO_RESULT_SGF = "(;GM[1]FF[4]SZ[9]RE[?];B[cc];W[gg];B[ce];W[ge])"


def one_game(sgf_text):
    games = parse_sgf(sgf_text)
    assert len(games) == 1
    return games[0]


# ---------------------------------------------------------------------------
# 1. SGF parser
# ---------------------------------------------------------------------------

def test_sgf_parser_properties_moves_passes():
    g = one_game(CAPTURE_SGF)
    assert g.board_size == 9
    assert g.komi == 5.5
    assert g.result == "B+2.5"
    assert g.player_black == "Alice" and g.player_white == "Bob"
    assert g.rank_black == "3d" and g.rank_white == "2d"
    assert g.handicap == 0 and not g.setup_black and not g.setup_white
    assert g.moves == [
        ("B", (2, 3)), ("W", (3, 3)), ("B", (3, 2)), ("W", (6, 6)),
        ("B", (3, 4)), ("W", (5, 6)), ("B", (4, 3)),
        ("W", None), ("B", None),                       # [] passes
    ]


def test_sgf_parser_tt_pass_and_handicap_setup():
    g19 = one_game(TINY19_SGF)
    assert g19.board_size == 19
    assert g19.moves[4] == ("B", None) and g19.moves[5] == ("W", None)  # [tt]

    gh = one_game(HANDICAP_SGF)
    assert gh.handicap == 2
    assert gh.setup_black == [(2, 2), (6, 6)]
    assert gh.moves[0] == ("W", (4, 4))                 # White moves first


def test_sgf_parser_variations_and_collections():
    # Main line only: first variation followed, sibling discarded.
    g = one_game("(;GM[1]SZ[9];B[aa](;W[bb];B[cc])(;W[dd]))")
    assert g.moves == [("B", (0, 0)), ("W", (1, 1)), ("B", (2, 2))]

    games = parse_sgf(CAPTURE_SGF + "\n" + MULTI_CAPTURE_SGF)
    assert len(games) == 2
    assert games[1].result == "B+R"


# ---------------------------------------------------------------------------
# 2. Rules engine
# ---------------------------------------------------------------------------

def test_rules_single_capture():
    b = GoBoard(9)
    b.play(WHITE, (3, 3))
    for p in [(2, 3), (3, 2), (3, 4)]:
        b.play(BLACK, p)
    captured = b.play(BLACK, (4, 3))
    assert captured == [(3, 3)]
    assert b.grid[3, 3] == EMPTY
    assert b.ko_point is None                           # capturer has 4 group libs


def test_rules_multi_stone_capture():
    b = GoBoard(9)
    b.play(WHITE, (0, 0))
    b.play(WHITE, (1, 0))
    b.play(BLACK, (0, 1))
    b.play(BLACK, (1, 1))
    captured = b.play(BLACK, (2, 0))
    assert set(captured) == {(0, 0), (1, 0)}
    assert b.grid[0, 0] == EMPTY and b.grid[1, 0] == EMPTY
    assert (b.grid == BLACK).sum() == 3


def test_rules_suicide_rejected():
    b = GoBoard(9)
    b.play(BLACK, (0, 1))
    b.play(BLACK, (1, 0))
    with pytest.raises(IllegalMove, match="suicide"):
        b.play(WHITE, (0, 0))
    assert b.grid[0, 0] == EMPTY                        # board unchanged


def test_rules_occupied_and_off_board():
    b = GoBoard(9)
    b.play(BLACK, (4, 4))
    with pytest.raises(IllegalMove, match="occupied"):
        b.play(WHITE, (4, 4))
    with pytest.raises(IllegalMove):
        b.play(WHITE, (9, 0))


def test_rules_simple_ko_forbidden_then_reopened():
    b = GoBoard(9)
    # Build the KO_SGF shape by hand (colors as recorded).
    for color, p in [
        (BLACK, (0, 2)), (WHITE, (0, 3)), (BLACK, (1, 1)), (WHITE, (1, 4)),
        (BLACK, (2, 2)), (WHITE, (2, 3)), (WHITE, (1, 2)),
    ]:
        b.play(color, p)
    captured = b.play(BLACK, (1, 3))                    # ko capture
    assert captured == [(1, 2)]
    assert b.ko_point == (1, 2)
    with pytest.raises(IllegalMove, match="ko"):
        b.play(WHITE, (1, 2))                           # immediate recapture
    b.play(WHITE, (5, 5))                               # ko threat elsewhere
    assert b.ko_point is None                           # ko reopened
    b.play(BLACK, (5, 6))
    captured = b.play(WHITE, (1, 2))                    # legal recapture now
    assert captured == [(1, 3)]
    assert b.ko_point == (1, 3)                         # ko flips sides

    b2 = GoBoard(9)
    b2.play(BLACK, (3, 3))
    b2.play(WHITE, (5, 5))
    b2.play(BLACK, None)                                # pass also clears ko
    assert b2.ko_point is None


# ---------------------------------------------------------------------------
# 3. Converter: shapes, POV, planes
# ---------------------------------------------------------------------------

def test_converter_shapes_dtypes_and_pov():
    sample, reason = goconvert.game_to_sample(
        one_game(CAPTURE_SGF), 9, min_moves=1, source="inline",
    )
    assert reason == "ok"
    board = sample["obs"]["board_seq"]
    assert board.shape == (9, 4, 9, 9) and board.dtype == torch.float32
    assert set(sample["obs"]) == {"board_seq"}          # no globals
    tgt = sample["tgt"]
    assert tgt["place_idx"].shape == (9,) and tgt["place_idx"].dtype == torch.long
    assert tgt["passed"].shape == (9,) and tgt["passed"].dtype == torch.bool
    assert tgt["value"].shape == (9,) and tgt["value"].dtype == torch.float32

    f = board.numpy()
    # Frame 0 (B to move, empty board): both stone planes empty.
    assert f[0, 0].sum() == 0 and f[0, 1].sum() == 0 and f[0, 2].sum() == 81
    # Frame 1 (W to move): B's dc=(2,3) is an OPPONENT stone for White.
    assert f[1, 0].sum() == 0
    assert f[1, 1, 2, 3] == 1 and f[1, 1].sum() == 1
    # Frame 2 (B to move): the same stone is OWN — the channel swap IS the POV.
    assert f[2, 0, 2, 3] == 1 and f[2, 0].sum() == 1 and f[2, 1].sum() == 1
    # Empty plane identity on every frame.
    assert np.array_equal(f[:, 2], 1.0 - f[:, 0] - f[:, 1])
    # No geometric flip: stone coordinates identical across both POVs.

    # Frame 7 (after B de captured W dd): (3,3) empty again, White (POV) has
    # exactly 2 stones left, Black 4 around the capture point.
    assert f[7, 2, 3, 3] == 1
    assert f[7, 0].sum() == 2                           # own = white: gg, gf
    assert f[7, 1].sum() == 4                           # opp = black
    for r, c in [(2, 3), (3, 2), (3, 4), (4, 3)]:
        assert f[7, 1, r, c] == 1

    # Labels: place_idx = row*9+col of the recorded move.
    assert tgt["place_idx"][0] == 2 * 9 + 3
    assert tgt["place_idx"][6] == 4 * 9 + 3


def test_converter_capture_updates_board_multi_stone():
    sample, reason = goconvert.game_to_sample(
        one_game(MULTI_CAPTURE_SGF), 9, min_moves=1,
    )
    assert reason == "ok"
    f = sample["obs"]["board_seq"].numpy()
    # Frame 5 (W to move, after B captured {aa, ab}): both points empty,
    # White has no stones anywhere.
    assert f[5, 2, 0, 0] == 1 and f[5, 2, 1, 0] == 1
    assert f[5, 0].sum() == 0 and f[5, 1].sum() == 3


def test_converter_ko_plane():
    sample, reason = goconvert.game_to_sample(
        one_game(KO_SGF), 9, min_moves=1,
    )
    assert reason == "ok"                               # legal ko recapture replays
    f = sample["obs"]["board_seq"].numpy()
    assert f.shape == (12, 4, 9, 9)
    # Frames 0..8: no ko anywhere yet.
    assert f[:9, 3].sum() == 0
    # Frame 9 (W to move right after B's ko capture): (1,2) forbidden, one-hot.
    assert f[9, 3, 1, 2] == 1 and f[9, 3].sum() == 1
    # ...and the captured white stone is gone / B's capturer is present.
    assert f[9, 2, 1, 2] == 1 and f[9, 1, 1, 3] == 1
    # Frames 10, 11: ko cleared by W's threat move.
    assert f[10, 3].sum() == 0 and f[11, 3].sum() == 0


def test_converter_19x19_and_wrong_size_skip():
    g = one_game(TINY19_SGF)
    sample, reason = goconvert.game_to_sample(g, 19, min_moves=1)
    assert reason == "ok"
    assert sample["obs"]["board_seq"].shape == (6, 4, 19, 19)
    assert sample["tgt"]["passed"].tolist() == [False] * 4 + [True, True]
    assert sample["meta"]["players"] == {"black": "P1", "white": "P2"}
    assert sample["meta"]["komi"] == 6.5

    sample, reason = goconvert.game_to_sample(g, 9, min_moves=1)
    assert sample is None and reason == "wrong_board_size"


# ---------------------------------------------------------------------------
# 4. Value perspective + pass frames
# ---------------------------------------------------------------------------

def test_value_perspective_and_pass_frames():
    sample, _ = goconvert.game_to_sample(one_game(CAPTURE_SGF), 9, min_moves=1)
    tgt = sample["tgt"]
    # Black won: +1 on Black's frames (even), −1 on White's (odd).
    assert tgt["value"].tolist() == [1, -1, 1, -1, 1, -1, 1, -1, 1]
    # Pass frames included, flagged, place_idx = 0.
    assert tgt["passed"].tolist() == [False] * 7 + [True, True]
    assert tgt["place_idx"][7] == 0 and tgt["place_idx"][8] == 0

    # White won the ko game: signs flip.
    sample, _ = goconvert.game_to_sample(one_game(KO_SGF), 9, min_moves=1)
    assert sample["tgt"]["value"].tolist() == [-1, 1] * 6


def test_value_no_result_and_handicap_flags():
    g = one_game(NO_RESULT_SGF)
    sample, reason = goconvert.game_to_sample(g, 9, min_moves=1)
    assert sample is None and reason == "no_result"
    sample, reason = goconvert.game_to_sample(
        g, 9, min_moves=1, allow_no_result=True,
    )
    assert reason == "ok"
    assert sample["tgt"]["value"].abs().sum() == 0

    gh = one_game(HANDICAP_SGF)
    sample, reason = goconvert.game_to_sample(gh, 9, min_moves=1)
    assert sample is None and reason == "handicap"
    sample, reason = goconvert.game_to_sample(
        gh, 9, min_moves=1, allow_handicap=True,
    )
    assert reason == "ok"
    f = sample["obs"]["board_seq"].numpy()
    # Frame 0: White to move, the 2 AB stones are opponent stones.
    assert f[0, 0].sum() == 0 and f[0, 1].sum() == 2
    assert f[0, 1, 2, 2] == 1 and f[0, 1, 6, 6] == 1
    # W won → mover values start at +1 for White.
    assert sample["tgt"]["value"].tolist() == [1, -1, 1, -1]


def test_min_moves_skip():
    sample, reason = goconvert.game_to_sample(
        one_game(CAPTURE_SGF), 9, min_moves=30,
    )
    assert sample is None and reason == "too_short"


# ---------------------------------------------------------------------------
# 5. Illegal records
# ---------------------------------------------------------------------------

def test_corrupt_record_skipped():
    sample, reason = goconvert.game_to_sample(
        one_game(CORRUPT_SGF), 9, min_moves=1,
    )
    assert sample is None and reason == "illegal_move"


# ---------------------------------------------------------------------------
# 5b. Setup hygiene: mid-game AB/AW/AE, HA↔AB consistency, free placement
# ---------------------------------------------------------------------------

def test_midgame_ab_rejected():
    g = one_game(MIDGAME_AB_SGF)
    assert g.setup_after_move
    for kwargs in ({}, {"allow_handicap": True}):
        sample, reason = goconvert.game_to_sample(g, 9, min_moves=1, **kwargs)
        assert sample is None and reason == "handicap"


def test_root_handicap_accepted_when_ha_matches_ab():
    sample, reason = goconvert.game_to_sample(
        one_game(HANDICAP_SGF), 9, min_moves=1, allow_handicap=True,
    )
    assert reason == "ok"
    f = sample["obs"]["board_seq"].numpy()
    # Frame 0: White moves first, both AB stones are opponent stones.
    assert f[0, 0].sum() == 0
    assert f[0, 1, 2, 2] == 1 and f[0, 1, 6, 6] == 1 and f[0, 1].sum() == 2
    assert sample["tgt"]["value"][0] == 1               # W won, W to move


def test_ha_ab_count_mismatch_rejected():
    sample, reason = goconvert.game_to_sample(
        one_game(HA_MISMATCH_SGF), 9, min_moves=1, allow_handicap=True,
    )
    assert sample is None and reason == "handicap"


def test_free_placement_handicap_accepted():
    g = one_game(FREE_PLACEMENT_SGF)
    assert g.handicap == 2 and not g.setup_black

    # Without --allow_handicap: still rejected (HA > 0).
    sample, reason = goconvert.game_to_sample(g, 9, min_moves=1)
    assert sample is None and reason == "handicap"

    sample, reason = goconvert.game_to_sample(
        g, 9, min_moves=1, allow_handicap=True,
    )
    assert reason == "ok"
    f = sample["obs"]["board_seq"].numpy()
    # Frame 0: empty board — the placements happen AS moves 0 and 1.
    assert f[0, 0].sum() == 0 and f[0, 1].sum() == 0
    # Frame 2 = position before White's first move: both handicap stones
    # on the board, as opponent stones from White's POV.
    assert f[2, 0].sum() == 0
    assert f[2, 1, 2, 2] == 1 and f[2, 1, 6, 6] == 1 and f[2, 1].sum() == 2
    # W won → mover-POV values follow the B,B,W,B,W move colors.
    assert sample["tgt"]["value"].tolist() == [-1, -1, 1, -1, 1]


def test_free_placement_missing_stones_rejected():
    # HA[2] but alternating colors from move 0: the record carries neither
    # AB setup nor leading Black placement moves.
    sample, reason = goconvert.game_to_sample(
        one_game(FREE_PLACEMENT_MISSING_SGF), 9, min_moves=1,
        allow_handicap=True,
    )
    assert sample is None and reason == "handicap"


def test_root_ae_clears_setup_stone():
    g = one_game(ROOT_AE_SGF)
    assert g.setup_black == [(6, 6)] and not g.setup_after_move
    sample, reason = goconvert.game_to_sample(
        g, 9, min_moves=1, allow_handicap=True,
    )
    assert reason == "ok"
    f = sample["obs"]["board_seq"].numpy()
    # Frame 0 (B to move): the cleared point is empty, the survivor is own.
    assert f[0, 2, 2, 2] == 1
    assert f[0, 0, 6, 6] == 1 and f[0, 0].sum() == 1 and f[0, 1].sum() == 0


def test_midgame_ae_rejected():
    # The audit-finding record must never convert again: AE after a move
    # sets setup_after_move and the game is rejected like mid-game AB.
    g = one_game(MIDGAME_AE_SGF)
    assert g.setup_after_move
    assert not g.setup_black and not g.setup_white      # nothing pre-placed
    for kwargs in ({}, {"allow_handicap": True}):
        sample, reason = goconvert.game_to_sample(g, 9, min_moves=1, **kwargs)
        assert sample is None and reason == "handicap"


# ---------------------------------------------------------------------------
# 6. End-to-end: CLI → shards → dataset → collate → network IL step
# ---------------------------------------------------------------------------

def _write_sgf_dir(root):
    files = {
        "capture.sgf": CAPTURE_SGF,
        "sub/ko.sgf": KO_SGF,                           # recursive discovery
        "sub/multicap.sgf": MULTI_CAPTURE_SGF,
        "corrupt.sgf": CORRUPT_SGF,
        "handicap.sgf": HANDICAP_SGF,
        "tiny19.sgf": TINY19_SGF,
        "notes.txt": "not an sgf",
    }
    for rel, text in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)


def test_cli_end_to_end(tmp_path):
    sgf_dir = tmp_path / "sgf"
    out_dir = tmp_path / "shards"
    _write_sgf_dir(str(sgf_dir))

    stats = goconvert.main([
        "--input", str(sgf_dir), "--output_dir", str(out_dir),
        "--shard_size", "2", "--board_size", "9", "--min_moves", "2",
    ])
    assert stats["converted"] == 3                      # capture, ko, multicap
    assert stats["illegal_move"] == 1                   # corrupt.sgf
    assert stats["handicap"] == 1
    assert stats["wrong_board_size"] == 1               # tiny19.sgf
    shard_files = sorted(os.listdir(out_dir))
    assert shard_files == ["go_00000.pt", "go_00001.pt"]  # 2 + 1 games

    dataset = GameShardDataset(str(out_dir), game="go")
    assert len(dataset) == 3

    batch = collate_game_sequences([dataset[i] for i in range(3)])
    B, T = 3, 12                                        # ko game is longest
    assert batch["obs"]["board_seq"].shape == (B, T, 4, 9, 9)
    assert batch["tgt"]["place_idx"].shape == (B, T)
    assert batch["meta"]["frame_mask"].shape == (B, T)
    assert sorted(batch["meta"]["lengths"]) == [6, 9, 12]

    from model.games import build_game_network

    torch.manual_seed(0)
    net = build_game_network(
        "go", board_h=9, board_w=9,
        d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, dropout=0.0, encoder_channels=16,
        # The go preset is moving to piece_channels=4 / no globals; pass
        # explicitly so this test is preset-independent.
        piece_channels=4, board_global_features=0,
    )
    frame_mask = batch["meta"]["frame_mask"]
    pred = net(batch["obs"], frame_mask=frame_mask)
    assert pred["place_logits"].shape == (B, T, 81)
    assert pred["pass_logits"].shape == (B, T)
    assert pred["value"].shape == (B, T)

    losses = net.action_decoder.il_loss(pred, batch["tgt"], frame_mask)
    mask_f = frame_mask.float()
    v_loss = (
        (pred["value"].float() - batch["tgt"]["value"].float()).pow(2) * mask_f
    ).sum() / mask_f.sum()
    loss = losses["loss"] + 0.2 * v_loss
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in net.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"

    # max_games cap respected on a fresh run.
    out2 = tmp_path / "shards2"
    stats2 = goconvert.main([
        "--input", str(sgf_dir), "--output_dir", str(out2),
        "--shard_size", "2", "--board_size", "9", "--min_moves", "2",
        "--max_games", "1",
    ])
    assert stats2["converted"] == 1
