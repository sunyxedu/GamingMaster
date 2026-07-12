"""Tests for the riichi mahjong data pipeline (mjai .mjson → shards).

Covers:
  1. tile-kind index map + dora successor arithmetic
  2. conversion shapes / dtypes; one sample per seat
  3. label correctness: full per-seat action/tile sequences (chi with the
     base-of-run tile, pon, riichi, kan (ankan + daiminkan), tsumo/ron
     hora, eligibility-based pass frames)
  4. state tracking: hand counts after calls, melds, discard ponds,
     riichi flags, dora markers, kyotaku/scores, red-five flag, wall
  5. call-opportunity policy: riichi'd seats emit no pass frames
  6. value: final placement (with the Tenhou seat tie-break) → ±1, ±1/3
  7. skips: hidden tiles, inconsistent hands
  8. e2e: CLI (dir + zip input) → shard → GameShardDataset → collate →
     GameNetwork forward + il_loss + backward
  9. integration: real Tenhou games (mjai fixtures) through the same path

The hand-written fixture is a full 2-kyoku game with a chi, a pon, an
ankan, a daiminkan, a riichi, a tsumo win and a ron; tile counts are
deck-consistent (≤4 per kind).
"""

import json
import os
import zipfile

import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
)
from data_processing.games.mahjong.convert import (
    SkipGame,
    TILE_KIND,
    convert_game,
    dora_from_marker,
    main as convert_main,
    parse_events,
)
from model.game_network import GameNetwork
from model.games.mahjong import (
    ACTION_CHI,
    ACTION_DAHAI,
    ACTION_HORA,
    ACTION_KAN,
    ACTION_PASS,
    ACTION_PON,
    ACTION_RIICHI,
    MahjongConfig,
    MahjongDecoder,
    MahjongEncoder,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "mahjong")

# ----------------------------------------------------------------------
# Hand-written mjai fixture (see module docstring)
# ----------------------------------------------------------------------

K1 = [
    {"type": "start_kyoku", "bakaze": "E", "dora_marker": "1m", "kyoku": 1,
     "honba": 0, "kyotaku": 0, "oya": 0, "scores": [25000] * 4,
     "tehais": [
         ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "E", "E"],
         ["1p", "3p", "4p", "5p", "6p", "7p", "2s", "3s", "4s", "5s", "P", "P", "P"],
         ["1s", "1s", "2s", "2s", "3s", "3s", "6s", "7s", "9p", "4m", "4m", "W", "W"],
         ["9m", "9m", "9p", "9p", "9s", "9s", "E", "S", "W", "N", "P", "F", "C"],
     ]},
    {"type": "tsumo", "actor": 0, "pai": "5mr"},
    {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
    {"type": "tsumo", "actor": 1, "pai": "8s"},
    {"type": "dahai", "actor": 1, "pai": "8s", "tsumogiri": True},
    {"type": "chi", "actor": 2, "target": 1, "pai": "8s", "consumed": ["6s", "7s"]},
    {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": False},
    {"type": "tsumo", "actor": 3, "pai": "9s"},
    {"type": "dahai", "actor": 3, "pai": "F", "tsumogiri": False},
    {"type": "tsumo", "actor": 0, "pai": "1p"},
    {"type": "reach", "actor": 0},
    {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
    {"type": "reach_accepted", "actor": 0},
    {"type": "tsumo", "actor": 1, "pai": "6p"},
    {"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": False},
    {"type": "tsumo", "actor": 2, "pai": "4s"},
    {"type": "dahai", "actor": 2, "pai": "4s", "tsumogiri": True},
    {"type": "tsumo", "actor": 3, "pai": "1m"},
    {"type": "dahai", "actor": 3, "pai": "W", "tsumogiri": False},
    {"type": "pon", "actor": 2, "target": 3, "pai": "W", "consumed": ["W", "W"]},
    {"type": "dahai", "actor": 2, "pai": "1s", "tsumogiri": False},
    {"type": "tsumo", "actor": 0, "pai": "5m"},
    {"type": "hora", "actor": 0, "target": 0,
     "deltas": [7000, -2000, -2000, -2000], "ura_markers": ["1s"]},
    {"type": "end_kyoku"},
]

K2 = [
    {"type": "start_kyoku", "bakaze": "E", "dora_marker": "6p", "kyoku": 1,
     "honba": 1, "kyotaku": 0, "oya": 0,
     "scores": [31000, 23000, 23000, 23000],
     "tehais": [
         ["2m", "2m", "2m", "2m", "3p", "4p", "5p", "6p", "7p", "8p", "3s", "4s", "5s"],
         ["4m", "5m", "6m", "1p", "2p", "3p", "5p", "5p", "7s", "8s", "9s", "W", "W"],
         ["4m", "5m", "6m", "7m", "8m", "9m", "4p", "4p", "1s", "2s", "3s", "N", "N"],
         ["1m", "9m", "1p", "9p", "1s", "E", "E", "E", "S", "W", "N", "F", "C"],
     ]},
    {"type": "tsumo", "actor": 0, "pai": "9s"},
    {"type": "ankan", "actor": 0, "consumed": ["2m", "2m", "2m", "2m"]},
    {"type": "dora", "dora_marker": "1s"},
    {"type": "tsumo", "actor": 0, "pai": "6s"},
    {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False},
    {"type": "tsumo", "actor": 1, "pai": "W"},
    {"type": "dahai", "actor": 1, "pai": "5p", "tsumogiri": False},
    {"type": "tsumo", "actor": 2, "pai": "E"},
    {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": True},
    {"type": "daiminkan", "actor": 3, "target": 2, "pai": "E",
     "consumed": ["E", "E", "E"]},
    {"type": "tsumo", "actor": 3, "pai": "5s"},
    {"type": "dahai", "actor": 3, "pai": "5s", "tsumogiri": True},
    {"type": "dora", "dora_marker": "2p"},
    {"type": "tsumo", "actor": 0, "pai": "1s"},
    {"type": "dahai", "actor": 0, "pai": "1s", "tsumogiri": True},
    {"type": "tsumo", "actor": 1, "pai": "9p"},
    {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
    {"type": "tsumo", "actor": 2, "pai": "5p"},
    {"type": "dahai", "actor": 2, "pai": "5p", "tsumogiri": True},
    {"type": "hora", "actor": 1, "target": 2, "deltas": [0, 8000, -8000, 0],
     "ura_markers": ["3m"]},
    {"type": "end_kyoku"},
]

GAME = (
    [{"type": "start_game", "names": ["A", "B", "C", "D"],
      "kyoku_first": 0, "aka_flag": True}]
    + K1 + K2 + [{"type": "end_game"}]
)


@pytest.fixture(scope="module")
def samples():
    return {s["meta"]["seat"]: s for s in convert_game(GAME, name="fixture")}


# ----------------------------------------------------------------------
# 1. tile-kind index map + dora arithmetic
# ----------------------------------------------------------------------

def test_tile_kind_map():
    assert TILE_KIND["1m"] == 0 and TILE_KIND["9m"] == 8
    assert TILE_KIND["1p"] == 9 and TILE_KIND["9p"] == 17
    assert TILE_KIND["1s"] == 18 and TILE_KIND["9s"] == 26
    assert [TILE_KIND[h] for h in "ESWN"] == [27, 28, 29, 30]
    assert [TILE_KIND[h] for h in "PFC"] == [31, 32, 33]
    # red fives → the plain 5 kind, both notations
    assert TILE_KIND["5mr"] == TILE_KIND["5m"] == 4
    assert TILE_KIND["5pr"] == TILE_KIND["0p"] == 13
    assert TILE_KIND["5sr"] == TILE_KIND["0s"] == 22


def test_dora_from_marker():
    assert dora_from_marker(TILE_KIND["1m"]) == TILE_KIND["2m"]
    assert dora_from_marker(TILE_KIND["9m"]) == TILE_KIND["1m"]   # suit wraps
    assert dora_from_marker(TILE_KIND["9s"]) == TILE_KIND["1s"]
    assert dora_from_marker(TILE_KIND["N"]) == TILE_KIND["E"]     # winds wrap
    assert dora_from_marker(TILE_KIND["C"]) == TILE_KIND["P"]     # dragons wrap


# ----------------------------------------------------------------------
# 2. shapes / dtypes
# ----------------------------------------------------------------------

def test_conversion_shapes_and_dtypes(samples):
    assert set(samples) == {0, 1, 2, 3}
    expected_T = {0: 7, 1: 6, 2: 7, 3: 5}
    for seat, s in samples.items():
        T = expected_T[seat]
        assert s["obs"]["tiles_seq"].shape == (T, 34, 16)
        assert s["obs"]["players_seq"].shape == (T, 4, 12)
        assert s["obs"]["table_seq"].shape == (T, 12)
        for k in s["obs"]:
            assert s["obs"][k].dtype == torch.float32
        assert s["tgt"]["action_type"].shape == (T,)
        assert s["tgt"]["action_type"].dtype == torch.long
        assert s["tgt"]["tile_idx"].shape == (T,)
        assert s["tgt"]["tile_idx"].dtype == torch.long
        assert s["tgt"]["value"].shape == (T,)
        assert s["tgt"]["value"].dtype == torch.float32
        assert s["meta"]["kyokus"] == 2


# ----------------------------------------------------------------------
# 3. label correctness (full per-seat sequences)
# ----------------------------------------------------------------------

def test_labels_seat0(samples):
    # dahai E | riichi (discard E) | tsumo hora | ankan 2m |
    # dahai 9s | pass (chi-eligible on P3's 5s) | dahai 1s
    tgt = samples[0]["tgt"]
    assert tgt["action_type"].tolist() == [
        ACTION_DAHAI, ACTION_RIICHI, ACTION_HORA,
        ACTION_KAN, ACTION_DAHAI, ACTION_PASS, ACTION_DAHAI,
    ]
    assert tgt["tile_idx"].tolist() == [27, 27, 0, 1, 26, 0, 18]


def test_labels_seat1(samples):
    # dahai 8s | dahai 1p | pass (chi-eligible on P0's 9s) |
    # dahai 5p | dahai 9p | ron hora
    tgt = samples[1]["tgt"]
    assert tgt["action_type"].tolist() == [
        ACTION_DAHAI, ACTION_DAHAI, ACTION_PASS,
        ACTION_DAHAI, ACTION_DAHAI, ACTION_HORA,
    ]
    assert tgt["tile_idx"].tolist() == [25, 9, 0, 13, 17, 0]


def test_labels_seat2(samples):
    # chi (6s7s8s → base 6s) | dahai 9p | dahai 4s | pon W | dahai 1s |
    # dahai E | dahai 5p
    tgt = samples[2]["tgt"]
    assert tgt["action_type"].tolist() == [
        ACTION_CHI, ACTION_DAHAI, ACTION_DAHAI, ACTION_PON,
        ACTION_DAHAI, ACTION_DAHAI, ACTION_DAHAI,
    ]
    # chi tile = base/lowest kind of the run (6s), NOT the claimed 8s
    assert tgt["tile_idx"].tolist() == [23, 17, 21, 29, 18, 27, 13]


def test_labels_seat3(samples):
    # pass (pon-eligible 9p) | dahai F | dahai W | daiminkan E | dahai 5s
    tgt = samples[3]["tgt"]
    assert tgt["action_type"].tolist() == [
        ACTION_PASS, ACTION_DAHAI, ACTION_DAHAI, ACTION_KAN, ACTION_DAHAI,
    ]
    assert tgt["tile_idx"].tolist() == [0, 32, 29, 27, 22]


# ----------------------------------------------------------------------
# 4. state tracking in the encoded frames
# ----------------------------------------------------------------------

def test_own_turn_frame_and_red_five(samples):
    tiles = samples[0]["obs"]["tiles_seq"]
    table = samples[0]["obs"]["table_seq"]
    # frame 0: just drew 5mr → 5m-kind count 2 (5m + red), red flag set,
    # focus on nothing but the drawn kind... focus == drawn kind 4
    assert tiles[0, 4, 2] == 1.0            # hand count one-hot
    assert tiles[0, 4, 14] == 1.0           # red five in hand
    assert tiles[0, 4, 15] == 1.0           # focus = drawn tile
    assert tiles[0, 1, 11] == 1.0           # marker 1m → dora 2m
    assert table[0, 0] == 1.0               # bakaze E
    assert table[0, 7] == pytest.approx(69 / 70)   # wall after 1 draw
    assert table[0, 9] == 1.0               # own-turn flag
    # frame 1 (riichi decision): drew 1p → 1p count 3; not yet riichi
    assert tiles[1, 9, 3] == 1.0
    assert samples[0]["obs"]["players_seq"][1, 0, 1] == 0.0
    # frame 2 (tsumo hora): riichi active, kyotaku 1, 5m-kind count 3
    players = samples[0]["obs"]["players_seq"]
    assert players[2, 0, 1] == 1.0
    assert table[2, 6] == 0.25              # kyotaku / 4
    assert tiles[2, 4, 3] == 1.0


def test_chi_frame_state(samples):
    s = samples[2]
    tiles, players, table = (s["obs"]["tiles_seq"], s["obs"]["players_seq"],
                             s["obs"]["table_seq"])
    # frame 0 = chi decision: discard already in pond, focus on 8s (25),
    # discarder (P1) is kamicha of seat 2 → relative row 3
    assert table[0, 9] == 0.0               # call frame, not own turn
    assert tiles[0, 23, 1] == 1.0           # one 6s in hand
    assert tiles[0, 25, 15] == 1.0          # focus = claimable 8s
    assert players[0, 3, 10] == 1.0         # focus actor = kamicha
    # frame 1 = post-chi discard: 6s/7s/8s moved hand+pond → own meld
    assert table[1, 9] == 1.0               # own-turn (forced discard)
    assert tiles[1, :, 15].sum() == 0.0     # no drawn tile → no focus
    assert tiles[1, 23, 0] == 1.0           # 6s left the hand
    for k in (23, 24, 25):
        assert tiles[1, k, 5] == 0.25       # 1 tile each in own meld
    assert players[1, 0, 7] == 0.25         # own meld count 1/4


def test_pon_frame_state(samples):
    s = samples[2]
    tiles, players = s["obs"]["tiles_seq"], s["obs"]["players_seq"]
    # frame 4 = post-pon discard: WWW meld (3 tiles of kind 29)
    assert tiles[4, 29, 5] == 0.75
    assert tiles[4, 29, 0] == 1.0           # no W left in hand
    assert players[4, 0, 7] == 0.5          # two melds now


def test_pass_frame_state(samples):
    s = samples[3]
    tiles, players = s["obs"]["tiles_seq"], s["obs"]["players_seq"]
    # frame 0 = pass on P2's 9p: seat 3 holds 9p9p (pon-eligible),
    # focus on 9p, kamicha (P2) discarded it (pond count 1)
    assert tiles[0, 17, 2] == 1.0
    assert tiles[0, 17, 15] == 1.0
    assert tiles[0, 17, 10] == 0.25         # kamicha discard count / 4
    assert players[0, 3, 10] == 1.0


def test_kan_and_dora_tracking(samples):
    s = samples[0]
    tiles, players, table = (s["obs"]["tiles_seq"], s["obs"]["players_seq"],
                             s["obs"]["table_seq"])
    # frame 4 = post-ankan discard decision (kyoku 2): 4×2m in own meld,
    # kan count 1, two dora markers (6p → 7p, 1s → 2s), honba 1
    assert tiles[4, 1, 5] == 1.0            # ankan: 4 tiles / 4
    assert tiles[4, 1, 0] == 1.0            # none left concealed
    assert players[4, 0, 9] == 0.25         # kan count / 4
    assert table[4, 8] == pytest.approx(2 / 5)
    assert tiles[4, 15, 11] == 1.0          # dora 7p live
    assert tiles[4, 19, 11] == 1.0          # dora 2s live
    assert table[4, 5] == 0.125             # honba / 8
    assert table[4, 7] == pytest.approx(68 / 70)   # two draws so far
    # seat 3 frame 3 = daiminkan decision: focus on the claimed E
    t3 = samples[3]["obs"]["tiles_seq"]
    assert t3[3, 27, 15] == 1.0
    assert samples[3]["obs"]["players_seq"][3, 3, 10] == 1.0  # discarder P2


def test_opponent_riichi_visible_and_scores(samples):
    # seat 1 frame 1 (dahai 1p) happens after P0's riichi was accepted:
    # P0 is seat 1's kamicha (relative row 3) — riichi flag + 24000 score
    players = samples[1]["obs"]["players_seq"]
    table = samples[1]["obs"]["table_seq"]
    assert players[1, 3, 1] == 1.0
    assert players[1, 3, 0] == pytest.approx(0.24)
    assert table[1, 6] == 0.25              # kyotaku 1


def test_riichi_seat_emits_no_pass_frames(samples):
    """After riichi, seat 0 is pon-eligible on P1's 1p discard (holds
    three 1p) and chi-eligible on P3's W... but riichi'd seats cannot
    call — no pass frame may appear between the riichi and the hora."""
    types = samples[0]["tgt"]["action_type"].tolist()
    assert types[1] == ACTION_RIICHI
    assert types[2] == ACTION_HORA          # nothing in between


# ----------------------------------------------------------------------
# 6. value by placement (Tenhou seat tie-break)
# ----------------------------------------------------------------------

def test_value_by_placement(samples):
    # final scores [31000, 31000, 15000, 23000]: seats 0 and 1 tie and
    # the earlier seat wins the tie → placements 1, 2, 4, 3
    expected = {0: (1, 1.0), 1: (2, 1 / 3), 2: (4, -1.0), 3: (3, -1 / 3)}
    for seat, (place, val) in expected.items():
        s = samples[seat]
        assert s["meta"]["placement"] == place
        assert torch.allclose(s["tgt"]["value"],
                              torch.full_like(s["tgt"]["value"], val))
    assert samples[0]["meta"]["final_score"] == 31000
    assert samples[2]["meta"]["final_score"] == 15000


# ----------------------------------------------------------------------
# 7. skips
# ----------------------------------------------------------------------

def test_skip_hidden_tiles():
    bad = [dict(K1[0])]
    bad[0] = json.loads(json.dumps(K1[0]))
    bad[0]["tehais"][1][0] = "?"
    with pytest.raises(SkipGame, match="hidden"):
        convert_game(bad)


def test_skip_inconsistent_hand():
    game = json.loads(json.dumps(
        [GAME[0]] + K1[:2] + [{"type": "dahai", "actor": 0, "pai": "9s",
                               "tsumogiri": False}]))
    with pytest.raises(SkipGame, match="not in hand"):
        convert_game(game)


def test_skip_no_kyoku():
    with pytest.raises(SkipGame, match="no kyoku"):
        convert_game([{"type": "start_game", "names": ["a"] * 4}])


# ----------------------------------------------------------------------
# 7b. bookkeeping events between decision and reaction (regressions)
# ----------------------------------------------------------------------

def test_discard_calls_skip_reach_accepted_and_dora():
    """Tenhou orders a called riichi discard as reach → dahai →
    reach_accepted → call; a post-kan discard as dahai → dora → call.
    Both bookkeeping events must be skipped or the call is lost and the
    caller's frame mislabeled as a PASS."""
    from data_processing.games.mahjong.convert import _discard_calls

    # riichi discard gets ponned across a reach_accepted
    ev = [
        {"type": "dahai", "actor": 0, "pai": "3p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "pon", "actor": 1, "target": 0, "pai": "3p",
         "consumed": ["3p", "3p"]},
    ]
    assert _discard_calls(ev, 0, 0) == {1: (ACTION_PON, TILE_KIND["3p"])}

    # ron across a reach_accepted
    ev = [
        {"type": "dahai", "actor": 0, "pai": "3p", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 0},
        {"type": "hora", "actor": 2, "target": 0, "deltas": [0, 0, 0, 0]},
    ]
    assert _discard_calls(ev, 0, 0) == {2: (ACTION_HORA, 0)}

    # post-kan discard called across a dora reveal
    ev = [
        {"type": "dahai", "actor": 3, "pai": "W", "tsumogiri": True},
        {"type": "dora", "dora_marker": "2p"},
        {"type": "pon", "actor": 1, "target": 3, "pai": "W",
         "consumed": ["W", "W"]},
    ]
    assert _discard_calls(ev, 0, 3) == {1: (ACTION_PON, TILE_KIND["W"])}


def test_own_turn_label_skips_kan_dora_reveal():
    """Real mjai logs place the kan-dora reveal between the replacement
    tsumo and the following action (kakan → tsumo → dora → dahai); the
    post-kan decision frame must not be dropped."""
    from data_processing.games.mahjong.convert import _own_turn_label

    ev = [
        {"type": "tsumo", "actor": 0, "pai": "6s"},
        {"type": "dora", "dora_marker": "1s"},
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": False},
    ]
    assert _own_turn_label(ev, 0, 0) == (ACTION_DAHAI, TILE_KIND["9s"])

    # rinshan tsumo-hora across the dora reveal
    ev = [
        {"type": "tsumo", "actor": 0, "pai": "6s"},
        {"type": "dora", "dora_marker": "1s"},
        {"type": "hora", "actor": 0, "target": 0, "deltas": [0, 0, 0, 0]},
    ]
    assert _own_turn_label(ev, 0, 0) == (ACTION_HORA, 0)


# ----------------------------------------------------------------------
# 8. e2e: CLI (dir + zip) → shards → dataset → collate → network
# ----------------------------------------------------------------------

def _tiny_net():
    torch.manual_seed(0)
    cfg = MahjongConfig(
        d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, dropout=0.0,
    )
    return GameNetwork(cfg, MahjongEncoder(cfg), MahjongDecoder(cfg))


def _run_e2e(shard_dir, expected_samples):
    ds = GameShardDataset(str(shard_dir), game="mahjong")
    assert len(ds) == expected_samples
    batch = collate_game_sequences([ds[i] for i in range(len(ds))])
    obs, tgt = batch["obs"], batch["tgt"]
    frame_mask = batch["meta"]["frame_mask"]
    B, T_max = frame_mask.shape
    assert obs["tiles_seq"].shape == (B, T_max, 34, 16)

    net = _tiny_net()
    net.train()
    pred = net(obs, labels={"action_type": tgt["action_type"]},
               frame_mask=frame_mask)
    losses = net.action_decoder.il_loss(pred, tgt, frame_mask)
    value_loss = ((pred["value"] - tgt["value"]) ** 2 * frame_mask).sum() \
        / frame_mask.sum()
    loss = losses["loss"] + 0.2 * value_loss
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    return losses


def test_e2e_cli_dir_dataset_network(tmp_path):
    src = tmp_path / "mjson"
    src.mkdir()
    text = "\n".join(json.dumps(e) for e in GAME) + "\n"
    (src / "g1.mjson").write_text(text)
    out_dir = tmp_path / "shards"

    stats = convert_main([
        "--input", str(src), "--output_dir", str(out_dir),
        "--shard_size", "3", "--min_frames", "1",
    ])
    assert stats["games"] == 1 and stats["samples"] == 4
    assert stats["shards"] == 2                      # 4 samples, size 3
    assert stats["frames"] == 7 + 6 + 7 + 5
    assert stats["action_hist"] == {
        "dahai": 15, "riichi": 1, "chi": 1, "pon": 1, "kan": 2,
        "hora": 2, "ryukyoku": 0, "pass": 3,
    }
    _run_e2e(out_dir, expected_samples=4)


def test_e2e_cli_zip_input(tmp_path):
    zpath = tmp_path / "games.zip"
    text = "\n".join(json.dumps(e) for e in GAME) + "\n"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("g1.mjson", text)
        z.writestr("notes.txt", "ignored")
    out_dir = tmp_path / "shards"
    stats = convert_main([
        "--input", str(zpath), "--output_dir", str(out_dir),
        "--shard_size", "128", "--min_frames", "1",
    ])
    assert stats["games"] == 1 and stats["samples"] == 4
    assert stats["shards"] == 1


# ----------------------------------------------------------------------
# 9. integration: real Tenhou mjai games
# ----------------------------------------------------------------------

def test_real_games_convert_and_train(tmp_path):
    real = sorted(os.listdir(FIXTURE_DIR))
    assert len(real) >= 2, "real mjai fixtures missing"

    out_dir = tmp_path / "shards"
    stats = convert_main([
        "--input", FIXTURE_DIR, "--output_dir", str(out_dir),
        "--shard_size", "128", "--min_frames", "1",
    ])
    assert stats["skipped"] == 0
    assert stats["games"] == len(real)
    assert stats["samples"] == 4 * len(real)
    assert stats["action_hist"]["dahai"] > 0
    assert stats["action_hist"]["hora"] > 0

    losses = _run_e2e(out_dir, expected_samples=4 * len(real))
    for k in ("loss", "type_loss", "tile_loss", "type_acc", "tile_acc"):
        assert torch.isfinite(losses[k])


def test_real_games_state_invariants():
    """Replaying real logs must keep every count in range: hand counts
    0..4 (one-hot rows sum to 1), visible counts ≤ 1, probabilities in
    [0, 1] for every scaled feature."""
    for fname in sorted(os.listdir(FIXTURE_DIR)):
        with open(os.path.join(FIXTURE_DIR, fname)) as fh:
            samples = convert_game(parse_events(fh.read()), name=fname)
        assert len(samples) == 4
        for s in samples:
            tiles = s["obs"]["tiles_seq"]
            assert torch.equal(tiles[:, :, :5].sum(-1),
                               torch.ones(tiles.shape[:2]))
            assert (tiles >= 0).all() and (tiles <= 1).all()
            assert (s["obs"]["players_seq"] >= 0).all()
            assert (s["obs"]["table_seq"] >= 0).all()
            assert (s["obs"]["table_seq"] <= 1).all()
            types = s["tgt"]["action_type"]
            assert ((types >= 0) & (types < 8)).all()
