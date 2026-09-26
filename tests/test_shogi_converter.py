"""gmdata shogi converter (converter_version 2): legality masks and repetition globals.

Every committed floodgate fixture game is replayed with python-shogi (the rules reference
the Rust engine mirrors): on every frame the set of legal moves, mapped to POV indices, must
equal the converter's masks (legal_mode / legal_from / legal_to_of / legal_promo_of /
legal_drop_piece / legal_drop_to_of, the `_of` rows taken at the frame's stored indices), and
the repetition flags obs.globals_seq[:, 4:7] must equal the sennichite occurrence counts.
"""
from __future__ import annotations

import os
import subprocess
from collections import Counter

import pytest
import shogi  # python-shogi (project dependency): the rules reference
import shogi.CSA
import torch

from data_processing.games.dataset import GameShardDataset, load_gm_samples
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (gmdata: pytest fixture)

MASK_KEYS = ("legal_mode", "legal_from", "legal_to_of", "legal_promo_of",
             "legal_drop_piece", "legal_drop_to_of")
MASK_WIDTH = {"legal_mode": 2, "legal_from": 81, "legal_to_of": 81, "legal_promo_of": 2,
              "legal_drop_piece": 14, "legal_drop_to_of": 81}


@pytest.fixture(scope="module")
def converted(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("shogi") / "rs"
    subprocess.run([gmdata, "shogi", "convert", "--input", os.path.join(FIX, "shogi"),
                    "--allow-no-result", "--out", str(out)], check=True, capture_output=True)
    samples = load_gm_samples(str(out))
    assert len(samples) > 0
    return str(out), samples


def test_shard_meta_and_schema(converted):
    out, samples = converted
    ds = GameShardDataset(out)
    assert ds.converter_version == 2
    assert ds.shard_meta == {"global_features": 7}
    for s in samples:
        T = s["tgt"]["dropped"].shape[0]
        assert s["obs"]["globals_seq"].shape == (T, 7)
        for k in MASK_KEYS:
            m = s["tgt"][k]
            assert m.shape == (T, MASK_WIDTH[k]) and m.dtype == torch.float32
            assert ((m == 0) | (m == 1)).all()
        assert (s["tgt"]["legal_drop_piece"][:, 7:] == 0).all()   # opponent's hand: never


def test_ground_truth_inside_masks(converted):
    _, samples = converted
    for i, s in enumerate(samples):
        t = s["tgt"]
        ar = torch.arange(t["dropped"].shape[0])
        drop = t["dropped"] == 1
        mode = t["legal_mode"].bool()
        assert mode[ar, t["dropped"]].all(), i
        move_ok = (t["legal_from"].bool()[ar, t["from_idx"]]
                   & t["legal_to_of"].bool()[ar, t["to_idx"]]
                   & t["legal_promo_of"].bool()[ar, t["promoted"].long()])
        drop_ok = (t["legal_drop_piece"].bool()[ar, t["piece_idx"]]
                   & t["legal_drop_to_of"].bool()[ar, t["drop_to_idx"]])
        assert torch.where(drop, drop_ok, move_ok).all(), i
        assert mode.any(-1).all(), i                       # the mover always has a move


def test_repetition_flags_monotone(converted):
    _, samples = converted
    for s in samples:
        rep = s["obs"]["globals_seq"][:, 4:7]
        assert ((rep == 0) | (rep == 1)).all()
        assert (rep[:, 1] <= rep[:, 0]).all() and (rep[:, 2] <= rep[:, 1]).all()
        assert (s["obs"]["globals_seq"][:, 3] == 1).all()


def _reference_masks(board, mover_black: bool, from_idx: int, to_idx: int, piece_idx: int):
    pv = (lambda sq: sq) if mover_black else (lambda sq: 80 - sq)
    m = {k: torch.zeros(w, dtype=torch.bool) for k, w in MASK_WIDTH.items()}
    for mv in board.generate_legal_moves():
        to = pv(mv.to_square)
        if mv.from_square is None:
            slot = mv.drop_piece_type - 1
            m["legal_mode"][1] = True
            m["legal_drop_piece"][slot] = True
            if slot == piece_idx:
                m["legal_drop_to_of"][to] = True
        else:
            fr = pv(mv.from_square)
            m["legal_mode"][0] = True
            m["legal_from"][fr] = True
            if fr == from_idx:
                m["legal_to_of"][to] = True
                if to == to_idx:
                    m["legal_promo_of"][int(bool(mv.promotion))] = True
    return m


def test_masks_and_repetitions_match_python_shogi(converted):
    _, samples = converted
    seen = Counter()
    for s in samples:
        game = shogi.CSA.Parser.parse_file(os.path.join(FIX, "shogi", s["meta"]["source"]))[0]
        board = shogi.Board(game["sfen"])
        t, glb = s["tgt"], s["obs"]["globals_seq"]
        assert len(game["moves"]) == t["dropped"].shape[0]
        occurrences = Counter()
        for k, usi in enumerate(game["moves"]):
            mover_black = board.turn == shogi.BLACK
            pv = (lambda sq: sq) if mover_black else (lambda sq: 80 - sq)
            # stored action = the record's move, in POV indices
            mv = shogi.Move.from_usi(usi)
            if mv.from_square is None:
                assert int(t["dropped"][k]) == 1
                assert int(t["piece_idx"][k]) == mv.drop_piece_type - 1
                assert int(t["drop_to_idx"][k]) == pv(mv.to_square)
            else:
                assert int(t["dropped"][k]) == 0
                assert int(t["from_idx"][k]) == pv(mv.from_square)
                assert int(t["to_idx"][k]) == pv(mv.to_square)
                assert bool(t["promoted"][k]) == bool(mv.promotion)
            # legality masks == python-shogi's legal move set
            ref = _reference_masks(board, mover_black, int(t["from_idx"][k]),
                                   int(t["to_idx"][k]), int(t["piece_idx"][k]))
            for key in MASK_KEYS:
                assert torch.equal(t[key][k].bool(), ref[key]), (s["meta"]["source"], k, key)
            # sennichite state: occurrences of this position (pieces, hands, side to move) so far
            pos = " ".join(board.sfen().split()[:3])
            rep = occurrences[pos]
            occurrences[pos] += 1
            want = torch.tensor([rep >= 1, rep >= 2, rep >= 3], dtype=torch.float32)
            assert torch.equal(glb[k, 4:7], want), (s["meta"]["source"], k)
            assert float(glb[k, 1]) == float(mover_black)
            assert float(glb[k, 2]) == float(board.is_check())
            seen["frames"] += 1
            seen["move_and_drop"] += bool(ref["legal_mode"].all())
            seen["optional_promo"] += bool(ref["legal_promo_of"].all())
            seen["repeated"] += rep >= 1
            board.push(mv)
    # the fixtures exercise both modes on one frame, optional promotions and repetitions
    assert seen["frames"] > 1000
    assert seen["move_and_drop"] > 0 and seen["optional_promo"] > 0 and seen["repeated"] > 0, seen


def test_decoder_consumes_converter_masks(converted):
    """The shard contract end to end: a 9×9 shogi net (7 globals) teacher-
    forced on converted games with legal = the tgt.legal_* rows resolves
    every mask (the `_of` rows apply at the labels), finds no ground truth
    outside them, and its il_head_terms add up to il_loss."""
    from data_processing.games.dataset import collate_game_sequences
    from model.games.registry import build_game_network

    _, samples = converted
    torch.manual_seed(0)
    net = build_game_network("shogi", d_model=32, n_heads=4, d_ff=64, n_st_layers=1,
                             core_type="trm", trm_H_cycles=1, trm_L_cycles=1,
                             trm_block_layers=1, halt_max_steps=1, encoder_channels=8,
                             shogi_global_features=7)
    net.eval()
    batch = collate_game_sequences(samples[:3])
    tgt, mask = batch["tgt"], batch["meta"]["frame_mask"]
    legal = {"mode": tgt["legal_mode"], "from": tgt["legal_from"], "to_of": tgt["legal_to_of"],
             "promo_of": tgt["legal_promo_of"], "drop_piece": tgt["legal_drop_piece"],
             "drop_to_of": tgt["legal_drop_to_of"]}
    labels = {k: tgt[k] for k in ("dropped", "from_idx", "to_idx", "piece_idx")}
    with torch.no_grad():
        pred = net(batch["obs"], legal=legal, labels=labels, frame_mask=mask)
    dec = net.action_decoder
    for k in ("mode_mask", "from_mask", "to_mask", "promo_mask", "piece_mask", "drop_to_mask"):
        assert pred[k].dtype == torch.bool
    # teacher forcing: every `_of` row applied, so the echoes are the shard rows
    assert torch.equal(pred["to_mask"], tgt["legal_to_of"].bool())
    assert torch.equal(pred["promo_mask"], tgt["legal_promo_of"].bool())
    assert torch.equal(pred["drop_to_mask"], tgt["legal_drop_to_of"].bool())
    out = dec.il_loss(pred, tgt, mask)
    assert torch.isfinite(out["loss"]) and out["illegal_target_frac"].item() == 0.0
    terms = dec.il_head_terms(pred, tgt, mask)
    total = sum((t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    assert torch.allclose(total, out["loss"], atol=1e-5)
