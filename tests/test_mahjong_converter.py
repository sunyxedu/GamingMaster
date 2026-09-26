"""`gmdata mahjong convert` v3 legality masks and loss weights on the committed
fixture games.

tgt.legal_action_type [T, 8] marks the action types legal at the frame (exact
under Tenhou's rules: the one-han minimum and furiten included) and
tgt.legal_tile_of [T, 34] the legal tile arguments of the ground-truth type
(all true for the tile-less hora / ryukyoku / pass).  Every recorded action
must lie inside its masks (the converter's `mask_mismatch` stat is 0), the
masks must actually constrain, and they must drive the decoder's IL legality
path (tgt.legal_<key> → legal[<key>]).  tgt.il_weight [T] is 0 only on claim
frames whose recorded pass was pre-empted by another seat's call."""
from __future__ import annotations

import json
import subprocess

import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples
from model.games.mahjong import (
    ACTION_CHI,
    ACTION_DAHAI,
    ACTION_HORA,
    ACTION_KAN,
    ACTION_PASS,
    ACTION_PON,
    ACTION_RIICHI,
    ACTION_RYUKYOKU,
    N_ACTION_TYPES,
    N_TILE_KINDS,
    MahjongDecoder,
)
from tests.test_rust_converters_parity import CASES, gmdata  # noqa: F401  (fixture)

USES = torch.tensor(MahjongDecoder.ARG_USES_DEFAULT)


@pytest.fixture(scope="module")
def converted(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("mahjong") / "rs"
    args, _ = CASES["mahjong"]
    r = subprocess.run([gmdata, *args, "--out", str(out)], check=True, capture_output=True,
                       text=True)
    return str(out), json.loads(r.stdout)


def _frames(samples, key):
    return torch.cat([s["tgt"][key] for s in samples])


def test_schema_and_mismatch_stat(converted):
    out, stats = converted
    assert stats["mask_mismatch"] == 0
    assert stats["frames"] > 0
    ds = GameShardDataset(out)
    assert ds.converter_version == 3 and ds.shard_meta == {}
    for s in load_gm_samples(out):
        T = s["tgt"]["action_type"].shape[0]
        assert s["tgt"]["legal_action_type"].shape == (T, N_ACTION_TYPES)
        assert s["tgt"]["legal_tile_of"].shape == (T, N_TILE_KINDS)
        assert s["tgt"]["il_weight"].shape == (T,)
        for k in ("legal_action_type", "legal_tile_of", "il_weight"):   # packed bits → float {0, 1}
            v = s["tgt"][k]
            assert v.dtype == torch.float32 and ((v == 0) | (v == 1)).all()


def test_every_ground_truth_inside_its_masks(converted):
    samples = load_gm_samples(converted[0])
    types = _frames(samples, "legal_action_type").bool()
    tiles = _frames(samples, "legal_tile_of").bool()
    at = _frames(samples, "action_type")
    ti = _frames(samples, "tile_idx")
    n = torch.arange(at.numel())
    assert types[n, at].all()
    uses = USES[at]
    assert tiles[n[uses], ti[uses]].all()
    assert tiles[~uses].all()          # tile-less GT types: unconstrained tile row


def test_masks_constrain(converted):
    samples = load_gm_samples(converted[0])
    types = _frames(samples, "legal_action_type").bool()
    tiles = _frames(samples, "legal_tile_of").bool()
    at = _frames(samples, "action_type")
    own = types[:, ACTION_DAHAI]
    claim = types[:, ACTION_PASS]
    assert not (own & claim).any()     # own-turn and call frames never mix
    assert (own | claim).all()
    assert (types.sum(-1) < N_ACTION_TYPES).all()
    # discards: only held kinds (≤ 14 kinds of a 14-tile hand)
    dahai = at == ACTION_DAHAI
    assert (tiles[dahai].sum(-1) <= 14).all() and tiles[dahai].sum(-1).float().mean() < 14
    # riichi discards keep tenpai: far fewer than the held kinds
    riichi = at == ACTION_RIICHI
    assert riichi.any() and tiles[riichi].sum(-1).float().mean() < 4
    # pon: exactly the called kind; chi: at most 3 run bases
    assert (tiles[at == ACTION_PON].sum(-1) == 1).all()
    assert (tiles[at == ACTION_CHI].sum(-1) <= 3).all()
    # hora is legal on every winning frame and exact (yaku, furiten): experts decline few
    hora_legal = types[:, ACTION_HORA]
    assert hora_legal[at == ACTION_HORA].all()
    assert hora_legal.sum() < 1.1 * (at == ACTION_HORA).sum()
    assert not types[claim, ACTION_RYUKYOKU].any() and not types[claim, ACTION_RIICHI].any()
    # every claim frame offers something besides pass (v3: claim frames exist by legality)
    assert (types[claim][:, [ACTION_CHI, ACTION_PON, ACTION_KAN, ACTION_HORA]].any(-1)).all()


def test_il_weight_marks_only_pre_empted_passes(converted):
    """Weight 0 only on claim frames labelled pass whose chi / pon / kan was pre-empted by a
    higher-priority call; own-turn frames and every other claim keep weight 1."""
    samples = load_gm_samples(converted[0])
    types = _frames(samples, "legal_action_type").bool()
    at = _frames(samples, "action_type")
    w = _frames(samples, "il_weight")
    zero = w == 0
    assert zero.any() and zero.float().mean() < 0.05
    assert (at[zero] == ACTION_PASS).all()
    assert types[zero][:, [ACTION_CHI, ACTION_PON, ACTION_KAN]].any(-1).all()
    assert not zero[types[:, ACTION_DAHAI]].any()


def test_masks_feed_the_decoder_il_path(converted):
    """tgt.legal_* → legal[*] (what --il_legal_mask does): the teacher-forced
    tile mask equals legal_tile_of, the masked IL loss is finite and never
    zeroes a target (every GT is legal)."""
    from model.game_network import GameNetwork
    from model.games.mahjong import MahjongConfig, MahjongEncoder

    samples = load_gm_samples(converted[0])[:4]
    batch = collate_game_sequences([{"obs": s["obs"], "tgt": s["tgt"], "meta": s["meta"]}
                                    for s in samples])
    tgt, fm = batch["tgt"], batch["meta"]["frame_mask"]
    cfg = MahjongConfig(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, core_type="trm",
                        trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1, halt_max_steps=1)
    torch.manual_seed(0)
    net = GameNetwork(cfg, MahjongEncoder(cfg), MahjongDecoder(cfg))
    legal = {k[len("legal_"):]: v.bool() for k, v in tgt.items() if k.startswith("legal_")}
    assert set(legal) == {"action_type", "tile_of"}
    pred = net(batch["obs"], legal=legal, labels={"action_type": tgt["action_type"]},
               frame_mask=fm)
    assert torch.equal(pred["tile_mask"][fm], tgt["legal_tile_of"].bool()[fm])
    out = net.action_decoder.il_loss(pred, tgt, fm)
    assert torch.isfinite(out["loss"])
    assert out["illegal_target_frac"].item() == 0.0
