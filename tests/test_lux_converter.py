"""`gmdata lux convert` v2 (frozen pre-training schema) on the committed fixture.

obs.globals_seq grew 6 → 11 (env parameters: unit move / sap cost, sap range,
sensor range, env_cfg_known), tgt.sap_offset [T, 16] holds the sap target
class (dy+7)·15 + (dx+7) (−1 = none), and the shard meta announces
{"global_features": 11, "sap_range": 7} for the trainers' config defaults.
The fixture's env_cfg carries none of the unit parameters: Lux S3 defaults,
env_cfg_known = 0."""
from __future__ import annotations

import json
import subprocess

import pytest
import torch

from data_processing.games.dataset import GameShardDataset, collate_game_sequences, load_gm_samples
from model.games.lux import LuxConfig, lux_sap_delta
from tests.test_rust_converters_parity import CASES, gmdata  # noqa: F401  (fixture)

SAP_R = 7


@pytest.fixture(scope="module")
def converted(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("lux") / "rs"
    args, _ = CASES["lux"]
    r = subprocess.run([gmdata, *args, "--out", str(out)], check=True, capture_output=True,
                       text=True)
    return str(out), json.loads(r.stdout)


def test_meta_and_stats(converted):
    out, stats = converted
    assert stats["sap_out_of_range"] == 0 and stats["samples"] == 8
    ds = GameShardDataset(out)
    assert ds.converter_version == 2
    assert ds.shard_meta == {"global_features": 11, "sap_range": 7}
    for s in load_gm_samples(out):
        assert s["meta"]["env_cfg"] == {"unit_move_cost": 2.0, "unit_sap_cost": 10.0,
                                        "unit_sap_range": 4.0, "unit_sensor_range": 2.0,
                                        "known": False}


def test_globals_env_params(converted):
    for s in load_gm_samples(converted[0]):
        g = s["obs"]["globals_seq"]
        T = s["tgt"]["unit_actions"].shape[0]
        assert g.shape == (T, 11) and g.dtype == torch.float32
        # defaults / normalisers: move 2/5, sap cost 10/50, sap range 4/8, sensor 2/8; unknown
        want = torch.tensor([0.4, 0.2, 0.5, 0.25, 0.0])
        assert torch.equal(g[:, 6:], want.expand(T, 5))


def test_sap_offsets(converted):
    samples = load_gm_samples(converted[0])
    for s in samples:
        so, ua = s["tgt"]["sap_offset"], s["tgt"]["unit_actions"]
        assert so.shape == ua.shape and so.dtype == torch.int64
        assert torch.equal(so >= 0, ua == 5)            # a target exactly on sap units
        assert ((so >= -1) & (so < (2 * SAP_R + 1) ** 2)).all()
        # slot 2 always plays the raw action [9, 2, -1]: clamped to sap, offset 99
        assert (so[:, 2] == 99).all()
        assert lux_sap_delta(99, SAP_R) == (2, -1)
    # a raw [5, 0, 0] (sap on the own cell) appears once per (agent, match 1)
    allv = torch.cat([s["tgt"]["sap_offset"].flatten() for s in samples])
    assert set(allv.unique().tolist()) == {-1, 99, 112}
    assert lux_sap_delta(112, SAP_R) == (0, 0)


@pytest.mark.parametrize("head", ["mlp", "pointer"])
def test_v2_shards_train_the_sap_head(converted, head):
    """A net configured from the shard meta (lux_global_features 11,
    lux_sap_range 7) consumes the v2 batch and trains the sap head (both
    heads; the pointer reads the unit cells from the real units_seq)."""
    from model.game_network import GameNetwork
    from model.games.lux import LuxDecoder, LuxEncoder

    samples = load_gm_samples(converted[0])
    batch = collate_game_sequences([{"obs": s["obs"], "tgt": s["tgt"], "meta": s["meta"]}
                                    for s in samples[:3]])
    meta = GameShardDataset(converted[0]).shard_meta
    cfg = LuxConfig(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, core_type="trm",
                    trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1, halt_max_steps=1,
                    encoder_channels=8, lux_global_features=meta["global_features"],
                    lux_sap_range=meta["sap_range"], lux_sap_head=head)
    torch.manual_seed(0)
    net = GameNetwork(cfg, LuxEncoder(cfg), LuxDecoder(cfg))
    fm = batch["meta"]["frame_mask"]
    pred = net(batch["obs"], legal={"unit": batch["obs"]["unit_mask"].bool()}, frame_mask=fm)
    out = net.action_decoder.il_loss(pred, batch["tgt"], fm)
    assert torch.isfinite(out["loss"]) and out["sap_loss"] > 0
    out["loss"].backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in net.action_decoder.sap_head.parameters())


def test_unit_cells_align_with_the_map_planes(converted):
    """The converter's unit features (x/W, y/H) and its [y][x] planes agree
    with the model's row-major cell y·W + x (LuxEncoder.unit_cells: the
    unit-cell PE and the pointer head's sap window centre): every present
    own unit stands on a cell with own-unit count > 0 (plane 6), and its sap
    target class o = (dy+R)(2R+1) + (dx+R) gathers the cell (x+dx, y+dy).
    Off-diagonal units make an x/y transposition fail the check."""
    from model.games.lux import LuxEncoder, lux_sap_window

    cfg = LuxConfig(lux_global_features=11, lux_sap_range=SAP_R)
    enc = LuxEncoder(cfg)
    H, W = cfg.map_h, cfg.map_w
    off_diagonal = n_sap = 0
    for s in load_gm_samples(converted[0]):
        u, m = s["obs"]["units_seq"], s["obs"]["unit_mask"].bool()
        cell = enc.unit_cells(u, m)                                    # [T, U]
        own = s["obs"]["map_seq"][:, 6].reshape(len(cell), H * W)
        assert (own.gather(1, cell) > 0)[m].all()
        x, y = cell % W, cell // W
        off_diagonal += int((x != y)[m].sum())
        so = s["tgt"]["sap_offset"]
        sel = m & (so >= 0)
        idx, in_map = lux_sap_window(cell, H, W, SAP_R)
        dx, dy = lux_sap_delta(so.clamp(min=0), SAP_R)
        tx, ty = x + dx, y + dy
        on_map = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
        target = idx.gather(-1, so.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        hit = in_map.gather(-1, so.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        assert torch.equal(hit[sel], on_map[sel])
        assert torch.equal(target[sel & on_map], (ty * W + tx)[sel & on_map])
        n_sap += int(sel.sum())
    assert off_diagonal > 0 and n_sap > 0
