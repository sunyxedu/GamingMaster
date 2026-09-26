"""generals as a first-class game of the multi-game IL pipeline.

- data: GeneralsGioDataset yields GameShardDataset samples through
  SeqReplayDataset's own decoding (prefix truncation, cached record lengths,
  shard meta for META_ARCH_KEYS); open_game_dataset dispatches gio
  directories vs shards;
- model: the generic GameNetwork call (obs / legal / labels dicts) reaches
  GeneralsNetwork next to its unchanged legacy tensor API;
- objective: batch_loss (GeneralsDecoder.il_loss + value + TRM q_halt) is
  EXACTLY training/train_il.py's ImitationLoss on the same batch — per core
  and decoder, deep-supervision carries included;
- end to end: a joint generals + othello train_multi_game run on one shared
  core, eval_multi_game on its checkpoint, and train_il_game on generals.

The Rust `gmdata generals shard` parity lives in test_rust_generals_parity.py.
"""

import json
import math
import os
import sys

import pytest
import torch

from data_processing.dataset_seq import SeqReplayDataset
from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
    open_game_dataset,
    save_shard,
    training_sampler,
)
from data_processing.games import generals as gdata
from model.games import generals as gmodel
from model.losses import ImitationLoss
from model.network import GeneralsNetwork
from training.deep_supervision import Q_HALT_WEIGHT
from training.train_il_game import (
    apply_meta_arch_defaults,
    batch_loss,
    build_labels,
    build_net_tolerant,
    check_meta_arch,
    check_value_support,
    open_dataset,
    prepare_batch,
    value_losses,
)

sys.path.insert(0, os.path.dirname(__file__))
from generals_gio import write_gio_dir  # noqa: E402
from test_core_moe import _board_samples  # noqa: E402

CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, encoder_channels=16, dropout=0.0)
CORES = {
    "transformer": dict(core_type="transformer", core_layers=1),
    "trm": dict(core_type="trm", trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1,
                halt_max_steps=2),
    "recurrent_depth": dict(rd_prelude_layers=1, rd_recurrent_layers=1, rd_coda_layers=1,
                            rd_mean_iters=2, rd_backprop_iters=1),
}


@pytest.fixture(scope="module")
def gio_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("gio")
    write_gio_dir(str(root), [9, 6, 11, 7, 8])
    return str(root)


def _net(ds, core="transformer", **extra):
    kw = apply_meta_arch_defaults("generals", {**TINY, **CORES[core], **extra}, ds.shard_meta)
    net, dropped = build_net_tolerant("generals", kw)
    assert not dropped
    check_meta_arch("generals", net.cfg, ds.shard_meta)
    return net


def _batch(ds, idx=(0, 1, 2)):
    return collate_game_sequences([ds[i] for i in idx])


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def test_gio_dataset_is_seq_replay_decoding(gio_dir):
    ds = gdata.GeneralsGioDataset(gio_dir)
    seq = SeqReplayDataset(gio_dir)
    assert len(ds) == len(seq) == 5
    for i in range(len(ds)):
        s, item = ds[i], seq[i]
        assert set(s["obs"]) == set(gdata.SAMPLE_OBS_KEYS)
        assert set(s["tgt"]) == set(gdata.SAMPLE_TGT_KEYS)
        assert s["meta"]["file"] == item["meta"]["file"]
        for grp in ("obs", "tgt"):
            for k, v in s[grp].items():
                assert torch.equal(v, item[grp][k][0]), k
                assert v.dtype == item[grp][k].dtype
    assert ds.shard_meta == {"grid_h": 5, "grid_w": 5, "grid_channels": 12,
                              "global_features": 4, "max_actions_per_tick": 4}
    s = ds[0]
    assert s["obs"]["grid_seq"].shape == (9, 12, 5, 5)
    assert s["obs"]["legal_source_mask_seq"].dtype == torch.bool
    # one-hot planes: exactly one state per on-map cell, none off-map
    onehot = s["obs"]["grid_seq"][:, :11].sum(1)
    assert torch.equal(onehot[:, :, :4], torch.ones(9, 5, 4))
    assert onehot[:, :, 4].abs().sum() == 0


def test_recursive_listing_in_path_component_order(tmp_path):
    write_gio_dir(str(tmp_path / "a-b"), [3], prefix="y")
    write_gio_dir(str(tmp_path / "a"), [4, 5], prefix="x")
    (tmp_path / "notes.txt").write_text("not a record")
    ds = gdata.GeneralsGioDataset(str(tmp_path))
    rel = [os.path.relpath(p, tmp_path) for p in ds.paths]
    # components compare as strings: "a" < "a-b" (a plain string sort would
    # put "a-b/..." first because "-" < "/"); matches Rust's PathBuf order
    assert rel == [os.path.join("a", "x000.json"), os.path.join("a", "x001.json"),
                   os.path.join("a-b", "y000.json")]
    assert [len(ds[i]["tgt"]["count"]) for i in range(3)] == [4, 5, 3]


def test_prefix_truncation_and_cached_total_frames(gio_dir, monkeypatch):
    full = gdata.GeneralsGioDataset(gio_dir)
    crop = gdata.GeneralsGioDataset(gio_dir, max_len=7)
    assert full.total_frames() == 9 + 6 + 11 + 7 + 8
    assert crop.total_frames() == 7 + 6 + 7 + 7 + 7
    for i in range(len(full)):
        a, b = full[i], crop[i]
        T = min(7, len(a["tgt"]["count"]))
        for grp in ("obs", "tgt"):
            for k in a[grp]:
                assert torch.equal(b[grp][k], a[grp][k][:T]), k   # the FIRST frames
    seq7 = SeqReplayDataset(gio_dir, max_sequence_length=7)       # train_il.py's truncation
    for i in range(len(crop)):
        assert _same_sample(crop[i], gdata.sample_from_seq_item(seq7[i]))
    cache = os.path.join(gio_dir, gdata.LENGTH_CACHE)
    assert os.path.exists(cache) and not cache.endswith(".json")
    assert len(SeqReplayDataset(gio_dir)) == 5                    # invisible to train_il.py
    assert not [f for f in os.listdir(gio_dir) if f.endswith(".tmp")]

    def boom(path):
        raise AssertionError("length should come from the cache")
    monkeypatch.setattr(gdata, "record_length", boom)
    assert gdata.GeneralsGioDataset(gio_dir, max_len=7).total_frames() == crop.total_frames()


def _same_sample(a, b):
    assert a["meta"]["file"] == b["meta"]["file"]
    for grp in ("obs", "tgt"):
        assert a[grp].keys() == b[grp].keys(), grp
        for k in a[grp]:
            assert a[grp][k].dtype == b[grp][k].dtype, k
            assert torch.equal(a[grp][k], b[grp][k]), k
    return True


@pytest.mark.parametrize("chunk", [None, 5, 64])
def test_record_length_reads_count_or_grid(tmp_path, monkeypatch, chunk):
    if chunk is not None:      # tiny read-back chunks: the key straddles chunk borders
        monkeypatch.setattr(gdata, "_CHUNK", chunk)
    p = write_gio_dir(str(tmp_path), [6])[0]
    assert gdata.record_length(p) == 6
    rec = json.load(open(p))
    q = tmp_path / "indented.json"
    q.write_text(json.dumps(rec, indent=1))
    assert gdata.record_length(str(q)) == 6
    del rec["count"]
    q = tmp_path / "nocount.json"
    q.write_text(json.dumps(rec, indent=1))
    assert gdata.record_length(str(q)) == 6
    (tmp_path / "bad.json").write_text("{truncated")
    assert gdata.record_length(str(tmp_path / "bad.json")) == 0
    (tmp_path / "open.json").write_text('{"grid_seq": [], "count": [1, 0')
    assert gdata.record_length(str(tmp_path / "open.json")) == 0


def test_absent_fields_take_seq_replay_defaults(tmp_path):
    p = write_gio_dir(str(tmp_path), [5])[0]
    rec = json.load(open(p))
    for k in ("globals_seq", "legal_source_mask_seq", "max_actions_per_tick", "count",
              "src_y", "value", "enemy_heatmap", "tick_seq"):
        del rec[k]
    rec["src_x"] = [[1, 2, 3, 4]] * 5            # K defaults to 4
    json.dump(rec, open(p, "w"))
    for ml in (None, 3):
        a = gdata.GeneralsGioDataset(str(tmp_path), max_len=ml)[0]
        b = gdata.sample_from_seq_item(SeqReplayDataset(str(tmp_path),
                                                        max_sequence_length=ml)[0])
        assert _same_sample(a, b)


def test_length_scan_split_over_ranks_and_read_only_cache(tmp_path, monkeypatch):
    """Data-parallel trainers scan a strided share per rank and adopt the
    union; a read-only data directory caches under $GM_CACHE_DIR."""
    root = tmp_path / "gio"
    write_gio_dir(str(root), [4, 7, 5, 9, 6])
    monkeypatch.setenv("GM_CACHE_DIR", str(tmp_path / "usercache"))
    ds = gdata.GeneralsGioDataset(str(root))
    parts = [ds.scan_lengths(r, 2) for r in range(2)]
    assert sorted(parts[0]) == [0, 2, 4] and sorted(parts[1]) == [1, 3]
    fresh = gdata.GeneralsGioDataset(str(root))
    fresh.adopt_lengths({**parts[0], **parts[1]}, write_cache=False)
    assert fresh.lengths() == [4, 7, 5, 9, 6]
    assert not os.path.exists(root / gdata.LENGTH_CACHE)
    with pytest.raises(ValueError, match="missing"):
        gdata.GeneralsGioDataset(str(root)).adopt_lengths(parts[0])
    # read-only data directory: the user cache takes the lengths
    os.chmod(root, 0o555)
    try:
        if os.access(root, os.W_OK):
            pytest.skip("running with permissions that ignore read-only directories")
        assert gdata.GeneralsGioDataset(str(root)).total_frames() == 31
        assert not os.path.exists(root / gdata.LENGTH_CACHE)
        user = gdata._user_cache_path(str(root))
        assert os.path.exists(user) and user.startswith(str(tmp_path / "usercache"))
        assert not [f for f in os.listdir(os.path.dirname(user)) if f.endswith(".tmp")]

        def boom(path):
            raise AssertionError("length should come from the user cache")
        monkeypatch.setattr(gdata, "record_length", boom)
        assert gdata.GeneralsGioDataset(str(root)).total_frames() == 31
    finally:
        os.chmod(root, 0o755)


def test_gio_dataset_advice():
    assert gdata.gio_dataset_advice(10, 4) is None
    assert "gmdata generals shard" in gdata.gio_dataset_advice(gdata.LARGE_GIO_DATASET, 4)
    assert "--num_workers" in gdata.gio_dataset_advice(10, 0)


def test_open_game_dataset_dispatch_and_crop_modes(gio_dir, tmp_path):
    ds = open_dataset("generals", gio_dir, 5)
    assert isinstance(ds, gdata.GeneralsGioDataset) and ds.max_len == 5
    assert isinstance(training_sampler(ds), torch.utils.data.RandomSampler)
    with pytest.raises(ValueError, match="prefix"):
        open_game_dataset(gio_dir, "generals", max_len=5, crop="random")
    # generals samples in a .pt shard: GameShardDataset with the prefix crop
    save_shard(str(tmp_path / "g.pt"), "generals", [ds[i] for i in range(len(ds))])
    sh = open_dataset("generals", str(tmp_path), 4)
    assert isinstance(sh, GameShardDataset) and sh.crop == "prefix"
    full = gdata.GeneralsGioDataset(gio_dir)
    for i in range(len(sh)):
        for k, v in sh[i]["obs"].items():
            assert torch.equal(v, full[i]["obs"][k][:4])
    assert GameShardDataset(str(tmp_path), max_len=4).crop == "random"
    with pytest.raises(ValueError, match="crop"):
        GameShardDataset(str(tmp_path), max_len=4, crop="middle")


# ---------------------------------------------------------------------------
# model interface
# ---------------------------------------------------------------------------

def test_generic_call_reaches_the_legacy_network(gio_dir):
    ds = gdata.GeneralsGioDataset(gio_dir)
    net = _net(ds).eval()
    b = _batch(ds)
    obs, fm = b["obs"], b["meta"]["frame_mask"]
    lab = gmodel.source_labels(b["tgt"])
    legal = gmodel.legal_from_obs(obs)
    with torch.no_grad():
        torch.manual_seed(1)
        a = net(obs["grid_seq"], obs["globals_seq"], legal_mask=obs["legal_source_mask_seq"],
                source_y=lab["source_y"], source_x=lab["source_x"], frame_mask=fm)
        torch.manual_seed(1)
        g = net(obs, legal=legal, labels=lab, frame_mask=fm)
    assert a.keys() == g.keys()
    for k in a:
        assert torch.equal(a[k], g[k]), k
    # single-step inference, both APIs
    with torch.no_grad():
        o1 = {"grid_seq": obs["grid_seq"][:, 0], "globals_seq": obs["globals_seq"][:, 0]}
        torch.manual_seed(2)
        s1 = net.step(o1["grid_seq"], o1["globals_seq"],
                      legal_mask=obs["legal_source_mask_seq"][:, 0])
        torch.manual_seed(2)
        s2 = net.step(o1, legal={"source": obs["legal_source_mask_seq"][:, 0]})
    for k in ("source_logits", "value", "action_source", "action_log_prob"):
        assert torch.equal(s1[k], s2[k]), k
    with pytest.raises(TypeError):
        net(obs, obs["globals_seq"])
    with pytest.raises(TypeError):
        net(obs["grid_seq"], obs["globals_seq"], legal=legal)
    with pytest.raises(TypeError):
        net.step(o1, legal_mask=obs["legal_source_mask_seq"][:, 0])


def test_teacher_forcing_labels_and_legality(gio_dir):
    ds = gdata.GeneralsGioDataset(gio_dir)
    b = _batch(ds)
    tgt, fm = b["tgt"], b["meta"]["frame_mask"]
    move = tgt["count"] > 0
    lab = build_labels("generals", tgt, 1.0, fm)
    assert set(lab) == {"source_y", "source_x"}
    assert torch.equal(lab["source_y"][move], tgt["src_y"][..., 0][move])
    assert (lab["source_y"][~move] == -1).all() and (lab["source_x"][~move] == -1).all()
    torch.manual_seed(0)
    half = build_labels("generals", tgt, 0.5, fm)
    kept = half["source_y"] >= 0
    assert kept.any() and (kept & move).sum() < move.sum() and not (kept & ~move).any()
    legal = gmodel.legal_from_obs(b["obs"])
    assert set(legal) == {"source"} and legal["source"].dtype == torch.bool
    dm = torch.ones(*b["obs"]["grid_seq"].shape[:2], 4, 5, 5, dtype=torch.bool)
    assert "direction" in gmodel.legal_from_obs({**b["obs"], "direction_mask_seq": dm})


def _legacy_loss(net, batch, seed, tf=1.0, **kw):
    """training/train_il.py prepare_rows + forward_backward's loss on
    ``batch`` at teacher-forcing ratio ``tf`` (ground-truth source on the
    kept move frames, sampled elsewhere; no labels at all at tf 0).  At tf 1
    train_il.py still draws its keep mask, which keeps every frame (rand < 1)
    — skipped here so both paths reach the decoder with the same RNG state."""
    obs, tgt, fm = batch["obs"], batch["tgt"], batch["meta"]["frame_mask"]
    torch.manual_seed(seed)
    sy = sx = use_gt = None
    if tf > 0:
        source_y, source_x = tgt["src_y"][:, :, 0], tgt["src_x"][:, :, 0]
        tf_mask = (torch.rand_like(source_y, dtype=torch.float32) < tf if tf < 1
                   else torch.ones_like(source_y, dtype=torch.bool))
        use_gt = tf_mask & (tgt["count"] > 0)
        sy = torch.where(use_gt, source_y, torch.full_like(source_y, -1))
        sx = torch.where(use_gt, source_x, torch.full_like(source_x, -1))
    pred = net(obs["grid_seq"], obs["globals_seq"], legal_mask=obs["legal_source_mask_seq"],
               source_y=sy, source_x=sx, frame_mask=fm, return_carry=True, **kw)
    crit = ImitationLoss(grid_w=net.cfg.grid_w,
                         q_halt_weight=0.5 if net.core_type == "trm" else 0.0)
    active = kw.get("active")
    return crit(pred, tgt, tf_mask=use_gt, frame_mask=active if active is not None else fm), pred


def _generic_loss(net, batch, seed, tf=1.0, **kw):
    torch.manual_seed(seed)         # the teacher-forcing draw, then the decoder's
    mb = prepare_batch("generals", batch, CPU, tf_prob=tf, cfg=net.cfg)
    return batch_loss(net, mb, CPU, 0.2, Q_HALT_WEIGHT, False, **kw)


def _grads(net):
    return {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("core,joint", [("transformer", False), ("trm", False),
                                        ("recurrent_depth", False), ("transformer", True)])
def test_generic_objective_equals_train_il(gio_dir, core, joint):
    ds = gdata.GeneralsGioDataset(gio_dir)
    torch.manual_seed(0)
    net = _net(ds, core, joint_decoder=joint)
    b = _batch(ds, (0, 1, 2, 3))
    legacy, lpred = _legacy_loss(net, b, seed=7)
    legacy["loss"].backward()
    g_legacy = _grads(net)
    net.zero_grad(set_to_none=True)
    loss, stats, pred = _generic_loss(net, b, seed=7)
    loss.backward()
    g_generic = _grads(net)

    assert torch.equal(loss, legacy["loss"])               # bit-identical objective
    for k in ("source_loss", "direction_loss", "split_loss", "pass_loss", "value_loss",
              "src_acc", "dir_acc", "full_acc", "complete_acc"):
        assert stats[k] == float(legacy[k]), k
    assert stats["exact_acc"] == stats["complete_acc"]
    if core == "trm":
        assert stats["q_halt_loss"] == float(legacy["q_halt_loss"])
    else:
        assert "q_halt_loss" not in stats                  # no halting head to train
    assert g_legacy.keys() == g_generic.keys()
    for n in g_legacy:
        assert torch.allclose(g_legacy[n], g_generic[n], rtol=1e-5, atol=1e-7), n

    if core == "trm":
        # a second deep-supervision step: carried latents, halted frames, resets
        fm = b["meta"]["frame_mask"]
        torch.manual_seed(3)
        active = fm & (torch.rand(fm.shape) < 0.6)
        reset = fm & (torch.rand(fm.shape) < 0.3)
        kw = dict(carry=pred["carry"], active=active, reset=reset)
        net.zero_grad(set_to_none=True)
        legacy2, _ = _legacy_loss(net, b, seed=11, **kw)
        loss2, stats2, _ = _generic_loss(net, b, seed=11, **kw)
        assert torch.equal(loss2, legacy2["loss"])
        assert stats2["q_halt_loss"] == float(legacy2["q_halt_loss"])


@pytest.mark.parametrize("joint", [False, True])
@pytest.mark.parametrize("tf", [0.0, 0.5])
def test_generic_objective_equals_train_il_below_full_teacher_forcing(gio_dir, joint, tf):
    """tf < 1 (train_il.py's default is 0): the same teacher-forcing draw,
    the same self-sampled sources — for the joint decoder too, which used to
    pin unlabelled frames to cell 0 under the generic trainers' all-−1 labels."""
    ds = gdata.GeneralsGioDataset(gio_dir)
    torch.manual_seed(0)
    net = _net(ds, "transformer", joint_decoder=joint)
    b = _batch(ds, (0, 1, 2, 3))
    legacy, lpred = _legacy_loss(net, b, seed=7, tf=tf)
    legacy["loss"].backward()
    g_legacy = _grads(net)
    net.zero_grad(set_to_none=True)
    loss, stats, pred = _generic_loss(net, b, seed=7, tf=tf)
    loss.backward()
    assert torch.equal(pred["src_idx"], lpred["src_idx"])
    assert torch.equal(loss, legacy["loss"])
    for k in ("source_loss", "direction_loss", "split_loss", "pass_loss"):
        assert stats[k] == float(legacy[k]), k
    g_generic = _grads(net)
    assert g_legacy.keys() == g_generic.keys()
    for n in g_legacy:
        assert torch.allclose(g_legacy[n], g_generic[n], rtol=1e-5, atol=1e-7), n


def test_tf0_means_no_labels():
    tgt = {"count": torch.tensor([[1, 0, -1]]), "src_y": torch.ones(1, 3, 4, dtype=torch.long),
           "src_x": torch.ones(1, 3, 4, dtype=torch.long)}
    fm = torch.ones(1, 3, dtype=torch.bool)
    state = torch.get_rng_state()
    assert build_labels("generals", tgt, 0.0, fm) is None
    assert build_labels("chess", {"from_idx": torch.zeros(1, 3, dtype=torch.long),
                                  "to_idx": torch.zeros(1, 3, dtype=torch.long)}, 0.0, fm) is None
    assert torch.equal(torch.get_rng_state(), state)          # no teacher-forcing draw
    # state labels are never dropped: pokemon keeps active_idx at tf 0
    poke = build_labels("pokemon", {"action_type": torch.zeros(1, 3, dtype=torch.long),
                                    "active_idx": torch.full((1, 3), 2)}, 0.0, fm)
    assert torch.equal(poke["active_idx"], torch.full((1, 3), 2))
    assert (poke["action_type"] == -1).all()


def test_joint_decoder_samples_unlabelled_frames(gio_dir):
    """−1 labels mean "sample from the policy", as for GeneralsDecoder: the
    joint decoder anchors its factorized views on its own draw there (not
    on cell 0), and keeps the ground truth where labelled."""
    ds = gdata.GeneralsGioDataset(gio_dir)
    torch.manual_seed(0)
    net = _net(ds, joint_decoder=True).eval()
    b = _batch(ds, (0, 1, 2, 3))
    obs, fm = b["obs"], b["meta"]["frame_mask"]
    legal = gmodel.legal_from_obs(obs)
    minus = {"source_y": torch.full(fm.shape, -1), "source_x": torch.full(fm.shape, -1)}
    with torch.no_grad():
        torch.manual_seed(4)
        free = net(obs, legal=legal, frame_mask=fm)
        torch.manual_seed(4)
        unl = net(obs, legal=legal, labels=minus, frame_mask=fm)
        lab = gmodel.source_labels(b["tgt"])
        torch.manual_seed(4)
        mixed = net(obs, legal=legal, labels=lab, frame_mask=fm)
    assert torch.equal(unl["src_idx"], free["src_idx"])
    assert (free["src_idx"][fm] != 0).any()
    for k in ("direction_logits", "split_logits", "source_logits", "pass_logits"):
        assert torch.equal(unl[k], free[k]), k
    move = lab["source_y"] >= 0
    gt = lab["source_y"] * net.cfg.grid_w + lab["source_x"]
    assert torch.equal(mixed["src_idx"][move], gt[move])
    assert torch.equal(mixed["src_idx"][~move], free["src_idx"][~move])


def test_value_loss_ignores_non_finite_masked_frames(gio_dir):
    """A NaN / inf value prediction on a masked (padded / halted) frame must
    not poison the loss — ImitationLoss's nan_to_num-before-mask rule."""
    target = torch.tensor([[0.5, -0.5, 0.0]])
    mask = torch.tensor([[True, True, False]])
    for bad in (float("nan"), float("inf")):
        pred = {"value": torch.tensor([[0.25, -0.25, bad]])}
        v_loss, v_mse = value_losses(None, pred, target, mask)
        assert float(v_loss) == float(v_mse) == pytest.approx(0.0625)
    ds = gdata.GeneralsGioDataset(gio_dir)
    net = _net(ds, hl_gauss_value=True, hl_gauss_bins=16)
    logits = torch.zeros(1, 3, 16)
    logits[0, 2] = float("nan")
    v_loss, v_mse = value_losses(net, {"value": torch.tensor([[0.0, 0.0, float("nan")]]),
                                       "value_logits": logits}, target, mask)
    assert torch.isfinite(v_loss) and torch.isfinite(v_mse)


def test_il_loss_contract(gio_dir):
    ds = gdata.GeneralsGioDataset(gio_dir)
    net = _net(ds)
    b = _batch(ds)
    mb = prepare_batch("generals", b, CPU, 1.0, cfg=net.cfg)
    pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"], frame_mask=mb["frame_mask"])
    out = net.action_decoder.il_loss(pred, mb["tgt"], mb["frame_mask"])
    assert out["loss"].requires_grad and out["loss"].dim() == 0
    assert out["answer_correct"].dtype == torch.bool and out["answer_mask"].dtype == torch.bool
    assert out["answer_correct"].shape == out["answer_mask"].shape == mb["frame_mask"].shape
    # dropped moves (count -1) and padding carry no answer
    assert torch.equal(out["answer_mask"], mb["frame_mask"] & (mb["tgt"]["count"] >= 0))
    assert "value_loss" not in out and "q_halt_loss" not in out   # the trainer's terms
    none = net.action_decoder.il_loss(pred, mb["tgt"], torch.zeros_like(mb["frame_mask"]))
    assert torch.isfinite(none["loss"]) and float(none["loss"].detach()) == 0.0


def test_turn_globals_from_the_frame_index(gio_dir):
    """turn_globals / derive_army_channels stay config choices on the data's
    4 globals and 12 planes (the phase-5+ RL input features)."""
    ds = gdata.GeneralsGioDataset(gio_dir)
    kw = apply_meta_arch_defaults("generals", {**TINY, **CORES["transformer"],
                                               "turn_globals": True,
                                               "derive_army_channels": True}, ds.shard_meta)
    assert kw["global_features"] == 7 and kw["grid_channels"] == 12
    net, _ = build_net_tolerant("generals", kw)
    check_meta_arch("generals", net.cfg, ds.shard_meta)
    b = _batch(ds)
    mb = prepare_batch("generals", b, CPU, 1.0, cfg=net.cfg)
    g = mb["obs"]["globals_seq"]
    assert g.shape[-1] == 7 and torch.equal(g[..., :4], b["obs"]["globals_seq"])
    # training/train_rl.py's IL-batch features (frame t is turn t)
    t = torch.arange(g.shape[1], dtype=torch.float32).view(1, -1, 1).expand(g.shape[0], -1, -1)
    assert torch.equal(g[..., 4:], torch.cat([(t % 50) / 50.0, t % 2, t / 500.0], -1))
    loss, _, _ = batch_loss(net, mb, CPU, 0.2, Q_HALT_WEIGHT, False)
    assert torch.isfinite(loss)
    # without turn_globals the data's 4 globals pass through unchanged
    plain = prepare_batch("generals", b, CPU, 1.0, cfg=_net(ds).cfg)
    assert plain["obs"]["globals_seq"].shape[-1] == 4
    with pytest.raises(ValueError, match="global_features"):
        check_meta_arch("generals", _net(ds).cfg, {**ds.shard_meta, "global_features": 7})
    with pytest.raises(ValueError, match="grid_h"):
        check_meta_arch("generals", _net(ds).cfg, {**ds.shard_meta, "grid_h": 25})


def test_generals_validation_is_deterministic(gio_dir):
    """Pass frames carry no source label: evaluation conditions them on the
    argmax source (greedy_decoding), so metrics ignore the RNG state; the
    flag is restored for training, which keeps sampling."""
    from training.train_il_game import evaluate

    ds = gdata.GeneralsGioDataset(gio_dir)
    net = _net(ds)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, collate_fn=collate_game_sequences)
    runs = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        runs.append(evaluate(net, "generals", loader, CPU, 0.2, Q_HALT_WEIGHT, False))
    assert runs[0] == runs[1]
    assert net.action_decoder.greedy is False and net.training


def test_value_targets_respect_the_hl_gauss_support(gio_dir):
    ds = gdata.GeneralsGioDataset(gio_dir)
    b = _batch(ds, range(len(ds)))
    check_value_support(b["tgt"]["value"], b["meta"]["frame_mask"], 1.5)
    net = _net(ds, hl_gauss_value=True, hl_gauss_bins=16)
    mb = prepare_batch("generals", b, CPU, 1.0, cfg=net.cfg)
    loss, stats, _ = batch_loss(net, mb, CPU, 0.2, Q_HALT_WEIGHT, False)
    assert torch.isfinite(loss) and stats["value_loss"] > 1.0     # CE, not MSE
    pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"], frame_mask=mb["frame_mask"])
    with pytest.raises(ValueError, match="outside the HL-Gauss support"):
        value_losses(net, pred, torch.full_like(mb["tgt"]["value"], 3.0), mb["frame_mask"])


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def test_joint_generals_othello_train_and_eval(gio_dir, tmp_path, capsys):
    from training import eval_multi_game, train_multi_game

    (tmp_path / "othello").mkdir()
    save_shard(str(tmp_path / "othello" / "a.pt"), "othello", _board_samples("othello", 4, 6))
    core = [f"{k}={v}" for k, v in {**TINY, **CORES["recurrent_depth"]}.items()]
    torch.manual_seed(0)
    train_multi_game.main([
        "--data", f"generals={gio_dir}", "--data", f"othello={tmp_path / 'othello'}",
        "--val", f"generals={gio_dir}", "--val", f"othello={tmp_path / 'othello'}",
        "--steps", "3", "--batch_size", "2", "--game_batch", "generals=3",
        "--game_max_len", "generals=8", "--game_val_max_len", "generals=10",
        "--game_tf", "generals=0.5", "--warmup_steps", "1", "--eval_every", "3",
        "--log_every", "1", "--save_dir", str(tmp_path / "ck"), "--device", "cpu",
        "--override", *core])
    out = capsys.readouterr().out
    mix = next(ln for ln in out.splitlines() if ln.startswith("[mixture]"))
    assert "generals: frames=37" in mix                   # 8+6+8+7+8: prefix-capped
    val = next(ln for ln in out.splitlines() if ln.startswith("[val 3] generals:"))
    for k in ("source_loss", "direction_loss", "pass_loss", "src_acc", "exact_acc",
              "value_mse", "loss"):
        assert f" {k}=" in val, k
    assert any(ln.startswith("[val 3] othello:") for ln in out.splitlines())

    ckp = tmp_path / "ck" / "multi_final.pt"
    ck = torch.load(ckp, weights_only=False)
    assert ck["games"] == ["generals", "othello"]
    assert ck["configs"]["generals"]["grid_h"] == 5 and ck["configs"]["generals"]["grid_w"] == 5
    assert ck["core_state"] and not any(k.startswith("obs_encoder") for k in ck["core_state"])

    model = eval_multi_game.load_multi(str(ckp), CPU)
    assert isinstance(model["generals"], GeneralsNetwork)
    assert model["generals"].core is model["othello"].core
    assert model["generals"].st_layers is model["othello"].st_layers
    assert model["generals"].obs_encoder is not model["othello"].obs_encoder
    js = tmp_path / "eval.json"
    eval_multi_game.main(["--ckpt", str(ckp), "--val", f"generals={gio_dir}",
                          "--val", f"othello={tmp_path / 'othello'}", "--depths", "1", "3",
                          "--batch_size", "2", "--max_len", "10", "--device", "cpu",
                          "--json", str(js)])
    res = json.loads(js.read_text())
    assert set(res["generals"]) == {"1", "3"}
    for m in res["generals"].values():
        assert 0.0 <= m["exact_acc"] <= 1.0 and math.isfinite(m["loss"])
        assert math.isfinite(m["value_mse"]) and "source_loss" in m

    # per-game post-training entry point: the generals network (shared core
    # included) as a single-game checkpoint, loadable the way train_il.py
    # --finetune_from / train_rl.py --il_checkpoint load theirs
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts"))
    from export_game_checkpoint import main as export
    from model.config import config_from_dict
    from model.rl_utils import load_model_state

    one = torch.load(export(["--ckpt", str(ckp), "--game", "generals",
                             "--out", str(tmp_path / "generals_brain.pt")]), weights_only=False)
    assert one["game"] == "generals" and "train_model_state" in one
    solo = GeneralsNetwork(config_from_dict(one["config"]))
    load_model_state(solo, one["model_state"], ctx="exported generals")
    solo.eval()
    b = _batch(gdata.GeneralsGioDataset(gio_dir))
    mb = prepare_batch("generals", b, CPU, 1.0, cfg=solo.cfg)
    with torch.no_grad():               # pass frames sample their source: same seed
        torch.manual_seed(5)
        want = model["generals"](mb["obs"], legal=mb["legal"], labels=mb["labels"],
                                 frame_mask=mb["frame_mask"])
        torch.manual_seed(5)
        got = solo(b["obs"]["grid_seq"], b["obs"]["globals_seq"],
                   legal_mask=b["obs"]["legal_source_mask_seq"],
                   source_y=mb["labels"]["source_y"], source_x=mb["labels"]["source_x"],
                   frame_mask=mb["frame_mask"])
    for k in ("source_logits", "direction_logits", "pass_logits", "value"):
        assert torch.equal(want[k], got[k]), k

    # the brain initialises a generals-only adaptation run (frozen core)
    train_multi_game.main([
        "--data", f"generals={gio_dir}", "--steps", "2", "--batch_size", "2",
        "--game_max_len", "generals=6", "--warmup_steps", "1", "--ema_rate", "0",
        "--save_dir", str(tmp_path / "ck2"), "--device", "cpu",
        "--init_core", str(ckp), "--freeze_core"])
    ck2 = torch.load(tmp_path / "ck2" / "multi_final.pt", weights_only=False)
    for k, v in ck["core_state"].items():
        if not k.endswith("load"):
            assert torch.equal(ck2["core_state"][k], v), k


def test_train_il_game_trains_generals(gio_dir, tmp_path, monkeypatch, capsys):
    from training import train_il_game

    monkeypatch.setattr(sys, "argv", [
        "train_il_game", "--game", "generals", "--data_dir", gio_dir, "--val_dir", gio_dir,
        "--epochs", "1", "--batch_size", "2", "--max_len", "8", "--warmup_steps", "1",
        "--log_every", "1", "--save_dir", str(tmp_path / "ck"),
        "--override", *[f"{k}={v}" for k, v in {**TINY, **CORES["transformer"]}.items()]])
    monkeypatch.setattr(train_il_game, "pick_device", lambda: CPU)
    train_il_game.main()
    out = capsys.readouterr().out
    assert "[e0 val]" in out and "exact_acc=" in out
    ck = torch.load(tmp_path / "ck" / "generals_epoch0.pt", weights_only=False)
    assert ck["game"] == "generals" and ck["config"]["grid_h"] == 5
    # the saved weights load into a GeneralsNetwork for train_rl.py / play
    net, _ = build_net_tolerant("generals", ck["config"])
    assert isinstance(net, GeneralsNetwork)
    net.load_state_dict(ck["model_state"])


def test_multi_game_generals_recipe_defaults(gio_dir, tmp_path, capsys, monkeypatch):
    """generals in train_multi_game: teacher forcing 0 by default (train_il.py),
    per-game schedules and loss weights, validation on the training crop,
    and a global --max_len refused for its prefix crop."""
    from training import train_multi_game as tmg

    assert tmg.tf_schedule("0.4:0") == (0.4, 0.0) and tmg.tf_schedule("1") == (1.0, 1.0)
    assert [tmg.tf_at((0.4, 0.0), s, 5) for s in (1, 3, 5)] == pytest.approx([0.4, 0.2, 0.0])
    with pytest.raises(SystemExit):
        tmg.tf_schedule("1.5")
    core = [f"{k}={v}" for k, v in {**TINY, **CORES["transformer"]}.items()]
    base = ["--data", f"generals={gio_dir}", "--steps", "1", "--batch_size", "2",
            "--warmup_steps", "1", "--ema_rate", "0", "--device", "cpu",
            "--save_dir", str(tmp_path / "ck"), "--override", *core]
    with pytest.raises(SystemExit, match="game_max_len generals=N"):
        tmg.main(base + ["--max_len", "64"])

    crops, tfs = [], []
    real_open, real_prep = tmg.open_dataset, tmg.prepare_rows
    monkeypatch.setattr(tmg, "open_dataset",
                        lambda g, d, ml=None: crops.append((g, ml)) or real_open(g, d, ml))
    monkeypatch.setattr(tmg, "prepare_rows",
                        lambda g, b, dev, tf, cfg=None: tfs.append((g, tf))
                        or real_prep(g, b, dev, tf, cfg=cfg))
    (tmp_path / "othello").mkdir()
    save_shard(str(tmp_path / "othello" / "a.pt"), "othello", _board_samples("othello", 4, 6))
    tmg.main(base + ["--data", f"othello={tmp_path / 'othello'}", "--val", f"generals={gio_dir}",
                     "--game_max_len", "generals=6", "--games_per_step", "4",
                     "--game_loss_weight", "generals=0.4", "--eval_every", "1"])
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[objective]"))
    assert "generals: tf=0 λ=0.4" in line and "othello: tf=1" in line
    assert {tf for g, tf in tfs if g == "generals"} == {0.0}
    assert {tf for g, tf in tfs if g == "othello"} <= {1.0}
    assert crops == [("generals", 6), ("othello", None), ("generals", 6)]   # val = train crop

    tfs.clear()
    tmg.main(base + ["--game_max_len", "generals=6", "--steps", "3",
                     "--game_tf", "generals=0.4:0", "--games_per_step", "1"])
    assert [tf for _, tf in tfs] == pytest.approx([0.4, 0.2, 0.0])


def test_rl_trainers_balance_moe_experts(gio_dir):
    """train_rl / train_rl_game update the MoE routing bias after their
    optimizer steps (the load resets); rate 0 only clears the tally."""
    from model.components.core import moe_layers
    from training.train_rl import moe_balance_step

    ds = gdata.GeneralsGioDataset(gio_dir)
    net = _net(ds, moe_experts=4, moe_top_k=1).train()
    b = _batch(ds)
    mb = prepare_batch("generals", b, CPU, 1.0, cfg=net.cfg)
    net(mb["obs"], legal=mb["legal"], labels=mb["labels"], frame_mask=mb["frame_mask"])
    layers = moe_layers(net)
    assert layers and all(m.load.sum() > 0 for m in layers)
    before = [m.balance_bias.clone() for m in layers]
    moe_balance_step(net, 1e-2)
    assert all(m.load.sum() == 0 for m in layers)
    assert any(not torch.equal(m.balance_bias, b0) for m, b0 in zip(layers, before))
    net(mb["obs"], legal=mb["legal"], labels=mb["labels"], frame_mask=mb["frame_mask"])
    frozen = [m.balance_bias.clone() for m in layers]
    moe_balance_step(net, 0.0)
    assert all(m.load.sum() == 0 for m in layers)
    assert all(torch.equal(m.balance_bias, b0) for m, b0 in zip(layers, frozen))


# ---------------------------------------------------------------------------
# data parallel
# ---------------------------------------------------------------------------

def _ddp_worker(rank: int, port: int, gio: str, othello: str, out: str) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      LOCAL_RANK=str(rank), WORLD_SIZE="2")
    torch.set_num_threads(1)
    from torch.utils.data import DataLoader

    from training import train_multi_game as tmg

    evals, shard_eval = [], tmg.evaluate_sharded

    def checking_eval(model, loaders, device, value_weight, bf16):
        res = shard_eval(model, loaders, device, value_weight, bf16)
        full = {g: DataLoader(ld.dataset, batch_size=2, collate_fn=collate_game_sequences)
                for g, ld in loaders.items()}
        evals.append((res, tmg.evaluate(model, full, device, value_weight, bf16)))
        return res

    tmg.evaluate_sharded = checking_eval
    core = [f"{k}={v}" for k, v in {**TINY, **CORES["recurrent_depth"], "moe_experts": 2}.items()]
    res = tmg.main([
        "--data", f"generals={gio}", "--data", f"othello={othello}",
        "--val", f"generals={gio}", "--batch_size", "2", "--games_per_step", "1",
        "--steps", "4", "--eval_every", "2", "--log_every", "2", "--warmup_steps", "1",
        "--game_max_len", "generals=8", "--game_val_max_len", "generals=8",
        "--ema_rate", "0.9", "--seed", "5", "--device", "cpu", "--dist_timeout", "2",
        "--save_dir", os.path.join(out, "ck"), "--override", *core])
    model, ema = res["model"], res["ema"]
    torch.save({"params": {n: p.detach().clone() for n, p in model.named_parameters()},
                "buffers": {n: b.detach().clone() for n, b in model.named_buffers()},
                "ema": {n: v.clone() for n, v in ema.shadow.items()},
                "evals": evals}, os.path.join(out, f"rank{rank}.pt"))


def test_ddp_generals_othello_ranks_stay_identical(gio_dir, tmp_path):
    """2 gloo ranks, one micro-batch each per step (the ranks draw different
    games, so generals' adapter often has a gradient on one rank only):
    parameters, buffers (generals BatchNorm statistics, MoE biases) and EMA
    stay bit-identical, and sharded generals validation equals the
    single-process one."""
    from multiprocessing.connection import wait

    from tests.test_multi_game_ddp import _assert_rank_identical, _free_port

    (tmp_path / "othello").mkdir()
    save_shard(str(tmp_path / "othello" / "a.pt"), "othello", _board_samples("othello", 4, 6))
    ctx = torch.multiprocessing.get_context("spawn")
    port = _free_port()
    procs = [ctx.Process(target=_ddp_worker,
                         args=(r, port, gio_dir, str(tmp_path / "othello"), str(tmp_path)))
             for r in range(2)]
    for p in procs:
        p.start()
    try:
        while any(p.is_alive() for p in procs):
            wait([p.sentinel for p in procs if p.is_alive()], timeout=1.0)
            bad = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
            assert not bad, f"rank failed with exit code {bad[0]}"
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=10)
    r0, r1 = (torch.load(tmp_path / f"rank{r}.pt", weights_only=False) for r in range(2))
    _assert_rank_identical(r0, r1)
    bn = [n for n in r0["buffers"] if n.startswith("nets.generals.obs_encoder")
          and n.endswith("running_mean")]
    assert bn and any(r0["buffers"][n].abs().sum() > 0 for n in bn)
    assert len(r0["evals"]) == 2
    for sharded, full in r0["evals"]:
        assert sharded.keys() == full.keys() == {"generals"}
        for k, v in full["generals"].items():
            assert sharded["generals"][k] == pytest.approx(v, rel=1e-5, abs=1e-6), k
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", weights_only=False)
    assert ck["world_size"] == 2 and ck["games"] == ["generals", "othello"]
