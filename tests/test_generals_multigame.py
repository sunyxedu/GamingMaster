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
    cache = os.path.join(gio_dir, gdata.LENGTH_CACHE)
    assert os.path.exists(cache) and not cache.endswith(".json")
    assert len(SeqReplayDataset(gio_dir)) == 5                    # invisible to train_il.py

    def boom(path):
        raise AssertionError("length should come from the cache")
    monkeypatch.setattr(gdata, "record_length", boom)
    assert gdata.GeneralsGioDataset(gio_dir, max_len=7).total_frames() == crop.total_frames()


def test_record_length_reads_count_or_grid(tmp_path):
    p = write_gio_dir(str(tmp_path), [6])[0]
    assert gdata.record_length(p) == 6
    rec = json.load(open(p))
    del rec["count"]
    q = tmp_path / "nocount.json"
    q.write_text(json.dumps(rec, indent=1))
    assert gdata.record_length(str(q)) == 6
    (tmp_path / "bad.json").write_text("{truncated")
    assert gdata.record_length(str(tmp_path / "bad.json")) == 0


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


def _legacy_loss(net, batch, seed, **kw):
    """training/train_il.py forward_backward's loss on ``batch`` (tf_ratio 1:
    ground-truth source on move frames, sampled elsewhere)."""
    obs, tgt, fm = batch["obs"], batch["tgt"], batch["meta"]["frame_mask"]
    use_gt = tgt["count"] > 0
    sy = torch.where(use_gt, tgt["src_y"][:, :, 0], torch.full_like(tgt["src_y"][:, :, 0], -1))
    sx = torch.where(use_gt, tgt["src_x"][:, :, 0], torch.full_like(tgt["src_x"][:, :, 0], -1))
    torch.manual_seed(seed)
    pred = net(obs["grid_seq"], obs["globals_seq"], legal_mask=obs["legal_source_mask_seq"],
               source_y=sy, source_x=sx, frame_mask=fm, return_carry=True, **kw)
    crit = ImitationLoss(grid_w=net.cfg.grid_w,
                         q_halt_weight=0.5 if net.core_type == "trm" else 0.0)
    active = kw.get("active")
    return crit(pred, tgt, tf_mask=use_gt, frame_mask=active if active is not None else fm), pred


def _generic_loss(net, batch, seed, **kw):
    mb = prepare_batch("generals", batch, CPU, tf_prob=1.0, cfg=net.cfg)
    torch.manual_seed(seed)
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
