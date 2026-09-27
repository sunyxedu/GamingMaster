"""Tests for the shared shard dataset and the generic multi-game IL trainer."""

import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
    save_shard,
)
from training.train_il_game import (
    batch_loss,
    build_labels,
    parse_overrides,
    prepare_batch,
)

SMALL = dict(
    d_model=48, n_heads=4, d_ff=64, n_st_layers=1,
    core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
    halt_max_steps=2, dropout=0.0, encoder_channels=16,
)


def _othello_samples(n=5, t0=6):
    samples = []
    for i in range(n):
        T = t0 + i
        samples.append({
            "obs": {"board_seq": torch.rand(T, 3, 8, 8)},
            "tgt": {
                "place_idx": torch.randint(0, 64, (T,)),
                "passed": torch.zeros(T, dtype=torch.bool),
                "value": torch.rand(T) * 2 - 1,
            },
            "meta": {"src": f"g{i}"},
        })
    return samples


def test_shard_roundtrip_and_index(tmp_path):
    save_shard(str(tmp_path / "a.pt"), "othello", _othello_samples(3))
    save_shard(str(tmp_path / "b.pt"), "othello", _othello_samples(2))
    ds = GameShardDataset(str(tmp_path), game="othello")
    assert len(ds) == 5
    s = ds[4]
    assert set(s["obs"]) == {"board_seq"}
    assert s["obs"]["board_seq"].shape[1:] == (3, 8, 8)


def test_shard_validation(tmp_path):
    save_shard(str(tmp_path / "a.pt"), "othello", _othello_samples(2))
    with pytest.raises(ValueError, match="shard is for game"):
        GameShardDataset(str(tmp_path), game="chess")
    # mismatched T across keys is rejected at save time
    bad = _othello_samples(1)
    bad[0]["tgt"]["value"] = torch.rand(3)
    with pytest.raises(ValueError, match="expected"):
        save_shard(str(tmp_path / "bad.pt"), "othello", bad)


def test_collate_tail_padding():
    batch = collate_game_sequences(_othello_samples(3, t0=4))  # T = 4, 5, 6
    fm = batch["meta"]["frame_mask"]
    assert batch["obs"]["board_seq"].shape == (3, 6, 3, 8, 8)
    assert fm.shape == (3, 6)
    assert fm[0].tolist() == [True] * 4 + [False] * 2
    # padding is zeros beyond each length
    assert batch["obs"]["board_seq"][0, 4:].abs().sum() == 0


def test_max_len_crop(tmp_path):
    save_shard(str(tmp_path / "a.pt"), "othello", _othello_samples(1, t0=20))
    ds = GameShardDataset(str(tmp_path), max_len=8)
    assert ds[0]["obs"]["board_seq"].shape[0] == 8


def test_build_labels_tf_masking():
    tgt = {"from_idx": torch.arange(12).reshape(2, 6), "to_idx": torch.arange(12).reshape(2, 6)}
    fm = torch.ones(2, 6, dtype=torch.bool)
    full = build_labels("chess", tgt, tf_prob=1.0, frame_mask=fm)
    assert torch.equal(full["from_idx"], tgt["from_idx"])
    torch.manual_seed(0)
    half = build_labels("chess", tgt, tf_prob=0.5, frame_mask=fm)
    dropped = half["from_idx"] == -1
    assert dropped.any() and not dropped.all()
    assert torch.equal(half["from_idx"][~dropped], tgt["from_idx"][~dropped])
    assert torch.equal(half["to_idx"] == -1, dropped)        # one draw per frame
    # tf 0 = no labels at all (train_il.py's convention; decoders treat None
    # exactly like all-−1 labels: sample from their own policy)
    assert build_labels("chess", tgt, tf_prob=0.0, frame_mask=fm) is None
    assert build_labels("othello", {"place_idx": tgt["from_idx"]}, 0.5, fm) is None


def test_parse_overrides():
    out = parse_overrides(["d_model=64", "temporal_mixer=recurrent", "dropout=0.0"])
    assert out == {"d_model": 64, "temporal_mixer": "recurrent", "dropout": 0.0}


def test_run_batch_trains_othello(tmp_path):
    """One real optimizer step through the generic trainer path."""
    from model.games import build_game_network

    torch.manual_seed(0)
    net = build_game_network("othello", **SMALL)
    batch = collate_game_sequences(_othello_samples(3))
    mb = prepare_batch("othello", batch, torch.device("cpu"), tf_prob=1.0)
    loss, stats, pred = batch_loss(net, mb, torch.device("cpu"), value_weight=0.2,
                                   q_halt_weight=0.5, bf16=False, act_steps=2)
    assert torch.isfinite(loss)
    assert "place_loss" in stats and "value_loss" in stats
    # q_halt is trained on every game (target: whole action right)
    assert "q_halt_loss" in stats and "exact_acc" in stats
    assert pred["carry"][0].shape[0] == pred["q_halt"].numel()
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_shard_locality_sampler_coverage_and_locality(tmp_path):
    from data_processing.games.dataset import ShardLocalitySampler

    save_shard(str(tmp_path / "a.pt"), "othello", _othello_samples(4))
    save_shard(str(tmp_path / "b.pt"), "othello", _othello_samples(3))
    save_shard(str(tmp_path / "c.pt"), "othello", _othello_samples(5))
    ds = GameShardDataset(str(tmp_path), game="othello")
    sampler = ShardLocalitySampler(ds)
    torch.manual_seed(0)
    order = list(iter(sampler))
    assert sorted(order) == list(range(len(ds)))
    # shard-contiguous: one cache reload per shard per epoch, no more
    shards = [ds._index[i][0] for i in order]
    switches = sum(1 for x, y in zip(shards, shards[1:]) if x != y)
    assert switches == len(ds.paths) - 1
    # reshuffles from the global RNG between epochs
    assert list(iter(sampler)) != order


def test_resume_arch_prefers_checkpoint_config(capsys):
    from training.train_il_game import resume_arch_overrides

    ckpt = {"config": {"d_model": 48, "board_h": 6}, "config_overrides": {}}
    # fields newer than the checkpoint get their LEGACY values
    assert resume_arch_overrides(ckpt, {}) == {
        "d_model": 48, "board_h": 6,
        "core_type": "trm", "trm_arch": "legacy", "halt_exploration_prob": 0.0,
        "qk_norm": False, "memory_vision": False,
        "encoder_norm": "batch"}
    # explicit CLI override is the escape hatch and applies loudly
    arch = resume_arch_overrides(ckpt, {"d_model": 64})
    assert arch == {"d_model": 64, "board_h": 6,
                    "core_type": "trm", "trm_arch": "legacy", "halt_exploration_prob": 0.0,
                    "qk_norm": False, "memory_vision": False,
        "encoder_norm": "batch"}
    assert "replaces checkpoint arch values" in capsys.readouterr().out
    # legacy checkpoint without a full config falls back to overrides
    assert resume_arch_overrides({"config_overrides": {"d_ff": 32}}, {}) == {
        "d_ff": 32, "core_type": "trm", "trm_arch": "legacy", "halt_exploration_prob": 0.0,
        "qk_norm": False, "memory_vision": False,
        "encoder_norm": "batch"}


def test_resume_rebuilds_exact_arch_from_config():
    """A checkpoint with empty config_overrides but a non-default config
    must resume with the saved shapes (the pre-fix code rebuilt from
    overrides + registry defaults and crashed or silently drifted)."""
    from model.games import build_game_network
    from training.train_il_game import build_net_tolerant, resume_arch_overrides

    net = build_game_network("othello", **SMALL)
    ckpt = {
        "game": "othello",
        "model_state": net.state_dict(),
        "config": dict(net.cfg.__dict__),
        "config_overrides": {},
    }
    net2, _ = build_net_tolerant("othello", resume_arch_overrides(ckpt, {}))
    net2.load_state_dict(ckpt["model_state"])  # exact shapes or it raises


def test_build_net_tolerant_drops_unknown_keys():
    from training.train_il_game import build_net_tolerant

    net, dropped = build_net_tolerant(
        "othello", {**SMALL, "some_removed_field": 1},
    )
    assert dropped == ["some_removed_field"]
    assert net.cfg.d_model == SMALL["d_model"]


def test_build_labels_state_keys_never_dropped():
    tgt = {
        "action_type": torch.ones(2, 6, dtype=torch.long),
        "active_idx": torch.full((2, 6), 3, dtype=torch.long),
    }
    fm = torch.ones(2, 6, dtype=torch.bool)
    torch.manual_seed(0)
    labels = build_labels("pokemon", tgt, tf_prob=0.0, frame_mask=fm)
    assert (labels["active_idx"] == 3).all()
    assert (labels["action_type"] == -1).all()


def test_legal_from_obs_wiring():
    from training.train_il_game import LEGAL_FROM_OBS

    hs = LEGAL_FROM_OBS["hearthstone"](
        {"entity_mask": torch.tensor([[[1, 0, 1]]], dtype=torch.uint8)},
    )
    assert hs["actor"].dtype == torch.bool
    assert hs["actor"].tolist() == [[[True, False, True]]]
    sc = LEGAL_FROM_OBS["starcraft"]({"unit_mask": torch.ones(1, 2, 4)})
    assert sc["actor"].dtype == torch.bool
    ids = torch.arange(8).reshape(1, 2, 4)
    assert torch.equal(LEGAL_FROM_OBS["pokemon"]({"move_ids": ids})["move_ids"], ids)
    assert LEGAL_FROM_OBS["pokemon"]({"other": ids}) is None


def test_mixed_converter_versions_rejected(tmp_path):
    save_shard(str(tmp_path / "a.pt"), "othello", _othello_samples(2))
    save_shard(str(tmp_path / "b.pt"), "othello", _othello_samples(2),
               converter_version=2)
    with pytest.raises(ValueError, match="mixes shard generations"):
        GameShardDataset(str(tmp_path), game="othello")


def test_mixed_shard_meta_rejected(tmp_path):
    save_shard(str(tmp_path / "a.pt"), "othello", _othello_samples(2),
               meta={"board_size": 15})
    save_shard(str(tmp_path / "b.pt"), "othello", _othello_samples(2),
               meta={"board_size": 20})
    with pytest.raises(ValueError, match="different settings"):
        GameShardDataset(str(tmp_path), game="othello")


def test_mixed_sample_schema_rejected(tmp_path):
    old = _othello_samples(2)
    new = _othello_samples(2)
    for s in new:
        T = s["obs"]["board_seq"].shape[0]
        s["obs"]["move_ids"] = torch.zeros(T, 4, dtype=torch.long)
    save_shard(str(tmp_path / "a.pt"), "othello", old)
    save_shard(str(tmp_path / "b.pt"), "othello", new)
    with pytest.raises(ValueError, match="mixes converter outputs"):
        GameShardDataset(str(tmp_path), game="othello")


def test_collate_schema_mismatch_rejected():
    a, b = _othello_samples(2)
    del b["tgt"]["value"]
    with pytest.raises(ValueError, match="disagree on obs/tgt keys"):
        collate_game_sequences([a, b])


def test_dataset_exposes_converter_version_and_meta(tmp_path):
    save_shard(str(tmp_path / "a.pt"), "gomoku", _othello_samples(1),
               converter_version=3, meta={"board_size": 9})
    ds = GameShardDataset(str(tmp_path), game="gomoku")
    assert ds.converter_version == 3
    assert ds.shard_meta == {"board_size": 9}


def test_check_meta_arch():
    from types import SimpleNamespace

    from training.train_il_game import check_meta_arch

    cfg = SimpleNamespace(board_h=15, board_w=15)
    check_meta_arch("gomoku", cfg, {"board_size": 15})
    check_meta_arch("gomoku", cfg, {})
    with pytest.raises(ValueError, match="board_h=15"):
        check_meta_arch("gomoku", cfg, {"board_size": 20})


def test_clash_royale_max_len_guard(monkeypatch, tmp_path):
    import sys

    from training import train_il_game

    monkeypatch.setattr(sys, "argv", [
        "training/train_il_game.py", "--game", "clash_royale",
        "--data_dir", str(tmp_path), "--max_len", "64",
    ])
    with pytest.raises(SystemExit, match="max_len"):
        train_il_game.main()


def test_run_batch_teacher_forced_chess():
    from model.games import build_game_network

    torch.manual_seed(0)
    net = build_game_network("chess", board_h=6, board_w=6, **SMALL)
    T, N = 5, 36
    samples = [{
        "obs": {
            "board_seq": torch.rand(T, 14, 6, 6),
            "globals_seq": torch.rand(T, 8),
        },
        "tgt": {
            "from_idx": torch.randint(0, N, (T,)),
            "to_idx": torch.randint(0, N, (T,)),
            "promotion": torch.zeros(T, dtype=torch.long),
            "value": torch.zeros(T),
        },
        "meta": {},
    } for _ in range(2)]
    batch = collate_game_sequences(samples)
    mb = prepare_batch("chess", batch, torch.device("cpu"), tf_prob=1.0)
    loss, _, _ = batch_loss(net, mb, torch.device("cpu"), value_weight=0.2,
                            q_halt_weight=0.5, bf16=False, act_steps=1)
    assert torch.isfinite(loss)


def test_logit_z_loss_masks_and_bounds():
    from training.train_il_game import logit_z_loss

    B, T, N = 2, 3, 5
    logits = torch.zeros(B, T, N)
    logits[0, 0] = 100.0                                   # log Z = 100 + log 5
    logits[1, 2] = float("-inf")                           # fully masked row: excluded
    pred = {"place_logits": logits, "pass_logits": torch.zeros(B, T)}   # rank-2: ignored
    mask = torch.tensor([[True, True, False], [True, True, True]])
    z, lse_max = logit_z_loss(pred, mask)
    lz = torch.logsumexp(logits.float(), -1)
    expect = (lz[0, 0] ** 2 + lz[0, 1] ** 2 + lz[1, 0] ** 2 + lz[1, 1] ** 2) / 4
    assert torch.allclose(z, expect)
    assert abs(lse_max.item() - lz[0, 0].item()) < 1e-4
    # gradient pushes the log-partition down
    x = torch.randn(B, T, N, requires_grad=True)
    logit_z_loss({"a_logits": x * 10}, mask)[0].backward()
    assert torch.isfinite(x.grad).all()
