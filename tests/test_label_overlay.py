"""Engine-teacher overlay shards (`gmdata label`, gmshard OverlayWriter) reach the
Python loader and the IL trainer unchanged.

`gmdata label selftest` writes toy gm-shards with the Rust ShardWriter (chess-shaped:
the same tensors as `gmdata chess convert`) and labels them through the real driver,
with the gmdata binary itself as a fake UCI engine child process. The overlays must
load with GmShard / GameShardDataset, keep every base tensor bit-identical (hard
links), add the documented teacher columns with the documented dtypes, and feed
training/train_il_game.py's soft-target and engine-value losses on a chess network.
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, GmShard, collate_game_sequences
from tests.test_rust_converters_parity import gmdata  # noqa: F401  (fixture: builds gmdata)

K = 16                  # --max-moves default of the toy labeller
HEADS_SPARSE = {"from": 8, "to": 16}
HEADS_DENSE = {"promotion": 5}
TAU = 0.05


@pytest.fixture(scope="module")
def selftest(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("label") / "run"
    r = subprocess.run([gmdata, "label", "selftest", "--out", str(out), "--workers", "2"],
                       check=True, capture_output=True, text=True)
    return out, json.loads(r.stdout)


def _shards(root):
    return sorted(p for p in os.listdir(root) if not p.startswith("."))


def test_summary_and_meta(selftest):
    out, summary = selftest
    st = summary["stats"]
    assert summary["shards_written"] == 2 and st["samples"] == 12
    assert st["verify_failed"] == st["replay_failed"] == st["engine_errors"] == 0
    assert st["labelled"] + sum(st["skipped"].values()) == st["frames"]
    teacher = summary["teacher"]
    assert teacher["engine"] == "FakeUci 1.0" and teacher["policy"] == {"kind": "softmax", "tau": TAU}
    for name in _shards(out / "labelled"):
        idx = json.load(open(out / "labelled" / name / "index.json"))
        base = json.load(open(out / "base" / name / "index.json"))
        assert idx["meta"]["teacher"] == teacher
        assert {k: v for k, v in idx["meta"].items() if k != "teacher"} == base["meta"]
        assert idx["samples"] == base["samples"] and idx["n_frames"] == base["n_frames"]
        assert idx["overlay"]["link"] == "hardlink" and "teacher_stats" in idx
        for t, info in base["tensors"].items():
            assert idx["tensors"][t] == info
            a, b = os.stat(out / "labelled" / name / info["file"]), os.stat(out / "base" / name / info["file"])
            assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), f"{t} not hard-linked"


def test_loader_sees_new_keys_and_unchanged_old_ones(selftest):
    out, _ = selftest
    base = GameShardDataset(str(out / "base"), game="toy")
    lab = GameShardDataset(str(out / "labelled"), game="toy")
    assert len(base) == len(lab) == 12
    assert lab.converter_version == base.converter_version
    assert lab.shard_meta["teacher"]["max_moves"] == K
    new = {
        "teacher_idx": (torch.int64, (K,)), "teacher_q": (torch.float32, (K,)),
        "teacher_raw": (torch.float32, (K,)), "engine_value": (torch.float32, ()),
        "engine_value_mask": (torch.bool, ()), "promotion_probs": (torch.float32, (5,)),
        **{f"{h}_probs_idx": (torch.int64, (k,)) for h, k in HEADS_SPARSE.items()},
        **{f"{h}_probs_val": (torch.float32, (k,)) for h, k in HEADS_SPARSE.items()},
    }
    for i in range(len(lab)):
        b, s = base[i], lab[i]
        assert set(s["obs"]) == set(b["obs"])
        assert set(s["tgt"]) == set(b["tgt"]) | set(new)
        for group in ("obs", "tgt"):
            for key, v in b[group].items():
                assert s[group][key].dtype == v.dtype and torch.equal(s[group][key], v), (i, key)
        T = s["tgt"]["from_idx"].shape[0]
        for key, (dt, shape) in new.items():
            v = s["tgt"][key]
            assert v.dtype == dt and tuple(v.shape) == (T, *shape), (key, v.dtype, v.shape)
        m = s["tgt"]["engine_value_mask"]
        idx, q = s["tgt"]["teacher_idx"], s["tgt"]["teacher_q"]
        assert torch.equal(idx == -1, torch.isnan(q)), "teacher_idx pads == teacher_q NaNs"
        assert not (idx[~m] != -1).any() and (s["tgt"]["engine_value"][~m] == 0).all()
        ev = s["tgt"]["engine_value"][m]
        assert ((ev >= -1) & (ev <= 1)).all()
        assert torch.allclose(ev, 2 * q[m][:, 0] - 1, atol=1e-6), "engine value = 2·Q(best) − 1"
        for h in HEADS_SPARSE:
            val = s["tgt"][f"{h}_probs_val"]
            assert torch.isfinite(val).all() and (val >= 0).all()
            assert torch.equal(s["tgt"][f"{h}_probs_idx"] == -1, val == 0)
        # `from` is a marginal: always present on labelled frames; the conditionals may be empty
        assert torch.allclose(s["tgt"]["from_probs_val"][m].sum(-1), torch.ones(int(m.sum())), atol=1e-5)
        assert (s["tgt"]["from_probs_val"][~m] == 0).all()
        for key in ("to_probs_val", "promotion_probs"):
            sums = s["tgt"][key].sum(-1)
            assert (((sums - 1).abs() < 1e-5) | (sums == 0)).all() and (sums[~m] == 0).all()


def test_soft_targets_rederive_from_raw_columns(selftest):
    """from_probs = marginal over from-squares of softmax(Q / τ) over the scored moves."""
    out, _ = selftest
    for name in _shards(out / "labelled"):
        sh = GmShard(str(out / "labelled" / name))
        a = sh.arrays()
        idx, q = np.asarray(a["tgt.teacher_idx"]), np.asarray(a["tgt.teacher_q"])
        fi, fv = np.asarray(a["tgt.from_probs_idx"]), np.asarray(a["tgt.from_probs_val"])
        for f in range(sh.index["n_frames"]):
            ok = idx[f] >= 0
            if not ok.any():
                continue
            e = np.exp((q[f][ok].astype(np.float64) - q[f][ok].max()) / TAU)
            pi = e / e.sum()
            frm = (idx[f][ok] // 5) // 64
            want = {int(s): float(pi[frm == s].sum()) for s in np.unique(frm)}
            got = {int(c): float(p) for c, p in zip(fi[f], fv[f]) if c >= 0}
            assert got.keys() == want.keys() and all(math.isclose(got[c], want[c], abs_tol=1e-5) for c in got)


def test_trainer_consumes_teacher_targets(selftest):
    from model.games import build_game_network
    from training.train_il_game import batch_loss, prepare_batch
    from tests.test_train_il_game import SMALL

    out, _ = selftest
    ds = GameShardDataset(str(out / "labelled"), game="toy")
    torch.manual_seed(0)
    net = build_game_network("chess", board_global_features=10, **SMALL)
    batch = collate_game_sequences([ds[i] for i in range(4)])
    mb = prepare_batch("chess", batch, torch.device("cpu"), tf_prob=1.0, il_legal_mask=True)
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), value_weight=0.2, q_halt_weight=0.5,
                                bf16=False, act_steps=1, soft_target_weight=1.0, engine_value_weight=0.5)
    assert torch.isfinite(loss)
    for h in ("from", "to", "promotion"):
        assert math.isfinite(float(stats[f"{h}_teacher_kl"])), h
        assert float(stats[f"{h}_teacher_frac"]) > 0, h
        # the toy engine only scores legal moves
        assert float(stats[f"{h}_teacher_illegal_mass"]) < 1e-6, h
    assert 0 < float(stats["engine_value_frac"]) <= 1
    loss.backward()
