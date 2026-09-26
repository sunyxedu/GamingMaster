"""`gmdata label othello` overlays reach the Python loader and the IL trainer unchanged.

Real WTHOR games (tests/fixtures/othello, converted by `gmdata othello convert`) are
labelled through the real CLI and driver with the gmdata binary itself as a fake
Egaroucid child process (`gmdata label fake-engine othello`: deterministic disc scores,
exact at <= 20 empties) — and, when Egaroucid is installed
(scripts/engines/fetch_engines.sh egaroucid, or $GM_EGAROUCID), with the real engine at a
fast search setting. The overlays must load with GameShardDataset next to the base
columns, carry the documented teacher columns (PlacementDecoder heads `place` / `pass`,
raw disc scores, engine values), re-derive from the raw columns (π ∝ exp(Q/τ + w·tanh(discs/16))),
and feed
training/train_il_game.py's soft-target and engine-value losses on the othello network.
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
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture: builds gmdata)

K = 32              # --max-moves default of the othello labeller
TAU = 0.05          # --tau default
W = 2.0             # --tie-weight default
DISC_SCALE = 2.3    # --disc-scale default
FAKE_EXACT = 20     # the fake engine reports exact results at <= 20 empties


def _egaroucid():
    p = os.environ.get("GM_EGAROUCID") or os.path.join(
        os.environ.get("GM_ENGINES", os.path.expanduser("~/gm-engines")), "bin", "egaroucid")
    return p if os.path.isfile(p) else None


ENGINES = ["fake"] + (["egaroucid"] if _egaroucid() else [])


@pytest.fixture(scope="module")
def base(gmdata, tmp_path_factory):  # noqa: F811
    out = tmp_path_factory.mktemp("oth") / "base"
    subprocess.run([gmdata, "othello", "convert", "--input", os.path.join(FIX, "othello"), "--out", str(out),
                    "--max-games", "8", "--shard-size", "4"], check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope="module", params=ENGINES)
def labelled(request, gmdata, base):  # noqa: F811
    out = base.parent / f"labelled-{request.param}"
    if request.param == "fake":
        engine = [gmdata, "--engine-arg", "label", "--engine-arg", "fake-engine", "--engine-arg", "othello"]
        limits = []
    else:
        engine = [_egaroucid()]
        limits = ["--limit", "level=5", "--limit", "exact_empties=12"]
    r = subprocess.run([gmdata, "label", "othello", "--input", str(base), "--out", str(out), "--engine", *engine,
                        "--workers", "2", "--progress-s", "0", *limits], check=True, capture_output=True, text=True)
    return request.param, out, json.loads(r.stdout)


def _shards(root):
    return sorted(p for p in os.listdir(root) if not p.startswith("."))


def test_summary_and_meta(base, labelled):
    kind, out, summary = labelled
    st = summary["stats"]
    assert summary["shards_written"] == 2 and st["samples"] == 8
    assert st["verify_failed"] == st["replay_failed"] == st["engine_errors"] == st["bad_analysis"] == 0
    assert st["labelled"] + sum(st["skipped"].values()) == st["frames"]
    assert set(st["skipped"]) <= {"forced"} and st["labelled"] > 0
    t = summary["teacher"]
    assert t["engine"].startswith("Egaroucid for Console ")
    assert t["raw_unit"] == "discs" and t["max_moves"] == K
    assert t["policy"] == {"kind": "softmax", "tau": TAU, "tie_weight": W,
                           "secondary": {"of": "teacher_raw", "discs": "u = tanh(raw / 16)"}}
    assert t["labeller_version"] == 2 and t["core_version"] == 2
    assert t["heads"] == {"place": {"kind": "categorical", "n": 64, "k": 64, "storage": "dense"},
                          "pass": {"kind": "bernoulli"}}
    assert t["game"]["disc_scale"] == DISC_SCALE
    for name in _shards(out):
        idx = json.load(open(out / name / "index.json"))
        b = json.load(open(base / name / "index.json"))
        assert idx["meta"]["teacher"] == t and idx["samples"] == b["samples"]
        for key, info in b["tensors"].items():
            assert idx["tensors"][key] == info


def _empties(board_seq):
    return board_seq[:, 2].reshape(board_seq.shape[0], -1).sum(-1)


def test_loader_columns(base, labelled):
    kind, out, _ = labelled
    b_ds = GameShardDataset(str(base), game="othello")
    ds = GameShardDataset(str(out), game="othello")
    assert len(ds) == len(b_ds) == 8
    new = {"teacher_idx": (torch.int64, (K,)), "teacher_q": (torch.float32, (K,)), "teacher_raw": (torch.float32, (K,)),
           "engine_value": (torch.float32, ()), "engine_value_mask": (torch.bool, ()),
           "place_probs": (torch.float32, (64,)), "pass_probs": (torch.float32, ()), "pass_probs_mask": (torch.bool, ())}
    for i in range(len(ds)):
        s, b = ds[i], b_ds[i]
        assert set(s["tgt"]) == set(b["tgt"]) | set(new)
        for group in ("obs", "tgt"):
            for key, v in b[group].items():
                assert torch.equal(s[group][key], v), (i, key)
        T = s["tgt"]["place_idx"].shape[0]
        for key, (dt, shape) in new.items():
            v = s["tgt"][key]
            assert v.dtype == dt and tuple(v.shape) == (T, *shape), (key, v.dtype, v.shape)
        m, idx = s["tgt"]["engine_value_mask"], s["tgt"]["teacher_idx"]
        passed = s["tgt"]["passed"]
        # pass frames: the pass teacher P(pass) = 1, the pass scored (id 64), no place teacher
        assert m[passed].all() and (idx[passed][:, 0] == 64).all() and (idx[passed][:, 1:] == -1).all()
        assert torch.equal(s["tgt"]["pass_probs_mask"], m)
        assert torch.equal(s["tgt"]["pass_probs"], passed.float() * m)
        mv = m & ~passed
        pp = s["tgt"]["place_probs"]
        assert torch.allclose(pp[mv].sum(-1), torch.ones(int(mv.sum())), atol=1e-5) and (pp[~mv] == 0).all()
        assert (idx[~m] == -1).all()
        # teacher squares are empty cells (legal placements); the played move is among them
        empty = s["obs"]["board_seq"][:, 2].reshape(T, 64)
        for t in torch.nonzero(mv).flatten().tolist():
            sq = idx[t][idx[t] >= 0]
            assert (empty[t, sq] == 1).all() and int(s["tgt"]["place_idx"][t]) in sq.tolist()
            assert (pp[t][empty[t] == 0] == 0).all()


def test_policy_and_q_rederive_from_raw_columns(labelled):
    kind, out, summary = labelled
    exact_empties = FAKE_EXACT if kind == "fake" else int(summary["teacher"]["limits"]["exact_empties"])
    n_pass = n_split = 0
    for name in _shards(out):
        a = GmShard(str(out / name)).arrays()
        idx, q, raw = (np.asarray(a[k]) for k in ("tgt.teacher_idx", "tgt.teacher_q", "tgt.teacher_raw"))
        ev, pp = np.asarray(a["tgt.engine_value"]), np.asarray(a["tgt.place_probs"])
        packed = np.asarray(a["obs.board_seq"])  # packbits storage: 192 bits per frame
        empties = _empties(np.unpackbits(packed.reshape(len(packed), -1), axis=1, count=192).reshape(-1, 3, 8, 8))
        n_exact = 0
        for f in range(len(idx)):
            ok = idx[f] >= 0
            if not ok.any():
                continue
            qs, rs = q[f][ok].astype(np.float64), raw[f][ok].astype(np.float64)
            if empties[f] <= exact_empties:
                n_exact += 1
                want = np.where(rs > 0, 1.0, np.where(rs < 0, 0.0, 0.5))
            else:
                want = 1.0 / (1.0 + np.exp(-rs / DISC_SCALE))
            assert np.allclose(qs, want, atol=1e-6), (name, f)
            assert math.isclose(ev[f], 2 * qs.max() - 1, abs_tol=1e-6)
            if idx[f][0] == 64:  # a pass frame: no place teacher
                n_pass += 1
                assert (pp[f] == 0).all()
                continue
            # best first by Q, among equal Q by discs; π ∝ exp(Q/τ + w·tanh(discs/16))
            assert all(qs[j - 1] > qs[j] or (qs[j - 1] == qs[j] and rs[j - 1] >= rs[j]) for j in range(1, len(qs)))
            s = (qs - qs.max()) / TAU + W * np.tanh(rs / 16.0)
            e = np.exp(s - s.max())
            dense = np.zeros(64)
            dense[idx[f][ok]] = e / e.sum()
            assert np.allclose(pp[f], dense, atol=1e-5), (name, f)
            # an exactly decided position (all Q equal) with different margins: the teacher
            # prefers the smaller loss / larger win instead of spreading uniformly
            if empties[f] <= exact_empties and (qs == qs[0]).all() and qs[0] != 0.5 and rs[0] > rs[-1]:
                n_split += 1
                assert pp[f][idx[f][ok][0]] > pp[f][idx[f][ok][-1]], (name, f)
        assert n_exact > 0
    assert n_pass > 0, "the fixture games include pass frames"
    assert n_split > 0, "exactly decided WTHOR endgames reach the tie-break"


def test_trainer_consumes_teacher_targets(labelled):
    from model.games import build_game_network
    from training.train_il_game import batch_loss, prepare_batch
    from tests.test_train_il_game import SMALL

    kind, out, _ = labelled
    ds = GameShardDataset(str(out), game="othello")
    torch.manual_seed(0)
    net = build_game_network("othello", **SMALL)
    batch = collate_game_sequences([ds[i] for i in range(4)])
    mb = prepare_batch("othello", batch, torch.device("cpu"), tf_prob=1.0)
    kw = dict(value_weight=0.2, q_halt_weight=0.5, bf16=False, act_steps=1)
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), soft_target_weight=1.0, engine_value_weight=0.5, **kw)
    assert torch.isfinite(loss)
    assert math.isfinite(stats["place_teacher_kl"]) and stats["place_teacher_kl"] > 0
    assert 0 < stats["place_teacher_frac"] <= 1
    assert math.isfinite(stats["pass_teacher_kl"]) and stats["pass_teacher_frac"] > 0
    assert 0 < stats["engine_value_frac"] <= 1 and math.isfinite(stats["engine_value_mse"])
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in net.parameters())
    # the teacher term is live: dropping it (alpha = beta = 0) changes the objective
    with torch.no_grad():
        plain, _, _ = batch_loss(net, mb, torch.device("cpu"), soft_target_weight=0.0, engine_value_weight=0.0, **kw)
    assert abs(float(plain) - float(loss.detach())) > 1e-6
