"""`gmdata label chess` overlays feed the Python loader and the IL trainer unchanged.

Real chess shards (`gmdata chess convert` on the committed Lichess fixture) are
labelled through the real CLI and driver with a fake UCI engine child process
(`gmdata label fake-engine chess`: shakmaty move generation, deterministic scores,
real UCI protocol), so no engine binary is needed. The overlays must load with
GameShardDataset, keep every base tensor bit-identical, carry the documented teacher
columns (flat action id = promotion·4096 + from·64 + to in POV squares), re-derive
from the raw columns exactly as FromToDecoder.il_head_terms defines the soft targets,
and train the real chess network through training.train_il_game.batch_loss with the
teacher KL and the engine-value term active. With Stockfish installed
(scripts/engines/fetch_engines.sh stockfish, or STOCKFISH=/path) the same loss check
also runs on real engine labels; otherwise that test is skipped.
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
from tests.test_rust_converters_parity import PGN, gmdata  # noqa: F401  (fixture: builds gmdata)

K = 8                   # --max-moves default of the chess labeller (= MultiPV)
TAU = 0.05


def _convert(gmdata, out, games=6, shard_size=3):  # noqa: F811
    subprocess.run([gmdata, "chess", "convert", "--input", PGN, "--out", str(out),
                    "--max-games", str(games), "--shard-size", str(shard_size)],
                   check=True, capture_output=True, text=True)


def _label(gmdata, base, out, engine, engine_args=(), nodes=1000):  # noqa: F811
    cmd = [gmdata, "label", "chess", "--input", str(base), "--out", str(out), "--engine", engine,
           "--workers", "2", "--limit", f"nodes={nodes}", "--progress-s", "1000"]
    for a in engine_args:
        cmd += ["--engine-arg", a]
    r = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def labelled(gmdata, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("label_chess")
    _convert(gmdata, root / "base")
    summary = _label(gmdata, root / "base", root / "lab", gmdata, ("label", "fake-engine", "chess"))
    return root, summary


def _shards(root):
    return sorted(p for p in os.listdir(root) if not p.startswith("."))


def _decode(ids):
    ids = np.asarray(ids)
    return ids % 4096 // 64, ids % 64, ids // 4096


def test_summary_and_meta(labelled):
    root, summary = labelled
    st = summary["stats"]
    assert summary["shards_written"] == 2 and st["samples"] == 6
    assert st["verify_failed"] == st["replay_failed"] == st["engine_errors"] == st["bad_analysis"] == 0
    assert st["labelled"] + sum(st["skipped"].values()) == st["frames"]
    assert set(st["skipped"]) <= {"forced"}
    t = summary["teacher"]
    assert t["engine"] == "FakeUci 1.0" and t["labeller"] == "gmdata label chess"
    assert t["policy"] == {"kind": "softmax", "tau": TAU} and t["max_moves"] == K
    assert t["limits"] == {"nodes": "1000"} and t["raw_unit"] == "cp"
    assert t["heads"]["from"]["storage"] == t["heads"]["to"]["storage"] == "sparse"
    assert t["heads"]["promotion"]["storage"] == "dense"
    for name in _shards(root / "lab"):
        idx = json.load(open(root / "lab" / name / "index.json"))
        base = json.load(open(root / "base" / name / "index.json"))
        assert idx["meta"]["teacher"] == t
        assert {k: v for k, v in idx["meta"].items() if k != "teacher"} == base["meta"]
        assert idx["samples"] == base["samples"] and idx["n_frames"] == base["n_frames"]
        for key, info in base["tensors"].items():
            assert idx["tensors"][key] == info
            a = os.stat(root / "lab" / name / info["file"])
            b = os.stat(root / "base" / name / info["file"])
            assert (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino), f"{key} not hard-linked"


def test_loader_sees_teacher_columns(labelled):
    root, _ = labelled
    base = GameShardDataset(str(root / "base"), game="chess")
    lab = GameShardDataset(str(root / "lab"), game="chess")
    assert len(base) == len(lab) == 6 and lab.converter_version == base.converter_version == 3
    new = {
        "teacher_idx": (torch.int64, (K,)), "teacher_q": (torch.float32, (K,)),
        "teacher_raw": (torch.float32, (K,)), "engine_value": (torch.float32, ()),
        "engine_value_mask": (torch.bool, ()), "promotion_probs": (torch.float32, (5,)),
        "from_probs_idx": (torch.int64, (K,)), "from_probs_val": (torch.float32, (K,)),
        "to_probs_idx": (torch.int64, (K,)), "to_probs_val": (torch.float32, (K,)),
    }
    for i in range(len(lab)):
        b, s = base[i], lab[i]
        assert set(s["tgt"]) == set(b["tgt"]) | set(new)
        for group in ("obs", "tgt"):
            for key, v in b[group].items():
                assert torch.equal(s[group][key], v), (i, key)
        T = s["tgt"]["from_idx"].shape[0]
        for key, (dt, shape) in new.items():
            v = s["tgt"][key]
            assert v.dtype == dt and tuple(v.shape) == (T, *shape), (key, v.dtype, v.shape)
        m, idx, q = s["tgt"]["engine_value_mask"], s["tgt"]["teacher_idx"], s["tgt"]["teacher_q"]
        assert m.float().mean() > 0.9
        assert torch.equal(idx == -1, torch.isnan(q)) and torch.equal(idx == -1, torch.isnan(s["tgt"]["teacher_raw"]))
        assert (idx[~m] == -1).all() and (s["tgt"]["engine_value"][~m] == 0).all()
        # teacher moves are legal: their from-squares are in tgt.legal_from
        f, _, _ = _decode(idx[m].clamp(min=0).numpy())
        legal_from = s["tgt"]["legal_from"][m].numpy()
        ok = idx[m].numpy() >= 0
        assert (np.take_along_axis(legal_from, f, 1)[ok] == 1).all()
        # the fake's multipv 1 line is its best-scored move → engine value = 2·Q[0] − 1
        assert torch.allclose(s["tgt"]["engine_value"][m], 2 * q[m][:, 0] - 1, atol=1e-6)
        assert torch.allclose(s["tgt"]["from_probs_val"][m].sum(-1), torch.ones(int(m.sum())), atol=1e-5)
        for key in ("to_probs_val", "promotion_probs"):
            sums = s["tgt"][key].sum(-1)
            assert (((sums - 1).abs() < 1e-5) | (sums == 0)).all() and (sums[~m] == 0).all()


def test_soft_targets_rederive_from_raw_columns(labelled):
    """from = marginal of softmax(Q/τ) over from-squares; to = p(to | from = tgt.from_idx);
    promotion = p(promo | tgt.from_idx, tgt.to_idx) — FromToDecoder.il_head_terms."""
    root, _ = labelled
    n_to = n_promo = 0
    for name in _shards(root / "lab"):
        a = GmShard(str(root / "lab" / name)).arrays()
        idx, q = np.asarray(a["tgt.teacher_idx"]), np.asarray(a["tgt.teacher_q"])
        gf, gt = np.asarray(a["tgt.from_idx"]), np.asarray(a["tgt.to_idx"])
        fi, fv = np.asarray(a["tgt.from_probs_idx"]), np.asarray(a["tgt.from_probs_val"])
        ti, tv = np.asarray(a["tgt.to_probs_idx"]), np.asarray(a["tgt.to_probs_val"])
        pp = np.asarray(a["tgt.promotion_probs"])
        for f in range(len(idx)):
            ok = idx[f] >= 0
            if not ok.any():
                continue
            e = np.exp((q[f][ok].astype(np.float64) - q[f][ok].max()) / TAU)
            pi = e / e.sum()
            frm, to, pro = _decode(idx[f][ok])
            want = {int(s): pi[frm == s].sum() for s in np.unique(frm)}
            got = {int(c): float(p) for c, p in zip(fi[f], fv[f]) if c >= 0}
            assert got.keys() == want.keys() and all(math.isclose(got[c], want[c], abs_tol=1e-5) for c in got)
            sel = frm == gf[f]
            got_to = {int(c): float(p) for c, p in zip(ti[f], tv[f]) if c >= 0}
            if sel.any():
                n_to += 1
                w = {int(s): pi[sel & (to == s)].sum() / pi[sel].sum() for s in np.unique(to[sel])}
                assert got_to.keys() == w.keys() and all(math.isclose(got_to[c], w[c], abs_tol=1e-5) for c in w)
            else:
                assert not got_to
            sel2 = sel & (to == gt[f])
            want_p = np.zeros(5)
            if sel2.any():
                n_promo += 1
                for c in range(5):
                    want_p[c] = pi[sel2 & (pro == c)].sum() / pi[sel2].sum()
            assert np.allclose(pp[f], want_p, atol=1e-5)
    assert n_to > 0 and n_promo > 0


def _train_step(ds, n=4, **weights):
    from model.games import build_game_network
    from training.train_il_game import batch_loss, prepare_batch
    from tests.test_train_il_game import SMALL

    torch.manual_seed(0)
    net = build_game_network("chess", board_global_features=10, **SMALL)
    batch = collate_game_sequences([ds[i] for i in range(n)])
    mb = prepare_batch("chess", batch, torch.device("cpu"), tf_prob=1.0, il_legal_mask=True)
    return batch_loss(net, mb, torch.device("cpu"), value_weight=0.2, q_halt_weight=0.5,
                      bf16=False, act_steps=1, **weights)


def _check_teacher_terms(ds, n=4):
    loss, stats, _ = _train_step(ds, n, soft_target_weight=1.0, engine_value_weight=0.5)
    assert torch.isfinite(loss)
    for h in ("from", "to", "promotion"):
        assert math.isfinite(float(stats[f"{h}_teacher_kl"])), h
        assert float(stats[f"{h}_teacher_frac"]) > 0, h
        # the teacher scores legal moves only
        assert float(stats[f"{h}_teacher_illegal_mass"]) < 1e-6, h
    assert float(stats["from_teacher_kl"]) > 0 and float(stats["to_teacher_kl"]) > 0
    assert 0 < float(stats["engine_value_frac"]) <= 1
    assert math.isfinite(float(stats["engine_value_mse"]))
    # the teacher terms change the objective (same net, same batch)
    plain, _, _ = _train_step(ds, n, soft_target_weight=0.0, engine_value_weight=0.0)
    assert abs(loss.item() - plain.item()) > 1e-4
    loss.backward()


def test_trainer_consumes_chess_teacher_targets(labelled):
    root, _ = labelled
    _check_teacher_terms(GameShardDataset(str(root / "lab"), game="chess"))


def _stockfish():
    p = os.environ.get("STOCKFISH")
    if p:
        return p if os.path.isfile(p) else None
    cand = os.path.join(os.environ.get("GM_ENGINES", os.path.expanduser("~/gm-engines")), "bin", "stockfish")
    return cand if os.path.isfile(cand) else None


def test_real_stockfish_labels_train(gmdata, tmp_path):  # noqa: F811
    sf = _stockfish()
    if sf is None:
        pytest.skip("stockfish not installed (scripts/engines/fetch_engines.sh stockfish)")
    _convert(gmdata, tmp_path / "base", games=2, shard_size=2)
    summary = _label(gmdata, tmp_path / "base", tmp_path / "lab", sf, nodes=5000)
    st = summary["stats"]
    assert summary["teacher"]["engine"].startswith("Stockfish")
    assert st["engine_errors"] == st["verify_failed"] == st["bad_analysis"] == 0 and st["labelled"] > 0
    ds = GameShardDataset(str(tmp_path / "lab"), game="chess")
    s = ds[0]["tgt"]
    m = s["engine_value_mask"]
    q = s["teacher_q"][m]
    assert ((q[~torch.isnan(q)] >= 0) & (q[~torch.isnan(q)] <= 1)).all()
    assert (s["teacher_idx"][m][:, 0] >= 0).all()
    loss, stats, _ = _train_step(ds, 2, soft_target_weight=1.0, engine_value_weight=0.5)
    assert torch.isfinite(loss) and float(stats["from_teacher_kl"]) > 0
    assert float(stats["from_teacher_illegal_mass"]) < 1e-6
