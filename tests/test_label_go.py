"""KataGo teacher labels for go shards (`gmdata label go`) reach the loader and the IL
trainer unchanged.

Tiny 9x9 records (a capture-to-live position, a ko, passes / irregular turn order /
a double pass, a white-first handicap game) are converted with `gmdata go convert`
and labelled by `gmdata label go` with the gmdata binary itself as a fake KataGo
analysis engine child process (`gmdata label fake-engine katago`, the JSON protocol
with computed answers). The overlay must load with GameShardDataset, carry the
documented columns (visit-count policy over row*N+col / pass = N*N, place / pass head
soft targets, engine value, raw Q / score lead / prior), and feed
training/train_il_game.py's teacher-KL and engine-value losses through the real go
network. When KataGo is installed (scripts/engines/fetch_engines.sh katago) the same
records are also labelled by the real engine.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, GmShard, collate_game_sequences
from tests.test_rust_converters_parity import gmdata  # noqa: F401  (fixture: builds gmdata)

N = 9
K = 16                    # --max-moves default of the go labeller
PASS = N * N
SGFS = {
    # black's corner group is in atari; it lives only by capturing at "cc" = point 20
    "a_capture": "(;GM[1]SZ[9]KM[7.5]RU[Chinese]AB[ba][ab][bb][ad][bd][cg]AW[ca][cb][ac][bc]"
                 ";B[cc];W[gc];B[gg];W[ee];B[eg])",
    # after B takes at "cb", white may not retake the ko at "bb" = point 10
    "b_ko": "(;GM[1]SZ[9]KM[6.5]RU[Japanese]AB[ba][ab][bc]AW[ca][db][cc][bb];B[cb];W[gg];B[bb];W[cg];B[gc])",
    "c_passes": "(;GM[1]SZ[9]KM[7]RU[Chinese];B[ee];W[];W[cc];B[gg];W[];B[];W[dd];B[ff])",
    "d_handicap": "(;GM[1]SZ[9]HA[2]RU[Japanese]AB[cc][gg];W[ee];B[cg];W[gc])",
}
CAPTURE_POINT, KO_POINT = 2 * N + 2, N + 1
PASSES_MASK = [True, True, False, True, True, True, False, False]
SMALL = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, core_type="trm",
             trm_H_cycles=1, trm_L_cycles=1, trm_block_layers=1, halt_max_steps=1,
             dropout=0.0, encoder_channels=8)


def _convert(gm, root):
    sgf = root / "sgf"
    sgf.mkdir(parents=True)
    for name, text in SGFS.items():
        (sgf / f"{name}.sgf").write_text(text)
    subprocess.run([gm, "go", "convert", "--input", str(sgf), "--out", str(root / "base"), "--board-size", str(N),
                    "--min-moves", "1", "--allow-handicap", "--allow-no-result"], check=True, capture_output=True)


def _label(gm, root, out, engine_argv, *extra):
    r = subprocess.run([gm, "label", "go", "--input", str(root / "base"), "--out", str(out), "--board-size", str(N),
                        "--progress-s", "0", *engine_argv, *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout)


@pytest.fixture(scope="module")
def labelled(gmdata, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("label_go")
    _convert(gmdata, root)
    fake = ["--engine", gmdata, "--engine-arg", "label", "--engine-arg", "fake-engine", "--engine-arg", "katago"]
    return root, _label(gmdata, root, root / "out", fake, "--workers", "2")


def _katago():
    engines = os.environ.get("GM_ENGINES", os.path.expanduser("~/gm-engines"))
    exe = os.environ.get("GM_KATAGO") or (os.path.join(engines, "bin", "katago")
                                          if os.path.exists(os.path.join(engines, "bin", "katago")) else shutil.which("katago"))
    model = os.environ.get("GM_KATAGO_MODEL", os.path.join(engines, "nets", "katago", "default.bin.gz"))
    return (exe, model) if exe and os.path.isfile(exe) and os.path.isfile(model) else None


def test_summary_and_meta(labelled):
    root, s = labelled
    st = s["stats"]
    assert s["shards_written"] == 1 and st["samples"] == 4
    assert st["verify_failed"] == st["replay_failed"] == st["engine_errors"] == st["bad_analysis"] == 0
    assert st["skipped"] == {"turn_order": 1, "after_double_pass": 2}
    assert st["labelled"] + 3 == st["frames"]
    t = s["teacher"]
    assert t["engine"] == "KataGo 1.18.2-fake fake-b1c1"
    assert t["policy"] == {"kind": "visits"} and t["raw_unit"] == "score_lead" and t["visits"] and t["prior"]
    assert t["limits"] == {"visits": "100"} and t["max_moves"] == K
    assert t["heads"] == {"place": {"kind": "categorical", "n": PASS, "k": K, "storage": "sparse"},
                          "pass": {"kind": "bernoulli"}}
    idx = json.load(open(root / "out" / "go_00000" / "index.json"))
    base = json.load(open(root / "base" / "go_00000" / "index.json"))
    assert idx["meta"]["teacher"] == t and {k: v for k, v in idx["meta"].items() if k != "teacher"} == base["meta"]
    assert idx["samples"] == base["samples"]


def test_loader_columns(labelled):
    root, _ = labelled
    base = GameShardDataset(str(root / "base"), game="go")
    lab = GameShardDataset(str(root / "out"), game="go")
    assert len(lab) == len(base) == 4 and lab.shard_meta["board_size"] == N
    new = {"teacher_idx": (torch.int64, (K,)), "teacher_q": (torch.float32, (K,)),
           "teacher_raw": (torch.float32, (K,)), "teacher_visits": (torch.float32, (K,)),
           "teacher_prior": (torch.float32, (K,)), "engine_value": (torch.float32, ()),
           "engine_value_mask": (torch.bool, ()), "place_probs_idx": (torch.int64, (K,)),
           "place_probs_val": (torch.float32, (K,)), "pass_probs": (torch.float32, ()),
           "pass_probs_mask": (torch.bool, ())}
    for i in range(len(lab)):
        b, s = base[i], lab[i]
        assert set(s["tgt"]) == set(b["tgt"]) | set(new)
        for group in ("obs", "tgt"):
            for key, v in b[group].items():
                assert torch.equal(s[group][key], v), (i, key)
        T = s["tgt"]["place_idx"].shape[0]
        for key, (dt, shape) in new.items():
            assert s["tgt"][key].dtype == dt and tuple(s["tgt"][key].shape) == (T, *shape), key
        m = s["tgt"]["engine_value_mask"]
        assert torch.equal(m, s["tgt"]["pass_probs_mask"])
        idx, vis = s["tgt"]["teacher_idx"], s["tgt"]["teacher_visits"]
        assert torch.equal(idx == -1, torch.isnan(vis)) and not (idx[~m] != -1).any()
        assert ((idx >= -1) & (idx <= PASS)).all()
        ev = s["tgt"]["engine_value"]
        assert ((ev >= -1) & (ev <= 1)).all() and (ev[~m] == 0).all()
        # place = p(place | not pass): a distribution over points (or no teacher); pass = P(pass)
        sums = s["tgt"]["place_probs_val"].sum(-1)
        assert (((sums - 1).abs() < 1e-5) | (sums == 0)).all() and (sums[~m] == 0).all()
        assert not (s["tgt"]["place_probs_idx"] == PASS).any()
    # scenario checks: capture = top move, no ko retake, pass-scenario frame mask
    cap, ko, passes = lab[0]["tgt"], lab[1]["tgt"], lab[2]["tgt"]
    assert int(cap["teacher_idx"][0, 0]) == CAPTURE_POINT and int(cap["place_probs_idx"][0, 0]) == CAPTURE_POINT
    assert bool(ko["engine_value_mask"][1]) and KO_POINT not in ko["teacher_idx"][1].tolist()
    assert lab[1]["obs"]["board_seq"][1, 3].flatten()[KO_POINT] == 1       # the shard's ko plane
    assert passes["engine_value_mask"].tolist() == PASSES_MASK


def test_head_targets_rederive_from_visits(labelled):
    """place_probs / pass_probs = normalised visits split on pass, per the PlacementDecoder docstring."""
    root, _ = labelled
    a = GmShard(str(root / "out" / "go_00000")).arrays()
    idx, vis = np.asarray(a["tgt.teacher_idx"]), np.asarray(a["tgt.teacher_visits"])
    pi, pv, pp = np.asarray(a["tgt.place_probs_idx"]), np.asarray(a["tgt.place_probs_val"]), np.asarray(a["tgt.pass_probs"])
    for f in range(idx.shape[0]):
        ok = idx[f] >= 0
        if not ok.any():
            continue
        w = vis[f][ok].astype(np.float64) / vis[f][ok].sum()
        p_pass = float(w[idx[f][ok] == PASS].sum())
        assert math.isclose(pp[f], p_pass, abs_tol=1e-6)
        want = {int(a_): float(x / (1 - p_pass)) for a_, x in zip(idx[f][ok], w) if a_ != PASS and x > 0}
        got = {int(c): float(p) for c, p in zip(pi[f], pv[f]) if c >= 0}
        assert got.keys() == want.keys() and all(math.isclose(got[c], want[c], abs_tol=1e-5) for c in got)


def _loss(root_out):
    from model.games import build_game_network
    from training.train_il_game import batch_loss, prepare_batch

    ds = GameShardDataset(str(root_out), game="go")
    torch.manual_seed(0)
    net = build_game_network("go", board_h=N, board_w=N, board_global_features=6, **SMALL)
    batch = collate_game_sequences([ds[i] for i in range(len(ds))])
    mb = prepare_batch("go", batch, torch.device("cpu"), tf_prob=1.0)
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), value_weight=0.2, q_halt_weight=0.5, bf16=False,
                                act_steps=1, soft_target_weight=1.0, engine_value_weight=0.5)
    return loss, stats


def test_trainer_consumes_go_teacher_targets(labelled):
    root, _ = labelled
    loss, stats = _loss(root / "out")
    assert torch.isfinite(loss)
    for h in ("place", "pass"):
        kl = float(stats[f"{h}_teacher_kl"])
        assert math.isfinite(kl) and kl > 0, (h, kl)          # an untrained net disagrees with the teacher
        assert float(stats[f"{h}_teacher_frac"]) > 0, h
    assert 0 < float(stats["engine_value_frac"]) <= 1 and math.isfinite(float(stats["engine_value_mse"]))
    loss.backward()


def test_real_katago_labels_feed_the_trainer(gmdata, tmp_path):  # noqa: F811
    kg = _katago()
    if kg is None:
        pytest.skip("KataGo or its network not installed (scripts/engines/fetch_engines.sh katago)")
    exe, model = kg
    _convert(gmdata, tmp_path)
    s = _label(gmdata, tmp_path, tmp_path / "out", ["--engine", exe, "--model", model],
               "--limit", "visits=64", "--analysis-threads", "8", "--workers", "1")
    st = s["stats"]
    assert st["engine_errors"] == st["bad_analysis"] == st["verify_failed"] == 0 and st["labelled"] == 18
    assert s["teacher"]["engine"].startswith("KataGo 1.")
    lab = GameShardDataset(str(tmp_path / "out"), game="go")
    assert int(lab[0]["tgt"]["teacher_idx"][0, 0]) == CAPTURE_POINT          # KataGo captures to live
    assert KO_POINT not in lab[1]["tgt"]["teacher_idx"][1].tolist()
    loss, stats = _loss(tmp_path / "out")
    assert torch.isfinite(loss) and float(stats["place_teacher_kl"]) > 0
