"""`gmdata label shogi` overlays reach the Python loader and the IL trainer unchanged.

The floodgate fixture games (tests/fixtures/shogi) are converted with `gmdata shogi
convert` and labelled with `gmdata label shogi`, the gmdata binary itself serving as the
USI engine child process (`gmdata label shogi fake-usi`: the converter's legal moves with
hashed scores, some of them mates) — no engine binary needed. The overlays must load with
GameShardDataset, keep every base tensor, carry the documented teacher columns (heads of
ShogiDecoder.il_head_terms), re-derive from the raw columns (π ∝ exp(Q/τ + w·u), u = the
secondary utility of teacher_raw), and feed
training/train_il_game.batch_loss's soft-target and engine-value terms through a small
shogi network. With YaneuraOu installed (scripts/engines/fetch_engines.sh yaneuraou, or
GM_YANEURAOU=/path), the last test repeats the pipeline with the real engine.
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
from tests.test_rust_converters_parity import FIX, gmdata  # noqa: F401  (fixture: builds gmdata)

K = 8                 # --max-moves default (MultiPV 8)
TAU = 0.05
W = 2.0               # --tie-weight default
N_BOARD = 81 * 81 * 2
HEADS = {  # dense heads: (tgt key, frame shape)
    "mode": ("mode_probs", (2,)),
    "promo": ("promo_probs", (2,)),
    "piece": ("piece_probs", (14,)),
}
SPARSE = ("from", "to", "drop_to")


def _label(gmdata, root, engine_args, extra=()):  # noqa: F811
    base, out = root / "base", root / "out"
    subprocess.run([gmdata, "shogi", "convert", "--input", os.path.join(FIX, "shogi"), "--out", str(base),
                    "--shard-size", "16"], check=True, capture_output=True)
    r = subprocess.run([gmdata, "label", "shogi", "--input", str(base), "--out", str(out), "--workers", "2",
                        "--progress-s", "1000", *engine_args, *extra], check=True, capture_output=True, text=True)
    return base, out, json.loads(r.stdout)


@pytest.fixture(scope="module")
def labelled(gmdata, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("label_shogi")
    return _label(gmdata, root, ["--engine", gmdata, "--engine-arg", "label", "--engine-arg", "shogi",
                                 "--engine-arg", "fake-usi"])


def _decode(a):
    """flat action id → ('board', from, to, promote) | ('drop', slot, to)."""
    if a < N_BOARD:
        return ("board", a // 2 // 81, a // 2 % 81, a % 2)
    r = a - N_BOARD
    return ("drop", r // 81, r % 81)


def _u(raw):
    """Secondary utility of teacher_raw (label/shogi.rs `secondary_of_raw`): a mate in p
    plies (|raw| > 90000, p = 100000 - |raw|) -> the chess mate utility ±(0.5 + 0.5·5/(n + 4))
    of n = (p + 1)/2 moves (the mover mates) or p/2 (it is mated), at least 1; else
    0.5·tanh(cp/3000)."""
    raw = np.asarray(raw, np.float64)
    p = 100000.0 - np.abs(raw)
    n = np.maximum(np.where(raw > 0, (p + 1) / 2, p / 2), 1.0)
    return np.where(np.abs(raw) > 90000, np.sign(raw) * (0.5 + 2.5 / (n + 4.0)), 0.5 * np.tanh(raw / 3000.0))


def _expected(idx, q, raw, dropped, gf, gt, gp):
    """The six head rows of one frame from its raw columns (ShogiDecoder.il_head_terms;
    π ∝ exp(Q/τ + w·u))."""
    ok = idx >= 0
    s = (q[ok].astype(np.float64) - q[ok].max()) / TAU + W * _u(raw[ok])
    e = np.exp(s - s.max())
    pi = e / e.sum()
    acts = [_decode(int(a)) for a in idx[ok]]
    rows = {h: {} for h in ("mode", "from", "to", "promo", "piece", "drop_to")}
    for a, p in zip(acts, pi):
        rows["mode"][int(a[0] == "drop")] = rows["mode"].get(int(a[0] == "drop"), 0) + p
        if a[0] == "board":
            rows["from"][a[1]] = rows["from"].get(a[1], 0) + p
            if dropped == 0 and a[1] == gf:
                rows["to"][a[2]] = rows["to"].get(a[2], 0) + p
                if a[2] == gt:
                    rows["promo"][a[3]] = rows["promo"].get(a[3], 0) + p
        else:
            rows["piece"][a[1]] = rows["piece"].get(a[1], 0) + p
            if dropped == 1 and a[1] == gp:
                rows["drop_to"][a[2]] = rows["drop_to"].get(a[2], 0) + p
    for h, r in rows.items():
        s = sum(r.values())
        rows[h] = {c: v / s for c, v in r.items()} if s > 0 else {}
    return rows


def test_summary_and_meta(labelled):
    base, out, summary = labelled
    st = summary["stats"]
    assert summary["shards_written"] == len(os.listdir(base)) >= 2
    assert st["verify_failed"] == st["replay_failed"] == st["engine_errors"] == st["bad_analysis"] == 0
    assert st["labelled"] + sum(st["skipped"].values()) == st["frames"] and st["labelled"] > 3000
    assert st["cache_hits"] > 0, "shared opening lines"
    t = summary["teacher"]
    assert t["engine"] == "FakeUsi 1.0" and t["labeller"] == "gmdata label shogi"
    assert t["limits"] == {"nodes": "1000000"} and t["max_moves"] == K and t["raw_unit"] == "cp"
    assert t["policy"]["kind"] == "softmax" and t["policy"]["tau"] == TAU and t["policy"]["tie_weight"] == W
    assert t["policy"]["secondary"]["of"] == "teacher_raw" and t["labeller_version"] == 3 and t["core_version"] == 2
    assert t["game"]["cp_scale"] == 600 and t["game"]["usi_options"]["Threads"] == "1"
    for name in sorted(os.listdir(out)):
        idx = json.load(open(out / name / "index.json"))
        b = json.load(open(base / name / "index.json"))
        assert idx["meta"]["teacher"] == t
        assert {k: v for k, v in idx["meta"].items() if k != "teacher"} == b["meta"] == {"global_features": 7}
        assert idx["samples"] == b["samples"] and idx["n_frames"] == b["n_frames"]
        for key, info in b["tensors"].items():
            assert idx["tensors"][key] == info


def test_loader_columns(labelled):
    base, out, _ = labelled
    b_ds, ds = GameShardDataset(str(base), game="shogi"), GameShardDataset(str(out), game="shogi")
    assert len(ds) == len(b_ds) and ds.converter_version == 2
    new = {"teacher_idx": (torch.int64, (K,)), "teacher_q": (torch.float32, (K,)),
           "teacher_raw": (torch.float32, (K,)), "engine_value": (torch.float32, ()),
           "engine_value_mask": (torch.bool, ()),
           **{key: (torch.float32, shape) for key, shape in HEADS.values()},
           **{f"{h}_probs_idx": (torch.int64, (K,)) for h in SPARSE},
           **{f"{h}_probs_val": (torch.float32, (K,)) for h in SPARSE}}
    drops = mates = 0
    for i in range(len(ds)):
        s, b = ds[i], b_ds[i]
        assert set(s["tgt"]) == set(b["tgt"]) | set(new)
        for key, v in b["tgt"].items():
            assert torch.equal(s["tgt"][key], v), key
        T = s["tgt"]["from_idx"].shape[0]
        for key, (dt, shape) in new.items():
            assert s["tgt"][key].dtype == dt and tuple(s["tgt"][key].shape) == (T, *shape), key
        tg, m = s["tgt"], s["tgt"]["engine_value_mask"]
        assert torch.equal(tg["teacher_idx"] == -1, torch.isnan(tg["teacher_q"]))
        assert torch.equal(torch.isnan(tg["teacher_q"]), torch.isnan(tg["teacher_raw"]))
        assert not (tg["teacher_idx"][~m] != -1).any() and (tg["engine_value"][~m] == 0).all()
        ev = tg["engine_value"][m]
        assert ((ev >= -1) & (ev <= 1)).all()
        # teacher_raw: cp, mates at +-(100000 - n) — beyond any cp YaneuraOu prints (35281)
        raw, tq = tg["teacher_raw"], tg["teacher_q"]
        mate = raw.abs() > 90000
        assert torch.equal(tq[mate], (raw[mate] > 0).float())
        assert (raw[~mate & ~torch.isnan(raw)].abs() <= 600).all(), "the fake engine's cp range"
        mates += int(mate.sum())
        assert torch.allclose(ev, 2 * tg["teacher_q"][m][:, 0] - 1, atol=1e-6)
        assert torch.allclose(tg["mode_probs"][m].sum(-1), torch.ones(int(m.sum())), atol=1e-5)
        mv, dr = tg["dropped"] == 0, tg["dropped"] == 1
        # GT-conditioned rows only on frames of their mode
        assert (tg["to_probs_val"][dr] == 0).all() and (tg["promo_probs"][dr] == 0).all()
        assert (tg["drop_to_probs_val"][mv] == 0).all()
        assert (tg["piece_probs"][:, 7:] == 0).all(), "only the mover's hand slots"
        for key in ("from_probs_val", "to_probs_val", "promo_probs", "piece_probs", "drop_to_probs_val"):
            sums = tg[key].sum(-1)
            assert (((sums - 1).abs() < 1e-5) | (sums == 0)).all() and (sums[~m] == 0).all(), key
        drops += int((dr & m & (tg["drop_to_probs_val"].sum(-1) > 0)).sum())
    assert drops > 0, "drop frames carry drop-square teachers"
    assert mates > 0, "mate scores reach teacher_raw"


def test_soft_targets_rederive_from_raw_columns(labelled):
    _, out, _ = labelled
    checked = 0
    for name in sorted(os.listdir(out)):
        a = GmShard(str(out / name)).arrays()
        idx, q, raw = (np.asarray(a[f"tgt.teacher_{k}"]) for k in ("idx", "q", "raw"))
        gt = {k: np.asarray(a[f"tgt.{k}"]) for k in ("dropped", "from_idx", "to_idx", "piece_idx")}
        for f in range(idx.shape[0]):
            if idx[f, 0] < 0:
                continue
            ok = idx[f] >= 0
            qs, rs = q[f][ok], raw[f][ok]
            # best first: by Q, among equal Q by the raw score (mates: faster first)
            assert all(qs[j - 1] > qs[j] or (qs[j - 1] == qs[j] and rs[j - 1] >= rs[j]) for j in range(1, len(qs)))
            want = _expected(idx[f], q[f], raw[f], gt["dropped"][f], gt["from_idx"][f], gt["to_idx"][f], gt["piece_idx"][f])
            for h in ("mode", "promo", "piece"):
                row = np.asarray(a[f"tgt.{h}_probs"])[f]
                got = {c: float(p) for c, p in enumerate(row) if p > 0}
                assert got.keys() == want[h].keys(), (h, f)
                assert all(math.isclose(got[c], want[h][c], abs_tol=1e-5) for c in got), (h, f)
            for h in SPARSE:
                ids, vals = np.asarray(a[f"tgt.{h}_probs_idx"])[f], np.asarray(a[f"tgt.{h}_probs_val"])[f]
                got = {int(c): float(p) for c, p in zip(ids, vals) if c >= 0}
                assert got.keys() == want[h].keys(), (h, f)
                assert all(math.isclose(got[c], want[h][c], abs_tol=1e-5) for c in got), (h, f)
            checked += 1
    assert checked > 3000


def _batch_with_drops(ds, n=4):
    """n samples whose labelled frames include drop frames (the drop heads get teachers)."""
    pick = []
    for i in range(len(ds)):
        tg = ds[i]["tgt"]
        if ((tg["dropped"] == 1) & tg["engine_value_mask"]).sum() >= 3:
            pick.append(ds[i])
        if len(pick) == n:
            break
    assert len(pick) == n
    return collate_game_sequences(pick)


def _net_and_batch(out, drops=True):
    from model.games import build_game_network
    from training.train_il_game import prepare_batch
    from tests.test_train_il_game import SMALL

    ds = GameShardDataset(str(out), game="shogi")
    torch.manual_seed(0)
    net = build_game_network("shogi", shogi_global_features=7, **SMALL)
    batch = _batch_with_drops(ds) if drops else collate_game_sequences([ds[i] for i in range(4)])
    mb = prepare_batch("shogi", batch, torch.device("cpu"), tf_prob=1.0, il_legal_mask=True)
    return net, mb


def test_trainer_consumes_teacher_targets(labelled):
    from training.train_il_game import batch_loss

    _, out, _ = labelled
    net, mb = _net_and_batch(out)
    kw = dict(value_weight=0.2, q_halt_weight=0.5, bf16=False, act_steps=1)
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), soft_target_weight=1.0,
                                engine_value_weight=0.5, **kw)
    assert torch.isfinite(loss)
    for h in ("mode", "from", "to", "promo", "piece", "drop_to"):
        kl = float(stats[f"{h}_teacher_kl"])
        assert math.isfinite(kl) and kl > 0, (h, kl)          # the teacher KL term is live
        assert float(stats[f"{h}_teacher_frac"]) > 0, h
        # the fake engine scores legal moves only, and the GT-conditioned rows match the
        # converter's legal_*_of masks
        assert float(stats.get(f"{h}_teacher_illegal_mass", 0.0)) < 1e-6, h
    assert 0 < float(stats["engine_value_frac"]) <= 1
    assert math.isfinite(float(stats["engine_value_mse"]))
    # the soft targets change the objective (α = 0 only reports them)
    with torch.no_grad():
        plain, _, _ = batch_loss(net, mb, torch.device("cpu"), soft_target_weight=0.0,
                                 engine_value_weight=0.5, **kw)
    assert abs(float(plain) - float(loss.detach())) > 1e-4
    loss.backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def _yaneuraou():
    p = os.environ.get("GM_YANEURAOU")
    if p:
        return p if os.path.isfile(p) else None
    p = os.path.join(os.environ.get("GM_ENGINES", os.path.expanduser("~/gm-engines")), "bin", "yaneuraou")
    return p if os.path.isfile(p) else shutil.which("yaneuraou")


def test_real_yaneuraou_labels_train(gmdata, tmp_path):  # noqa: F811
    exe = _yaneuraou()
    if exe is None:
        pytest.skip("yaneuraou not installed (scripts/engines/fetch_engines.sh yaneuraou)")
    from training.train_il_game import batch_loss

    _, out, summary = _label(gmdata, tmp_path, ["--engine", exe],
                             ["--limit", "nodes=5000", "--max-samples", "6", "--max-shards", "1"])
    st = summary["stats"]
    assert st["engine_errors"] == st["verify_failed"] == st["bad_analysis"] == 0 and st["labelled"] > 100
    assert summary["teacher"]["engine"].startswith("YaneuraOu")
    net, mb = _net_and_batch(out, drops=False)
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), value_weight=0.2, q_halt_weight=0.5,
                                bf16=False, act_steps=1, soft_target_weight=1.0, engine_value_weight=0.5)
    assert torch.isfinite(loss) and float(stats["from_teacher_kl"]) > 0
    assert float(stats["from_teacher_illegal_mass"]) < 1e-6
