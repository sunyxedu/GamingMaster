"""Skat (ISS rules) end to end — rules, converter, simulator, adapter, trainers:

    gmrules skat ─► gmdata skat convert (real ISS records, tests/fixtures/skat) ─► shards
      ─► CandidateEncoder / CandidateDecoder ─► train_multi_game (with kuhn_poker)
      ─► gmsim env (3 per-seat rows) ─► train_rl_joint (1 iteration, with kuhn_poker)

- the converter replays every real record and reproduces every recorded result
  (`gmdata skat verify`: 0 illegal, 0 mismatches), the values are ISS's zero-sum
  money scores (re-derived here from the recorded results);
- golden: converter frames == `gmsim.replay` frames of each record, and simulator
  frames == converter frames of the simulator's own episodes (env → trajectories
  → `gmdata skat render` → ISS records → `gmdata skat convert`): IL frames == RL frames;
- hidden information: the simulator battery (causality + privacy) and, on the real
  converter binary, causality (ISS records cut after every move) and privacy (the
  hidden cards re-dealt by `gmsim.perturb_hidden`, rendered back to ISS records),
  with a negative control;
- quality filters (players / corpus statistics / ISS ratings);
- `scripts/fetch_skat.sh` is valid bash.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, load_gm_samples
from env.vec_gmsim import GmSimVecEnv
from tests import invariance

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
FIXTURE = ROOT / "tests" / "fixtures" / "skat" / "iss_sample.sgf"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
MONEY_SCALE = 1260.0


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return str(WS / "target" / "release" / "gmdata")


def _run(*args) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def shards(gmdata, tmp_path_factory):
    out = tmp_path_factory.mktemp("skat")
    st = _run(gmdata, "skat", "convert", "--input", FIXTURE, "--out", out / "train",
              "--shard-size", "64", "--store-events")
    return out, st


def _money(result: str) -> List[float]:
    """ISS's zero-sum money scores from a recorded result string (independent of the
    Rust code): declarer won v + 50, lost v - 50 and 40 per defender; 3 s - sum."""
    if not result.startswith("d:"):
        return [0.0, 0.0, 0.0]
    d, v = int(result.split()[0][2:]), int(re.search(r"v:(-?\d+)", result).group(1))
    s = [0.0] * 3 if v > 0 else [40.0] * 3
    s[d] = v + 50.0 if v > 0 else v - 50.0
    return [3 * x - sum(s) for x in s]


def _lines() -> List[str]:
    return [line for line in FIXTURE.read_text().splitlines() if line.strip()]


def _convert(gmdata, tmp: Path, lines: List[str], *extra) -> List[dict]:
    """The converter as a function of ISS record lines."""
    d = tmp / f"c{abs(hash(tuple(lines)))}"
    if d.exists():
        shutil.rmtree(d)
    (d / "raw").mkdir(parents=True)
    (d / "raw" / "g.sgf").write_text("\n".join(lines) + "\n")
    _run(gmdata, "skat", "convert", "--input", d / "raw", "--out", d / "out", "--allow-truncated",
         "--store-events", *extra)
    return load_gm_samples(str(d / "out")) if (d / "out").exists() else []


def _render(gmdata, tmp: Path, trajs: Dict[str, np.ndarray], players=("a", "b", "c")) -> List[str]:
    """gmrules trajectories → ISS record lines (`gmdata skat render`)."""
    d = tmp / f"r{abs(hash(tuple(trajs)))}"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "t.jsonl", "w") as fh:
        for k, ev in trajs.items():
            fh.write(json.dumps({"id": k, "events": np.asarray(ev).tolist(),
                                 "meta": {"players": list(players)}}) + "\n")
    _run(gmdata, "skat", "render", "--input", d / "t.jsonl", "--out", d / "g.sgf")
    return (d / "g.sgf").read_text().splitlines()


# ---------------------------------------------------------------------------
# converter
# ---------------------------------------------------------------------------

def test_every_fixture_record_verifies(gmdata):
    st = _run(gmdata, "skat", "verify", "--input", FIXTURE)
    assert (st["unparseable"], st["illegal"], st["result_mismatch"]) == (0, 0, 0), st["errors"]
    assert st["verified"] == st["records"] == len(_lines())
    for k in ("finished", "passed", "resigned", "aborted"):
        assert st[k] > 0, k
    assert st["jump_bids"] > 0
    # every declaration type of ISS occurs
    games = set(st["games"])
    assert {"D", "H", "S", "C", "G", "N", "NO", "NH", "NOH", "GH", "GHS", "GHZ", "GO",
            "CO"} <= games


def test_shards_schema_and_meta(shards):
    out, st = shards
    iss = st["iss"]
    assert (st["illegal"], st["returns_mismatch"], st["unparseable"]) == (0, 0, 0)
    assert st["converted"] == iss["finished"] + iss["passed"] + iss["resigned"]
    assert st["conceded"] == iss["resigned"] and st["samples"] == 3 * st["converted"]
    ds = GameShardDataset(str(out / "train"), game="skat")
    assert len(ds) == st["samples"] and ds.total_frames() == st["frames"]
    spec = gmsim.game_info("skat", {})["spec"]
    m = ds.shard_meta
    assert (m["schema"], m["layout"], m["n_seats"]) == ("candidate.v1", "per_seat", 3)
    for k, v in spec["meta"].items():
        assert m[k] == v, k
    s = ds[0]
    assert set(s["obs"]) == {"entities", "entity_mask", "globals", "actions", "action_mask"}
    assert set(s["tgt"]) == {"action_idx", "value"}
    assert s["obs"]["entities"].shape[1:] == (35, 49) and s["obs"]["actions"].shape[1:] == (12, 68)
    for k in ("entities", "globals", "actions"):
        v = s["obs"][k]
        assert v.dtype == torch.float32 and bool(((v == 0) | (v == 1)).all()), k   # bit-packed 0/1
    assert bool(s["obs"]["action_mask"].gather(1, s["tgt"]["action_idx"].unsqueeze(1)).all())


def test_converter_frames_are_the_rules_frames(shards):
    """Golden: every sample == the gmsim replay stream of its record; the value is the
    record's result (ISS money / 1260), also for games ended by resignations."""
    out, _ = shards
    samples = load_gm_samples(str(out / "train"))
    kinds = set()
    for s in samples:
        meta = s["meta"]
        r = gmsim.replay("skat", {}, meta["events"])
        seat = meta["seat"]
        idx = dict(r["streams"])[seat]
        for k in r["obs"]:
            assert np.array_equal(s["obs"][k].numpy(), r["obs"][k][idx]), k
        assert s["tgt"]["action_idx"].tolist() == r["action_idx"][idx].tolist()
        money = _money(meta["result"])
        want = money[seat] / MONEY_SCALE
        assert s["tgt"]["value"].numpy() == pytest.approx(np.full(len(idx), want), abs=1e-6)
        assert meta["returns"] == pytest.approx(money)
        kinds.add(meta["ending"])
        # resigned games stop before the rules' end; the others replay to it
        assert r["terminal"] == (meta["ending"] != "resigned")
    assert kinds == {"finished", "passed", "resigned"}


def test_simulator_frames_are_the_converter_frames(gmdata, tmp_path):
    """Golden IL = RL: play the gmsim env, write its episodes as ISS records (gmdata
    skat render), convert them — the shards' frames are exactly the env's frames."""
    n = 12
    env = GmSimVecEnv("skat", n, seed=0)
    env.reset(seed=5)
    gen = torch.Generator().manual_seed(1)
    obs = env.observe()
    seen = {r: [] for r in range(obs.n_rows)}
    while bool(obs.active.any()):
        for r in obs.active.nonzero().squeeze(1).tolist():
            assert int(obs.step_idx[r]) == len(seen[r])
            seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        obs, _, _, _ = env.step(env.random_actions(obs, generator=gen))
    lines = _render(gmdata, tmp_path, {str(e): env.trajectory(e) for e in range(n)})
    samples = _convert(gmdata, tmp_path, lines)
    assert len(samples) == 3 * n
    for s in samples:
        e, seat = int(s["meta"]["iss_id"]), s["meta"]["seat"]
        frames = seen[e * 3 + seat]
        assert s["obs"]["entities"].shape[0] == len(frames)
        for k in s["obs"]:
            ours = torch.stack([f[k] for f in frames])
            assert ours.dtype == s["obs"][k].dtype and torch.equal(ours, s["obs"][k]), (e, seat, k)
        # the stored trajectory is the env's
        assert s["meta"]["events"] == env.trajectory(e).tolist()


def test_render_roundtrip_of_real_records(gmdata, shards, tmp_path):
    """ISS record → trajectory → rendered ISS record → the same frames (finished games)."""
    out, _ = shards
    samples = [s for s in load_gm_samples(str(out / "train")) if s["meta"]["ending"] == "finished"]
    trajs = {s["meta"]["iss_id"]: np.asarray(s["meta"]["events"]) for s in samples[:60]}
    back = _convert(gmdata, tmp_path, _render(gmdata, tmp_path, trajs))
    orig = {(s["meta"]["iss_id"], s["meta"]["seat"]): s for s in samples}
    assert back
    for s in back:
        o = orig[(s["meta"]["iss_id"], s["meta"]["seat"])]
        for k in s["obs"]:
            assert torch.equal(s["obs"][k], o["obs"][k]), k
        assert s["meta"]["result"] == o["meta"]["result"]


# ---------------------------------------------------------------------------
# hidden information
# ---------------------------------------------------------------------------

def test_simulator_invariance_battery():
    st = invariance.check_sim_battery("skat", {}, n_games=30, seed=3)
    assert st["privacy"] > 20 and st["decisions"] > 300
    assert "privacy_checks" in gmsim.check_game("skat", {}, 300, 2)


def _cuts(line: str) -> List[str]:
    """The record cut after every player move (result unknown)."""
    m = re.search(r"MV\[(.*?)\]", line)
    toks = m.group(1).split()
    pairs = list(zip(toks[0::2], toks[1::2]))
    out = []
    for k in range(2, len(pairs)):
        if pairs[k - 1][0] == "w":
            continue
        mv = " ".join(f"{w} {a}" for w, a in pairs[:k])
        out.append(line[:m.start(1)] + mv + " " + line[m.end(1):].replace(
            re.search(r"R\[(.*?)\]", line).group(0), "R[unknown]"))
    return out


def _pick(pred, n=1) -> List[str]:
    """Converted fixture records (not ended by a timeout / a player leaving) matching."""
    got = [line for line in _lines()
           if pred(line) and " w TI." not in line and " w LE." not in line]
    assert len(got) >= n
    return got[:n]


def test_converter_causality(gmdata, tmp_path):
    """Every stream of a record cut after any move is a frame-prefix of the full
    record's stream (no encoder looks ahead), for pickup / hand / null ouvert /
    resigned games."""
    recs = (_pick(lambda r: re.search(r" \d [DHSCG]\.", r) and "R[d:" in r and " RE" not in r)
            + _pick(lambda r: re.search(r" \d [DHSCG]H ", r) is not None)
            + _pick(lambda r: re.search(r" \d NO\.", r) is not None)
            + _pick(lambda r: " RE ]" in r))
    for rec in recs:
        cuts = _cuts(rec)
        assert invariance.check_converter_causality(lambda ls: _convert(gmdata, tmp_path, ls), rec,
                                                    cuts[:: max(1, len(cuts) // 12)]) > 0


def test_converter_privacy(gmdata, shards, tmp_path):
    """A seat's frames do not change when the cards hidden from it are re-dealt
    (gmsim.perturb_hidden; the perturbed game rendered back to an ISS record and
    converted by the real binary) — for defenders and declarers, pickup and hand
    games; and the check does catch a leak (negative control)."""
    out, _ = shards
    samples = [s for s in load_gm_samples(str(out / "train")) if s["meta"]["ending"] == "finished"]
    by_game: Dict[str, np.ndarray] = {}
    for s in samples:
        by_game.setdefault(s["meta"]["iss_id"], np.asarray(s["meta"]["events"]))
    convert = lambda ls: _convert(gmdata, tmp_path, ls)  # noqa: E731
    checked = 0
    for gid, ev in list(by_game.items())[:25]:
        base = _render(gmdata, tmp_path, {gid: ev})[0]
        for seat in range(3):
            perts = []
            for seed in range(3):
                p = gmsim.perturb_hidden("skat", {}, ev, seat, seed)
                if p is not None:
                    perts.append(_render(gmdata, tmp_path, {gid: p})[0])
            if perts:
                checked += invariance.check_converter_privacy(convert, base, perts, seat)
    assert checked >= 10
    # negative control: re-dealing a seat's OWN cards changes its frames (a game passed
    # in — no card play, so any re-deal keeps the record legal)
    passed = next(s for s in load_gm_samples(str(out / "train")) if s["meta"]["ending"] == "passed")
    ev = np.asarray(passed["meta"]["events"])
    swapped = ev.copy()
    swapped[0, 1], swapped[10, 1] = ev[10, 1], ev[0, 1]     # forehand's 1st card <-> middlehand's
    base = _render(gmdata, tmp_path, {"base": ev})[0]
    leak = _render(gmdata, tmp_path, {"leak": swapped})[0]
    assert invariance.check_converter_privacy(convert, base, [base], 0) == 1
    with pytest.raises(AssertionError, match="converter privacy"):
        invariance.check_converter_privacy(convert, base, [leak], 0)


# ---------------------------------------------------------------------------
# quality filters
# ---------------------------------------------------------------------------

def test_quality_filters(gmdata, tmp_path):
    st = _run(gmdata, "skat", "stats", "--input", FIXTURE, "--out", tmp_path / "players.json")
    players = json.loads((tmp_path / "players.json").read_text())["players"]
    assert st["players"] == len(players) > 10
    # money is zero-sum over all players
    assert sum(p["money"] * p["games"] for p in players.values()) == pytest.approx(0.0, abs=1e-6)
    top = max((p for p in players.items() if p[1]["games"] >= 20), key=lambda x: x[1]["money"])[0]
    out = _run(gmdata, "skat", "convert", "--input", FIXTURE, "--out", tmp_path / "f",
               "--players", tmp_path / "players.json", "--min-games", "20",
               "--min-money", str(players[top]["money"] - 1e-9))
    for s in load_gm_samples(str(tmp_path / "f")):
        assert s["meta"]["players"][s["meta"]["seat"]].split(":")[0] == top
    assert out["filtered_out"] > 0
    # explicit players, and the ISS ratings page (program instances count as the base name)
    (tmp_path / "r.html").write_text(
        "<pre>  1 kermit       31216 Decl[ 33 87 13 ] Def[ 67 22 78 ] 30.79  16.77 1106.5  24.9"
        "\n</pre>")
    for extra in (["--player", "kermit1"],
                  ["--ratings", tmp_path / "r.html", "--min-rating", "1100"]):
        d = tmp_path / f"k{len(extra)}"
        _run(gmdata, "skat", "convert", "--input", FIXTURE, "--out", d, *extra)
        names = {s["meta"]["players"][s["meta"]["seat"]] for s in load_gm_samples(str(d))}
        assert names and all(n.startswith("kermit") for n in names), names
    with pytest.raises(subprocess.CalledProcessError):
        _run(gmdata, "skat", "convert", "--input", FIXTURE, "--out", tmp_path / "x", "--hq")


# ---------------------------------------------------------------------------
# model + training
# ---------------------------------------------------------------------------

def test_adapter_on_real_frames(shards):
    """The generic candidate adapter on skat frames: every labelled action is a legal
    candidate, the IL loss is finite and decomposes into its head terms."""
    from data_processing.games.dataset import collate_game_sequences
    from model.games import build_game_network
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS, batch_loss

    assert LABEL_KEYS["skat"] == () and "max_actions" in META_ARCH_KEYS["skat"]
    out, _ = shards
    samples = load_gm_samples(str(out / "train"))[:8]
    torch.manual_seed(0)
    net = build_game_network("skat", d_model=32, n_heads=4, d_ff=64, n_st_layers=1,
                             core_type="transformer", core_layers=1)
    assert net.obs_encoder.n_tokens == 35 + 1 + 12
    b = collate_game_sequences(samples)
    mb = {"obs": b["obs"], "tgt": b["tgt"], "frame_mask": b["meta"]["frame_mask"],
          "labels": None, "legal": None}
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), 0.2, 0.0, False)
    assert torch.isfinite(loss) and 0.0 <= stats["action_acc"] <= 1.0
    pred = net(b["obs"], frame_mask=b["meta"]["frame_mask"])
    terms = net.action_decoder.il_head_terms(pred, b["tgt"], b["meta"]["frame_mask"])
    full = net.action_decoder.il_loss(pred, b["tgt"], b["meta"]["frame_mask"])["loss"]
    t = terms["action"]
    assert torch.allclose(full, (t["ce"] * t["weight"]).sum() / t["norm"], atol=1e-6)


@pytest.fixture(scope="module")
def stage1(gmdata, shards, tmp_path_factory):
    from training import train_multi_game

    out, _ = shards
    root = tmp_path_factory.mktemp("stage1")
    _run(gmdata, "kuhn-poker", "generate", "--out", root / "raw" / "h.log", "--hands", "200",
         "--players", "2", "--seed", "3")
    _run(gmdata, "kuhn-poker", "convert", "--input", root / "raw", "--out", root / "kuhn")
    torch.manual_seed(0)
    res = train_multi_game.main([
        "--data", f"skat={out / 'train'}", "--data", f"kuhn_poker={root / 'kuhn'}",
        "--val", f"skat={out / 'train'}", "--eval_every", "2", "--eval_batches", "1",
        "--steps", "3", "--batch_size", "4", "--warmup_steps", "1", "--device", "cpu",
        "--save_dir", str(root / "ck"), "--override", *CORE])
    return root, res


def test_train_multi_game_with_skat(stage1):
    root, res = stage1
    ck = torch.load(root / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["games"] == ["skat", "kuhn_poker"]
    assert ck["configs"]["skat"]["cand_max_actions"] == 12
    assert ck["configs"]["skat"]["cand_global_features"] == 149
    for p in res["model"].parameters():
        assert torch.isfinite(p).all()


def test_train_rl_joint_with_skat(stage1, shards):
    from training import train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _ = stage1
    out, _ = shards
    args = ["--init", str(root / "ck" / "multi_final.pt"), "--rl", "skat", "--rl", "kuhn_poker",
            "--data", f"skat={out / 'train'}", "--replay_batch_size", "4",
            "--tokens_per_iter", "200", "--envs", "skat=4", "--envs", "kuhn_poker=4",
            "--epochs", "1", "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu",
            "--seed", "3"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(root / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"]["skat"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    for k in ("entropy", "kl", "value_loss", "adv_std"):
        assert np.isfinite(row[k]), k
    assert (root / "rl" / "rl_final.pt").exists()
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(root / "rl2")]))
    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {"skat": 120, "kuhn_poker": 40}, diag)
    streams = tr.rl["skat"].streams
    assert streams
    for s in streams:
        # a per-seat stream: its seat's frames only, the game's result on its last frame
        seat = int(s.seat[0])
        assert bool((s.seat == seat).all())
        assert bool(s.done[-1]) and not bool(s.done[:-1].any())
    tr._recompute("skat")
    for s in tr.rl["skat"].streams:
        assert torch.allclose(s.lp_old, s.mu, atol=1e-4)
    with pytest.raises(SystemExit, match="actor stream"):
        JointRLTrainer(parse_args(args + ["--save_dir", str(root / "rl3"), "--actor", "fixed"]))


def test_fetch_script_is_valid_bash():
    script = ROOT / "scripts" / "fetch_skat.sh"
    assert script.exists()
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
