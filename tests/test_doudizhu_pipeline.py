"""Dou Dizhu end to end on REAL generated records (tests/fixtures/doudizhu/
douzero_selfplay.jsonl.gz: 150 games of the public DouZero ADP / WP agents from
scripts/generate_doudizhu.py, heuristic bidding):

    gmdata doudizhu convert (replays + verifies every record) ─► gm-shards
      ─► CandidateEncoder / CandidateDecoder ─► train_multi_game (with kuhn_poker)
      ─► gmsim env (env/vec_gmsim.py) ─► train_rl_joint (with kuhn_poker)

- converter: every record verified (0 illegal, 0 returns mismatches), shard meta =
  the simulator's FrameSpec, the --hq filter, teacher soft targets from DouZero's Q;
- golden: converter frames == gmsim.replay frames of every record; the RL env
  (``GmSimVecEnv.reset_from`` the record's deal, then stepped with the recorded
  actions) emits exactly the converter's frames, rewards and returns — and from
  every decision's prefix; the env's own random episodes, written as records and
  converted, give the env's frames (IL = RL);
- hidden information: converter causality (records cut after every play) and
  privacy (the other hands re-dealt, every recorded play kept legal) with
  tests/invariance.py, plus the simulator battery;
- the candidate adapter's IL contract on real batches (soft targets included); a
  train_multi_game smoke with kuhn_poker; a train_rl_joint iteration with
  kuhn_poker (per-seat streams, the auction in the env, shared peasant wins).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
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
FIXTURE = ROOT / "tests" / "fixtures" / "doudizhu" / "douzero_selfplay.jsonl.gz"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
GAME = "doudizhu"
CFG: Dict = {}
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1)


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


def _records() -> List[dict]:
    with gzip.open(FIXTURE, "rt") as fh:
        return [json.loads(line) for line in fh]


@pytest.fixture(scope="module")
def shards(gmdata, tmp_path_factory):
    root = tmp_path_factory.mktemp("ddz")
    st = _run(gmdata, "doudizhu", "convert", "--input", FIXTURE, "--out", root / "train",
              "--shard-size", "64", "--store-events")
    return root, st


def _convert(gmdata, tmp: Path, recs: List[dict], *extra) -> List[dict]:
    """The converter as a function of records (for the invariance checks)."""
    d = tmp / f"c{abs(hash(json.dumps(recs, sort_keys=True)))}"
    if d.exists():
        shutil.rmtree(d)
    (d / "raw").mkdir(parents=True)
    with open(d / "raw" / "r.jsonl", "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    _run(gmdata, "doudizhu", "convert", "--input", d / "raw", "--out", d / "out",
         "--allow-truncated", *extra)
    return load_gm_samples(str(d / "out")) if (d / "out").exists() else []


def in_auction(st: gmsim.State, seat: int) -> bool:
    return any(1 <= a <= 3 for a in st.legal_actions(seat))


def record_of(events, rec_id: str) -> dict:
    """A gmrules trajectory in the converter's record format (the inverse of its
    parser up to the canonical deal order)."""
    ev = gmsim.events_array(events).tolist()
    cards = [i - 51 for _, i in ev[1:52]]
    st = gmsim.State(GAME, CFG)
    for e in ev[:52]:
        st.apply(e)
    rec = {"id": rec_id, "hands": [sorted(cards[k] for k in range(51) if k % 3 == s)
                                   for s in range(3)],
           "kitty": sorted(set(range(54)) - set(cards)), "face_up": cards[ev[0][1]],
           "auction": [], "plays": []}
    for seat, a in ev[52:]:
        if in_auction(st, seat):
            rec["auction"].append([seat, a])
        else:
            name = st.action_string(seat, a)
            if "*" not in name:
                rec["plays"].append([seat, name])
        st.apply_action(seat, a)
    if st.turn()[0] == "terminal":
        rec["returns"] = st.returns()
    return rec


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

def test_registration_everywhere():
    from env.vec_game import has_simulator, make_vec_env
    from model.games.gmrules import CANDIDATE_META_ARCH, candidate_games
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS
    from training.train_rl_joint import GAME_SPECS, CandidateSpec

    assert GAME in candidate_games() and GAME in gmsim.games()
    assert LABEL_KEYS[GAME] == () and META_ARCH_KEYS[GAME] == CANDIDATE_META_ARCH
    assert isinstance(GAME_SPECS[GAME], CandidateSpec) and has_simulator(GAME)
    env = make_vec_env(GAME, 2, seed=0)
    assert isinstance(env, GmSimVecEnv)
    assert (env.n_seats, env.rows_per_env, env.layout, env.lockstep) == (3, 3, "per_seat", False)
    spec = gmsim.game_info(GAME)["spec"]
    assert spec["hidden_info"] and spec["meta"]["max_actions"] == 92
    assert spec["meta"]["max_entities"] == 20 and spec["meta"]["action_features"] == 48


# ---------------------------------------------------------------------------
# converter on the real records
# ---------------------------------------------------------------------------

def test_fixture_records_all_verify(shards, gmdata, tmp_path):
    root, st = shards
    recs = _records()
    assert len(recs) == 150 and st["records"] == st["converted"] == 150
    assert st["illegal"] == st["returns_mismatch"] == st["unparseable"] == 0
    assert st["samples"] == 450 and st["frames"] > 4000
    # the real records cover the rules: bids, bombs, the rocket, kickers, springs
    plays = [p for r in recs for _, p in r["plays"]]
    assert any(p == "(BWJ)(CJ)" for p in plays) and any("-" in p for p in plays)
    assert any(len(p) == 4 and len(set(p)) == 1 for p in plays)            # a bomb
    assert {r["agents"][0] for r in recs if r["plays"]} == {"douzero_adp", "douzero_wp"}
    assert {max(b for _, b in r["auction"]) for r in recs} == {1, 2, 3}
    assert all(r["meta"]["bidding"] == "heuristic_v1" for r in recs)
    # --hq keeps every seat (the default mix is ADP / WP only); a corrupted record is
    # rejected by the replay
    hq = _run(gmdata, "doudizhu", "convert", "--input", FIXTURE, "--out", tmp_path / "hq", "--hq")
    assert hq["samples"] == 450 and hq["dropped_seats"] == 0
    bad = dict(recs[0])
    bad["plays"] = [list(p) for p in bad["plays"]]
    bad["plays"][0][0] = (bad["plays"][0][0] + 1) % 3
    bad_ret = dict(recs[1], returns=[r * 2 for r in recs[1]["returns"]])
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "b.jsonl").write_text(json.dumps(bad) + "\n" + json.dumps(bad_ret) + "\n")
    st2 = _run(gmdata, "doudizhu", "convert", "--input", tmp_path / "bad", "--out", tmp_path / "x")
    assert (st2["illegal"], st2["returns_mismatch"], st2["converted"]) == (1, 1, 0)


def test_shards_schema_and_meta(shards):
    root, st = shards
    ds = GameShardDataset(str(root / "train"), game=GAME)
    assert len(ds) == st["samples"] and ds.total_frames() == st["frames"]
    spec = gmsim.game_info(GAME)["spec"]
    m = ds.shard_meta
    assert m["schema"] == "candidate.v1" and m["layout"] == "per_seat" and m["n_seats"] == 3
    assert m["config"] == {"rules": "standard", "scoring": "score", "score_scale": 24.0}
    for k, v in spec["meta"].items():
        assert m[k] == v, k
    s = ds[0]
    assert set(s["obs"]) == {"entities", "entity_mask", "globals", "actions", "action_mask"}
    assert set(s["tgt"]) == {"action_idx", "value"}
    assert s["obs"]["actions"].dtype == torch.float32
    assert s["obs"]["entity_mask"].dtype == torch.bool
    idx = s["tgt"]["action_idx"]
    assert bool(s["obs"]["action_mask"].gather(1, idx.unsqueeze(1)).all())
    assert s["tgt"]["value"].abs().max() <= 1.0


def test_teacher_soft_targets_from_douzero_q(gmdata, tmp_path):
    st = _run(gmdata, "doudizhu", "convert", "--input", FIXTURE, "--out", tmp_path / "soft",
              "--soft-k", "4", "--soft-temp", "douzero_wp=0.05", "--soft-temp", "douzero_adp=0.25")
    assert st["converted"] == 150 and st["soft_frames"] > 2000 and st["soft_illegal_mass"] == 0
    samples = load_gm_samples(str(tmp_path / "soft"))
    n = 0
    for s in samples:
        idx, val = s["tgt"]["action_probs_idx"], s["tgt"]["action_probs_val"]
        has = idx[:, 0] >= 0
        n += int(has.sum())
        # the K most probable candidates (the trainer renormalises the kept mass)
        mass = val[has].sum(-1)
        assert bool((mass > 0).all() and (mass <= 1 + 1e-5).all())
        assert bool((val[has][:, 1:] <= val[has][:, :-1] + 1e-7).all())
        # the teacher's candidates are legal ones of the frame
        legal = s["obs"]["action_mask"]
        for t in has.nonzero().squeeze(1).tolist():
            for i, v in zip(idx[t].tolist(), val[t].tolist()):
                if i >= 0:
                    assert bool(legal[t, i]) and v > 0
    assert n == st["soft_frames"]


def test_candidates_are_distinguishable(shards):
    """Within a frame every legal candidate has its own feature vector (the pointer
    head scores candidates by their features only: two identical rows would be one
    action to the policy) — on the real records and on random games."""
    root, _ = shards
    frames = [s["obs"] for s in load_gm_samples(str(root / "train"))]
    for seed in range(40):
        r = gmsim.replay(GAME, CFG, gmsim.random_trajectory(GAME, CFG, seed))
        frames.append({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in r["obs"].items()})
    n = 0
    for f in frames:
        for t in range(f["actions"].shape[0]):
            k = int(f["action_mask"][t].sum())
            rows = f["actions"][t, :k]
            assert torch.unique(rows, dim=0).shape[0] == k
            n += 1
    assert n > 6000


def test_converter_frames_are_the_rules_frames(shards):
    """Golden: every sample == the replay stream of its record (gmrules encode_frame)."""
    root, _ = shards
    for s in load_gm_samples(str(root / "train")):
        r = gmsim.replay(GAME, CFG, s["meta"]["events"])
        idx = dict(r["streams"])[s["meta"]["seat"]]
        for k in r["obs"]:
            assert np.array_equal(s["obs"][k].numpy(), r["obs"][k][idx]), k
        assert s["tgt"]["action_idx"].tolist() == r["action_idx"][idx].tolist()
        want = [r["outcome"][s["meta"]["seat"]]] * len(idx)
        assert s["tgt"]["value"].tolist() == pytest.approx(want)


def _samples_by_record(root) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}
    for s in load_gm_samples(str(root)):
        out.setdefault(s["meta"]["record"], {})[s["meta"]["seat"]] = s
    return out


def test_env_on_real_records_emits_the_converter_frames(shards):
    """Golden IL = RL on real records: the gmsim env started from each record's deal
    (``reset_from``) and stepped with the record's actions emits exactly the
    converter's frames (every row, every step), rewards and returns."""
    root, _ = shards
    by_rec = _samples_by_record(root / "train")
    recs = _records()
    events = [gmsim.events_array(next(iter(by_rec[r["id"]].values()))["meta"]["events"])
              for r in recs]
    env = GmSimVecEnv(GAME, len(recs), seed=0)
    obs = env.reset_from([ev[:52] for ev in events], seed=3)
    acts = [[(int(s), int(a)) for s, a in ev[52:]] for ev in events]
    pos = [0] * len(recs)
    seen = {(e, s): 0 for e in range(len(recs)) for s in range(3)}
    rewards = torch.zeros(len(recs), 3)
    returns = torch.zeros(len(recs), 3, dtype=torch.float64)
    steps = 0
    while bool(obs.active.any()):
        a = torch.zeros(obs.n_rows, dtype=torch.long)
        for row in obs.active.nonzero().squeeze(1).tolist():
            e, seat = divmod(row, 3)
            want_seat, want = acts[e][pos[e]]
            assert seat == want_seat and int(obs.to_play[row]) == seat
            s = by_rec[recs[e]["id"]][seat]
            t = seen[(e, seat)]
            assert int(obs.step_idx[row]) == t
            for k, v in obs.obs.items():
                assert torch.equal(v[row], s["obs"][k][t]), (e, seat, t, k)
            legal = env.legal_ids(row)
            a[row] = legal.index(want)
            assert int(a[row]) == int(s["tgt"]["action_idx"][t])
            seen[(e, seat)] += 1
            pos[e] += 1
        obs, rew, done, info = env.step({"action_idx": a})
        rewards += rew
        returns += torch.where(done.unsqueeze(-1), info["returns"], torch.zeros_like(returns))
        steps += 1
        assert steps < 400
    for e, r in enumerate(recs):
        assert pos[e] == len(acts[e])
        assert returns[e].tolist() == pytest.approx(r["returns"])
        for seat, s in by_rec[r["id"]].items():
            assert seen[(e, seat)] == len(s["tgt"]["action_idx"])
            assert float(rewards[e, seat]) == pytest.approx(float(s["tgt"]["value"][0]))


def test_env_from_every_decision_prefix(shards):
    """``reset_from`` the record prefix before any decision: the env row holds exactly
    that decision's converter frame (streams restart at the seat's stream position)."""
    root, _ = shards
    by_rec = _samples_by_record(root / "train")
    for r in _records()[:6]:
        ev = gmsim.events_array(next(iter(by_rec[r["id"]].values()))["meta"]["events"])
        rep = gmsim.replay(GAME, CFG, ev)
        n = rep["n"]
        env = GmSimVecEnv(GAME, n, seed=0)
        obs = env.reset_from([ev[:int(rep["event"][d])] for d in range(n)], seed=1)
        streams = dict(rep["streams"])
        for d in range(n):
            seat = int(rep["seat"][d])
            row = d * 3 + seat
            t = streams[seat].index(d)
            assert bool(obs.active[row]) and int(obs.step_idx[row]) == t
            s = by_rec[r["id"]][seat]
            for k, v in obs.obs.items():
                assert torch.equal(v[row], s["obs"][k][t]), (r["id"], d, k)


def test_simulator_episodes_convert_to_the_env_frames(gmdata, tmp_path):
    """The env's own random episodes, written as records and converted by gmdata: the
    shards hold exactly the frames the env emitted (IL = RL)."""
    env = GmSimVecEnv(GAME, 12, seed=0)
    obs = env.reset(seed=21)
    gen = torch.Generator().manual_seed(0)
    seen: Dict[int, list] = {r: [] for r in range(obs.n_rows)}
    while bool(obs.active.any()):
        for r in obs.active.nonzero().squeeze(1).tolist():
            seen[r].append({k: v[r].clone() for k, v in obs.obs.items()})
        obs, _, _, _ = env.step(env.random_actions(obs, generator=gen))
    recs = [record_of(env.trajectory(e), f"sim{e}") for e in range(12)]
    samples = _convert(gmdata, tmp_path, recs)
    assert len(samples) == sum(len({s for s, _ in r["auction"]} | {s for s, _ in r["plays"]})
                               for r in recs)
    for s in samples:
        e, seat = int(s["meta"]["record"][3:]), s["meta"]["seat"]
        frames = seen[e * 3 + seat]
        for k in s["obs"]:
            ours = torch.stack([f[k] for f in frames])
            assert ours.dtype == s["obs"][k].dtype and torch.equal(ours, s["obs"][k]), (e, seat, k)


# ---------------------------------------------------------------------------
# hidden information: causality + privacy
# ---------------------------------------------------------------------------

def test_simulator_invariance_battery():
    st = invariance.check_sim_battery(GAME, CFG, n_games=12, seed=4)
    assert st["privacy"] > 20 and st["decisions"] > 300
    assert "privacy_checks" in gmsim.check_game(GAME, CFG, 40, 2)
    gmsim.check_game(GAME, {"rules": "openspiel", "scoring": "wp"}, 20, 3)


def _truncations(rec: dict) -> List[dict]:
    """The record cut after every bid and every play (returns unknown)."""
    out = []
    for k in range(1, len(rec["auction"]) + 1):
        out.append(dict(rec, auction=rec["auction"][:k], plays=[], returns=None))
    for k in range(1, len(rec["plays"])):
        out.append(dict(rec, plays=rec["plays"][:k], returns=None))
    return out


def _redeals(rec: dict, events, seat: int, n: int) -> List[dict]:
    """The record with what ``seat`` never sees re-dealt (the other hands, and the
    kitty while hidden; every recorded play stays legal): gmrules' own perturbation
    of the converter's canonical events, rendered back as a record."""
    out = []
    for k in range(n):
        p = gmsim.perturb_hidden(GAME, CFG, events, seat, 100 + k)
        if p is not None:
            r = record_of(p, rec["id"])
            assert r["plays"] == rec["plays"] and r["auction"] == rec["auction"]
            out.append(r)
    return out


def test_converter_causality_and_privacy(gmdata, shards, tmp_path):
    root, _ = shards
    by_rec = _samples_by_record(root / "train")

    def convert(recs):
        return _convert(gmdata, tmp_path, recs)

    for rec in _records()[:4]:
        assert invariance.check_converter_causality(convert, rec, _truncations(rec)) > 0
        events = next(iter(by_rec[rec["id"]].values()))["meta"]["events"]
        for seat in range(3):
            perturbed = _redeals(rec, events, seat, 2)
            assert perturbed, "nothing to perturb"
            assert invariance.check_converter_privacy(convert, rec, perturbed, seat) > 0


def test_privacy_check_catches_a_leak(gmdata, tmp_path):
    """Negative control: a card of the seat's OWN hand swapped with another seat's
    (different rank) changes its frames, and the converter-level check notices."""
    rec = _records()[0]
    seat = rec["auction"][0][0]                     # the first bidder: it has a frame
    other = (seat + 1) % 3
    base = dict(rec, auction=rec["auction"][:1], plays=[], returns=None)
    hands = [list(h) for h in rec["hands"]]
    i = next(i for i, c in enumerate(hands[seat]) if c != rec["face_up"] and c < 52)
    j = next(j for j, c in enumerate(hands[other]) if c < 52 and c % 13 != hands[seat][i] % 13)
    hands[seat][i], hands[other][j] = hands[other][j], hands[seat][i]
    with pytest.raises(AssertionError, match="entities"):
        invariance.check_converter_privacy(lambda r: _convert(gmdata, tmp_path, r), base,
                                           [dict(base, hands=hands)], seat)


# ---------------------------------------------------------------------------
# model, IL, RL
# ---------------------------------------------------------------------------

def test_il_contract_on_real_batches(gmdata, tmp_path):
    from data_processing.games.dataset import collate_game_sequences
    from model.games import build_game_network
    from training.train_il_game import batch_loss

    _run(gmdata, "doudizhu", "convert", "--input", FIXTURE, "--out", tmp_path / "soft",
         "--soft-k", "4", "--soft-temp", "douzero_wp=0.05", "--soft-temp", "douzero_adp=0.25")
    samples = load_gm_samples(str(tmp_path / "soft"))
    torch.manual_seed(0)
    net = build_game_network(GAME, **TINY)
    spec = gmsim.game_info(GAME)["spec"]["meta"]
    assert net.cfg.cand_max_actions == spec["max_actions"] == 92
    assert net.obs_encoder.n_tokens == 20 + 1 + 92
    b = collate_game_sequences(samples[:6])
    mb = {"obs": b["obs"], "tgt": b["tgt"], "frame_mask": b["meta"]["frame_mask"],
          "labels": None, "legal": None}
    loss, stats, _ = batch_loss(net, mb, torch.device("cpu"), 0.2, 0.0, False)
    assert torch.isfinite(loss) and 0 < stats["action_teacher_frac"] < 1
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    # the head terms reproduce il_loss (hard labels)
    pred = net(b["obs"])
    fm = b["meta"]["frame_mask"]
    tgt = {"action_idx": b["tgt"]["action_idx"]}
    dec = net.action_decoder
    ref = dec.il_loss(pred, tgt, fm)["loss"]
    terms = dec.il_head_terms(pred, tgt, fm)["action"]
    assert torch.allclose(ref, (terms["ce"] * terms["weight"]).sum() / terms["norm"])
    # legal-only sampling on real frames
    out = dec.sample({k: v[:, :1] for k, v in pred.items()})
    am = b["obs"]["action_mask"][:, 0]
    assert bool(am.gather(1, out["action_idx"].view(-1, 1)).all())


@pytest.fixture(scope="module")
def stage1(gmdata, shards, tmp_path_factory):
    from training import train_multi_game

    root, _ = shards
    out = tmp_path_factory.mktemp("ddz_stage1")
    _run(gmdata, "kuhn-poker", "generate", "--out", out / "kraw" / "h.log", "--hands", "200",
         "--seed", "1")
    _run(gmdata, "kuhn-poker", "convert", "--input", out / "kraw", "--out", out / "kuhn")
    torch.manual_seed(0)
    res = train_multi_game.main([
        "--data", f"doudizhu={root / 'train'}", "--data", f"kuhn_poker={out / 'kuhn'}",
        "--val", f"doudizhu={root / 'train'}", "--eval_every", "2", "--eval_batches", "1",
        "--steps", "2", "--batch_size", "4", "--warmup_steps", "1", "--device", "cpu",
        "--save_dir", str(out / "ck"), "--override", *CORE])
    return out, res


def test_train_multi_game_with_kuhn(stage1):
    out, res = stage1
    ck = torch.load(out / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["games"] == ["doudizhu", "kuhn_poker"]
    assert ck["configs"]["doudizhu"]["cand_max_actions"] == 92
    assert ck["configs"]["doudizhu"]["cand_entity_features"] == 43
    for p in res["model"].parameters():
        assert torch.isfinite(p).all()


def test_train_rl_joint_with_kuhn(stage1):
    from training import train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    out, _ = stage1
    args = ["--init", str(out / "ck" / "multi_final.pt"), "--rl", "doudizhu", "--rl", "kuhn_poker",
            "--tokens_per_iter", "160", "--envs", "doudizhu=4", "--envs", "kuhn_poker=4",
            "--mirror_prob", "doudizhu=0.5", "--snapshot_every", "1", "--epochs", "1",
            "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "3"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(out / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    for g in ("doudizhu", "kuhn_poker"):
        row = rec["games"][g]
        assert row["tokens"] >= 1 and row["episodes"] > 0
        assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
        for k in ("entropy", "kl", "value_loss", "adv_std"):
            assert np.isfinite(row[k]), (g, k)
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(out / "rl2")]))
    diag = Diag(tr.games, tr.n_moe)
    tr._rollouts(0, {"doudizhu": 400, "kuhn_poker": 40}, diag)
    auction = 0
    for s in tr.rl["doudizhu"].streams:
        # per-seat stream: the seat's own frames only, its terminal reward on its LAST
        # frame (also when another seat ended the game)
        seat = int(s.seat[0])
        assert bool((s.seat == seat).all()) and bool(s.done[-1]) and not bool(s.done[:-1].any())
        assert float(s.rewards[:-1].abs().sum()) == 0.0
        # the auction is part of the RL game: bid candidates (type bit 1) appear
        auction += int((s.obs["actions"][..., 1] > 0).any())
    assert auction > 0
    for g in tr.rl_games:
        tr._recompute(g)
        for s in tr.rl[g].streams:
            assert torch.allclose(s.lp_old, s.mu, atol=1e-4)


def test_league_scores_peasant_wins_as_team_wins():
    """Learner vs a random opponent on one seat: every decided game is a win or a
    loss of the learner's SIDE (shared peasant wins included), never a draw; only
    games nobody bid on (all returns 0) are draws."""
    from env.league import League, RandomPolicy, play_episodes

    env = GmSimVecEnv(GAME, 24, seed=0)
    league = League(env, {"rand": RandomPolicy(env, seed=1)}, mirror_prob=0.0, seed=2)
    res = play_episodes(env, league, RandomPolicy(env, seed=3))
    seats = res["learner_seats"]
    for e in range(env.num_envs):
        r = gmsim.replay(GAME, CFG, env.trajectory(e))["returns"]
        mine = r[int(seats[e].nonzero()[0])]
        want = 1.0 if mine > 0 else 0.0 if mine < 0 else 0.5
        assert float(res["score"][e]) == want, (e, r, seats[e])
    assert set(res["score"].tolist()) <= {0.0, 0.5, 1.0} and 1.0 in res["score"].tolist()


def test_fetch_scripts_are_valid():
    for script in ("fetch_doudizhu.sh",):
        assert subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)]).returncode == 0
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "generate_doudizhu.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "--agents" in out.stdout
