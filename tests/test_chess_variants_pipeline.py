"""Chess variants end to end on REAL lichess records (tests/fixtures/chess_variants:
49 games of the 2026-08 dumps, CC0 — every variant, drops, promotions, castling,
resignations / time forfeits / agreed draws and rules ends):

    lichess PGN ─► gmdata chess-variants convert (SAN resolved by shakmaty, every
      record replayed by the gmrules rules) ─► gm-shards (chessvar.v1)
      ─► ChessVariantsEncoder / FromToDecoder ─► train_multi_game
      ─► gmsim env (GmSimVecEnv) ─► train_rl_joint (FromToSpec)

- spec + registrations (model registry, LABEL_KEYS / META_ARCH_KEYS, env, RL spec);
- shard frames == gmsim.replay frames; value target = the mover's outcome;
- GOLDEN: the RL env (configured with the record's variant / chess960 start, without the
  RL-only automatic threefold draw) stepped with the records' own action keys
  reproduces every converter frame bit for bit (obs, the exact legality tables = the IL
  masks) and ends exactly where the rules end each record, with the outcome as reward;
- a network's sampled (from, to, promotion) — drops and promotions included — is always
  legal in the env;
- IL contract on real batches: the shards' IL masks make every target legal,
  il_head_terms decomposes il_loss (from / to / promotion heads);
- train_multi_game, then train_rl_joint: μ = π_old, the compact FromToSpec replay
  legality (from / to_of / promotion_of);
- scripts/fetch_chess_variants.sh offline (SKIP_FETCH): one dump per variant and month,
  per-variant filtering, resumable, relative paths from the caller's directory.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import (
    GameShardDataset,
    collate_game_sequences,
    load_gm_samples,
)
from env.vec_game import has_simulator
from env.vec_gmsim import GmSimVecEnv
from model.games import build_game_network
from model.games.gmrules import rules_games

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
REAL = ROOT / "tests" / "fixtures" / "chess_variants" / "lichess_variants_2026-08_sample.pgn"
SCRIPT = ROOT / "scripts" / "fetch_chess_variants.sh"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
N, P = 74, 6
VARIANTS = ["chess960", "crazyhouse", "atomic", "antichess", "threecheck", "kingofthehill",
            "horde", "racingkings"]
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent")
# no MoE here: a top-k routing near-tie (gaps of ~1e-7 occur on these frames) can flip
# between the rollout's batched stepping and the full-sequence replay (fp32 noise), a
# game-independent ~1e-3 log-prob jump that would make the exact μ = π_old check flaky
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1"]
CPU = torch.device("cpu")


def _run(*args, **kw) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True, **kw)
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def gmdata():
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    return str(WS / "target" / "release" / "gmdata")


@pytest.fixture(scope="module")
def shards(gmdata, tmp_path_factory):
    root = tmp_path_factory.mktemp("cv")
    st = _run(gmdata, "chess-variants", "convert", "--input", REAL, "--out", root / "all",
              "--store-events")
    return root, st


def _env_config(sample: Dict) -> Dict:
    """The env config that replays a record: its variant (and chess960 start), no
    automatic threefold draw (lichess: a claim; the converter keeps records past it)."""
    ev = sample["meta"]["events"]
    cfg = {"variants": [sample["meta"]["variant"]], "auto_threefold": False}
    if sample["meta"]["variant"] == "chess960":
        cfg["chess960_positions"] = [ev[1][1]]
    return cfg


# ---------------------------------------------------------------------------
# spec, registrations
# ---------------------------------------------------------------------------

def test_spec_and_registrations():
    spec = gmsim.spec_of("chess_variants", {})
    assert spec["schema"] == "chessvar.v1" and spec["candidate"] is None
    assert spec["layout"] == "interleaved" and not spec["hidden_info"]
    assert [(t["name"], t["shape"], t["bits"]) for t in spec["obs"]] == [
        ("board_seq", [17, 8, 8], True), ("hand_seq", [10, 8], True),
        ("globals_seq", [24], False)]
    assert [(t["name"], t["shape"]) for t in spec["legal"]] == [
        ("from", [N]), ("to_cond", [N, N]), ("promotion_cond", [N, N, P])]
    assert spec["action_keys"] == ["from_idx", "to_idx", "promotion"]
    assert spec["meta"]["variants"] == VARIANTS and spec["meta"]["n_tokens"] == N
    entry = rules_games()["chess_variants"]
    assert not entry.is_candidate and has_simulator("chess_variants")
    assert entry.sim_config == {"auto_threefold": True}
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS
    from training.train_rl_joint import GAME_SPECS, FromToSpec
    assert LABEL_KEYS["chess_variants"] == ("from_idx", "to_idx")
    assert META_ARCH_KEYS["chess_variants"]["hand_piece_types"] == ("n_hand_piece_types",)
    assert isinstance(GAME_SPECS["chess_variants"], FromToSpec)
    net = build_game_network("chess_variants", **TINY)        # dims: the simulator's spec
    c = net.cfg
    assert (c.board_h, c.board_w, c.piece_channels, c.board_global_features, c.n_promotions,
            c.n_hand_piece_types, c.hand_feature_dim) == (8, 8, 17, 24, P, 5, 8)
    assert net.obs_encoder.n_tokens == N


# ---------------------------------------------------------------------------
# the converter's shards on real records
# ---------------------------------------------------------------------------

def test_shards_are_the_rules_frames(shards):
    root, st = shards
    assert (st["records"], st["converted"], st["illegal"], st["returns_mismatch"],
            st["adjudicated"]) == (49, 49, 0, 0, 27)
    ds = GameShardDataset(str(root / "all"), game="chess_variants")
    assert ds.shard_meta["schema"] == "chessvar.v1" and ds.total_frames() == st["frames"]
    samples = load_gm_samples(str(root / "all"))
    assert {s["meta"]["variant"] for s in samples} == set(VARIANTS)
    for smp in samples:
        r = gmsim.replay("chess_variants", {}, smp["meta"]["events"])
        for k in r["obs"]:
            assert np.array_equal(smp["obs"][k].numpy(), r["obs"][k]), k
        for k in r["target"]:
            assert np.array_equal(smp["tgt"][k].numpy(), r["target"][k]), k
        # interleaved: the value target is the MOVER's outcome, frame by frame
        assert smp["tgt"]["value"].tolist() == [smp["meta"]["outcome"][s] for s in r["seat"]]
        assert r["terminal"] == (not smp["meta"].get("adjudicated", False))


# ---------------------------------------------------------------------------
# GOLDEN: env frames == converter frames on the real records
# ---------------------------------------------------------------------------

def test_env_reproduces_the_converter_frames_of_real_records(shards):
    root, _ = shards
    samples = load_gm_samples(str(root / "all"))
    compared = ended = drops = promotions = 0
    for i, s in enumerate(samples):
        T = s["obs"]["board_seq"].shape[0]
        env = GmSimVecEnv("chess_variants", 1, seed=i, max_turns=T + 5, config=_env_config(s))
        obs = env.reset(seed=i)
        for t in range(T):
            assert bool(obs.active[0]) and int(obs.step_idx[0]) == t, (i, t)
            for k in ("board_seq", "hand_seq", "globals_seq"):
                assert torch.equal(obs.obs[k][0], s["obs"][k][t]), (i, t, k)
            f, to, p = (int(s["tgt"][k][t]) for k in ("from_idx", "to_idx", "promotion"))
            assert torch.equal(obs.legal["from"][0].float(), s["tgt"]["legal_from"][t]), (i, t)
            assert torch.equal(obs.legal["to_cond"][0][f].float(), s["tgt"]["legal_to_of"][t])
            assert torch.equal(obs.legal["promotion_cond"][0][f][to].float(),
                               s["tgt"]["legal_promotion_of"][t])
            drops += f >= 64
            promotions += p > 0
            obs, rew, done, info = env.step({"from_idx": torch.tensor([f]),
                                             "to_idx": torch.tensor([to]),
                                             "promotion": torch.tensor([p])})
            assert (int(info["applied"]["from_idx"][0]), int(info["applied"]["to_idx"][0]),
                    int(info["applied"]["promotion"][0])) == (f, to, p)
            compared += 1
        terminal = not s["meta"].get("adjudicated", False)
        assert bool(done[0]) == terminal, s["meta"]["record"]
        if terminal:
            assert rew[0].tolist() == s["meta"]["outcome"], s["meta"]["record"]
            ended += 1
        assert [list(e) for e in env.trajectory(0).tolist()][:len(s["meta"]["events"])] \
            == s["meta"]["events"]
    assert compared == sum(s["obs"]["board_seq"].shape[0] for s in samples)
    assert ended == 22 and drops > 0 and promotions > 0


def test_rl_env_draws_at_the_first_threefold_repetition(shards):
    """With its RL setting (auto_threefold: lichess' claim, always made) the env plays a
    record exactly like the converter until the first threefold repetition, and ends
    there with a draw: the 7 fixture games drawn by a threefold claim end there with the
    recorded draw; 1 game went on past a threefold (its later IL frames are positions the
    RL env never reaches — the documented converter / env difference)."""
    root, _ = shards
    samples = load_gm_samples(str(root / "all"))
    repeated = claimed = 0
    for i, s in enumerate(samples):
        cfg = {k: v for k, v in _env_config(s).items() if k != "auto_threefold"}
        T = s["obs"]["board_seq"].shape[0]
        env = GmSimVecEnv("chess_variants", 1, seed=i, max_turns=T + 5, config=cfg)
        assert env.config["auto_threefold"] is True
        obs = env.reset(seed=i)
        for t in range(T):
            assert torch.equal(obs.obs["globals_seq"][0], s["obs"]["globals_seq"][t]), (i, t)
            keys = {k: s["tgt"][k][t:t + 1].clone() for k in ("from_idx", "to_idx", "promotion")}
            obs, rew, done, _ = env.step(keys)
            # globals_seq[11]: the position was seen >= 2 times before (a threefold)
            third = t + 1 < T and bool(s["obs"]["globals_seq"][t + 1][11] == 1.0)
            if bool(done[0]) and t + 1 < T:
                assert third and rew[0].tolist() == [0.0, 0.0], (s["meta"]["record"], t)
                repeated += 1
                break
            assert not third, (s["meta"]["record"], t)
            if bool(done[0]) and s["meta"].get("adjudicated", False):
                # the record ends with a threefold claim: the env draws right there
                assert rew[0].tolist() == s["meta"]["outcome"] == [0.0, 0.0], s["meta"]["record"]
                claimed += 1
    assert (repeated, claimed) == (1, 7)


def test_network_samples_legal_moves():
    from env.league import StreamStepper

    torch.manual_seed(0)
    net = build_game_network("chess_variants", **TINY).eval()
    # crazyhouse (drops) and antichess (forced captures, king promotions) among others
    env = GmSimVecEnv("chess_variants", 8, seed=2, max_turns=80)
    stepper = StreamStepper(net)
    obs = env.observe()
    steps = drops = 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1)
        restart = obs.stream_id[rows][obs.new_episode[rows] | (obs.step_idx[rows] == 0)]
        if restart.numel():
            stepper.reset(restart.tolist())
        with torch.no_grad():
            out = stepper.step({k: v[rows] for k, v in obs.obs.items()},
                               {k: v[rows] for k, v in obs.legal.items()},
                               obs.stream_id[rows].tolist(), obs.step_idx[rows].tolist())
        f, to, p = (out[k].cpu() for k in ("action_from", "action_to", "action_promotion"))
        for j, r in enumerate(rows.tolist()):       # every sampled (from, to, promotion) is legal
            assert bool(obs.legal["promotion_cond"][r, f[j], to[j], p[j]])
        drops += int((f >= 64).sum())
        acts = {k: torch.zeros(obs.n_rows, dtype=torch.long)
                for k in ("action_from", "action_to", "action_promotion")}
        for k, v in (("action_from", f), ("action_to", to), ("action_promotion", p)):
            acts[k][rows] = v
        obs, _, _, info = env.step(acts)
        assert bool(info["acted"][rows].all())
        steps += 1
    assert steps > 0 and bool(env.finished.all())


# ---------------------------------------------------------------------------
# IL contract, IL + RL
# ---------------------------------------------------------------------------

def test_il_contract_on_real_batches(shards):
    from training.train_il_game import check_meta_arch, prepare_batch

    root, _ = shards
    ds = GameShardDataset(str(root / "all"), game="chess_variants", max_len=64, crop="prefix")
    torch.manual_seed(0)
    net = build_game_network("chess_variants", **TINY)
    check_meta_arch("chess_variants", net.cfg, ds.shard_meta)
    batch = collate_game_sequences([ds[i] for i in range(len(ds))])
    mb = prepare_batch("chess_variants", batch, CPU, tf_prob=1.0, cfg=net.cfg, il_legal_mask=True)
    assert set(mb["legal"]) == {"from", "to_of", "promotion_of"}
    net.eval()
    pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"])
    dec = net.action_decoder
    mask = mb["frame_mask"]
    il = dec.il_loss(pred, mb["tgt"], mask)
    terms = dec.il_head_terms(pred, mb["tgt"], mask)
    assert set(terms) == {"from", "to", "promotion"}
    total = sum((t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    assert float(total.detach()) == pytest.approx(float(il["loss"].detach()), abs=1e-5)
    # the shards' IL masks make every recorded move legal — drops and promotions included
    assert float(il["illegal_target_frac"]) == 0.0
    assert int((mb["tgt"]["from_idx"][mask] >= 64).sum()) > 0
    assert int((mb["tgt"]["promotion"][mask] > 0).sum()) > 0
    assert torch.equal(pred["from_idx"][mask], mb["tgt"]["from_idx"][mask])


def test_train_multi_game_then_rl(shards, tmp_path):
    from training import train_multi_game, train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _ = shards
    torch.manual_seed(0)
    out = train_multi_game.main([
        "--data", f"chess_variants={root / 'all'}", "--steps", "2", "--batch_size", "4",
        "--max_len", "48", "--warmup_steps", "1", "--eval_every", "2", "--device", "cpu",
        "--il_legal_mask", "--save_dir", str(tmp_path / "ck"), "--override", *CORE])
    assert all(torch.isfinite(p).all() for p in out["model"].parameters())
    args = ["--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "chess_variants",
            "--tokens_per_iter", "160", "--envs", "chess_variants=4",
            "--env_kw", "chess_variants:max_turns=50", "--snapshot_every", "1", "--epochs", "1",
            "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"]["chess_variants"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(tmp_path / "rl2")]))
    tr._rollouts(0, {"chess_variants": 100}, Diag(tr.games, tr.n_moe))
    for s in tr.rl["chess_variants"].streams:
        # FromToSpec stores the masks the rollout sampled under, [N] / [P] per frame
        assert set(s.legal) == {"from", "to_of", "promotion_of"}
        assert s.legal["to_of"].shape[1:] == (N,) and s.legal["promotion_of"].shape[1:] == (P,)
    tr._recompute("chess_variants")
    for s in tr.rl["chess_variants"].streams:
        assert torch.allclose(s.lp_old, s.mu, atol=1e-4)


# ---------------------------------------------------------------------------
# the fetch script (offline)
# ---------------------------------------------------------------------------

def test_fetch_script_offline(gmdata, tmp_path):
    """scripts/fetch_chess_variants.sh with SKIP_FETCH=1 on dumps already in RAW (the
    fixture, which mixes all eight variants, stands in for each variant's dump): each
    (variant, month) keeps only its variant, the stats land in logs/, a second run skips
    what is done, and relative directories are the caller's."""
    assert os.access(SCRIPT, os.X_OK)
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    raw = tmp_path / "raw"
    raw.mkdir()
    for name in ("atomic_rated_2026-08", "racingKings_rated_2026-08", "atomic_rated_2026-07"):
        shutil.copy(REAL, raw / f"lichess_db_{name}.head.pgn")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = dict(os.environ, SKIP_FETCH="1", MAX_BYTES="1", HQ="0", VARIANTS="atomic racingkings",
               MONTHS="2026-08", VAL_MONTHS="2026-07", RAW="../raw", GMDATA=gmdata)
    env.pop("KEEP_RAW", None)
    run = subprocess.run(["bash", str(SCRIPT), "out"], cwd=cwd, env=env, capture_output=True,
                         text=True)
    # the val month has no racing kings dump: the script stops there, after the rest
    assert run.returncode != 0 and "no dump" in run.stderr
    out = cwd / "out"
    logs = {p.name: json.loads(p.read_text()) for p in (out / "logs").glob("*.json")}
    assert set(logs) == {"train_atomic_2026-08.json", "train_racingkings_2026-08.json",
                         "val_atomic_2026-07.json"}
    assert logs["train_atomic_2026-08.json"]["converted"] == 8
    assert logs["train_racingkings_2026-08.json"]["converted"] == 6
    assert all(v["illegal"] == 0 and v["returns_mismatch"] == 0 for v in logs.values())
    for v, n in (("atomic", 8), ("racingkings", 6)):
        samples = load_gm_samples(str(out / "train" / v / "2026-08"))
        assert len(samples) == n and {s["meta"]["variant"] for s in samples} == {v}
    assert sorted(p.name for p in raw.iterdir()) == sorted(
        f"lichess_db_{n}.head.pgn" for n in ("atomic_rated_2026-08", "racingKings_rated_2026-08",
                                             "atomic_rated_2026-07"))   # SKIP_FETCH: kept
    # resumable: done work is skipped, nothing is appended
    run = subprocess.run(["bash", str(SCRIPT), "out"], cwd=cwd, env={**env, "VAL_MONTHS": ""},
                         capture_output=True, text=True, check=True)
    assert run.stdout.count("have train") == 2
    shard_dirs = [d.name for d in (out / "train" / "atomic" / "2026-08").iterdir()]
    assert shard_dirs == ["chess_variants_00000"]
    assert not (ROOT / "out").exists()
