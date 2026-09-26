"""The CUSTOM-SCHEMA path of the gmrules framework end to end, on its reference game
breakthrough (a from-to board game, the pattern for chess variants / xiangqi):

    gmrules rules + fromto.v1 frames ─► gmdata rules random / convert ─► gm-shards
      ─► BoardEncoder / FromToDecoder (build_fromto_network) ─► train_multi_game
      ─► gmsim env (legal from / to_cond, action keys from_idx / to_idx)
      ─► train_rl_joint (FromToSpec, jointly with a candidate game)

- the spec, the registrations and the model built from the simulator's dims;
- env: interleaved rows, the exact legality tables, frames = replay frames, applied
  action keys = the replay's targets, a network's sampled (from, to) is always legal;
- converter: shards carry the custom obs / tgt tensors, bit-exactly the replay's;
- invariance battery (causality; perfect information: no privacy perturbation);
- IL 2 steps + RL 1 iteration with kuhn_poker in the same run: μ = π_old, the
  compact replay legality (from / to_of) of FromToSpec.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import gmsim
import numpy as np
import pytest
import torch

from data_processing.games.dataset import GameShardDataset, load_gm_samples
from env.vec_game import has_simulator, make_vec_env
from env.vec_gmsim import GmSimVecEnv
from model.games import build_game_network
from model.games.gmrules import rules_games
from tests import invariance

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "data_processing"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
CFG = {"rows": 6, "columns": 5}
N = 30
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent")
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]


def _run(*args) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True)
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


def test_spec_and_registrations():
    spec = gmsim.spec_of("breakthrough", CFG)
    assert spec["schema"] == "fromto.v1" and spec["candidate"] is None
    assert spec["layout"] == "interleaved" and not spec["hidden_info"]
    assert [(t["name"], t["shape"], t["dtype"], t["bits"]) for t in spec["obs"]] == [
        ("board_seq", [3, 6, 5], "f32", True), ("globals_seq", [2], "f32", False)]
    assert [t["name"] for t in spec["legal"]] == ["from", "to_cond"]
    assert [t["name"] for t in spec["target"]] == ["from_idx", "to_idx", "legal_from", "legal_to_of"]
    assert spec["action_keys"] == ["from_idx", "to_idx"]
    assert spec["meta"] == {"board_h": 6, "board_w": 5, "piece_channels": 3, "global_features": 2}
    entry = rules_games()["breakthrough"]
    assert not entry.is_candidate and has_simulator("breakthrough")
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS
    from training.train_rl_joint import GAME_SPECS, FromToSpec
    assert LABEL_KEYS["breakthrough"] == ("from_idx", "to_idx")
    assert META_ARCH_KEYS["breakthrough"]["global_features"] == ("board_global_features",)
    assert isinstance(GAME_SPECS["breakthrough"], FromToSpec)
    net = build_game_network("breakthrough", **TINY)          # dims: the 8x8 sim default
    assert (net.cfg.board_h, net.cfg.board_w, net.cfg.piece_channels,
            net.cfg.board_global_features, net.cfg.n_promotions) == (8, 8, 3, 2, 0)
    assert net.obs_encoder.n_tokens == 64


def _play(env: GmSimVecEnv, gen: torch.Generator, net=None):
    """Play to the end (random legal actions, or the network's samples); returns the
    frames / legal / applied keys every row saw."""
    obs = env.observe()
    seen = {r: [] for r in range(obs.n_rows)}
    stepper = None
    if net is not None:
        from env.league import StreamStepper
        stepper = StreamStepper(net)
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1)
        # interleaved and lockstep: every live row at one frame index
        assert len(set(obs.step_idx[rows].tolist())) == 1
        if net is None:
            acts = env.random_actions(obs, generator=gen)
        else:
            restart = obs.stream_id[rows][obs.new_episode[rows] | (obs.step_idx[rows] == 0)]
            if restart.numel():
                stepper.reset(restart.tolist())
            with torch.no_grad():
                out = stepper.step({k: v[rows] for k, v in obs.obs.items()},
                                   {k: v[rows] for k, v in obs.legal.items()},
                                   obs.stream_id[rows].tolist(), obs.step_idx[rows].tolist())
            acts = {k: torch.zeros(obs.n_rows, dtype=torch.long) for k in ("action_from",
                                                                           "action_to")}
            acts["action_from"][rows] = out["action_from"].cpu()
            acts["action_to"][rows] = out["action_to"].cpu()
        prev = obs
        obs, rew, done, info = env.step(acts)
        for r in rows.tolist():
            seen[r].append(({k: v[r].clone() for k, v in prev.obs.items()},
                            {k: v[r].clone() for k, v in prev.legal.items()},
                            (int(info["applied"]["from_idx"][r]), int(info["applied"]["to_idx"][r]))))
        assert bool(info["acted"][rows].all())
    return seen


def test_env_frames_legality_and_replay():
    env = make_vec_env("breakthrough", 4, seed=0, **CFG)
    assert isinstance(env, GmSimVecEnv)
    assert (env.interleaved, env.lockstep, env.rows_per_env, env.action_keys) \
        == (True, True, 1, ("from_idx", "to_idx"))
    assert env.ACTION_KEYS == {"from_idx": "from_idx", "to_idx": "to_idx",
                               "action_from": "from_idx", "action_to": "to_idx"}
    obs = env.reset(seed=3)
    assert obs.obs["board_seq"].shape == (4, 3, 6, 5) and obs.obs["board_seq"].dtype == torch.float32
    assert obs.legal["to_cond"].shape == (4, N, N) and obs.legal["to_cond"].dtype == torch.bool
    # the from mask is exactly the rows of the to-table with a legal destination
    assert torch.equal(obs.legal["from"], obs.legal["to_cond"].any(-1))
    seen = _play(env, torch.Generator().manual_seed(0))
    for e in range(4):
        r = gmsim.replay("breakthrough", CFG, env.trajectory(e))
        assert r["terminal"] and sorted(r["outcome"]) == [-1.0, 1.0]
        (seat, idx), = r["streams"]
        assert seat == -1 and len(idx) == len(seen[e])
        for j, d in enumerate(idx):
            o, lg, keys = seen[e][j]
            for k in o:
                assert torch.equal(o[k], torch.from_numpy(r["obs"][k][d].copy())), (e, j, k)
            for k in lg:
                assert torch.equal(lg[k], torch.from_numpy(r["legal"][k][d].copy())), (e, j, k)
            assert keys == (int(r["target"]["from_idx"][d]), int(r["target"]["to_idx"][d]))
            # the IL masks are the legality tables at the taken from-square
            assert np.array_equal(r["target"]["legal_from"][d], r["legal"]["from"][d].astype(np.float32))
            assert np.array_equal(r["target"]["legal_to_of"][d],
                                  r["legal"]["to_cond"][d][keys[0]].astype(np.float32))
    # keys that are no legal move raise before anything is applied
    env.reset(seed=4)
    bad = {"from_idx": torch.zeros(4, dtype=torch.long), "to_idx": torch.full((4,), N - 1)}
    with pytest.raises(ValueError, match="no legal action"):
        env.step(bad)
    st = gmsim.State("breakthrough", CFG)
    a = st.legal_actions(0)[3]
    t = st.target(0, a)
    assert st.decode(0, [int(t["from_idx"]), int(t["to_idx"])]) == a


def test_network_samples_legal_moves():
    torch.manual_seed(0)
    net = build_game_network("breakthrough", **TINY, board_h=6, board_w=5).eval()
    env = GmSimVecEnv("breakthrough", 3, seed=1, **CFG)
    seen = _play(env, torch.Generator().manual_seed(0), net=net)
    assert all(len(v) > 0 for v in seen.values())
    assert bool(env.finished.all())


def test_invariance_battery():
    st = invariance.check_sim_battery("breakthrough", CFG, n_games=4, seed=1)
    assert st["decisions"] > 0 and st["privacy"] == 0
    assert "terminal: 20" in gmsim.check_game("breakthrough", {"rows": 5, "columns": 4}, 20, 3)


@pytest.fixture(scope="module")
def shards(gmdata, tmp_path_factory):
    root = tmp_path_factory.mktemp("bt")
    st = _run(gmdata, "rules", "random", "--game", "breakthrough", "--config", json.dumps(CFG),
              "--episodes", "24", "--out", root / "bt", "--store-events")
    assert st["converted"] == 24 and st["samples"] == 24
    _run(gmdata, "kuhn-poker", "generate", "--out", root / "raw" / "h.log", "--hands", "200")
    _run(gmdata, "kuhn-poker", "convert", "--input", root / "raw", "--out", root / "kuhn")
    return root, st


def test_converter_writes_the_custom_tensors(shards):
    root, st = shards
    ds = GameShardDataset(str(root / "bt"), game="breakthrough")
    m = ds.shard_meta
    assert (m["schema"], m["layout"], m["board_h"], m["board_w"], m["global_features"]) \
        == ("fromto.v1", "interleaved", 6, 5, 2)
    assert ds.total_frames() == st["frames"]
    s = ds[0]
    assert set(s["obs"]) == {"board_seq", "globals_seq"}
    assert set(s["tgt"]) == {"from_idx", "to_idx", "legal_from", "legal_to_of", "value"}
    assert s["obs"]["board_seq"].dtype == torch.float32 and s["obs"]["board_seq"].shape[1:] == (3, 6, 5)
    for smp in load_gm_samples(str(root / "bt"))[:6]:
        r = gmsim.replay("breakthrough", CFG, smp["meta"]["events"])
        for k in r["obs"]:
            assert np.array_equal(smp["obs"][k].numpy(), r["obs"][k]), k
        for k in r["target"]:
            assert np.array_equal(smp["tgt"][k].numpy(), r["target"][k]), k
        # interleaved: the value target is the MOVER's outcome, frame by frame
        want = [r["outcome"][s] for s in r["seat"]]
        assert smp["tgt"]["value"].tolist() == want


def test_il_then_rl_with_a_candidate_game(shards, tmp_path):
    from training import train_multi_game, train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _ = shards
    torch.manual_seed(0)
    train_multi_game.main([
        "--data", f"breakthrough={root / 'bt'}", "--data", f"kuhn_poker={root / 'kuhn'}",
        "--steps", "2", "--batch_size", "4", "--warmup_steps", "1", "--device", "cpu",
        "--il_legal_mask", "--save_dir", str(tmp_path / "ck"), "--override", *CORE])
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["configs"]["breakthrough"]["board_h"] == 6
    assert ck["configs"]["breakthrough"]["n_promotions"] == 0
    args = ["--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "breakthrough",
            "--rl", "kuhn_poker", "--tokens_per_iter", "80", "--envs", "breakthrough=3",
            "--envs", "kuhn_poker=4", "--env_kw", "breakthrough:rows=6",
            "--env_kw", "breakthrough:columns=5", "--mirror_prob", "breakthrough=0.5",
            "--snapshot_every", "1", "--epochs", "1", "--minibatches", "2",
            "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    for g in ("breakthrough", "kuhn_poker"):
        row = rec["games"][g]
        assert row["tokens"] >= 1 and row["episodes"] > 0
        assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
        for k in ("entropy", "kl", "value_loss"):
            assert np.isfinite(row[k]), (g, k)
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(tmp_path / "rl2")]))
    tr._rollouts(0, {"breakthrough": 60, "kuhn_poker": 40}, Diag(tr.games, tr.n_moe))
    for s in tr.rl["breakthrough"].streams:
        # FromToSpec stores the masks the rollout sampled under, [N] per frame
        assert set(s.legal) == {"from", "to_of"} and s.legal["to_of"].shape[1:] == (N,)
        assert bool(s.done[-1]) and s.rewards[:-1].abs().sum() == 0
    for g in tr.rl_games:
        tr._recompute(g)
        for s in tr.rl[g].streams:
            assert torch.allclose(s.lp_old, s.mu, atol=1e-4), g
    # an env variant the network was not built for is refused
    bad = ["breakthrough:rows=7" if a == "breakthrough:rows=6" else a for a in args]
    with pytest.raises(SystemExit, match="board_seq"):
        JointRLTrainer(parse_args(bad + ["--save_dir", str(tmp_path / "rl3")]))
