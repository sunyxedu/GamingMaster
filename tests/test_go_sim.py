"""Go as a SIMULATED game (gmrules ``go``, model/games/gmrules/go.py) for the existing go
adapter — the IL side is unchanged:

    KGS SGF ─► gmdata go convert (the gmrules go rules and frame encoders, converter v2)
      ─► gm-shards ─► BoardEncoder / PlacementDecoder (pass-gated) ─► train_multi_game
      ─► gmsim env (GmSimVecEnv) ─► train_rl_joint (GoSpec)

- spec + registrations: the simulator's frame schema is the v2 shards' (obs, action
  keys, meta), the IL tables keep their go entries, the env / RL spec exist;
- GOLDEN: the RL env with each record's own game config (komi, area scoring — read from
  its first frame) stepped with the record's action keys reproduces every converter
  frame of the committed KGS 4d+ sample bit for bit, and every recorded move is in the
  env's legal set;
- a network's sampled placement or pass is always legal; two passes end a game, which
  is scored by area;
- causality of the simulator's frames on random playouts;
- train_multi_game on the converted shards, then train_rl_joint on go: finite losses,
  μ = π_old (GoSpec, routing replay on).

The Rust side pins rules and parity (data_processing/rs/gmrules/src/games/go/tests.rs:
captures, suicide, ko, superko ends, Tromp-Taylor scores; the converter goldens of
tests/test_rust_converters_parity.py — byte-identical before / after the rules moved
into gmrules, incl. the hand-written 9x9 / 13x13 records of tests/fixtures/go_sim).
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
SGF = ROOT / "tests" / "fixtures" / "go"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent",
            board_global_features=6)
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]


def _run(*args) -> dict:
    res = subprocess.run([str(a) for a in args], capture_output=True, text=True, check=True)
    return json.loads(res.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def shards(tmp_path_factory):
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "gmdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    gmdata = WS / "target" / "release" / "gmdata"
    root = tmp_path_factory.mktemp("go")
    st = _run(gmdata, "go", "convert", "--input", SGF, "--out", root / "shards")   # even games
    return root / "shards", st


def _record_config(sample) -> dict:
    """The gmrules go config of an even record, from its first (Black's) frame's globals:
    komi (mover's view / 20, clipped), komi known, area scoring, handicap."""
    g = sample["obs"]["globals_seq"][0]
    assert float(g[0]) == 1.0 and float(g[4]) == 0.0        # Black moves first, no handicap
    komi = -20.0 * float(g[1]) if float(g[2]) else None
    return {"board_size": int(sample["obs"]["board_seq"].shape[-1]), "komi": komi,
            "area_scoring": bool(float(g[3]))}


# ---------------------------------------------------------------------------
# spec, registrations
# ---------------------------------------------------------------------------

def test_spec_and_registrations():
    spec = gmsim.spec_of("go", {})
    assert spec["schema"] == "go.v2" and spec["candidate"] is None
    assert spec["layout"] == "interleaved" and not spec["hidden_info"]
    assert [(t["name"], t["shape"], t["dtype"]) for t in spec["obs"]] == [
        ("board_seq", [4, 19, 19], "f32"), ("globals_seq", [6], "f32")]
    assert [(t["name"], t["shape"]) for t in spec["legal"]] == [("place", [361]), ("pass", [])]
    assert spec["action_keys"] == ["place_idx", "passed"]
    assert spec["meta"] == {"board_size": 19, "global_features": 6}
    assert gmsim.spec_of("go", {"board_size": 9})["meta"]["board_size"] == 9
    entry = rules_games()["go"]
    assert not entry.is_candidate and has_simulator("go")
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS, META_LEGACY
    from training.train_rl_joint import GAME_SPECS, GoSpec
    assert LABEL_KEYS["go"] == ()
    assert META_ARCH_KEYS["go"] == {"global_features": ("board_global_features",),
                                    "board_size": ("board_h", "board_w")}
    assert META_LEGACY["go"] == {"global_features": 0}
    assert isinstance(GAME_SPECS["go"], GoSpec)
    net = build_game_network("go", **TINY)
    assert (net.cfg.board_h, net.cfg.piece_channels, net.cfg.board_global_features) == (19, 4, 6)
    GAME_SPECS["go"].check_env(make_vec_env("go", 2), net)
    small = build_game_network("go", **TINY, board_h=9, board_w=9)
    env9 = make_vec_env("go", 2, **GAME_SPECS["go"].env_kwargs(small.cfg))   # the net's board
    assert tuple(env9.obs_schema()["board_seq"][0]) == (4, 9, 9)
    GAME_SPECS["go"].check_env(env9, small)
    with pytest.raises(SystemExit, match="board_seq"):
        GAME_SPECS["go"].check_env(make_vec_env("go", 1), small)


# ---------------------------------------------------------------------------
# GOLDEN: env frames == converter frames
# ---------------------------------------------------------------------------

def test_env_reproduces_the_converter_frames(shards):
    """Each record in its own one-row env (its komi / rules): stepped with the record's
    own keys, every observation equals the converter's frame bit for bit, every
    recorded move is legal in the env, and the applied keys are the record's."""
    root, st = shards
    samples = load_gm_samples(str(root))
    assert len(samples) == st["converted"] >= 20 and st["illegal_move"] == 0
    compared, passes = 0, 0
    configs = set()
    for s in samples:
        cfg = _record_config(s)
        configs.add((cfg["komi"], cfg["area_scoring"]))
        T = s["obs"]["board_seq"].shape[0]
        env = GmSimVecEnv("go", 1, seed=0, max_turns=T + 5,
                          config={**cfg, "superko": False, "max_moves": 10 * T})
        obs = env.reset(seed=0)
        for t in range(T):
            assert bool(obs.active[0]), t
            for k in ("board_seq", "globals_seq"):
                assert torch.equal(obs.obs[k][0], s["obs"][k][t]), (t, k)
            place, passed = int(s["tgt"]["place_idx"][t]), bool(s["tgt"]["passed"][t])
            if not passed:
                assert bool(obs.legal["place"][0, place]), t        # the move is legal here
            assert bool(obs.legal["pass"][0])
            acts = {"place_idx": torch.tensor([place]), "passed": torch.tensor([passed])}
            obs, _, done, info = env.step(acts)
            assert int(info["applied"]["passed"][0]) == int(passed)
            if not passed:
                assert int(info["applied"]["place_idx"][0]) == place
            passes += passed
            compared += 1
    assert compared == sum(s["obs"]["board_seq"].shape[0] for s in samples)
    assert len(configs) >= 2                  # 7.5 area and 6.5 territory records at least


def test_shard_meta_and_dataset(shards):
    root, st = shards
    ds = GameShardDataset(str(root), game="go")
    assert ds.converter_version == 2 and ds.shard_meta == {"board_size": 19,
                                                           "global_features": 6}
    assert len(ds) == st["converted"]
    assert {"place_idx", "passed", "value"} <= set(ds[0]["tgt"])


# ---------------------------------------------------------------------------
# play, invariance
# ---------------------------------------------------------------------------

def test_network_samples_legal_moves_and_passes_end_games():
    from env.league import StreamStepper

    torch.manual_seed(0)
    net = build_game_network("go", **TINY, board_h=9, board_w=9).eval()
    with torch.no_grad():                       # a pass-happy policy: games end by passes
        net.action_decoder.pass_head.net[-1].bias.fill_(1.5)
    env = GmSimVecEnv("go", 4, seed=1, max_turns=200, board_size=9)
    stepper = StreamStepper(net)
    obs = env.observe()
    ended = 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1)
        restart = obs.stream_id[rows][obs.new_episode[rows] | (obs.step_idx[rows] == 0)]
        if restart.numel():
            stepper.reset(restart.tolist())
        with torch.no_grad():
            out = stepper.step({k: v[rows] for k, v in obs.obs.items()},
                               {k: v[rows] for k, v in obs.legal.items()},
                               obs.stream_id[rows].tolist(), obs.step_idx[rows].tolist())
        place, did_pass = out["action_place"].cpu(), out["action_pass"].cpu()
        for j, r in enumerate(rows.tolist()):
            if not bool(did_pass[j]):
                assert bool(obs.legal["place"][r, place[j]])
        acts = {"action_place": torch.zeros(obs.n_rows, dtype=torch.long),
                "action_pass": torch.zeros(obs.n_rows, dtype=torch.bool)}
        acts["action_place"][rows], acts["action_pass"][rows] = place, did_pass.bool()
        obs, rew, done, info = env.step(acts)
        for r in done.nonzero().squeeze(1).tolist():
            assert not bool(info["truncated"][r])
            assert rew[r].abs().sum() > 0 or float(rew[r].sum()) == 0.0   # scored
            ended += 1
    assert ended == 4


def test_simulator_causality_on_random_playouts():
    for seed in (1, 2):
        ev = gmsim.random_trajectory("go", {"board_size": 9}, seed)[:80]
        assert invariance.check_sim_causality("go", {"board_size": 9}, ev) == len(ev)
    ev = gmsim.random_trajectory("go", {"board_size": 9}, 7)
    assert all(gmsim.perturb_hidden("go", {"board_size": 9}, ev, seat, 0) is None
               for seat in (0, 1))


# ---------------------------------------------------------------------------
# IL then RL
# ---------------------------------------------------------------------------

def test_train_multi_game_then_rl_joint(shards, tmp_path):
    from training import train_multi_game, train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _ = shards
    torch.manual_seed(0)
    out = train_multi_game.main([
        "--data", f"go={root}", "--val", f"go={root}", "--steps", "3",
        "--batch_size", "2", "--max_len", "32", "--val_max_len", "32", "--warmup_steps", "1",
        "--eval_every", "3", "--device", "cpu",
        "--save_dir", str(tmp_path / "ck"), "--override", *CORE])
    assert all(torch.isfinite(p).all() for p in out["model"].parameters())
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["configs"]["go"]["board_global_features"] == 6         # from the shard meta
    args = ["--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "go",
            "--tokens_per_iter", "120", "--envs", "go=3", "--env_kw", "go:max_turns=40",
            "--mirror_prob", "go=0.5", "--snapshot_every", "1", "--epochs", "1",
            "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"]["go"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    for k in ("entropy", "kl", "value_loss"):
        assert np.isfinite(row[k]), k
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(tmp_path / "rl2")]))
    tr._rollouts(0, {"go": 60}, Diag(tr.games, tr.n_moe))
    for s in tr.rl["go"].streams:
        assert s.routes is not None and s.routes.shape[0] == s.length   # routing replay
    tr._recompute("go")
    for s in tr.rl["go"].streams:
        assert torch.allclose(s.lp_old, s.mu, atol=1e-4)
