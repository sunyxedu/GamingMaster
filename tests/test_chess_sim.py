"""Standard chess as a SIMULATED game (gmrules ``chess``, model/games/gmrules/chess.py)
for the existing chess adapter — the IL side is unchanged:

    PGN ─► gmdata chess convert (frames through the gmrules chess game, converter v3)
      ─► gm-shards ─► BoardEncoder / FromToDecoder (the chess adapter) ─► train_multi_game
      ─► gmsim env (GmSimVecEnv) ─► train_rl_joint (FromToSpec, promotion head)

- spec + registrations: the simulator's frame schema is the v3 shards' (obs, legality,
  action keys, meta), the IL tables keep their chess entries, the env / RL spec exist;
- GOLDEN: the RL env stepped with the records' own action keys (the shard targets:
  from / to / promotion) reproduces every converter frame of the edge-case fixture bit
  for bit — observations, the legality tables (= the IL masks), the applied keys — and
  ends where the rules end a record (checkmate: the outcome as reward);
- a network's sampled (from, to, promotion) is always legal in the env;
- causality of the simulator's frames on random playouts;
- train_multi_game on the converted shards, then train_rl_joint on chess: finite losses,
  μ = π_old (routing replay on), the promotion replay legality.

The Rust side pins rules and parity in depth (data_processing/rs/gmrules/src/games/chess/
tests.rs: perft through the action enumeration and the keys, castling / en passant /
promotions, automatic draws; data_processing/rs/gmdata/src/chess/parity.rs: converter
frames = the pre-refactor v3 reference frames, simulator steps = converter frames).
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
PGN = ROOT / "tests" / "fixtures" / "chess" / "edge_cases.pgn"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent",
            board_global_features=10)
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
    root = tmp_path_factory.mktemp("chess")
    st = _run(gmdata, "chess", "convert", "--input", PGN, "--out", root / "shards")
    return root / "shards", st


# ---------------------------------------------------------------------------
# spec, registrations
# ---------------------------------------------------------------------------

def test_spec_and_registrations():
    spec = gmsim.spec_of("chess", {})
    assert spec["schema"] == "chess.v3" and spec["candidate"] is None
    assert spec["layout"] == "interleaved" and not spec["hidden_info"]
    assert [(t["name"], t["shape"], t["dtype"]) for t in spec["obs"]] == [
        ("board_seq", [14, 8, 8], "f32"), ("globals_seq", [10], "f32")]
    assert [(t["name"], t["shape"]) for t in spec["legal"]] == [
        ("from", [64]), ("to_cond", [64, 64]), ("promotion_cond", [64, 64, 5])]
    assert [t["name"] for t in spec["target"]] == [
        "from_idx", "to_idx", "promotion", "legal_from", "legal_to_of", "legal_promotion_of"]
    assert spec["action_keys"] == ["from_idx", "to_idx", "promotion"]
    assert spec["meta"] == {"global_features": 10}          # the v3 shards' meta
    entry = rules_games()["chess"]
    assert not entry.is_candidate and has_simulator("chess")
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS, META_LEGACY
    from training.train_rl_joint import GAME_SPECS, FromToSpec
    # the IL tables keep their own chess entries (registration never overrides them)
    assert LABEL_KEYS["chess"] == ("from_idx", "to_idx")
    assert META_ARCH_KEYS["chess"] == {"global_features": ("board_global_features",)}
    assert META_LEGACY["chess"] == {"global_features": 8}
    assert isinstance(GAME_SPECS["chess"], FromToSpec)
    net = build_game_network("chess", **TINY)
    assert (net.cfg.board_h, net.cfg.board_w, net.cfg.piece_channels,
            net.cfg.board_global_features, net.cfg.n_promotions) == (8, 8, 14, 10, 5)
    assert net.obs_encoder.n_tokens == 64 and net.action_decoder.promotion_head is not None
    env = make_vec_env("chess", 2)
    assert isinstance(env, GmSimVecEnv)
    GAME_SPECS["chess"].check_env(env, net)
    with pytest.raises(SystemExit, match="board_seq"):       # the legacy 8-global width
        GAME_SPECS["chess"].check_env(env, build_game_network("chess", **{
            **TINY, "board_global_features": 8}))


# ---------------------------------------------------------------------------
# GOLDEN: env frames == converter frames
# ---------------------------------------------------------------------------

#: the converter's replay config (gmrules ChessConfig::records): records continue past
#: threefold repetitions, 50-move clocks and dead material, only mate / stalemate end
RECORDS = {"threefold_draw": False, "fifty_move_draw": False,
           "insufficient_material_draw": False}


def test_env_reproduces_the_converter_frames(shards):
    """One GmSimVecEnv row per record (the records' config), stepped with the record's
    own action keys: every observation and legality table equals the converter's frame
    and IL masks bit for bit, the applied keys are the record's; a record that ends by
    the rules (mate, stalemate) ends the row there with its outcome as reward."""
    root, st = shards
    samples = load_gm_samples(str(root))
    n = len(samples)
    assert n == st["kept"] == st["seen"] >= 8 and st["mask_mismatch"] == 0
    lengths = [s["obs"]["board_seq"].shape[0] for s in samples]
    env = GmSimVecEnv("chess", n, seed=0, max_turns=max(lengths) + 5, threads=2,
                      config=RECORDS)
    obs = env.reset(seed=0)
    gen = torch.Generator().manual_seed(0)
    compared, t, promotions, ended = 0, 0, 0, 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1).tolist()
        assert set(obs.step_idx[rows].tolist()) == {t}          # interleaved, lockstep
        acts = env.random_actions(obs, generator=gen)
        for r in rows:
            s = samples[r]
            if t >= lengths[r]:
                continue
            for k in ("board_seq", "globals_seq"):
                assert torch.equal(obs.obs[k][r], s["obs"][k][t]), (r, t, k)
            f, to = int(s["tgt"]["from_idx"][t]), int(s["tgt"]["to_idx"][t])
            p = int(s["tgt"]["promotion"][t])
            assert torch.equal(obs.legal["from"][r].float(), s["tgt"]["legal_from"][t]), (r, t)
            assert torch.equal(obs.legal["to_cond"][r][f].float(), s["tgt"]["legal_to_of"][t])
            assert torch.equal(obs.legal["promotion_cond"][r][f, to].float(),
                               s["tgt"]["legal_promotion_of"][t]), (r, t)
            assert torch.equal(obs.legal["from"][r], obs.legal["to_cond"][r].any(-1))
            acts["from_idx"][r], acts["to_idx"][r], acts["promotion"][r] = f, to, p
            promotions += p > 0
            compared += 1
        obs, rew, done, info = env.step(acts)
        for r in rows:
            if t < lengths[r]:
                s = samples[r]
                assert (int(info["applied"]["from_idx"][r]), int(info["applied"]["to_idx"][r]),
                        int(info["applied"]["promotion"][r])) == (
                    int(s["tgt"]["from_idx"][t]), int(s["tgt"]["to_idx"][t]),
                    int(s["tgt"]["promotion"][t])), (r, t)
            if t == lengths[r] - 1 and bool(done[r]):
                # a rules end (mate / stalemate / automatic draw): the record's result
                v = samples[r]["tgt"]["value"][0].item()          # White's first frame
                assert rew[r].tolist() == [v, -v], r
                ended += 1
        t += 1
    assert compared == sum(lengths)
    assert promotions >= 4                     # every promotion piece is in the fixture
    assert ended >= 1


def test_default_env_ends_records_at_their_automatic_draws(shards):
    """The RL env's defaults make threefold repetition, the 50-move clock and dead
    material automatic draws: replaying the records, a row the env ends before its
    record does ends on such a position (the record's next frame says so: seen twice
    before — globals 9 — or the halfmove clock at 100 — globals 7), with a draw."""
    root, _ = shards
    samples = load_gm_samples(str(root))
    n = len(samples)
    lengths = [s["obs"]["board_seq"].shape[0] for s in samples]
    env = GmSimVecEnv("chess", n, seed=0, max_turns=max(lengths) + 5)
    obs = env.reset(seed=0)
    gen = torch.Generator().manual_seed(0)
    t, early = 0, 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1).tolist()
        acts = env.random_actions(obs, generator=gen)
        for r in rows:
            if t < lengths[r]:
                s = samples[r]
                acts["from_idx"][r] = int(s["tgt"]["from_idx"][t])
                acts["to_idx"][r] = int(s["tgt"]["to_idx"][t])
                acts["promotion"][r] = int(s["tgt"]["promotion"][t])
        obs, rew, done, info = env.step(acts)
        for r in rows:
            if bool(done[r]) and t < lengths[r] - 1:          # ended before its record
                g = samples[r]["obs"]["globals_seq"][t + 1]
                assert bool(g[9] == 1.0) or bool(g[7] >= 1.0), (r, t, g.tolist())
                assert rew[r].tolist() == [0.0, 0.0] and not bool(info["truncated"][r])
                early += 1
        t += 1
    assert early >= 1                  # the fixture has unclaimed repetitions


def test_shard_meta_and_dataset(shards):
    root, st = shards
    ds = GameShardDataset(str(root), game="chess")
    assert ds.converter_version == 3 and ds.shard_meta == {"global_features": 10}
    assert len(ds) == st["kept"]
    assert ds.total_frames() == sum(s["obs"]["board_seq"].shape[0] for s in load_gm_samples(str(root)))
    s = ds[0]
    assert set(s["obs"]) == {"board_seq", "globals_seq"}
    assert {"from_idx", "to_idx", "promotion", "legal_from", "legal_to_of",
            "legal_promotion_of", "value"} <= set(s["tgt"])


# ---------------------------------------------------------------------------
# play, invariance
# ---------------------------------------------------------------------------

def test_network_samples_legal_moves():
    from env.league import StreamStepper

    torch.manual_seed(0)
    net = build_game_network("chess", **TINY).eval()
    env = GmSimVecEnv("chess", 3, seed=1, max_turns=60)
    stepper = StreamStepper(net)
    obs = env.observe()
    steps = 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1)
        restart = obs.stream_id[rows][obs.new_episode[rows] | (obs.step_idx[rows] == 0)]
        if restart.numel():
            stepper.reset(restart.tolist())
        with torch.no_grad():
            out = stepper.step({k: v[rows] for k, v in obs.obs.items()},
                               {k: v[rows] for k, v in obs.legal.items()},
                               obs.stream_id[rows].tolist(), obs.step_idx[rows].tolist())
        f, to, p = out["action_from"].cpu(), out["action_to"].cpu(), out["action_promotion"].cpu()
        for j, r in enumerate(rows.tolist()):        # every sampled move is legal
            assert bool(obs.legal["to_cond"][r, f[j], to[j]])
            assert bool(obs.legal["promotion_cond"][r, f[j], to[j], p[j]])
        acts = {k: torch.zeros(obs.n_rows, dtype=torch.long)
                for k in ("action_from", "action_to", "action_promotion")}
        acts["action_from"][rows], acts["action_to"][rows] = f, to
        acts["action_promotion"][rows] = p
        obs, _, _, info = env.step(acts)
        assert bool(info["acted"][rows].all())
        steps += 1
    assert steps > 20


def test_simulator_causality_on_random_playouts():
    for seed in (1, 2):
        ev = gmsim.random_trajectory("chess", {}, seed)[:80]
        assert invariance.check_sim_causality("chess", {}, ev) == len(ev)
    ev = gmsim.random_trajectory("chess", {}, 7)
    assert all(gmsim.perturb_hidden("chess", {}, ev, seat, 0) is None for seat in (0, 1))


# ---------------------------------------------------------------------------
# IL then RL
# ---------------------------------------------------------------------------

def test_train_multi_game_then_rl_joint(shards, tmp_path):
    from training import train_multi_game, train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _ = shards
    torch.manual_seed(0)
    out = train_multi_game.main([
        "--data", f"chess={root}", "--val", f"chess={root}", "--steps", "3",
        "--batch_size", "4", "--max_len", "48", "--val_max_len", "48", "--warmup_steps", "1",
        "--eval_every", "3", "--device", "cpu", "--il_legal_mask",
        "--save_dir", str(tmp_path / "ck"), "--override", *CORE])
    assert all(torch.isfinite(p).all() for p in out["model"].parameters())
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["configs"]["chess"]["board_global_features"] == 10     # from the shard meta
    args = ["--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "chess",
            "--tokens_per_iter", "120", "--envs", "chess=3", "--env_kw", "chess:max_turns=40",
            "--mirror_prob", "chess=0.5", "--snapshot_every", "1", "--epochs", "1",
            "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"]["chess"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    for k in ("entropy", "kl", "value_loss"):
        assert np.isfinite(row[k]), k
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(tmp_path / "rl2")]))
    tr._rollouts(0, {"chess": 60}, Diag(tr.games, tr.n_moe))
    for s in tr.rl["chess"].streams:
        assert s.routes is not None and s.routes.shape[0] == s.length   # routing replay
    tr._recompute("chess")
    for s in tr.rl["chess"].streams:
        assert torch.allclose(s.lp_old, s.mu, atol=1e-4)
