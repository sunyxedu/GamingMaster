"""Shogi as a SIMULATED game (gmrules ``shogi``, model/games/gmrules/shogi.py) for the
existing shogi adapter — the IL side is unchanged:

    floodgate CSA ─► gmdata shogi convert (the gmrules shogi engine and frame encoders,
      converter v2) ─► gm-shards ─► ShogiEncoder / ShogiDecoder ─► train_multi_game
      ─► gmsim env (GmSimVecEnv) ─► train_rl_joint (ShogiSpec)

- spec + registrations: the simulator's frame schema is the v2 shards' (obs, legality,
  action keys, meta), the IL tables keep their shogi entries, the env / RL spec exist;
- GOLDEN: the RL env (records' config: no automatic declaration, no ply cap in reach)
  stepped with the records' own action keys reproduces every converter frame of the 40
  committed floodgate records bit for bit — observations, the legality tables (= the IL
  masks at the stored conditioning), the applied keys;
- a network's sampled move or drop is always legal in the env;
- causality of the simulator's frames on random playouts;
- train_multi_game on the converted shards, then train_rl_joint on shogi: finite losses,
  μ = π_old (ShogiSpec replay under the sampled masks, routing replay on).

The Rust side pins rules and parity (data_processing/rs/gmrules/src/games/shogi/tests.rs:
perft 30 / 900 / 25470 / 719731 and a drop position, every game end; gmdata shogi.rs:
byte-identical converter output, the floodgate server's own rulings).
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
CSA = ROOT / "tests" / "fixtures" / "shogi"
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent",
            shogi_global_features=7)
CORE = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
        "rd_prelude_layers=1", "rd_recurrent_layers=1", "rd_coda_layers=1", "rd_mean_iters=2",
        "rd_backprop_iters=1", "moe_experts=4"]
KEYS = ("dropped", "from_idx", "to_idx", "promoted", "piece_idx", "drop_to_idx")
#: the records' view of the rules: floodgate players may play on past a position where
#: they could declare, and 2025 records run to 512 plies
RECORDS = {"entering_king": "none", "max_plies": 1024}


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
    root = tmp_path_factory.mktemp("shogi")
    st = _run(gmdata, "shogi", "convert", "--input", CSA, "--allow-no-result",
              "--out", root / "shards")
    return root / "shards", st


# ---------------------------------------------------------------------------
# spec, registrations
# ---------------------------------------------------------------------------

def test_spec_and_registrations():
    spec = gmsim.spec_of("shogi", {})
    assert spec["schema"] == "shogi.v2" and spec["candidate"] is None
    assert spec["layout"] == "interleaved" and not spec["hidden_info"]
    assert [(t["name"], t["shape"]) for t in spec["obs"]] == [
        ("board_seq", [28, 9, 9]), ("hand_seq", [14, 4]), ("globals_seq", [7])]
    assert [(t["name"], t["shape"]) for t in spec["legal"]] == [
        ("mode", [2]), ("from", [81]), ("to_cond", [81, 81]), ("promo_cond", [81, 81, 2]),
        ("drop_piece", [14]), ("drop_to_cond", [14, 81])]
    assert spec["action_keys"] == list(KEYS)
    assert spec["meta"] == {"global_features": 7}           # the v2 shards' meta
    entry = rules_games()["shogi"]
    assert not entry.is_candidate and has_simulator("shogi")
    from training.train_il_game import LABEL_KEYS, META_ARCH_KEYS, META_LEGACY
    from training.train_rl_joint import GAME_SPECS, ShogiSpec
    assert LABEL_KEYS["shogi"] == ("dropped", "from_idx", "to_idx", "piece_idx")
    assert META_ARCH_KEYS["shogi"] == {"global_features": ("shogi_global_features",)}
    assert META_LEGACY["shogi"] == {"global_features": 4}
    assert isinstance(GAME_SPECS["shogi"], ShogiSpec)
    net = build_game_network("shogi", **TINY)
    env = make_vec_env("shogi", 2)
    assert isinstance(env, GmSimVecEnv)
    GAME_SPECS["shogi"].check_env(env, net)
    with pytest.raises(SystemExit, match="globals_seq"):     # the legacy 4-global width
        GAME_SPECS["shogi"].check_env(env, build_game_network("shogi", **{
            **TINY, "shogi_global_features": 4}))


# ---------------------------------------------------------------------------
# GOLDEN: env frames == converter frames
# ---------------------------------------------------------------------------

def test_env_reproduces_the_converter_frames(shards):
    """One GmSimVecEnv row per record, stepped with the record's own action keys: every
    observation and legality table equals the converter's frame and IL masks bit for
    bit (the ``_of`` masks are the tables' rows at the stored indices), and the applied
    keys are the record's."""
    root, st = shards
    samples = load_gm_samples(str(root))
    n = len(samples)
    assert n == st["converted"] >= 30 and st["illegal_move"] == 0 and st["mask_mismatch"] == 0
    lengths = [s["obs"]["board_seq"].shape[0] for s in samples]
    env = GmSimVecEnv("shogi", n, seed=0, max_turns=max(lengths) + 5, threads=2,
                      config=RECORDS)
    obs = env.reset(seed=0)
    gen = torch.Generator().manual_seed(0)
    compared, t, drops, promotions = 0, 0, 0, 0
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1).tolist()
        assert set(obs.step_idx[rows].tolist()) == {t}          # interleaved, lockstep
        acts = env.random_actions(obs, generator=gen)
        for r in rows:
            s = samples[r]
            if t >= lengths[r]:
                continue
            for k in ("board_seq", "hand_seq", "globals_seq"):
                assert torch.equal(obs.obs[k][r].float(), s["obs"][k][t].float()), (r, t, k)
            tg = {k: s["tgt"][k][t] for k in KEYS}
            f, to, pc = int(tg["from_idx"]), int(tg["to_idx"]), int(tg["piece_idx"])
            lg = obs.legal
            assert torch.equal(lg["mode"][r].float(), s["tgt"]["legal_mode"][t]), (r, t)
            assert torch.equal(lg["from"][r].float(), s["tgt"]["legal_from"][t]), (r, t)
            assert torch.equal(lg["to_cond"][r][f].float(), s["tgt"]["legal_to_of"][t]), (r, t)
            assert torch.equal(lg["promo_cond"][r][f, to].float(),
                               s["tgt"]["legal_promo_of"][t]), (r, t)
            assert torch.equal(lg["drop_piece"][r].float(), s["tgt"]["legal_drop_piece"][t])
            assert torch.equal(lg["drop_to_cond"][r][pc].float(),
                               s["tgt"]["legal_drop_to_of"][t]), (r, t)
            for k in KEYS:
                acts[k][r] = tg[k].to(acts[k].dtype)
            drops += int(bool(tg["dropped"]))
            promotions += int(bool(tg["promoted"]))
            compared += 1
        obs, rew, done, info = env.step(acts)
        for r in rows:
            if t < lengths[r]:
                s = samples[r]
                dropped = bool(s["tgt"]["dropped"][t])
                live = ("dropped", "piece_idx", "drop_to_idx") if dropped else \
                    ("dropped", "from_idx", "to_idx", "promoted")
                for k in live:                  # the branch the step decodes
                    assert int(info["applied"][k][r]) == int(s["tgt"][k][t]), (r, t, k)
        t += 1
    assert compared == sum(lengths)
    assert drops > 100 and promotions > 20


def test_shard_meta_and_dataset(shards):
    root, st = shards
    ds = GameShardDataset(str(root), game="shogi")
    assert ds.converter_version == 2 and ds.shard_meta == {"global_features": 7}
    assert ds.seat_split and len(ds) == 2 * st["converted"]     # two seat streams per record
    s = ds[0]
    assert set(s["obs"]) == {"board_seq", "hand_seq", "globals_seq"}
    assert set(KEYS) <= set(s["tgt"])


# ---------------------------------------------------------------------------
# play, invariance
# ---------------------------------------------------------------------------

def test_network_samples_legal_moves():
    from env.league import StreamStepper

    torch.manual_seed(0)
    net = build_game_network("shogi", **TINY).eval()
    env = GmSimVecEnv("shogi", 3, seed=1, max_turns=60)
    stepper = StreamStepper(net)
    obs = env.observe()
    steps = 0
    keys = rules_games()["shogi"].sample_keys
    while bool(obs.active.any()):
        rows = obs.active.nonzero().squeeze(1)
        restart = obs.stream_id[rows][obs.new_episode[rows] | (obs.step_idx[rows] == 0)]
        if restart.numel():
            stepper.reset(restart.tolist())
        with torch.no_grad():
            out = stepper.step({k: v[rows] for k, v in obs.obs.items()},
                               {k: v[rows] for k, v in obs.legal.items()},
                               obs.stream_id[rows].tolist(), obs.step_idx[rows].tolist())
        a = {keys[k]: out[k].cpu() for k in keys}
        for j, r in enumerate(rows.tolist()):        # every sampled move / drop is legal
            lg = obs.legal
            if bool(a["dropped"][j]):
                assert bool(lg["mode"][r, 1]) and bool(lg["drop_piece"][r, a["piece_idx"][j]])
                assert bool(lg["drop_to_cond"][r, a["piece_idx"][j], a["drop_to_idx"][j]])
            else:
                assert bool(lg["mode"][r, 0])
                f, to = a["from_idx"][j], a["to_idx"][j]
                assert bool(lg["to_cond"][r, f, to])
                assert bool(lg["promo_cond"][r, f, to, int(bool(a["promoted"][j]))])
        acts = {k: torch.zeros(obs.n_rows, dtype=out[k].dtype) for k in keys}
        for k in keys:
            acts[k][rows] = out[k].cpu()
        obs, _, _, info = env.step(acts)
        assert bool(info["acted"][rows].all())
        steps += 1
    assert steps > 20


def test_simulator_causality_on_random_playouts():
    for seed in (1, 2):
        ev = gmsim.random_trajectory("shogi", {}, seed)[:80]
        assert invariance.check_sim_causality("shogi", {}, ev) == len(ev)
    ev = gmsim.random_trajectory("shogi", {}, 7)
    assert all(gmsim.perturb_hidden("shogi", {}, ev, seat, 0) is None for seat in (0, 1))


# ---------------------------------------------------------------------------
# IL then RL
# ---------------------------------------------------------------------------

def test_train_multi_game_then_rl_joint(shards, tmp_path):
    from training import train_multi_game, train_rl_joint
    from training.train_rl_joint import Diag, JointRLTrainer, parse_args

    root, _ = shards
    torch.manual_seed(0)
    out = train_multi_game.main([
        "--data", f"shogi={root}", "--val", f"shogi={root}", "--steps", "3",
        "--batch_size", "4", "--max_len", "48", "--val_max_len", "48", "--warmup_steps", "1",
        "--eval_every", "3", "--device", "cpu", "--il_legal_mask",
        "--save_dir", str(tmp_path / "ck"), "--override", *CORE])
    assert all(torch.isfinite(p).all() for p in out["model"].parameters())
    ck = torch.load(tmp_path / "ck" / "multi_final.pt", map_location="cpu", weights_only=False)
    assert ck["configs"]["shogi"]["shogi_global_features"] == 7      # from the shard meta
    args = ["--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "shogi",
            "--tokens_per_iter", "120", "--envs", "shogi=3", "--env_kw", "shogi:max_turns=40",
            "--mirror_prob", "shogi=0.5", "--snapshot_every", "1", "--epochs", "1",
            "--minibatches", "2", "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1"]
    res = train_rl_joint.main(args + ["--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    rec = res["history"][-1]
    assert rec["dropped"] == 0 and rec["nonfinite"] == 0
    row = rec["games"]["shogi"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    for k in ("entropy", "kl", "value_loss"):
        assert np.isfinite(row[k]), k
    tr = JointRLTrainer(parse_args(args + ["--save_dir", str(tmp_path / "rl2")]))
    tr._rollouts(0, {"shogi": 60}, Diag(tr.games, tr.n_moe))
    for s in tr.rl["shogi"].streams:
        # ShogiSpec stores the masks the rollout sampled under, one row per frame
        assert set(s.legal) == {"mode", "from", "to_of", "promo_of", "drop_piece",
                                "drop_to_of"}
        assert s.routes is not None and s.routes.shape[0] == s.length
    tr._recompute("shogi")
    for s in tr.rl["shogi"].streams:
        assert torch.allclose(s.lp_old, s.mu, atol=1e-4)
