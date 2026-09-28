"""StarCraft II simulator (env/vec_sc2.py on gmsim's Sc2Sim): the headless 3.16.1 client
against the built-in AI.  Needs the Linux install (``GM_SC2_INSTALL``, default
~/gm-data/sc2/3.16.1/StarCraftII) — skipped elsewhere.

* the obs are the shards' tensors (names, shapes; the converter and the env share gmsc2's
  encoder, so the frames are the same function of the game state);
* random decisions play: commands reach the game, windows advance 8 loops, units keep their
  slots while they live;
* the network's rollout drives the env (decode → sample → step), and the stored actions
  replay to the rollout's log-prob (the RL spec's teacher-forced labels).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

INSTALL = os.environ.get("GM_SC2_INSTALL") or os.path.expanduser("~/gm-data/sc2/3.16.1/StarCraftII")
pytestmark = pytest.mark.skipif(not (sys.platform.startswith("linux") and os.path.isdir(os.path.join(INSTALL, "Versions"))),
                                reason="needs the Linux SC2 install")

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, encoder_channels=8,
            core_type="transformer", core_layers=1)


@pytest.fixture(scope="module")
def env():
    from env.vec_sc2 import Sc2VecEnv

    e = Sc2VecEnv(2, autoreset=True, difficulty="very_easy", races=("terran", "zerg"), max_loops=22 * 60 * 3, seed=0)
    yield e
    del e


def test_obs_are_the_shard_tensors(env):
    obs = env.observe()
    schema = env.obs_schema()
    for k, (shape, dtype) in schema.items():
        assert tuple(obs.obs[k].shape[1:]) == tuple(shape), k
    assert obs.obs["unit_own"].any(-1).all(), "every game starts with own units"
    assert obs.new_episode.all() and obs.active.all()


def test_random_decisions_play(env):
    obs = env.observe()
    g = torch.Generator().manual_seed(0)
    sent = 0
    loops0 = None
    for _ in range(200):
        a = env.random_actions(obs, g)
        obs, rew, done, info = env.step(a)
        sent += int(info["sent"].sum())
        if loops0 is None:
            loops0 = info["game_loop"].clone()
    assert env.failures == 0, env._sim.last_error
    assert sent > 20
    assert (info["game_loop"] > loops0).all(), "the windows advance"


def test_the_network_drives_the_env_and_replays_its_log_prob(env):
    from model.games.registry import build_game_network
    from training.train_rl_joint import game_spec

    torch.manual_seed(0)
    net = build_game_network("sc2", **TINY).eval()
    spec = game_spec("sc2")
    spec.check_env(env, net)
    obs = env.observe()
    for _ in range(20):
        seq = {k: v.unsqueeze(1) for k, v in obs.obs.items()}
        with torch.no_grad():
            out = net(seq)
            pred = out["pred"] if "pred" in out else out
            s = net.action_decoder.sample({k: v[:, 0] for k, v in pred.items()})
            acts = env.replay_actions(s)
            labels = {k: v.unsqueeze(1) for k, v in spec.labels(acts, net.cfg).items()}
            out2 = net(seq, labels=labels)
            pred2 = out2["pred"] if "pred" in out2 else out2
            lp, _, _ = net.action_decoder.log_prob_entropy(pred2, {k: v.unsqueeze(1) for k, v in acts.items()})
        torch.testing.assert_close(lp[:, 0], s["action_log_prob"], rtol=0, atol=1e-4)
        obs, _, _, info = env.step(s)
    assert env.failures == 0, env._sim.last_error


SHARDS = os.environ.get("GM_SC2_TEST_SHARDS") or os.path.expanduser("~/gm-data/sc2/convert_test/train")


@pytest.mark.skipif(not os.path.isdir(SHARDS), reason="needs converted sc2 shards (gmdata sc2 convert)")
def test_il_then_rl_on_the_simulator(tmp_path):
    """IL on converted replays, then one RL iteration against the built-in AI: the
    learner's replay of the rollout is exact (μ = π_old)."""
    import json

    import numpy as np

    from training import train_multi_game, train_rl_joint

    core = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0", "encoder_channels=8",
            "core_type=transformer", "core_layers=1", "temporal_mixer=recurrent"]
    train_multi_game.main(["--data", f"sc2={SHARDS}", "--steps", "3", "--batch_size", "2", "--max_len", "16",
                           "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu",
                           "--save_dir", str(tmp_path / "ck"), "--override", *core])
    res = train_rl_joint.main([
        "--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "sc2", "--tokens_per_iter", "200",
        "--envs", "sc2=2", "--env_kw", "sc2:difficulty=very_easy", "--epochs", "1", "--minibatches", "2",
        "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1", "--iterations", "1",
        "--save_dir", str(tmp_path / "rl")])
    row = res["history"][-1]["games"]["sc2"]
    assert row["tokens"] >= 1
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    assert np.isfinite(row["entropy"]) and np.isfinite(row["value_loss"])
    json.dumps(res["history"][-1])
