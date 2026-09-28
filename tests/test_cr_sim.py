"""Clash Royale simulator (env/vec_cr.py: RoyaleSim battles observed through crdata's frame
builder).  Needs the ``royalesim`` extension (a RoyaleSim checkout built with maturin) —
skipped without it.

* a battle the env played, exported as a RoyaleAPI replay and converted by ``crdata
  convert``, gives the env's own frames (the IL and RL frames are one function of the
  public record), except what the env knows and a replay does not: the own hand (the
  deck tokens' P(in hand) / P(next) and the cards they make legal);
* random decisions play battles to the end with ±1 / 0 rewards;
* the network's rollout drives the env and replays to its log-prob.
"""

from __future__ import annotations

import json
import os
import subprocess

import pytest
import torch

royalesim = pytest.importorskip("royalesim")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
#: deck-token feature columns that hold the own-hand belief (expand.py)
BELIEF = (6, 7, 8)


@pytest.fixture(scope="module")
def crdata():
    subprocess.run(["cargo", "build", "--release", "-p", "crdata"], cwd=WS, check=True, capture_output=True)
    return os.path.join(WS, "target", "release", "crdata")


def _play(env, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = env.observe()
    seen = [obs]
    rewards, done_at = [], None
    for t in range(steps):
        obs, rew, done, info = env.step(env.random_actions(obs, g))
        rewards.append(rew)
        seen.append(obs)
        if bool(done[0]):
            done_at = t + 1
            break
    return seen, rewards, done_at


def test_a_simulated_battle_converts_to_the_envs_frames(crdata, tmp_path):
    from data_processing.games.dataset import load_gm_samples
    from env.vec_cr import ClashRoyaleVecEnv

    env = ClashRoyaleVecEnv(1, seed=3)
    seen, _, _ = _play(env, 240)
    assert len(env._plays[0]) >= 6, "random play plays cards"
    rec = tmp_path / "rec.jsonl"
    rec.write_text(json.dumps({"tag": "sim-0", "payload": env.export_replay(0)}) + "\n")
    out = tmp_path / "shards"
    subprocess.run([crdata, "convert", "--input", str(rec), "--out", str(out)], check=True, capture_output=True)
    samples = load_gm_samples(str(out))
    assert len(samples) == 2
    for seat, s in enumerate(sorted(samples, key=lambda s: s["meta"]["side"] != "team")):
        obs = s["obs"]
        T = min(obs["arena_seq"].shape[0], len(seen))
        for t in range(T):
            got = {k: v[2 * 0 + seat] for k, v in seen[t].obs.items()}
            for k, v in obs.items():
                if k not in got:
                    continue
                a, b = v[t], got[k]
                if k == "deck_feats":
                    keep = [j for j in range(a.shape[-1]) if j not in BELIEF]
                    a, b = a[..., keep], b[..., keep]
                if k == "globals_seq":
                    a, b = a[:5], b[:5]              # [5]: the expected next-card cost (belief)
                torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=1e-5, msg=f"seat {seat} frame {t} {k}")


def test_random_battles_end_with_the_result():
    from env.vec_cr import ClashRoyaleVecEnv

    env = ClashRoyaleVecEnv(2, seed=1)
    _, rewards, done_at = _play(env, 700, seed=1)
    assert done_at is not None and done_at <= 601
    r = rewards[-1][0]
    assert float(r.sum()) == 0.0 and set(r.tolist()) <= {-1.0, 0.0, 1.0}


def test_the_network_drives_the_env_and_replays_its_log_prob():
    from env.vec_cr import ClashRoyaleVecEnv
    from model.games.registry import build_game_network

    torch.manual_seed(0)
    net = build_game_network("clash_royale", d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
                             encoder_channels=8, core_type="transformer", core_layers=1).eval()
    env = ClashRoyaleVecEnv(1, seed=5)
    obs = env.observe()
    for _ in range(12):
        seq = {k: v.unsqueeze(1) for k, v in obs.obs.items()}
        legal = {k: v.unsqueeze(1) for k, v in obs.legal.items()}
        with torch.no_grad():
            out = net(seq, legal=legal)
            pred = out["pred"] if "pred" in out else out
            s = net.action_decoder.sample({k: v[:, 0] for k, v in pred.items()}, legal=obs.legal)
            acts = env.replay_actions(s)
            labels = {"card_slot": acts["card_slot"].unsqueeze(1), "place_idx": acts["place_idx"].unsqueeze(1)}
            out2 = net(seq, legal=legal, labels=labels)
            pred2 = out2["pred"] if "pred" in out2 else out2
            lp, _, _ = net.action_decoder.log_prob_entropy(pred2, {k: v.unsqueeze(1) for k, v in acts.items()}, legal)
        torch.testing.assert_close(lp[:, 0], s["action_log_prob"], rtol=0, atol=1e-4)
        obs, _, _, _ = env.step(s)


def test_il_then_rl_on_the_simulator(crdata, tmp_path):
    """IL on the fixture replays (crdata convert), then one RL iteration of self-play in the
    simulator: the learner's replay of the rollout is exact (μ = π_old)."""
    import numpy as np

    from training import train_multi_game, train_rl_joint

    fix = os.path.join(ROOT, "tests", "fixtures", "clash_royale", "replays.jsonl")
    subprocess.run([crdata, "convert", "--input", fix, "--out", str(tmp_path / "il")], check=True, capture_output=True)
    core = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0", "encoder_channels=8",
            "core_type=transformer", "core_layers=1", "temporal_mixer=recurrent"]
    train_multi_game.main(["--data", f"clash_royale={tmp_path / 'il'}", "--steps", "3", "--batch_size", "2",
                           "--max_len", "32", "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu",
                           "--save_dir", str(tmp_path / "ck"), "--override", *core])
    res = train_rl_joint.main([
        "--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "clash_royale", "--tokens_per_iter", "300",
        "--envs", "clash_royale=2", "--epochs", "1", "--minibatches", "2", "--snapshot_dir", "memory",
        "--device", "cpu", "--seed", "1", "--iterations", "1", "--save_dir", str(tmp_path / "rl")])
    row = res["history"][-1]["games"]["clash_royale"]
    assert row["tokens"] >= 1
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    assert np.isfinite(row["entropy"]) and np.isfinite(row["value_loss"])
    json.dumps(res["history"][-1])
