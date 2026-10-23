"""Pokémon Showdown simulator (env/vec_pokemon.py: Showdown's own sim in Node workers, frames
from gmsim ``pokemon_live_frame`` = the converter's parser): a live request's frame IS the IL
frame of that request — battles played in the env, their complete logs converted with
``gmdata pokemon convert`` as replays, give exactly the frames the env showed each side —
plus the env contract (rows, legality from the requests, ±1 rewards) and IL → RL end to end.

Needs Node.js and a built Showdown checkout (``GM_SHOWDOWN_DIR``, default
~/gm-tools/pokemon-showdown: ``git clone`` + ``node build`` at the vocabulary's commit).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from tests.test_rust_converters_parity import gmdata  # noqa: F401  (fixture)

SHOWDOWN = os.environ.get("GM_SHOWDOWN_DIR") or os.path.expanduser("~/gm-tools/pokemon-showdown")
pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.isfile(os.path.join(SHOWDOWN, "dist", "sim", "battle.js")),
    reason="Node.js or a built Pokémon Showdown checkout (GM_SHOWDOWN_DIR) not available")
gmsim = pytest.importorskip("gmsim")
if not hasattr(gmsim._native, "pokemon_live_frame"):
    pytest.skip("gmsim was built without pokemon_live_frame (uv sync --all-extras --reinstall-package gmsim)",
                allow_module_level=True)

OBS_KEYS = ("species_ids", "move_ids", "poke_feats", "slot_mask", "field_seq", "slot_moves", "slot_item",
            "slot_no_item", "slot_ability", "slot_tera_type", "format_id")


def _play(env, gen, keep_frames=True):
    """Random legal play to the end; per env and side the frames it showed."""
    frames = [[[], []] for _ in range(env.num_envs)]
    obs = env.observe()
    steps = 0
    while not bool(env.finished.all()):
        act = obs.active
        for row in torch.nonzero(act).flatten().tolist():
            e, s = divmod(row, 2)
            if keep_frames:
                frames[e][s].append({k: obs.obs[k][row].clone() for k in OBS_KEYS})
            lg = {k: v[row] for k, v in obs.legal.items()}
            assert bool(lg["action_type"].any()), "an active row without a legal action type"
            if lg["action_type"][0]:
                assert bool(lg["move_id"].any()), "a move-legal row without a usable move"
        a = env.random_actions(obs, gen)
        obs, rew, done, info = env.step(a)
        for e in torch.nonzero(done).flatten().tolist():
            if bool(info["terminated"][e]) and int(info["winner"][e]) >= 0:
                assert sorted(rew[e].tolist()) == [-1.0, 1.0]
        steps += 1
        assert steps < 3000
    return frames


TEAMS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pokemon_teams",
                     "gen9ou_hl_05_26_sample.jsonl")


@pytest.mark.parametrize("fmt,seed", [("gen9randombattle", 26), ("gen9ou", 41)])
def test_live_frames_are_the_converters_frames(gmdata, tmp_path, fmt, seed):  # noqa: F811
    """Random battles (no team preview; seed 26 holds a Revival Blessing, whose pick the env
    makes itself — the IL data has no frame for it) and gen9ou with team preview."""
    from data_processing.games.dataset import load_gm_samples
    from env.vec_pokemon import PokemonVecEnv

    env = PokemonVecEnv(8, format=fmt, teams=None if "random" in fmt else TEAMS, seed=seed, workers=2,
                        showdown_dir=SHOWDOWN)
    frames = _play(env, torch.Generator().manual_seed(0))
    raw = tmp_path / "raw"
    raw.mkdir()
    for e in range(env.num_envs):
        (raw / f"{fmt}-{e}.json").write_text(json.dumps(
            {"id": f"{fmt}-{e}", "formatid": fmt, "log": "\n".join(env.logs[e]), "players": ["p1", "p2"]}))
    env.close()
    subprocess.run([gmdata, "pokemon", "convert", "--input", str(raw), "--out", str(tmp_path / "out")],
                   check=True, capture_output=True)
    by = {(s["meta"]["replay_id"], s["meta"]["side"]): s for s in load_gm_samples(str(tmp_path / "out"))}
    compared = 0
    for e in range(len(frames)):
        for side in range(2):
            s = by.get((f"{fmt}-{e}", "p1" if side == 0 else "p2"))
            live = frames[e][side]
            if s is None:
                assert not live
                continue
            T = s["obs"]["species_ids"].shape[0]
            assert T == len(live), (e, side, T, len(live))
            for t in range(T):
                for k in OBS_KEYS:
                    got = s["obs"][k][t]
                    want = live[t][k]
                    assert torch.equal(got.to(want.dtype), want), (e, side, t, k, got, want)
                compared += 1
    assert compared > 200, compared


def test_il_then_rl_on_the_simulator(tmp_path, gmdata):  # noqa: F811
    from training import train_multi_game, train_rl_joint

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([gmdata, "pokemon", "convert", "--input", os.path.join(root, "tests", "fixtures", "pokemon"),
                    "--out", str(tmp_path / "il")], check=True, capture_output=True)
    core = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
            "core_type=transformer", "core_layers=1", "moe_experts=4", "temporal_mixer=recurrent"]
    train_multi_game.main(["--data", f"pokemon={tmp_path / 'il'}", "--steps", "3", "--batch_size", "2",
                           "--warmup_steps", "1", "--eval_every", "0", "--device", "cpu", "--il_legal_mask",
                           "--save_dir", str(tmp_path / "ck"), "--override", *core])
    res = train_rl_joint.main([
        "--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "pokemon", "--tokens_per_iter", "120",
        "--envs", "pokemon=2", "--env_kw", "pokemon:format='gen9randombattle'",
        "--env_kw", f"pokemon:showdown_dir='{SHOWDOWN}'", "--epochs", "1", "--minibatches", "2",
        "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1", "--iterations", "1",
        "--privileged_critic", "pokemon", "--save_dir", str(tmp_path / "rl")])
    tr = res["trainer"]
    assert tr.model["pokemon"].priv_value is not None
    streams = tr.rl["pokemon"].streams
    assert streams and streams[0].privileged["oracle"].shape[-1] == tr.rl["pokemon"].env.privileged_dim
    row = res["history"][-1]["games"]["pokemon"]
    assert row["tokens"] >= 1
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    assert np.isfinite(row["entropy"]) and np.isfinite(row["value_loss"])


def test_privileged_oracle_is_both_real_teams():
    """observe().privileged["oracle"]: on every active row the row's own team, then the foe's,
    as the sides' requests have them (env.teams); zeros on the other rows; HP moves as the
    battles go on."""
    from env.vec_pokemon import ORACLE_DIM, PokemonVecEnv, team_oracle

    env = PokemonVecEnv(4, format="gen9randombattle", seed=3, workers=1, showdown_dir=SHOWDOWN)
    assert env.privileged_dim == ORACLE_DIM
    gen = torch.Generator().manual_seed(0)
    obs = env.observe()
    damaged = checked = 0
    for _ in range(40):
        o = obs.privileged["oracle"]
        assert o.shape == (env.num_rows, ORACLE_DIM)
        for row in torch.nonzero(obs.active).flatten().tolist():
            e, s = divmod(row, 2)
            want = team_oracle(env.teams[e][s], env.teams[e][1 - s])
            assert np.allclose(o[row].numpy(), want)
            assert not np.allclose(want[:ORACLE_DIM // 2], want[ORACLE_DIM // 2:])   # two teams
            hp = o[row].reshape(2, 6, -1)[:, :, 0]
            damaged += int(((hp > 0) & (hp < 1)).any())
            checked += 1
        assert not o[~obs.active].any()
        if bool(env.finished.all()):
            break
        obs, *_ = env.step(env.random_actions(obs, gen))
    assert checked > 20 and damaged > 0
    env.close()
