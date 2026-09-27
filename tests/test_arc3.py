"""ARC-AGI-3 in the tier-3 pipeline: the adapter (model/games/arc3.py), the env
(env/vec_arc3.py, the offline SDK), the solver's demonstrations (scripts/arc3_solve.py
→ scripts/arc3_shards.py) and IL → RL end to end.

Adapter (no SDK needed): cell ↔ (patch, cell-in-patch) round trip; the rollout's
μ = sample()'s log-prob equals the replay's log π of the stored action (teacher-forced
click, the compact masks the rollout sampled with); legality of every sampled action;
greedy mode; il_head_terms decomposes il_loss.  Solver (no SDK needed): an action the
engine raises on is a dead end of the search, and a game that fails leaves results.
Env / solver / pipeline (skipped without the SDK and the cached games,
``ENVIRONMENTS_DIR``): the frame schema, a known solution completes a level (reward
1 / levels), game over forces a RESET, the action cap truncates, the per-reset game
rotation; the solver's solution replays into shards whose frames are the env's; a tiny
train_multi_game on them, then train_rl_joint on arc3 with μ = π_old.
"""

from __future__ import annotations

import os

import pytest
import torch

from model.games import build_game_network
from model.games.arc3 import CLICK, arc3_replay_legal, click_of, patch_of, sub_of

TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0,
            core_type="transformer", core_layers=1, temporal_mixer="recurrent",
            encoder_channels=16, arc_patch=8)
ENV_DIR = os.environ.get("ENVIRONMENTS_DIR") or os.path.expanduser(
    "~/Documents/arc3-wm/environment_files")


def _sdk_ok() -> bool:
    try:
        import arc_agi  # noqa: F401
        import arcengine  # noqa: F401
    except ImportError:
        return False
    return os.path.isdir(os.path.join(ENV_DIR, "vc33"))


needs_sdk = pytest.mark.skipif(not _sdk_ok(), reason="ARC-AGI-3 SDK (arc3 extra) or the cached "
                               "games (ENVIRONMENTS_DIR) not available")


def _obs(B, T, g):
    return {"grid": torch.randint(0, 16, (B, T, 64, 64), generator=g).to(torch.uint8),
            "delta": torch.rand(B, T, 64, 64, generator=g) < 0.05,
            "globals": torch.rand(B, T, 22, generator=g)}


def _legal(B, T, g, click=True):
    t = torch.rand(B, T, 8, generator=g) < 0.6
    t[..., 0] = True
    t[..., CLICK] = click
    c = (torch.rand(B, T, 4096, generator=g) < 0.3) if click else torch.zeros(B, T, 4096,
                                                                                dtype=torch.bool)
    return {"type": t, "click": c}


# ---------------------------------------------------------------------------
# adapter
# ---------------------------------------------------------------------------

def test_click_decomposition_round_trips():
    c = torch.arange(4096)
    for p in (4, 8):
        assert torch.equal(click_of(patch_of(c, 64, p), sub_of(c, 64, p), 64, p), c)
    # (x, y) = (5, 9) at patch 4: patch (x 1, y 2) = 2 * 16 + 1, cell (1, 1) = 1 * 4 + 1
    assert int(patch_of(torch.tensor(9 * 64 + 5), 64, 4)) == 33
    assert int(sub_of(torch.tensor(9 * 64 + 5), 64, 4)) == 5


def test_rollout_logprob_is_the_replay_logprob():
    """μ of the step's sampled action == log π of it under the replay's forward
    (teacher-forced click, arc3_replay_legal masks), and every sample is legal."""
    torch.manual_seed(0)
    net = build_game_network("arc3", **TINY).eval()
    g = torch.Generator().manual_seed(1)
    B = 6
    obs = _obs(B, 1, g)
    legal = _legal(B, 1, g)
    with torch.no_grad():
        out = net.step({k: v[:, 0] for k, v in obs.items()},
                       legal={k: v[:, 0] for k, v in legal.items()})
    t, c = out["action_type"], out["action_click"]
    for b in range(B):
        assert bool(legal["type"][b, 0, t[b]])
        if int(t[b]) == CLICK:
            assert bool(legal["click"][b, 0, c[b]])
    replay = {k: v.unsqueeze(1) for k, v in arc3_replay_legal(out).items()}
    acts = {"type_idx": t.unsqueeze(1), "click_idx": c.unsqueeze(1)}
    with torch.no_grad():
        pred = net(obs, legal=replay, labels={"click_idx": acts["click_idx"]}, act_steps=None)
        lp, ent, _ = net.action_decoder.log_prob_entropy(pred, acts, replay)
    torch.testing.assert_close(lp[:, 0], out["action_log_prob"].float(), rtol=1e-4, atol=1e-5)
    assert torch.isfinite(ent).all()


def test_greedy_and_no_click_frames():
    torch.manual_seed(0)
    net = build_game_network("arc3", **TINY).eval()
    g = torch.Generator().manual_seed(2)
    obs = _obs(3, 1, g)
    legal = _legal(3, 1, g, click=False)
    net.action_decoder.greedy = True
    with torch.no_grad():
        a = net.step({k: v[:, 0] for k, v in obs.items()},
                     legal={k: v[:, 0] for k, v in legal.items()})
        b = net.step({k: v[:, 0] for k, v in obs.items()},
                     legal={k: v[:, 0] for k, v in legal.items()})
    assert torch.equal(a["action_type"], b["action_type"])
    assert not bool((a["action_type"] == CLICK).any())


def test_il_head_terms_decompose_il_loss():
    torch.manual_seed(0)
    net = build_game_network("arc3", **TINY).train()
    g = torch.Generator().manual_seed(3)
    B, T = 2, 4
    obs = _obs(B, T, g)
    legal = _legal(B, T, g)
    tgt = {"type_idx": torch.full((B, T), CLICK), "click_idx": torch.randint(0, 4096, (B, T),
                                                                             generator=g)}
    tgt["type_idx"][:, 0] = 0
    pred = net(obs, legal=legal, labels={"click_idx": tgt["click_idx"]})
    mask = torch.ones(B, T, dtype=torch.bool)
    dec = net.action_decoder
    loss = dec.il_loss(pred, tgt, mask)["loss"]
    terms = dec.il_head_terms(pred, tgt, mask)
    total = sum((t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    torch.testing.assert_close(loss, total)
    loss.backward()
    assert net.action_decoder.sub_head.net[0].weight.grad is not None


# ---------------------------------------------------------------------------
# env (the offline SDK)
# ---------------------------------------------------------------------------

@needs_sdk
def test_env_frames_rewards_and_resets():
    from env.vec_arc3 import Arc3VecEnv

    env = Arc3VecEnv(2, games=("vc33", "sp80"), max_actions=50, environments_dir=ENV_DIR)
    obs = env.observe()
    assert env.obs_schema() == {"grid": ((64, 64), torch.uint8),
                                "delta": ((64, 64), torch.bool), "globals": ((22,), torch.float32)}
    assert env.game_of == ["vc33", "sp80"]
    assert not obs.obs["delta"].any() and bool(obs.new_episode.all())
    # the solver's vc33 level-0 solution (3 clicks) completes the level: reward 1 / 7
    rew_sum = torch.zeros(2)
    for _ in range(3):
        obs, rew, done, info = env.step({"type_idx": torch.tensor([6, 1]),
                                         "click_idx": torch.tensor([34 * 64 + 62, 0])})
        rew_sum += rew[:, 0]
    assert rew_sum[0] == pytest.approx(1 / 7) and int(info["levels"][0]) == 1
    assert float(obs.obs["globals"][0, 20]) == 1.0                   # just leveled
    assert float(obs.obs["globals"][0, 16]) == 1.0                   # last action ACTION6
    assert obs.obs["delta"][0].any()
    # the action cap truncates; the next reset rotates the games
    while bool(obs.active.any()):
        obs, rew, done, info = env.step(env.random_actions(obs))
        forced = obs.forced & obs.active
        assert bool((obs.legal["type"][forced, 1:] == 0).all())      # game over: RESET only
    assert int(env.actions_used.max()) <= 50
    env.reset()
    assert env.game_of == ["sp80", "vc33"]


# ---------------------------------------------------------------------------
# solver → shards → IL → RL
# ---------------------------------------------------------------------------

class _Frame:
    """A 1×4 corridor frame: the agent's cell is 1, the level is won at cell 3."""

    def __init__(self, pos):
        import numpy as np
        grid = np.zeros((1, 4), dtype=np.uint8)
        grid[0, pos] = 1
        self.frame, self.available_actions = [grid], [1, 2]
        self.state, self.levels_completed = "PLAY", int(pos == 3)


class _Corridor:
    """ACTION1 steps right; ACTION2 makes the engine raise (a game's own code failing)."""

    def __init__(self):
        self.pos = 0

    def perform_action(self, action, raw=True):
        if action.id == 2:
            raise RuntimeError("sprite not attached to its layer")
        self.pos += 1
        return _Frame(self.pos)


_FAKE_SDK = (None, lambda id, data: type("A", (), {"id": id, "data": data})(),
             type("GA", (), {"from_id": staticmethod(lambda t: t)}),
             type("GS", (), {"GAME_OVER": "OVER"}))


def test_solver_prunes_engine_errors_and_isolates_failed_games(monkeypatch):
    from scripts import arc3_solve

    actions, _, fd, n, errors = arc3_solve.solve_level(_Corridor(), _Frame(0), 0, 100, 10,
                                                       _FAKE_SDK)
    assert actions == [(1, 0, 0)] * 3 and fd.levels_completed == 1
    assert errors["count"] == 2 and "not attached" in errors["first"]

    def broken_sdk():
        raise RuntimeError("cannot load the game")
    monkeypatch.setattr(arc3_solve, "_sdk", broken_sdk)
    game, res = arc3_solve.solve_game(("xx00", 10, 5, None, None))
    assert game == "xx00" and res["levels"] == [] and "cannot load" in res["error"]

@needs_sdk
def test_solver_shards_il_and_rl(tmp_path, monkeypatch):
    import json

    import numpy as np

    from data_processing.games.dataset import GameShardDataset
    from scripts import arc3_shards, arc3_solve
    from training import train_multi_game, train_rl_joint

    monkeypatch.setenv("ENVIRONMENTS_DIR", ENV_DIR)
    sol = arc3_solve.main(["--games", "vc33", "sp80", "--budget", "600", "--max_depth", "12",
                           "--workers", "1", "--out", str(tmp_path / "sol.json")])
    assert sol["vc33"]["levels"][0]["actions"] == [[6, 62, 34]] * 3
    # the per-game partial results (written after every solved level) replay the same
    partial = tmp_path / "sol.json.partial"
    assert (partial / "vc33.json").is_file()
    st_p = arc3_shards.main(["--solutions", str(partial), "--out", str(tmp_path / "from_partial")])
    st = arc3_shards.main(["--solutions", str(tmp_path / "sol.json"),
                           "--out", str(tmp_path / "train")])
    assert st_p == st
    assert st["samples"] == 2 and st["levels"]["vc33"] >= 1
    ds = GameShardDataset(str(tmp_path / "train"), game="arc3")
    s = ds[0]
    assert s["tgt"]["value"][0] > 0 and s["tgt"]["value"][-1] > 0     # levels still to come
    assert torch.equal(s["tgt"]["legal_type"][:, 0], torch.ones(s["tgt"]["value"].shape[0],
                                                                dtype=torch.bool))
    core = ["d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
            "core_type=transformer", "core_layers=1", "moe_experts=4",
            "temporal_mixer=recurrent", "encoder_channels=16", "arc_patch=8"]
    train_multi_game.main(["--data", f"arc3={tmp_path / 'train'}", "--steps", "3",
                           "--batch_size", "2", "--warmup_steps", "1", "--eval_every", "0",
                           "--device", "cpu", "--il_legal_mask",
                           "--save_dir", str(tmp_path / "ck"), "--override", *core])
    res = train_rl_joint.main([
        "--init", str(tmp_path / "ck" / "multi_final.pt"), "--rl", "arc3",
        "--tokens_per_iter", "100", "--envs", "arc3=3", "--env_kw", "arc3:max_actions=20",
        "--env_kw", "arc3:games=('vc33', 'sp80')", "--epochs", "1", "--minibatches", "2",
        "--snapshot_dir", "memory", "--device", "cpu", "--seed", "1", "--iterations", "1",
        "--save_dir", str(tmp_path / "rl")])
    row = res["history"][-1]["games"]["arc3"]
    assert row["tokens"] >= 1 and row["episodes"] > 0
    assert abs(row["canary_log_old_mu"]) < 1e-4 and abs(row["canary_log_ratio"]) < 1e-4
    assert np.isfinite(row["entropy"]) and np.isfinite(row["value_loss"])
    json.dumps(res["history"][-1])
