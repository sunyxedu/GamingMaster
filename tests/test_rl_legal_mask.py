"""RL legal-masking tests.

The PPO-exactness invariant must hold on the MASKED path exactly as it
does on the unmasked one: step(legal_mask, direction_mask) during rollout
and forward(legal_mask, direction_mask) + compute_log_prob during the
update must produce identical log-probs (ratio == 1).

Also checks that masked sampling never emits an illegal source/direction
and that the env's dir_mask matches get_valid_directions.
"""

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.network import GeneralsNetwork


@pytest.fixture(scope="module")
def cfg():
    return ModelConfig(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
                       trm_block_layers=1)


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    m = GeneralsNetwork(cfg)
    m.eval()
    return m


def _rollout(model, env, T=6, act_steps=1):
    """Masked rollout via step(); returns stacked tensors for replay."""
    obs = env.reset(seed=7)
    cache = None
    rec = {k: [] for k in ("grid", "glb", "legal", "dirm", "src", "dir",
                           "spl", "passed", "lp")}
    for t in range(T):
        grid = obs[0]["grid"].unsqueeze(0)
        glb = obs[0]["globals"].unsqueeze(0)
        legal = obs[0]["legal_mask"].unsqueeze(0)
        dirm = obs[0]["dir_mask"].unsqueeze(0)
        with torch.no_grad():
            out = model.step(grid, glb, legal_mask=legal, st_cache=cache,
                             step_idx=t, act_steps=act_steps,
                             direction_mask=dirm)
        cache = out["st_cache"]
        rec["grid"].append(grid[0])
        rec["glb"].append(glb[0])
        rec["legal"].append(legal[0])
        rec["dirm"].append(dirm[0])
        rec["src"].append(out["action_source"].item())
        rec["dir"].append(out["action_direction"].item())
        rec["spl"].append(out["action_split"].item())
        rec["passed"].append(bool(out["action_pass"].item()))
        rec["lp"].append(out["action_log_prob"].item())

        action = None
        if not rec["passed"][-1]:
            action = {"source_idx": rec["src"][-1],
                      "direction": rec["dir"][-1],
                      "split": rec["spl"][-1]}
        obs, _, done, _ = env.step(action, None)
        if done:
            break
    return rec


class TestMaskedPPOExactness:
    def test_masked_rollout_replay_ratio_is_one(self, model, cfg):
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        rec = _rollout(model, env, T=6)
        T = len(rec["lp"])

        grid_seq = torch.stack(rec["grid"]).unsqueeze(0)
        glb_seq = torch.stack(rec["glb"]).unsqueeze(0)
        legal_seq = torch.stack(rec["legal"]).unsqueeze(0)   # [1,T,1,H,W]
        dirm_seq = torch.stack(rec["dirm"]).unsqueeze(0)     # [1,T,4,H,W]
        actions = {
            "source_idx": torch.tensor(rec["src"]).unsqueeze(0),
            "direction": torch.tensor(rec["dir"]).unsqueeze(0),
            "split": torch.tensor(rec["spl"]).unsqueeze(0),
            "passed": torch.tensor(rec["passed"]).unsqueeze(0),
        }
        with torch.no_grad():
            pred = model(
                grid_seq, glb_seq,
                legal_mask=legal_seq,
                source_y=actions["source_idx"] // cfg.grid_w,
                source_x=actions["source_idx"] % cfg.grid_w,
                act_steps=1,
                direction_mask=dirm_seq,
            )
            lp, ent, _ = model.compute_log_prob(
                pred, actions, legal={"source": legal_seq})

        old = torch.tensor(rec["lp"]).unsqueeze(0)
        ratio = (lp - old).exp()
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-4), \
            f"masked rollout/replay mismatch: max|ratio-1|={(ratio - 1).abs().max()}"
        assert torch.isfinite(ent).all()

    def test_masked_sampling_never_illegal(self, model, cfg):
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        obs = env.reset(seed=11)
        legal = obs[0]["legal_mask"]
        dirm = obs[0]["dir_mask"]
        cache = None
        for t in range(12):
            with torch.no_grad():
                out = model.step(
                    obs[0]["grid"].unsqueeze(0),
                    obs[0]["globals"].unsqueeze(0),
                    legal_mask=obs[0]["legal_mask"].unsqueeze(0),
                    st_cache=cache, step_idx=t, act_steps=1,
                    direction_mask=obs[0]["dir_mask"].unsqueeze(0))
            cache = out["st_cache"]
            src = out["action_source"].item()
            d = out["action_direction"].item()
            r, c = divmod(src, cfg.grid_w)
            legal = obs[0]["legal_mask"]
            dirm = obs[0]["dir_mask"]
            has_moves = bool(legal.any())
            if not bool(out["action_pass"].item()) and has_moves:
                assert legal[0, r, c], f"illegal source sampled at step {t}"
                assert dirm[d, r, c], f"illegal direction sampled at step {t}"
                action = {"source_idx": src, "direction": d,
                          "split": out["action_split"].item()}
            else:
                # No legal source (all-masked fallback) => collect sends pass.
                action = None
            obs, _, done, info = env.step(action, None)
            assert not info["illegal"][0], "masked action was illegal in env"
            if done:
                break


class TestDirMask:
    def test_env_dir_mask_matches_get_valid_directions(self):
        env = GeneralsEnv(n_mountains=25, n_cities=3)
        env.reset(seed=5)
        dirm = env.dir_valid
        for idx in list(env.player_land[0]) + list(env.player_land[1]):
            valid = env.get_valid_directions(idx)
            r, c = divmod(idx, env.grid_w)
            for d in range(4):
                assert bool(dirm[d, r, c]) == valid[d]

    def test_masked_direction_logits_suppressed(self, model, cfg):
        """Direction logits at the chosen source are -1e9 on invalid dirs."""
        env = GeneralsEnv(grid_h=cfg.grid_h, grid_w=cfg.grid_w)
        obs = env.reset(seed=3)
        # find an owned movable tile on the map edge (has an invalid dir)
        with torch.no_grad():
            out = model.step(
                obs[0]["grid"].unsqueeze(0), obs[0]["globals"].unsqueeze(0),
                legal_mask=obs[0]["legal_mask"].unsqueeze(0),
                act_steps=1,
                direction_mask=obs[0]["dir_mask"].unsqueeze(0))
        src = out["action_source"].item()
        r, c = divmod(src, cfg.grid_w)
        dl = out["direction_logits"][0]
        for d in range(4):
            if not obs[0]["dir_mask"][d, r, c]:
                assert dl[d] <= -1e8, f"invalid dir {d} not masked: {dl[d]}"
