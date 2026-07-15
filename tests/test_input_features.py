"""Input-feature upgrade tests (derive_army_channels / turn_globals).

The load-time zero-pad migration must be exactly function-preserving:
a checkpoint trained without the features, loaded into a model with them
enabled, must produce bit-identical outputs (the new input channels hit
zero weights). Also checks the derived channel arithmetic and the env's
7-dim globals.
"""

import math

import pytest
import torch

from env.generals_env import GeneralsEnv
from model.config import ModelConfig
from model.network import GeneralsNetwork


def _cfg(**kw):
    return ModelConfig(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
                       trm_block_layers=1, **kw)


class TestFunctionPreservingMigration:
    def test_derived_channels_zero_pad_is_identity(self):
        torch.manual_seed(0)
        old = GeneralsNetwork(_cfg())
        new = GeneralsNetwork(_cfg(derive_army_channels=True))
        new.load_state_dict(old.state_dict())  # zero-pads conv1 dim 1
        old.eval(), new.eval()

        env = GeneralsEnv()
        obs = env.reset(seed=1)
        grid = obs[0]["grid"].unsqueeze(0).unsqueeze(0)   # [1,1,C,H,W]
        glb = obs[0]["globals"].unsqueeze(0).unsqueeze(0)
        sy = torch.zeros(1, 1, dtype=torch.long)
        sx = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            p_old = old(grid, glb, source_y=sy, source_x=sx, act_steps=1)
            p_new = new(grid, glb, source_y=sy, source_x=sx, act_steps=1)
        assert torch.allclose(p_old["source_logits"], p_new["source_logits"],
                              atol=1e-5)
        assert torch.allclose(p_old["value"], p_new["value"], atol=1e-5)

    def test_turn_globals_zero_pad_is_identity(self):
        torch.manual_seed(1)
        old = GeneralsNetwork(_cfg())
        new = GeneralsNetwork(_cfg(turn_globals=True, global_features=7))
        new.load_state_dict(old.state_dict())  # zero-pads global_proj dim 1
        old.eval(), new.eval()

        env = GeneralsEnv()
        obs = env.reset(seed=2)
        grid = obs[0]["grid"].unsqueeze(0).unsqueeze(0)
        glb4 = obs[0]["globals"].unsqueeze(0).unsqueeze(0)
        glb7 = torch.cat([glb4, torch.rand(1, 1, 3)], dim=-1)
        sy = torch.zeros(1, 1, dtype=torch.long)
        sx = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            p_old = old(grid, glb4, source_y=sy, source_x=sx, act_steps=1)
            p_new = new(grid, glb7, source_y=sy, source_x=sx, act_steps=1)
        assert torch.allclose(p_old["source_logits"], p_new["source_logits"],
                              atol=1e-5)
        assert torch.allclose(p_old["value"], p_new["value"], atol=1e-5)

    def test_pool_snapshot_roundtrip_after_migration(self):
        """Old-shape pool snapshots keep loading (every iteration) after
        the migration, and new-shape state dicts load unchanged."""
        torch.manual_seed(2)
        old_sd = GeneralsNetwork(_cfg()).state_dict()
        new = GeneralsNetwork(_cfg(derive_army_channels=True,
                                   turn_globals=True, global_features=7))
        new.load_state_dict(old_sd)
        new.load_state_dict(new.state_dict())  # new-shape: no-op adaptation


class TestDerivedChannels:
    def test_linear_army_values(self):
        cfg = _cfg(derive_army_channels=True)
        net = GeneralsNetwork(cfg)
        enc = net.obs_encoder
        grid = torch.zeros(1, 12, 25, 25)
        # own general with 150 army at (0,0)
        grid[0, 7, 0, 0] = 1.0
        grid[0, 11, 0, 0] = math.log1p(150) / 10.0
        # enemy blank with 40 army at (0,1)
        grid[0, 8, 0, 1] = 1.0
        grid[0, 11, 0, 1] = math.log1p(40) / 10.0
        # neutral city with 45 army at (0,2)
        grid[0, 3, 0, 2] = 1.0
        grid[0, 11, 0, 2] = math.log1p(45) / 10.0
        aug = enc._augment(grid)
        assert aug.shape == (1, 15, 25, 25)
        assert aug[0, 12, 0, 0] == pytest.approx(150 / 50.0, rel=1e-3)
        assert aug[0, 13, 0, 0] == 0.0
        assert aug[0, 13, 0, 1] == pytest.approx(40 / 50.0, rel=1e-3)
        assert aug[0, 14, 0, 2] == pytest.approx(45 / 50.0, rel=1e-3)
        # clamp at 10 (~500 army)
        grid[0, 11, 0, 0] = math.log1p(5000) / 10.0
        aug = enc._augment(grid)
        assert aug[0, 12, 0, 0] == pytest.approx(10.0)

    def test_disabled_is_passthrough(self):
        net = GeneralsNetwork(_cfg())
        grid = torch.rand(2, 12, 25, 25)
        assert net.obs_encoder._augment(grid) is grid


class TestTurnGlobals:
    def test_env_emits_seven_and_cycle_features(self):
        env = GeneralsEnv()
        env.turn_globals = True
        obs = env.reset(seed=3)
        assert obs[0]["globals"].shape == (7,)
        for _ in range(7):
            obs, _, done, info = env.step(None, None)
        g = obs[0]["globals"]
        turn = info["turn"]
        assert g[4] == pytest.approx((turn % 50) / 50.0)
        assert g[5] == pytest.approx(float(turn % 2))
        assert g[6] == pytest.approx(turn / 500.0)

    def test_env_default_stays_four(self):
        env = GeneralsEnv()
        obs = env.reset(seed=3)
        assert obs[0]["globals"].shape == (4,)
