"""Phase-6 regime tests.

Covers the pieces the phase-6 job turns on together:
  - HL-Gauss distributional critic (target construction, head interface,
    network wiring, scalar->distributional checkpoint migration)
  - RLLoss: HL-Gauss CE value loss + advantage-filter policy mask
  - anchor_weight decay schedule
  - top-|adv| filtering end-to-end through ppo_update
"""

import math

import pytest
import torch

from model.components.heads import HLGaussValueHead
from model.config import ModelConfig
from model.losses import RLLoss, hl_gauss_target_probs
from model.network import GeneralsNetwork
from model.rl_utils import RolloutBuffer, load_model_state
from env.generals_env import GeneralsEnv
from training.train_rl import (anchor_weight, collect_selfplay_episode,
                      load_sparring_net, ppo_update, resolve_phase_start)

TINY = dict(d_model=64, n_heads=4, trm_H_cycles=1, trm_L_cycles=2,
            trm_block_layers=1)


# ======================================================================
# HL-Gauss target distribution
# ======================================================================

class TestHLGaussTargets:
    def test_rows_sum_to_one(self):
        y = torch.tensor([-2.0, -1.0, -0.3, 0.0, 0.7, 1.0, 3.0])
        p = hl_gauss_target_probs(y, n_bins=128, vmax=1.5, sigma=0.06)
        assert p.shape == (7, 128)
        assert torch.allclose(p.sum(-1), torch.ones(7), atol=1e-5)
        assert (p >= 0).all()

    def test_expectation_recovers_target(self):
        # In-support targets: the smeared distribution's mean must sit on
        # the target well within one bin width (2*1.5/128 ≈ 0.023).
        y = torch.linspace(-1.2, 1.2, 25)
        p = hl_gauss_target_probs(y, n_bins=128, vmax=1.5, sigma=0.06)
        centers = torch.linspace(-1.5, 1.5, 128)
        assert torch.allclose((p * centers).sum(-1), y, atol=0.03)

    def test_out_of_support_clamps(self):
        p = hl_gauss_target_probs(torch.tensor([10.0]), 128, 1.5, 0.06)
        centers = torch.linspace(-1.5, 1.5, 128)
        exp = (p * centers).sum(-1).item()
        assert 1.5 - 3 * 0.06 <= exp <= 1.5 + 1e-6

    def test_batch_shape_passthrough(self):
        y = torch.zeros(2, 5)
        p = hl_gauss_target_probs(y, 64, 1.5, 0.06)
        assert p.shape == (2, 5, 64)


# ======================================================================
# HL-Gauss head + network wiring + migration
# ======================================================================

class TestHLGaussHead:
    def test_scalar_interface_bounded(self):
        torch.manual_seed(0)
        head = HLGaussValueHead(32, n_bins=64, vmax=1.5)
        v = head(torch.randn(7, 32))
        assert v.shape == (7,)
        assert (v.abs() <= 1.5 + 1e-5).all()

    def test_network_emits_logits_and_expectation(self):
        torch.manual_seed(0)
        cfg = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32, **TINY)
        net = GeneralsNetwork(cfg).eval()
        B, T = 1, 3
        grid = torch.randn(B, T, 12, 25, 25)
        glb = torch.randn(B, T, cfg.global_features)
        with torch.no_grad():
            out = net(grid, glb,
                      source_y=torch.zeros(B, T, dtype=torch.long),
                      source_x=torch.zeros(B, T, dtype=torch.long),
                      act_steps=1)
        assert out["value"].shape == (B, T)
        assert out["value_logits"].shape == (B, T, 32)
        centers = net.value_head.centers
        probs = torch.softmax(out["value_logits"], dim=-1)
        assert torch.allclose(out["value"], (probs * centers).sum(-1),
                              atol=1e-5)

    def test_scalar_network_has_no_logits(self):
        cfg = ModelConfig(**TINY)
        net = GeneralsNetwork(cfg).eval()
        with torch.no_grad():
            out = net(torch.randn(1, 2, 12, 25, 25),
                      torch.randn(1, 2, cfg.global_features),
                      source_y=torch.zeros(1, 2, dtype=torch.long),
                      source_x=torch.zeros(1, 2, dtype=torch.long),
                      act_steps=1)
        assert "value_logits" not in out

    def test_migration_from_scalar_checkpoint(self):
        torch.manual_seed(0)
        scalar_net = GeneralsNetwork(ModelConfig(**TINY))
        sd = scalar_net.state_dict()

        cfg6 = ModelConfig(hl_gauss_value=True, hl_gauss_bins=32, **TINY)
        net6 = GeneralsNetwork(cfg6)
        # Strict load must fail (shapes differ) …
        with pytest.raises(Exception):
            load_model_state(net6, sd, ctx="test")
        # … and the reinit escape hatch must succeed, keeping everything
        # except the value head byte-identical to the checkpoint.
        load_model_state(net6, sd, ctx="test",
                         reinit_prefixes=("value_head",))
        for k, v in net6.state_dict().items():
            if k.startswith("value_head") or k.endswith("centers"):
                continue
            assert torch.equal(v, sd[k]), f"{k} not restored"


# ======================================================================
# RLLoss: policy mask + HL-Gauss value loss
# ======================================================================

def _loss_inputs(T=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        log_probs=torch.randn(1, T, generator=g),
        entropy=torch.rand(1, T, generator=g),
        values=torch.randn(1, T, generator=g),
        returns=torch.randn(1, T, generator=g).clamp(-1, 1),
        advantages=torch.randn(1, T, generator=g),
        old_log_probs=torch.randn(1, T, generator=g),
    )


class TestRLLossPhase6:
    def test_all_true_policy_mask_matches_none(self):
        crit = RLLoss()
        inp = _loss_inputs()
        a = crit(**inp)
        b = crit(**inp, policy_mask=torch.ones(1, 12, dtype=torch.bool))
        for k in ("loss", "policy_loss", "entropy"):
            assert torch.allclose(a[k], b[k]), k

    def test_policy_mask_math(self):
        crit = RLLoss(entropy_weight=0.0)
        inp = _loss_inputs()
        pm = torch.zeros(1, 12, dtype=torch.bool)
        pm[0, :3] = True
        out = crit(**inp, policy_mask=pm)

        ratio = torch.exp(inp["log_probs"] - inp["old_log_probs"])
        clip = torch.clamp(ratio, 1 - crit.clip_range, 1 + crit.clip_range)
        pl = -torch.min(ratio * inp["advantages"], clip * inp["advantages"])
        expected = pl[0, :3].mean()
        assert torch.allclose(out["policy_loss"], expected, atol=1e-6)

    def test_value_loss_unaffected_by_policy_mask(self):
        crit = RLLoss()
        inp = _loss_inputs()
        pm = torch.zeros(1, 12, dtype=torch.bool)
        pm[0, 0] = True
        assert torch.allclose(crit(**inp)["value_loss"],
                              crit(**inp, policy_mask=pm)["value_loss"])

    def test_hl_gauss_value_loss(self):
        crit = RLLoss(hl_gauss=True, hl_gauss_vmax=1.5, hl_gauss_sigma=0.06)
        inp = _loss_inputs()
        n_bins = 64
        target_p = hl_gauss_target_probs(inp["returns"], n_bins, 1.5, 0.06)
        perfect = torch.log(target_p.clamp_min(1e-9))
        random_logits = torch.randn(1, 12, n_bins)

        good = crit(**inp, value_logits=perfect)["value_loss"]
        bad = crit(**inp, value_logits=random_logits)["value_loss"]
        assert good < bad
        assert torch.isfinite(good) and torch.isfinite(bad)
        # Perfect logits achieve the target distribution's own entropy
        ent = -(target_p * target_p.clamp_min(1e-9).log()).sum(-1).mean()
        assert torch.allclose(good, ent, atol=1e-3)

    def test_hl_gauss_requires_logits(self):
        crit = RLLoss(hl_gauss=True)
        with pytest.raises(ValueError):
            crit(**_loss_inputs())


# ======================================================================
# Anchor-weight schedule
# ======================================================================

class TestAnchorWeight:
    def test_pow_zero_is_constant(self):
        assert anchor_weight(0.05, 0, 0.0, 0.0) == 0.05
        assert anchor_weight(0.05, 10_000, 0.0, 0.0) == 0.05

    def test_decay_and_floor(self):
        w0 = anchor_weight(0.08, 0, 0.2, 0.005)
        w100 = anchor_weight(0.08, 100, 0.2, 0.005)
        w_huge = anchor_weight(0.08, 10**9, 0.2, 0.005)
        assert w0 == pytest.approx(0.08)
        assert w100 == pytest.approx(0.08 / (101 ** 0.2))
        assert w0 > w100 > w_huge
        assert w_huge == 0.005

    def test_negative_t_clamped(self):
        assert anchor_weight(0.08, -5, 0.2, 0.0) == pytest.approx(0.08)


class TestResolvePhaseStart:
    def test_fresh_run(self):
        assert resolve_phase_start(None, 1, cross_phase=False) == 1

    def test_within_phase_recovery_continues_clock(self):
        ckpt = {"iteration": 1200, "phase_start_iter": 951}
        assert resolve_phase_start(ckpt, 1201, cross_phase=False) == 951

    def test_cross_phase_resets(self):
        # Resuming a phase-6 checkpoint into a NEW phase directory must
        # restart the anchor schedule even though the key is present.
        ckpt = {"iteration": 1200, "phase_start_iter": 951}
        assert resolve_phase_start(ckpt, 1201, cross_phase=True) == 1201

    def test_legacy_checkpoint_without_key(self):
        assert resolve_phase_start({"iteration": 1000}, 1001,
                                   cross_phase=False) == 1001


# ======================================================================
# Sparring partner: cross-architecture frozen opponent
# ======================================================================

def _save_scalar_ckpt(tmp_path):
    torch.manual_seed(1)
    cfg = ModelConfig(**TINY)
    net = GeneralsNetwork(cfg)
    path = str(tmp_path / "sparring.pt")
    torch.save({"iteration": 950, "model_state": net.state_dict(),
                "config": cfg.__dict__}, path)
    return path, cfg


class TestSparringPartner:
    def test_load_frozen_from_own_config(self, tmp_path):
        path, _ = _save_scalar_ckpt(tmp_path)
        net = load_sparring_net(path, torch.device("cpu"),
                                expected_globals=4)
        assert not any(p.requires_grad for p in net.parameters())
        assert not net.training
        # Built from ITS config: scalar critic, no HL-Gauss logits.
        assert not hasattr(net.value_head, "centers")

    def test_globals_mismatch_raises(self, tmp_path):
        path, _ = _save_scalar_ckpt(tmp_path)
        with pytest.raises(ValueError):
            load_sparring_net(path, torch.device("cpu"), expected_globals=7)

    def test_cross_arch_episode(self, tmp_path):
        # A HL-Gauss learner must be able to play a full masked episode
        # against a scalar-critic sparring net — the exact phase-6 vs
        # ema950 pairing the job runs.
        path, _ = _save_scalar_ckpt(tmp_path)
        sparring = load_sparring_net(path, torch.device("cpu"),
                                     expected_globals=4)
        torch.manual_seed(2)
        learner = GeneralsNetwork(
            ModelConfig(hl_gauss_value=True, hl_gauss_bins=32, **TINY))
        env = GeneralsEnv(max_turns=8)
        bufs, replay, _ = collect_selfplay_episode(
            learner, sparring, env, torch.device("cpu"),
            max_steps=6, record=False, act_steps=1, legal_mask=True,
            learner_pid=1, collect_both=False, reset_seed=5)
        assert len(bufs) == 1
        assert len(bufs[0]) == 6


# ======================================================================
# End-to-end: adv filter + HL-Gauss critic through ppo_update
# ======================================================================

def _make_buffer(T, cfg, seed=0):
    torch.manual_seed(seed)
    buf = RolloutBuffer()
    for t in range(T):
        obs = {
            "grid": torch.randn(12, 25, 25),
            "globals": torch.randn(cfg.global_features),
            "legal_mask": torch.ones(1, 25, 25, dtype=torch.bool),
        }
        action = {"source_idx": int(torch.randint(0, 625, (1,))),
                  "direction": int(torch.randint(0, 4, (1,))),
                  "split": 0, "passed": False}
        buf.add(obs, action, log_prob=-1.0,
                value=float(torch.randn(1)), reward=float(torch.randn(1)),
                done=(t == T - 1))
    buf.compute_returns_and_advantages(gamma=1.0, lam=0.9)
    return buf


class TestPPOUpdatePhase6:
    @pytest.fixture(scope="class")
    def cfg(self):
        return ModelConfig(hl_gauss_value=True, hl_gauss_bins=32, **TINY)

    @pytest.fixture(scope="class")
    def net(self, cfg):
        torch.manual_seed(0)
        return GeneralsNetwork(cfg)

    def _run(self, net, cfg, frac):
        buf = _make_buffer(8, cfg)
        crit = RLLoss(hl_gauss=True, hl_gauss_vmax=cfg.hl_gauss_vmax,
                      hl_gauss_sigma=cfg.hl_gauss_sigma)
        opt = torch.optim.SGD(net.parameters(), lr=0.0)
        return ppo_update(
            net, None, [buf], crit, opt, torch.device("cpu"), cfg,
            kl_weight=0.0, invalidity_weight=0.0, legal_masked=False,
            adv_filter_frac=frac,
        )

    def test_filter_keeps_top_quarter(self, net, cfg):
        m = self._run(net, cfg, 0.25)
        # T=8, k=2 → at least 25% kept; ties may keep slightly more but
        # never the majority of the batch.
        assert 0.25 <= m["adv_kept"] <= 0.5
        assert math.isfinite(m["value_loss"])
        assert math.isfinite(m["policy_loss"])

    def test_no_filter_keeps_all(self, net, cfg):
        assert self._run(net, cfg, 0.0)["adv_kept"] == 1.0
