"""training/rl_joint/losses.py: decoupled PPO, value, entropy, KL terms."""

import math

import pytest
import torch
import torch.nn.functional as F

from model.losses import RLLoss, hl_gauss_cross_entropy
from training.rl_joint import (
    categorical_kl,
    combine_game_losses,
    decoupled_ppo_loss,
    entropy_loss,
    hl_gauss_value_loss,
    kl_penalty,
    value_loss,
)


def _policy_batch(seed=0, shape=(4, 16), spread=0.3):
    g = torch.Generator().manual_seed(seed)
    logp_old = -torch.rand(shape, generator=g) * 3 - 0.1
    logp = (logp_old + torch.randn(shape, generator=g) * spread).requires_grad_(True)
    adv = torch.randn(shape, generator=g)
    mask = torch.rand(shape, generator=g) > 0.2
    return logp, logp_old, adv, mask


# ======================================================================
# decoupled PPO
# ======================================================================

class TestDecoupledPPO:
    def test_reduces_to_standard_ppo_when_mu_equals_pi_old(self):
        logp, logp_old, adv, mask = _policy_batch()
        ours = decoupled_ppo_loss(logp, logp_old, logp_old.clone(), adv, mask, clip=0.2)
        (g_ours,) = torch.autograd.grad(ours["loss"], logp)

        logp2 = logp.detach().clone().requires_grad_(True)
        ref = RLLoss(value_weight=0.0, entropy_weight=0.0, clip_range=0.2)(
            log_probs=logp2, entropy=torch.zeros_like(adv), values=torch.zeros_like(adv),
            returns=torch.zeros_like(adv), advantages=adv, old_log_probs=logp_old, mask=mask)
        (g_ref,) = torch.autograd.grad(ref["loss"], logp2)

        torch.testing.assert_close(ours["loss"], ref["policy_loss"], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(g_ours, g_ref, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(ours["clip_frac"], ref["clip_fraction"])
        assert float(ours["is_weight"]) == 1.0
        assert float(ours["abs_log_old_mu"]) == 0.0
        assert float(ours["is_trunc_frac"]) == 0.0

    def test_textbook_formula(self):
        logp, logp_old, adv, mask = _policy_batch(seed=1, spread=0.5)
        out = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask, clip=0.1, clip_high=0.3)
        r = torch.exp(logp.detach() - logp_old)
        surr = torch.min(r * adv, r.clamp(0.9, 1.3) * adv)
        want = -(surr[mask]).mean()
        torch.testing.assert_close(out["loss"], want)
        clipped = ((r < 0.9) | (r > 1.3))[mask].float().mean()
        torch.testing.assert_close(out["clip_frac"], clipped)

    def test_gradient_is_truncated_is_weight_times_ppo_gradient(self):
        logp, logp_old, adv, mask = _policy_batch(seed=2)
        g = torch.Generator().manual_seed(9)
        logp_mu = logp_old + torch.randn(logp_old.shape, generator=g) * 0.7
        out = decoupled_ppo_loss(logp, logp_old, logp_mu, adv, mask, reduction="sum")
        (g_dec,) = torch.autograd.grad(out["loss"], logp)
        std = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask, reduction="sum")
        (g_std,) = torch.autograd.grad(std["loss"], logp)
        w = torch.exp(logp_old - logp_mu).clamp(max=1.0)
        torch.testing.assert_close(g_dec, w * g_std)
        assert 0.0 < float(out["is_trunc_frac"]) < 1.0
        torch.testing.assert_close(out["abs_log_old_mu"], (logp_old - logp_mu).abs()[mask].mean())
        torch.testing.assert_close(out["is_weight"], w[mask].mean())

    def test_is_clip_levels(self):
        logp, logp_old, adv, mask = _policy_batch(seed=3)
        logp_mu = logp_old - 1.0            # π_old/μ = e for every token
        full = decoupled_ppo_loss(logp, logp_old, logp_mu, adv, mask, is_clip=None,
                                  reduction="sum")["loss"]
        trunc2 = decoupled_ppo_loss(logp, logp_old, logp_mu, adv, mask, is_clip=2.0,
                                    reduction="sum")["loss"]
        std = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask, reduction="sum")["loss"]
        torch.testing.assert_close(full, math.e * std)
        torch.testing.assert_close(trunc2, 2.0 * std)

    def test_first_minibatch_canaries_are_zero(self):
        logp, logp_old, adv, mask = _policy_batch(seed=4)
        out = decoupled_ppo_loss(logp_old.clone().requires_grad_(True), logp_old, logp_old,
                                 adv, mask)
        assert float(out["abs_log_ratio"]) == 0.0
        assert float(out["approx_kl"]) == 0.0
        assert float(out["clip_frac"]) == 0.0

    def test_masked_garbage_never_leaks(self):
        logp, logp_old, adv, mask = _policy_batch(seed=5)
        clean = decoupled_ppo_loss(logp, logp_old, logp_old - 0.2, adv, mask)
        (g_clean,) = torch.autograd.grad(clean["loss"], logp)

        dirty_lp = torch.where(mask, logp.detach(), torch.tensor(float("-inf")))
        dirty_lp.requires_grad_(True)
        dirty_old = torch.where(mask, logp_old, torch.tensor(float("-inf")))
        dirty_mu = torch.where(mask, logp_old - 0.2, torch.tensor(float("nan")))
        dirty_adv = torch.where(mask, adv, torch.tensor(float("nan")))
        dirty = decoupled_ppo_loss(dirty_lp, dirty_old, dirty_mu, dirty_adv, mask)
        (g_dirty,) = torch.autograd.grad(dirty["loss"], dirty_lp)
        torch.testing.assert_close(dirty["loss"], clean["loss"])
        torch.testing.assert_close(g_dirty, g_clean)
        for k in ("approx_kl", "clip_frac", "is_weight", "abs_log_old_mu", "abs_log_ratio"):
            assert torch.isfinite(dirty[k]), k

    def test_policy_mask_trains_subset_diagnostics_full(self):
        logp, logp_old, adv, mask = _policy_batch(seed=6, spread=0.6)
        keep = adv.abs() > 0.8
        out = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask, policy_mask=keep)
        sub = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask & keep)
        full = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask)
        torch.testing.assert_close(out["loss"], sub["loss"])
        torch.testing.assert_close(out["approx_kl"], full["approx_kl"])
        torch.testing.assert_close(out["count"], (mask & keep).sum().float())
        torch.testing.assert_close(out["policy_frac"],
                                   (mask & keep).sum().float() / mask.sum().float())

    def test_dual_clip_bounds_negative_advantage_loss(self):
        logp_old = torch.zeros(3)
        logp = torch.tensor([3.0, 3.0, -3.0], requires_grad=True)   # ratios e^3, e^3, e^-3
        adv = torch.tensor([-1.0, 1.0, -1.0])
        plain = decoupled_ppo_loss(logp, logp_old, logp_old, adv, reduction="sum")["loss"]
        dual = decoupled_ppo_loss(logp, logp_old, logp_old, adv, dual_clip=3.0,
                                  reduction="sum")["loss"]
        # token 0: plain loss = e^3 ≈ 20, dual-clipped to 3; others unchanged
        torch.testing.assert_close(plain - dual, torch.tensor(math.exp(3.0) - 3.0))
        with pytest.raises(ValueError):
            decoupled_ppo_loss(logp, logp_old, logp_old, adv, dual_clip=0.5)

    def test_sum_reduction(self):
        logp, logp_old, adv, mask = _policy_batch(seed=7)
        m = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask)
        s = decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask, reduction="sum")
        torch.testing.assert_close(s["loss"], m["loss"] * m["count"])
        with pytest.raises(ValueError):
            decoupled_ppo_loss(logp, logp_old, logp_old, adv, mask, reduction="max")

    def test_bf16_inputs_compute_in_fp32(self):
        logp, logp_old, adv, mask = _policy_batch(seed=8)
        out = decoupled_ppo_loss(logp.detach().bfloat16().requires_grad_(True), logp_old,
                                 logp_old, adv, mask)
        assert out["loss"].dtype == torch.float32


# ======================================================================
# value / entropy
# ======================================================================

class TestValueLoss:
    def test_mse_matches_reference(self):
        g = torch.Generator().manual_seed(0)
        v, tgt = torch.randn(3, 7, generator=g), torch.randn(3, 7, generator=g)
        mask = torch.rand(3, 7, generator=g) > 0.3
        out = value_loss(v, tgt, mask)
        torch.testing.assert_close(out["loss"], F.mse_loss(v[mask], tgt[mask]))

    def test_huber_matches_generals_critic(self):
        g = torch.Generator().manual_seed(1)
        v, tgt = torch.randn(40, generator=g) * 3, torch.randn(40, generator=g) * 3
        ref = RLLoss(value_weight=1.0, entropy_weight=0.0)(
            log_probs=torch.zeros(40), entropy=torch.zeros(40), values=v, returns=tgt,
            advantages=torch.zeros(40), old_log_probs=torch.zeros(40))
        torch.testing.assert_close(value_loss(v, tgt, kind="huber")["loss"], ref["value_loss"])

    def test_clipped_value_loss(self):
        old = torch.tensor([0.0, 0.0, 0.0, 0.0])
        v = torch.tensor([0.5, 0.5, -0.05, 0.1], requires_grad=True)
        tgt = torch.tensor([1.0, -1.0, 1.0, 0.1])
        out = value_loss(v, tgt, old_pred=old, clip=0.2, reduction="sum")
        # token 0: max((0.5-1)^2, (0.2-1)^2) = 0.64 (clipped branch, no grad)
        # token 1: max((0.5+1)^2, (0.2+1)^2) = 2.25 (unclipped)
        # token 2: inside the clip range: (−0.05−1)^2 both
        # token 3: exact
        want = 0.64 + 2.25 + 1.05 ** 2 + 0.0
        torch.testing.assert_close(out["loss"], torch.tensor(want))
        (grad,) = torch.autograd.grad(out["loss"], v)
        torch.testing.assert_close(grad, torch.tensor([0.0, 3.0, -2.1, 0.0]))
        torch.testing.assert_close(out["clip_frac"], torch.tensor(0.5))
        with pytest.raises(ValueError):
            value_loss(v, tgt, clip=0.2)

    def test_hl_gauss(self):
        g = torch.Generator().manual_seed(2)
        logits = torch.randn(5, 6, 32, generator=g)
        tgt = torch.rand(5, 6, generator=g) * 2 - 1
        tgt[0, 0] = 2.0                             # outside vmax = 1.5
        mask = torch.rand(5, 6, generator=g) > 0.2
        mask[0, 0] = True
        out = hl_gauss_value_loss(logits, tgt, mask, vmax=1.5, sigma=0.06)
        want = hl_gauss_cross_entropy(logits, tgt, 1.5, 0.06)[mask].mean()
        torch.testing.assert_close(out["loss"], want)
        torch.testing.assert_close(out["out_of_support"], 1.0 / mask.sum().float())
        # garbage logits on padded rows: same loss, finite gradient
        dirty = torch.where(mask.unsqueeze(-1), logits, torch.tensor(float("nan")))
        dirty.requires_grad_(True)
        d = hl_gauss_value_loss(dirty, tgt, mask, vmax=1.5, sigma=0.06)
        torch.testing.assert_close(d["loss"], want)
        d["loss"].backward()
        assert torch.isfinite(dirty.grad).all()

    def test_entropy_loss(self):
        h = torch.tensor([1.0, 2.0, 100.0], requires_grad=True)
        mask = torch.tensor([True, True, False])
        out = entropy_loss(h, mask, coef=0.1)
        torch.testing.assert_close(out["loss"], torch.tensor(-0.15))
        torch.testing.assert_close(out["entropy"], torch.tensor(1.5))
        neg = entropy_loss(h, mask, coef=-0.1)       # suppressor
        torch.testing.assert_close(neg["loss"], torch.tensor(0.15))


# ======================================================================
# KL
# ======================================================================

class TestKL:
    def _dists(self, n=7, seed=0):
        g = torch.Generator().manual_seed(seed)
        theta = torch.randn(n, generator=g, dtype=torch.float64).requires_grad_(True)
        ref = torch.randn(n, generator=g, dtype=torch.float64)
        return theta, ref

    @pytest.mark.parametrize("estimator,direction", [("k3", "ref||pi"), ("k2", "pi||ref")])
    def test_sample_estimator_gradients_in_expectation(self, estimator, direction):
        """E_{a~π}[∇ k(a)] computed exactly (every action weighted by the
        detached π): k3 → ∇KL(π_ref‖π), k2 → ∇KL(π‖π_ref)."""
        theta, ref_logits = self._dists()
        logp = F.log_softmax(theta, -1)
        logp_ref = F.log_softmax(ref_logits, -1)
        out = kl_penalty(logp, logp_ref, beta=logp.detach().exp(), estimator=estimator,
                         reduction="sum")
        (g_est,) = torch.autograd.grad(out["loss"], theta)
        kl = categorical_kl(theta, ref_logits, direction=direction)
        (g_true,) = torch.autograd.grad(kl, theta)
        torch.testing.assert_close(g_est, g_true)

    def test_k3_value_is_unbiased_and_nonnegative(self):
        theta, ref_logits = self._dists(seed=1)
        logp = F.log_softmax(theta.detach(), -1)
        logp_ref = F.log_softmax(ref_logits, -1)
        out = kl_penalty(logp, logp_ref, beta=logp.exp(), estimator="k3", reduction="sum")
        want = categorical_kl(theta.detach(), ref_logits, direction="pi||ref")
        torch.testing.assert_close(out["loss"], want)
        k3_each = kl_penalty(logp, logp_ref, beta=1.0, reduction="sum")
        assert float(k3_each["kl_k3"]) >= 0.0
        # π ≪ π_ref on the taken action: large but finite
        far = kl_penalty(torch.tensor([-200.0]), torch.tensor([-0.1]), beta=1.0)
        assert torch.isfinite(far["loss"]) and float(far["loss"]) > 1e8
        with pytest.raises(ValueError):
            kl_penalty(logp, logp_ref, beta=1.0, estimator="k4")

    def test_categorical_kl_formula_and_legal_mask(self):
        g = torch.Generator().manual_seed(3)
        cur = torch.randn(4, 6, generator=g).requires_grad_(True)
        ref = torch.randn(4, 6, generator=g)
        legal = torch.rand(4, 6, generator=g) > 0.4
        legal[:, 0] = True
        legal[3] = False                           # no legal entry: treated as unmasked
        kl = categorical_kl(cur, ref, legal)
        for i in range(4):
            ok = legal[i] if legal[i].any() else torch.ones(6, dtype=torch.bool)
            lp_c = F.log_softmax(cur[i][ok], -1)
            lp_r = F.log_softmax(ref[i][ok], -1)
            torch.testing.assert_close(kl[i], (lp_r.exp() * (lp_r - lp_c)).sum())
        kl.sum().backward()
        assert torch.isfinite(cur.grad).all()
        assert (cur.grad[:3][~legal[:3]] == 0).all()
        assert (categorical_kl(ref, ref, legal).abs() < 1e-6).all()
        # decoder-masked −inf logits, no legal mask, both directions: finite grads
        for direction in ("ref||pi", "pi||ref"):
            pre = torch.where(legal[:3], cur.detach()[:3], torch.tensor(float("-inf")))
            pre.requires_grad_(True)
            ref_pre = torch.where(legal[:3], ref[:3], torch.tensor(float("-inf")))
            kl_pre = categorical_kl(pre, ref_pre, direction=direction)
            torch.testing.assert_close(
                kl_pre, categorical_kl(cur.detach()[:3], ref[:3], legal[:3], direction=direction))
            kl_pre.sum().backward()
            assert torch.isfinite(pre.grad).all()
        with pytest.raises(ValueError):
            categorical_kl(cur, ref, direction="both")


# ======================================================================
# combining games
# ======================================================================

class TestCombine:
    def test_token_and_game_modes(self):
        g = torch.Generator().manual_seed(0)
        x_a, x_b = torch.randn(10, generator=g), torch.randn(30, generator=g)
        ta = entropy_loss(x_a, coef=-1.0, reduction="sum")     # loss = Σ x
        tb = entropy_loss(x_b, coef=-1.0, reduction="sum")
        terms = {"a": ta, "b": tb}
        torch.testing.assert_close(combine_game_losses(terms), torch.cat([x_a, x_b]).mean())
        torch.testing.assert_close(combine_game_losses(terms, mode="game"),
                                   (x_a.mean() + x_b.mean()) / 2)
        torch.testing.assert_close(combine_game_losses(terms, {"a": 3.0}, mode="game"),
                                   (3 * x_a.mean() + x_b.mean()) / 4)
        torch.testing.assert_close(combine_game_losses(terms, {"a": 2.0}),
                                   (2 * x_a.sum() + x_b.sum()) / 50)
        # a global (per-rank average) count replaces the local one
        torch.testing.assert_close(combine_game_losses(terms, counts={"a": 20, "b": 20}),
                                   (x_a.sum() + x_b.sum()) / 40)

    def test_game_without_tokens(self):
        empty = entropy_loss(torch.zeros(0), coef=-1.0, reduction="sum")
        full = entropy_loss(torch.ones(4), coef=-1.0, reduction="sum")
        torch.testing.assert_close(combine_game_losses({"a": empty, "b": full}, mode="game"),
                                   torch.tensor(1.0))
        with pytest.raises(ValueError):
            combine_game_losses({})
