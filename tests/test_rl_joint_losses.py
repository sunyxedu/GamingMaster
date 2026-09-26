"""training/rl_joint/losses.py: decoupled PPO, value, entropy, KL terms."""

import math

import pytest
import torch
import torch.nn.functional as F

from model.losses import RLLoss, hl_gauss_cross_entropy
from training.rl_joint import (
    bernoulli_kl,
    categorical_kl,
    combine_game_losses,
    decoupled_ppo_loss,
    entropy_loss,
    head_terms_kl,
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

    def test_k3_keeps_pulling_toward_the_reference_past_the_cap(self):
        """Past the overflow cap k3 continues linearly: continuous value, and
        the gradient w.r.t. log π stays negative (descent RAISES π(a)), never
        the +1 a clamp inside the exp alone would leave."""
        log_rho = torch.tensor([5.0, 19.0, 19.999, 20.0, 20.001, 21.0, 30.0, 200.0],
                               dtype=torch.float64)
        logp = (-log_rho).clone().requires_grad_(True)      # log π_ref = 0
        out = kl_penalty(logp, torch.zeros_like(logp), beta=1.0, reduction="sum")
        (grad,) = torch.autograd.grad(out["loss"], logp)
        assert (grad < 0).all()
        exact = 1.0 - log_rho.exp()                           # dk3/dlogπ below the cap
        torch.testing.assert_close(grad[:3], exact[:3])
        torch.testing.assert_close(grad[4:], torch.full((4,), 1.0 - math.exp(20.0),
                                                        dtype=torch.float64))
        per = [float(kl_penalty(logp[i:i + 1].detach(), torch.zeros(1, dtype=torch.float64),
                                beta=1.0)["loss"]) for i in range(len(log_rho))]
        assert all(b > a for a, b in zip(per, per[1:]))      # still increasing in log ρ
        slope = math.exp(20.0) - 1.0                          # continuous at the cap
        assert per[3] - per[2] == pytest.approx(slope * 0.001, rel=1e-3)
        assert per[4] - per[3] == pytest.approx(slope * 0.001, rel=1e-3)
        k3 = kl_penalty(logp.detach(), torch.zeros_like(logp), beta=1.0)["kl_k3"]
        assert float(k3) == pytest.approx(sum(per) / len(per), rel=1e-12)

    def test_bernoulli_kl(self):
        g = torch.Generator().manual_seed(5)
        cur = (torch.randn(3, 4, generator=g, dtype=torch.float64) * 3).requires_grad_(True)
        ref = torch.randn(3, 4, generator=g, dtype=torch.float64) * 3
        p, q = torch.sigmoid(ref), torch.sigmoid(cur)
        want = p * (p.log() - q.log()) + (1 - p) * ((1 - p).log() - (1 - q).log())
        torch.testing.assert_close(bernoulli_kl(cur, ref), want)
        torch.testing.assert_close(bernoulli_kl(cur, ref, direction="pi||ref"),
                                   bernoulli_kl(ref, cur.detach()))
        big = torch.tensor([60.0, -60.0], requires_grad=True)
        kl = bernoulli_kl(big, torch.tensor([-60.0, 60.0]))
        kl.sum().backward()
        assert torch.isfinite(kl).all() and torch.isfinite(big.grad).all()


def _two(logit):
    """Bernoulli logit → 2-class logits [0, l] (a masked Bernoulli head)."""
    return torch.stack([torch.zeros_like(logit), logit], -1)


class TestHeadTermsKL:
    @staticmethod
    def _terms(seed, B=3, T=5, N=6, U=2, garbage=False):
        """il_head_terms-shaped dicts: a masked categorical, a multi-unit
        categorical [B, T, U, N], a bernoulli and a masked (2-class)
        bernoulli, with conditional weights."""
        g = torch.Generator().manual_seed(seed)
        src = torch.randn(B, T, N, generator=g, dtype=torch.float64)
        unit = torch.randn(B, T, U, N, generator=g, dtype=torch.float64)
        pas = torch.randn(B, T, generator=g, dtype=torch.float64) * 2
        promo = torch.randn(B, T, generator=g, dtype=torch.float64)
        # weights / masks are shared by both sides (same teacher-forced actions)
        g = torch.Generator().manual_seed(1234)
        legal = torch.rand(B, T, N, generator=g) > 0.3
        legal[..., 0] = True
        w_src = (torch.rand(B, T, generator=g) > 0.2).double()
        w_unit = (torch.rand(B, T, U, generator=g) > 0.4).double()
        w_pass = torch.ones(B, T, dtype=torch.float64)
        w_promo = (torch.rand(B, T, generator=g) > 0.5).double()
        for w in (w_src, w_unit, w_pass, w_promo):
            w[-1, -1] = 0.0                          # a padded frame
        promo_mask = torch.ones(B, T, 2, dtype=torch.bool)
        promo_mask[0, :, 1] = False                  # promotion illegal on row 0
        if garbage:                                   # garbage where unsupervised
            src = torch.where((w_src > 0).unsqueeze(-1), src, torch.tensor(float("nan")))
            pas = torch.where(w_pass > 0, pas, torch.tensor(float("inf")))
        return {
            "src": {"logits": src, "weight": w_src, "norm": w_src.sum().clamp(min=1),
                    "mask": legal},
            "unit": {"logits": unit, "weight": w_unit, "norm": w_unit.sum().clamp(min=1),
                     "scale": 0.5},
            "pass": {"logits": pas, "kind": "bernoulli", "weight": w_pass,
                     "norm": w_pass.sum().clamp(min=1)},
            "promo": {"logits": promo, "kind": "bernoulli", "weight": w_promo,
                      "norm": w_promo.sum().clamp(min=1), "mask": promo_mask},
        }

    def test_matches_manual_composition(self):
        cur = self._terms(0)
        for t in cur.values():
            t["logits"].requires_grad_(True)
        ref = self._terms(1)
        out = head_terms_kl(cur, ref, beta=2.0)
        want = 0.0
        for h, t in cur.items():
            if t.get("kind") == "bernoulli":
                if "mask" in t:
                    kl = categorical_kl(_two(t["logits"]), _two(ref[h]["logits"]), t["mask"])
                else:
                    kl = bernoulli_kl(t["logits"], ref[h]["logits"])
            else:
                kl = categorical_kl(t["logits"], ref[h]["logits"], t.get("mask"))
            want = want + t.get("scale", 1.0) * (kl * t["weight"]).sum() / t["norm"]
        torch.testing.assert_close(out["loss"], 2.0 * want)
        torch.testing.assert_close(out["kl"], want.detach())
        # frames with any supervised head: every frame but the padded one has pass weight
        assert int(out["count"]) == 3 * 5 - 1
        s = head_terms_kl(cur, ref, beta=2.0, reduction="sum")
        torch.testing.assert_close(s["loss"], out["loss"] * out["count"])
        assert set(out) >= {"kl_src", "kl_unit", "kl_pass", "kl_promo"}
        # the promotion KL on the illegal-promotion row is exactly zero
        zero_row = self._terms(0)
        zero_row["promo"]["weight"] = torch.zeros(3, 5, dtype=torch.float64)
        zero_row["promo"]["weight"][0] = 1.0
        z = head_terms_kl({"promo": zero_row["promo"]}, {"promo": ref["promo"]})
        assert float(z["kl"]) == pytest.approx(0.0, abs=1e-12)

    def test_identical_policies_and_gradient(self):
        cur = self._terms(0)
        ref = self._terms(0)
        for t in cur.values():
            t["logits"].requires_grad_(True)
        out = head_terms_kl(cur, ref)
        assert float(out["kl"]) == pytest.approx(0.0, abs=1e-12)
        grads = torch.autograd.grad(out["loss"], [t["logits"] for t in cur.values()])
        for gr in grads:                              # KL minimum: zero gradient
            assert gr.abs().max() < 1e-12

    def test_garbage_in_unsupervised_slots_never_leaks(self):
        cur = self._terms(0, garbage=True)
        for t in cur.values():
            t["logits"].requires_grad_(True)
        ref = self._terms(1, garbage=True)
        for direction in ("ref||pi", "pi||ref"):
            out = head_terms_kl(cur, ref, direction=direction)
            assert torch.isfinite(out["loss"])
            grads = torch.autograd.grad(out["loss"], [t["logits"] for t in cur.values()])
            assert all(torch.isfinite(gr).all() for gr in grads)
            clean = self._terms(0)
            torch.testing.assert_close(
                out["loss"], head_terms_kl(clean, self._terms(1), direction=direction)["loss"])

    def test_validation(self):
        cur, ref = self._terms(0), self._terms(1)
        with pytest.raises(KeyError):
            head_terms_kl(cur, {"src": ref["src"]})
        bad = dict(cur, src=dict(cur["src"], kind="gaussian"))
        with pytest.raises(ValueError, match="kind"):
            head_terms_kl(bad, ref)
        bad = dict(cur, src=dict(cur["src"], weight=torch.ones(3, 4)))
        with pytest.raises(ValueError, match="weight"):
            head_terms_kl(bad, ref)

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

    @pytest.mark.parametrize("mode", ["token", "game"])
    def test_ddp_ranks_with_disjoint_games(self, mode):
        """Two ranks holding DIFFERENT game sets, gradients averaged (DDP):
        with global counts over every game, the averaged gradient is that of
        the global token mean (game mode: of the mean of global per-game
        means) — also when one rank has no tokens of a game at all."""
        g = torch.Generator().manual_seed(1)
        theta = torch.zeros((), dtype=torch.float64, requires_grad=True)
        xs = {("r0", "a"): torch.randn(10, generator=g, dtype=torch.float64),
              ("r0", "b"): torch.randn(30, generator=g, dtype=torch.float64),
              ("r1", "a"): torch.randn(10, generator=g, dtype=torch.float64),
              ("r1", "c"): torch.randn(7, generator=g, dtype=torch.float64)}

        def term(x):          # per-token loss (x + θ)², sum-reduced
            return value_loss(x + theta, torch.zeros_like(x), reduction="sum")

        world = 2
        global_n = {"a": 20.0, "b": 30.0, "c": 7.0, "d": 0.0}   # "d": configured, no data
        counts = {k: v / world for k, v in global_n.items()}
        per_rank = []
        for rank in ("r0", "r1"):
            terms = {gm: term(x) for (r, gm), x in xs.items() if r == rank}
            per_rank.append(combine_game_losses(terms, mode=mode, counts=counts))
        ddp = sum(per_rank) / world
        by_game = {gm: torch.cat([x for (r, k), x in xs.items() if k == gm])
                   for gm in ("a", "b", "c")}
        if mode == "token":
            want = sum(((x + theta) ** 2).sum() for x in by_game.values()) / 57.0
        else:
            want = sum(((x + theta) ** 2).mean() for x in by_game.values()) / 3.0
        torch.testing.assert_close(ddp, want)
        (g_ddp,) = torch.autograd.grad(ddp, theta)
        (g_want,) = torch.autograd.grad(want, theta)
        torch.testing.assert_close(g_ddp, g_want)
        # a game with loss terms must have a count
        with pytest.raises(KeyError, match="counts missing"):
            combine_game_losses({"a": term(xs[("r0", "a")])}, counts={"b": 1.0})
