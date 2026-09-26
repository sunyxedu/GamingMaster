"""training/rl_joint: advantage normaliser, adaptive controllers, top-q filter,
token-quota scheduler, diagnostics."""

import math

import pytest
import torch

from training.rl_joint import (
    AdaptiveEntropyCoef,
    AdaptiveKLBeta,
    AdvantageNormalizer,
    LearningProgressBandit,
    RunningMoments,
    ShareProvider,
    TokenQuotaScheduler,
    expert_load_entropy,
    explained_variance,
    largest_remainder,
    per_game_moments,
    project_shares,
    top_q_mask,
)

GAMES = ["chess", "generals", "othello"]


# ======================================================================
# advantage normaliser
# ======================================================================

class TestAdvantageNormalizer:
    def test_first_update_is_exact_and_games_are_independent(self):
        norm = AdvantageNormalizer(GAMES, decay=0.99)
        g = torch.Generator().manual_seed(0)
        big = torch.randn(500, generator=g) * 100 + 30
        small = torch.randn(3, 40, generator=g) * 0.01
        mask = torch.rand(3, 40, generator=g) > 0.25
        small[~mask] = float("nan")                  # padding garbage is excluded
        norm.update({"generals": big, "othello": (small, mask)}, sync=False)
        assert norm.mean("generals") == pytest.approx(float(big.double().mean()), rel=1e-9)
        assert norm.std("generals") == pytest.approx(float(big.double().std(unbiased=False)),
                                                     rel=1e-6)
        out = norm.normalize("othello", small, mask)
        assert torch.isfinite(out).all() and (out[~mask] == 0).all()
        assert float(out[mask].mean()) == pytest.approx(0.0, abs=1e-5)
        assert float(out[mask].std(unbiased=False)) == pytest.approx(1.0, rel=1e-4)
        # a game without data stays the identity
        x = torch.randn(7)
        torch.testing.assert_close(norm.normalize("chess", x), x)
        assert norm.moments.count("chess") == 0

    def test_running_statistics_follow_a_drift(self):
        norm = AdvantageNormalizer(["a"], decay=0.5)
        g = torch.Generator().manual_seed(1)
        for _ in range(20):
            norm.update({"a": torch.randn(2000, generator=g) * 10.0 - 5.0}, sync=False)
        assert norm.std("a") == pytest.approx(10.0, rel=0.1)
        for _ in range(20):                          # the advantage scale shifts
            norm.update({"a": torch.randn(2000, generator=g) * 3.0 + 1.0}, sync=False)
        assert norm.std("a") == pytest.approx(3.0, rel=0.1)
        assert norm.mean("a") == pytest.approx(1.0, abs=0.3)

    def test_uncentered(self):
        norm = AdvantageNormalizer(["a"], center=False)
        x = torch.tensor([1.0, 3.0])
        norm.update({"a": x}, sync=False)
        torch.testing.assert_close(norm.normalize("a", x), x / 1.0)   # std of {1, 3} = 1

    def test_state_dict_round_trip_and_mismatch(self):
        norm = AdvantageNormalizer(GAMES, decay=0.9)
        for i in range(3):
            norm.update({"chess": torch.randn(50) + i}, sync=False)
        fresh = AdvantageNormalizer(GAMES, decay=0.9)
        fresh.load_state_dict(norm.state_dict())
        for game in GAMES:
            assert fresh.mean(game) == norm.mean(game) and fresh.std(game) == norm.std(game)
        other = AdvantageNormalizer(["chess", "go"], decay=0.5)   # a retuned decay
        with pytest.raises(KeyError):
            other.load_state_dict(norm.state_dict())
        other.load_state_dict(norm.state_dict(), strict=False)
        assert other.mean("chess") == norm.mean("chess") and other.mean("go") == 0.0

    def test_validation(self):
        with pytest.raises(ValueError):
            RunningMoments([])
        with pytest.raises(ValueError):
            RunningMoments(["a", "a"])
        with pytest.raises(ValueError):
            RunningMoments(["a"], decay=1.0)
        with pytest.raises(KeyError):
            RunningMoments(["a", "b"], decay={"a": 0.9})
        with pytest.raises(KeyError):
            per_game_moments({"zzz": torch.ones(2)}, ["a"], sync=False)
        with pytest.raises(ValueError, match="non-finite"):
            per_game_moments({"a": torch.tensor([1.0, float("inf")])}, ["a"], sync=False)


# ======================================================================
# controllers
# ======================================================================

class TestEntropyController:
    def test_log_mode_converges_to_target(self):
        """Plant: the policy's entropy rises with the bonus, H = 1 + 0.3·log(α/0.01)."""
        ctl = AdaptiveEntropyCoef(GAMES, {"chess": 1.5, "generals": 0.6, "othello": 1.0},
                                  init=0.01, lr=0.5, min_coef=1e-6, max_coef=10.0)

        def plant(a):
            return 1.0 + 0.3 * math.log(a / 0.01)

        for _ in range(200):
            ctl.update({g: plant(ctl[g]) for g in GAMES})
        for g, target in ctl.target.items():
            assert plant(ctl[g]) == pytest.approx(target, abs=1e-6)

    def test_linear_mode_turns_into_a_suppressor(self):
        ctl = AdaptiveEntropyCoef(["a"], 1.0, init=0.01, lr=0.02, min_coef=-0.05,
                                  max_coef=0.05, mode="linear")
        for _ in range(20):
            ctl.update({"a": 3.0})                 # entropy runaway above the target
        assert ctl["a"] == -0.05                   # clamped at the (negative) floor

    def test_tolerance_bounds_and_partial_updates(self):
        ctl = AdaptiveEntropyCoef(GAMES, 1.0, init=0.02, lr=1.0, tolerance=0.1, max_coef=0.03)
        ctl.update({"chess": 1.05})                # inside the dead band
        assert ctl["chess"] == 0.02
        ctl.update({"chess": 0.0})                 # far below target: hits the cap
        assert ctl["chess"] == 0.03 and ctl["othello"] == 0.02
        with pytest.raises(KeyError):
            ctl.update({"zzz": 1.0})
        with pytest.raises(ValueError):
            ctl.update({"chess": float("nan")})
        with pytest.raises(ValueError):
            AdaptiveEntropyCoef(["a"], 1.0, min_coef=0.0)
        with pytest.raises(ValueError):
            AdaptiveEntropyCoef(["a"], 1.0, mode="pid")


class TestKLController:
    @pytest.mark.parametrize("mode", ["proportional", "step"])
    def test_converges_to_budget(self, mode):
        """Plant: KL to the reference falls as the anchor strengthens, KL = a/β."""
        a = {"chess": 0.02, "generals": 0.5, "othello": 0.003}
        budget = {"chess": 0.01, "generals": 0.05, "othello": 0.02}
        ctl = AdaptiveKLBeta(GAMES, budget, init=0.1, mode=mode, min_beta=1e-6, max_beta=1e3)
        for _ in range(400):
            ctl.update({g: a[g] / ctl[g] for g in GAMES})
        for g in GAMES:
            kl = a[g] / ctl[g]
            if mode == "proportional":
                assert kl == pytest.approx(budget[g], rel=1e-3)
            else:                                  # settles inside the tolerance band
                assert budget[g] / 1.5 <= kl <= budget[g] * 1.5 * 2

    def test_proportional_step_is_bounded(self):
        ctl = AdaptiveKLBeta(["a"], 0.01, init=1.0, rate=0.5, err_clip=0.2)
        ctl.update({"a": 100.0})
        assert ctl["a"] == pytest.approx(1.1)
        ctl.update({"a": 0.0})
        assert ctl["a"] == pytest.approx(1.1 * 0.9)

    def test_state_dict_round_trip(self):
        ctl = AdaptiveKLBeta(GAMES, 0.02)
        ent = AdaptiveEntropyCoef(GAMES, 1.0)
        for i in range(5):
            ctl.update({"chess": 0.1 * i, "othello": 0.001})
            ent.update({"generals": 0.2 * i})
        for c, cls, arg in ((ctl, AdaptiveKLBeta, 0.02), (ent, AdaptiveEntropyCoef, 1.0)):
            fresh = cls(GAMES, arg)
            fresh.load_state_dict(c.state_dict())
            assert fresh.as_dict() == c.as_dict()
            with pytest.raises(KeyError):
                cls(["chess"], arg).load_state_dict(c.state_dict())
            part = cls(["chess", "go"], arg)
            part.load_state_dict(c.state_dict(), strict=False)
            assert part["chess"] == c["chess"]

    def test_validation(self):
        with pytest.raises(ValueError):
            AdaptiveKLBeta(["a"], 0.0)
        with pytest.raises(ValueError):
            AdaptiveKLBeta(["a"], 0.1, min_beta=0.0)
        with pytest.raises(ValueError):
            AdaptiveKLBeta(["a"], 0.1, min_beta=2.0, max_beta=1.0)
        with pytest.raises(KeyError):
            AdaptiveKLBeta(["a", "b"], {"a": 0.1})


# ======================================================================
# top-q filter
# ======================================================================

class TestTopQ:
    def test_global_count_and_order(self):
        adv = torch.tensor([0.1, -3.0, 0.5, 2.0, -0.2, 0.05, 1.0, -0.7, 0.3, 0.0, 4.0, -1.5])
        keep = top_q_mask(adv, q=0.25)
        assert keep.sum() == 3
        assert set(keep.nonzero().flatten().tolist()) == {1, 3, 10}
        signed = top_q_mask(adv, q=0.25, key="signed")
        assert set(signed.nonzero().flatten().tolist()) == {3, 6, 10}

    def test_mask_ties_small_groups_per_row(self):
        adv = torch.tensor([[5.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1, 9.0],
                            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])
        mask = torch.ones_like(adv, dtype=torch.bool)
        mask[0, -1] = False                        # the biggest |A| is padding
        keep = top_q_mask(adv, mask, q=0.25, per_row=True)
        # row 0: 8 valid, k = 2 → {5.0, 1.0-tie ×3} (ties kept), never the padded 9.0
        assert keep[0].tolist() == [True, True, True, True] + [False] * 5
        # row 1: 9 valid, k = round(2.25) = 2 → {9, 8}
        assert keep[1].nonzero().flatten().tolist() == [7, 8]
        few = top_q_mask(adv[:, :4], q=0.25, per_row=True, min_count=8)
        assert few.all()                           # groups below min_count keep everything
        assert torch.equal(top_q_mask(adv, mask, q=1.0), mask)
        assert not top_q_mask(adv, torch.zeros_like(mask), q=0.25).any()
        with pytest.raises(ValueError):
            top_q_mask(adv, q=0.0)
        with pytest.raises(ValueError):
            top_q_mask(adv, key="other")

    def test_matches_generals_trainer_rule(self):
        """Same kthvalue rule as training/train_rl.py's per-episode filter."""
        g = torch.Generator().manual_seed(0)
        adv = torch.randn(1, 200, generator=g)
        n = adv.numel()
        k = max(1, int(round(n * 0.25)))
        thresh = adv.abs().flatten().kthvalue(n - k + 1).values
        assert torch.equal(top_q_mask(adv, q=0.25, per_row=True), adv.abs() >= thresh)


# ======================================================================
# token quotas
# ======================================================================

class TestProjectShares:
    def test_no_bounds_is_normalisation(self):
        p = project_shares({"a": 1.0, "b": 3.0})
        assert p == pytest.approx({"a": 0.25, "b": 0.75})

    def test_floor_and_cap_keep_relative_weights(self):
        p = project_shares({"a": 100.0, "b": 1.0, "c": 2.0, "d": 3.0}, floor=0.1, cap=0.5)
        assert sum(p.values()) == pytest.approx(1.0)
        assert p["a"] == pytest.approx(0.5)
        assert p["b"] == pytest.approx(0.1)        # raised to its floor
        assert p["d"] / p["c"] == pytest.approx(1.5)
        assert all(0.1 - 1e-12 <= v <= 0.5 + 1e-12 for v in p.values())

    def test_edge_cases(self):
        assert project_shares({"a": 0.0, "b": 0.0}) == pytest.approx({"a": 0.5, "b": 0.5})
        # positive-weight game capped, the zero-weight games absorb the rest
        p = project_shares({"a": 1.0, "b": 0.0, "c": 0.0}, cap=0.4)
        assert p["a"] == pytest.approx(0.4) and p["b"] == pytest.approx(0.3)
        with pytest.raises(ValueError, match="infeasible"):
            project_shares({"a": 1.0, "b": 1.0}, floor=0.6)
        with pytest.raises(ValueError, match="infeasible"):
            project_shares({"a": 1.0, "b": 1.0}, cap=0.4)
        with pytest.raises(ValueError):
            project_shares({"a": -1.0})

    def test_largest_remainder(self):
        q = largest_remainder({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}, 100)
        assert sum(q.values()) == 100 and q == {"a": 34, "b": 33, "c": 33}
        assert largest_remainder({"a": 0.5, "b": 0.5}, 0) == {"a": 0, "b": 0}


class TestTokenQuotaScheduler:
    def test_static_quotas(self):
        sched = TokenQuotaScheduler({"chess": 2, "generals": 1, "othello": 1})
        q = sched.quotas(4000)
        assert q == {"chess": 2000, "generals": 1000, "othello": 1000}
        # Hamilton: 3.5 / 1.75 / 1.75 → floors 3/1/1, the two largest remainders win
        assert sched.allocate(7) == {"chess": 3, "generals": 2, "othello": 2}

    def test_debt_tops_up_an_under_served_game(self):
        sched = TokenQuotaScheduler({"a": 1, "b": 1}, floor=0.1, cap=0.9, debt_decay=1.0)
        sched.record({"a": 1000, "b": 200})        # b's long episodes did not finish
        q = sched.quotas(1000)
        # b is owed 400 tokens: ideal allotment 100 (a) / 900 (b), cap-bounded
        assert q == {"a": 100, "b": 900}
        no_debt = TokenQuotaScheduler({"a": 1, "b": 1}, carry_debt=False)
        no_debt.record({"a": 1000, "b": 200})
        assert no_debt.quotas(1000) == {"a": 500, "b": 500}

    def test_long_run_token_fractions_match_shares(self):
        shares = {"chess": 0.5, "generals": 0.3, "othello": 0.2}
        sched = TokenQuotaScheduler(shares, floor=0.05, cap=0.8, debt_decay=0.95)
        g = torch.Generator().manual_seed(0)
        for _ in range(300):
            q = sched.quotas(10_000)
            # the actors deliver a noisy, biased fraction of every quota
            got = {k: int(v * float(torch.rand(1, generator=g) * 0.6 + 0.4)) for k, v in q.items()}
            sched.record(got)
        total = sum(sched.total_tokens.values())
        for k, s in shares.items():
            assert sched.total_tokens[k] / total == pytest.approx(s, abs=0.01)

    def test_bandit_hook_shifts_shares_within_bounds(self):
        bandit = LearningProgressBandit(GAMES, eps=0.0)
        assert isinstance(bandit, ShareProvider)
        sched = TokenQuotaScheduler({g: 1 for g in GAMES}, floor=0.1, cap=0.7,
                                    bandit=bandit, bandit_mix=1.0)
        for i in range(20):
            bandit.update({"chess": 0.5, "generals": 0.5 + 0.02 * i, "othello": 0.5})
        s = sched.shares()
        assert s["generals"] == pytest.approx(0.7)       # the progressing game, capped
        assert s["chess"] == pytest.approx(0.15) and s["othello"] == pytest.approx(0.15)
        half = TokenQuotaScheduler({g: 1 for g in GAMES}, bandit=bandit, bandit_mix=0.5)
        assert half.shares()["chess"] == pytest.approx(0.5 / 3)

    def test_bandit_optimism_and_forgetting(self):
        bandit = LearningProgressBandit(["a", "b"], eps=0.2)
        for i in range(10):
            bandit.update({"a": 1.0 - 0.05 * i})       # a is being forgotten
        w = bandit.weights()
        assert w["a"] == pytest.approx(w["b"])          # b never measured: optimistic
        bandit.update({"b": 0.3})
        bandit.update({"b": 0.3})
        w = bandit.weights()
        assert w["a"] > w["b"] and w["b"] == pytest.approx(0.1)   # only the ε floor

    def test_sample_is_reproducible(self):
        sched = TokenQuotaScheduler({"a": 3, "b": 1})
        d1 = sched.sample(400, torch.Generator().manual_seed(1))
        d2 = sched.sample(400, torch.Generator().manual_seed(1))
        assert d1 == d2 and d1.count("a") / 400 == pytest.approx(0.75, abs=0.07)

    def test_state_dict_round_trip(self):
        bandit = LearningProgressBandit(GAMES)
        sched = TokenQuotaScheduler({g: 1 for g in GAMES}, bandit=bandit, bandit_mix=0.5)
        for i in range(4):
            sched.record(sched.quotas(999))
            bandit.update({"chess": 0.1 * i, "othello": 0.3})
        fresh = TokenQuotaScheduler({g: 1 for g in GAMES},
                                    bandit=LearningProgressBandit(GAMES), bandit_mix=0.5)
        fresh.load_state_dict(sched.state_dict())
        assert fresh.quotas(12345) == sched.quotas(12345)
        assert fresh.shares() == sched.shares() and fresh.iterations == 4
        with pytest.raises(KeyError):
            TokenQuotaScheduler({"chess": 1}).load_state_dict(sched.state_dict())

    def test_validation(self):
        with pytest.raises(ValueError):
            TokenQuotaScheduler({})
        with pytest.raises(ValueError):
            TokenQuotaScheduler({"a": 1, "b": 1}, floor=0.7)
        with pytest.raises(TypeError):
            TokenQuotaScheduler({"a": 1}, bandit=object())
        sched = TokenQuotaScheduler({"a": 1})
        with pytest.raises(KeyError):
            sched.record({"b": 1})
        with pytest.raises(ValueError):
            sched.record({"a": -1})


# ======================================================================
# diagnostics
# ======================================================================

def test_explained_variance():
    t = torch.randn(100)
    assert explained_variance(t, t) == pytest.approx(1.0)
    assert explained_variance(torch.zeros(100), t) == pytest.approx(0.0, abs=1e-6)
    assert math.isnan(explained_variance(t, torch.ones(100)))


def test_expert_load_entropy():
    loads = torch.tensor([[5.0, 5.0, 5.0, 5.0], [8.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    h = expert_load_entropy(loads)
    assert h[0] == pytest.approx(1.0) and h[1] == pytest.approx(0.0) and math.isnan(h[2])
