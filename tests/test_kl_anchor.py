"""Full-head KL anchor tests: KL(ref || cur) over source/direction/split/pass.

Identical predictions must give ~0 on every head; perturbing one head must
move exactly that head's KL; masked (-1e9) direction entries must not
contribute; the pass mask must exclude passed steps from dir/split KLs.
"""

import torch

from model.rl_utils import compute_policy_kl_heads


def _make_pred(B=2, T=5, N=625, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "source_logits": torch.randn(B, T, N, generator=g),
        "direction_logits": torch.randn(B, T, 4, generator=g),
        "split_logits": torch.randn(B, T, 2, generator=g),
        "pass_logits": torch.randn(B, T, generator=g),
    }


class TestKLHeads:
    def test_identical_preds_zero_kl(self):
        pred = _make_pred()
        passed = torch.zeros(2, 5, dtype=torch.bool)
        kls = compute_policy_kl_heads(pred, pred, passed=passed)
        for k, v in kls.items():
            assert v.abs() < 1e-6, f"{k} nonzero for identical dists: {v}"

    def test_perturbation_isolated_per_head(self):
        cur = _make_pred(seed=1)
        ref = {k: v.clone() for k, v in cur.items()}
        ref["direction_logits"] = ref["direction_logits"] + torch.randn_like(
            ref["direction_logits"])
        passed = torch.zeros(2, 5, dtype=torch.bool)
        kls = compute_policy_kl_heads(cur, ref, passed=passed)
        assert kls["kl_dir"] > 1e-3
        assert kls["kl_src"].abs() < 1e-6
        assert kls["kl_spl"].abs() < 1e-6
        assert kls["kl_pass"].abs() < 1e-6

    def test_pass_kl_moves_with_pass_logits(self):
        cur = _make_pred(seed=2)
        ref = {k: v.clone() for k, v in cur.items()}
        ref["pass_logits"] = ref["pass_logits"] + 2.0
        kls = compute_policy_kl_heads(cur, ref)
        assert kls["kl_pass"] > 1e-2

    def test_masked_direction_entries_do_not_contribute(self):
        """Entries masked to -1e9 in BOTH dists contribute ~0."""
        cur = _make_pred(seed=3)
        ref = {k: v.clone() for k, v in cur.items()}
        # mask direction 2 everywhere in both — remaining dists identical
        cur["direction_logits"][..., 2] = -1e9
        ref["direction_logits"][..., 2] = -1e9
        passed = torch.zeros(2, 5, dtype=torch.bool)
        kls = compute_policy_kl_heads(cur, ref, passed=passed)
        assert kls["kl_dir"].abs() < 1e-5
        assert torch.isfinite(kls["kl_dir"])

    def test_passed_steps_excluded_from_move_heads(self):
        cur = _make_pred(seed=4)
        ref = {k: v.clone() for k, v in cur.items()}
        # perturb direction ONLY on step 0; mark step 0 as passed
        ref["direction_logits"][:, 0] += 5.0
        passed = torch.zeros(2, 5, dtype=torch.bool)
        passed[:, 0] = True
        kls = compute_policy_kl_heads(cur, ref, passed=passed)
        assert kls["kl_dir"].abs() < 1e-6, \
            "passed step leaked into direction KL"

    def test_gradient_flows_to_current_only(self):
        cur = _make_pred(seed=5)
        for v in cur.values():
            v.requires_grad_(True)
        ref = _make_pred(seed=6)
        passed = torch.zeros(2, 5, dtype=torch.bool)
        kls = compute_policy_kl_heads(cur, ref, passed=passed)
        total = sum(kls.values())
        total.backward()
        for k, v in cur.items():
            assert v.grad is not None and torch.isfinite(v.grad).all(), k


class TestEMA:
    def test_update_ema_math_and_buffers(self):
        import torch.nn as nn
        from train_rl import init_ema, update_ema

        m = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
        ema = init_ema(m)
        before = {k: v.clone() for k, v in ema.items()}
        # perturb weights and BN stats
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
            m[1].running_mean.add_(2.0)
            m[1].num_batches_tracked.add_(3)
        update_ema(ema, m, decay=0.9)
        sd = m.state_dict()
        for k, v in sd.items():
            if v.dtype.is_floating_point:
                expected = 0.9 * before[k] + 0.1 * v.cpu().float()
                assert torch.allclose(ema[k], expected, atol=1e-6), k
            else:
                assert torch.equal(ema[k], v.cpu()), k

    def test_ema_state_loads_strict(self):
        from model.config import ModelConfig
        from model.network import GeneralsNetwork
        from train_rl import init_ema

        cfg = ModelConfig(d_model=64, n_heads=4, trm_H_cycles=1,
                          trm_L_cycles=2, trm_block_layers=1)
        m = GeneralsNetwork(cfg)
        ema = init_ema(m)
        m2 = GeneralsNetwork(cfg)
        m2.load_state_dict(ema)  # strict — must not raise
