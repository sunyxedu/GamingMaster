"""Phase-7 regime tests.

Covers the anti-stalling fixes layered on the phase-6 regime after the
iter ~1355-1462 collapse (mutual clock-farming + entropy runaway):
  - truncation soft-terminal knobs (material scale + flat stalling penalty)
  - negative entropy weight (active suppression when the ceiling gate fires)
  - argparse backwards-compat defaults
"""

import sys
from unittest import mock

import torch

from env.generals_env import GeneralsEnv
from model.losses import RLLoss
from train_rl import parse_args


# ======================================================================
# Truncation soft-terminal knobs
# ======================================================================

class TestTruncationKnobs:
    def test_defaults_match_legacy_formula(self):
        env = GeneralsEnv()
        env.reset(seed=3)
        assert env.trunc_material_scale == 0.35
        assert env.trunc_penalty == 0.0
        for pid in (0, 1):
            assert env.truncation_score(pid) == 0.35 * env._heuristic(pid)

    def test_penalty_and_scale_applied(self):
        env = GeneralsEnv()
        env.reset(seed=3)
        env.trunc_material_scale = 0.15
        env.trunc_penalty = 0.3
        for pid in (0, 1):
            expected = 0.15 * env._heuristic(pid) - 0.3
            assert abs(env.truncation_score(pid) - expected) < 1e-9

    def test_penalty_charged_to_both_players(self):
        env = GeneralsEnv()
        env.reset(seed=7)
        env.trunc_material_scale = 0.15
        env.trunc_penalty = 0.3
        s = env.truncation_score(0) + env.truncation_score(1)
        material = 0.15 * (env._heuristic(0) + env._heuristic(1))
        assert abs(s - (material - 0.6)) < 1e-9

    def test_leader_strictly_prefers_winning(self):
        # The heuristic is tanh-bounded in (-1, 1): even a maximal lead at
        # the clock scores below scale - penalty, far under the +1 win.
        env = GeneralsEnv()
        env.reset(seed=3)
        env.trunc_material_scale = 0.15
        env.trunc_penalty = 0.3
        best_possible = env.trunc_material_scale * 1.0 - env.trunc_penalty
        assert best_possible == -0.15 < 1.0
        for pid in (0, 1):
            assert env.truncation_score(pid) < best_possible + 1e-9


# ======================================================================
# Negative entropy weight (active suppression above the ceiling)
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


class TestNegativeEntropyWeight:
    def test_negative_weight_raises_loss_by_entropy_term(self):
        inp = _loss_inputs()
        pos = RLLoss(entropy_weight=0.005)(**inp)
        neg = RLLoss(entropy_weight=-0.005)(**inp)
        # loss = ... - w * entropy_bonus, so flipping w by 0.01 raises the
        # total by exactly 0.01 * entropy_bonus (entropy_bonus > 0 here).
        assert neg["entropy"] == pos["entropy"]
        assert torch.allclose(neg["loss"] - pos["loss"],
                              0.01 * pos["entropy"], atol=1e-6)

    def test_gradient_pushes_entropy_down(self):
        inp = _loss_inputs()
        ent = inp["entropy"].clone().requires_grad_(True)
        inp["entropy"] = ent
        out = RLLoss(entropy_weight=-0.005)(**inp)
        out["loss"].backward()
        # d loss / d entropy = +|w|/T on every sample: minimising the loss
        # lowers entropy (the phase-6 zeroing gate could only stop adding).
        assert (ent.grad > 0).all()

    def test_zero_weight_ignores_entropy(self):
        inp = _loss_inputs()
        out_a = RLLoss(entropy_weight=0.0)(**inp)
        inp2 = dict(inp)
        inp2["entropy"] = inp["entropy"] * 10.0
        out_b = RLLoss(entropy_weight=0.0)(**inp2)
        assert torch.allclose(out_a["loss"], out_b["loss"])


# ======================================================================
# Argparse defaults (backwards compatibility)
# ======================================================================

class TestPhase7ArgDefaults:
    def test_new_flags_default_off(self):
        with mock.patch.object(sys, "argv", ["train_rl.py"]):
            args = parse_args()
        assert args.trunc_material == 0.35
        assert args.trunc_penalty == 0.0
        assert args.entropy_penalty == 0.0

    def test_phase7_values_parse(self):
        argv = ["train_rl.py", "--trunc_penalty", "0.3",
                "--trunc_material", "0.15", "--entropy_penalty", "0.005",
                "--entropy_ceiling", "1.9"]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        assert args.trunc_penalty == 0.3
        assert args.trunc_material == 0.15
        assert args.entropy_penalty == 0.005
        assert args.entropy_ceiling == 1.9
