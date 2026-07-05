"""Tests for the Hearthstone game adapter.

Covers the adapter contract against the game-agnostic core:
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling
  6. il_loss: finite, differentiable, robust to degenerate masks

Plus the ARG_USES-specific invariant: argument heads only contribute
log-prob/entropy where the action type actually uses them.
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.hearthstone import (
    EntityActionDecoder,
    HearthstoneConfig,
    HearthstoneEncoder,
)


# ======================================================================
# Fixtures / helpers
# ======================================================================

@pytest.fixture
def cfg():
    """Tiny config for fast tests."""
    return HearthstoneConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2,
        hand_slots=5, board_slots=3, entity_feature_dim=10,
        n_entity_types=6, hs_global_features=8, n_action_types=4,
    )


@pytest.fixture
def model(cfg):
    return GameNetwork(cfg, HearthstoneEncoder(cfg), EntityActionDecoder(cfg))


def _n_tokens(cfg):
    return cfg.hand_slots + 2 * cfg.board_slots + 3


def _make_obs(cfg, B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = _n_tokens(cfg)
    obs = {
        "entities_seq": torch.randn(B, T, S, cfg.entity_feature_dim, generator=g),
        # Some empty slots so the "empty" embedding participates
        "entity_mask": torch.rand(B, T, S, generator=g) < 0.7,
        "entity_types": torch.randint(0, cfg.n_entity_types, (B, T, S), generator=g),
        "globals_seq": torch.randn(B, T, cfg.hs_global_features, generator=g),
    }
    obs["entity_mask"][..., -3:] = True  # heroes + hero power always present
    return obs


def _make_legal(cfg, B, T=None, seed=0):
    """Random legal masks with at least one legal entry per row."""
    g = torch.Generator().manual_seed(seed)
    S = _n_tokens(cfg)
    shape = (B,) if T is None else (B, T)
    legal = {
        "action_type": torch.rand(*shape, cfg.n_action_types, generator=g) < 0.5,
        "actor": torch.rand(*shape, S, generator=g) < 0.4,
        "target": torch.rand(*shape, S, generator=g) < 0.4,
    }
    for m in legal.values():
        m[..., 0] |= ~m.any(-1)
    return legal


# ======================================================================
# Encoder slot layout
# ======================================================================

def test_slot_layout(cfg):
    enc = HearthstoneEncoder(cfg)
    assert enc.n_tokens == _n_tokens(cfg)
    assert enc.hand_slice == slice(0, cfg.hand_slots)
    assert enc.own_board_slice == slice(cfg.hand_slots, cfg.hand_slots + cfg.board_slots)
    assert enc.hero_power_idx == enc.n_tokens - 1
    assert enc.positional_embedding().shape == (1, enc.n_tokens, cfg.d_model)


# ======================================================================
# 1. Forward shapes
# ======================================================================

def test_forward_shapes(cfg, model):
    B, T = 2, 3
    S = _n_tokens(cfg)
    out = model(_make_obs(cfg, B, T))
    assert out["action_type_logits"].shape == (B, T, cfg.n_action_types)
    assert out["actor_logits"].shape == (B, T, S)
    assert out["target_logits"].shape == (B, T, S)
    assert out["position_logits"].shape == (B, T, cfg.board_slots + 1)
    assert out["action_type_used"].shape == (B, T)
    assert out["actor_idx"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


def test_forward_teacher_forcing_uses_labels(cfg, model):
    B, T = 2, 3
    labels = {
        "action_type": torch.randint(0, cfg.n_action_types, (B, T)),
        "actor_idx": torch.randint(0, _n_tokens(cfg), (B, T)),
    }
    out = model(_make_obs(cfg, B, T), labels=labels)
    assert torch.equal(out["action_type_used"], labels["action_type"])
    assert torch.equal(out["actor_idx"], labels["actor_idx"])


# ======================================================================
# 2. Gradient flow
# ======================================================================

def test_gradient_flow(cfg, model):
    model.train()
    out = model(_make_obs(cfg, B=2, T=3))
    loss = sum(v.sum() for v in out.values() if v.is_floating_point())
    loss.backward()

    for name, module in [
        ("encoder", model.obs_encoder),
        ("decoder", model.action_decoder),
        ("st_layers", model.st_layers),
        ("trm", model.trm),
    ]:
        for pname, p in module.named_parameters():
            assert p.grad is not None, f"{name}.{pname} got no gradient"
        got_signal = any(
            p.grad.abs().sum() > 0 for p in module.parameters()
        )
        assert got_signal, f"{name} gradients are all zero"


# ======================================================================
# 3. step() chaining with st_cache
# ======================================================================

def test_step_chaining_cache_growth(cfg, model):
    model.eval()
    B = 2
    S_game = _n_tokens(cfg)
    S = S_game + 2  # + HL/LL decision tokens
    obs = _make_obs(cfg, B, T=3)
    cache = None
    for t in range(3):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)  # attention KV cache grows
        assert out["action_type_logits"].shape == (B, cfg.n_action_types)
        assert out["actor_logits"].shape == (B, S_game)
        assert out["target_logits"].shape == (B, S_game)
        assert out["position_logits"].shape == (B, cfg.board_slots + 1)
        assert out["value"].shape == (B,)
        assert out["q_halt"].shape == (B,)
        assert out["action_log_prob"].shape == (B,)


# ======================================================================
# 4. PPO exactness: rollout vs teacher-forced replay
# ======================================================================

def test_step_vs_full_consistency_act_steps(cfg, model):
    """With a fixed ACT depth, step() rollout log-probs must equal
    forward()+log_prob_entropy() replay of the stored actions — this is
    what makes the PPO ratio valid."""
    K = 2
    model.eval()
    B, T = 2, 4
    obs = _make_obs(cfg, B, T, seed=42)

    cache = None
    lps, stored = [], {k: [] for k in ("action_type", "actor_idx", "target_idx", "position")}
    for t in range(T):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        for k in stored:
            stored[k].append(out[k])

    actions = {k: torch.stack(v, dim=1) for k, v in stored.items()}
    with torch.no_grad():
        full = model(
            obs,
            labels={"action_type": actions["action_type"],
                    "actor_idx": actions["actor_idx"]},
            act_steps=K,
        )
        lp, ent, head_ent = model.compute_log_prob(full, actions)

    assert torch.equal(full["action_type_used"], actions["action_type"])
    assert torch.equal(full["actor_idx"], actions["actor_idx"])
    rollout_lp = torch.stack(lps, dim=1)
    torch.testing.assert_close(lp, rollout_lp, atol=1e-5, rtol=1e-5)
    assert ent.shape == (B, T) and (ent >= 0).all()
    assert set(head_ent) == {"ent_type", "ent_actor", "ent_target", "ent_position"}


def test_arg_uses_gating(cfg, model):
    """end_turn actions must ignore actor/target/position log-probs
    entirely: total lp equals the type lp alone."""
    import torch.nn.functional as F

    model.eval()
    B, T = 2, 3
    obs = _make_obs(cfg, B, T)
    end_turn = torch.zeros(B, T, dtype=torch.long)   # type 0 = end_turn (F,F,F)
    actions = {
        "action_type": end_turn,
        "actor_idx": torch.randint(0, _n_tokens(cfg), (B, T)),
        "target_idx": torch.randint(0, _n_tokens(cfg), (B, T)),
        "position": torch.randint(0, cfg.board_slots + 1, (B, T)),
    }
    with torch.no_grad():
        pred = model(obs, labels={"action_type": end_turn,
                                  "actor_idx": actions["actor_idx"]})
        lp, _, _ = model.compute_log_prob(pred, actions)
        type_lp = F.log_softmax(pred["action_type_logits"], dim=-1).gather(
            -1, end_turn.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lp, type_lp)


# ======================================================================
# 5. Legal-mask respect
# ======================================================================

def test_sampling_respects_legal_masks(cfg, model):
    model.eval()
    B = 4
    for trial in range(20):
        obs = {k: v[:, 0] for k, v in _make_obs(cfg, B, T=1, seed=trial).items()}
        legal = _make_legal(cfg, B, seed=trial)
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        b = torch.arange(B)
        assert legal["action_type"][b, out["action_type"]].all(), "sampled illegal type"
        assert legal["actor"][b, out["actor_idx"]].all(), "sampled illegal actor"
        assert legal["target"][b, out["target_idx"]].all(), "sampled illegal target"
        assert torch.isfinite(out["action_log_prob"]).all()


# ======================================================================
# 6. il_loss
# ======================================================================

def _make_tgt(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = _n_tokens(cfg)
    return {
        "action_type": torch.randint(0, cfg.n_action_types, (B, T), generator=g),
        "actor_idx": torch.randint(0, S, (B, T), generator=g),
        "target_idx": torch.randint(0, S, (B, T), generator=g),
        "position": torch.randint(0, cfg.board_slots + 1, (B, T), generator=g),
    }


def test_il_loss_finite_and_differentiable(cfg, model):
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"],
                         "actor_idx": tgt["actor_idx"]})
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False

    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.parameters())
    for k in ("type_loss", "actor_loss", "target_loss", "position_loss",
              "type_acc", "actor_acc", "target_acc", "position_acc"):
        assert torch.isfinite(out[k]), f"{k} not finite"
        assert not out[k].requires_grad, f"{k} should be detached"


def test_il_loss_degenerate_masks(cfg, model):
    """All-padded sequences and all-end_turn targets (every argument −1)
    must not produce NaN."""
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"],
                         "actor_idx": tgt["actor_idx"]})
    pred = {k: v.detach() for k, v in pred.items()}

    out = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])

    tgt_end = {
        "action_type": torch.zeros(B, T, dtype=torch.long),  # end_turn
        "actor_idx": torch.full((B, T), -1),
        "target_idx": torch.full((B, T), -1),
        "position": torch.full((B, T), -1),
    }
    out = model.action_decoder.il_loss(pred, tgt_end, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])
