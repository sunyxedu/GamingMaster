"""Tests for the riichi mahjong game adapter.

Covers the adapter contract against the game-agnostic core:
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling
  6. il_loss: finite, differentiable, robust to degenerate masks

Plus the ARG_USES-specific invariant: the tile head only contributes
log-prob/entropy where the action type actually takes a tile.
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.mahjong import (
    ACTION_PASS,
    MahjongConfig,
    MahjongDecoder,
    MahjongEncoder,
    N_ACTION_TYPES,
    N_MJ_PLAYERS,
    N_TILE_KINDS,
)


# ======================================================================
# Fixtures / helpers
# ======================================================================

@pytest.fixture
def cfg():
    """Tiny config for fast tests."""
    return MahjongConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2,
    )


@pytest.fixture
def model(cfg):
    return GameNetwork(cfg, MahjongEncoder(cfg), MahjongDecoder(cfg))


N_TOKENS = N_TILE_KINDS + N_MJ_PLAYERS + 1  # 39


def _make_obs(cfg, B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "tiles_seq": torch.randn(
            B, T, N_TILE_KINDS, cfg.mahjong_tile_feature_dim, generator=g),
        "players_seq": torch.randn(
            B, T, N_MJ_PLAYERS, cfg.mahjong_player_feature_dim, generator=g),
        "table_seq": torch.randn(B, T, cfg.mahjong_table_feature_dim, generator=g),
    }


def _make_legal(cfg, B, T=None, seed=0):
    """Random legal masks with at least one legal entry per row."""
    g = torch.Generator().manual_seed(seed)
    shape = (B,) if T is None else (B, T)
    legal = {
        "action_type": torch.rand(*shape, N_ACTION_TYPES, generator=g) < 0.5,
        "tile": torch.rand(*shape, N_TILE_KINDS, generator=g) < 0.3,
    }
    for m in legal.values():
        m[..., 0] |= ~m.any(-1)
    return legal


# ======================================================================
# Encoder token layout
# ======================================================================

def test_token_layout(cfg):
    enc = MahjongEncoder(cfg)
    assert enc.n_tokens == N_TOKENS == 39
    assert enc.tile_slice == slice(0, N_TILE_KINDS)
    assert enc.player_slice == slice(N_TILE_KINDS, N_TILE_KINDS + N_MJ_PLAYERS)
    assert enc.table_idx == N_TOKENS - 1
    assert enc.positional_embedding().shape == (1, N_TOKENS, cfg.d_model)
    tokens = enc(_make_obs(cfg, B=2, T=3))
    assert tokens.shape == (2, 3, N_TOKENS, cfg.d_model)


# ======================================================================
# 1. Forward shapes
# ======================================================================

def test_forward_shapes(cfg, model):
    B, T = 2, 3
    out = model(_make_obs(cfg, B, T))
    assert out["action_type_logits"].shape == (B, T, N_ACTION_TYPES)
    assert out["tile_logits"].shape == (B, T, N_TILE_KINDS)
    assert out["action_type_used"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


def test_forward_teacher_forcing_uses_labels(cfg, model):
    B, T = 2, 3
    labels = {"action_type": torch.randint(0, N_ACTION_TYPES, (B, T))}
    out = model(_make_obs(cfg, B, T), labels=labels)
    assert torch.equal(out["action_type_used"], labels["action_type"])


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
        got_signal = any(p.grad.abs().sum() > 0 for p in module.parameters())
        assert got_signal, f"{name} gradients are all zero"


# ======================================================================
# 3. step() chaining with st_cache
# ======================================================================

def test_step_chaining_cache_growth(cfg, model):
    model.eval()
    B = 2
    S = N_TOKENS + 2  # + HL/LL decision tokens
    obs = _make_obs(cfg, B, T=3)
    cache = None
    for t in range(3):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)  # KV cache grows
        assert out["action_type_logits"].shape == (B, N_ACTION_TYPES)
        assert out["tile_logits"].shape == (B, N_TILE_KINDS)
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
    lps, stored = [], {k: [] for k in ("action_type", "tile_idx")}
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
            labels={"action_type": actions["action_type"]},
            act_steps=K,
        )
        lp, ent, head_ent = model.compute_log_prob(full, actions)

    assert torch.equal(full["action_type_used"], actions["action_type"])
    rollout_lp = torch.stack(lps, dim=1)
    torch.testing.assert_close(lp, rollout_lp, atol=1e-5, rtol=1e-5)
    assert ent.shape == (B, T) and (ent >= 0).all()
    assert set(head_ent) == {"ent_type", "ent_tile"}


def test_arg_uses_gating(cfg, model):
    """pass/hora actions must ignore the tile log-prob entirely: total lp
    equals the type lp alone."""
    import torch.nn.functional as F

    model.eval()
    B, T = 2, 3
    obs = _make_obs(cfg, B, T)
    pass_type = torch.full((B, T), ACTION_PASS, dtype=torch.long)
    actions = {
        "action_type": pass_type,
        "tile_idx": torch.randint(0, N_TILE_KINDS, (B, T)),
    }
    with torch.no_grad():
        pred = model(obs, labels={"action_type": pass_type})
        lp, _, _ = model.compute_log_prob(pred, actions)
        type_lp = F.log_softmax(pred["action_type_logits"], dim=-1).gather(
            -1, pass_type.unsqueeze(-1)).squeeze(-1)
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
        assert legal["action_type"][b, out["action_type"]].all(), "illegal type"
        assert legal["tile"][b, out["tile_idx"]].all(), "illegal tile"
        assert torch.isfinite(out["action_log_prob"]).all()


# ======================================================================
# 6. il_loss
# ======================================================================

def _make_tgt(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "action_type": torch.randint(0, N_ACTION_TYPES, (B, T), generator=g),
        "tile_idx": torch.randint(0, N_TILE_KINDS, (B, T), generator=g),
    }


def test_il_loss_finite_and_differentiable(cfg, model):
    B, T = 2, 3
    tgt = _make_tgt(B, T)
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"]})
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False

    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.parameters())
    for k in ("type_loss", "tile_loss", "type_acc", "tile_acc"):
        assert torch.isfinite(out[k]), f"{k} not finite"
        assert not out[k].requires_grad, f"{k} should be detached"


def test_il_loss_degenerate_masks(cfg, model):
    """All-padded sequences and all-pass targets (tile −1 everywhere)
    must not produce NaN."""
    B, T = 2, 3
    tgt = _make_tgt(B, T)
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"]})
    pred = {k: v.detach() for k, v in pred.items()}

    out = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])

    tgt_pass = {
        "action_type": torch.full((B, T), ACTION_PASS, dtype=torch.long),
        "tile_idx": torch.full((B, T), -1),
    }
    out = model.action_decoder.il_loss(pred, tgt_pass, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])
