"""Tests for the Clash Royale game adapter.

Covers the adapter contract against the game-agnostic core:
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling
  6. il_loss: finite, differentiable, robust to degenerate masks
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.clash_royale import (
    CardPlacementDecoder,
    ClashRoyaleConfig,
    ClashRoyaleEncoder,
)


# ======================================================================
# Fixtures / helpers
# ======================================================================

@pytest.fixture
def cfg():
    """Tiny config for fast tests."""
    return ClashRoyaleConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, encoder_channels=24,
        arena_h=8, arena_w=6, arena_channels=5,
        hand_size=4, n_cards=20, card_feature_dim=8, cr_global_features=6,
    )


@pytest.fixture
def model(cfg):
    return GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))


def _make_obs(cfg, B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = {
        "arena_seq": torch.randn(B, T, cfg.arena_channels, cfg.arena_h, cfg.arena_w,
                                 generator=g),
        "hand_ids": torch.randint(0, cfg.n_cards, (B, T, cfg.hand_size), generator=g),
        "hand_feats": torch.randn(B, T, cfg.hand_size, cfg.card_feature_dim, generator=g),
        # Some empty slots so the "empty" embedding participates
        "hand_mask": torch.rand(B, T, cfg.hand_size, generator=g) < 0.8,
        "globals_seq": torch.randn(B, T, cfg.cr_global_features, generator=g),
    }
    obs["hand_mask"][..., 0] = True  # at least one real card
    return obs


def _make_legal(cfg, B, T=None, seed=0):
    """Random legal masks with at least one legal entry per row."""
    g = torch.Generator().manual_seed(seed)
    shape = (B,) if T is None else (B, T)
    n_arena = cfg.arena_h * cfg.arena_w
    card = torch.rand(*shape, cfg.hand_size, generator=g) < 0.5
    place = torch.rand(*shape, n_arena, generator=g) < 0.3
    card[..., 0] |= ~card.any(-1)
    place[..., 0] |= ~place.any(-1)
    return {"card": card, "placement": place}


# ======================================================================
# 1. Forward shapes
# ======================================================================

def test_forward_shapes(cfg, model):
    B, T = 2, 3
    n_arena = cfg.arena_h * cfg.arena_w
    out = model(_make_obs(cfg, B, T))
    assert out["wait_logits"].shape == (B, T)
    assert out["card_logits"].shape == (B, T, cfg.hand_size)
    assert out["placement_logits"].shape == (B, T, n_arena)
    assert out["card_slot"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)
    assert (out["card_slot"] >= 0).all() and (out["card_slot"] < cfg.hand_size).all()


def test_forward_teacher_forcing_uses_labels(cfg, model):
    B, T = 2, 3
    labels = {"card_slot": torch.randint(0, cfg.hand_size, (B, T))}
    out = model(_make_obs(cfg, B, T), labels=labels)
    assert torch.equal(out["card_slot"], labels["card_slot"])


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
    n_arena = cfg.arena_h * cfg.arena_w
    S = model.obs_encoder.n_tokens + 2  # + HL/LL decision tokens
    obs = _make_obs(cfg, B, T=3)
    cache = None
    for t in range(3):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)  # attention KV cache grows
        assert out["wait_logits"].shape == (B,)
        assert out["card_logits"].shape == (B, cfg.hand_size)
        assert out["placement_logits"].shape == (B, n_arena)
        assert out["value"].shape == (B,)
        assert out["q_halt"].shape == (B,)
        assert out["action_log_prob"].shape == (B,)
        assert out["waited"].dtype == torch.bool


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
    lps, slots, places, waits = [], [], [], []
    for t in range(T):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        slots.append(out["card_slot"])
        places.append(out["place_idx"])
        waits.append(out["waited"])

    actions = {
        "card_slot": torch.stack(slots, dim=1),
        "place_idx": torch.stack(places, dim=1),
        "waited": torch.stack(waits, dim=1),
    }
    with torch.no_grad():
        full = model(obs, labels={"card_slot": actions["card_slot"]}, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(full, actions)

    assert torch.equal(full["card_slot"], actions["card_slot"])  # conditioning replayed
    rollout_lp = torch.stack(lps, dim=1)
    torch.testing.assert_close(lp, rollout_lp, atol=1e-5, rtol=1e-5)
    assert ent.shape == (B, T) and (ent >= 0).all()
    assert set(head_ent) == {"ent_wait", "ent_card", "ent_place"}


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
        assert legal["card"][b, out["card_slot"]].all(), "sampled illegal card"
        assert legal["placement"][b, out["place_idx"]].all(), "sampled illegal placement"
        assert torch.isfinite(out["action_log_prob"]).all()


# ======================================================================
# 6. il_loss
# ======================================================================

def _make_tgt(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_arena = cfg.arena_h * cfg.arena_w
    return {
        "waited": torch.rand(B, T, generator=g) < 0.4,
        "card_slot": torch.randint(0, cfg.hand_size, (B, T), generator=g),
        "place_idx": torch.randint(0, n_arena, (B, T), generator=g),
    }


def test_il_loss_finite_and_differentiable(cfg, model):
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T), labels={"card_slot": tgt["card_slot"]})
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False

    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.parameters())
    for k in ("wait_loss", "card_loss", "place_loss",
              "wait_acc", "card_acc", "place_acc"):
        assert torch.isfinite(out[k]), f"{k} not finite"
        assert not out[k].requires_grad, f"{k} should be detached"


def test_il_loss_degenerate_masks(cfg, model):
    """All-padded sequences and all-wait targets must not produce NaN."""
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T), labels={"card_slot": tgt["card_slot"]})

    # Everything masked out
    out = model.action_decoder.il_loss(
        {k: v.detach() for k, v in pred.items()},
        tgt, torch.zeros(B, T, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])

    # Every frame waited: card/place targets are meaningless (−1)
    tgt_wait = {
        "waited": torch.ones(B, T, dtype=torch.bool),
        "card_slot": torch.full((B, T), -1),
        "place_idx": torch.full((B, T), -1),
    }
    out = model.action_decoder.il_loss(
        {k: v.detach() for k, v in pred.items()},
        tgt_wait, torch.ones(B, T, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])
