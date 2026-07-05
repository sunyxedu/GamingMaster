"""Tests for the Pokémon game adapter.

Covers the adapter contract against the game-agnostic core (mirrors the
Hearthstone suite):
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling (incl. the tera gate)
  6. il_loss: finite, differentiable, robust to degenerate masks

Plus the ARG_USES invariant: the move head + tera Bernoulli contribute
log-prob only on move frames, the switch pointer only on switch frames.
"""

import pytest
import torch
import torch.nn.functional as F

from model.game_network import GameNetwork
from model.games.pokemon import (
    POKE_FEAT_ACTIVE,
    PokemonConfig,
    PokemonDecoder,
    PokemonEncoder,
)


# ======================================================================
# Fixtures / helpers
# ======================================================================

@pytest.fixture
def cfg():
    """Tiny config for fast tests."""
    return PokemonConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, pokemon_species_buckets=64,
    )


@pytest.fixture
def model(cfg):
    return GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))


def _make_obs(cfg, B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    S = 2 * cfg.team_slots
    obs = {
        "species_ids": torch.randint(1, cfg.pokemon_species_buckets, (B, T, S), generator=g),
        "poke_feats": torch.randn(B, T, S, cfg.pokemon_feature_dim, generator=g),
        # some unknown opponent slots so the "empty" embedding participates
        "slot_mask": torch.rand(B, T, S, generator=g) < 0.7,
        "field_seq": torch.randn(B, T, cfg.pokemon_field_features, generator=g),
    }
    obs["slot_mask"][..., : cfg.team_slots] = True   # own team always known
    obs["poke_feats"][..., POKE_FEAT_ACTIVE] = 0.0
    obs["poke_feats"][..., 0, POKE_FEAT_ACTIVE] = 1.0
    return obs


def _make_legal(cfg, B, T=None, seed=0):
    """Random legal masks with at least one legal entry per row."""
    g = torch.Generator().manual_seed(seed)
    shape = (B,) if T is None else (B, T)
    legal = {
        "action_type": torch.rand(*shape, cfg.n_action_types, generator=g) < 0.5,
        "move": torch.rand(*shape, cfg.n_move_slots, generator=g) < 0.5,
        "switch": torch.rand(*shape, cfg.team_slots, generator=g) < 0.4,
        "tera": torch.rand(*shape, generator=g) < 0.5,
        "active_idx": torch.randint(0, cfg.team_slots, shape, generator=g),
    }
    for k in ("action_type", "move", "switch"):
        legal[k][..., 0] |= ~legal[k].any(-1)
    return legal


# ======================================================================
# Encoder slot layout
# ======================================================================

def test_slot_layout(cfg):
    enc = PokemonEncoder(cfg)
    assert enc.n_tokens == 2 * cfg.team_slots + 1 == 13
    assert enc.own_slice == slice(0, cfg.team_slots)
    assert enc.opp_slice == slice(cfg.team_slots, 2 * cfg.team_slots)
    assert enc.field_idx == enc.n_tokens - 1
    assert enc.positional_embedding().shape == (1, enc.n_tokens, cfg.d_model)


def test_encoder_empty_slots_invariant(cfg):
    """Absent slots must be invariant to their (garbage) features."""
    enc = PokemonEncoder(cfg)
    obs = _make_obs(cfg, B=1, T=1)
    obs["slot_mask"][..., 7] = False
    out1 = enc(obs)
    obs2 = {k: v.clone() for k, v in obs.items()}
    obs2["poke_feats"][..., 7, :] = 123.0
    obs2["species_ids"][..., 7] = 3
    out2 = enc(obs2)
    torch.testing.assert_close(out1, out2)


# ======================================================================
# 1. Forward shapes
# ======================================================================

def test_forward_shapes(cfg, model):
    B, T = 2, 3
    out = model(_make_obs(cfg, B, T))
    assert out["action_type_logits"].shape == (B, T, cfg.n_action_types)
    assert out["move_logits"].shape == (B, T, cfg.n_move_slots)
    assert out["switch_logits"].shape == (B, T, cfg.team_slots)
    assert out["tera_logit"].shape == (B, T)
    assert out["action_type_used"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


def test_forward_teacher_forcing_uses_labels(cfg, model):
    B, T = 2, 3
    labels = {
        "action_type": torch.randint(0, cfg.n_action_types, (B, T)),
        "active_idx": torch.randint(0, cfg.team_slots, (B, T)),
    }
    out = model(_make_obs(cfg, B, T), labels=labels)
    assert torch.equal(out["action_type_used"], labels["action_type"])
    assert torch.equal(out["active_idx_used"], labels["active_idx"])


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
    S_game = model.obs_encoder.n_tokens
    S = S_game + 2  # + HL/LL decision tokens
    obs = _make_obs(cfg, B, T=3)
    cache = None
    for t in range(3):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)  # KV cache grows
        assert out["action_type_logits"].shape == (B, cfg.n_action_types)
        assert out["move_logits"].shape == (B, cfg.n_move_slots)
        assert out["switch_logits"].shape == (B, cfg.team_slots)
        assert out["tera_logit"].shape == (B,)
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
    legal_full = _make_legal(cfg, B, T, seed=7)

    cache = None
    lps, stored = [], {k: [] for k in ("action_type", "move_idx", "switch_idx", "tera")}
    for t in range(T):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        step_legal = {k: v[:, t] for k, v in legal_full.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=step_legal, st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        for k in stored:
            stored[k].append(out[k])

    actions = {k: torch.stack(v, dim=1) for k, v in stored.items()}
    with torch.no_grad():
        full = model(
            obs,
            legal=legal_full,
            labels={"action_type": actions["action_type"],
                    "active_idx": legal_full["active_idx"]},
            act_steps=K,
        )
        lp, ent, head_ent = model.compute_log_prob(full, actions, legal_full)

    assert torch.equal(full["action_type_used"], actions["action_type"])
    rollout_lp = torch.stack(lps, dim=1)
    torch.testing.assert_close(lp, rollout_lp, atol=1e-5, rtol=1e-5)
    assert ent.shape == (B, T) and (ent >= 0).all()
    assert set(head_ent) == {"ent_type", "ent_move", "ent_switch", "ent_tera"}


def test_arg_uses_gating(cfg, model):
    """switch actions must ignore move/tera log-probs entirely (total lp
    = type lp + switch lp); move actions must ignore the switch head."""
    model.eval()
    B, T = 2, 3
    obs = _make_obs(cfg, B, T)
    switch_type = torch.ones(B, T, dtype=torch.long)   # type 1 = switch
    actions = {
        "action_type": switch_type,
        "move_idx": torch.randint(0, cfg.n_move_slots, (B, T)),
        "switch_idx": torch.randint(0, cfg.team_slots, (B, T)),
        "tera": torch.randint(0, 2, (B, T)),
    }
    with torch.no_grad():
        pred = model(obs, labels={"action_type": switch_type})
        lp, _, _ = model.compute_log_prob(pred, actions)
        type_lp = F.log_softmax(pred["action_type_logits"], dim=-1).gather(
            -1, switch_type.unsqueeze(-1)).squeeze(-1)
        switch_lp = F.log_softmax(pred["switch_logits"], dim=-1).gather(
            -1, actions["switch_idx"].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lp, type_lp + switch_lp)

    # move actions with the tera gate closed: lp = type + move only
    move_type = torch.zeros(B, T, dtype=torch.long)
    actions["action_type"] = move_type
    legal = {"tera": torch.zeros(B, T, dtype=torch.bool)}
    with torch.no_grad():
        pred = model(obs, labels={"action_type": move_type})
        lp, _, _ = model.compute_log_prob(pred, actions, legal)
        type_lp = F.log_softmax(pred["action_type_logits"], dim=-1).gather(
            -1, move_type.unsqueeze(-1)).squeeze(-1)
        move_lp = F.log_softmax(pred["move_logits"], dim=-1).gather(
            -1, actions["move_idx"].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lp, type_lp + move_lp)


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
        assert legal["move"][b, out["move_idx"]].all(), "sampled illegal move"
        assert legal["switch"][b, out["switch_idx"]].all(), "sampled illegal switch"
        assert (out["tera"].bool() <= legal["tera"]).all(), "tera despite closed gate"
        assert torch.isfinite(out["action_log_prob"]).all()


# ======================================================================
# 6. il_loss
# ======================================================================

def _make_tgt(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "action_type": torch.randint(0, cfg.n_action_types, (B, T), generator=g),
        "move_idx": torch.randint(0, cfg.n_move_slots, (B, T), generator=g),
        "switch_idx": torch.randint(0, cfg.team_slots, (B, T), generator=g),
        "tera": torch.randint(0, 2, (B, T), generator=g),
        "active_idx": torch.randint(0, cfg.team_slots, (B, T), generator=g),
    }


def test_il_loss_finite_and_differentiable(cfg, model):
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"],
                         "active_idx": tgt["active_idx"]})
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False

    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.parameters())
    for k in ("type_loss", "move_loss", "switch_loss", "tera_loss",
              "type_acc", "move_acc", "switch_acc", "tera_acc"):
        assert torch.isfinite(out[k]), f"{k} not finite"
        assert not out[k].requires_grad, f"{k} should be detached"


def test_il_loss_degenerate_masks(cfg, model):
    """All-padded sequences and all-switch targets (every move argument
    −1) must not produce NaN."""
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"],
                         "active_idx": tgt["active_idx"]})
    pred = {k: v.detach() for k, v in pred.items()}

    out = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])

    tgt_switch = {
        "action_type": torch.ones(B, T, dtype=torch.long),  # switch
        "move_idx": torch.full((B, T), -1),
        "switch_idx": torch.randint(0, cfg.team_slots, (B, T)),
        "tera": torch.full((B, T), -1),
    }
    out = model.action_decoder.il_loss(pred, tgt_switch, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])
