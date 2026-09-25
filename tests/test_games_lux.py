"""Tests for the Lux AI-style adapter (LuxEncoder + LuxDecoder).

Covers the frozen adapter contract:
  1. forward output shapes (+ token-layout attrs)
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal stored-action replay
  5. legal-mask respect for the per-unit action head
  6. absent-unit invariance — masked-out slots cannot influence the policy
  7. il_loss finiteness (incl. the zero-units batch)
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.board import _log_prob_and_entropy
from model.games.lux import LuxConfig, LuxDecoder, LuxEncoder


def make_cfg(**overrides) -> LuxConfig:
    kw = dict(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, encoder_channels=16,
        # shrunken game dims for CPU speed (6×6 map → 36 map tokens)
        map_h=6, map_w=6, map_channels=4,
        max_units=5, unit_feature_dim=6, n_unit_actions=6,
        lux_global_features=3,
    )
    kw.update(overrides)
    return LuxConfig(**kw)


def make_obs(cfg: LuxConfig, B: int, T: int) -> dict:
    unit_mask = torch.rand(B, T, cfg.max_units) < 0.6
    unit_mask[..., 0] = True   # keep at least one real unit around
    unit_mask[..., -1] = False  # and at least one guaranteed-empty slot
    return {
        "map_seq": torch.randn(B, T, cfg.map_channels, cfg.map_h, cfg.map_w),
        "units_seq": torch.randn(B, T, cfg.max_units, cfg.unit_feature_dim),
        "unit_mask": unit_mask,
        "globals_seq": torch.randn(B, T, cfg.lux_global_features),
    }


def make_legal(cfg: LuxConfig, obs: dict) -> dict:
    """Per-unit legality with noop always legal + the presence mask."""
    B, T, U = obs["unit_mask"].shape
    m = torch.rand(B, T, U, cfg.n_unit_actions) < 0.5
    m[..., 0] = True  # noop is always legal
    return {"unit_action": m, "unit": obs["unit_mask"].clone()}


def build(cfg: LuxConfig) -> GameNetwork:
    return GameNetwork(cfg, LuxEncoder(cfg), LuxDecoder(cfg))


def assert_grad_flow(model: GameNetwork):
    for name, module in [
        ("obs_encoder", model.obs_encoder),
        ("action_decoder", model.action_decoder),
        ("st_layers", model.st_layers),
        ("trm", model.trm),
    ]:
        got = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in module.parameters()
        )
        assert got, f"no gradient reached {name}"


# ======================================================================
# 1. Forward shapes + token layout
# ======================================================================

def test_forward_shapes():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    M = model.obs_encoder.n_map_tokens
    U = cfg.max_units
    A = cfg.n_unit_actions
    assert M == cfg.map_h * cfg.map_w
    assert model.obs_encoder.n_tokens == M + U
    assert model.action_decoder.n_map_tokens == M
    assert model.obs_encoder.positional_embedding().shape == (1, M + U, cfg.d_model)

    B, T = 2, 3
    obs = make_obs(cfg, B, T)
    with torch.no_grad():
        out = model(obs, legal=make_legal(cfg, obs))
    assert out["unit_action_logits"].shape == (B, T, U, A)
    assert out["unit_present"].shape == (B, T, U)
    assert out["unit_present"].dtype == torch.bool
    assert torch.equal(out["unit_present"], obs["unit_mask"])
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)

    # without legal["unit"], decode defaults to all-present
    with torch.no_grad():
        out = model(obs)
    assert out["unit_present"].all()


# ======================================================================
# 2. Gradient flow
# ======================================================================

def test_gradient_flow():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    out = model(make_obs(cfg, 2, 3))
    loss = out["unit_action_logits"].sum() + out["value"].sum() + out["q_halt"].sum()
    loss.backward()
    assert_grad_flow(model)
    # unit-slot machinery must train
    assert model.obs_encoder.empty_embed.grad.abs().sum() > 0
    assert model.obs_encoder.unit_slot_embed.grad.abs().sum() > 0


# ======================================================================
# 3. step() chaining with Stage-1 cache
# ======================================================================

def test_step_cache_growth():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B = 2
    U = cfg.max_units
    S = model.obs_encoder.n_tokens + 2  # + HL/LL decision tokens

    cache = None
    for t in range(3):
        obs = {k: v[:, 0] for k, v in make_obs(cfg, B, 1).items()}
        with torch.no_grad():
            out = model.step(obs, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)
        assert out["unit_action_logits"].shape == (B, U, cfg.n_unit_actions)
        assert out["unit_present"].shape == (B, U)
        assert out["unit_actions"].shape == (B, U)
        assert out["action_log_prob"].shape == (B,)
        assert out["value"].shape == (B,)


# ======================================================================
# 4. PPO exactness: rollout log-prob == stored-action replay
# ======================================================================

def test_ppo_exactness():
    """step() rollout and forward()+log_prob_entropy replay must define the
    same policy.  No teacher forcing exists (no cross-unit conditioning),
    so replay needs only the stored actions + the same legal masks."""
    torch.manual_seed(1)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T, K = 2, 4, 2
    obs = make_obs(cfg, B, T)
    legal = make_legal(cfg, obs)

    cache = None
    lps, acts = [], []
    for t in range(T):
        with torch.no_grad():
            out = model.step(
                {k: v[:, t] for k, v in obs.items()},
                legal={k: v[:, t] for k, v in legal.items()},
                st_cache=cache, act_steps=K,
            )
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        acts.append(out["unit_actions"])

    rollout_lp = torch.stack(lps, 1)                     # [B, T]
    actions = {"unit_actions": torch.stack(acts, 1)}     # [B, T, U]
    with torch.no_grad():
        pred = model(obs, legal=legal, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(pred, actions, legal)

    assert torch.equal(pred["unit_present"], obs["unit_mask"])
    assert torch.allclose(lp, rollout_lp, atol=1e-5)
    assert ent.shape == (B, T) and torch.isfinite(ent).all()
    assert set(head_ent) == {"ent_unit_action"}
    assert all(v.dim() == 0 and not v.requires_grad for v in head_ent.values())


# ======================================================================
# 5. Legal-mask respect
# ======================================================================

def test_legal_mask_respected():
    """Every present unit's sampled action must be legal, over many draws."""
    torch.manual_seed(3)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B = 16
    for _ in range(10):
        obs_seq = make_obs(cfg, B, 1)
        legal_seq = make_legal(cfg, obs_seq)
        obs = {k: v[:, 0] for k, v in obs_seq.items()}
        legal = {k: v[:, 0] for k, v in legal_seq.items()}
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        chosen_legal = legal["unit_action"].gather(
            -1, out["unit_actions"].unsqueeze(-1),
        ).squeeze(-1)                                    # [B, U]
        present = obs["unit_mask"].bool()
        assert chosen_legal[present].all()


# ======================================================================
# 6. Absent-unit invariance
# ======================================================================

def test_absent_unit_invariance():
    """Masked-out unit slots must be inert: garbage features in absent
    slots cannot change the joint log-prob, and absent units contribute
    exactly 0 to it."""
    torch.manual_seed(5)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T = 2, 3
    U = cfg.max_units
    obs = make_obs(cfg, B, T)
    legal = make_legal(cfg, obs)
    present = obs["unit_mask"].bool()
    assert (~present).any()  # make_obs guarantees an empty slot

    actions = {"unit_actions": torch.randint(0, cfg.n_unit_actions, (B, T, U))}

    # (a) scrambling absent slots' features leaves the policy untouched
    obs2 = {k: v.clone() for k, v in obs.items()}
    obs2["units_seq"] = torch.where(
        present.unsqueeze(-1), obs["units_seq"],
        torch.randn_like(obs["units_seq"]) * 100.0,
    )
    with torch.no_grad():
        pred1 = model(obs, legal=legal)
        pred2 = model(obs2, legal=legal)
        lp1, ent1, _ = model.compute_log_prob(pred1, actions, legal)
        lp2, ent2, _ = model.compute_log_prob(pred2, actions, legal)
    assert torch.allclose(lp1, lp2, atol=1e-6)
    assert torch.allclose(ent1, ent2, atol=1e-6)

    # (b) absent units contribute exactly 0: changing THEIR stored actions
    # never moves the joint lp, which equals the sum over present units
    actions3 = {
        "unit_actions": torch.where(
            present, actions["unit_actions"],
            (actions["unit_actions"] + 1) % cfg.n_unit_actions,
        ),
    }
    with torch.no_grad():
        lp3, _, _ = model.compute_log_prob(pred1, actions3, legal)
        unit_lp, _ = _log_prob_and_entropy(
            pred1["unit_action_logits"], actions["unit_actions"],
            legal["unit_action"],
        )
        # gate with where (not multiply): an absent unit whose arbitrary
        # stored action is illegal has unit_lp = -inf, and -inf * 0 = nan
        expected = torch.where(
            present, unit_lp, torch.zeros_like(unit_lp),
        ).sum(-1)
    assert torch.equal(lp1, lp3)
    assert torch.allclose(lp1, expected, atol=1e-6)


# ======================================================================
# 7. il_loss
# ======================================================================

def test_il_loss():
    torch.manual_seed(4)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    obs = make_obs(cfg, B, T)
    legal = make_legal(cfg, obs)
    pred = model(obs, legal=legal)
    tgt = {"unit_actions": torch.randint(0, cfg.n_unit_actions, (B, T, cfg.max_units))}
    full = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    empty = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))

    assert torch.isfinite(full["loss"])
    for k in ("unit_action_loss", "unit_action_acc"):
        assert k in full and torch.isfinite(full[k]) and not full[k].requires_grad
    # fully-masked-out batch: zero loss, never NaN
    assert torch.isfinite(empty["loss"])
    assert empty["loss"].item() == pytest.approx(0.0, abs=1e-6)

    (full["loss"] + empty["loss"]).backward()
    assert_grad_flow(model)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_il_loss_zero_units():
    """A batch where NO unit exists anywhere must still give a finite zero."""
    torch.manual_seed(6)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T = 2, 2
    obs = make_obs(cfg, B, T)
    obs["unit_mask"] = torch.zeros(B, T, cfg.max_units, dtype=torch.bool)
    legal = make_legal(cfg, obs)
    with torch.no_grad():
        pred = model(obs, legal=legal)
    assert not pred["unit_present"].any()
    tgt = {"unit_actions": torch.randint(0, cfg.n_unit_actions, (B, T, cfg.max_units))}
    out = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)
    # and the joint log-prob of a unit-less turn is exactly 0
    with torch.no_grad():
        lp, ent, _ = model.compute_log_prob(pred, tgt, legal)
    assert torch.equal(lp, torch.zeros(B, T))
    assert torch.equal(ent, torch.zeros(B, T))
