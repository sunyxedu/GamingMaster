"""Tests for the Lux AI-style adapter (LuxEncoder + LuxDecoder).

Covers the frozen adapter contract:
  1. forward output shapes (+ token-layout attrs)
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal stored-action replay
  5. legal-mask respect for the per-unit action head
  6. absent-unit invariance — masked-out slots cannot influence the policy
  7. il_loss finiteness (incl. the zero-units batch)
  8. sap target head (lux_sap_range > 0): legacy configs unchanged, offset
     encoding + legality helper, rollout/replay exactness with legal['sap'],
     il_loss with sap targets, il_head_terms reconstruction, z-loss
"""

import pytest
import torch

from model.config import config_from_dict
from model.game_network import GameNetwork
from model.games.base import masked_sample
from model.games.board import _log_prob_and_entropy
from model.games.lux import (
    LuxConfig,
    LuxDecoder,
    LuxEncoder,
    lux_sap_delta,
    lux_sap_legal,
    lux_sap_offset,
)
from training.train_il_game import logit_z_loss


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


# ======================================================================
# 8. Sap target head
# ======================================================================

R = 7
N_OFF = (2 * R + 1) ** 2
SAP = 5


def make_sap_obs_legal(cfg: LuxConfig, B: int, T: int, sap_range: int = 3):
    """obs + legal with integer unit cells and lux_sap_legal sap masks."""
    obs = make_obs(cfg, B, T)
    legal = make_legal(cfg, obs)
    xy = torch.stack([torch.randint(0, cfg.map_w, (B, T, cfg.max_units)),
                      torch.randint(0, cfg.map_h, (B, T, cfg.max_units))], -1)
    legal["sap"] = lux_sap_legal(xy, obs["unit_mask"], sap_range, R, cfg.map_w, cfg.map_h)
    return obs, legal


def make_sap_tgt(cfg: LuxConfig, legal: dict, B: int, T: int):
    """Unit-action targets with ~half saps, legal sap offsets (a few −1)."""
    ua = torch.randint(0, cfg.n_unit_actions, (B, T, cfg.max_units))
    ua[..., 1::2] = SAP
    logits = torch.randn(B, T, cfg.max_units, N_OFF)
    so = masked_sample(logits.reshape(-1, N_OFF), legal["sap"].reshape(-1, N_OFF)
                       ).reshape(B, T, cfg.max_units)
    so[0, 0] = -1
    return {"unit_actions": ua, "sap_offset": torch.where(ua == SAP, so, torch.full_like(so, -1))}


def reconstruct(terms):
    return sum(h.get("scale", 1.0) * (h["ce"] * h["weight"]).sum() / h["norm"]
               for h in terms.values() if h["ce"] is not None)


def test_legacy_config_builds_the_legacy_decoder():
    """lux_sap_range defaults to 0: the same modules / state_dict keys as
    before the field existed (old checkpoints load strictly), no sap output."""
    cfg = make_cfg()
    assert cfg.lux_sap_range == 0 and cfg.lux_sap_action == 5
    dec = LuxDecoder(cfg)
    assert dec.sap_head is None
    assert set(dec.state_dict()) == {
        "action_head.net.0.weight", "action_head.net.0.bias",
        "action_head.net.2.weight", "action_head.net.2.bias",
    }
    # a checkpoint config written before the field existed
    d = {k: v for k, v in vars(make_cfg()).items() if k not in ("lux_sap_range", "lux_sap_action")}
    old = config_from_dict(d, LuxConfig)
    assert old.lux_sap_range == 0
    torch.manual_seed(0)
    model = build(old)
    fresh = build(make_cfg())
    fresh.load_state_dict(model.state_dict(), strict=True)
    obs = make_obs(old, 2, 2)
    with torch.no_grad():
        out = model(obs)
    assert "sap_logits" not in out and "sap_mask" not in out
    tgt = {"unit_actions": torch.randint(0, 6, (2, 2, old.max_units)),
           "sap_offset": torch.zeros(2, 2, old.max_units, dtype=torch.long)}
    res = model.action_decoder.il_loss(out, tgt, torch.ones(2, 2, dtype=torch.bool))
    assert "sap_loss" not in res and torch.equal(res["loss"], res["unit_action_loss"])
    assert set(model.action_decoder.il_head_terms(out, tgt, torch.ones(2, 2, dtype=torch.bool))
               ) == {"unit_action"}


def test_sap_config_validation():
    with pytest.raises(ValueError):
        LuxDecoder(make_cfg(lux_sap_range=R, lux_sap_action=6))
    with pytest.raises(ValueError):
        LuxDecoder(make_cfg(lux_sap_range=-1))


def test_sap_offset_encoding_and_legality():
    for o in range(N_OFF):
        dx, dy = lux_sap_delta(o, R)
        assert abs(dx) <= R and abs(dy) <= R
        assert lux_sap_offset(dx, dy, R) == o
    assert lux_sap_offset(2, -1, R) == 99            # the converter fixture's [9, 2, -1]
    assert lux_sap_delta(99, R) == (2, -1)
    o = torch.arange(N_OFF)
    dx, dy = lux_sap_delta(o, R)
    assert torch.equal(lux_sap_offset(dx, dy, R), o)

    xy = torch.tensor([[[0, 0], [5, 5], [2, 3]]])    # [1, U=3, 2]
    mask = torch.tensor([[True, True, False]])
    legal = lux_sap_legal(xy, mask, 2, R, map_w=6, map_h=6)
    assert legal.shape == (1, 3, N_OFF) and legal.dtype == torch.bool
    ddx, ddy = lux_sap_delta(torch.arange(N_OFF), R)
    want0 = (ddx >= 0) & (ddx <= 2) & (ddy >= 0) & (ddy <= 2)       # corner: 3×3
    want1 = (ddx >= -2) & (ddx <= 0) & (ddy >= -2) & (ddy <= 0)
    assert torch.equal(legal[0, 0], want0) and torch.equal(legal[0, 1], want1)
    assert not legal[0, 2].any()                                      # absent unit
    # per-episode range tensor broadcast over the leading dims
    legal2 = lux_sap_legal(xy.expand(2, 3, 2), mask.expand(2, 3), torch.tensor([2, 0]),
                           R, 6, 6)
    assert torch.equal(legal2[0], legal[0])
    assert legal2[1, 0].sum() == 1 and legal2[1, 0, lux_sap_offset(0, 0, R)]


def test_sap_head_shapes_and_z_loss():
    torch.manual_seed(0)
    cfg = make_cfg(lux_sap_range=R)
    model = build(cfg)
    assert model.action_decoder.sap_head is not None
    B, T = 2, 3
    obs, legal = make_sap_obs_legal(cfg, B, T)
    out = model(obs, legal=legal)
    assert out["sap_logits"].shape == (B, T, cfg.max_units, N_OFF)
    assert torch.equal(out["sap_mask"], legal["sap"])
    # the z-loss picks the sap head up like every other *_logits head
    m = torch.ones(B, T, dtype=torch.bool)
    z_all, _ = logit_z_loss(out, m)
    z_unit, _ = logit_z_loss({"unit_action_logits": out["unit_action_logits"]}, m)
    z_sap, _ = logit_z_loss({"sap_logits": out["sap_logits"]}, m)
    torch.testing.assert_close(z_all, z_unit + z_sap)
    assert z_sap > 0
    # gradient reaches the sap head
    out["sap_logits"].sum().backward()
    assert model.action_decoder.sap_head.net[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("with_sap_legal", [False, True])
def test_sap_rollout_replay_exactness(with_sap_legal):
    torch.manual_seed(2)
    cfg = make_cfg(lux_sap_range=R)
    model = build(cfg)
    model.eval()
    B, T, K = 3, 4, 2
    obs, legal = make_sap_obs_legal(cfg, B, T)
    legal["unit_action"][..., SAP] = True      # plenty of sap actions
    if not with_sap_legal:
        del legal["sap"]
    cache = None
    lps, acts, saps = [], [], []
    for t in range(T):
        with torch.no_grad():
            out = model.step({k: v[:, t] for k, v in obs.items()},
                             legal={k: v[:, t] for k, v in legal.items()},
                             st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        acts.append(out["unit_actions"])
        saps.append(out["sap_offsets"])
        if with_sap_legal:     # sampled offsets are legal for every present unit
            chosen = legal["sap"][:, t].gather(-1, out["sap_offsets"].unsqueeze(-1)).squeeze(-1)
            assert chosen[obs["unit_mask"][:, t]].all()
    actions = {"unit_actions": torch.stack(acts, 1), "sap_offsets": torch.stack(saps, 1)}
    assert (actions["unit_actions"] == SAP).any()
    with torch.no_grad():
        pred = model(obs, legal=legal, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(pred, actions, legal)
    torch.testing.assert_close(lp, torch.stack(lps, 1), atol=1e-5, rtol=1e-5)
    assert torch.isfinite(ent).all()
    assert set(head_ent) == {"ent_unit_action", "ent_sap"}
    # the sap lp counts only for sapping units: re-drawing the offsets of
    # non-sapping units leaves the joint lp untouched
    other = torch.where(actions["unit_actions"] == SAP, actions["sap_offsets"],
                        (actions["sap_offsets"] + 1) % N_OFF)
    with torch.no_grad():
        lp2, _, _ = model.compute_log_prob(pred, {**actions, "sap_offsets": other}, legal)
    assert torch.equal(lp, lp2)
    with pytest.raises(KeyError):
        model.compute_log_prob(pred, {"unit_actions": actions["unit_actions"]}, legal)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sap_sample_replay_bitwise_decoder_level(dtype):
    torch.manual_seed(3)
    cfg = make_cfg(lux_sap_range=R)
    dec = LuxDecoder(cfg)
    B, T, U = 4, 5, cfg.max_units
    obs, legal = make_sap_obs_legal(cfg, B, T)
    pred = {
        "unit_action_logits": (torch.randn(B, T, U, cfg.n_unit_actions) * 3).to(dtype),
        "sap_logits": (torch.randn(B, T, U, N_OFF) * 3).to(dtype),
        "unit_present": obs["unit_mask"],
    }
    lps, acts, saps = [], [], []
    for t in range(T):
        out = dec.sample({k: v[:, t] for k, v in pred.items()},
                         {k: v[:, t] for k, v in legal.items()})
        lps.append(out["action_log_prob"])
        acts.append(out["unit_actions"])
        saps.append(out["sap_offsets"])
    actions = {"unit_actions": torch.stack(acts, 1), "sap_offsets": torch.stack(saps, 1)}
    lp, ent, _ = dec.log_prob_entropy(pred, actions, legal)
    assert lp.dtype == torch.float32
    assert torch.equal(lp, torch.stack(lps, 1))
    assert torch.isfinite(ent).all()


@pytest.mark.parametrize("with_sap_legal", [False, True])
def test_sap_il_loss_and_head_terms(with_sap_legal):
    torch.manual_seed(4)
    cfg = make_cfg(lux_sap_range=R)
    model = build(cfg)
    model.train()
    B, T = 2, 3
    obs, legal = make_sap_obs_legal(cfg, B, T)
    tgt = make_sap_tgt(cfg, legal, B, T)
    if not with_sap_legal:
        del legal["sap"]
    pred = model(obs, legal=legal)
    seq = torch.ones(B, T, dtype=torch.bool)
    seq[1, -1] = False
    dec = model.action_decoder
    out = dec.il_loss(pred, tgt, seq)
    assert torch.isfinite(out["loss"])
    for k in ("sap_loss", "sap_acc", "unit_action_loss", "unit_action_acc"):
        assert torch.isfinite(out[k]) and not out[k].requires_grad
    torch.testing.assert_close(out["loss"], out["unit_action_loss"] + out["sap_loss"])
    # the sap CE runs exactly over present, in-sequence sap targets with an offset
    present = obs["unit_mask"].float() * seq.float().unsqueeze(-1)
    w_sap = present * (tgt["unit_actions"] == SAP).float() * (tgt["sap_offset"] >= 0).float()
    terms = dec.il_head_terms(pred, tgt, seq)
    assert set(terms) == {"unit_action", "sap"}
    assert torch.equal(terms["sap"]["weight"], w_sap)   # all targets legal
    assert (terms["sap"]["mask"] is not None) == with_sap_legal
    torch.testing.assert_close(reconstruct(terms), out["loss"], atol=1e-6, rtol=1e-6)
    for h in terms.values():
        assert h["logits"].requires_grad and not h["weight"].requires_grad
        assert h["weight"].shape == h["logits"].shape[:-1] == h["ce"].shape
    out["loss"].backward()
    assert model.action_decoder.sap_head.net[0].weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    # answer_correct also needs the sap offsets: a perfect unit-action head
    # with a wrong offset on a supervised sap unit is not an exact answer
    with torch.no_grad():
        fake = {k: v.detach() for k, v in pred.items() if torch.is_tensor(v)}
        fake["unit_action_logits"] = torch.nn.functional.one_hot(
            tgt["unit_actions"], cfg.n_unit_actions).float() * 50
        good = torch.nn.functional.one_hot(tgt["sap_offset"].clamp(min=0), N_OFF).float() * 50
        fake["sap_logits"] = good
        fake.pop("sap_mask", None)
        ok = dec.il_loss(fake, tgt, seq)
        assert ok["answer_correct"].all() and ok["sap_acc"] == 1.0
        b, t, u = (w_sap > 0).nonzero()[0].tolist()
        bad = good.clone()
        bad[b, t, u] = good[b, t, u].roll(1)
        res = dec.il_loss({**fake, "sap_logits": bad}, tgt, seq)
        assert not res["answer_correct"][b, t] and res["answer_correct"].sum() == B * T - 1

    # no sap_offset in tgt: the historical loss; the sap term stays addressable
    # for soft targets (additive, ce None)
    tgt_old = {"unit_actions": tgt["unit_actions"]}
    res = dec.il_loss(pred, tgt_old, seq)
    assert "sap_loss" not in res and torch.equal(res["loss"], res["unit_action_loss"])
    terms = dec.il_head_terms(pred, tgt_old, seq)
    assert terms["sap"]["ce"] is None
    assert torch.equal(terms["sap"]["weight"], present * (tgt["unit_actions"] == SAP).float())


def test_sap_il_loss_illegal_target_and_zero_units():
    torch.manual_seed(5)
    cfg = make_cfg(lux_sap_range=R)
    model = build(cfg)
    B, T = 2, 2
    obs, legal = make_sap_obs_legal(cfg, B, T)
    obs["unit_mask"][0, 0, 0] = True
    legal["unit"][0, 0, 0] = True
    tgt = make_sap_tgt(cfg, legal, B, T)
    tgt["unit_actions"][0, 0, 0] = SAP
    legal["sap"][0, 0, 0] = False
    legal["sap"][0, 0, 0, 0] = True
    tgt["sap_offset"][0, 0, 0] = 1                     # outside its mask
    pred = model(obs, legal=legal)
    seq = torch.ones(B, T, dtype=torch.bool)
    out = model.action_decoder.il_loss(pred, tgt, seq)
    assert torch.isfinite(out["loss"])
    terms = model.action_decoder.il_head_terms(pred, tgt, seq)
    assert terms["sap"]["weight"][0, 0, 0] == 0 and terms["sap"]["ce"][0, 0, 0] == 0
    torch.testing.assert_close(reconstruct(terms), out["loss"], atol=1e-6, rtol=1e-6)
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)

    obs["unit_mask"][:] = False
    legal["unit"][:] = False
    with torch.no_grad():
        pred = model(obs, legal=legal)
    out = model.action_decoder.il_loss(pred, tgt, seq)
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)
