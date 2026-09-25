"""Tests for the StarCraft-style RTS adapter (StarCraftEncoder + RTSDecoder).

Covers the frozen adapter contract:
  1. forward output shapes (+ token-layout attrs)
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal teacher-forced replay
  5. legal-mask respect for all four maskable heads
  6. il_loss finiteness (incl. the fully-masked-out batch)
  plus ARG_USES gating and the constructor override.
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.base import bernoulli_log_prob_entropy, masked_log_prob
from model.games.starcraft import (
    RTSDecoder,
    StarCraftConfig,
    StarCraftEncoder,
    default_arg_uses,
)


def make_cfg(**overrides) -> StarCraftConfig:
    kw = dict(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, encoder_channels=16,
        # shrunken game dims for CPU speed (16×16 minimap → 4×4 map tokens)
        minimap_h=16, minimap_w=16, minimap_channels=4,
        max_units=8, unit_feature_dim=6, sc_global_features=4,
    )
    kw.update(overrides)
    return StarCraftConfig(**kw)


def make_obs(cfg: StarCraftConfig, B: int, T: int) -> dict:
    unit_mask = torch.rand(B, T, cfg.max_units) < 0.7
    unit_mask[..., 0] = True  # keep at least one real unit around
    return {
        "minimap_seq": torch.randn(B, T, cfg.minimap_channels, cfg.minimap_h, cfg.minimap_w),
        "units_seq": torch.randn(B, T, cfg.max_units, cfg.unit_feature_dim),
        "unit_mask": unit_mask,
        "globals_seq": torch.randn(B, T, cfg.sc_global_features),
    }


def build(cfg: StarCraftConfig, **decoder_kw) -> GameNetwork:
    return GameNetwork(cfg, StarCraftEncoder(cfg), RTSDecoder(cfg, **decoder_kw))


def random_mask(B: int, n: int, p: float = 0.4) -> torch.Tensor:
    m = torch.rand(B, n) < p
    m[torch.arange(B), torch.randint(0, n, (B,))] = True
    return m


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
    assert M == (cfg.minimap_h // 4) * (cfg.minimap_w // 4)
    assert model.obs_encoder.n_tokens == M + U
    assert model.obs_encoder.positional_embedding().shape == (1, M + U, cfg.d_model)

    B, T = 2, 3
    with torch.no_grad():
        out = model(make_obs(cfg, B, T))
    assert out["type_logits"].shape == (B, T, cfg.n_action_types)
    assert out["actor_logits"].shape == (B, T, U)
    assert out["target_logits"].shape == (B, T, U)
    assert out["location_logits"].shape == (B, T, M)
    assert out["queued_logits"].shape == (B, T)
    assert out["action_type_used"].shape == (B, T)
    assert out["actor_idx"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


def test_arg_uses_table_and_override():
    """Row 0 is always no_op; non-8 sizes cycle sensibly; the constructor
    accepts a custom table and rejects wrong shapes."""
    table = default_arg_uses(8)
    assert table.shape == (8, 3)
    assert not table[0].any()          # no_op uses nothing
    assert table[1:, 0].all()          # everything else has an actor
    assert default_arg_uses(5).shape == (5, 3)
    assert not default_arg_uses(5)[0].any()

    cfg = make_cfg()
    custom = torch.ones(cfg.n_action_types, 3, dtype=torch.bool)
    dec = RTSDecoder(cfg, arg_uses=custom)
    assert dec.arg_uses.all()
    with pytest.raises(ValueError):
        RTSDecoder(cfg, arg_uses=torch.ones(3, 3, dtype=torch.bool))


# ======================================================================
# 2. Gradient flow
# ======================================================================

def test_gradient_flow():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    out = model(make_obs(cfg, 2, 3))
    loss = (
        out["type_logits"].sum() + out["actor_logits"].sum()
        + out["target_logits"].sum() + out["location_logits"].sum()
        + out["queued_logits"].sum() + out["value"].sum() + out["q_halt"].sum()
    )
    loss.backward()
    assert_grad_flow(model)
    # the type embedding conditions actor/location/queued heads → must train
    assert model.action_decoder.type_embed.weight.grad.abs().sum() > 0


# ======================================================================
# 3. step() chaining with Stage-1 cache
# ======================================================================

def test_step_cache_growth():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B = 2
    S = model.obs_encoder.n_tokens + 2  # + HL/LL decision tokens

    cache = None
    for t in range(3):
        obs = {k: v[:, 0] for k, v in make_obs(cfg, B, 1).items()}
        with torch.no_grad():
            out = model.step(obs, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)
        assert out["type_logits"].shape == (B, cfg.n_action_types)
        assert out["action_type"].shape == (B,)
        assert out["action_actor"].shape == (B,)
        assert out["action_target"].shape == (B,)
        assert out["action_location"].shape == (B,)
        assert out["action_queued"].dtype == torch.bool
        assert out["action_log_prob"].shape == (B,)
        assert out["value"].shape == (B,)


# ======================================================================
# 4. PPO exactness: rollout log-prob == teacher-forced replay
# ======================================================================

def test_ppo_exactness():
    """Mirror of test_step_vs_full_consistency_act_steps: step() rollout and
    forward()+log_prob_entropy replay must define the same policy."""
    torch.manual_seed(1)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T, K = 2, 4, 2
    obs = make_obs(cfg, B, T)

    cache = None
    rollout = {k: [] for k in
               ("lp", "type", "actor", "target", "location", "queued")}
    for t in range(T):
        with torch.no_grad():
            out = model.step(
                {k: v[:, t] for k, v in obs.items()},
                legal=None, st_cache=cache, act_steps=K,
            )
        cache = out["st_cache"]
        rollout["lp"].append(out["action_log_prob"])
        rollout["type"].append(out["action_type"])
        rollout["actor"].append(out["action_actor"])
        rollout["target"].append(out["action_target"])
        rollout["location"].append(out["action_location"])
        rollout["queued"].append(out["action_queued"])

    stack = {k: torch.stack(v, 1) for k, v in rollout.items()}
    labels = {"action_type": stack["type"], "actor_idx": stack["actor"]}
    actions = {
        "action_type": stack["type"],
        "actor_idx": stack["actor"],
        "target_idx": stack["target"],
        "location_idx": stack["location"],
        "queued": stack["queued"],
    }
    with torch.no_grad():
        pred = model(obs, labels=labels, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(pred, actions)

    # decode must echo back the conditioning it was teacher-forced with
    assert torch.equal(pred["action_type_used"], labels["action_type"])
    assert torch.equal(pred["actor_idx"], labels["actor_idx"])

    assert torch.allclose(lp, stack["lp"], atol=1e-5)
    assert ent.shape == (B, T) and torch.isfinite(ent).all()
    assert all(v.dim() == 0 and not v.requires_grad for v in head_ent.values())


def test_arg_uses_gates_log_prob():
    """A no_op action (type 0) must be scored by type + queued alone —
    argument heads are inapplicable and must not enter the log-prob."""
    torch.manual_seed(2)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T = 2, 3
    obs = make_obs(cfg, B, T)
    actions = {
        "action_type": torch.zeros(B, T, dtype=torch.long),  # no_op everywhere
        "actor_idx": torch.randint(0, cfg.max_units, (B, T)),
        "target_idx": torch.randint(0, cfg.max_units, (B, T)),
        "location_idx": torch.zeros(B, T, dtype=torch.long),
        "queued": torch.rand(B, T) < 0.5,
    }
    with torch.no_grad():
        pred = model(obs, labels={"action_type": actions["action_type"],
                                  "actor_idx": actions["actor_idx"]})
        lp, _, _ = model.compute_log_prob(pred, actions)
        expected = (
            masked_log_prob(pred["type_logits"], actions["action_type"])
            + bernoulli_log_prob_entropy(
                pred["queued_logits"], actions["queued"].bool())[0]
        )
    assert torch.allclose(lp, expected, atol=1e-6)


# ======================================================================
# 5. Legal-mask respect
# ======================================================================

def test_legal_mask_respected():
    torch.manual_seed(3)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B = 16
    M = model.obs_encoder.n_map_tokens
    legal = {
        "action_type": random_mask(B, cfg.n_action_types),
        "actor": random_mask(B, cfg.max_units),
        "target_unit": random_mask(B, cfg.max_units),
        "location": random_mask(B, M),
    }
    arange = torch.arange(B)
    for _ in range(10):
        obs = {k: v[:, 0] for k, v in make_obs(cfg, B, 1).items()}
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        assert legal["action_type"][arange, out["action_type"]].all()
        assert legal["actor"][arange, out["action_actor"]].all()
        assert legal["target_unit"][arange, out["action_target"]].all()
        assert legal["location"][arange, out["action_location"]].all()


# ======================================================================
# 6. il_loss
# ======================================================================

def test_il_loss():
    torch.manual_seed(4)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    M = model.obs_encoder.n_map_tokens
    pred = model(make_obs(cfg, B, T))
    tgt = {
        "action_type": torch.randint(0, cfg.n_action_types, (B, T)),
        "actor_idx": torch.randint(0, cfg.max_units, (B, T)),
        "target_idx": torch.randint(0, cfg.max_units, (B, T)),
        "location_idx": torch.randint(0, M, (B, T)),
        "queued": torch.rand(B, T) < 0.5,
    }
    full = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    empty = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))

    assert torch.isfinite(full["loss"])
    for k in (
        "type_loss", "queued_loss", "actor_loss", "target_loss", "location_loss",
        "type_acc", "queued_acc", "actor_acc", "target_acc", "location_acc",
    ):
        assert k in full and torch.isfinite(full[k]) and not full[k].requires_grad
    # fully-masked-out batch: zero-ish loss, never NaN
    assert torch.isfinite(empty["loss"])
    assert empty["loss"].item() == pytest.approx(0.0, abs=1e-6)

    (full["loss"] + empty["loss"]).backward()
    assert_grad_flow(model)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
