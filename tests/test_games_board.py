"""Tests for the board-game adapters (BoardEncoder + FromToDecoder / PlacementDecoder).

Covers the frozen adapter contract:
  1. forward output shapes for both decoders
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal teacher-forced replay
  5. legal-mask respect during sampling
  6. il_loss finiteness (incl. the fully-masked-out batch)
"""

import pytest
import torch
import torch.nn.functional as F

from model.game_network import GameNetwork
from model.games.base import masked_sample
from model.games.board import (
    BoardConfig,
    BoardEncoder,
    FromToDecoder,
    PlacementDecoder,
)

N = 64  # 8×8 board


def make_cfg(**overrides) -> BoardConfig:
    kw = dict(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, encoder_channels=16,
    )
    kw.update(overrides)
    return BoardConfig(**kw)


def make_obs(cfg: BoardConfig, B: int, T: int) -> dict:
    obs = {"board_seq": torch.randn(B, T, cfg.piece_channels, cfg.board_h, cfg.board_w)}
    if cfg.board_global_features > 0:
        obs["globals_seq"] = torch.randn(B, T, cfg.board_global_features)
    return obs


def build(cfg: BoardConfig, decoder_cls) -> GameNetwork:
    return GameNetwork(cfg, BoardEncoder(cfg), decoder_cls(cfg))


def random_mask(B: int, n: int, p: float = 0.25) -> torch.Tensor:
    """Random legality mask with at least one legal entry per row."""
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
# 1. Forward shapes
# ======================================================================

def test_fromto_forward_shapes():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg, FromToDecoder)
    model.eval()
    B, T = 2, 3
    with torch.no_grad():
        out = model(make_obs(cfg, B, T))

    assert out["from_logits"].shape == (B, T, N)
    assert out["to_logits"].shape == (B, T, N)
    assert out["promotion_logits"].shape == (B, T, cfg.n_promotions)
    assert out["from_idx"].shape == (B, T) and out["from_idx"].dtype == torch.long
    assert out["to_idx"].shape == (B, T) and out["to_idx"].dtype == torch.long
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


def test_fromto_no_promotion_no_globals():
    """n_promotions ≤ 1 disables the promotion head; board_global_features=0
    means no globals_seq is required at all."""
    torch.manual_seed(0)
    cfg = make_cfg(n_promotions=0, board_global_features=0)
    model = build(cfg, FromToDecoder)
    model.eval()
    obs = {"board_seq": torch.randn(2, 3, cfg.piece_channels, 8, 8)}
    with torch.no_grad():
        out = model(obs)
    assert "promotion_logits" not in out
    assert out["from_logits"].shape == (2, 3, N)


def test_placement_forward_shapes():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg, PlacementDecoder)
    model.eval()
    B, T = 2, 3
    with torch.no_grad():
        out = model(make_obs(cfg, B, T))
    assert out["place_logits"].shape == (B, T, N)
    assert out["pass_logits"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


# ======================================================================
# 2. Gradient flow
# ======================================================================

def test_fromto_gradient_flow():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg, FromToDecoder)
    model.train()
    out = model(make_obs(cfg, 2, 3))
    loss = (
        out["from_logits"].sum() + out["to_logits"].sum()
        + out["promotion_logits"].sum() + out["value"].sum() + out["q_halt"].sum()
    )
    loss.backward()
    assert_grad_flow(model)


def test_placement_gradient_flow():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg, PlacementDecoder)
    model.train()
    out = model(make_obs(cfg, 2, 3))
    loss = out["place_logits"].sum() + out["pass_logits"].sum() + out["value"].sum()
    loss.backward()
    assert_grad_flow(model)


# ======================================================================
# 3. step() chaining with Stage-1 cache
# ======================================================================

@pytest.mark.parametrize("decoder_cls", [FromToDecoder, PlacementDecoder])
def test_step_cache_growth(decoder_cls):
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg, decoder_cls)
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
        assert out["value"].shape == (B,)
        assert out["action_log_prob"].shape == (B,)
        if decoder_cls is FromToDecoder:
            assert out["from_logits"].shape == (B, N)
            assert out["action_from"].shape == (B,)
        else:
            assert out["place_logits"].shape == (B, N)
            assert out["action_place"].shape == (B,)


# ======================================================================
# 4. PPO exactness: rollout log-prob == teacher-forced replay
# ======================================================================

def test_ppo_exactness_fromto():
    """Mirror of test_step_vs_full_consistency_act_steps: step() rollout and
    forward()+log_prob_entropy replay must define the same policy."""
    torch.manual_seed(1)
    cfg = make_cfg()
    model = build(cfg, FromToDecoder)
    model.eval()
    B, T, K = 2, 4, 2
    obs = make_obs(cfg, B, T)

    cache = None
    lps, froms, tos, promos = [], [], [], []
    for t in range(T):
        with torch.no_grad():
            out = model.step(
                {k: v[:, t] for k, v in obs.items()},
                legal=None, st_cache=cache, act_steps=K,
            )
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        froms.append(out["action_from"])
        tos.append(out["action_to"])
        promos.append(out["action_promotion"])

    labels = {"from_idx": torch.stack(froms, 1), "to_idx": torch.stack(tos, 1)}
    actions = {**labels, "promotion": torch.stack(promos, 1)}
    with torch.no_grad():
        pred = model(obs, labels=labels, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(pred, actions)

    # decode must echo back the conditioning it was teacher-forced with
    assert torch.equal(pred["from_idx"], labels["from_idx"])
    assert torch.equal(pred["to_idx"], labels["to_idx"])

    assert torch.allclose(lp, torch.stack(lps, 1), atol=1e-5)
    assert ent.shape == (B, T) and torch.isfinite(ent).all()
    assert all(v.dim() == 0 and not v.requires_grad for v in head_ent.values())


def test_ppo_exactness_placement():
    torch.manual_seed(2)
    cfg = make_cfg()
    model = build(cfg, PlacementDecoder)
    model.eval()
    B, T, K = 2, 4, 2
    obs = make_obs(cfg, B, T)

    cache = None
    lps, places, passes = [], [], []
    for t in range(T):
        with torch.no_grad():
            out = model.step(
                {k: v[:, t] for k, v in obs.items()},
                legal=None, st_cache=cache, act_steps=K,
            )
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        places.append(out["action_place"])
        passes.append(out["action_pass"])

    actions = {"place_idx": torch.stack(places, 1), "passed": torch.stack(passes, 1)}
    with torch.no_grad():
        pred = model(obs, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(pred, actions)

    assert torch.allclose(lp, torch.stack(lps, 1), atol=1e-5)
    assert ent.shape == (B, T) and torch.isfinite(ent).all()
    assert set(head_ent) == {"ent_pass", "ent_place"}


def test_ppo_exactness_fromto_bf16():
    """With bf16 logits, sample()'s bookkeeping and log_prob_entropy()'s
    replay must still agree bit-for-bit — both evaluate the fp32
    distribution (masked_log_prob / _log_prob_and_entropy cast to float32).
    Decoder-level so the two paths see literally the same logits."""
    torch.manual_seed(7)
    cfg = make_cfg()
    decoder = FromToDecoder(cfg)
    B, T = 4, 5
    legal = {
        "from": torch.stack([random_mask(B, N) for _ in range(T)], 1),
        "to": torch.stack([random_mask(B, N) for _ in range(T)], 1),
    }
    pred = {
        "from_logits": (torch.randn(B, T, N) * 5).bfloat16(),
        "to_logits": (torch.randn(B, T, N) * 5).bfloat16(),
        "promotion_logits": (torch.randn(B, T, cfg.n_promotions) * 5).bfloat16(),
    }

    lps, froms, tos, promos = [], [], [], []
    for t in range(T):
        pred_t = {k: v[:, t] for k, v in pred.items()}
        legal_t = {k: v[:, t] for k, v in legal.items()}
        # decode() commits from/to via masked_sample; replicate that here
        pred_t["from_idx"] = masked_sample(pred_t["from_logits"], legal_t["from"])
        pred_t["to_idx"] = masked_sample(pred_t["to_logits"], legal_t["to"])
        out = decoder.sample(pred_t, legal_t)
        lps.append(out["action_log_prob"])
        froms.append(out["action_from"])
        tos.append(out["action_to"])
        promos.append(out["action_promotion"])

    actions = {
        "from_idx": torch.stack(froms, 1),
        "to_idx": torch.stack(tos, 1),
        "promotion": torch.stack(promos, 1),
    }
    lp, ent, head_ent = decoder.log_prob_entropy(pred, actions, legal)

    rollout_lp = torch.stack(lps, 1)
    assert lp.dtype == torch.float32 and rollout_lp.dtype == torch.float32
    assert torch.equal(lp, rollout_lp)  # bitwise: PPO ratio is exactly 1
    assert ent.dtype == torch.float32 and torch.isfinite(ent).all()


def test_ppo_exactness_placement_bf16():
    torch.manual_seed(8)
    cfg = make_cfg()
    decoder = PlacementDecoder(cfg)
    B, T = 4, 6
    legal = {"place": torch.stack([random_mask(B, N) for _ in range(T)], 1)}
    pass_logits = torch.randn(B, T) * 5
    pass_logits[:, 0] = 20.0   # sigmoid ≈ 1: guaranteed pass frames
    pass_logits[:, 1] = -20.0  # sigmoid ≈ 0: guaranteed move frames
    pred = {
        "place_logits": (torch.randn(B, T, N) * 5).bfloat16(),
        "pass_logits": pass_logits.bfloat16(),
    }

    lps, places, passes = [], [], []
    for t in range(T):
        out = decoder.sample(
            {k: v[:, t] for k, v in pred.items()},
            {"place": legal["place"][:, t]},
        )
        lps.append(out["action_log_prob"])
        places.append(out["action_place"])
        passes.append(out["action_pass"])

    actions = {"place_idx": torch.stack(places, 1), "passed": torch.stack(passes, 1)}
    assert actions["passed"][:, 0].all() and not actions["passed"][:, 1].any()

    lp, ent, head_ent = decoder.log_prob_entropy(pred, actions, legal)
    rollout_lp = torch.stack(lps, 1)
    assert lp.dtype == torch.float32 and rollout_lp.dtype == torch.float32
    assert torch.equal(lp, rollout_lp)
    assert ent.dtype == torch.float32 and torch.isfinite(ent).all()

    # Sample/replay agreeing is not enough — a bf16 bookkeeping pair would
    # also agree with itself.  Pin both to the fp32 evaluation of the same
    # masked distribution (every row has a legal entry, so no guard needed).
    place_f = pred["place_logits"].float().masked_fill(~legal["place"], float("-inf"))
    ref_place = F.log_softmax(place_f, -1).gather(
        -1, actions["place_idx"].unsqueeze(-1),
    ).squeeze(-1)
    pass_f = pred["pass_logits"].float()
    ref_pass = torch.where(
        actions["passed"], F.logsigmoid(pass_f), F.logsigmoid(-pass_f),
    )
    assert torch.equal(lp, torch.where(actions["passed"], ref_pass, ref_pass + ref_place))


# ======================================================================
# 5. Legal-mask respect
# ======================================================================

def test_legal_mask_respected_fromto():
    torch.manual_seed(3)
    cfg = make_cfg()
    model = build(cfg, FromToDecoder)
    model.eval()
    B = 16
    legal = {"from": random_mask(B, N), "to": random_mask(B, N)}
    arange = torch.arange(B)
    for _ in range(10):
        obs = {k: v[:, 0] for k, v in make_obs(cfg, B, 1).items()}
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        assert legal["from"][arange, out["action_from"]].all()
        assert legal["to"][arange, out["action_to"]].all()


def test_legal_mask_respected_placement():
    torch.manual_seed(4)
    cfg = make_cfg()
    model = build(cfg, PlacementDecoder)
    model.eval()
    B = 16
    legal = {"place": random_mask(B, N)}
    arange = torch.arange(B)
    for _ in range(10):
        obs = {k: v[:, 0] for k, v in make_obs(cfg, B, 1).items()}
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        assert legal["place"][arange, out["action_place"]].all()


# ======================================================================
# 6. il_loss
# ======================================================================

def test_il_loss_fromto():
    torch.manual_seed(5)
    cfg = make_cfg()
    model = build(cfg, FromToDecoder)
    model.train()
    B, T = 2, 3
    pred = model(make_obs(cfg, B, T))
    tgt = {
        "from_idx": torch.randint(0, N, (B, T)),
        "to_idx": torch.randint(0, N, (B, T)),
        "promotion": torch.randint(0, cfg.n_promotions, (B, T)),
    }
    full = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    empty = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))

    assert torch.isfinite(full["loss"])
    for k in ("from_loss", "to_loss", "promo_loss", "from_acc", "to_acc", "promo_acc"):
        assert k in full and torch.isfinite(full[k]) and not full[k].requires_grad
    # fully-masked-out batch: zero-ish loss, never NaN
    assert torch.isfinite(empty["loss"])
    assert empty["loss"].item() == pytest.approx(0.0, abs=1e-6)

    (full["loss"] + empty["loss"]).backward()
    assert_grad_flow(model)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_il_loss_placement():
    torch.manual_seed(6)
    cfg = make_cfg()
    model = build(cfg, PlacementDecoder)
    model.train()
    B, T = 2, 3
    pred = model(make_obs(cfg, B, T))
    tgt = {
        "place_idx": torch.randint(0, N, (B, T)),
        "passed": torch.rand(B, T) < 0.3,
    }
    full = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    empty = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))

    assert torch.isfinite(full["loss"])
    for k in ("pass_loss", "place_loss", "pass_acc", "place_acc"):
        assert torch.isfinite(full[k]) and not full[k].requires_grad
    assert torch.isfinite(empty["loss"])
    assert empty["loss"].item() == pytest.approx(0.0, abs=1e-6)

    (full["loss"] + empty["loss"]).backward()
    assert_grad_flow(model)


# ---------------------------------------------------------------------------
# 7. masked entropy gradients — 0·(−inf) must not NaN the backward
# ---------------------------------------------------------------------------


def test_categorical_entropy_masked_backward_finite():
    from model.games.base import categorical_entropy

    torch.manual_seed(0)
    logits = torch.randn(4, 7, requires_grad=True)
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[:, :3] = True
    ent = categorical_entropy(logits.masked_fill(~mask, float("-inf")))
    # forward: entropy of the 3 unmasked entries only (≤ log 3)
    assert torch.isfinite(ent).all()
    assert (ent <= torch.log(torch.tensor(3.0)) + 1e-5).all()
    ent.sum().backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[:, 3:].abs().sum() == 0


def test_log_prob_and_entropy_masked_backward_finite():
    from model.games.board import _log_prob_and_entropy

    torch.manual_seed(0)
    logits = torch.randn(5, N, requires_grad=True)
    mask = torch.zeros(5, N, dtype=torch.bool)
    mask[:, :4] = True
    idx = torch.randint(0, 4, (5,))
    lp, ent = _log_prob_and_entropy(logits, idx, mask)
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()
    (lp.sum() + ent.sum()).backward()
    assert torch.isfinite(logits.grad).all()
