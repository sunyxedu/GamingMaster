"""Tests for the Shogi adapter (ShogiEncoder + ShogiDecoder).

Covers the frozen adapter contract:
  1. forward output shapes (both branches always decoded)
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal teacher-forced replay
  5. legal-mask respect for from / to / drop_piece / drop_to
  6. il_loss finiteness (incl. the fully-masked-out batch)

Uses a tiny 5×5 board to keep the suite fast: n_tokens = 25 + 14 = 39.
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.shogi import ShogiConfig, ShogiDecoder, ShogiEncoder

NB = 25  # 5×5 board tokens
NH = 14  # 7 hand piece types × 2 players


def make_cfg(**overrides) -> ShogiConfig:
    kw = dict(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, encoder_channels=16,
        board_h=5, board_w=5, piece_channels=8,
    )
    kw.update(overrides)
    return ShogiConfig(**kw)


def make_obs(cfg: ShogiConfig, B: int, T: int) -> dict:
    obs = {
        "board_seq": torch.randn(B, T, cfg.piece_channels, cfg.board_h, cfg.board_w),
        "hand_seq": torch.randn(B, T, 2 * cfg.n_hand_piece_types, cfg.hand_feature_dim),
    }
    if cfg.shogi_global_features > 0:
        obs["globals_seq"] = torch.randn(B, T, cfg.shogi_global_features)
    return obs


def build(cfg: ShogiConfig) -> GameNetwork:
    return GameNetwork(cfg, ShogiEncoder(cfg), ShogiDecoder(cfg))


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

def test_forward_shapes_both_branches():
    torch.manual_seed(0)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    assert model.obs_encoder.n_tokens == NB + NH
    B, T = 2, 3
    with torch.no_grad():
        out = model(make_obs(cfg, B, T))

    # mode + MOVE branch
    assert out["mode_logits"].shape == (B, T)
    assert out["from_logits"].shape == (B, T, NB)
    assert out["to_logits"].shape == (B, T, NB)
    assert out["promo_logits"].shape == (B, T)
    # DROP branch
    assert out["piece_logits"].shape == (B, T, NH)
    assert out["drop_to_logits"].shape == (B, T, NB)
    # echoed conditioning
    assert out["dropped"].shape == (B, T) and out["dropped"].dtype == torch.bool
    assert out["from_idx"].shape == (B, T) and out["from_idx"].dtype == torch.long
    assert out["to_idx"].shape == (B, T) and out["to_idx"].dtype == torch.long
    assert out["piece_idx"].shape == (B, T) and out["piece_idx"].dtype == torch.long
    # shared core
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)


def test_forward_no_globals():
    """shogi_global_features=0 → no globals_seq required at all."""
    torch.manual_seed(0)
    cfg = make_cfg(shogi_global_features=0)
    model = build(cfg)
    model.eval()
    obs = make_obs(cfg, 2, 3)
    assert "globals_seq" not in obs
    with torch.no_grad():
        out = model(obs)
    assert out["from_logits"].shape == (2, 3, NB)
    assert out["piece_logits"].shape == (2, 3, NH)


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
        out["mode_logits"].sum()
        + out["from_logits"].sum() + out["to_logits"].sum()
        + out["promo_logits"].sum()
        + out["piece_logits"].sum() + out["drop_to_logits"].sum()
        + out["value"].sum() + out["q_halt"].sum()
    )
    loss.backward()
    assert_grad_flow(model)


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
        assert out["value"].shape == (B,)
        assert out["mode_logits"].shape == (B,)
        assert out["from_logits"].shape == (B, NB)
        assert out["piece_logits"].shape == (B, NH)
        assert out["drop_to_logits"].shape == (B, NB)
        assert out["action_dropped"].shape == (B,)
        assert out["action_from"].shape == (B,)
        assert out["action_to"].shape == (B,)
        assert out["action_promoted"].shape == (B,)
        assert out["action_piece"].shape == (B,)
        assert out["action_drop_to"].shape == (B,)
        assert out["action_log_prob"].shape == (B,)


# ======================================================================
# 4. PPO exactness: rollout log-prob == teacher-forced replay
# ======================================================================

def test_ppo_exactness():
    """step() rollout and forward()+log_prob_entropy replay must define the
    same policy: store every action component plus the mode, replay with
    all conditioning teacher-forced."""
    torch.manual_seed(1)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T, K = 2, 4, 2
    obs = make_obs(cfg, B, T)

    cache = None
    lps, drops, froms, tos, promos, pieces, drop_tos = [], [], [], [], [], [], []
    for t in range(T):
        with torch.no_grad():
            out = model.step(
                {k: v[:, t] for k, v in obs.items()},
                legal=None, st_cache=cache, act_steps=K,
            )
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        drops.append(out["action_dropped"])
        froms.append(out["action_from"])
        tos.append(out["action_to"])
        promos.append(out["action_promoted"])
        pieces.append(out["action_piece"])
        drop_tos.append(out["action_drop_to"])

    dropped = torch.stack(drops, 1)
    labels = {
        "dropped": dropped.long(),
        "from_idx": torch.stack(froms, 1),
        "to_idx": torch.stack(tos, 1),
        "piece_idx": torch.stack(pieces, 1),
    }
    actions = {
        "dropped": dropped,
        "from_idx": labels["from_idx"],
        "to_idx": labels["to_idx"],
        "promoted": torch.stack(promos, 1),
        "piece_idx": labels["piece_idx"],
        "drop_to_idx": torch.stack(drop_tos, 1),
    }
    with torch.no_grad():
        pred = model(obs, labels=labels, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(pred, actions)

    # decode must echo back the conditioning it was teacher-forced with
    assert torch.equal(pred["dropped"], dropped)
    assert torch.equal(pred["from_idx"], labels["from_idx"])
    assert torch.equal(pred["to_idx"], labels["to_idx"])
    assert torch.equal(pred["piece_idx"], labels["piece_idx"])

    assert torch.allclose(lp, torch.stack(lps, 1), atol=1e-5)
    assert ent.shape == (B, T) and torch.isfinite(ent).all()
    assert set(head_ent) == {
        "ent_mode", "ent_from", "ent_to", "ent_promo", "ent_piece", "ent_drop_to",
    }
    assert all(v.dim() == 0 and not v.requires_grad for v in head_ent.values())


# ======================================================================
# 5. Legal-mask respect
# ======================================================================

def test_legal_mask_respected():
    torch.manual_seed(3)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B = 16
    legal = {
        "from": random_mask(B, NB),
        "to": random_mask(B, NB),
        "drop_piece": random_mask(B, NH),
        "drop_to": random_mask(B, NB),
    }
    arange = torch.arange(B)
    for _ in range(10):
        obs = {k: v[:, 0] for k, v in make_obs(cfg, B, 1).items()}
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        assert legal["from"][arange, out["action_from"]].all()
        assert legal["to"][arange, out["action_to"]].all()
        assert legal["drop_piece"][arange, out["action_piece"]].all()
        assert legal["drop_to"][arange, out["action_drop_to"]].all()


# ======================================================================
# 6. il_loss
# ======================================================================

def make_targets(B: int, T: int) -> dict:
    return {
        "dropped": (torch.rand(B, T) < 0.4).long(),
        "from_idx": torch.randint(0, NB, (B, T)),
        "to_idx": torch.randint(0, NB, (B, T)),
        "promoted": torch.rand(B, T) < 0.3,
        "piece_idx": torch.randint(0, NH, (B, T)),
        "drop_to_idx": torch.randint(0, NB, (B, T)),
    }


def test_il_loss():
    torch.manual_seed(5)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    pred = model(make_obs(cfg, B, T))
    tgt = make_targets(B, T)
    full = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    empty = model.action_decoder.il_loss(pred, tgt, torch.zeros(B, T, dtype=torch.bool))

    assert torch.isfinite(full["loss"])
    for k in (
        "mode_loss", "from_loss", "to_loss", "promo_loss",
        "piece_loss", "drop_to_loss",
        "mode_acc", "from_acc", "to_acc", "promo_acc",
        "piece_acc", "drop_to_acc",
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


def test_il_loss_unlabeled_mode_frames():
    """dropped = −1 frames are excluded from every head but stay finite."""
    torch.manual_seed(6)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    pred = model(make_obs(cfg, B, T))
    tgt = make_targets(B, T)
    tgt["dropped"] = torch.full((B, T), -1, dtype=torch.long)
    out = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() == pytest.approx(0.0, abs=1e-6)


# ======================================================================
# 7. il_loss: branch supervision is gated on the TRUE mode label only
# ======================================================================
#
# No branch logit conditions on the mode (decode always evaluates both
# branches), so a frame's branch heads must be supervised whenever its
# ground-truth mode is labeled — even when the mode head's own argmax is
# wrong on EVERY frame.  The old `* mode_ok` gating on move_w / drop_w
# would zero the true-mode branch out entirely here, so these tests fail
# if it is ever reintroduced.

def head_has_grad(head: torch.nn.Module) -> bool:
    return any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters()
    )


def forced_pred(model: GameNetwork, cfg: ShogiConfig, tgt: dict) -> dict:
    """Forward pass with all conditioning teacher-forced from tgt, so the
    downstream chain weights (to_w, promo_w, drop_to_w) are fully on."""
    labels = {
        "dropped": tgt["dropped"],
        "from_idx": tgt["from_idx"],
        "to_idx": tgt["to_idx"],
        "piece_idx": tgt["piece_idx"],
    }
    return model(make_obs(cfg, *tgt["dropped"].shape), labels=labels)


def test_il_loss_move_supervision_independent_of_mode_head():
    """All-move batch, mode head confidently (and wrongly) says drop on
    every frame: the move branch must still be fully supervised."""
    torch.manual_seed(7)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    tgt = make_targets(B, T)
    tgt["dropped"] = torch.zeros(B, T, dtype=torch.long)
    pred = forced_pred(model, cfg, tgt)
    pred["mode_logits"] = torch.full((B, T), 8.0, requires_grad=True)

    out = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    assert out["mode_acc"].item() == 0.0  # mode argmax contradicts every label

    # true-mode branch supervised despite the wrong mode head
    assert out["from_loss"].item() > 0.0
    assert out["to_loss"].item() > 0.0
    assert out["promo_loss"].item() > 0.0

    out["loss"].backward()
    dec = model.action_decoder
    for name, head in [
        ("from_head", dec.from_head),
        ("to_head", dec.to_head),
        ("promo_head", dec.promo_head),
    ]:
        assert head_has_grad(head), f"no gradient reached {name}"
    for name, head in [
        ("piece_head", dec.piece_head),
        ("drop_to_head", dec.drop_to_head),
    ]:
        assert not head_has_grad(head), f"unexpected gradient reached {name}"


def test_il_loss_drop_supervision_independent_of_mode_head():
    """All-drop batch, mode head confidently (and wrongly) says move on
    every frame: the drop branch must still be fully supervised."""
    torch.manual_seed(8)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    tgt = make_targets(B, T)
    tgt["dropped"] = torch.ones(B, T, dtype=torch.long)
    pred = forced_pred(model, cfg, tgt)
    pred["mode_logits"] = torch.full((B, T), -8.0, requires_grad=True)

    out = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    assert out["mode_acc"].item() == 0.0  # mode argmax contradicts every label

    # true-mode branch supervised despite the wrong mode head
    assert out["piece_loss"].item() > 0.0
    assert out["drop_to_loss"].item() > 0.0

    out["loss"].backward()
    dec = model.action_decoder
    for name, head in [
        ("piece_head", dec.piece_head),
        ("drop_to_head", dec.drop_to_head),
    ]:
        assert head_has_grad(head), f"no gradient reached {name}"
    for name, head in [
        ("from_head", dec.from_head),
        ("to_head", dec.to_head),
        ("promo_head", dec.promo_head),
    ]:
        assert not head_has_grad(head), f"unexpected gradient reached {name}"


def test_il_loss_single_mode_batch_denominators():
    """Single-mode batches pin the clamp(min=1) normalization: the absent
    branch's losses are exactly 0 (not NaN from a 0/0), every metric stays
    finite, and the present branch keeps its full weight."""
    torch.manual_seed(9)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 2, 3
    tgt = make_targets(B, T)
    tgt["dropped"] = torch.zeros(B, T, dtype=torch.long)
    pred = forced_pred(model, cfg, tgt)
    seq_mask = torch.ones(B, T, dtype=torch.bool)

    all_move = model.action_decoder.il_loss(pred, tgt, seq_mask)
    tgt["dropped"] = torch.ones(B, T, dtype=torch.long)
    all_drop = model.action_decoder.il_loss(pred, tgt, seq_mask)

    for out in (all_move, all_drop):
        assert torch.isfinite(out["loss"])
        for k, v in out.items():
            if k.startswith("answer_"):   # per-frame q_halt target masks
                continue
            assert torch.isfinite(v)

    # all-move batch: drop branch contributes exactly nothing …
    assert all_move["piece_loss"].item() == 0.0
    assert all_move["drop_to_loss"].item() == 0.0
    assert all_move["from_loss"].item() > 0.0
    # … and vice versa
    assert all_drop["from_loss"].item() == 0.0
    assert all_drop["to_loss"].item() == 0.0
    assert all_drop["promo_loss"].item() == 0.0
    assert all_drop["piece_loss"].item() > 0.0
