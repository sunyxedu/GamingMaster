"""Tests for the Shogi adapter (ShogiEncoder + ShogiDecoder).

Covers the frozen adapter contract:
  1. forward output shapes (both branches always decoded)
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal teacher-forced replay
  5. legal-mask respect for from / to / drop_piece / drop_to
  6. il_loss finiteness (incl. the fully-masked-out batch)
  7. il_loss branch supervision independent of the mode head
  8. conditional legality masks: tables / X_of resolution, masked 2-class
     mode & promotion, exact replay, masked IL loss, il_head_terms

Uses a tiny 5×5 board to keep the suite fast: n_tokens = 25 + 14 = 39.
"""

import random

import pytest
import torch
import torch.nn.functional as F

from model.game_network import GameNetwork
from model.games.base import masked_sample
from model.games.shogi import (
    ShogiConfig,
    ShogiDecoder,
    ShogiEncoder,
    shogi_legal_tables,
    shogi_replay_legal,
)

NB = 25  # 5×5 board tokens
NH = 14  # 7 hand piece types × 2 players


def make_cfg(**overrides) -> ShogiConfig:
    kw = dict(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
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


# ======================================================================
# 8. Conditional legality masks (schema freeze): tables, X_of rows,
#    masked Bernoullis, exact replay, masked IL loss, il_head_terms
# ======================================================================

HEADS = {"mode", "from", "to", "promo", "piece", "drop_to"}
REPLAY_KEYS = {"mode", "from", "to_of", "promo_of", "drop_piece", "drop_to_of"}


def two_class(logit: torch.Tensor) -> torch.Tensor:
    return torch.stack([torch.zeros_like(logit), logit], -1)


def random_move_set(rng: random.Random, kind: str) -> list:
    """(dropped, from, to, promote, piece, drop_to) tuples: board moves with
    no-promotion-only / forced-promotion / optional-promotion targets and
    drops of a few hand slots; kind ∈ both / move (move-only) / drop."""
    moves = []
    if kind in ("both", "move"):
        for f in rng.sample(range(NB), rng.randint(1, 3)):
            for t in rng.sample(range(NB), rng.randint(1, 3)):
                for p in rng.choice(([False], [True], [False, True])):
                    moves.append((0, f, t, p, 0, 0))
    if kind in ("both", "drop"):
        for pc in rng.sample(range(7), rng.randint(1, 3)):
            for s in rng.sample(range(NB), rng.randint(1, 4)):
                moves.append((1, 0, 0, False, pc, s))
    return moves


def random_tables(B: int, T: int, seed: int) -> dict:
    """legal tables [B, T, ...] (shogi_legal_tables per frame)."""
    rng = random.Random(seed)
    kinds = ("both", "move", "drop")
    rows = [[shogi_legal_tables(random_move_set(rng, kinds[(b + t) % 3]), NB, NH)
             for t in range(T)] for b in range(B)]
    return {k: torch.stack([torch.stack([rows[b][t][k] for t in range(T)]) for b in range(B)])
            for k in rows[0][0]}


def assert_actions_legal(legal: dict, a: dict):
    """Every sampled action lies in the full conditional tables."""
    B, T = a["dropped"].shape
    for b in range(B):
        for t in range(T):
            L = {k: v[b, t] for k, v in legal.items()}
            if bool(a["dropped"][b, t]):
                pc, sq = int(a["piece_idx"][b, t]), int(a["drop_to_idx"][b, t])
                assert L["mode"][1] and L["drop_piece"][pc] and L["drop_to_cond"][pc, sq]
            else:
                f, to = int(a["from_idx"][b, t]), int(a["to_idx"][b, t])
                p = int(a["promoted"][b, t])
                assert L["mode"][0] and L["from"][f] and L["to_cond"][f, to]
                assert L["promo_cond"][f, to, p]


def stacked_actions(st: dict) -> dict:
    return {
        "dropped": st["action_dropped"], "from_idx": st["action_from"],
        "to_idx": st["action_to"], "promoted": st["action_promoted"],
        "piece_idx": st["action_piece"], "drop_to_idx": st["action_drop_to"],
    }


def test_legal_tables_helper():
    moves = [(0, 3, 7, False, 0, 0), (0, 3, 7, True, 0, 0), (0, 3, 8, True, 0, 0),
             (1, 0, 0, False, 2, 11)]
    L = shogi_legal_tables(moves, NB, NH)
    assert L["mode"].tolist() == [True, True]
    assert L["from"].nonzero().flatten().tolist() == [3]
    assert L["to_cond"][3].nonzero().flatten().tolist() == [7, 8]
    assert L["promo_cond"][3, 7].tolist() == [True, True]
    assert L["promo_cond"][3, 8].tolist() == [False, True]      # forced promotion
    assert L["drop_piece"].nonzero().flatten().tolist() == [2]
    assert L["drop_to_cond"][2].nonzero().flatten().tolist() == [11]
    assert shogi_legal_tables([], NB, NH)["mode"].tolist() == [False, False]


def test_conditional_mask_rollout_legal_and_exact_replay():
    """step() rollout under full conditional tables (forced promotions,
    move-only and drop-only frames): every sampled action is legal; the
    replay with the masks sample() used (X_of rows) + labels reproduces
    the rollout log-probs, and replaying with the full tables instead gives
    bit-identical log-probs (same resolved rows)."""
    torch.manual_seed(21)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T, K = 3, 6, 2
    obs = make_obs(cfg, B, T)
    legal = random_tables(B, T, seed=21)

    cache, outs = None, []
    for t in range(T):
        with torch.no_grad():
            out = model.step({k: v[:, t] for k, v in obs.items()},
                             legal={k: v[:, t] for k, v in legal.items()},
                             st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        outs.append(out)
    st = {k: torch.stack([o[k] for o in outs], 1)
          for k in outs[0] if k.startswith(("action_", "legal_"))}
    actions = stacked_actions(st)
    assert_actions_legal(legal, actions)
    assert actions["dropped"].any() and not actions["dropped"].all()

    replay_legal = shogi_replay_legal(st)
    assert set(replay_legal) == REPLAY_KEYS
    labels = {"dropped": actions["dropped"].long(), "from_idx": actions["from_idx"],
              "to_idx": actions["to_idx"], "piece_idx": actions["piece_idx"]}
    rollout_lp = st["action_log_prob"]
    with torch.no_grad():
        pred = model(obs, legal=replay_legal, labels=labels, act_steps=K)
        lp, ent, _ = model.compute_log_prob(pred, actions)
        pred_full = model(obs, legal=legal, labels=labels, act_steps=K)
        lp_full, ent_full, _ = model.compute_log_prob(pred_full, actions)

    assert torch.isfinite(rollout_lp).all()
    assert torch.allclose(lp, rollout_lp, atol=1e-5)
    assert torch.equal(lp_full, lp) and torch.equal(ent_full, ent)
    for head, key in (("mode", "legal_mode"), ("to", "legal_to"), ("promo", "legal_promo"),
                      ("drop_to", "legal_drop_to")):
        assert torch.equal(pred[f"{head}_mask"], st[key])
        assert torch.equal(pred_full[f"{head}_mask"], st[key])


def test_legacy_step_has_no_new_outputs():
    """Without mode / promotion masks the legacy Bernoulli paths run: no
    mode / promo echoes, and no legal_* outputs at all without legal."""
    torch.manual_seed(22)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    obs = {k: v[:, 0] for k, v in make_obs(cfg, 2, 1).items()}
    with torch.no_grad():
        out = model.step(obs)
        out_l = model.step(obs, legal={"from": random_mask(2, NB)})
    assert not any(k.endswith("_mask") or k.startswith("legal_") for k in out)
    assert "from_mask" in out_l and "legal_from" in out_l
    assert "mode_mask" not in out_l and "legal_promo" not in out_l


def test_float_masks_match_bool_masks():
    torch.manual_seed(23)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T = 2, 3
    obs = make_obs(cfg, B, T)
    legal = random_tables(B, T, seed=23)
    labels = {"dropped": torch.zeros(B, T, dtype=torch.long),
              "from_idx": legal["from"].float().argmax(-1)}
    with torch.no_grad():
        torch.manual_seed(0)
        a = model(obs, legal=legal, labels=labels)
        torch.manual_seed(0)
        b = model(obs, legal={k: v.float() for k, v in legal.items()}, labels=labels)
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_decode_mask_resolution_precedence():
    """cond > of > flat; an X_of row applies only where every committed
    index equals its label (label −1 → flat mask, or unconstrained)."""
    torch.manual_seed(24)
    cfg = make_cfg()
    dec = ShogiDecoder(cfg)
    B, T = 2, 3
    BT, d = B * T, cfg.d_model
    tokens = torch.randn(BT, NB + NH, d)
    ll = torch.randn(BT, d)

    def rnd(*shape):
        m = torch.rand(*shape) < 0.3
        m[..., 0] = True
        return m

    to_cond, to_of, to_flat = rnd(B, T, NB, NB), rnd(B, T, NB), rnd(B, T, NB)
    promo_of = torch.tensor([True, False]).expand(B, T, 2)
    drop_to_cond, drop_to_of = rnd(B, T, NH, NB), rnd(B, T, NB)
    lbl_from = torch.tensor([[3, -1, 5], [7, 7, -1]])
    lbl_to = torch.tensor([[4, 4, -1], [2, 2, 2]])
    lbl_piece = torch.tensor([[1, -1, 2], [0, 0, 6]])
    labels = {"from_idx": lbl_from, "to_idx": lbl_to, "piece_idx": lbl_piece}
    with torch.no_grad():
        full = dec.decode(tokens, ll, ll, B, T, legal={
            "to_cond": to_cond, "to_of": to_of, "to": to_flat,
            "drop_to_cond": drop_to_cond, "drop_to_of": drop_to_of,
        }, labels=labels)
        of = dec.decode(tokens, ll, ll, B, T, legal={
            "to_of": to_of, "to": to_flat, "promo_of": promo_of, "drop_to_of": drop_to_of,
        }, labels=labels)
        bare = dec.decode(tokens, ll, ll, B, T, legal={"to_of": to_of, "promo_of": promo_of})
    bi, ti = torch.arange(B)[:, None], torch.arange(T)[None, :]
    assert torch.equal(full["to_mask"], to_cond[bi, ti, full["from_idx"]])
    assert torch.equal(full["drop_to_mask"], drop_to_cond[bi, ti, full["piece_idx"]])
    assert "promo_mask" not in full and "mode_mask" not in full

    ok_from = lbl_from >= 0
    assert torch.equal(of["from_idx"][ok_from], lbl_from[ok_from])
    assert torch.equal(of["to_mask"], torch.where(ok_from[..., None], to_of, to_flat))
    everything = torch.ones(B, T, 2, dtype=torch.bool)
    applies_promo = (ok_from & (lbl_to >= 0))[..., None]
    assert torch.equal(of["promo_mask"], torch.where(applies_promo, promo_of, everything))
    ok_piece = (lbl_piece >= 0)[..., None]
    everywhere = torch.ones(B, T, NB, dtype=torch.bool)
    assert torch.equal(of["drop_to_mask"], torch.where(ok_piece, drop_to_of, everywhere))
    # no labels at all: X_of rows never apply
    assert bare["to_mask"].all() and bare["promo_mask"].all()


def make_masked_pred(B: int, T: int, dtype=torch.float32, seed: int = 0) -> dict:
    """Decoder-level pred with random logits, echoed masks (mode / promo
    rows incl. forced options, ≥1 legal entry everywhere) and committed
    indices sampled under them, as decode() would."""
    g = torch.Generator().manual_seed(seed)
    opts = torch.tensor([[True, True], [True, False], [False, True]])
    pred = {
        "mode_logits": torch.randn(B, T, generator=g) * 3,
        "from_logits": torch.randn(B, T, NB, generator=g) * 5,
        "to_logits": torch.randn(B, T, NB, generator=g) * 5,
        "promo_logits": torch.randn(B, T, generator=g) * 3,
        "piece_logits": torch.randn(B, T, NH, generator=g) * 5,
        "drop_to_logits": torch.randn(B, T, NB, generator=g) * 5,
    }
    pred = {k: v.to(dtype) for k, v in pred.items()}
    pred["mode_mask"] = opts[torch.randint(0, 3, (B, T), generator=g)]
    pred["promo_mask"] = opts[torch.randint(0, 3, (B, T), generator=g)]
    for k, n in (("from", NB), ("to", NB), ("piece", NH), ("drop_to", NB)):
        pred[f"{k}_mask"] = torch.stack([random_mask(B, n) for _ in range(T)], 1)
    pred["dropped"] = masked_sample(two_class(pred["mode_logits"].float()),
                                    pred["mode_mask"]).bool()
    pred["from_idx"] = masked_sample(pred["from_logits"], pred["from_mask"])
    pred["to_idx"] = masked_sample(pred["to_logits"], pred["to_mask"])
    pred["piece_idx"] = masked_sample(pred["piece_logits"], pred["piece_mask"])
    return pred


def rollout_decoder(dec: ShogiDecoder, pred: dict):
    T = pred["dropped"].shape[1]
    outs = [dec.sample({k: v[:, t] for k, v in pred.items()}) for t in range(T)]
    st = {k: torch.stack([o[k] for o in outs], 1) for k in outs[0]}
    return st, stacked_actions(st)


def test_ppo_exactness_bf16_conditional_masks():
    """bf16 logits + masked 2-class mode / promotion: sample()'s bookkeeping
    and log_prob_entropy()'s replay agree bit-for-bit, and both equal the
    fp32 evaluation of the masked distributions."""
    torch.manual_seed(25)
    dec = ShogiDecoder(make_cfg())
    B, T = 4, 6
    pred = make_masked_pred(B, T, dtype=torch.bfloat16, seed=25)
    st, actions = rollout_decoder(dec, pred)
    lp, ent, head_ent = dec.log_prob_entropy(pred, actions)

    assert lp.dtype == torch.float32
    assert torch.equal(lp, st["action_log_prob"])
    assert torch.isfinite(ent).all()
    # sampled leaves respect their masks
    assert pred["promo_mask"].gather(-1, actions["promoted"].long()[..., None]).all()
    assert pred["drop_to_mask"].gather(-1, actions["drop_to_idx"][..., None]).all()

    def ref(logits, idx, mask):
        m = logits.float().masked_fill(~mask, float("-inf"))
        return F.log_softmax(m, -1).gather(-1, idx[..., None]).squeeze(-1)

    mode = ref(two_class(pred["mode_logits"].float()), actions["dropped"].long(),
               pred["mode_mask"])
    promo = ref(two_class(pred["promo_logits"].float()), actions["promoted"].long(),
                pred["promo_mask"])
    move = (ref(pred["from_logits"], actions["from_idx"], pred["from_mask"])
            + ref(pred["to_logits"], actions["to_idx"], pred["to_mask"]) + promo)
    drop = (ref(pred["piece_logits"], actions["piece_idx"], pred["piece_mask"])
            + ref(pred["drop_to_logits"], actions["drop_to_idx"], pred["drop_to_mask"]))
    assert torch.equal(lp, mode + torch.where(actions["dropped"], drop, move))


def test_forced_options_zero_log_prob_and_entropy():
    """Single legal option on every head (forced move / forced drop, forced
    promotion): log-prob 0 and entropy 0 in rollout and replay."""
    dec = ShogiDecoder(make_cfg())
    B, T = 4, 1
    pred = make_masked_pred(B, T, seed=26)
    pred["mode_mask"] = torch.tensor([[[True, False]], [[False, True]]] * 2)
    pred["promo_mask"] = torch.tensor([False, True]).expand(B, T, 2)
    for k, n in (("from", NB), ("to", NB), ("piece", NH), ("drop_to", NB)):
        pred[f"{k}_mask"] = F.one_hot(torch.randint(0, n, (B, T)), n).bool()
    pred["dropped"] = pred["mode_mask"][..., 1]
    for k in ("from", "to", "piece"):
        pred[f"{k}_idx"] = pred[f"{k}_mask"].long().argmax(-1)
    st, actions = rollout_decoder(dec, pred)
    assert actions["promoted"].all()
    assert torch.equal(actions["dropped"], pred["mode_mask"][..., 1])
    lp, ent, head_ent = dec.log_prob_entropy(pred, actions)
    assert (st["action_log_prob"] == 0).all() and (lp == 0).all()
    assert (ent == 0).all()
    assert head_ent["ent_mode"] == 0 and head_ent["ent_promo"] == 0


def converter_style_legal(tgt: dict, B: int, T: int, seed: int) -> dict:
    """tgt.legal_* style float masks (rows for the ground-truth
    conditioning) that contain every target."""
    g = torch.Generator().manual_seed(seed)

    def rnd(n, target):
        m = torch.rand(B, T, n, generator=g) < 0.3
        m.scatter_(-1, target[..., None], True)
        return m.float()

    return {
        "mode": rnd(2, tgt["dropped"].clamp(min=0)),
        "from": rnd(NB, tgt["from_idx"]),
        "to_of": rnd(NB, tgt["to_idx"]),
        "promo_of": rnd(2, tgt["promoted"].long()),
        "drop_piece": rnd(NH, tgt["piece_idx"]),
        "drop_to_of": rnd(NB, tgt["drop_to_idx"]),
    }


def tf_labels(tgt: dict) -> dict:
    return {k: tgt[k] for k in ("dropped", "from_idx", "to_idx", "piece_idx")}


def test_masked_il_loss_padding_and_illegal_targets():
    torch.manual_seed(27)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 3, 4
    tgt = make_targets(B, T)
    tgt["dropped"][1, 1] = 0                     # a move frame …
    legal = converter_style_legal(tgt, B, T, seed=27)
    legal["from"][1, 1, tgt["from_idx"][1, 1]] = 0.0   # … whose from target is illegal
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[0, -1] = False                      # padded frame: all-zero masks
    for v in legal.values():
        v[0, -1] = 0.0

    pred = model(make_obs(cfg, B, T), legal=legal, labels=tf_labels(tgt))
    for k in ("mode_mask", "from_mask", "to_mask", "promo_mask", "piece_mask", "drop_to_mask"):
        assert pred[k].dtype == torch.bool
    pred["from_logits"].retain_grad()
    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    assert all(torch.isfinite(v).all() for k, v in out.items() if not k.startswith("answer_"))
    assert out["illegal_target_frac"].item() == pytest.approx(1.0 / (B * T - 1))
    out["loss"].backward()
    assert_grad_flow(model)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    g = pred["from_logits"].grad
    assert torch.isfinite(g).all() and (g[1, 1] == 0).all() and (g[0, -1] == 0).all()
    terms = model.action_decoder.il_head_terms(pred, tgt, seq_mask)
    assert terms["from"]["weight"][1, 1] == 0 and terms["mode"]["weight"][1, 1] == 1


def test_masked_il_loss_with_all_legal_masks_matches_unmasked():
    torch.manual_seed(28)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T = 2, 3
    tgt = make_targets(B, T)
    obs = make_obs(cfg, B, T)
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    ones = {"mode": torch.ones(B, T, 2), "from": torch.ones(B, T, NB),
            "to_of": torch.ones(B, T, NB), "promo_of": torch.ones(B, T, 2),
            "drop_piece": torch.ones(B, T, NH), "drop_to_of": torch.ones(B, T, NB)}
    with torch.no_grad():
        plain = model.action_decoder.il_loss(model(obs, labels=tf_labels(tgt)), tgt, seq_mask)
        masked = model.action_decoder.il_loss(
            model(obs, legal=ones, labels=tf_labels(tgt)), tgt, seq_mask)
    assert masked["illegal_target_frac"] == 0 and "illegal_target_frac" not in plain
    for k, v in plain.items():
        if k.startswith("answer_"):
            assert torch.equal(v, masked[k]), k
        else:
            assert torch.allclose(v, masked[k], atol=1e-5), k


def _legacy_il_loss(pred, tgt, seq_mask):
    """The pre-mask ShogiDecoder.il_loss (loss + metrics), verbatim."""
    B, T, Nb = pred["from_logits"].shape
    Nh = pred["piece_logits"].shape[-1]
    valid = seq_mask.float()
    d_lbl = tgt["dropped"].long()
    labeled = valid * (d_lbl >= 0).float()
    n_labeled = labeled.sum().clamp(min=1.0)
    tgt_dropped = (d_lbl > 0).float()
    mode_bce = F.binary_cross_entropy_with_logits(
        pred["mode_logits"], tgt_dropped, reduction="none").nan_to_num(0.0)
    mode_loss = (mode_bce * labeled).sum() / n_labeled
    mode_pred_drop = (pred["mode_logits"] > 0).long()
    mode_ok = (mode_pred_drop == d_lbl).float()
    move_w = labeled * (d_lbl == 0).float()
    drop_w = labeled * (d_lbl == 1).float()

    def _ce(logits, idx, n):
        safe = idx.long().clamp(0, n - 1)
        return F.cross_entropy(logits.reshape(-1, n), safe.reshape(-1),
                               reduction="none").reshape(B, T).nan_to_num(0.0)

    tgt_from = tgt["from_idx"].long().clamp(0, Nb - 1)
    tgt_to = tgt["to_idx"].long().clamp(0, Nb - 1)
    n_move = move_w.sum().clamp(min=1.0)
    from_loss = (_ce(pred["from_logits"], tgt_from, Nb) * move_w).sum() / n_move
    to_w = move_w * (pred["from_idx"] == tgt_from).float()
    n_to = to_w.sum().clamp(min=1.0)
    to_loss = (_ce(pred["to_logits"], tgt_to, Nb) * to_w).sum() / n_to
    promo_w = to_w * (pred["to_idx"] == tgt_to).float()
    n_promo = promo_w.sum().clamp(min=1.0)
    promo_bce = F.binary_cross_entropy_with_logits(
        pred["promo_logits"], tgt["promoted"].float(), reduction="none").nan_to_num(0.0)
    promo_loss = (promo_bce * promo_w).sum() / n_promo
    tgt_piece = tgt["piece_idx"].long().clamp(0, Nh - 1)
    tgt_drop_to = tgt["drop_to_idx"].long().clamp(0, Nb - 1)
    n_drop = drop_w.sum().clamp(min=1.0)
    piece_loss = (_ce(pred["piece_logits"], tgt_piece, Nh) * drop_w).sum() / n_drop
    drop_to_w = drop_w * (pred["piece_idx"] == tgt_piece).float()
    n_drop_to = drop_to_w.sum().clamp(min=1.0)
    drop_to_loss = (_ce(pred["drop_to_logits"], tgt_drop_to, Nb) * drop_to_w).sum() / n_drop_to
    loss = mode_loss + from_loss + to_loss + promo_loss + piece_loss + drop_to_loss
    from_ok = pred["from_logits"].argmax(-1) == tgt_from
    to_ok = pred["to_logits"].argmax(-1) == tgt_to
    promo_ok = (pred["promo_logits"] > 0) == tgt["promoted"].bool()
    piece_ok = pred["piece_logits"].argmax(-1) == tgt_piece
    drop_to_ok = pred["drop_to_logits"].argmax(-1) == tgt_drop_to
    answer_ok = (mode_pred_drop == d_lbl) & torch.where(
        d_lbl == 1, piece_ok & drop_to_ok, from_ok & to_ok & promo_ok)
    return {
        "loss": loss, "answer_correct": answer_ok,
        "answer_mask": seq_mask.bool() & (d_lbl >= 0),
        "mode_loss": mode_loss, "from_loss": from_loss, "to_loss": to_loss,
        "promo_loss": promo_loss, "piece_loss": piece_loss, "drop_to_loss": drop_to_loss,
        "mode_acc": (mode_ok * labeled).sum() / n_labeled,
        "from_acc": (from_ok.float() * move_w).sum() / n_move,
        "to_acc": (to_ok.float() * to_w).sum() / n_to,
        "promo_acc": (promo_ok.float() * promo_w).sum() / n_promo,
        "piece_acc": (piece_ok.float() * drop_w).sum() / n_drop,
        "drop_to_acc": (drop_to_ok.float() * drop_to_w).sum() / n_drop_to,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unmasked_il_loss_bit_identical_to_legacy(dtype):
    torch.manual_seed(29)
    cfg = make_cfg()
    model = build(cfg)
    model.eval()
    B, T = 3, 4
    tgt = make_targets(B, T)
    tgt["dropped"][0, 0] = -1
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[2, 2:] = False
    labels = tf_labels(tgt)
    labels["from_idx"] = torch.where(torch.rand(B, T) < 0.5, labels["from_idx"], -1)
    with torch.no_grad():
        pred = model(make_obs(cfg, B, T), labels=labels)
    pred = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in pred.items()}
    new = model.action_decoder.il_loss(pred, tgt, seq_mask)
    old = _legacy_il_loss(pred, tgt, seq_mask)
    assert set(new) == set(old)
    for k in old:
        assert torch.equal(new[k], old[k]), k


@pytest.mark.parametrize("masked", [False, True])
def test_il_head_terms_reconstruct_il_loss(masked):
    torch.manual_seed(30)
    cfg = make_cfg()
    model = build(cfg)
    model.train()
    B, T = 3, 4
    tgt = make_targets(B, T)
    tgt["dropped"][0, 1] = -1
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False
    legal = converter_style_legal(tgt, B, T, seed=30) if masked else None
    labels = tf_labels(tgt)
    labels["from_idx"] = torch.where(torch.rand(B, T) < 0.7, labels["from_idx"], -1)
    pred = model(make_obs(cfg, B, T), legal=legal, labels=labels)
    dec = model.action_decoder
    out = dec.il_loss(pred, tgt, seq_mask)
    terms = dec.il_head_terms(pred, tgt, seq_mask)
    assert set(terms) == HEADS
    total = sum((t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    assert torch.allclose(total, out["loss"], atol=1e-6, rtol=1e-6)
    for h, t in terms.items():
        assert torch.allclose((t["ce"] * t["weight"]).sum() / t["norm"], out[f"{h}_loss"])
        assert t["ce"].shape == t["weight"].shape == (B, T) and t["norm"].dim() == 0
        assert ("mask" in t) == masked
        if masked:
            assert t["mask"].shape == t["logits"].shape and t["mask"].dtype == torch.bool
    widths = {"mode": 2, "from": NB, "to": NB, "promo": 2, "piece": NH, "drop_to": NB}
    assert {h: t["logits"].shape[-1] for h, t in terms.items()} == widths
    # the 2-class view of a Bernoulli head: softmax([0, l])[1] = sigmoid(l)
    p = F.softmax(terms["promo"]["logits"].float(), -1)[..., 1]
    assert torch.allclose(p, torch.sigmoid(pred["promo_logits"].float()), atol=1e-6)
    total.backward()
    assert_grad_flow(model)
