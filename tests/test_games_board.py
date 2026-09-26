"""Tests for the board-game adapters (BoardEncoder + FromToDecoder / PlacementDecoder).

Covers the frozen adapter contract:
  1. forward output shapes for both decoders
  2. gradient flow into encoder / decoder / Stage-1 / TRM
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness — rollout log-probs equal teacher-forced replay
  5. legal-mask respect during sampling
  6. il_loss finiteness (incl. the fully-masked-out batch)
  7. masked-entropy gradients
  8. conditional legality masks (to_cond / to_of, promotion_cond / _of):
     legal RL sampling, exact replay, masked IL loss, il_head_terms
"""

import random

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
    fromto_legal_tables,
    fromto_replay_legal,
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


# ======================================================================
# 8. Conditional legality masks (to_cond / to_of, promotion_cond / _of)
# ======================================================================

def random_move_sets(rng: random.Random, n_promotions: int, n: int = N):
    """A random legal move list [(from, to, promo)]: a few from-squares, each
    with ≥ 1 destination; some (from, to) pairs are promotions (all promotion
    pieces legal, no plain move — as in chess)."""
    moves = []
    for f in rng.sample(range(n), rng.randint(1, 6)):
        for t in rng.sample(range(n), rng.randint(1, 4)):
            if n_promotions > 1 and rng.random() < 0.35:
                moves += [(f, t, p) for p in range(1, n_promotions)]
            else:
                moves.append((f, t, 0))
    return moves


def stack_tables(move_sets, n_promotions):
    tables = [fromto_legal_tables(m, N, n_promotions) for m in move_sets]
    return {k: torch.stack([tb[k] for tb in tables]) for k in tables[0]}


def test_fromto_legal_tables():
    tb = fromto_legal_tables([(3, 5, 0), (3, 9, 2), (3, 9, 4), (7, 1, 0)], N, 5)
    assert tb["from"].dtype == torch.bool and tb["from"].nonzero().flatten().tolist() == [3, 7]
    assert tb["to_cond"][3].nonzero().flatten().tolist() == [5, 9]
    assert tb["to_cond"][7].nonzero().flatten().tolist() == [1]
    assert tb["promotion_cond"][3, 9].nonzero().flatten().tolist() == [2, 4]
    assert tb["promotion_cond"][3, 5].nonzero().flatten().tolist() == [0]
    assert tb["promotion_cond"].shape == (N, N, 5)
    empty = fromto_legal_tables([], N, 0)
    assert not empty["from"].any() and empty["promotion_cond"].shape == (N, N, 1)
    out = {"legal_from": 1, "legal_to": 2, "legal_promotion": 3, "action_from": 4}
    assert fromto_replay_legal(out) == {"from": 1, "to_of": 2, "promotion_of": 3}
    assert fromto_replay_legal({"legal_from": 1}) == {"from": 1}


@pytest.mark.parametrize("n_promotions", [5, 0])
def test_ppo_exactness_fromto_conditional(n_promotions):
    """step() rollouts under exact conditional tables only ever sample legal
    moves, and replaying them — with the recorded masks via
    fromto_replay_legal, or with the full tables — gives the rollout
    log-probs back."""
    torch.manual_seed(11)
    rng = random.Random(11)
    cfg = make_cfg(n_promotions=n_promotions)
    has_promo = n_promotions > 1
    model = build(cfg, FromToDecoder)
    model.eval()
    B, T, K = 3, 5, 2
    obs = make_obs(cfg, B, T)

    cache = None
    rec = {k: [] for k in ("lp", "from", "to", "promo", "legal_from", "legal_to",
                           "legal_promotion", "t_from", "t_to_cond", "t_promotion_cond")}
    for t in range(T):
        move_sets = [random_move_sets(rng, n_promotions) for _ in range(B)]
        legal = stack_tables(move_sets, n_promotions)
        with torch.no_grad():
            out = model.step({k: v[:, t] for k, v in obs.items()}, legal=legal,
                             st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        for b in range(B):
            mv = (int(out["action_from"][b]), int(out["action_to"][b]),
                  int(out["action_promotion"][b]))
            assert mv in set(move_sets[b]), (t, b, mv)
        assert out["legal_from"].shape == (B, N) and out["legal_to"].shape == (B, N)
        assert ("legal_promotion" in out) == has_promo
        rec["lp"].append(out["action_log_prob"])
        rec["from"].append(out["action_from"])
        rec["to"].append(out["action_to"])
        rec["promo"].append(out["action_promotion"])
        for k in ("legal_from", "legal_to", "legal_promotion"):
            if k in out:
                rec[k].append(out[k])
        for k in ("from", "to_cond", "promotion_cond"):
            rec["t_" + k].append(legal[k])

    st = {k: torch.stack(v, 1) for k, v in rec.items() if v}
    rollout_lp = st["lp"]
    labels = {"from_idx": st["from"], "to_idx": st["to"]}
    actions = {**labels, "promotion": st["promo"]}

    replay_legal = fromto_replay_legal(st)
    assert set(replay_legal) == ({"from", "to_of", "promotion_of"} if has_promo
                                 else {"from", "to_of"})
    table_legal = {"from": st["t_from"], "to_cond": st["t_to_cond"],
                   "promotion_cond": st["t_promotion_cond"]}
    with torch.no_grad():
        for legal in (replay_legal, table_legal):
            pred = model(obs, labels=labels, legal=legal, act_steps=K)
            assert torch.equal(pred["from_mask"], st["legal_from"])
            assert torch.equal(pred["to_mask"], st["legal_to"])
            if has_promo:
                assert torch.equal(pred["promotion_mask"], st["legal_promotion"])
            lp, ent, _ = model.compute_log_prob(pred, actions)
            assert torch.allclose(lp, rollout_lp, atol=1e-5)
            assert torch.isfinite(ent).all()
        # the masks matter: the unmasked replay is a different distribution
        pred = model(obs, labels=labels, act_steps=K)
        lp_free, _, _ = model.compute_log_prob(pred, actions)
    assert not torch.allclose(lp_free, rollout_lp, atol=1e-3)


def test_ppo_exactness_fromto_conditional_bf16():
    """Decoder level (same logits on both paths), bf16 logits, conditional
    masks: sample()'s bookkeeping equals log_prob_entropy() bit for bit —
    with the full tables, with the recorded replay masks, and with decode
    echoes — and all three equal the fp32 masked distribution."""
    torch.manual_seed(12)
    rng = random.Random(12)
    cfg = make_cfg()
    P = cfg.n_promotions
    decoder = FromToDecoder(cfg)
    B, T = 4, 6
    sets = [[random_move_sets(rng, P) for _ in range(B)] for _ in range(T)]
    per_t = [stack_tables(s, P) for s in sets]
    legal = {k: torch.stack([lt[k] for lt in per_t], 1) for k in per_t[0]}
    pred = {
        "from_logits": (torch.randn(B, T, N) * 5).bfloat16(),
        "to_logits": (torch.randn(B, T, N) * 5).bfloat16(),
        "promotion_logits": (torch.randn(B, T, P) * 5).bfloat16(),
    }
    arange = torch.arange(B)
    lps, acts, used = [], {"from_idx": [], "to_idx": [], "promotion": []}, {}
    for t in range(T):
        pred_t = {k: v[:, t] for k, v in pred.items()}
        legal_t = per_t[t]
        # decode() commits from/to under the tables; replicate that here
        pred_t["from_idx"] = masked_sample(pred_t["from_logits"], legal_t["from"])
        pred_t["to_idx"] = masked_sample(
            pred_t["to_logits"], legal_t["to_cond"][arange, pred_t["from_idx"]])
        out = decoder.sample(pred_t, legal_t)
        for b in range(B):
            mv = (int(out["action_from"][b]), int(out["action_to"][b]),
                  int(out["action_promotion"][b]))
            assert mv in set(sets[t][b])
        lps.append(out["action_log_prob"])
        acts["from_idx"].append(out["action_from"])
        acts["to_idx"].append(out["action_to"])
        acts["promotion"].append(out["action_promotion"])
        for k in ("legal_from", "legal_to", "legal_promotion"):
            used.setdefault(k, []).append(out[k])
    actions = {k: torch.stack(v, 1) for k, v in acts.items()}
    used = {k: torch.stack(v, 1) for k, v in used.items()}
    rollout_lp = torch.stack(lps, 1)

    echoed = {**pred, "from_mask": used["legal_from"], "to_mask": used["legal_to"],
              "promotion_mask": used["legal_promotion"]}
    for p, lg in ((pred, legal), (pred, fromto_replay_legal(used)), (echoed, None)):
        lp, ent, head_ent = decoder.log_prob_entropy(p, actions, lg)
        assert lp.dtype == torch.float32 and torch.equal(lp, rollout_lp)
        assert torch.isfinite(ent).all() and "ent_promo" in head_ent

    def ref_lp(logits, idx, mask):
        m = logits.float().masked_fill(~mask, float("-inf"))
        return F.log_softmax(m, -1).gather(-1, idx.unsqueeze(-1)).squeeze(-1)

    ref = (ref_lp(pred["from_logits"], actions["from_idx"], used["legal_from"])
           + ref_lp(pred["to_logits"], actions["to_idx"], used["legal_to"])
           + ref_lp(pred["promotion_logits"], actions["promotion"], used["legal_promotion"]))
    assert torch.equal(rollout_lp, ref)


def test_fromto_of_masks_only_constrain_labelled_rows():
    """to_of / promotion_of describe the LABELLED move: rows without a label
    (none given, or −1) are unconstrained; the rollout path (no labels)
    ignores them; float {0, 1} masks (gm-shard bits) are accepted."""
    torch.manual_seed(13)
    cfg = make_cfg()
    P = cfg.n_promotions
    decoder = FromToDecoder(cfg)
    B, T = 3, 4
    tokens = torch.randn(B * T, N, cfg.d_model)
    ll = torch.randn(B * T, cfg.d_model)
    to_of = (torch.rand(B, T, N) < 0.2).float()
    to_of[..., 0] = 1.0
    promo_of = torch.zeros(B, T, P)
    promo_of[..., 0] = 1.0
    legal = {"to_of": to_of, "promotion_of": promo_of}

    pred = decoder.decode(tokens, ll, ll, B, T, legal=legal)
    assert pred["to_mask"].dtype == torch.bool and pred["to_mask"].all()
    assert pred["promotion_mask"].all()
    assert "from_mask" not in pred

    labels = {"from_idx": torch.randint(0, N, (B, T)), "to_idx": torch.randint(0, N, (B, T))}
    labels["from_idx"][0, :2] = -1
    labels["to_idx"][1, 1] = -1
    pred = decoder.decode(tokens, ll, ll, B, T, legal=legal, labels=labels)
    rows = labels["from_idx"] >= 0
    assert torch.equal(pred["to_mask"][rows], to_of.bool()[rows])
    assert pred["to_mask"][~rows].all()
    prow = rows & (labels["to_idx"] >= 0)
    assert torch.equal(pred["promotion_mask"][prow], promo_of.bool()[prow])
    assert pred["promotion_mask"][~prow].all()
    # labelled rows commit to the labels; the −1 rows sampled their own squares
    assert torch.equal(pred["from_idx"][rows], labels["from_idx"][rows])

    # sample() without echoes: a rollout has no labels, so *_of is ignored
    pred_t = {k: v[:, 0] for k, v in pred.items() if not k.endswith("_mask")}
    torch.manual_seed(0)
    got = decoder.sample(pred_t, {"to_of": to_of[:, 0], "promotion_of": promo_of[:, 0]})
    torch.manual_seed(0)
    ref = decoder.sample(pred_t, None)
    assert torch.equal(got["action_log_prob"], ref["action_log_prob"])
    assert torch.equal(got["action_promotion"], ref["action_promotion"])
    assert not any(k.startswith("legal_") for k in got)


def _masked_il_setup(n_promotions=5, seed=14):
    torch.manual_seed(seed)
    cfg = make_cfg(n_promotions=n_promotions)
    model = build(cfg, FromToDecoder)
    model.train()
    B, T = 3, 4
    P = max(n_promotions, 1)
    obs = make_obs(cfg, B, T)
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[0, 2:] = False                      # padded tail
    tgt = {
        "from_idx": torch.randint(0, N, (B, T)),
        "to_idx": torch.randint(0, N, (B, T)),
        "promotion": torch.randint(0, P, (B, T)),
    }
    lf = (torch.rand(B, T, N) < 0.3).float()
    lt = (torch.rand(B, T, N) < 0.3).float()
    lp = (torch.rand(B, T, P) < 0.5).float()
    bi, ti = torch.arange(B)[:, None], torch.arange(T)[None, :]
    lf[bi, ti, tgt["from_idx"]] = 1.0
    lt[bi, ti, tgt["to_idx"]] = 1.0
    lp[bi, ti, tgt["promotion"]] = 1.0
    lt[1, 1, tgt["to_idx"][1, 1]] = 0.0          # one illegal to-target
    for m in (lf, lt, lp):
        m[0, 2:] = 0.0                            # padded frames: all-False rows
    tgt = {k: v.masked_fill(~seq_mask, 0) for k, v in tgt.items()}   # collate pads 0
    legal = {"from": lf, "to_of": lt, "promotion_of": lp}
    labels = {"from_idx": tgt["from_idx"], "to_idx": tgt["to_idx"]}
    return model, obs, seq_mask, tgt, legal, labels


def test_il_loss_fromto_masked():
    model, obs, seq_mask, tgt, legal, labels = _masked_il_setup()
    pred = model(obs, legal=legal, labels=labels)
    assert {"from_mask", "to_mask", "promotion_mask"} <= set(pred)
    assert not pred["from_mask"][0, 2:].any()     # padded rows stay all-False
    dec = model.action_decoder
    out = dec.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    for k in ("from_loss", "to_loss", "promo_loss", "from_acc", "to_acc", "promo_acc",
              "illegal_target_frac"):
        assert torch.isfinite(out[k]) and not out[k].requires_grad, k
    n_valid = seq_mask.sum().item()
    assert out["illegal_target_frac"].item() == pytest.approx(1.0 / n_valid)
    # masked CE ≤ unmasked CE (the legal set is a subset of the squares)
    free = dec.il_loss({k: v for k, v in pred.items() if not k.endswith("_mask")},
                       tgt, seq_mask)
    assert "illegal_target_frac" not in free
    assert out["from_loss"] <= free["from_loss"] + 1e-6
    # masked argmax: accuracy is measured among the legal squares only
    ml = pred["from_logits"].detach().masked_fill(~pred["from_mask"], float("-inf"))
    ok = (ml.argmax(-1) == tgt["from_idx"]).float() * seq_mask.float()
    assert out["from_acc"].item() == pytest.approx((ok.sum() / seq_mask.sum()).item())
    assert torch.equal(dec.il_head_terms(pred, tgt, seq_mask)["from"]["mask"],
                       pred["from_mask"])
    out["loss"].backward()
    assert_grad_flow(model)
    assert all(torch.isfinite(p.grad).all()
               for p in model.parameters() if p.grad is not None)


def test_il_loss_fromto_masked_bf16_autocast():
    model, obs, seq_mask, tgt, legal, labels = _masked_il_setup(seed=15)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = model(obs, legal=legal, labels=labels)
        out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(torch.isfinite(p.grad).all()
               for p in model.parameters() if p.grad is not None)


def _reconstruct(terms):
    total = None
    for t in terms.values():
        part = (t["ce"] * t["weight"]).sum() / t["norm"]
        total = part if total is None else total + part
    return total


@pytest.mark.parametrize("n_promotions", [5, 0])
@pytest.mark.parametrize("masked", [False, True])
def test_il_head_terms_reconstruct_fromto(n_promotions, masked):
    model, obs, seq_mask, tgt, legal, labels = _masked_il_setup(n_promotions, seed=16)
    pred = model(obs, legal=legal if masked else None, labels=labels)
    dec = model.action_decoder
    terms = dec.il_head_terms(pred, tgt, seq_mask)
    assert list(terms) == ["from", "to"] + (["promotion"] if n_promotions > 1 else [])
    B, T = seq_mask.shape
    for t in terms.values():
        assert t["kind"] == "categorical"
        assert t["ce"].shape == (B, T) and t["weight"].shape == (B, T) and t["norm"].dim() == 0
        assert ("mask" in t) == masked
        if masked:
            assert t["mask"].shape == t["logits"].shape and t["mask"].dtype == torch.bool
    assert terms["from"]["logits"] is pred["from_logits"]
    assert torch.equal(terms["from"]["weight"], seq_mask.float())
    assert torch.equal(_reconstruct(terms), dec.il_loss(pred, tgt, seq_mask)["loss"])


@pytest.mark.parametrize("allow_pass", [True, False])
def test_il_head_terms_reconstruct_placement(allow_pass):
    torch.manual_seed(17)
    cfg = make_cfg(allow_pass=allow_pass)
    model = build(cfg, PlacementDecoder)
    B, T = 3, 4
    pred = model(make_obs(cfg, B, T))
    tgt = {"place_idx": torch.randint(0, N, (B, T)), "passed": torch.rand(B, T) < 0.3}
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, 3:] = False
    dec = model.action_decoder
    terms = dec.il_head_terms(pred, tgt, seq_mask)
    assert list(terms) == (["place", "pass"] if allow_pass else ["place"])
    assert terms["place"]["kind"] == "categorical"
    assert terms["place"]["logits"] is pred["place_logits"]
    if allow_pass:
        assert terms["pass"]["kind"] == "bernoulli"
        assert terms["pass"]["logits"] is pred["pass_logits"]
        assert torch.equal(terms["place"]["weight"],
                           seq_mask.float() * (1.0 - tgt["passed"].float()))
    assert torch.equal(_reconstruct(terms), dec.il_loss(pred, tgt, seq_mask)["loss"])
