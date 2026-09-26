"""Tests for the riichi mahjong game adapter.

Covers the adapter contract against the game-agnostic core:
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling
  6. il_loss: finite, differentiable, robust to degenerate masks

Plus the ARG_USES-specific invariant: the tile head only contributes
log-prob/entropy where the action type actually takes a tile, and the
conditional legality contract (7): tile masks conditioned on the action
type (tile_cond / tile_of) for rollout, exact replay and the masked IL
loss, and the il_head_terms decomposition.
"""

import pytest
import torch
import torch.nn.functional as F

from model.game_network import GameNetwork
from model.games.base import masked_sample
from model.games.mahjong import (
    ACTION_CHI,
    ACTION_DAHAI,
    ACTION_HORA,
    ACTION_PASS,
    ACTION_PON,
    ACTION_RIICHI,
    N_ACTION_TYPES,
    N_MJ_PLAYERS,
    N_TILE_KINDS,
    MahjongConfig,
    MahjongDecoder,
    MahjongEncoder,
    _mask_logits,
    mahjong_legal_tables,
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


# ======================================================================
# 7. Conditional legality: tile masks conditioned on the action type
# ======================================================================

def _make_cond_legal(B, T=None, seed=0):
    """Random exact legality: type mask + per-type tile rows (≥1 legal
    entry per row, so no row is degenerate)."""
    g = torch.Generator().manual_seed(seed)
    shape = (B,) if T is None else (B, T)
    types = torch.rand(*shape, N_ACTION_TYPES, generator=g) < 0.5
    types[..., ACTION_DAHAI] |= ~types.any(-1)
    cond = torch.rand(*shape, N_ACTION_TYPES, N_TILE_KINDS, generator=g) < 0.2
    cond[..., 0] |= ~cond.any(-1)
    return {"action_type": types, "tile_cond": cond}


def test_mahjong_legal_tables():
    t = mahjong_legal_tables([(ACTION_DAHAI, 3), (ACTION_DAHAI, 7), (ACTION_CHI, 1),
                              (ACTION_PASS, None), (ACTION_HORA, None)])
    assert t["action_type"].tolist() == [True, False, True, False, False, True, False, True]
    assert t["tile_cond"][ACTION_DAHAI].nonzero().flatten().tolist() == [3, 7]
    assert t["tile_cond"][ACTION_CHI].nonzero().flatten().tolist() == [1]
    assert t["tile_cond"][ACTION_PASS].all() and t["tile_cond"][ACTION_HORA].all()
    assert not t["tile_cond"][ACTION_PON].any()
    with pytest.raises(ValueError):
        mahjong_legal_tables([(ACTION_DAHAI, 34)])


def test_tile_mask_resolution_priority(cfg):
    """tile_cond row of the committed type > tile_of (only where the
    committed type equals a non-negative label) > legacy tile."""
    dec = MahjongDecoder(cfg)
    g = torch.Generator().manual_seed(0)
    B = 6
    types = torch.tensor([0, 1, 2, 3, 0, 1])
    cond = torch.rand(B, N_ACTION_TYPES, N_TILE_KINDS, generator=g) < 0.5
    flat = torch.rand(B, N_TILE_KINDS, generator=g) < 0.5
    of = torch.rand(B, N_TILE_KINDS, generator=g) < 0.5
    got = dec._resolve_tile_mask({"tile_cond": cond, "tile_of": of, "tile": flat}, types, None)
    assert torch.equal(got, cond[torch.arange(B), types])
    label = torch.tensor([0, 2, 2, -1, 0, 1])      # match, mismatch, match, −1, ...
    applies = torch.tensor([True, False, True, False, True, True])
    got = dec._resolve_tile_mask({"tile_of": of, "tile": flat}, types, label)
    assert torch.equal(got, torch.where(applies[:, None], of, flat))
    got = dec._resolve_tile_mask({"tile_of": of}, types, label)
    assert torch.equal(got[applies], of[applies]) and got[~applies].all()
    got = dec._resolve_tile_mask({"tile_of": of.float()}, types, None)   # packed-bit floats
    assert got.dtype == torch.bool and got.all()   # no label → unconstrained
    assert dec._resolve_tile_mask({"action_type": cond[:, 0]}, types, None) is None


def test_tile_cond_rollout_never_samples_illegal_tile(cfg, model):
    model.eval()
    B = 8
    for trial in range(15):
        obs = {k: v[:, 0] for k, v in _make_obs(cfg, B, T=1, seed=trial).items()}
        legal = _make_cond_legal(B, seed=trial)
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        b = torch.arange(B)
        assert legal["action_type"][b, out["action_type"]].all(), "illegal type"
        tile_row = legal["tile_cond"][b, out["action_type"]]          # [B, 34]
        assert tile_row[b, out["tile_idx"]].all(), "tile illegal for its type"
        # the masks actually used are returned for replay
        assert torch.equal(out["legal_tile"], tile_row)
        assert torch.equal(out["legal_action_type"], legal["action_type"])
        assert torch.isfinite(out["action_log_prob"]).all()


def test_tile_cond_rollout_replay_with_tile_of(cfg, model):
    """Rollout under exact tile_cond masks; replay from the stored
    (legal_action_type, legal_tile) as {'action_type', 'tile_of'} + the
    stored type as label reproduces the rollout log-probs."""
    K = 2
    model.eval()
    B, T = 3, 5
    obs = _make_obs(cfg, B, T, seed=11)
    legal_seq = _make_cond_legal(B, T, seed=11)
    cache = None
    lps = []
    rec = {k: [] for k in ("action_type", "tile_idx", "legal_action_type", "legal_tile")}
    for t in range(T):
        with torch.no_grad():
            out = model.step({k: v[:, t] for k, v in obs.items()},
                             legal={k: v[:, t] for k, v in legal_seq.items()},
                             st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        for k in rec:
            rec[k].append(out[k])
    stored = {k: torch.stack(v, 1) for k, v in rec.items()}
    actions = {"action_type": stored["action_type"], "tile_idx": stored["tile_idx"]}
    replay_legal = {"action_type": stored["legal_action_type"], "tile_of": stored["legal_tile"]}
    with torch.no_grad():
        full = model(obs, legal=replay_legal,
                     labels={"action_type": actions["action_type"]}, act_steps=K)
        lp, ent, _ = model.compute_log_prob(full, actions, replay_legal)
        # the fallback path (no echoes in pred) resolves the same masks
        bare = {k: v for k, v in full.items() if not k.endswith("_mask")}
        lp2, _, _ = model.compute_log_prob(bare, actions, replay_legal)
    assert torch.equal(full["tile_mask"], stored["legal_tile"])
    torch.testing.assert_close(lp, torch.stack(lps, 1), atol=1e-5, rtol=1e-5)
    assert torch.equal(lp, lp2)
    assert torch.isfinite(ent).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sample_replay_bitwise_decoder_level(cfg, dtype):
    """Same logits through sample() (per step) and log_prob_entropy()
    (whole sequence): bitwise-equal fp32 log-probs, bf16 logits included,
    replaying either the exact tile_cond masks or the returned legal_tile
    as tile_of."""
    torch.manual_seed(0)
    dec = MahjongDecoder(cfg)
    B, T = 4, 6
    g = torch.Generator().manual_seed(5)
    pred = {
        "action_type_logits": (torch.randn(B, T, N_ACTION_TYPES, generator=g) * 5).to(dtype),
        "tile_logits": (torch.randn(B, T, N_TILE_KINDS, generator=g) * 5).to(dtype),
    }
    legal = _make_cond_legal(B, T, seed=5)
    lps, types, tiles, ltiles = [], [], [], []
    for t in range(T):
        pred_t = {k: v[:, t] for k, v in pred.items()}
        legal_t = {k: v[:, t] for k, v in legal.items()}
        # decode() commits the type under the type mask; replicate that here
        pred_t["action_type_used"] = masked_sample(pred_t["action_type_logits"],
                                                   legal_t["action_type"])
        out = dec.sample(pred_t, legal_t)
        lps.append(out["action_log_prob"])
        types.append(out["action_type"])
        tiles.append(out["tile_idx"])
        ltiles.append(out["legal_tile"])
    actions = {"action_type": torch.stack(types, 1), "tile_idx": torch.stack(tiles, 1)}
    for replay_legal in (legal, {"action_type": legal["action_type"],
                                 "tile_of": torch.stack(ltiles, 1)}):
        lp, ent, _ = dec.log_prob_entropy(pred, actions, replay_legal)
        assert lp.dtype == torch.float32
        assert torch.equal(lp, torch.stack(lps, 1))
        assert torch.isfinite(ent).all()


def _masked_batch(cfg, model, B=3, T=4, seed=0, illegal=False):
    """Forward with converter-style legality (tile_of valid for the
    labelled type; the tgt copies are packed-bit floats) and teacher-forced
    types.  ``illegal``: one frame whose GT type, one whose GT tile, falls
    outside its mask."""
    g = torch.Generator().manual_seed(seed)
    tgt = _make_tgt(B, T, seed=seed)
    if illegal:
        tgt["action_type"][0, 1] = ACTION_RIICHI
    types = torch.rand(B, T, N_ACTION_TYPES, generator=g) < 0.4
    tile_of = torch.rand(B, T, N_TILE_KINDS, generator=g) < 0.3
    b, t = torch.meshgrid(torch.arange(B), torch.arange(T), indexing="ij")
    types[b, t, tgt["action_type"]] = True
    tile_of[b, t, tgt["tile_idx"]] = True
    if illegal:
        types[0, 0, tgt["action_type"][0, 0]] = False
        tile_of[0, 1, tgt["tile_idx"][0, 1]] = False
    tgt["legal_action_type"] = types.float()
    tgt["legal_tile_of"] = tile_of.float()
    legal = {"action_type": types, "tile_of": tile_of}
    pred = model(_make_obs(cfg, B, T, seed=seed), legal=legal,
                 labels={"action_type": tgt["action_type"]})
    return pred, tgt


def _reconstruct(terms):
    return sum(h.get("scale", 1.0) * (h["ce"] * h["weight"]).sum() / h["norm"]
               for h in terms.values())


def test_masked_il_loss_finite_and_zeroes_illegal_targets(cfg, model):
    model.train()
    B, T = 3, 4
    pred, tgt = _masked_batch(cfg, model, B, T, seed=1, illegal=True)
    assert torch.equal(pred["tile_mask"], tgt["legal_tile_of"].bool())  # labels committed
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[2, -1] = False
    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    assert out["illegal_target_frac"].item() == pytest.approx(2 / seq_mask.sum().item())
    out["loss"].backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    # masked argmax metrics: the argmax never lands on an illegal type
    with torch.no_grad():
        scored = _mask_logits(pred["action_type_logits"], pred["action_type_mask"])
        assert pred["action_type_mask"].gather(-1, scored.argmax(-1, keepdim=True)).all()
    # degenerate rows (all-False masks, e.g. padded frames) stay finite
    pred_d = {k: v.detach() for k, v in pred.items()}
    pred_d["action_type_mask"] = torch.zeros_like(pred_d["action_type_mask"])
    pred_d["tile_mask"] = torch.zeros_like(pred_d["tile_mask"])
    out = model.action_decoder.il_loss(pred_d, tgt, seq_mask)
    assert torch.isfinite(out["loss"]) and out["loss"].item() == 0.0


def test_il_loss_unmasked_bit_identical_to_reference(cfg, model):
    """No mask echoes → exactly the historical loss and stat keys."""
    B, T = 2, 4
    tgt = _make_tgt(B, T, seed=3)
    pred = model(_make_obs(cfg, B, T, seed=3), labels={"action_type": tgt["action_type"]})
    assert "action_type_mask" not in pred and "tile_mask" not in pred
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False
    out = model.action_decoder.il_loss(pred, tgt, seq_mask)

    seq_f = seq_mask.float()
    n_seq = seq_f.sum().clamp(min=1)
    uses = model.action_decoder.ARG_USES[tgt["action_type"]]
    w_tile = uses.float() * (pred["action_type_used"] == tgt["action_type"]).float() * seq_f
    n_tile = w_tile.sum().clamp(min=1)
    type_all = F.cross_entropy(pred["action_type_logits"].reshape(-1, N_ACTION_TYPES),
                               tgt["action_type"].reshape(-1), reduction="none"
                               ).reshape(B, T).nan_to_num(0.0)
    tile_all = F.cross_entropy(pred["tile_logits"].reshape(-1, N_TILE_KINDS),
                               tgt["tile_idx"].reshape(-1), reduction="none"
                               ).reshape(B, T).nan_to_num(0.0)
    ref = (type_all * seq_f).sum() / n_seq + (tile_all * w_tile).sum() / n_tile
    assert torch.equal(out["loss"], ref)
    assert set(out) == {"answer_correct", "answer_mask", "loss", "type_loss", "tile_loss",
                        "type_acc", "tile_acc"}


@pytest.mark.parametrize("masked", [False, True])
def test_il_head_terms_reconstruct_il_loss(cfg, model, masked):
    B, T = 3, 4
    if masked:
        pred, tgt = _masked_batch(cfg, model, B, T, seed=2, illegal=True)
    else:
        tgt = _make_tgt(B, T, seed=2)
        pred = model(_make_obs(cfg, B, T, seed=2), labels={"action_type": tgt["action_type"]})
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[0, -1] = False
    dec = model.action_decoder
    loss = dec.il_loss(pred, tgt, seq_mask)["loss"]
    terms = dec.il_head_terms(pred, tgt, seq_mask)
    assert set(terms) == {"action_type", "tile"}
    torch.testing.assert_close(_reconstruct(terms), loss, atol=1e-6, rtol=1e-6)
    for h in terms.values():
        assert h["kind"] == "categorical" and h["scale"] == 1.0
        assert h["logits"].requires_grad and not h["weight"].requires_grad
        assert h["weight"].shape == h["logits"].shape[:-1] == h["ce"].shape
        assert h["norm"].dim() == 0
        assert (h["mask"] is not None) == masked
        if masked:
            assert h["mask"].shape == h["logits"].shape and h["mask"].dtype == torch.bool
