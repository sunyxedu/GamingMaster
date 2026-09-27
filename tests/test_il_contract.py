"""The generic IL contract of the schema freeze (training/train_il_game.py).

- soft (teacher-distribution) targets through ``ActionDecoder.il_head_terms``:
  absent keys = the historical objective bit for bit; one-hot teachers = the
  hard labels; genuine teachers = soft cross-entropy (value + gradient);
  sparse == dense; Bernoulli heads; the additive (ce None) path; legality-
  masked terms; teacher KL / coverage stats; bf16;
- engine value targets (``engine_value`` / ``value_mask``), scalar and
  HL-Gauss heads;
- IL legality masks from the shards (``tgt["legal_<key>"]`` → ``legal``
  under ``--il_legal_mask``), through BatchCarry;
- the schema-freeze shard meta registry (META_ARCH_KEYS / META_LEGACY,
  schema_freeze_warning, ``--require_frozen_schema``);
- the per-head decomposition of every owned decoder's il_loss (pokemon,
  clash_royale, hearthstone, starcraft, generals factorized + joint, lux
  with the pointer sap head);
- end to end: train_multi_game with soft / engine / legal targets.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from data_processing.games.dataset import collate_game_sequences, save_shard
from model.game_network import GameNetwork
from model.games import build_game_network
from model.games.base import ActionDecoder, densify_soft_target, soft_cross_entropy
from model.games.board import BoardConfig, BoardEncoder, PlacementDecoder
from training.deep_supervision import BatchCarry, merge_rows
from training.train_il_game import (
    META_ARCH_KEYS,
    META_LEGACY,
    SCHEMA_FREEZE_CONVERTER_VERSION,
    apply_meta_arch_defaults,
    batch_loss,
    check_meta_arch,
    legal_from_tgt,
    logit_z_loss,
    mb_from_merged,
    prepare_batch,
    prepare_rows,
    schema_freeze_warning,
    value_losses,
)

CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=1, dropout=0.0, encoder_channels=16,
            core_type="transformer", core_layers=1)
N = 16          # 4x4 toy board
VW = 0.2        # value weight


# ---------------------------------------------------------------------------
# toy decoder: PlacementDecoder + an il_head_terms mirroring its il_loss
# ---------------------------------------------------------------------------

class ToyDecoder(PlacementDecoder):
    """``place`` (categorical, optional term mask) + ``pass`` (Bernoulli);
    ``additive`` drops the place ``ce`` (soft term then purely additive)."""

    additive = False
    term_mask = None

    def il_head_terms(self, pred, tgt, seq_mask):
        B, T, n = pred["place_logits"].shape
        valid = seq_mask.float()
        n_valid = valid.sum().clamp(min=1.0)
        tgt_pass = tgt["passed"].float()
        pass_bce = F.binary_cross_entropy_with_logits(
            pred["pass_logits"], tgt_pass, reduction="none").nan_to_num(0.0)
        move_w = valid * (1.0 - tgt_pass)
        n_moves = move_w.sum().clamp(min=1.0)
        tgt_place = tgt["place_idx"].long().clamp(0, n - 1)
        place_ce = F.cross_entropy(pred["place_logits"].reshape(-1, n), tgt_place.reshape(-1),
                                   reduction="none").reshape(B, T).nan_to_num(0.0)
        return {
            "place": {"logits": pred["place_logits"],
                      "ce": None if self.additive else place_ce,
                      "weight": move_w, "norm": n_moves, "mask": self.term_mask},
            "pass": {"logits": pred["pass_logits"], "kind": "bernoulli", "ce": pass_bce,
                     "weight": valid, "norm": n_valid},
        }


def _toy_net(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = BoardConfig(board_h=4, board_w=4, piece_channels=3, board_global_features=0,
                      n_promotions=0, **{**TINY, **kw})
    return GameNetwork(cfg, BoardEncoder(cfg), ToyDecoder(cfg))


def _toy_mb(B=3, T=5, seed=0, extra=None):
    """Collated toy batch (lengths T, T-1, ...); ``extra(i, L, g)`` adds tgt keys."""
    g = torch.Generator().manual_seed(seed)
    samples = []
    for i in range(B):
        L = T - i
        tgt = {"place_idx": torch.randint(0, N, (L,), generator=g),
               "passed": torch.rand(L, generator=g) < 0.25,
               "value": torch.rand(L, generator=g) * 2 - 1}
        if extra is not None:
            tgt.update(extra(i, L, g))
        samples.append({"obs": {"board_seq": torch.rand(L, 3, 4, 4, generator=g)},
                        "tgt": tgt, "meta": {}})
    return prepare_batch("othello", collate_game_sequences(samples), CPU, 1.0)


def _with(mb, **tgt):
    return {**mb, "tgt": {**mb["tgt"], **tgt}}


def _loss(net, mb, **kw):
    return batch_loss(net, mb, CPU, VW, 0.5, kw.pop("bf16", False), **kw)


def _forward(net, mb):
    return net(mb["obs"], legal=mb["legal"], labels=mb["labels"],
               frame_mask=mb["frame_mask"], return_carry=True)


def _head_sums(net, mb):
    """(pred, il_loss, value loss, per-head (w, norm, hard ce, logits)) of one forward."""
    pred = _forward(net, mb)
    il = net.action_decoder.il_loss(pred, mb["tgt"], mb["frame_mask"])
    v_loss, _ = value_losses(net, pred, mb["tgt"]["value"], mb["frame_mask"])
    terms = net.action_decoder.il_head_terms(pred, mb["tgt"], mb["frame_mask"])
    return pred, il, v_loss, terms


def _soft_ce(logits, p):
    return -(p * F.log_softmax(logits.float(), -1)).sum(-1)


def _bce(logits, p):
    return F.binary_cross_entropy_with_logits(logits.float(), p, reduction="none")


def _dirichlet(shape, g):
    x = torch.rand(*shape, generator=g) + 1e-3
    return x / x.sum(-1, keepdim=True)


# ---------------------------------------------------------------------------
# absent keys: the historical objective, bit for bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("zw", [0.0, 1e-4])
def test_absent_keys_reproduce_reference_objective(zw):
    torch.manual_seed(0)
    net = build_game_network("othello", board_h=4, board_w=4, **TINY)
    mb = _toy_mb()
    loss, stats, _ = _loss(net, mb, z_loss_weight=zw)
    pred = _forward(net, mb)
    il = net.action_decoder.il_loss(pred, mb["tgt"], mb["frame_mask"])
    z, _ = logit_z_loss(pred, mb["frame_mask"])
    v_loss, _ = value_losses(net, pred, mb["tgt"]["value"], mb["frame_mask"])
    ref = il["loss"]
    if zw > 0:
        ref = ref + zw * z
    ref = ref + VW * v_loss
    assert torch.equal(loss, ref)
    expect = ({k for k in il if k not in ("loss", "answer_correct", "answer_mask")}
              | {"loss", "logit_lse_max", "value_loss", "value_mse", "exact_acc"}
              | ({"z_loss"} if zw > 0 else set()))
    assert set(stats) == expect


# ---------------------------------------------------------------------------
# soft targets
# ---------------------------------------------------------------------------

def test_one_hot_soft_targets_equal_hard_labels():
    net = _toy_net()
    mb = _toy_mb()
    base, base_stats, _ = _loss(net, mb)
    move = (~mb["tgt"]["passed"]).float().unsqueeze(-1)
    soft = _with(mb, place_probs=F.one_hot(mb["tgt"]["place_idx"], N).float() * move,
                 pass_probs=mb["tgt"]["passed"].float())
    loss, stats, _ = _loss(net, soft, soft_target_weight=1.0)
    assert abs(float(loss) - float(base)) < 1e-6
    assert stats["place_teacher_frac"] == pytest.approx(1.0)
    assert stats["pass_teacher_frac"] == pytest.approx(1.0)
    assert stats["pass_teacher_kl"] >= -1e-6
    assert set(stats) - set(base_stats) == {"place_teacher_kl", "place_teacher_frac",
                                            "pass_teacher_kl", "pass_teacher_frac"}


def test_genuine_teacher_replaces_the_hard_terms_value_and_gradient():
    g = torch.Generator().manual_seed(1)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    p_place = _dirichlet((B, T, N), g)
    p_pass = torch.rand(B, T, generator=g)
    soft = _with(mb, place_probs=p_place, pass_probs=p_pass)

    net = _toy_net()
    loss, stats, _ = _loss(net, soft)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in net.named_parameters() if p.grad is not None}

    net2 = _toy_net()
    pred, il, v_loss, terms = _head_sums(net2, soft)
    t_pl, t_pa = terms["place"], terms["pass"]
    manual = ((_soft_ce(pred["place_logits"], p_place) * t_pl["weight"]).sum() / t_pl["norm"]
              + (_bce(pred["pass_logits"], p_pass) * t_pa["weight"]).sum() / t_pa["norm"]
              + VW * v_loss)
    assert float(loss) == pytest.approx(float(manual), abs=1e-5)
    manual.backward()
    for n, p in net2.named_parameters():
        if n in grads:
            assert torch.allclose(grads[n], p.grad, atol=1e-6, rtol=1e-4), n
    # KL(p‖q) ≥ 0, and the stat is the teacher-weighted mean of it
    ent = -(torch.special.xlogy(p_place, p_place)).sum(-1)
    kl = ((_soft_ce(pred["place_logits"], p_place) - ent) * t_pl["weight"]).sum() / (
        t_pl["weight"].sum())
    assert stats["place_teacher_kl"] == pytest.approx(float(kl), rel=1e-4)
    assert stats["place_teacher_kl"] > 0 and stats["pass_teacher_kl"] > 0


def test_zero_rows_keep_the_hard_label_and_alpha_blends():
    g = torch.Generator().manual_seed(2)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    teach = torch.rand(B, T, generator=g) < 0.5
    p = _dirichlet((B, T, N), g) * teach.unsqueeze(-1)
    net = _toy_net()
    pred, il, v_loss, terms = _head_sums(net, mb)
    t = terms["place"]
    w = t["weight"] * teach.float()
    for alpha in (1.0, 0.25, 0.0):
        loss, stats, _ = _loss(net, _with(mb, place_probs=p), soft_target_weight=alpha)
        manual = (il["loss"] + alpha * ((_soft_ce(pred["place_logits"], p) - t["ce"]) * w).sum()
                  / t["norm"] + VW * v_loss)
        assert float(loss) == pytest.approx(float(manual), abs=1e-5), alpha
        assert stats["place_teacher_frac"] == pytest.approx(
            float(w.sum() / t["weight"].sum()), abs=1e-6)
    assert "place_teacher_kl" in stats                 # α = 0 still reports


def test_sparse_equals_dense_and_densify_contract():
    g = torch.Generator().manual_seed(3)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    K = 3
    idx = torch.randint(0, N, (B, T, K), generator=g)
    idx[..., 2] = idx[..., 0]                           # repeated class: mass adds up
    idx[0, 0, 1] = -1                                   # pad
    val = torch.rand(B, T, K, generator=g)
    dense = densify_soft_target(idx, val, N)
    ref = torch.zeros(B, T, N)
    for b in range(B):
        for t in range(T):
            for k in range(K):
                if idx[b, t, k] >= 0:
                    ref[b, t, idx[b, t, k]] += val[b, t, k]
    assert torch.allclose(dense, ref)
    net = _toy_net()
    l_d, s_d, _ = _loss(net, _with(mb, place_probs=dense))
    l_s, s_s, _ = _loss(net, _with(mb, place_probs_idx=idx, place_probs_val=val))
    assert float(l_d) == pytest.approx(float(l_s), abs=1e-6)
    assert s_d["place_teacher_kl"] == pytest.approx(s_s["place_teacher_kl"], abs=1e-6)
    with pytest.raises(ValueError, match="outside"):
        densify_soft_target(torch.tensor([[N]]), torch.ones(1, 1), N)
    with pytest.raises(ValueError, match="outside"):
        densify_soft_target(torch.tensor([[-2]]), torch.ones(1, 1), N)


def test_bernoulli_only_and_additive_path():
    g = torch.Generator().manual_seed(4)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    p_pass = torch.rand(B, T, generator=g)
    net = _toy_net()
    pred, il, v_loss, terms = _head_sums(net, mb)
    t = terms["pass"]
    loss, _, _ = _loss(net, _with(mb, pass_probs=p_pass))
    manual = (il["loss"] + ((_bce(pred["pass_logits"], p_pass) - t["ce"]) * t["weight"]).sum()
              / t["norm"] + VW * v_loss)
    assert float(loss) == pytest.approx(float(manual), abs=1e-5)

    net.action_decoder.additive = True
    p = _dirichlet((B, T, N), g)
    t = terms["place"]
    loss, _, _ = _loss(net, _with(mb, place_probs=p), soft_target_weight=0.5)
    manual = (il["loss"] + 0.5 * (_soft_ce(pred["place_logits"], p) * t["weight"]).sum()
              / t["norm"] + VW * v_loss)
    assert float(loss) == pytest.approx(float(manual), abs=1e-5)


def test_malformed_soft_targets_are_rejected():
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    net = _toy_net()
    with pytest.raises(ValueError, match=r"ToyDecoder.*\['pass', 'place'\]"):
        _loss(net, _with(mb, foo_probs=torch.rand(B, T, 4)))
    with pytest.raises(ValueError, match="shape"):
        _loss(net, _with(mb, place_probs=torch.rand(B, T, N + 1)))
    with pytest.raises(ValueError, match="negative"):
        _loss(net, _with(mb, place_probs=-torch.rand(B, T, N)))
    with pytest.raises(ValueError, match="> 1"):
        _loss(net, _with(mb, pass_probs=torch.full((B, T), 1.5)))
    with pytest.raises(ValueError, match="without"):
        _loss(net, _with(mb, place_probs_idx=torch.zeros(B, T, 2, dtype=torch.long)))
    # legal_* keys are never soft targets; the base contract exposes no heads
    loss, stats, _ = _loss(net, _with(mb, legal_place_probs=torch.ones(B, T, N)))
    assert not any("teacher" in k for k in stats)
    pred = _forward(net, mb)
    assert ActionDecoder.il_head_terms(net.action_decoder, pred, mb["tgt"],
                                       mb["frame_mask"]) == {}


def test_teacher_kl_vanishes_at_the_model_distribution():
    net = _toy_net()
    mb = _toy_mb()
    with torch.no_grad():
        pred = _forward(net, mb)
    soft = _with(mb, place_probs=F.softmax(pred["place_logits"].float(), -1),
                 pass_probs=torch.sigmoid(pred["pass_logits"].float()))
    _, stats, _ = _loss(net, soft)
    assert abs(stats["place_teacher_kl"]) < 1e-5
    assert abs(stats["pass_teacher_kl"]) < 1e-5


def test_masked_term_drops_and_reports_illegal_teacher_mass():
    g = torch.Generator().manual_seed(5)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    tmask = torch.rand(B, T, N, generator=g) < 0.5
    tmask[..., 0] = True
    p = torch.full((B, T, N), 1.0 / N)
    # a teacher row with all its mass on illegal moves keeps the hard label
    tmask[0, 0] = False
    tmask[0, 0, 0] = True
    p[0, 0] = 0.0
    p[0, 0, 1] = 1.0
    net = _toy_net()
    net.action_decoder.term_mask = tmask
    pred, il, v_loss, terms = _head_sums(net, mb)
    t = terms["place"]
    loss, stats, _ = _loss(net, _with(mb, place_probs=p))
    legal_p = p * tmask
    has = legal_p.sum(-1) > 0
    legal_p = legal_p / legal_p.sum(-1, keepdim=True).clamp_min(1e-12)
    masked = pred["place_logits"].float().masked_fill(~tmask, float("-inf"))
    soft = -(legal_p * F.log_softmax(masked, -1).clamp_min(-1e4)).sum(-1)
    w = t["weight"] * has.float()
    manual = il["loss"] + ((soft - t["ce"]) * w).sum() / t["norm"] + VW * v_loss
    assert float(loss) == pytest.approx(float(manual), abs=1e-5)
    rows = (t["weight"] > 0)
    illegal = (p * ~tmask).sum(-1)
    assert stats["place_teacher_illegal_mass"] == pytest.approx(
        float((illegal * rows).sum() / rows.sum()), abs=1e-6)
    # soft_cross_entropy: masked entries never produce NaN in the backward
    x = torch.randn(2, N, requires_grad=True)
    m = torch.zeros(2, N, dtype=torch.bool)
    m[0, :3] = True                                     # row 1 fully masked: index-0 guard
    ce, ent = soft_cross_entropy(x, torch.full((2, N), 1.0 / N) * m, m)
    ce.sum().backward()
    assert torch.isfinite(ce).all() and torch.isfinite(x.grad).all()
    assert ent[1] == 0


def test_soft_targets_under_bf16_autocast():
    g = torch.Generator().manual_seed(6)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    net = _toy_net()
    loss, stats, _ = _loss(net, _with(mb, place_probs=_dirichlet((B, T, N), g),
                                      pass_probs=torch.rand(B, T, generator=g)), bf16=True)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    assert all(v == v for v in stats.values())          # no NaN stat


# ---------------------------------------------------------------------------
# engine values
# ---------------------------------------------------------------------------

def _value_manual(pred, target, m):
    m = m.float()
    return (((pred["value"].float() - target) ** 2) * m).sum() / m.sum().clamp(min=1)


def test_engine_value_blend_and_masks():
    g = torch.Generator().manual_seed(7)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    fm = mb["frame_mask"]
    eng = torch.rand(B, T, generator=g) * 2 - 1
    val = mb["tgt"]["value"]
    net = _toy_net()
    pred, il, _, _ = _head_sums(net, mb)

    beta = 0.3
    loss, stats, _ = _loss(net, _with(mb, engine_value=eng), engine_value_weight=beta)
    target = beta * eng + (1 - beta) * val
    v = _value_manual(pred, target, fm)
    assert stats["value_loss"] == pytest.approx(float(v), abs=1e-6)
    assert float(loss) == pytest.approx(float(il["loss"] + VW * v), abs=1e-6)
    assert stats["engine_value_frac"] == pytest.approx(1.0)
    assert stats["engine_value_mse"] == pytest.approx(float(_value_manual(pred, eng, fm)),
                                                      abs=1e-6)
    sign_ok = (torch.sign(pred["value"]) == torch.sign(eng)).float() * fm
    assert stats["engine_value_sign_acc"] == pytest.approx(float(sign_ok.sum() / fm.sum()),
                                                           abs=1e-6)

    # outcome invalid → engine only; no engine eval → outcome only; neither → dropped
    val_m = torch.rand(B, T, generator=g) < 0.6
    eng_m = torch.rand(B, T, generator=g) < 0.6
    eng_nan = torch.where(eng_m, eng, torch.full_like(eng, float("nan")))
    _, stats, _ = _loss(net, _with(mb, engine_value=eng_nan, engine_value_mask=eng_m,
                                   value_mask=val_m), engine_value_weight=beta)
    target = torch.where(eng_m & val_m, beta * eng + (1 - beta) * val,
                         torch.where(eng_m, eng, val))
    sup = fm & (eng_m | val_m)
    assert stats["value_loss"] == pytest.approx(
        float(_value_manual(pred, torch.where(sup, target, 0.0), sup)), abs=1e-6)
    assert stats["engine_value_frac"] == pytest.approx(
        float((fm & eng_m).sum() / fm.sum()), abs=1e-6)
    assert all(v == v for v in stats.values())

    # value_mask alone: outcome loss on the valid frames, no engine stats
    _, stats, _ = _loss(net, _with(mb, value_mask=val_m))
    assert stats["value_loss"] == pytest.approx(float(_value_manual(pred, val, fm & val_m)),
                                                abs=1e-6)
    assert not any(k.startswith("engine_value") for k in stats)


def test_engine_value_hl_gauss_head():
    g = torch.Generator().manual_seed(8)
    mb = _toy_mb()
    B, T = mb["frame_mask"].shape
    eng = torch.rand(B, T, generator=g) * 2 - 1
    net = _toy_net(hl_gauss_value=True, hl_gauss_bins=16)
    pred = _forward(net, mb)
    beta = 0.5
    target = beta * eng + (1 - beta) * mb["tgt"]["value"]
    ref, _ = value_losses(net, pred, target, mb["frame_mask"])
    _, stats, _ = _loss(net, _with(mb, engine_value=eng), engine_value_weight=beta)
    assert stats["value_loss"] == pytest.approx(float(ref), abs=1e-6)
    # the HL-Gauss support check applies to the blended target
    with pytest.raises(ValueError, match="HL-Gauss support"):
        _loss(net, _with(mb, engine_value=torch.full((B, T), 5.0)), engine_value_weight=1.0)


# ---------------------------------------------------------------------------
# IL legality masks
# ---------------------------------------------------------------------------

def _lux_batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    samples = []
    for L in (4, 3):
        samples.append({
            "obs": {"unit_mask": torch.rand(L, 5, generator=g) < 0.7},
            "tgt": {"value": torch.zeros(L),
                    # packbits storage decodes to float {0, 1}
                    "legal_sap": (torch.rand(L, 5, 9, generator=g) < 0.5).float(),
                    "legal_move": torch.rand(L, 5, 6, generator=g) < 0.5},
            "meta": {}})
    return collate_game_sequences(samples)


def test_legal_from_tgt_merges_with_legal_from_obs():
    batch = _lux_batch()
    assert set(legal_from_tgt(batch["tgt"])) == {"sap", "move"}
    off = prepare_batch("lux", batch, CPU, 1.0)
    assert set(off["legal"]) == {"unit"}
    on = prepare_batch("lux", batch, CPU, 1.0, il_legal_mask=True)
    assert set(on["legal"]) == {"unit", "sap", "move"}
    assert on["legal"]["sap"].dtype == torch.bool
    assert torch.equal(on["legal"]["sap"], batch["tgt"]["legal_sap"].bool())
    assert torch.equal(on["legal"]["unit"], off["legal"]["unit"])
    assert "legal_sap" in on["tgt"]                     # stays in tgt
    # a game without LEGAL_FROM_OBS: None, or just the shard masks
    mb = _toy_mb(extra=lambda i, L, g: {"legal_place": torch.ones(L, N, dtype=torch.bool)})
    assert mb["legal"] is None
    raw = collate_game_sequences([{"obs": {"board_seq": torch.rand(3, 3, 4, 4)},
                                   "tgt": {"legal_place": torch.ones(3, N)}, "meta": {}}])
    assert set(prepare_batch("othello", raw, CPU, 1.0, il_legal_mask=True)["legal"]) == {
        "place"}


def test_legal_masks_survive_batch_carry():
    batch = _lux_batch()
    rows = prepare_rows("lux", batch, CPU, 1.0, il_legal_mask=True)
    merged = merge_rows(rows)
    mb = mb_from_merged("lux", merged, il_legal_mask=True)
    fm = batch["meta"]["frame_mask"]
    assert torch.equal(mb["legal"]["sap"][fm], batch["tgt"]["legal_sap"].bool()[fm])
    assert not mb["legal"]["sap"][~fm].any()            # padding: nothing legal
    assert set(mb_from_merged("lux", merged)["legal"]) == {"unit"}

    seen = []

    def fb(m, carry, active, reset, weight):
        seen.append(mb_from_merged("lux", m, il_legal_mask=True))
        return {"q_halt": torch.zeros(active.shape)}, {}

    BatchCarry(1, 0.0).step(rows, fb)
    assert torch.equal(seen[0]["legal"]["move"][fm], batch["tgt"]["legal_move"][fm])


# ---------------------------------------------------------------------------
# schema-freeze shard meta
# ---------------------------------------------------------------------------

def test_meta_registry_frozen_schema_keys():
    assert META_ARCH_KEYS["go"] == {"global_features": ("board_global_features",),
                                    "board_size": ("board_h", "board_w")}
    assert META_ARCH_KEYS["chess"] == {"global_features": ("board_global_features",)}
    assert META_ARCH_KEYS["shogi"] == {"global_features": ("shogi_global_features",)}
    assert META_ARCH_KEYS["lux"] == {"global_features": ("lux_global_features",),
                                     "sap_range": ("lux_sap_range",)}
    assert META_ARCH_KEYS["pokemon"] == {"move_buckets": ("pokemon_move_buckets",),
                                         "species_buckets": ("pokemon_species_buckets",),
                                         "field_features": ("pokemon_field_features",),
                                         "move_vocab": ("pokemon_move_vocab",),
                                         "species_vocab": ("pokemon_species_vocab",),
                                         "item_vocab": ("pokemon_item_vocab",),
                                         "ability_vocab": ("pokemon_ability_vocab",),
                                         "type_vocab": ("pokemon_type_vocab",),
                                         "poke_features": ("pokemon_feature_dim",),
                                         "format_vocab": ("pokemon_format_vocab",)}


#: games whose pre-freeze shards must NOT train a fresh default net: clash_royale
#: converter_version <= 2 inferred the hand from the whole match (a future leak),
#: pokemon converter v1 / v2 hashed the move / species names into colliding buckets,
#: so the default net refuses them (the legacy net still accepts them)
REFUSE_PRE_FREEZE = {"clash_royale": {"cr_card_tokens": "hand", "card_feature_dim": 8,
                                      "cr_arena_patch": 1, "cr_recent_tokens": 0,
                                      "cr_opp_tokens": 0},
                     "pokemon": {"pokemon_move_vocab": 0, "pokemon_species_vocab": 0}}


def test_meta_legacy_values_are_the_registry_defaults():
    """A fresh run on pre-freeze shards (no meta key) builds the legacy
    architecture without anything being injected — except REFUSE_PRE_FREEZE
    games, where it fails loudly and only the legacy config passes."""
    for game, legacy in META_LEGACY.items():
        cfg = build_game_network(game, **TINY).cfg
        if game in REFUSE_PRE_FREEZE:
            with pytest.raises(ValueError, match="implied"):
                check_meta_arch(game, cfg, {})
            cfg = build_game_network(game, **TINY, **REFUSE_PRE_FREEZE[game]).cfg
        for meta_key, want in legacy.items():
            for ck in META_ARCH_KEYS[game][meta_key]:
                if hasattr(cfg, ck):
                    assert getattr(cfg, ck) == want, (game, ck)
        check_meta_arch(game, cfg, {})


def test_chess_meta_defaults_and_checks():
    kw = apply_meta_arch_defaults("chess", dict(TINY), {"global_features": 10})
    assert kw["board_global_features"] == 10
    net = build_game_network("chess", **kw)
    assert net.cfg.board_global_features == 10
    check_meta_arch("chess", net.cfg, {"global_features": 10})
    assert apply_meta_arch_defaults("chess", {}, {}) == {}          # legacy: never injected
    check_meta_arch("chess", SimpleNamespace(board_global_features=8), {})
    with pytest.raises(ValueError, match="implied"):
        check_meta_arch("chess", net.cfg, {})                       # new net, old shards
    with pytest.raises(ValueError, match="board_global_features=8"):
        check_meta_arch("chess", SimpleNamespace(board_global_features=8),
                        {"global_features": 10})                    # old net, new shards


def test_go_and_lux_meta():
    kw = apply_meta_arch_defaults("go", {}, {"board_size": 9, "global_features": 6})
    assert kw == {"board_h": 9, "board_w": 9, "board_global_features": 6}
    check_meta_arch("go", SimpleNamespace(board_h=9, board_w=9, board_global_features=6),
                    {"board_size": 9, "global_features": 6})
    check_meta_arch("go", SimpleNamespace(board_h=19, board_w=19, board_global_features=0), {})
    with pytest.raises(ValueError, match="board_h=19"):
        check_meta_arch("go", SimpleNamespace(board_h=19, board_w=19, board_global_features=6),
                        {"board_size": 9, "global_features": 6})
    # a config class without lux_sap_range: that field is skipped, not an error
    old = SimpleNamespace(lux_global_features=6)
    check_meta_arch("lux", old, {})
    check_meta_arch("lux", SimpleNamespace(lux_global_features=11),
                    {"global_features": 11, "sap_range": 7})
    with pytest.raises(ValueError, match="lux_global_features=6"):
        check_meta_arch("lux", old, {"global_features": 11, "sap_range": 7})
    with pytest.raises(ValueError, match="lux_sap_range=0"):
        check_meta_arch("lux", SimpleNamespace(lux_global_features=11, lux_sap_range=0),
                        {"global_features": 11, "sap_range": 7})
    with pytest.raises(ValueError, match="sap_range=0 .implied"):
        check_meta_arch("lux", SimpleNamespace(lux_global_features=6, lux_sap_range=7), {})


def test_pokemon_meta():
    """converter v3 shards (exact Showdown vocabularies: move_vocab / species_vocab sizes, 22
    field features) build the exact tables in fresh runs.  Hashed-bucket shards (v2: 22 field
    features, v1: 20 — neither has a vocab key, i.e. implied 0) refuse a fresh default net
    unless the legacy hashed net is asked for explicitly; hashed checkpoints refuse v3."""
    from data_processing.games.pokemon import MOVE_VOCAB_SIZE, SPECIES_VOCAB_SIZE, VOCAB_VERSION

    v3 = {"field_features": 22, "move_vocab": 900, "species_vocab": 1400,
          "vocab_version": VOCAB_VERSION}
    kw = apply_meta_arch_defaults("pokemon", dict(TINY), v3)
    assert (kw["pokemon_field_features"], kw["pokemon_move_vocab"],
            kw["pokemon_species_vocab"]) == (22, 900, 1400)
    net = build_game_network("pokemon", **kw)
    assert net.action_decoder.move_embed.num_embeddings == 900
    assert net.obs_encoder.species_embed.num_embeddings == 1400
    check_meta_arch("pokemon", net.cfg, v3)
    fresh = build_game_network("pokemon", **TINY,                    # the vendored vocabulary
                               pokemon_field_features=22).cfg
    assert (fresh.pokemon_move_vocab, fresh.pokemon_species_vocab) == (MOVE_VOCAB_SIZE,
                                                                       SPECIES_VOCAB_SIZE)
    with pytest.raises(ValueError, match="move_vocab=900 but the model has "
                                         f"pokemon_move_vocab={MOVE_VOCAB_SIZE}"):
        check_meta_arch("pokemon", fresh, v3)          # another vocabulary version
    hashed = dict(pokemon_move_vocab=0, pokemon_species_vocab=0)
    with pytest.raises(ValueError, match="pokemon_move_vocab=0.*fresh pokemon adapter"):
        check_meta_arch("pokemon", build_game_network("pokemon", **TINY, **hashed,
                                                      pokemon_field_features=22).cfg, v3)
    # v2 (hashed) shards: the fresh default net refuses them, naming the way out ...
    v2 = {"field_features": 22, "move_buckets": 1024, "species_buckets": 512}
    kw = apply_meta_arch_defaults("pokemon", dict(TINY), v2)
    assert "pokemon_move_vocab" not in kw
    with pytest.raises(ValueError, match="move_vocab=0 .implied.*--override pokemon_move_vocab=0"):
        check_meta_arch("pokemon", build_game_network("pokemon", **kw).cfg, v2)
    # ... and the explicit legacy net trains on them (buckets from the meta)
    kw = apply_meta_arch_defaults("pokemon", {**TINY, **hashed}, v2)
    old = build_game_network("pokemon", **kw)
    assert (kw["pokemon_field_features"], kw["pokemon_move_buckets"]) == (22, 1024)
    assert old.action_decoder.move_embed.num_embeddings == 1024
    check_meta_arch("pokemon", old.cfg, v2)
    with pytest.raises(ValueError, match="pokemon_field_features=20"):
        check_meta_arch("pokemon", SimpleNamespace(pokemon_field_features=20,
                                                   pokemon_move_buckets=1024,
                                                   pokemon_species_buckets=512), v2)
    with pytest.raises(ValueError, match="field_features=20 .implied"):
        check_meta_arch("pokemon", old.cfg, {"move_buckets": 1024, "species_buckets": 512})
    # the move head: v2 / v3 shards take the config default (identity); v1 shards (no
    # field_features, no tgt.move_id) keep the legacy slot head in fresh runs — the identity
    # head cannot train on them — unless the command line says otherwise
    assert net.cfg.pokemon_move_head == "identity" and "pokemon_move_head" not in kw
    v1 = {"move_buckets": 256, "species_buckets": 512}
    kw = apply_meta_arch_defaults("pokemon", {**TINY, **hashed}, v1)
    old = build_game_network("pokemon", **kw).cfg
    assert (old.pokemon_move_head, old.pokemon_field_features) == ("slot", 20)
    check_meta_arch("pokemon", old, v1)
    with pytest.raises(ValueError, match="implied"):
        check_meta_arch("pokemon", build_game_network(
            "pokemon", **apply_meta_arch_defaults("pokemon", dict(TINY), v1)).cfg, v1)
    explicit = apply_meta_arch_defaults("pokemon", {**TINY, "pokemon_move_head": "identity"}, v1)
    assert explicit["pokemon_move_head"] == "identity"


def test_schema_freeze_warning():
    assert SCHEMA_FREEZE_CONVERTER_VERSION == {"chess": 3, "go": 2, "shogi": 2,
                                               "mahjong": 4, "lux": 3, "clash_royale": 3,
                                               "pokemon": 5}
    msg = schema_freeze_warning("lux", SimpleNamespace(converter_version=2))
    assert msg.startswith("lux shards have converter_version 2 < 3") and "one sample per game" in msg
    assert "move-identity" in schema_freeze_warning("pokemon",
                                                    SimpleNamespace(converter_version=1))
    msg = schema_freeze_warning("pokemon", SimpleNamespace(converter_version=2))
    assert msg.startswith("pokemon shards have converter_version 2 < 5") and "exact" in msg
    msg = schema_freeze_warning("pokemon", SimpleNamespace(converter_version=3))
    assert msg.startswith("pokemon shards have converter_version 3 < 5") and "revealed" in msg
    msg = schema_freeze_warning("pokemon", SimpleNamespace(converter_version=4))
    assert msg.startswith("pokemon shards have converter_version 4 < 5") and "format" in msg
    assert schema_freeze_warning("pokemon", SimpleNamespace(converter_version=5)) is None
    msg = schema_freeze_warning("chess", SimpleNamespace(converter_version=2))
    assert msg.startswith("chess shards have converter_version 2 < 3: they predate the "
                          "frozen pre-training schema")
    assert "repetition flags + legal masks" in msg and "re-convert" in msg
    assert "komi" in schema_freeze_warning("go", SimpleNamespace(converter_version=None))
    msg = schema_freeze_warning("mahjong", SimpleNamespace(converter_version=2))
    assert msg.startswith("mahjong shards have converter_version 2 < 4") and "il_weight" in msg
    msg = schema_freeze_warning("mahjong", SimpleNamespace(converter_version=3))
    assert msg.startswith("mahjong shards have converter_version 3 < 4") and "discard history" in msg
    assert schema_freeze_warning("mahjong", SimpleNamespace(converter_version=4)) is None
    assert schema_freeze_warning("chess", SimpleNamespace(converter_version=3)) is None
    assert schema_freeze_warning("othello", SimpleNamespace(converter_version=1)) is None
    assert schema_freeze_warning("generals", SimpleNamespace(converter_version=1)) is None


def _chess_samples(n=3, T=5):
    return [{"obs": {"board_seq": torch.rand(T, 14, 8, 8), "globals_seq": torch.rand(T, 8)},
             "tgt": {"from_idx": torch.randint(0, 64, (T,)), "to_idx": torch.randint(0, 64, (T,)),
                     "promotion": torch.zeros(T, dtype=torch.long),
                     "value": torch.rand(T) * 2 - 1},
             "meta": {}} for _ in range(n)]


def test_require_frozen_schema_exits(tmp_path, monkeypatch):
    from training import train_il_game, train_multi_game

    save_shard(str(tmp_path / "chess" / "a.pt"), "chess", _chess_samples(), converter_version=2)
    with pytest.raises(SystemExit, match="require_frozen_schema: chess shards"):
        train_multi_game.main(["--data", f"chess={tmp_path / 'chess'}",
                               "--require_frozen_schema", "--device", "cpu",
                               "--save_dir", str(tmp_path / "ck")])
    monkeypatch.setattr(sys, "argv", ["train_il_game", "--game", "chess", "--data_dir",
                                      str(tmp_path / "chess"), "--require_frozen_schema",
                                      "--save_dir", str(tmp_path / "ck2")])
    with pytest.raises(SystemExit, match="require_frozen_schema: chess shards"):
        train_il_game.main()


# ---------------------------------------------------------------------------
# per-head decomposition of the owned decoders
# ---------------------------------------------------------------------------

def _small(cls, **kw):
    return cls(**TINY, **kw)


def _decoder_cases():
    """(name, net, pred, tgt, mask) per owned decoder, fixed seeds, teacher
    forcing partly dropped (−1 labels) so condition weights are non-trivial."""
    from model.config import ModelConfig
    from model.games.clash_royale import (
        CardPlacementDecoder,
        ClashRoyaleConfig,
        ClashRoyaleEncoder,
    )
    from model.games.generals import legal_from_obs, source_labels
    from model.games.hearthstone import (
        EntityActionDecoder,
        HearthstoneConfig,
        HearthstoneEncoder,
    )
    from model.games.lux import LuxConfig, LuxDecoder, LuxEncoder, lux_sap_legal
    from model.games.pokemon import PokemonConfig, PokemonDecoder, PokemonEncoder, n_move_ids
    from model.games.starcraft import RTSDecoder, StarCraftConfig, StarCraftEncoder
    from model.network import GeneralsNetwork

    B, T = 3, 5
    g = torch.Generator().manual_seed(0)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 3:] = False

    def drop(x):
        return torch.where(torch.rand(B, T, generator=g) < 0.6, x, torch.full_like(x, -1))

    out = []
    torch.manual_seed(0)
    cfg = _small(PokemonConfig, pokemon_species_buckets=64)
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))
    S = 2 * cfg.team_slots
    obs = {"species_ids": torch.randint(1, 64, (B, T, S), generator=g),
           "poke_feats": torch.randn(B, T, S, cfg.pokemon_feature_dim, generator=g),
           "slot_mask": torch.rand(B, T, S, generator=g) < 0.7,
           "field_seq": torch.randn(B, T, cfg.pokemon_field_features, generator=g)}
    # converter v2 targets: move identity (−1 = none), unlabelled frames
    tgt = {"action_type": torch.randint(0, 2, (B, T), generator=g),
           "move_idx": torch.randint(-1, 4, (B, T), generator=g),
           "move_id": torch.randint(-1, n_move_ids(cfg), (B, T), generator=g),
           "switch_idx": torch.randint(0, 6, (B, T), generator=g),
           "tera": torch.randint(0, 2, (B, T), generator=g),
           "active_idx": torch.randint(0, 6, (B, T), generator=g),
           "action_known": torch.rand(B, T, generator=g) < 0.8}
    tgt["action_type"][0, 1] = -1
    pk_obs, pk_tgt = obs, tgt
    pred = net(obs, labels={"action_type": drop(tgt["action_type"]),
                            "active_idx": tgt["active_idx"]})
    out.append(("pokemon", net, pred, tgt, mask))

    # the CR patch decoder (cr_arena_patch 2) with the play trackers, IL under the masks
    torch.manual_seed(0)
    cfg = _small(ClashRoyaleConfig, arena_h=8, arena_w=6, n_cards=20, cr_arena_patch=2)
    net = GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))
    D = cfg.deck_size
    cr_obs = {"arena_seq": torch.randn(B, T, cfg.arena_channels, 8, 6, generator=g),
              "deck_ids": torch.randint(0, 20, (B, T, D), generator=g),
              "deck_feats": torch.randn(B, T, D, cfg.card_feature_dim, generator=g),
              "deck_mask": torch.rand(B, T, D, generator=g) < 0.8,
              "globals_seq": torch.randn(B, T, cfg.cr_global_features, generator=g),
              "recent_ids": torch.randint(1, 20, (B, T, 8), generator=g),
              "recent_feats": torch.rand(B, T, 8, 7, generator=g),
              "recent_mask": torch.rand(B, T, 8, generator=g) < 0.7,
              "recent_cell": torch.randint(-1, 48, (B, T, 8), generator=g),
              "opp_ids": torch.randint(1, 20, (B, T, 8), generator=g),
              "opp_feats": torch.rand(B, T, 8, 10, generator=g),
              "opp_mask": torch.rand(B, T, 8, generator=g) < 0.7}
    cr_tgt = {"waited": torch.rand(B, T, generator=g) < 0.4,
              "card_slot": torch.randint(0, D, (B, T), generator=g),
              "place_idx": torch.randint(0, 48, (B, T), generator=g)}
    card_ok = torch.rand(B, T, D, generator=g) < 0.7
    card_ok[..., 0] = True
    cells_ok = torch.rand(B, T, 48, generator=g) < 0.6
    cells_ok[..., 0] = True
    legal = {"card": card_ok, "placement": cells_ok}
    pred = net(cr_obs, legal=legal, labels={"card_slot": drop(cr_tgt["card_slot"]),
                                            "place_idx": drop(cr_tgt["place_idx"])})
    out.append(("clash_royale", net, pred, cr_tgt, mask))

    torch.manual_seed(0)
    cfg = _small(HearthstoneConfig)
    net = GameNetwork(cfg, HearthstoneEncoder(cfg), EntityActionDecoder(cfg))
    S = cfg.hand_slots + 2 * cfg.board_slots + 3
    obs = {"entities_seq": torch.randn(B, T, S, cfg.entity_feature_dim, generator=g),
           "entity_mask": torch.rand(B, T, S, generator=g) < 0.7,
           "entity_types": torch.randint(0, cfg.n_entity_types, (B, T, S), generator=g),
           "globals_seq": torch.randn(B, T, cfg.hs_global_features, generator=g)}
    tgt = {"action_type": torch.randint(0, 4, (B, T), generator=g),
           "actor_idx": torch.randint(-1, S, (B, T), generator=g),
           "target_idx": torch.randint(-1, S, (B, T), generator=g),
           "position": torch.randint(-1, cfg.board_slots + 1, (B, T), generator=g)}
    pred = net(obs, labels={"action_type": drop(tgt["action_type"]),
                            "actor_idx": drop(tgt["actor_idx"])})
    out.append(("hearthstone", net, pred, tgt, mask))

    torch.manual_seed(0)
    cfg = _small(StarCraftConfig, minimap_h=16, minimap_w=16, max_units=8)
    net = GameNetwork(cfg, StarCraftEncoder(cfg), RTSDecoder(cfg))
    obs = {"minimap_seq": torch.randn(B, T, cfg.minimap_channels, 16, 16, generator=g),
           "units_seq": torch.randn(B, T, 8, cfg.unit_feature_dim, generator=g),
           "unit_mask": torch.rand(B, T, 8, generator=g) < 0.7,
           "globals_seq": torch.randn(B, T, cfg.sc_global_features, generator=g)}
    tgt = {"action_type": torch.randint(0, cfg.n_action_types, (B, T), generator=g),
           "actor_idx": torch.randint(0, 8, (B, T), generator=g),
           "target_idx": torch.randint(0, 8, (B, T), generator=g),
           "location_idx": torch.randint(0, 16, (B, T), generator=g),
           "queued": torch.rand(B, T, generator=g) < 0.3}
    pred = net(obs, labels={"action_type": drop(tgt["action_type"]),
                            "actor_idx": drop(tgt["actor_idx"])})
    out.append(("starcraft", net, pred, tgt, mask))

    for joint in (False, True):
        torch.manual_seed(0)
        cfg = _small(ModelConfig, grid_h=5, grid_w=5, joint_decoder=joint)
        net = GeneralsNetwork(cfg)
        obs = {"grid_seq": torch.rand(B, T, 12, 5, 5, generator=g),
               "globals_seq": torch.randn(B, T, 4, generator=g),
               "legal_source_mask_seq": torch.rand(B, T, 1, 5, 5, generator=g) < 0.5}
        tgt = {"count": torch.randint(-1, 3, (B, T), generator=g),
               "src_y": torch.randint(0, 5, (B, T, 4), generator=g),
               "src_x": torch.randint(0, 5, (B, T, 4), generator=g),
               "dir_id": torch.randint(0, 4, (B, T, 4), generator=g),
               "ratio_id": torch.randint(0, 2, (B, T, 4), generator=g)}
        pred = net(obs, legal=legal_from_obs(obs), labels=source_labels(tgt))
        out.append(("generals_joint" if joint else "generals", net, pred, tgt, mask))
    # lux with the pointer sap head: its sap term carries the in-map mask
    # the decoder echoes (every target here is on the map, so legal)
    torch.manual_seed(0)
    cfg = _small(LuxConfig, map_h=6, map_w=6, map_channels=4, max_units=5,
                 unit_feature_dim=6, lux_global_features=3, lux_sap_range=7,
                 lux_sap_head="pointer")
    net = GameNetwork(cfg, LuxEncoder(cfg), LuxDecoder(cfg))
    unit_mask = torch.rand(B, T, 5, generator=g) < 0.7
    xy = torch.stack([torch.randint(0, 6, (B, T, 5), generator=g),
                      torch.randint(0, 6, (B, T, 5), generator=g)], -1)
    units = torch.randn(B, T, 5, 6, generator=g)
    units[..., 0], units[..., 1] = xy[..., 0] / 6, xy[..., 1] / 6
    obs = {"map_seq": torch.randn(B, T, 4, 6, 6, generator=g), "units_seq": units,
           "unit_mask": unit_mask, "globals_seq": torch.randn(B, T, 3, generator=g)}
    on_map = lux_sap_legal(xy, unit_mask, 7, 7, 6, 6)
    so = torch.multinomial(on_map.reshape(-1, 225).float() + 1e-9, 1, generator=g)
    tgt = {"unit_actions": torch.randint(0, 6, (B, T, 5), generator=g),
           "sap_offset": so.reshape(B, T, 5)}
    pred = net(obs, legal={"unit": unit_mask})
    out.append(("lux", net, pred, tgt, mask))

    # the legacy pokemon slot head (configs saved before pokemon_move_head)
    torch.manual_seed(0)
    cfg = _small(PokemonConfig, pokemon_species_buckets=64, pokemon_move_head="slot")
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))
    pred = net(pk_obs, labels={"action_type": drop(pk_tgt["action_type"]),
                               "active_idx": pk_tgt["active_idx"]})
    out.append(("pokemon_slot", net, pred, pk_tgt, mask))

    # the legacy CR layout (one token per arena cell, no trackers): unmasked IL
    torch.manual_seed(0)
    cfg = _small(ClashRoyaleConfig, arena_h=8, arena_w=6, n_cards=20, cr_arena_patch=1,
                 cr_recent_tokens=0, cr_opp_tokens=0)
    net = GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))
    pred = net(cr_obs, labels={"card_slot": drop(cr_tgt["card_slot"])})
    out.append(("clash_royale_cells", net, pred, cr_tgt, mask))
    return out


#: il_loss output keys of each decoder (unchanged by the per-head refactor)
IL_KEYS = {
    "pokemon": {"type", "move", "switch", "tera"},
    "clash_royale": {"wait", "card", "patch", "sub"},
    "clash_royale_cells": {"wait", "card", "place"},
    "hearthstone": {"type", "actor", "target", "position"},
    "starcraft": {"type", "queued", "actor", "target", "location"},
    "generals": {"source", "direction", "split", "pass"},
    "lux": {"unit_action", "sap"},
}
HEADS = {
    "pokemon": {"action_type", "move", "switch", "tera"},
    "clash_royale": {"wait", "card", "patch", "sub"},
    "clash_royale_cells": {"wait", "card", "placement"},
    "hearthstone": {"action_type", "actor", "target", "position"},
    "starcraft": {"type", "queued", "actor", "target", "location"},
    "generals": {"source", "direction", "split", "pass"},
    "lux": {"unit_action", "sap"},
}


@pytest.mark.parametrize("case", range(9))
def test_owned_decoders_reconstruct_il_loss(case):
    name, net, pred, tgt, mask = _decoder_cases()[case]
    key = name.split("_")[0] if name.startswith(("generals", "pokemon")) else name
    dec = net.action_decoder
    il = dec.il_loss(pred, tgt, mask)
    terms = dec.il_head_terms(pred, tgt, mask)
    assert set(terms) == HEADS[key]
    if key == "generals":
        stat_keys = {"source_loss", "direction_loss", "split_loss", "pass_loss",
                     "src_acc", "dir_acc", "full_acc", "complete_acc"}
    else:
        stat_keys = {f"{h}_{s}" for h in IL_KEYS[key] for s in ("loss", "acc")}
    if key == "clash_royale":
        # the cell NLL (patch + cell, comparable with the legacy cell head) and the
        # masked heads' illegal-target fraction
        stat_keys |= {"place_loss", "illegal_target_frac"}
    assert set(il) == {"loss", "answer_correct", "answer_mask"} | stat_keys
    total = 0.0
    for h, t in terms.items():
        kind = t.get("kind", "categorical")
        lg = t["logits"]
        wshape = lg.shape[:-1] if kind == "categorical" else lg.shape
        assert t["weight"].shape == wshape and t["ce"].shape == wshape, h
        assert not t["weight"].requires_grad and t["ce"].requires_grad, h
        assert torch.equal(t["norm"], t["weight"].sum().clamp(min=1)), h
        total = total + t.get("scale", 1.0) * (t["ce"] * t["weight"]).sum() / t["norm"]
    assert float(total) == pytest.approx(float(il["loss"]), abs=1e-5)
    if key != "generals":
        # the per-head hard terms ARE the il_loss head losses
        for h, t in terms.items():
            stat = {"action_type": "type", "placement": "place"}.get(h, h)
            part = (t["ce"] * t["weight"]).sum() / t["norm"]
            assert float(part) == pytest.approx(float(il[f"{stat}_loss"]), abs=1e-6), h


def test_imitation_loss_head_terms_match_compute():
    from model.losses import ImitationLoss

    name, net, pred, tgt, mask = _decoder_cases()[4]
    loss = ImitationLoss(grid_w=5)
    out = loss.compute(pred, {**tgt, "value": torch.zeros(mask.shape)}, frame_mask=mask)
    terms = loss.head_terms(pred, tgt, frame_mask=mask)
    for h, stat in (("source", "source_loss"), ("direction", "direction_loss"),
                    ("split", "split_loss"), ("pass", "pass_loss")):
        t = terms[h]
        assert float((t["ce"] * t["weight"]).sum() / t["norm"]) == pytest.approx(
            float(out[stat]), abs=1e-6)
    total = sum(t["scale"] * (t["ce"] * t["weight"]).sum() / t["norm"] for t in terms.values())
    assert float(total) == pytest.approx(float(out["policy_loss"]), abs=1e-5)


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

def _hs_samples(n=3, T=5, seed=0):
    from model.games.hearthstone import HearthstoneConfig

    cfg = HearthstoneConfig()
    S = cfg.hand_slots + 2 * cfg.board_slots + 3
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        idx = torch.randint(0, cfg.board_slots + 1, (T, 2), generator=g)
        idx[:, 1] = -1
        out.append({
            "obs": {"entities_seq": torch.randn(T, S, cfg.entity_feature_dim, generator=g),
                    "entity_mask": torch.rand(T, S, generator=g) < 0.7,
                    "entity_types": torch.randint(0, cfg.n_entity_types, (T, S), generator=g),
                    "globals_seq": torch.randn(T, cfg.hs_global_features, generator=g)},
            "tgt": {"action_type": torch.randint(0, 4, (T,), generator=g),
                    "actor_idx": torch.randint(0, S, (T,), generator=g),
                    "target_idx": torch.randint(0, S, (T,), generator=g),
                    "position": torch.randint(0, cfg.board_slots + 1, (T,), generator=g),
                    "value": torch.ones(T),
                    "target_probs": _dirichlet((T, S), g),
                    "position_probs_idx": idx,
                    "position_probs_val": torch.ones(T, 2),
                    "engine_value": torch.rand(T, generator=g) * 2 - 1,
                    "legal_actor": torch.ones(T, S)},
            "meta": {}})
    return out


def test_train_multi_game_with_engine_targets_and_legal_masks(tmp_path, capsys):
    from training import train_multi_game

    save_shard(str(tmp_path / "hs" / "a.pt"), "hearthstone", _hs_samples())
    save_shard(str(tmp_path / "chess" / "a.pt"), "chess", _chess_samples(), converter_version=2)
    core = [f"{k}={v}" for k, v in TINY.items()]
    out = train_multi_game.main([
        "--data", f"hearthstone={tmp_path / 'hs'}", "--data", f"chess={tmp_path / 'chess'}",
        "--val", f"hearthstone={tmp_path / 'hs'}", "--steps", "2", "--batch_size", "2",
        "--warmup_steps", "1", "--eval_every", "2", "--ema_rate", "0", "--device", "cpu",
        "--il_legal_mask", "--soft_target_weight", "0.5", "--engine_value_weight", "0.7",
        "--save_dir", str(tmp_path / "ck"), "--override", *core])
    log = capsys.readouterr().out
    assert "[WARN] chess shards have converter_version 2 < 3" in log
    val = next(ln for ln in log.splitlines() if ln.startswith("[val 2] hearthstone"))
    assert "target_teacher_kl=" in val and "position_teacher_kl=" in val
    assert "engine_value_mse=" in val
    assert all(torch.isfinite(p).all() for p in out["model"].parameters())


def test_train_il_game_with_engine_targets_and_legal_masks(tmp_path, capsys, monkeypatch):
    from training import train_il_game

    save_shard(str(tmp_path / "hs" / "a.pt"), "hearthstone", _hs_samples())
    monkeypatch.setattr(train_il_game, "pick_device", lambda: CPU)
    monkeypatch.setattr(sys, "argv", [
        "train_il_game", "--game", "hearthstone", "--data_dir", str(tmp_path / "hs"),
        "--val_dir", str(tmp_path / "hs"), "--epochs", "1", "--batch_size", "2",
        "--warmup_steps", "1", "--ema_rate", "0", "--il_legal_mask",
        "--soft_target_weight", "0.5", "--engine_value_weight", "0.7",
        "--save_dir", str(tmp_path / "ck"), "--override",
        *[f"{k}={v}" for k, v in TINY.items()]])
    train_il_game.main()
    val = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[e0 val]"))
    for k in ("target_teacher_kl=", "position_teacher_frac=", "engine_value_sign_acc="):
        assert k in val, k


@pytest.mark.parametrize("module,flags", [
    ("train_il_game", ("--il_legal_mask", "--soft_target_weight", "--engine_value_weight",
                       "--require_frozen_schema")),
    ("train_multi_game", ("--il_legal_mask", "--soft_target_weight", "--engine_value_weight",
                          "--require_frozen_schema")),
    ("eval_multi_game", ("--il_legal_mask",)),
])
def test_cli_help_lists_the_flags(module, flags):
    out = subprocess.run([sys.executable, "-m", f"training.{module}", "--help"],
                         capture_output=True, text=True, check=True).stdout
    for f in flags:
        assert f in out, (module, f)
