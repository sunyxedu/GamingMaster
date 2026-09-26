"""Tests for the Pokémon game adapter.

Covers the adapter contract against the game-agnostic core (mirrors the
Hearthstone suite), for both move heads (``pokemon_move_head`` "identity"
— a categorical over the move vocabulary — and the legacy "slot" pointer):
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling (incl. the tera gate and the
     identity head's request mask)
  6. il_loss: finite, differentiable, robust to degenerate masks; an
     unsupervisable move label (slot head: −1 = no observed slot; identity
     head: −1 = no move) contributes zero move loss and is safe through
     decode()/sample(); frames without an observed choice (converter v2
     action_type −1 / action_known 0) carry no policy loss; echoed legal
     masks give masked CE with illegal targets unsupervised
  7. the deploy hook (legal_from_request / request_move_mask /
     move_bucket) and legacy configs rebuilding the slot head

Plus the ARG_USES invariant: the move head + tera Bernoulli contribute
log-prob only on move frames, the switch pointer only on switch frames.
"""

import pytest
import torch
import torch.nn.functional as F

from model.config import upgrade_config_dict
from model.game_network import GameNetwork
from model.games.pokemon import (
    NO_MOVE_LOGIT,
    POKE_FEAT_ACTIVE,
    PokemonConfig,
    PokemonDecoder,
    PokemonEncoder,
    legal_from_request,
    move_bucket,
    request_move_mask,
)

HEADS = ("identity", "slot")


# ======================================================================
# Fixtures / helpers
# ======================================================================

def _cfg(head, **kw):
    """Tiny config for fast tests."""
    return PokemonConfig(**{**dict(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, pokemon_species_buckets=64, pokemon_move_buckets=32,
        pokemon_move_head=head,
    ), **kw})


@pytest.fixture(params=HEADS)
def cfg(request):
    return _cfg(request.param)


@pytest.fixture
def model(cfg):
    return GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))


def _n_move(cfg):
    return cfg.pokemon_move_buckets if cfg.pokemon_move_head == "identity" else cfg.n_move_slots


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
    """Random legal masks with at least one legal entry per row (both heads'
    move masks, revealed-move ids)."""
    g = torch.Generator().manual_seed(seed)
    shape = (B,) if T is None else (B, T)
    legal = {
        "action_type": torch.rand(*shape, cfg.n_action_types, generator=g) < 0.5,
        "move": torch.rand(*shape, cfg.n_move_slots, generator=g) < 0.5,
        "move_id": torch.rand(*shape, cfg.pokemon_move_buckets, generator=g) < 0.2,
        "switch": torch.rand(*shape, cfg.team_slots, generator=g) < 0.4,
        "tera": torch.rand(*shape, generator=g) < 0.5,
        "active_idx": torch.randint(0, cfg.team_slots, shape, generator=g),
        "move_ids": torch.randint(0, cfg.pokemon_move_buckets, (*shape, cfg.n_move_slots),
                                  generator=g),
    }
    legal["move_id"][..., 0] = False                 # id 0 = no move
    legal["move_id"][..., 1] |= ~legal["move_id"].any(-1)
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
    assert out["move_logits"].shape == (B, T, _n_move(cfg))
    assert out["switch_logits"].shape == (B, T, cfg.team_slots)
    assert out["tera_logit"].shape == (B, T)
    assert out["action_type_used"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)
    assert not any(k.endswith("_mask") for k in out)          # echoes only with legal


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
    legal = {"move_ids": _make_legal(cfg, 2, 3)["move_ids"]}
    out = model(_make_obs(cfg, B=2, T=3), legal=legal)
    loss = sum(v.sum() for k, v in out.items() if v.is_floating_point() and k != "move_logits")
    # the identity head's id-0 logit is a constant: sum the real classes
    loss = loss + out["move_logits"][..., 1:].sum()
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
        assert out["move_logits"].shape == (B, _n_move(cfg))
        assert out["switch_logits"].shape == (B, cfg.team_slots)
        assert out["tera_logit"].shape == (B,)
        assert out["value"].shape == (B,)
        assert out["q_halt"].shape == (B,)
        assert out["action_log_prob"].shape == (B,)
        assert out[model.action_decoder.move_action_key].shape == (B,)


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
    move_key = model.action_decoder.move_action_key

    cache = None
    lps, stored = [], {k: [] for k in ("action_type", move_key, "switch_idx", "tera")}
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
    move_key = model.action_decoder.move_action_key
    switch_type = torch.ones(B, T, dtype=torch.long)   # type 1 = switch
    actions = {
        "action_type": switch_type,
        move_key: torch.randint(1, _n_move(cfg), (B, T)),
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
        move_lp = F.log_softmax(pred["move_logits"].float(), dim=-1).gather(
            -1, actions[move_key].unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(lp, type_lp + move_lp)


# ======================================================================
# 5. Legal-mask respect
# ======================================================================

def test_sampling_respects_legal_masks(cfg, model):
    model.eval()
    B = 4
    dec = model.action_decoder
    for trial in range(20):
        obs = {k: v[:, 0] for k, v in _make_obs(cfg, B, T=1, seed=trial).items()}
        legal = _make_legal(cfg, B, seed=trial)
        with torch.no_grad():
            out = model.step(obs, legal=legal)
        b = torch.arange(B)
        assert legal["action_type"][b, out["action_type"]].all(), "sampled illegal type"
        assert legal[dec.move_legal_key][b, out[dec.move_action_key]].all(), "sampled illegal move"
        assert legal["switch"][b, out["switch_idx"]].all(), "sampled illegal switch"
        assert (out["tera"].bool() <= legal["tera"]).all(), "tera despite closed gate"
        assert torch.isfinite(out["action_log_prob"]).all()


def test_greedy_takes_the_masked_argmax(cfg, model):
    model.eval()
    dec = model.action_decoder
    obs = {k: v[:, 0] for k, v in _make_obs(cfg, 4, T=1, seed=3).items()}
    legal = _make_legal(cfg, 4, seed=3)
    dec.greedy = True
    try:
        with torch.no_grad():
            a = model.step(obs, legal=legal)
            b = model.step(obs, legal=legal)
    finally:
        dec.greedy = False
    for k in ("action_type", dec.move_action_key, "switch_idx", "tera"):
        assert torch.equal(a[k], b[k]), k
    want = a["move_logits"].masked_fill(~legal[dec.move_legal_key], float("-inf")).argmax(-1)
    assert torch.equal(a[dec.move_action_key], want)


# ======================================================================
# 6. il_loss
# ======================================================================

def _make_tgt(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "action_type": torch.randint(0, cfg.n_action_types, (B, T), generator=g),
        "move_idx": torch.randint(0, cfg.n_move_slots, (B, T), generator=g),
        "move_id": torch.randint(1, cfg.pokemon_move_buckets, (B, T), generator=g),
        "switch_idx": torch.randint(0, cfg.team_slots, (B, T), generator=g),
        "tera": torch.randint(0, 2, (B, T), generator=g),
        "active_idx": torch.randint(0, cfg.team_slots, (B, T), generator=g),
    }


def _move_tgt_key(cfg):
    return "move_id" if cfg.pokemon_move_head == "identity" else "move_idx"


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
    assert "illegal_target_frac" not in out


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
        "move_id": torch.full((B, T), -1),
        "switch_idx": torch.randint(0, cfg.team_slots, (B, T)),
        "tera": torch.full((B, T), -1),
    }
    out = model.action_decoder.il_loss(pred, tgt_switch, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["loss"])


def test_il_loss_unrepresentable_move_label_zero_loss(cfg, model):
    """An unsupervisable move label on a GT MOVE frame (slot head: move_idx
    −1, the move has no observed slot; identity head: move_id −1) must
    contribute zero move loss: perturbing the move logits at exactly those
    frames leaves every il_loss output unchanged — a clamped gather would
    instead train the clamp target's class."""
    B, T = 2, 3
    g = torch.Generator().manual_seed(3)
    key = _move_tgt_key(cfg)
    tgt = _make_tgt(cfg, B, T)
    tgt["action_type"] = torch.zeros(B, T, dtype=torch.long)   # all move frames
    tgt[key][0, 1] = -1
    tgt[key][1, 2] = -1
    pred = model(_make_obs(cfg, B, T),
                 labels={"action_type": tgt["action_type"],
                         "active_idx": tgt["active_idx"]})
    pred = {k: v.detach() for k, v in pred.items()}
    mask = torch.ones(B, T, dtype=torch.bool)
    out1 = model.action_decoder.il_loss(pred, tgt, mask)
    assert torch.isfinite(out1["loss"])

    pred2 = {k: v.clone() for k, v in pred.items()}
    pred2["move_logits"][0, 1] = torch.randn(_n_move(cfg), generator=g) * 10
    pred2["move_logits"][1, 2] = torch.randn(_n_move(cfg), generator=g) * 10
    out2 = model.action_decoder.il_loss(pred2, tgt, mask)
    for k in ("loss", "move_loss", "move_acc"):
        torch.testing.assert_close(out1[k], out2[k], msg=k)

    # every move label unsupervisable → the move head goes fully silent, no NaN
    tgt[key] = torch.full((B, T), -1)
    out3 = model.action_decoder.il_loss(pred, tgt, mask)
    assert out3["move_loss"] == 0.0
    assert torch.isfinite(out3["loss"])


def test_decode_negative_move_label_no_crash_self_samples(cfg, model):
    """labels carrying a −1 move label must be conditioning-safe: decode()
    never teacher-forces the move (moves are sampled downstream), so a −1
    label neither crashes nor leaks into the outputs, and sample() still
    picks a real move."""
    model.eval()
    B, T = 2, 3
    obs = _make_obs(cfg, B, T)
    tgt_type = torch.zeros(B, T, dtype=torch.long)
    labels = {
        "action_type": tgt_type,
        "active_idx": torch.randint(0, cfg.team_slots, (B, T)),
        "move_idx": torch.full((B, T), -1),
        "move_id": torch.full((B, T), -1),
    }
    with torch.no_grad():
        out = model(obs, labels=labels)
    for k in ("action_type_logits", "move_logits", "switch_logits", "tera_logit"):
        assert torch.isfinite(out[k]).all(), k
    assert torch.equal(out["action_type_used"], tgt_type)

    # rollout self-sampling picks a real move regardless of the −1 label
    pred0 = {k: v[:, 0] for k, v in out.items()}
    sampled = model.action_decoder.sample(pred0, legal=None)
    mv = sampled[model.action_decoder.move_action_key]
    lo = 1 if cfg.pokemon_move_head == "identity" else 0
    assert ((mv >= lo) & (mv < _n_move(cfg))).all()
    assert torch.isfinite(sampled["action_log_prob"]).all()


def test_frames_without_an_observed_choice_carry_no_policy_loss(cfg, model):
    """Converter v2 keeps the frame of a turn whose choice is not visible
    (action_type −1, action_known 0): perturbing every head's logits there
    changes no il_loss output, and such frames are no q_halt answers."""
    B, T = 2, 4
    g = torch.Generator().manual_seed(5)
    tgt = _make_tgt(cfg, B, T)
    tgt["action_type"][0, 1] = -1
    tgt["action_known"] = torch.ones(B, T, dtype=torch.bool)
    tgt["action_known"][1, 2] = False                           # a masked label, type kept
    labels = {"action_type": tgt["action_type"], "active_idx": tgt["active_idx"]}
    pred = model(_make_obs(cfg, B, T), labels=labels)
    pred = {k: v.detach() for k, v in pred.items()}
    mask = torch.ones(B, T, dtype=torch.bool)
    out1 = model.action_decoder.il_loss(pred, tgt, mask)
    pred2 = {k: v.clone() for k, v in pred.items()}
    for b, t in ((0, 1), (1, 2)):
        for k in ("action_type_logits", "move_logits", "switch_logits"):
            pred2[k][b, t] = torch.randn(pred2[k].shape[-1], generator=g) * 10
        pred2["tera_logit"][b, t] = 7.0
    out2 = model.action_decoder.il_loss(pred2, tgt, mask)
    for k in ("loss", "type_loss", "move_loss", "switch_loss", "tera_loss", "type_acc"):
        torch.testing.assert_close(out1[k], out2[k], msg=k)
    assert not out1["answer_mask"][0, 1] and not out1["answer_mask"][1, 2]
    assert out1["answer_mask"].sum() == B * T - 2
    # the type loss is a mean over the labelled frames only
    ce = F.cross_entropy(pred["action_type_logits"].reshape(-1, 2),
                         tgt["action_type"].clamp(min=0).reshape(-1), reduction="none")
    keep = out1["answer_mask"].reshape(-1).float()
    torch.testing.assert_close(out1["type_loss"], (ce * keep).sum() / keep.sum())


def test_legal_masks_are_echoed_and_score_il(cfg, model):
    """IL legality (--il_legal_mask: converter v2 legal_action_type /
    legal_switch): decode() echoes the masks, il_loss scores the masked
    heads, a target outside its mask is unsupervised and counted; the
    il_head_terms reconstruction stays exact."""
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    tgt["action_type"] = torch.tensor([[1, 1, 0], [1, 0, 1]])
    legal = {"action_type": torch.tensor([[[False, True]] * 3, [[True, True]] * 3]),
             "switch": torch.ones(B, T, cfg.team_slots, dtype=torch.bool)}
    legal["switch"][0, 0] = False
    legal["switch"][0, 0, int(tgt["switch_idx"][0, 0])] = True     # legal target
    legal["switch"][1, 2, int(tgt["switch_idx"][1, 2])] = False    # illegal target
    labels = {"action_type": tgt["action_type"], "active_idx": tgt["active_idx"]}
    pred = model(_make_obs(cfg, B, T), legal=legal, labels=labels)
    assert torch.equal(pred["action_type_mask"], legal["action_type"])
    assert torch.equal(pred["switch_mask"], legal["switch"])
    assert "move_mask" not in pred
    out = model.action_decoder.il_loss(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    # frame (0, 2) is a move frame whose type is masked out: illegal; (1, 2) its switch
    assert out["illegal_target_frac"] == pytest.approx(2 / 6)
    terms = model.action_decoder.il_head_terms(pred, tgt, torch.ones(B, T, dtype=torch.bool))
    assert terms["switch"]["weight"][1, 2] == 0 and terms["action_type"]["weight"][0, 2] == 0
    total = sum(float(t.get("scale", 1.0)) * (t["ce"] * t["weight"]).sum() / t["norm"]
                for t in terms.values())
    torch.testing.assert_close(out["loss"], total)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()


# ======================================================================
# The identity move head
# ======================================================================

def test_identity_head_scores_the_whole_vocabulary():
    """First uses are representable: every vocabulary id is a class, id 0
    ("no move") carries no mass, and a revealed move (move_ids) raises its
    own logit only."""
    cfg = _cfg("identity")
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg)).eval()
    B, T = 1, 2
    obs = _make_obs(cfg, B, T)
    ids = torch.tensor([[[5, 9, 0, 0], [5, 9, 0, 0]]])
    with torch.no_grad():
        net.action_decoder.move_revealed.fill_(2.0)
        a = net(obs, legal={"move_ids": ids}, labels={"action_type": torch.zeros(B, T).long()})
        net.action_decoder.move_revealed.fill_(0.0)
        b = net(obs, legal={"move_ids": ids}, labels={"action_type": torch.zeros(B, T).long()})
    assert a["move_logits"].shape == (B, T, cfg.pokemon_move_buckets)
    assert (a["move_logits"][..., 0] == NO_MOVE_LOGIT).all()
    diff = a["move_logits"] - b["move_logits"]
    torch.testing.assert_close(diff[..., [5, 9]], torch.full((B, T, 2), 2.0))
    assert diff[..., [1, 2, 3, 4, 6, 7, 8, 10]].abs().max() == 0
    # a never-revealed move is a finite, trainable target
    tgt = {"action_type": torch.zeros(B, T).long(), "move_id": torch.tensor([[17, 5]]),
           "move_idx": torch.tensor([[-1, 0]]), "switch_idx": torch.zeros(B, T).long(),
           "tera": torch.zeros(B, T).long()}
    out = net.action_decoder.il_loss(b, tgt, torch.ones(B, T, dtype=torch.bool))
    assert torch.isfinite(out["move_loss"]) and out["move_loss"] > 0


def test_identity_head_needs_the_v2_target():
    cfg = _cfg("identity")
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))
    tgt = _make_tgt(cfg, 1, 2)
    del tgt["move_id"]
    pred = net(_make_obs(cfg, 1, 2), labels={"action_type": tgt["action_type"]})
    with pytest.raises(ValueError, match="converter v2"):
        net.action_decoder.il_loss(pred, tgt, torch.ones(1, 2, dtype=torch.bool))


# ======================================================================
# 7. Deploy hook + legacy configs
# ======================================================================

def test_move_bucket_is_the_converter_hash():
    assert move_bucket("", 1024) == 0
    assert move_bucket("Earthquake", 1024) == move_bucket("earthquake", 1024)
    assert move_bucket("U-turn", 1024) == move_bucket("uturn", 1024)
    assert all(1 <= move_bucket(m, 32) < 32 for m in ("Splash", "Make It Rain", "Knock Off"))
    # the empty-after-normalisation name still hashes (crc32(b"") = 0 → id 1), as in Rust
    assert move_bucket("---", 32) == 1


def _request(**kw):
    side = [
        {"ident": "p1: Garchomp", "condition": "80/100", "active": True},
        {"ident": "p1: Ditto", "condition": "0 fnt", "active": False},
        {"ident": "p1: Corviknight", "condition": "100/100", "active": False},
    ]
    active = [{"moves": [{"move": "Earthquake", "id": "earthquake", "disabled": False},
                         {"move": "Swords Dance", "id": "swordsdance", "disabled": True},
                         {"move": "U-turn", "id": "uturn"}],
               "canTerastallize": "Ground"}]
    return {"side": {"pokemon": side}, "active": active, **kw}


def test_legal_from_request():
    V = 64
    # the model's slots: the team-preview order (Showdown reorders side.pokemon later)
    slot_of = {"p1: Corviknight": 0, "p1: Garchomp": 1, "p1: Ditto": 2}
    lg = legal_from_request(_request(), slot_of, V, move_ids=torch.tensor([7, 0, 0, 0]))
    assert lg["action_type"].tolist() == [[True, True]]
    assert lg["switch"].tolist() == [[True, False, False, False, False, False]]
    assert lg["active_idx"].tolist() == [1] and lg["tera"].tolist() == [True]
    want = {move_bucket("Earthquake", V), move_bucket("U-turn", V)}
    assert set(lg["move_id"][0].nonzero().squeeze(1).tolist()) == want
    assert lg["move_ids"].tolist() == [[7, 0, 0, 0]]
    assert torch.equal(lg["move_id"], request_move_mask([["earthquake", "U-turn"]], V))
    # forced switch: switch only; trapped: move only; team preview: every present slot
    forced = legal_from_request(_request(forceSwitch=[True]), slot_of, V)
    assert forced["action_type"].tolist() == [[False, True]] and not forced["move_id"].any()
    assert forced["switch"].tolist() == [[True, False, False, False, False, False]]
    req = _request()
    req["active"][0]["trapped"] = True
    assert legal_from_request(req, slot_of, V)["action_type"].tolist() == [[True, False]]
    tp = legal_from_request({"teamPreview": True, "side": {"pokemon": [
        {"ident": k, "condition": "100/100"} for k in slot_of]}}, slot_of, V)
    assert tp["action_type"].tolist() == [[False, True]]
    assert tp["switch"].tolist() == [[True, True, True, False, False, False]]
    assert tp["active_idx"].tolist() == [0]


def test_request_masks_drive_step_sampling():
    """The deploy path: GameNetwork.step with legal_from_request samples one
    of the request's usable moves (identity head) and its log-prob is exact."""
    cfg = _cfg("identity", pokemon_move_buckets=64)
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg)).eval()
    slot_of = {"p1: Garchomp": 0, "p1: Ditto": 1, "p1: Corviknight": 2}
    lg = legal_from_request(_request(), slot_of, 64, move_ids=torch.tensor([3, 0, 0, 0]))
    lg["action_type"] = torch.tensor([[True, False]])                # force a move
    obs = {k: v[:, 0] for k, v in _make_obs(cfg, 1, T=1).items()}
    allowed = set(lg["move_id"][0].nonzero().squeeze(1).tolist())
    for seed in range(10):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = net.step(obs, legal=lg)
        assert out["action_type"].item() == 0 and out["move_id"].item() in allowed
        pred = {k: v.unsqueeze(1) for k, v in out.items() if k.endswith(("_logits", "_logit"))}
        acts = {k: out[k].unsqueeze(1) for k in ("action_type", "move_id", "switch_idx", "tera")}
        lp, _, _ = net.compute_log_prob(pred, acts, {k: v.unsqueeze(1) for k, v in lg.items()})
        torch.testing.assert_close(lp.squeeze(1), out["action_log_prob"])


#: the pokemon decoder's parameter / persistent-buffer names of every
#: checkpoint written before the identity head (the slot head)
SLOT_HEAD_KEYS = {
    "ARG_USES", "type_head.net.0.weight", "type_head.net.0.bias", "type_head.net.2.weight",
    "type_head.net.2.bias", "type_embed.weight", "move_embed.weight", "move_slot_embed",
    "move_head.net.0.weight", "move_head.net.0.bias", "move_head.net.2.weight",
    "move_head.net.2.bias", "switch_head.net.0.weight", "switch_head.net.0.bias",
    "switch_head.net.2.weight", "switch_head.net.2.bias", "tera_head.net.0.weight",
    "tera_head.net.0.bias", "tera_head.net.2.weight", "tera_head.net.2.bias",
}


def test_legacy_configs_rebuild_the_slot_head():
    """A config saved before pokemon_move_head existed (it has
    pokemon_move_buckets) rebuilds the slot head: same state-dict keys, the
    checkpoint loads strictly and computes the same outputs."""
    cfg = _cfg("slot")
    old = {k: v for k, v in cfg.__dict__.items() if k != "pokemon_move_head"}
    up = upgrade_config_dict(old)
    assert up["pokemon_move_head"] == "slot"
    assert "pokemon_move_head" not in upgrade_config_dict({"d_model": 48})   # other games
    assert PokemonConfig().pokemon_move_head == "identity"                  # fresh configs
    torch.manual_seed(0)
    src = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg)).eval()
    assert set(src.action_decoder.state_dict()) == SLOT_HEAD_KEYS
    rebuilt_cfg = PokemonConfig(**up)
    dst = GameNetwork(rebuilt_cfg, PokemonEncoder(rebuilt_cfg), PokemonDecoder(rebuilt_cfg)).eval()
    dst.load_state_dict(src.state_dict(), strict=True)
    obs = _make_obs(cfg, 2, 3)
    legal = {"move_ids": _make_legal(cfg, 2, 3)["move_ids"]}
    labels = {"action_type": torch.zeros(2, 3).long(), "active_idx": torch.zeros(2, 3).long()}
    with torch.no_grad():
        a, b = src(obs, legal=legal, labels=labels), dst(obs, legal=legal, labels=labels)
    for k in ("action_type_logits", "move_logits", "switch_logits", "tera_logit", "value"):
        assert torch.equal(a[k], b[k]), k
    # the identity head has its own parameters: a slot checkpoint does not fit it
    ident = _cfg("identity")
    with pytest.raises(RuntimeError):
        GameNetwork(ident, PokemonEncoder(ident), PokemonDecoder(ident)).load_state_dict(
            src.state_dict(), strict=True)


def _reference_slot_il(pred, tgt, seq_mask):
    """The pre-v2 PokemonDecoder.il_loss (slot head, no masks) — the legacy
    objective the slot head must still compute bit for bit on v1 targets."""
    seq_f = seq_mask.float()
    n_seq = seq_f.sum().clamp(min=1)
    tgt_type = tgt["action_type"].long().clamp(0, 1)
    uses = torch.tensor([[True, False, True], [False, True, False]])[tgt_type]
    type_match = (pred["action_type_used"] == tgt_type).float()
    w_move = uses[..., 0].float() * (tgt["move_idx"] >= 0).float() * type_match * seq_f
    w_switch = uses[..., 1].float() * type_match * seq_f
    w_tera = uses[..., 2].float() * seq_f

    def ce(logits, target, n):
        B, T = target.shape
        return F.cross_entropy(logits.reshape(-1, n), target.long().clamp(0, n - 1).reshape(-1),
                               reduction="none").reshape(B, T).nan_to_num(0.0)

    tera = F.binary_cross_entropy_with_logits(pred["tera_logit"], tgt["tera"].float().clamp(0, 1),
                                              reduction="none").nan_to_num(0.0)
    return ((ce(pred["action_type_logits"], tgt_type, 2) * seq_f).sum() / n_seq
            + (ce(pred["move_logits"], tgt["move_idx"], 4) * w_move).sum() / w_move.sum().clamp(min=1)
            + (ce(pred["switch_logits"], tgt["switch_idx"], 6) * w_switch).sum()
            / w_switch.sum().clamp(min=1)
            + (tera * w_tera).sum() / w_tera.sum().clamp(min=1))


def test_slot_head_il_loss_is_the_legacy_objective():
    cfg = _cfg("slot")
    torch.manual_seed(1)
    net = GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))
    B, T = 3, 5
    tgt = _make_tgt(cfg, B, T, seed=4)
    tgt["move_idx"][0, :2] = -1
    del tgt["move_id"]
    g = torch.Generator().manual_seed(2)
    labels = {"action_type": torch.where(torch.rand(B, T, generator=g) < 0.6, tgt["action_type"],
                                         torch.full_like(tgt["action_type"], -1)),
              "active_idx": tgt["active_idx"]}
    pred = net(_make_obs(cfg, B, T), labels=labels)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 3:] = False
    out = net.action_decoder.il_loss(pred, tgt, mask)
    assert torch.equal(out["loss"], _reference_slot_il(pred, tgt, mask))
    assert torch.equal(out["answer_mask"], mask)
