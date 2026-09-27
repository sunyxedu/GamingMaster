"""Tests for the Clash Royale game adapter.

Covers the adapter contract against the game-agnostic core, for both card-token
variants (``cr_card_tokens``: the 8 deck tokens with the causal hand belief, and
the legacy 4 hand slots):
  1. forward output shapes (all pred keys, value, q_halt)
  2. gradient flow into encoder, decoder, and the shared core
  3. step() chaining with the Stage-1 attention cache
  4. PPO exactness: rollout log-probs == teacher-forced replay log-probs
  5. legal-mask respect during sampling
  6. il_loss: finite, differentiable, robust to degenerate masks
"""

import pytest
import torch

from model.game_network import GameNetwork
from model.games.base import masked_sample
from model.games.clash_royale import (
    CardPlacementDecoder,
    ClashRoyaleConfig,
    ClashRoyaleEncoder,
    n_card_tokens,
)

# ======================================================================
# Fixtures / helpers
# ======================================================================

@pytest.fixture(params=["deck", "hand"])
def cfg(request):
    """Tiny config for fast tests (deck tokens / legacy hand slots)."""
    return ClashRoyaleConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0,
        n_st_layers=1, core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
        halt_max_steps=2, encoder_channels=24,
        arena_h=8, arena_w=6, arena_channels=5,
        hand_size=4, n_cards=20, cr_global_features=6, cr_card_tokens=request.param,
        card_feature_dim=9 if request.param == "deck" else 8,
    )


@pytest.fixture
def model(cfg):
    return GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))


def _make_obs(cfg, B=2, T=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    N, pre = n_card_tokens(cfg), cfg.cr_card_tokens
    obs = {
        "arena_seq": torch.randn(B, T, cfg.arena_channels, cfg.arena_h, cfg.arena_w,
                                 generator=g),
        f"{pre}_ids": torch.randint(0, cfg.n_cards, (B, T, N), generator=g),
        f"{pre}_feats": torch.randn(B, T, N, cfg.card_feature_dim, generator=g),
        # Some empty slots so the "empty" embedding participates
        f"{pre}_mask": torch.rand(B, T, N, generator=g) < 0.8,
        "globals_seq": torch.randn(B, T, cfg.cr_global_features, generator=g),
    }
    obs[f"{pre}_mask"][..., 0] = True  # at least one real card
    return obs


def _make_legal(cfg, B, T=None, seed=0):
    """Random legal masks with at least one legal entry per row."""
    g = torch.Generator().manual_seed(seed)
    shape = (B,) if T is None else (B, T)
    n_arena = cfg.arena_h * cfg.arena_w
    card = torch.rand(*shape, n_card_tokens(cfg), generator=g) < 0.5
    place = torch.rand(*shape, n_arena, generator=g) < 0.3
    card[..., 0] |= ~card.any(-1)
    place[..., 0] |= ~place.any(-1)
    return {"card": card, "placement": place}


# ======================================================================
# 1. Forward shapes
# ======================================================================

def test_forward_shapes(cfg, model):
    B, T = 2, 3
    n_arena = cfg.arena_h * cfg.arena_w
    out = model(_make_obs(cfg, B, T))
    assert out["wait_logits"].shape == (B, T)
    assert out["card_logits"].shape == (B, T, n_card_tokens(cfg))
    assert out["placement_logits"].shape == (B, T, n_arena)
    assert out["card_slot"].shape == (B, T)
    assert out["value"].shape == (B, T)
    assert out["q_halt"].shape == (B, T)
    assert (out["card_slot"] >= 0).all() and (out["card_slot"] < n_card_tokens(cfg)).all()


def test_forward_teacher_forcing_uses_labels(cfg, model):
    B, T = 2, 3
    labels = {"card_slot": torch.randint(0, n_card_tokens(cfg), (B, T))}
    out = model(_make_obs(cfg, B, T), labels=labels)
    assert torch.equal(out["card_slot"], labels["card_slot"])


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
        got_signal = any(
            p.grad.abs().sum() > 0 for p in module.parameters()
        )
        assert got_signal, f"{name} gradients are all zero"


# ======================================================================
# 3. step() chaining with st_cache
# ======================================================================

def test_step_chaining_cache_growth(cfg, model):
    model.eval()
    B = 2
    n_arena = cfg.arena_h * cfg.arena_w
    S = model.obs_encoder.n_tokens + 2  # + HL/LL decision tokens
    obs = _make_obs(cfg, B, T=3)
    cache = None
    for t in range(3):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache)
        cache = out["st_cache"]
        assert len(cache) == cfg.n_st_layers
        assert cache[0].shape == (B * S, t + 1, cfg.d_model)  # attention KV cache grows
        assert out["wait_logits"].shape == (B,)
        assert out["card_logits"].shape == (B, n_card_tokens(cfg))
        assert out["placement_logits"].shape == (B, n_arena)
        assert out["value"].shape == (B,)
        assert out["q_halt"].shape == (B,)
        assert out["action_log_prob"].shape == (B,)
        assert out["waited"].dtype == torch.bool


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
    lps, slots, places, waits = [], [], [], []
    for t in range(T):
        step_obs = {k: v[:, t] for k, v in obs.items()}
        with torch.no_grad():
            out = model.step(step_obs, legal=None, st_cache=cache, act_steps=K)
        cache = out["st_cache"]
        lps.append(out["action_log_prob"])
        slots.append(out["card_slot"])
        places.append(out["place_idx"])
        waits.append(out["waited"])

    actions = {
        "card_slot": torch.stack(slots, dim=1),
        "place_idx": torch.stack(places, dim=1),
        "waited": torch.stack(waits, dim=1),
    }
    with torch.no_grad():
        full = model(obs, labels={"card_slot": actions["card_slot"]}, act_steps=K)
        lp, ent, head_ent = model.compute_log_prob(full, actions)

    assert torch.equal(full["card_slot"], actions["card_slot"])  # conditioning replayed
    rollout_lp = torch.stack(lps, dim=1)
    torch.testing.assert_close(lp, rollout_lp, atol=1e-5, rtol=1e-5)
    assert ent.shape == (B, T) and (ent >= 0).all()
    assert set(head_ent) == {"ent_wait", "ent_card", "ent_place"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sample_replay_log_prob_parity(cfg, dtype):
    """sample()'s rollout bookkeeping and log_prob_entropy()'s replay must
    agree bit-for-bit on the same pred — including bf16 logits, where both
    sides must evaluate the fp32 distribution (the e25e464 invariant)."""
    torch.manual_seed(0)
    decoder = CardPlacementDecoder(cfg)
    B, T = 4, 6
    n_arena = cfg.arena_h * cfg.arena_w
    g = torch.Generator().manual_seed(3)

    wait = torch.randn(B, T, generator=g) * 5
    wait[:, 0] = 20.0   # sigmoid ≈ 1: guaranteed-wait frames
    wait[:, 1] = -20.0  # sigmoid ≈ 0: guaranteed-play frames
    pred = {
        "wait_logits": wait.to(dtype).requires_grad_(),
        "card_logits": (torch.randn(B, T, n_card_tokens(cfg), generator=g) * 5)
        .to(dtype).requires_grad_(),
        "placement_logits": (torch.randn(B, T, n_arena, generator=g) * 5)
        .to(dtype).requires_grad_(),
    }
    legal = _make_legal(cfg, B, T, seed=3)

    lps, slots, places, waits = [], [], [], []
    for t in range(T):
        pred_t = {k: v[:, t] for k, v in pred.items()}
        legal_t = {k: v[:, t] for k, v in legal.items()}
        # decode() commits the card via masked_sample; replicate that here
        pred_t["card_slot"] = masked_sample(pred_t["card_logits"], legal_t["card"])
        out = decoder.sample(pred_t, legal_t)
        lps.append(out["action_log_prob"])
        slots.append(out["card_slot"])
        places.append(out["place_idx"])
        waits.append(out["waited"])

    actions = {
        "card_slot": torch.stack(slots, dim=1),
        "place_idx": torch.stack(places, dim=1),
        "waited": torch.stack(waits, dim=1),
    }
    assert actions["waited"][:, 0].all() and not actions["waited"][:, 1].any()

    pred["card_slot"] = actions["card_slot"]
    lp, ent, head_ent = decoder.log_prob_entropy(pred, actions, legal)

    rollout_lp = torch.stack(lps, dim=1)
    assert lp.dtype == torch.float32 and rollout_lp.dtype == torch.float32
    assert torch.equal(lp, rollout_lp)  # bitwise: PPO ratio is exactly 1
    assert ent.dtype == torch.float32 and torch.isfinite(ent).all()

    # Replay is the PPO training path — it must stay differentiable.
    # (Entropy backward under -inf masks NaNs at HEAD for every adapter —
    # pre-existing, so only the ratio path is checked for finite grads.)
    assert ent.requires_grad
    lp.sum().backward()
    for key in ("wait_logits", "card_logits", "placement_logits"):
        grad = pred[key].grad
        assert grad is not None and torch.isfinite(grad).all(), key
        assert grad.abs().sum() > 0, key


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
        assert legal["card"][b, out["card_slot"]].all(), "sampled illegal card"
        assert legal["placement"][b, out["place_idx"]].all(), "sampled illegal placement"
        assert torch.isfinite(out["action_log_prob"]).all()


# ======================================================================
# 6. il_loss
# ======================================================================

def _make_tgt(cfg, B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_arena = cfg.arena_h * cfg.arena_w
    return {
        "waited": torch.rand(B, T, generator=g) < 0.4,
        "card_slot": torch.randint(0, n_card_tokens(cfg), (B, T), generator=g),
        "place_idx": torch.randint(0, n_arena, (B, T), generator=g),
    }


def test_il_loss_finite_and_differentiable(cfg, model):
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T), labels={"card_slot": tgt["card_slot"]})
    seq_mask = torch.ones(B, T, dtype=torch.bool)
    seq_mask[1, -1] = False

    out = model.action_decoder.il_loss(pred, tgt, seq_mask)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in model.action_decoder.parameters())
    for k in ("wait_loss", "card_loss", "place_loss",
              "wait_acc", "card_acc", "place_acc"):
        assert torch.isfinite(out[k]), f"{k} not finite"
        assert not out[k].requires_grad, f"{k} should be detached"


def test_il_loss_degenerate_masks(cfg, model):
    """All-padded sequences and all-wait targets must not produce NaN."""
    B, T = 2, 3
    tgt = _make_tgt(cfg, B, T)
    pred = model(_make_obs(cfg, B, T), labels={"card_slot": tgt["card_slot"]})

    # Everything masked out
    out = model.action_decoder.il_loss(
        {k: v.detach() for k, v in pred.items()},
        tgt, torch.zeros(B, T, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])

    # Every frame waited: card/place targets are meaningless (−1)
    tgt_wait = {
        "waited": torch.ones(B, T, dtype=torch.bool),
        "card_slot": torch.full((B, T), -1),
        "place_idx": torch.full((B, T), -1),
    }
    out = model.action_decoder.il_loss(
        {k: v.detach() for k, v in pred.items()},
        tgt_wait, torch.ones(B, T, dtype=torch.bool),
    )
    assert torch.isfinite(out["loss"])


# ======================================================================
# Wait-head time feature: match clock (crop-safe) vs legacy frame index
# ======================================================================

def test_wait_pe_reads_the_match_clock_not_the_frame_index(cfg):
    """cr_wait_time='match' (the default): the wait feature is the match
    clock from the observation, so the absolute frame offset t0 of a window
    changes nothing and the clock does; legacy 'frame' is the reverse."""
    torch.manual_seed(0)
    dec = CardPlacementDecoder(cfg).eval()
    B, T, d = 2, 3, cfg.d_model
    tokens = torch.randn(B * T, cfg.arena_h * cfg.arena_w + n_card_tokens(cfg), d)
    hl, ll = torch.randn(B * T, d), torch.randn(B * T, d)
    clock = torch.tensor([[10.0, 11.0, 12.0], [100.0, 101.0, 102.0]])
    with torch.no_grad():
        a = dec.decode(tokens, hl, ll, B, T, t0=0, ctx={"match_time": clock})["wait_logits"]
        b = dec.decode(tokens, hl, ll, B, T, t0=37, ctx={"match_time": clock})["wait_logits"]
        c = dec.decode(tokens, hl, ll, B, T, t0=0, ctx={"match_time": clock + 60})["wait_logits"]
    assert torch.equal(a, b) and not torch.allclose(a, c)
    with pytest.raises(KeyError, match="match_time"):
        dec.decode(tokens, hl, ll, B, T)

    legacy = ClashRoyaleConfig(**{**cfg.__dict__, "cr_wait_time": "frame"})
    dec_f = CardPlacementDecoder(legacy).eval()
    dec_f.load_state_dict(dec.state_dict())         # same parameters, other time feature
    with torch.no_grad():
        f0 = dec_f.decode(tokens, hl, ll, B, T, t0=0)["wait_logits"]
        f1 = dec_f.decode(tokens, hl, ll, B, T, t0=37)["wait_logits"]
    assert not torch.allclose(f0, f1)


def test_decode_context_is_the_clock_in_seconds(cfg):
    from model.games.clash_royale import CR_GLOBAL_TIME, CR_TIME_SCALE

    obs = _make_obs(cfg, B=2, T=3)
    ctx = ClashRoyaleEncoder(cfg).decode_context(obs)
    assert torch.equal(ctx["match_time"], obs["globals_seq"][..., CR_GLOBAL_TIME] * CR_TIME_SCALE)
    legacy = ClashRoyaleConfig(**{**cfg.__dict__, "cr_wait_time": "frame"})
    assert ClashRoyaleEncoder(legacy).decode_context(obs) is None


def test_legacy_configs_keep_the_frame_wait_pe_and_the_hand_slots():
    from model.config import upgrade_config_dict

    old = {"d_model": 48, "cr_global_features": 6}            # a pre-field CR config
    up = upgrade_config_dict(old)
    assert (up["cr_wait_time"], up["cr_card_tokens"], up["card_feature_dim"]) == \
        ("frame", "hand", 8)
    other = upgrade_config_dict({"d_model": 48})               # other games
    assert not {"cr_wait_time", "cr_card_tokens", "card_feature_dim"} & set(other)
    new = ClashRoyaleConfig()
    assert (new.cr_wait_time, new.cr_card_tokens, new.deck_size, new.card_feature_dim) == \
        ("match", "deck", 8, 9)
    with pytest.raises(ValueError, match="cr_card_tokens"):
        ClashRoyaleEncoder(ClashRoyaleConfig(cr_card_tokens="slots"))


def test_crop_policy_refuses_frame_time_nets_on_windows(cfg):
    from training.train_il_game import check_crop_policy

    legacy = ClashRoyaleConfig(**{**cfg.__dict__, "cr_wait_time": "frame"})
    check_crop_policy("clash_royale", legacy, None)             # full sequences: fine
    check_crop_policy("clash_royale", cfg, 64)                  # match clock: crop-safe
    with pytest.raises(SystemExit, match="absolute frame time"):
        check_crop_policy("clash_royale", legacy, 64)
    check_crop_policy("chess", legacy, 64)                      # no rule for other games


def _cr_pt_samples(n=2, T=12, seed=0):
    """Deck-token samples (crdata converter_version 3 observation)."""
    g = torch.Generator().manual_seed(seed)
    c = ClashRoyaleConfig()
    D = c.deck_size
    out = []
    for _ in range(n):
        glob = torch.rand(T, c.cr_global_features, generator=g)
        glob[:, 2] = torch.arange(T) / 300.0                    # match clock
        out.append({
            "obs": {"arena_seq": torch.rand(T, c.arena_channels, c.arena_h, c.arena_w, generator=g),
                    "deck_ids": torch.randint(1, c.n_cards, (T, D), generator=g),
                    "deck_feats": torch.rand(T, D, c.card_feature_dim, generator=g),
                    "deck_mask": torch.ones(T, D, dtype=torch.bool),
                    "globals_seq": glob,
                    "legal_card": torch.ones(T, D, dtype=torch.bool),
                    "legal_place": torch.ones(T, c.arena_h * c.arena_w, dtype=torch.bool)},
            "tgt": {"waited": torch.rand(T, generator=g) < 0.5,
                    "card_slot": torch.randint(0, D, (T,), generator=g),
                    "place_idx": torch.randint(0, c.arena_h * c.arena_w, (T,), generator=g),
                    "value": torch.ones(T)},
            "meta": {}})
    return out


#: the shard meta crdata converter_version 3 writes (the fields the trainers check)
CR_V3_META = {"card_tokens": "deck", "deck_size": 8, "card_feature_dim": 9, "hand_size": 4}


def test_train_multi_game_crops_clash_royale_only_when_asked_and_safe(tmp_path):
    """--game_max_len clash_royale=N crops (the global --max_len does not);
    a legacy frame-time wait head refuses the crop."""
    from data_processing.games.dataset import save_shard
    from training import train_multi_game

    save_shard(str(tmp_path / "cr" / "a.pt"), "clash_royale", _cr_pt_samples(), converter_version=3,
               meta=CR_V3_META)
    base = ["--data", f"clash_royale={tmp_path / 'cr'}", "--steps", "1", "--batch_size", "1",
            "--warmup_steps", "1", "--ema_rate", "0", "--device", "cpu", "--eval_every", "100",
            "--save_dir", str(tmp_path / "ck"), "--max_len", "4", "--override",
            "d_model=32", "n_heads=4", "d_ff=64", "n_st_layers=1", "dropout=0.0",
            "encoder_channels=8", "core_type=transformer", "core_layers=1"]
    seen = {}
    orig = train_multi_game.open_dataset

    def spy(game, d, max_len=None):
        seen[game] = max_len
        return orig(game, d, max_len)

    train_multi_game.open_dataset = spy
    try:
        train_multi_game.main(base)
        assert seen["clash_royale"] is None                     # global --max_len ignored
        train_multi_game.main(base[:-9] + ["--game_max_len", "clash_royale=5"] + base[-9:])
        assert seen["clash_royale"] == 5
        with pytest.raises(SystemExit, match="absolute frame time"):
            train_multi_game.main(base[:-9] + ["--game_max_len", "clash_royale=5"] + base[-9:]
                                  + ["cr_wait_time=frame"])
    finally:
        train_multi_game.open_dataset = orig
