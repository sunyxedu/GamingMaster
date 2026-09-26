"""Clash Royale information-leak audit: everything the network sees at a decision
must be information the acting player has at that moment.

Two leak classes, each with an invariance test:

* FUTURE (causality): the observation of frame t is a function of the record up
  to t.  Converter side: ``crdata`` truncates replays at t and rebuilds
  (``data_processing/games/clash_royale/crdata/src/leak_audit.rs``, run below
  through cargo, on synthetic matches and on the real replays in
  ``tests/fixtures/clash_royale/replays.jsonl``).  Loader side: ``expand`` is
  frame-wise, so frames < k never depend on frames >= k.  Model side: the
  network's outputs at t (incl. the wait head's match-clock context) do not
  depend on later frames.
* PRIVACY: a player's frames are invariant to the opponent's private
  information (deck beyond the cards already played, deck order / cycle,
  levels, forms, tower troop, the result).  Converter side (cargo test).  The
  opponent's elixir — not shown by the client — must be a deterministic
  function of public history; the cargo test pins it to an integrator that
  sees only the opponent's visible ``(tick, cost)`` events.

``KNOWN_FUTURE_LEAKS`` in ``leak_audit.rs`` is empty since crdata
converter_version 3: the hand is no longer inferred from the whole match (the
card about to be played was marked "certainly in hand"); frames carry the 8 own
deck cards with the causal belief P(in hand | own plays so far), and ability
events are attributed from the public record.  The one pinned leak left here is
the train-mode BatchNorm test below (the arena encoder, not the data).
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from data_processing.games.clash_royale.expand import (
    DECK_OBS_KEYS,
    HAND_OBS_KEYS,
    expand,
    obs_keys,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "clash_royale", "replays.jsonl")

# raw.* columns crdata writes per frame that are decision-time inputs (every one is
# causal and private-information free: leak_audit.rs); the rest (``ability`` /
# ``ability_card``: the frame window's own action) are labels
COMMON_INPUTS = frozenset({
    "tick", "time_s", "phase", "overtime", "elixir", "opp_elixir", "recent_side", "recent_card",
    "recent_cost", "recent_x", "recent_y", "recent_age"})
INPUT_COLUMNS = COMMON_INPUTS | {"deck_ids", "deck_form", "deck_kind", "deck_cost", "deck_in_hand",
                                 "deck_next"}
# converter_version <= 2 (a hand inferred from the whole match — future-dependent at
# the converter; still expanded so legacy checkpoints can be evaluated)
LEGACY_INPUT_COLUMNS = COMMON_INPUTS | {"hand_ids", "hand_form", "hand_kind", "hand_cost",
                                        "hand_prob", "next_cost"}
LABEL_COLUMNS = frozenset({"ability", "ability_card"})
# dataset-level constants written once per shard directory by crdata
STATIC_META = frozenset({"arena_channels", "card_feature_dim", "tau_ticks"})
SHARD_META = {"arena_channels": 16, "card_feature_dim": 9, "tau_ticks": 60.0}
LEGACY_SHARD_META = {"arena_channels": 16, "card_feature_dim": 8, "tau_ticks": 60.0}
SCHEMAS = {"deck": (DECK_OBS_KEYS, SHARD_META, INPUT_COLUMNS),
           "hand": (HAND_OBS_KEYS, LEGACY_SHARD_META, LEGACY_INPUT_COLUMNS)}


# ======================================================================
# converter (Rust): truncation / perturbation invariance on synthetic and
# real replays
# ======================================================================

@pytest.mark.skipif(not os.path.exists(CARGO), reason="cargo not installed")
def test_crdata_leak_audit():
    """``cargo test -p crdata leak_audit``: every input column is causal
    (``KNOWN_FUTURE_LEAKS`` is empty), the hand belief is the deal prior at kickoff
    and calibrated on the coming play (the old hindsight inference, rebuilt as a
    control, is flagged), ability events name only cards already played, a side's
    frames are invariant to the opponent's private information, and ``opp_elixir``
    is the public-history integral — on synthetic matches and on the real-replay
    fixture."""
    build = subprocess.run([CARGO, "test", "--release", "-q", "-p", "crdata", "--no-run"], cwd=WS,
                           capture_output=True, text=True, timeout=1800)
    if build.returncode != 0 and "could not compile `crdata`" not in build.stderr:
        pytest.skip(f"cargo cannot build the crdata tests here: {build.stderr[-400:]}")
    assert build.returncode == 0, build.stderr[-3000:]
    res = subprocess.run(
        [CARGO, "test", "--release", "-q", "-p", "crdata", "leak_audit", "--", "--nocapture"],
        cwd=WS, capture_output=True, text=True, timeout=1800)
    out = res.stdout + res.stderr
    assert res.returncode == 0, out[-6000:]
    assert "test result: ok" in out and " 0 failed" in out
    assert "leaking columns {}" in out
    if os.path.exists(FIXTURE):
        assert "fixture (" in out, "the real-replay fixture was not audited"


# ======================================================================
# loader: expand() is frame-wise and reads decision-time columns only
# ======================================================================

def _raw(T=40, K=8, seed=0, schema="deck"):
    rng = np.random.default_rng(seed)
    side = rng.choice(np.array([0, 1, 255], np.uint8), size=(T, K))
    r = {
        "tick": (np.arange(T) * 10).astype(np.int32),
        "time_s": (np.arange(T) * 0.5).astype(np.float32),
        "phase": rng.integers(0, 3, T).astype(np.uint8),
        "overtime": rng.integers(0, 2, T).astype(np.uint8),
        "elixir": rng.uniform(0, 10, T).astype(np.float32),
        "opp_elixir": rng.uniform(0, 10, T).astype(np.float32),
        "recent_side": side,
        "recent_card": rng.integers(1, 123, (T, K)).astype(np.uint16),
        "recent_cost": rng.integers(1, 10, (T, K)).astype(np.float32),
        "recent_x": np.where(side == 255, -1, rng.integers(-1, 18, (T, K))).astype(np.int8),
        "recent_y": np.where(side == 255, -1, rng.integers(-1, 32, (T, K))).astype(np.int8),
        "recent_age": np.where(side == 255, -1, rng.integers(0, 400, (T, K))).astype(np.int32),
        "ability": rng.integers(0, 2, T).astype(np.uint8),
        "ability_card": rng.integers(0, 123, T).astype(np.uint16),
    }
    if schema == "deck":
        deck = np.sort(rng.choice(np.arange(1, 123), 8, replace=False)).astype(np.uint16)
        p_hand = rng.choice(np.array([0.0, 0.5, 4 / 7, 1.0], np.float32), size=(T, 8))
        r.update({
            "deck_ids": np.tile(deck, (T, 1)),
            "deck_form": np.tile(rng.choice(np.array([0, 1, 200], np.uint8), size=8), (T, 1)),
            "deck_kind": np.tile(rng.integers(0, 4, 8).astype(np.uint8), (T, 1)),
            "deck_cost": np.tile(rng.integers(1, 10, 8).astype(np.float32), (T, 1)),
            "deck_in_hand": p_hand,
            "deck_next": rng.uniform(0, 1, (T, 8)).astype(np.float32),
        })
    else:
        r.update({
            "hand_ids": rng.integers(1, 123, (T, 4)).astype(np.uint16),
            "hand_form": rng.choice(np.array([0, 1, 200], np.uint8), size=(T, 4)),
            "hand_kind": rng.integers(0, 4, (T, 4)).astype(np.uint8),
            "hand_cost": rng.integers(1, 10, (T, 4)).astype(np.float32),
            "hand_prob": rng.uniform(0, 1, (T, 4)).astype(np.float32),
            "next_cost": rng.integers(1, 10, T).astype(np.float32),
        })
    return {k: torch.from_numpy(v) for k, v in r.items()}


def _assert_obs_equal(a, b, n, keys):
    for k in keys:
        assert torch.equal(a[k][:n], b[k][:n]), f"obs[{k!r}] of frames < {n} changed"


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
@pytest.mark.parametrize("k", [1, 7, 23, 39])
def test_expand_prefix_is_the_expansion_of_the_prefix(k, schema):
    """Truncating the record at frame k (every raw column) and expanding gives
    exactly the first k frames of the full expansion: no whole-sequence
    normaliser, length feature or cross-frame smoothing."""
    keys, meta, _ = SCHEMAS[schema]
    raw = _raw(schema=schema)
    full = expand(raw, meta)
    part = expand({c: v[:k] for c, v in raw.items()}, meta)
    assert set(full) == set(keys) == obs_keys(raw)
    assert all(part[o].shape[0] == k for o in keys)
    _assert_obs_equal(full, part, k, keys)


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_expand_ignores_future_frames(schema):
    keys, meta, _ = SCHEMAS[schema]
    raw, other = _raw(seed=1, schema=schema), _raw(seed=2, schema=schema)
    k = 17
    mixed = {c: torch.cat([raw[c][:k], other[c][k:]]) for c in raw}
    _assert_obs_equal(expand(raw, meta), expand(mixed, meta), k, keys)


class _Recording(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read = set()

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)


def test_expand_ignores_recent_card_ids():
    """``raw.recent_card`` does not reach the observation: only the cost of plays
    with a cell reaches the arena planes; ability events (x = y = -1) do not.  crdata
    converter_version 3 attributes ability events from the public record (the unique
    candidate the side has already played, else unknown — pinned by leak_audit.rs
    ``ability_card_is_resolved_from_public_plays``; converter_version 2 named
    RoyaleAPI's first candidate, possibly an opponent card never played), so the ids
    could now be wired in — with a schema bump, not silently."""
    for schema in SCHEMAS:
        keys, meta, _ = SCHEMAS[schema]
        raw = _raw(seed=3, schema=schema)
        other = dict(raw)
        other["recent_card"] = torch.from_numpy((raw["recent_card"].numpy() % 97 + 5).astype(np.uint16))
        assert not torch.equal(raw["recent_card"], other["recent_card"])
        a, b = expand(raw, meta), expand(other, meta)
        _assert_obs_equal(a, b, raw["elixir"].shape[0], keys)
        # an ability entry (no cell) with any cost leaves the arena planes untouched
        abil = {k: v.clone() for k, v in raw.items()}
        abil["recent_side"][:, 0] = 1
        abil["recent_x"][:, 0] = -1
        abil["recent_y"][:, 0] = -1
        abil2 = {k: v.clone() for k, v in abil.items()}
        abil2["recent_cost"][:, 0] = abil["recent_cost"][:, 0] + 3.0
        _assert_obs_equal(expand(abil, meta), expand(abil2, meta), raw["elixir"].shape[0], keys)


@pytest.mark.parametrize("schema", sorted(SCHEMAS))
def test_expand_reads_only_decision_time_inputs(schema):
    """expand() must not read the label columns (the frame window's own action) and
    only static, dataset-level shard meta — never per-sample facts."""
    _, meta0, inputs = SCHEMAS[schema]
    raw, meta = _Recording(_raw(schema=schema)), _Recording(meta0)
    expand(raw, meta)
    assert raw.read <= inputs, raw.read - inputs
    assert not raw.read & LABEL_COLUMNS
    assert meta.read <= STATIC_META, meta.read - STATIC_META


def test_deck_legality_is_the_belief_support_and_affordability():
    """A deck card is playable iff the causal belief still allows it (P(in hand) > 0)
    and its cost is covered at the frame; the placement mask opens the enemy half
    only when a PLAYABLE card is a spell."""
    raw = _raw(seed=4)
    obs = expand(raw, SHARD_META)
    p = raw["deck_in_hand"]
    affordable = raw["deck_cost"] <= raw["elixir"][:, None] + 1e-6
    assert torch.equal(obs["legal_card"], (p > 0) & affordable)
    assert (p == 0).any() and ((p == 0) & affordable).any()       # the test is live
    spell_ok = ((p > 0) & affordable & (raw["deck_kind"] == 2)).any(-1)
    assert torch.equal(obs["legal_place"].all(-1), spell_ok)
    assert obs["legal_place"][:, : 16 * 18].all()
    f = obs["deck_feats"]
    assert f.shape == (40, 8, 9)
    assert torch.equal(f[..., 7], p) and torch.equal(f[..., 8], raw["deck_next"])
    assert torch.equal(f[..., 6].bool(), affordable)
    exp_next = (raw["deck_next"].double() * raw["deck_cost"]).sum(-1).float() / 10.0
    torch.testing.assert_close(obs["globals_seq"][:, 5], exp_next)
    assert torch.equal(obs["deck_ids"], raw["deck_ids"].long()) and obs["deck_mask"].all()


# ======================================================================
# model: outputs at frame t ignore frames > t (Stage-1 temporal state and
# the wait head's match-clock decode context included)
# ======================================================================

def _net_and_obs(B=2, T=6, seed=1, card_tokens="deck"):
    from model.game_network import GameNetwork
    from model.games.clash_royale import CardPlacementDecoder, ClashRoyaleConfig, ClashRoyaleEncoder

    N, F = (8, 9) if card_tokens == "deck" else (4, 8)
    cfg = ClashRoyaleConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0, n_st_layers=1, core_type="trm",
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, halt_max_steps=2, encoder_channels=24,
        arena_h=8, arena_w=6, arena_channels=5, hand_size=4, n_cards=20, card_feature_dim=F,
        cr_global_features=6, cr_card_tokens=card_tokens)
    torch.manual_seed(0)
    net = GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))
    g = torch.Generator().manual_seed(seed)
    pre = "deck" if card_tokens == "deck" else "hand"
    obs = {
        "arena_seq": torch.randn(B, T, 5, 8, 6, generator=g),
        f"{pre}_ids": torch.randint(1, 20, (B, T, N), generator=g),
        f"{pre}_feats": torch.randn(B, T, N, F, generator=g),
        f"{pre}_mask": torch.ones(B, T, N, dtype=torch.bool),
        "globals_seq": torch.rand(B, T, 6, generator=g),
    }
    labels = {"card_slot": torch.randint(0, N, (B, T), generator=g)}
    return net, obs, labels


def _future_perturbed(obs, k):
    out = {key: v.clone() for key, v in obs.items()}
    for key in out:
        if key.endswith("_ids"):
            out[key][:, k:] = (out[key][:, k:] + 7) % 19 + 1
        elif out[key].is_floating_point():      # incl. globals_seq[..., 2] = the match clock
            out[key][:, k:] = out[key][:, k:] * 3.0 + 1.0
    return out


def _max_past_change(net, obs, labels, k):
    with torch.no_grad():
        a = net(obs, labels=labels, act_steps=2)
        b = net(_future_perturbed(obs, k), labels=labels, act_steps=2)
    return {key: float((a[key][:, :k] - b[key][:, :k]).abs().max())
            for key in ("wait_logits", "card_logits", "placement_logits", "value")}


@pytest.mark.parametrize("card_tokens", ["deck", "hand"])
def test_network_outputs_are_causal_in_time_eval(card_tokens):
    net, obs, labels = _net_and_obs(card_tokens=card_tokens)
    net.eval()
    change = _max_past_change(net, obs, labels, k=4)
    assert all(v == 0.0 for v in change.values()), change


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN minor leak: the arena encoder's BatchNorm2d normalises frame t with batch statistics "
    "over all B*T frames in train mode (GameNetwork.forward passes whole sequences), so IL/PPO "
    "forward passes let later frames of the sequence shift frame t's features; eval / step() use "
    "running stats. Fix: per-frame normalisation (GroupNorm/LayerNorm) or frozen BN statistics."))
def test_network_outputs_are_causal_in_time_train_mode():
    net, obs, labels = _net_and_obs()
    net.train()
    change = _max_past_change(net, obs, labels, k=4)
    assert all(v == 0.0 for v in change.values()), change


def test_train_mode_time_mixing_comes_only_from_batchnorm():
    """Scope of the xfail above: with every BatchNorm layer on running statistics and
    the rest of the network in train mode, outputs at t ignore frames > t again, so
    the arena encoder's BatchNorm is the only non-causal path in training."""
    net, obs, labels = _net_and_obs()
    net.train()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns and all(n.startswith("obs_encoder.arena_encoder") for n, m in net.named_modules() if m in bns)
    for m in bns:
        m.eval()
    change = _max_past_change(net, obs, labels, k=4)
    assert all(v == 0.0 for v in change.values()), change
