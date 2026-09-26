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

Known leaks are pinned (they fail once fixed, so the pin gets removed):
``KNOWN_FUTURE_LEAKS`` in ``leak_audit.rs`` (the hand-cycle inference uses the
whole match) and the train-mode BatchNorm test below.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest
import torch

from data_processing.games.clash_royale.expand import OBS_KEYS, expand

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "clash_royale", "replays.jsonl")

# raw.* columns crdata writes per frame that are decision-time inputs; the rest
# (``ability`` / ``ability_card``: the frame window's own action) are labels
INPUT_COLUMNS = frozenset({
    "tick", "time_s", "phase", "overtime", "elixir", "opp_elixir", "hand_ids", "hand_form",
    "hand_kind", "hand_cost", "hand_prob", "next_cost", "recent_side", "recent_card", "recent_cost",
    "recent_x", "recent_y", "recent_age"})
LABEL_COLUMNS = frozenset({"ability", "ability_card"})
# dataset-level constants written once per shard directory by crdata
STATIC_META = frozenset({"arena_channels", "card_feature_dim", "tau_ticks"})
SHARD_META = {"arena_channels": 16, "card_feature_dim": 8, "tau_ticks": 60.0}


# ======================================================================
# converter (Rust): truncation / perturbation invariance on synthetic and
# real replays
# ======================================================================

@pytest.mark.skipif(not os.path.exists(CARGO), reason="cargo not installed")
def test_crdata_leak_audit():
    """``cargo test -p crdata leak_audit``: public columns are causal, the pinned
    hand-cycle columns are the only future-dependent ones, a side's frames are
    invariant to the opponent's private information, and ``opp_elixir`` is the
    public-history integral — on synthetic matches and on the real-replay fixture."""
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
    if os.path.exists(FIXTURE):
        assert "fixture (" in out, "the real-replay fixture was not audited"


# ======================================================================
# loader: expand() is frame-wise and reads decision-time columns only
# ======================================================================

def _raw(T=40, K=8, seed=0):
    rng = np.random.default_rng(seed)
    side = rng.choice(np.array([0, 1, 255], np.uint8), size=(T, K))
    r = {
        "tick": (np.arange(T) * 10).astype(np.int32),
        "time_s": (np.arange(T) * 0.5).astype(np.float32),
        "phase": rng.integers(0, 3, T).astype(np.uint8),
        "overtime": rng.integers(0, 2, T).astype(np.uint8),
        "elixir": rng.uniform(0, 10, T).astype(np.float32),
        "opp_elixir": rng.uniform(0, 10, T).astype(np.float32),
        "hand_ids": rng.integers(1, 123, (T, 4)).astype(np.uint16),
        "hand_form": rng.choice(np.array([0, 1, 200], np.uint8), size=(T, 4)),
        "hand_kind": rng.integers(0, 4, (T, 4)).astype(np.uint8),
        "hand_cost": rng.integers(1, 10, (T, 4)).astype(np.float32),
        "hand_prob": rng.uniform(0, 1, (T, 4)).astype(np.float32),
        "next_cost": rng.integers(1, 10, T).astype(np.float32),
        "recent_side": side,
        "recent_card": rng.integers(1, 123, (T, K)).astype(np.uint16),
        "recent_cost": rng.integers(1, 10, (T, K)).astype(np.float32),
        "recent_x": np.where(side == 255, -1, rng.integers(-1, 18, (T, K))).astype(np.int8),
        "recent_y": np.where(side == 255, -1, rng.integers(-1, 32, (T, K))).astype(np.int8),
        "recent_age": np.where(side == 255, -1, rng.integers(0, 400, (T, K))).astype(np.int32),
        "ability": rng.integers(0, 2, T).astype(np.uint8),
        "ability_card": rng.integers(0, 123, T).astype(np.uint16),
    }
    return {k: torch.from_numpy(v) for k, v in r.items()}


def _assert_obs_equal(a, b, n):
    for k in OBS_KEYS:
        assert torch.equal(a[k][:n], b[k][:n]), f"obs[{k!r}] of frames < {n} changed"


@pytest.mark.parametrize("k", [1, 7, 23, 39])
def test_expand_prefix_is_the_expansion_of_the_prefix(k):
    """Truncating the record at frame k (every raw column) and expanding gives
    exactly the first k frames of the full expansion: no whole-sequence
    normaliser, length feature or cross-frame smoothing."""
    raw = _raw()
    full = expand(raw, SHARD_META)
    part = expand({c: v[:k] for c, v in raw.items()}, SHARD_META)
    assert all(part[o].shape[0] == k for o in OBS_KEYS)
    _assert_obs_equal(full, part, k)


def test_expand_ignores_future_frames():
    raw, other = _raw(seed=1), _raw(seed=2)
    k = 17
    mixed = {c: torch.cat([raw[c][:k], other[c][k:]]) for c in raw}
    _assert_obs_equal(expand(raw, SHARD_META), expand(mixed, SHARD_META), k)


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


def test_expand_reads_only_decision_time_inputs():
    """expand() must not read the label columns (the frame window's own action) and
    only static, dataset-level shard meta — never per-sample facts."""
    raw, meta = _Recording(_raw()), _Recording(SHARD_META)
    expand(raw, meta)
    assert raw.read <= INPUT_COLUMNS, raw.read - INPUT_COLUMNS
    assert not raw.read & LABEL_COLUMNS
    assert meta.read <= STATIC_META, meta.read - STATIC_META


# ======================================================================
# model: outputs at frame t ignore frames > t (Stage-1 temporal state and
# the wait head's match-clock decode context included)
# ======================================================================

def _net_and_obs(B=2, T=6, seed=1):
    from model.game_network import GameNetwork
    from model.games.clash_royale import CardPlacementDecoder, ClashRoyaleConfig, ClashRoyaleEncoder

    cfg = ClashRoyaleConfig(
        d_model=48, n_heads=4, d_ff=64, dropout=0.0, n_st_layers=1, core_type="trm",
        trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1, halt_max_steps=2, encoder_channels=24,
        arena_h=8, arena_w=6, arena_channels=5, hand_size=4, n_cards=20, card_feature_dim=8,
        cr_global_features=6)
    torch.manual_seed(0)
    net = GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))
    g = torch.Generator().manual_seed(seed)
    obs = {
        "arena_seq": torch.randn(B, T, 5, 8, 6, generator=g),
        "hand_ids": torch.randint(1, 20, (B, T, 4), generator=g),
        "hand_feats": torch.randn(B, T, 4, 8, generator=g),
        "hand_mask": torch.ones(B, T, 4, dtype=torch.bool),
        "globals_seq": torch.rand(B, T, 6, generator=g),
    }
    labels = {"card_slot": torch.randint(0, 4, (B, T), generator=g)}
    return net, obs, labels


def _future_perturbed(obs, k):
    out = {key: v.clone() for key, v in obs.items()}
    for key in ("arena_seq", "hand_feats", "globals_seq"):
        out[key][:, k:] = out[key][:, k:] * 3.0 + 1.0      # globals_seq[..., 2] = the match clock
    out["hand_ids"][:, k:] = (out["hand_ids"][:, k:] + 7) % 19 + 1
    return out


def _max_past_change(net, obs, labels, k):
    with torch.no_grad():
        a = net(obs, labels=labels, act_steps=2)
        b = net(_future_perturbed(obs, k), labels=labels, act_steps=2)
    return {key: float((a[key][:, :k] - b[key][:, :k]).abs().max())
            for key in ("wait_logits", "card_logits", "placement_logits", "value")}


def test_network_outputs_are_causal_in_time_eval():
    net, obs, labels = _net_and_obs()
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
