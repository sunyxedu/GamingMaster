"""Clash Royale causal hand belief (crdata converter_version 3) and the deck-token
card head.

* ``belief.py`` (what a live bot runs on its own plays) is the converter's
  posterior: unit checks against the closed form, and bit-for-bit parity with
  ``crdata convert`` on the committed real replays.
* Legacy ``cr_card_tokens="hand"`` networks compute exactly what they computed
  before the deck tokens existed (reference outputs pinned from the previous
  adapter code), and legacy config dicts rebuild them.
* Deck tokens carry no slot semantics: permuting them permutes the card logits and
  changes nothing else.
* The trainers wire the schema: shard meta ↔ config checks, the schema-freeze
  warning for converter_version 2 shards, and the legality masks.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import zlib
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_processing.games.clash_royale.belief import (
    CYCLED,
    INCONSISTENT,
    NOT_IN_DECK,
    CycleBelief,
    deck_token_order,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WS = os.path.join(ROOT, "data_processing")
CARGO = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "clash_royale", "replays.jsonl")
CATALOG = os.path.join(WS, "games", "clash_royale", "catalog", "cards.json")
F32 = lambda x: np.float32(x)   # noqa: E731


# ======================================================================
# the belief
# ======================================================================

def test_deal_prior_then_the_first_play():
    b = CycleBelief([11, 12, 13, 14, 15, 16, 17, 18])
    assert b.n_consistent == 1680 and not b.determined
    assert (b.in_hand == F32(0.5)).all() and (b.next == F32(0.125)).all()
    assert b.observe(13) == CYCLED and b.n_consistent == 840
    assert b.in_hand[2] == 0 and b.next[2] == 0
    others = [0, 1, 3, 4, 5, 6, 7]
    assert (b.in_hand[others] == F32(4 / 7)).all() and (b.next[others] == F32(1 / 7)).all()
    assert b.observe(99) == NOT_IN_DECK and (b.plays, b.skipped) == (1, 1)
    before = b.in_hand
    assert b.observe(13) == INCONSISTENT          # needs 4 more plays to come back
    assert (b.in_hand == before).all() and b.n_consistent == 840
    with pytest.raises(ValueError):
        CycleBelief([1, 2, 3])


def _random_game(rng: random.Random, n_plays: int):
    deck = list(range(1, 9))
    rng.shuffle(deck)
    order = deck[:]
    rng.shuffle(order)
    hand, queue = order[:4], order[4:]
    plays, truth = [], []
    for _ in range(n_plays):
        truth.append((list(hand), queue[0]))
        c = hand[rng.randrange(4)]
        hand[hand.index(c)] = queue.pop(0)
        queue.append(c)
        plays.append(c)
    truth.append((list(hand), queue[0]))
    return deck, plays, truth


def test_matches_the_closed_form_and_never_excludes_the_truth():
    """After m <= 3 distinct plays: played cards out, the rest 4/(8-m) in hand and
    1/(8-m) next; from the 4th play on: exactly the true hand and next card."""
    rng = random.Random(0)
    for _ in range(80):
        deck, plays, truth = _random_game(rng, rng.randrange(25))
        b = CycleBelief(deck)
        for k in range(len(plays) + 1):
            hand, nxt = truth[k]
            for i, c in enumerate(deck):
                if k >= 4:
                    want = (float(c in hand), float(c == nxt))
                elif c in plays[:k]:
                    want = (0.0, 0.0)
                else:
                    want = (4 / (8 - k), 1 / (8 - k))
                got = (b.in_hand[i], b.next[i])
                assert got == (F32(want[0]), F32(want[1])), (deck, plays[:k], c)
                if c in hand:
                    assert b.in_hand[i] > 0
            assert b.determined == (k >= 4)
            if k < len(plays):
                assert b.observe(plays[k]) == CYCLED


def test_repeated_ids_and_token_order():
    b = CycleBelief([1, 2, 3, 4, 5, 6, 99, 99])
    assert b.observe(99) == CYCLED and b.observe(99) == CYCLED
    assert (b.in_hand[6], b.in_hand[7]) == (0, 0)
    assert abs(float(b.in_hand.sum()) - 4) < 1e-5 and abs(float(b.next.sum()) - 1) < 1e-6
    assert deck_token_order([40, 3, 22, 3], [0, 1, 0, 0]) == [3, 1, 2, 0]


# ======================================================================
# parity with the converter on the real replays
# ======================================================================

@pytest.fixture(scope="module")
def fixture_shards(tmp_path_factory):
    if not os.path.exists(CARGO):
        pytest.skip("cargo not installed")
    b = subprocess.run([CARGO, "build", "--release", "-q", "-p", "crdata"], cwd=WS,
                       capture_output=True, text=True)
    if b.returncode != 0:
        pytest.skip(f"cargo build failed: {b.stderr[-500:]}")
    out = tmp_path_factory.mktemp("cr") / "shards"
    # every side, including the two the default elixir-violation filter drops
    binary = os.path.join(WS, "target", "release", "crdata")
    res = subprocess.run([binary, "convert", "--input", FIXTURE, "--out", str(out),
                          "--max-elixir-violations", "1000000"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr[-2000:]
    return str(out)


def _resolve():
    cards = json.load(open(CATALOG))
    by_key = {c["key"]: c["id"] for c in cards["cards"]}

    def resolve(key: str):
        if key.endswith("-hero"):
            base, form = key[:-5], 200
        else:
            base, _, lvl = key.rpartition("-ev")
            base, form = (base, int(lvl)) if base and lvl.isdigit() else (key, 0)
        return by_key.get(base, cards["id_unknown"]), form
    return resolve


def _own_plays(payload, side, resolve):
    """(tick, card id) of `side`'s plays as crdata's parser keeps them (a cell inside
    the arena), in tick order."""
    out = []
    for e in payload["events"]:
        if e["side"] != side or e["kind"] != "play_card":
            continue
        key = (e.get("deck_card_key_candidates") or [None])[0] or e.get("card_key")
        c = e.get("coordinates")
        if key is None or c is None or c.get("inside_18x32_arena") is False:
            continue
        out.append((e["replay_tick_20hz"], resolve(key)[0]))
    return sorted(out, key=lambda p: p[0])


def test_python_belief_matches_the_converter_bit_for_bit(fixture_shards):
    """For every side of every real fixture replay: the deck tokens are the deck in
    deck_token_order and, frame by frame, CycleBelief fed the own plays strictly
    before the frame reproduces raw.deck_in_hand / raw.deck_next exactly."""
    from data_processing.games.dataset import GmShard, find_gm_shards

    resolve = _resolve()
    replays = {}
    for line in open(FIXTURE):
        if line.strip():
            row = json.loads(line)
            replays[row["tag"]] = row["payload"]
    n_sides = n_frames = n_determined = n_labels = 0
    for d in find_gm_shards(fixture_shards):
        sh = GmShard(d)
        assert sh.index["converter_version"] == 3
        arr = sh.arrays()
        for info in sh.index["samples"]:
            a, n = info["start"], info["length"]
            meta = info["meta"]
            payload = replays[meta["tag"]]
            team = "team" if meta["side"] == "team" else "opponent"
            deck = [resolve(c["card_key"]) for c in payload["battle"][team]["players"][0]["deck"]]
            order = deck_token_order([i for i, _ in deck], [f for _, f in deck])
            tokens = [deck[i][0] for i in order]
            assert np.array_equal(np.asarray(arr["raw.deck_ids"][a:a + n]), np.tile(tokens, (n, 1)))
            plays = _own_plays(payload, meta["side"], resolve)
            belief, j = CycleBelief(tokens), 0
            got_h = np.asarray(arr["raw.deck_in_hand"][a:a + n])
            got_n = np.asarray(arr["raw.deck_next"][a:a + n])
            slots = np.asarray(arr["tgt.card_slot"][a:a + n])
            for f in range(n):
                while j < len(plays) and plays[j][0] < f * 10:
                    assert belief.observe(plays[j][1]) != INCONSISTENT
                    j += 1
                where = (meta["tag"], meta["side"], f)
                assert got_h[f].tobytes() == belief.in_hand.tobytes(), where
                assert got_n[f].tobytes() == belief.next.tobytes(), where
                if slots[f] >= 0:
                    # the label is a card the belief (hence legal_card) lets the player hold
                    assert tokens[slots[f]] == plays[j][1] and belief.in_hand[slots[f]] > 0, where
                    n_labels += 1
                n_determined += belief.determined
            n_sides += 1
            n_frames += n
    assert n_sides == 26 and n_frames > 10_000 and 0 < n_determined < n_frames and n_labels > 500


# ======================================================================
# model: legacy hand networks unchanged, deck tokens without order semantics
# ======================================================================

_LEGACY_BASE = dict(d_model=32, n_heads=4, d_ff=48, dropout=0.0, n_st_layers=1, encoder_channels=8,
                    arena_h=8, arena_w=6, arena_channels=5, hand_size=4, n_cards=20,
                    card_feature_dim=8, cr_global_features=6)
_LEGACY_CORES = {
    "trm/frame": dict(core_type="trm", trm_H_cycles=2, trm_L_cycles=1, trm_block_layers=1,
                      halt_max_steps=2, cr_wait_time="frame"),
    "transformer/match": dict(core_type="transformer", core_layers=1, cr_wait_time="match"),
}
#: outputs of the adapter BEFORE the deck tokens existed (model/games/clash_royale.py
#: at 7dd8315), same weights / inputs / float64: the legacy path must not move
_LEGACY_REFERENCE = {
    "trm/frame": dict(
        sha="6ee5c83c4e4317e3bd5a31ab5a69a065c4007ef5b61515015744cd1606df1e33",
        wait=[-0.346817030584, -0.134034038925, -0.585706922539, -0.343069824123, -0.127882609426,
              -0.559740119432],
        card=[-0.771074525097, -0.764871651919, -0.762335240137, -0.770773741967, -0.820547916045,
              -0.818602175777, -0.817206770773, -0.824219411632],
        value=[0.017050264987, -0.015763895502, -0.038044194432, 0.013096428181, 0.01130784398,
               0.018048663222],
        place_sum=-464.123609348),
    "transformer/match": dict(
        sha="0f752646961c0fbde5967c0007468351ca9072f6ed1a10ebd68fa1c165e86b0c",
        wait=[-0.340295086705, -1.616823063248, -0.222495368267, -1.242955986986, -2.271078716286,
              -1.503108923418],
        card=[-1.249488862029, -1.234806707577, -1.24377423524, -1.247133130507, -1.231906276672,
              -1.239913718321, -1.234427549244, -1.243077707001],
        value=[0.384292031965, 0.384268666525, 0.387048330052, 0.383573878594, 0.383112121977,
               0.383403178749],
        place_sum=-433.020360418),
}


def _ints(shape, seed, lo=-1000, hi=1001):
    """Integer draws: platform-independent (unlike float sampling kernels)."""
    return torch.randint(lo, hi, tuple(shape), generator=torch.Generator().manual_seed(seed))


def _fill_by_name(net):
    for name, t in sorted(net.state_dict().items()):
        if t.is_floating_point():
            v = _ints(t.shape, zlib.crc32(name.encode())).double()
            with torch.no_grad():
                t.copy_(1.0 + (v + 1000) / 2000.0 if name.endswith("running_var") else v / 2500.0)


@pytest.mark.parametrize("case", sorted(_LEGACY_REFERENCE))
def test_legacy_hand_network_computes_what_it_computed(case):
    from model.config import config_from_dict
    from model.game_network import GameNetwork
    from model.games.clash_royale import CardPlacementDecoder, ClashRoyaleConfig, ClashRoyaleEncoder

    # a config dict saved before the deck tokens existed: every field of that time,
    # but no cr_card_tokens / deck_size
    saved = asdict(ClashRoyaleConfig(**_LEGACY_BASE, **_LEGACY_CORES[case]))
    for k in ("cr_card_tokens", "deck_size"):
        saved.pop(k)
    cfg = config_from_dict(saved, ClashRoyaleConfig)
    assert (cfg.cr_card_tokens, cfg.card_feature_dim) == ("hand", 8)
    net = GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg)).double()
    keys = sorted(f"{k}:{tuple(v.shape)}" for k, v in net.state_dict().items())
    ref = _LEGACY_REFERENCE[case]
    sha = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    assert sha == ref["sha"], "legacy state-dict layout changed"
    _fill_by_name(net)
    net.eval()
    B, T = 2, 3
    obs = {"arena_seq": _ints((B, T, 5, 8, 6), 1).double() / 1000.0,
           "hand_ids": _ints((B, T, 4), 2, 0, 20),
           "hand_feats": _ints((B, T, 4, 8), 3).double() / 1000.0,
           "hand_mask": _ints((B, T, 4), 4, 0, 5) > 0,
           "globals_seq": _ints((B, T, 6), 5, 0, 1001).double() / 1000.0}
    obs["hand_mask"][..., 0] = True
    with torch.no_grad():
        o = net(obs, labels={"card_slot": _ints((B, T), 6, 0, 4)}, act_steps=2)
    close = dict(rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(o["wait_logits"].flatten().numpy(), ref["wait"], **close)
    np.testing.assert_allclose(o["card_logits"].flatten()[:8].numpy(), ref["card"], **close)
    np.testing.assert_allclose(o["value"].flatten().numpy(), ref["value"], **close)
    np.testing.assert_allclose(float(o["placement_logits"].sum()), ref["place_sum"], rtol=1e-9)


def _deck_net(**kw):
    from model.games import build_game_network

    torch.manual_seed(0)
    arch = dict(d_model=32, n_heads=4, d_ff=48, n_st_layers=1, dropout=0.0, encoder_channels=8,
                arena_h=8, arena_w=6, arena_channels=5, n_cards=20, core_type="transformer",
                core_layers=1)
    return build_game_network("clash_royale", **{**arch, **kw})


#: cores whose computation treats tokens alike (no per-index state): exactly equivariant
#: to the deck order.  The recurrent-depth core's fixed inference state s_0 is one draw
#: per token index, so it is not; the order is canonical instead (by card id, the same in
#: crdata and belief.deck_token_order — test_python_belief_matches_the_converter_bit_for_bit).
_EQUIVARIANT_CORES = {
    "transformer": dict(core_type="transformer", core_layers=1),
    "trm": dict(core_type="trm", trm_H_cycles=2, trm_L_cycles=1, trm_block_layers=1,
                halt_max_steps=2),
}


def _deck_obs(B=2, T=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    decks = torch.stack([torch.randperm(19, generator=g)[:8] + 1 for _ in range(B)])
    return {"arena_seq": torch.randn(B, T, 5, 8, 6, generator=g),
            "deck_ids": decks[:, None].expand(B, T, 8).clone(),
            "deck_feats": torch.rand(B, T, 8, 9, generator=g),
            "deck_mask": torch.ones(B, T, 8, dtype=torch.bool),
            "globals_seq": torch.rand(B, T, 6, generator=g)}


@pytest.mark.parametrize("core", sorted(_EQUIVARIANT_CORES))
@pytest.mark.parametrize("mixer", ["attention", "recurrent"])
def test_deck_tokens_carry_no_order_semantics(mixer, core):
    """The deck index is only a pointer: permuting the deck tokens permutes the card
    logits (the label follows its card) and leaves wait / placement / value alone."""
    net = _deck_net(temporal_mixer=mixer, **_EQUIVARIANT_CORES[core]).eval()
    assert net.cfg.cr_card_tokens == "deck" and net.obs_encoder.n_tokens == 48 + 8
    obs = _deck_obs()
    labels = torch.randint(0, 8, (2, 4), generator=torch.Generator().manual_seed(1))
    perm = torch.tensor([5, 2, 7, 0, 1, 6, 3, 4])
    inv = torch.argsort(perm)
    pobs = {k: (v[:, :, perm] if k.startswith("deck_") else v) for k, v in obs.items()}
    with torch.no_grad():
        a = net(obs, labels={"card_slot": labels}, act_steps=2)
        b = net(pobs, labels={"card_slot": inv[labels]}, act_steps=2)
    close = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(b["card_logits"], a["card_logits"][:, :, perm], **close)
    for k in ("wait_logits", "placement_logits", "value"):
        torch.testing.assert_close(b[k], a[k], **close)


def test_deck_network_trains_on_its_legal_masks():
    """IL forward with the obs-derived masks: self-sampled cards (label -1) are
    always playable ones; the card head has deck_size classes."""
    from training.train_il_game import LEGAL_FROM_OBS

    net = _deck_net().eval()
    obs = _deck_obs()
    obs["legal_card"] = torch.zeros(2, 4, 8, dtype=torch.bool)
    obs["legal_card"][..., 3] = True
    obs["legal_card"][1, :, 6] = True
    obs["legal_place"] = torch.ones(2, 4, 48, dtype=torch.bool)
    legal = LEGAL_FROM_OBS["clash_royale"](obs)
    assert set(legal) == {"card", "placement"}
    with torch.no_grad():
        out = net(obs, legal=legal, labels={"card_slot": torch.full((2, 4), -1)}, act_steps=1)
    assert out["card_logits"].shape == (2, 4, 8)
    assert legal["card"].gather(-1, out["card_slot"][..., None]).all()
    hand_obs = {"hand_ids": torch.zeros(1, 1, 4, dtype=torch.long),
                "legal_card": torch.ones(1, 1, 4)}
    assert LEGAL_FROM_OBS["clash_royale"](hand_obs) is None       # legacy trainers passed no masks


# ======================================================================
# trainer glue
# ======================================================================

def test_meta_arch_and_schema_freeze_hold_the_generations_apart():
    from training.train_il_game import check_meta_arch, schema_freeze_warning

    v3 = {"arena_h": 32, "arena_w": 18, "arena_channels": 16, "hand_size": 4, "n_cards": 124,
          "card_tokens": "deck", "deck_size": 8, "card_feature_dim": 9, "cr_global_features": 6}
    v2 = {k: v for k, v in v3.items() if k not in ("card_tokens", "deck_size")}
    v2["card_feature_dim"] = 8
    from model.games.clash_royale import ClashRoyaleConfig

    fresh = ClashRoyaleConfig()
    legacy = ClashRoyaleConfig(cr_card_tokens="hand", card_feature_dim=8)
    check_meta_arch("clash_royale", fresh, v3)
    check_meta_arch("clash_royale", legacy, v2)
    with pytest.raises(ValueError, match="card_tokens=deck but the model has cr_card_tokens=hand"):
        check_meta_arch("clash_royale", legacy, {**v3, "card_feature_dim": 8})
    with pytest.raises(ValueError, match=r"card_tokens=hand \(implied"):
        check_meta_arch("clash_royale", ClashRoyaleConfig(card_feature_dim=8), v2)
    msg = schema_freeze_warning("clash_royale", SimpleNamespace(converter_version=2))
    assert msg.startswith("clash_royale shards have converter_version 2 < 3")
    assert "whole match" in msg
    assert schema_freeze_warning("clash_royale", SimpleNamespace(converter_version=3)) is None
