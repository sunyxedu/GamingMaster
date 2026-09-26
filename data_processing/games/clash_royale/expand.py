"""Load-time expansion of Clash Royale gm-shards into the adapter's observation.

The Rust converter (`crdata convert`) stores compact per-frame fields (`raw.*`);
materialising the dense arena planes on disk would cost ~36 KB per frame, so they
are built here when a sample is loaded (`GameShardDataset` calls
:func:`expand` for game ``clash_royale``).  Every output row t is a function of
the raw row t alone (and of static shard meta), and every raw input column is
information the player has at frame t (``tests/test_clash_royale_leaks.py``).

Observation (own perspective, row 0 = own back line) of converter_version >= 3
shards — ``raw.deck_*``, the model's ``cr_card_tokens="deck"``:

    arena_seq   [T, C, 32, 18]  0 own deployments exp(-age/tau)   1 opponent deployments
                                2 own deployment cost/10          3 opponent deployment cost/10
                                4 own half   5 enemy half   6 river   7 own elixir/10
                                8 opp elixir/10   9 time/300 s   10 double   11 triple
                                12 overtime   13.. reserved (0)
    deck_ids    [T, 8] long     the 8 own deck cards, token order (by card id)
    deck_feats  [T, 8, 9]       cost/10, troop, building, spell, evolved, hero, affordable,
                                P(in hand | own plays so far), P(next card | own plays so far)
    deck_mask   [T, 8] bool     True = a deck card
    globals_seq [T, 6]          elixir/10, opp/10, time/300, phase/2, overtime,
                                expected next-card cost/10 under the belief
    legal_card  [T, 8] bool     playable: P(in hand) > 0 and affordable
    legal_place [T, 576] bool   own half; whole arena when a playable card is a spell

The hand belief (`raw.deck_in_hand`, `raw.deck_next`) is crdata's causal
posterior over the hand cycle given the own plays before the frame
(`belief.py` is its Python twin, which a live bot uses to build the same
inputs).  The card head points over the deck tokens: label ``card_slot`` = deck
index of the played card.

converter_version <= 2 shards (``raw.hand_*``: a hand inferred from the WHOLE
match, which leaks the coming play — see crdata's ``leak_audit.rs``) still expand,
unchanged, to the legacy observation of ``cr_card_tokens="hand"`` checkpoints, so
those can be evaluated on the data they were trained on:

    hand_ids    [T, 4] long     hand_feats [T, 4, 8]  cost/10, troop, building, spell,
                                evolved, hero, affordable, P(card really in hand)
    hand_mask   [T, 4] bool     globals_seq [T, 6]  ... next-card cost/10
    legal_card  [T, 4] bool     affordable
    legal_place [T, 576] bool   own half; whole arena when the hand holds a spell

Deployment planes stand in for unit positions, which the replays do not carry.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Iterable

import numpy as np
import torch

GRID_H, GRID_W = 32, 18
#: deck-token features written by :func:`expand` (crdata shard meta card_feature_dim)
DECK_FEATURES = 9
DECK_OBS_KEYS = ("arena_seq", "deck_ids", "deck_feats", "deck_mask", "globals_seq", "legal_card",
                 "legal_place")
#: the observation of converter_version <= 2 shards (legacy hand tokens)
HAND_OBS_KEYS = ("arena_seq", "hand_ids", "hand_feats", "hand_mask", "globals_seq", "legal_card",
                 "legal_place")
#: the observation of the current converter's shards
OBS_KEYS = DECK_OBS_KEYS


def obs_keys(raw_names: Iterable[str]) -> FrozenSet[str]:
    """The observation keys :func:`expand` builds from a shard with these raw columns."""
    return frozenset(DECK_OBS_KEYS if "deck_ids" in set(raw_names) else HAND_OBS_KEYS)


def _np(v) -> np.ndarray:
    return v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v)


def _arena(raw: Dict, C: int, tau: float, elixir, opp_elixir, time_s, phase,
           overtime) -> np.ndarray:
    T = elixir.shape[0]
    arena = np.zeros((T, C, GRID_H, GRID_W), dtype=np.float32)
    side = _np(raw["recent_side"])
    x = _np(raw["recent_x"]).astype(np.int64)
    y = _np(raw["recent_y"]).astype(np.int64)
    age = _np(raw["recent_age"]).astype(np.float32)
    cost = _np(raw["recent_cost"]).astype(np.float32) / 10.0
    t_idx = np.arange(T)
    # oldest first, so the newest play at a cell writes its cost plane last
    for k in range(side.shape[1] - 1, -1, -1):
        ok = (side[:, k] != 255) & (x[:, k] >= 0) & (y[:, k] >= 0)
        if not ok.any():
            continue
        tt, yy, xx = t_idx[ok], y[ok, k], x[ok, k]
        plane = np.where(side[ok, k] == 0, 0, 1)
        w = np.exp(-age[ok, k] / tau).astype(np.float32)
        arena[tt, plane, yy, xx] = np.maximum(arena[tt, plane, yy, xx], w)
        arena[tt, plane + 2, yy, xx] = cost[ok, k]
    arena[:, 4, :16, :] = 1.0
    arena[:, 5, 16:, :] = 1.0
    arena[:, 6, 15:17, :] = 1.0
    arena[:, 7] = (elixir / 10.0)[:, None, None]
    arena[:, 8] = (opp_elixir / 10.0)[:, None, None]
    arena[:, 9] = (time_s / 300.0)[:, None, None]
    arena[:, 10] = (phase >= 1)[:, None, None]
    arena[:, 11] = (phase >= 2)[:, None, None]
    arena[:, 12] = overtime[:, None, None]
    return arena


def expand(raw: Dict, shard_meta: Dict) -> Dict[str, torch.Tensor]:
    """``raw.*`` columns of one sample → the adapter observation (the deck-token
    schema, or the legacy hand schema for converter_version <= 2 shards)."""
    C = int(shard_meta.get("arena_channels", 16))
    F = int(shard_meta.get("card_feature_dim", 8))
    tau = float(shard_meta.get("tau_ticks", 60.0))
    elixir = _np(raw["elixir"]).astype(np.float32)
    opp_elixir = _np(raw["opp_elixir"]).astype(np.float32)
    time_s = _np(raw["time_s"]).astype(np.float32)
    phase = _np(raw["phase"]).astype(np.float32)
    overtime = _np(raw["overtime"]).astype(np.float32)
    arena = _arena(raw, C, tau, elixir, opp_elixir, time_s, phase, overtime)
    if "deck_ids" in raw:
        return _deck_obs(raw, F, arena, elixir, opp_elixir, time_s, phase, overtime)
    return _hand_obs_legacy(raw, F, arena, elixir, opp_elixir, time_s, phase, overtime)


def _deck_obs(raw, F, arena, elixir, opp_elixir, time_s, phase,
              overtime) -> Dict[str, torch.Tensor]:
    if F < DECK_FEATURES:
        raise ValueError(f"deck tokens have {DECK_FEATURES} features, shard meta "
                         f"card_feature_dim={F}")
    T = elixir.shape[0]
    ids = _np(raw["deck_ids"]).astype(np.int64)
    kind = _np(raw["deck_kind"])
    form = _np(raw["deck_form"])
    cost = _np(raw["deck_cost"]).astype(np.float32)
    p_hand = _np(raw["deck_in_hand"]).astype(np.float32)
    p_next = _np(raw["deck_next"]).astype(np.float32)
    D = ids.shape[1]
    affordable = cost <= elixir[:, None] + 1e-6
    feats = np.zeros((T, D, F), dtype=np.float32)
    feats[..., 0] = cost / 10.0
    feats[..., 1] = kind == 0
    feats[..., 2] = kind == 1
    feats[..., 3] = kind == 2
    feats[..., 4] = (form >= 1) & (form < 200)
    feats[..., 5] = form == 200
    feats[..., 6] = affordable
    feats[..., 7] = p_hand
    feats[..., 8] = p_next
    # a deck card is playable iff the player may hold it and can pay for it
    playable = (p_hand > 0) & affordable
    next_cost = (p_next.astype(np.float64) * cost).sum(axis=1).astype(np.float32)
    globals_seq = np.stack([elixir / 10.0, opp_elixir / 10.0, time_s / 300.0, phase / 2.0, overtime,
                            next_cost / 10.0], axis=1).astype(np.float32)
    legal_place = np.zeros((T, GRID_H * GRID_W), dtype=bool)
    legal_place[:, : 16 * GRID_W] = True
    legal_place[(playable & (kind == 2)).any(axis=1)] = True
    return {
        "arena_seq": torch.from_numpy(arena),
        "deck_ids": torch.from_numpy(ids),
        "deck_feats": torch.from_numpy(feats),
        "deck_mask": torch.ones((T, D), dtype=torch.bool),
        "globals_seq": torch.from_numpy(globals_seq),
        "legal_card": torch.from_numpy(playable),
        "legal_place": torch.from_numpy(legal_place),
    }


def _hand_obs_legacy(raw, F, arena, elixir, opp_elixir, time_s, phase,
                     overtime) -> Dict[str, torch.Tensor]:
    """converter_version <= 2 (hindsight hand; legacy checkpoints only) — unchanged."""
    T = elixir.shape[0]
    kind = _np(raw["hand_kind"])
    form = _np(raw["hand_form"])
    hcost = _np(raw["hand_cost"]).astype(np.float32)
    feats = np.zeros((T, 4, F), dtype=np.float32)
    feats[..., 0] = hcost / 10.0
    feats[..., 1] = kind == 0
    feats[..., 2] = kind == 1
    feats[..., 3] = kind == 2
    feats[..., 4] = (form >= 1) & (form < 200)
    feats[..., 5] = form == 200
    affordable = hcost <= elixir[:, None] + 1e-6
    feats[..., 6] = affordable
    feats[..., 7] = _np(raw["hand_prob"])
    globals_seq = np.stack([elixir / 10.0, opp_elixir / 10.0, time_s / 300.0, phase / 2.0, overtime,
                            _np(raw["next_cost"]).astype(np.float32) / 10.0], axis=1).astype(np.float32)
    legal_place = np.zeros((T, GRID_H * GRID_W), dtype=bool)
    legal_place[:, : 16 * GRID_W] = True
    legal_place[(kind == 2).any(axis=1)] = True
    return {
        "arena_seq": torch.from_numpy(arena),
        "hand_ids": torch.from_numpy(_np(raw["hand_ids"]).astype(np.int64)),
        "hand_feats": torch.from_numpy(feats),
        "hand_mask": torch.ones((T, 4), dtype=torch.bool),
        "globals_seq": torch.from_numpy(globals_seq),
        "legal_card": torch.from_numpy(affordable),
        "legal_place": torch.from_numpy(legal_place),
    }
