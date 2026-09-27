"""Load-time expansion of Clash Royale gm-shards into the adapter's observation.

The Rust converter (`crdata convert`) stores compact per-frame fields (`raw.*`);
materialising the dense arena planes on disk would cost ~36 KB per frame, so they
are built here when a sample is loaded (`GameShardDataset` calls
:func:`expand` for game ``clash_royale``).  Every output row t is a function of
the raw rows 0..t (the arena planes, deck tokens and globals of row t alone; the play
trackers of the rows up to t) and of static shard meta, and every raw input column is
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

Both sides' plays, with WHICH card (the arena planes carry only a deployment's cost),
and the opponent's revealed deck with its hand cycle — both public, both functions of
the plays before the frame (:func:`recent_tokens`, :func:`opponent_tokens`):

    recent_ids   [T, K] long    card of the play in ring slot j (0 = empty); a play
                                keeps its slot (its index in the match mod K) while it
                                is among the K latest, so a slot tracks one play over
                                the frames it is shown in
    recent_feats [T, K, 7]      own, opponent, ability (no cell), exp(-age/tau),
                                age/600 ticks (clipped), cost/10, 1
    recent_mask  [T, K] bool    recent_cell [T, K] long  y*18 + x of the play, -1 none
    opp_ids      [T, 8] long    the opponent's cards in reveal order (0 = not yet)
    opp_feats    [T, 8, 10]     in queue, next, in hand, plays since its last play/8,
                                times played/8, cost/10, affordable by the opponent,
                                opponent plays/40, revealed cards/8, known cards in the
                                opponent's hand/4
    opp_mask     [T, 8] bool

The opponent's cycle: a played card goes to the back of the 4-card queue, so after
the opponent's n-th play (n >= 4) its last 4 plays are the queue (the oldest of them
next) and its other cards are in hand: a revealed card whose last play is 4 or more
of its plays back is in hand for certain.  Champion abilities are not plays of the
cycle (``recent_x`` -1).

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
#: recent-play token features (:func:`recent_tokens`)
RECENT_FEATURES = 7
#: opponent-deck token slots and features (:func:`opponent_tokens`)
OPP_SLOTS = 8
OPP_FEATURES = 10
#: plays of a cycle before the queue holds only known cards (the 4-card queue)
QUEUE = 4
#: age scale of the recent-play features (ticks; 600 = 30 s at 20 Hz)
RECENT_AGE_SCALE = 600.0
RECENT_KEYS = ("recent_ids", "recent_feats", "recent_mask", "recent_cell")
OPP_KEYS = ("opp_ids", "opp_feats", "opp_mask")
DECK_OBS_KEYS = ("arena_seq", "deck_ids", "deck_feats", "deck_mask", "globals_seq", "legal_card",
                 "legal_place") + RECENT_KEYS + OPP_KEYS
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


def _play_index(raw: Dict) -> np.ndarray:
    """[T, K] int64: the index in the match of every entry of the recent-play window
    (-1 = empty).  A play is the same play in every frame that shows it: its tick
    (frame tick - age), side, card and cell identify it.  Frames are read in order and
    each frame's entries oldest first, so the indices count the plays in time order."""
    side = _np(raw["recent_side"])
    T, K = side.shape
    key = np.stack([_np(raw["tick"]).astype(np.int64)[:, None] - _np(raw["recent_age"]).astype(np.int64),
                    side.astype(np.int64), _np(raw["recent_card"]).astype(np.int64),
                    _np(raw["recent_x"]).astype(np.int64), _np(raw["recent_y"]).astype(np.int64)],
                   axis=-1)
    index = np.full((T, K), -1, dtype=np.int64)
    seen: Dict[tuple, int] = {}
    for t in range(T):
        for k in range(K - 1, -1, -1):
            if side[t, k] == 255:
                continue
            index[t, k] = seen.setdefault(tuple(key[t, k].tolist()), len(seen))
    return index


def recent_tokens(raw: Dict, tau: float, index: np.ndarray) -> Dict[str, np.ndarray]:
    """The recent-play tokens (module docstring): the play with match index i sits in
    ring slot i mod K while it is shown (a slot another play of the frame already
    took — only in records that are not one match's history — falls back to the first
    free slot)."""
    side = _np(raw["recent_side"])
    T, K = side.shape
    card = _np(raw["recent_card"]).astype(np.int64)
    x = _np(raw["recent_x"]).astype(np.int64)
    y = _np(raw["recent_y"]).astype(np.int64)
    age = _np(raw["recent_age"]).astype(np.float32)
    cost = _np(raw["recent_cost"]).astype(np.float32)
    ids = np.zeros((T, K), dtype=np.int64)
    feats = np.zeros((T, K, RECENT_FEATURES), dtype=np.float32)
    mask = np.zeros((T, K), dtype=bool)
    cell = np.full((T, K), -1, dtype=np.int64)
    for t in range(T):
        for k in range(K):
            if index[t, k] < 0:
                continue
            j = int(index[t, k] % K)
            if mask[t, j]:
                j = int(np.flatnonzero(~mask[t])[0])
            mask[t, j] = True
            ids[t, j] = card[t, k]
            ability = x[t, k] < 0 or y[t, k] < 0
            if not ability:
                cell[t, j] = y[t, k] * GRID_W + x[t, k]
            feats[t, j] = (side[t, k] == 0, side[t, k] == 1, ability, np.exp(-age[t, k] / tau),
                           min(age[t, k] / RECENT_AGE_SCALE, 1.0), cost[t, k] / 10.0, 1.0)
    return {"recent_ids": ids, "recent_feats": feats, "recent_mask": mask, "recent_cell": cell}


def opponent_tokens(raw: Dict, index: np.ndarray, opp_elixir: np.ndarray) -> Dict[str, np.ndarray]:
    """The opponent-deck tokens (module docstring) from the opponent's plays before
    each frame: its cards in reveal order, where each is in its cycle, and a summary."""
    side = _np(raw["recent_side"])
    T, K = side.shape
    card = _np(raw["recent_card"]).astype(np.int64)
    x = _np(raw["recent_x"]).astype(np.int64)
    cost = _np(raw["recent_cost"]).astype(np.float32)
    ids = np.zeros((T, OPP_SLOTS), dtype=np.int64)
    feats = np.zeros((T, OPP_SLOTS, OPP_FEATURES), dtype=np.float32)
    mask = np.zeros((T, OPP_SLOTS), dtype=bool)
    done = -1                      # the latest play index already applied
    n_plays = 0
    slot: Dict[int, int] = {}      # card -> reveal slot
    last = np.zeros(OPP_SLOTS, dtype=np.int64)     # opponent-play number of its last play
    times = np.zeros(OPP_SLOTS, dtype=np.int64)
    card_cost = np.zeros(OPP_SLOTS, dtype=np.float32)
    for t in range(T):
        new = [(int(index[t, k]), k) for k in range(K)
               if index[t, k] > done and side[t, k] == 1 and x[t, k] >= 0]
        for i, k in sorted(new):
            c = int(card[t, k])
            if c not in slot and len(slot) < OPP_SLOTS:
                slot[c] = len(slot)
                card_cost[slot[c]] = cost[t, k]
            if c in slot:
                s = slot[c]
                last[s], times[s] = n_plays, times[s] + 1
            n_plays += 1
        if (index[t] >= 0).any():
            done = max(done, int(index[t].max()))
        n = len(slot)
        if n == 0:
            continue
        since = n_plays - 1 - last[:n]                  # 0 = its latest play
        queue = since < QUEUE
        nxt = (since == QUEUE - 1) & (n_plays >= QUEUE)
        hand = ~queue
        mask[t, :n] = True
        for c, s in slot.items():
            ids[t, s] = c
        f = feats[t, :n]
        f[:, 0], f[:, 1], f[:, 2] = queue, nxt, hand
        f[:, 3] = np.minimum(since, 8) / 8.0
        f[:, 4] = np.minimum(times[:n], 8) / 8.0
        f[:, 5] = card_cost[:n] / 10.0
        f[:, 6] = card_cost[:n] <= opp_elixir[t] + 1e-6
        f[:, 7] = min(n_plays, 40) / 40.0
        f[:, 8] = n / OPP_SLOTS
        f[:, 9] = min(int(hand.sum()), 4) / 4.0
    return {"opp_ids": ids, "opp_feats": feats, "opp_mask": mask}


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
        out = _deck_obs(raw, F, arena, elixir, opp_elixir, time_s, phase, overtime)
        index = _play_index(raw)
        for part in (recent_tokens(raw, tau, index), opponent_tokens(raw, index, opp_elixir)):
            out.update({k: torch.from_numpy(v) for k, v in part.items()})
        return out
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
