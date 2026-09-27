"""Pokémon (Showdown singles) game adapter — observation encoder and
action decoder.

Singles v1 token layout (``n_tokens = 13``):

    slots 0-5    own team   (team-preview order)
    slots 6-11   opponent team
    slot  12     field token (weather / terrain / turn / side conditions)

Each Pokémon slot carries a species embedding plus a small numeric
feature vector.  Species and move names become ids through one of two
schemes (the config's, :func:`n_species_ids` / :func:`n_move_ids`):

    exact    (``pokemon_species_vocab`` / ``pokemon_move_vocab`` > 0; converter
             v3, the default) the Showdown vocabularies of
             ``data_processing/games/pokemon`` (showdown_vocab.json, generated
             from Pokémon Showdown's data): 0 none, 1 UNK (a name the
             vocabulary does not know), 2.. one id per move / per species or
             forme that plays differently — no two names share an id.  The
             vocabulary is append-only, so a net whose table is smaller than
             the vendored vocabulary reads the newer names as UNK.
    hashed   (both 0: configs saved before the fields existed,
             model.config.LEGACY_CONFIG_DEFAULTS; converter v1 / v2 shards) crc32
             buckets of the name (:func:`move_bucket`) — ``pokemon_species_buckets``
             / ``pokemon_move_buckets`` ids, 0 = none; distinct names can share an
             id.

Slots whose species is unknown (``slot_mask`` False — e.g. no team
preview) are replaced by a dedicated learnable "empty" embedding, the
Hearthstone pattern.

The observer-view replay data cannot see hidden information (opponent
movesets, items, exact stats); the encoder deliberately only consumes
what the sim-protocol log reveals.

Numeric per-slot features (``pokemon_feature_dim``: 17 for converter v1-v4
shards and legacy configs, 18 from converter v5 on — shard meta
``poke_features``; the layout is the single source of truth shared with the
converter via the ``POKE_FEAT_*`` constants):

    0      hp_frac                  current HP fraction 0..1
    1      active                   is the currently active Pokémon
    2      fainted
    3-9    status one-hot           none/brn/par/psn/tox/slp/frz
    10-14  boosts / 6               atk/def/spa/spd/spe, clipped to ±1
    15     revealed                 has entered battle at least once
    16     tera                     is terastallized
    17     level / 100              (v5) 1 in the level-100 tiers, 0.05 in Little
                                    Cup, per Pokémon in the random battles

Field-token features (``pokemon_field_features``: 20 for converter v1
shards and legacy configs, 22 from converter v2 on — the trainers take it
from the shard meta ``field_features``):

    0-4    weather one-hot          sun/rain/sand/snow/other (all 0 = clear)
    5-8    terrain one-hot          electric/grassy/misty/psychic
    9      turn / 50                clipped to 1
    10-14  own side conditions      spikes/3, toxic spikes/2, stealth rock,
                                    sticky web, screens/3 (reflect + light
                                    screen + aurora veil count)
    15-19  opponent side conditions (same layout)
    20     forced-switch request    (v2) KO replacement / pivot / eject
    21     team-preview request     (v2) the lead choice

obs keys:
    species_ids [B, T, 12] long     species ids (0 = none)
    poke_feats  [B, T, 12, F]       numeric features above
    slot_mask   [B, T, 12] bool     True = species known (slot exists)
    field_seq   [B, T, 20 | 22]     field features above
    move_ids    [B, T, 4] long      ids of the first four moves the active
                                    Pokémon REVEALED before the frame
                                    (0 = none yet) — state for the decoder,
                                    forwarded as ``legal["move_ids"]``
                                    (training.train_il_game.LEGAL_FROM_OBS)
  converter v4 (``pokemon_item_vocab`` > 0: the shard meta's item_vocab /
  ability_vocab / type_vocab), every slot's revealed information, both sides —
  what the public log showed by the frame (id 0 = not revealed, embedded as zero):
    slot_moves     [B, T, 12, 4] long  its first four revealed moves (summed)
    slot_item      [B, T, 12] long     the held item the log revealed
    slot_no_item   [B, T, 12] bool     known to hold nothing (consumed, knocked off)
    slot_ability   [B, T, 12] long     the revealed ability
    slot_tera_type [B, T, 12] long     the tera type once terastallized
  v3 showed the own active's revealed moves only: the opponent's revealed moves
  and every revealed item / ability never reached the network.
  converter v5 (``pokemon_format_vocab`` > 0: the shard meta's format_vocab):
    format_id      [B, T] long         the battle's format, an id of
                                       data_processing.games.pokemon.FORMATS (Gen 9 OU,
                                       UU, RU, NU, PU, ZU, LC, Ubers, Ubers UU, Monotype,
                                       Anything Goes, the random battles; 0 = another),
                                       embedded into the field token

Frames (converter v2 / v3, data_processing/rs/gmdata/src/pokemon.rs): one
per request the player answers — the team-preview lead, every turn (with a
masked label, ``tgt.action_known`` 0 / ``action_type`` −1, when the choice
is not visible in the log) and every forced switch, snapshotted when the
request is issued.  A live bot calls the network at every request with the
same flags (field 20 on ``forceSwitch`` requests, 21 at team preview), and
builds ``species_ids`` / ``move_ids`` from its battle-log tracker through
:func:`species_id` / :func:`move_id` — converter v3's exact ids (a slot shows
the forme the log last revealed: the team-preview ``Urshifu-*`` until the
switch-in names the forme, permanent ``|detailschange|`` formes).

``frame_mask`` is intentionally a no-op: the encoder has no batch
statistics (no BatchNorm — only per-frame pointwise Linear/Embedding),
so padded frames cannot leak into real frames; they are excluded
downstream by loss masks and the causal temporal mixers, per the base
contract.

Action space:

    type 0 = move    argument (``pokemon_move_head``):
                     "identity" (default): a categorical over the whole move
                       vocabulary — the :func:`n_move_ids` move ids (id 0 = no
                       move, never chosen).  Move v is
                       scored by its embedding (the table that also embeds
                       ``move_ids``) against a query from (active_embed,
                       type_embed, ll, the revealed moves), plus a learned
                       bonus for the revealed moves.  IL target
                       ``tgt.move_id`` (the used move, first uses included);
                       IL scores the whole vocabulary — the moveset is
                       private until revealed — and a live bot masks it to
                       the moves of its request: ``legal["move_id"]``
                       [.., V] bool, built by :func:`request_move_mask` /
                       :func:`legal_from_request` through the config's id
                       scheme (:func:`move_id`).  Sampled / replayed as
                       ``actions["move_id"]``.
                     "slot" (legacy — configs saved before the field
                       existed, model.config.LEGACY_CONFIG_DEFAULTS): a
                       pointer over the 4 revealed-move slots of
                       ``move_ids``, target ``tgt.move_idx`` (−1 = the move
                       is not in obs: unsupervised), mask ``legal["move"]``,
                       ``actions["move_idx"]``.  It cannot choose a move
                       before it was revealed.
                     Plus an optional Bernoulli terastallize flag.
    type 1 = switch  argument: pointer over the 6 own slot tokens.

Because the acting Pokémon is state, not a decision, the decoder needs
its own-slot index from outside: ``decode`` reads ``active_idx`` [B, T]
long from ``labels`` (IL: the converter stores it in ``tgt``) or from
``legal`` (rollout / PPO replay), defaulting to slot 0 if absent.
``move_ids`` travels the same way (legal or labels); without it every
revealed-move input is id 0 ("nothing revealed").
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_processing.games.pokemon import ID_UNK, MOVE_VOCAB_SIZE, SPECIES_VOCAB_SIZE, VOCAB
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    PointerHead,
    bernoulli_log_prob_entropy,
    categorical_entropy,
    masked_sample,
    safe_masked_logits,
)

# Per-slot numeric feature indices (shared with the converter).
POKE_FEAT_HP = 0
POKE_FEAT_ACTIVE = 1
POKE_FEAT_FAINTED = 2
POKE_FEAT_STATUS = 3          # 7 one-hot dims: none/brn/par/psn/tox/slp/frz
POKE_FEAT_BOOSTS = 10         # 5 dims: atk/def/spa/spd/spe, /6 clipped
POKE_FEAT_REVEALED = 15
POKE_FEAT_TERA = 16
#: per-slot features of converter v1-v4 shards (the config default: legacy nets)
POKE_FEATURE_DIM = 17
#: converter v5 appends level / 100 (shard meta "poke_features": 18)
POKE_FEAT_LEVEL = 17
POKE_FEATURE_DIM_V5 = 18

#: field-token width of converter v1 shards (the config default: legacy nets)
FIELD_FEATURE_DIM = 20
#: converter v2 appends the decision-phase flags (shard meta "field_features")
FIELD_FORCED_SWITCH = 20      # the frame answers a forced-switch request
FIELD_TEAM_PREVIEW = 21       # the frame answers the team-preview (lead) request
FIELD_FEATURE_DIM_V2 = 22

N_MOVE_SLOTS = 4
TEAM_SLOTS = 6

#: the move argument heads (``PokemonConfig.pokemon_move_head``)
MOVE_HEADS = ("identity", "slot")

#: raw identity-head logit of id 0 ("no move"): finite, so mask-free consumers
#: (z-loss, KL diagnostics) stay finite, and low enough to carry no mass
NO_MOVE_LOGIT = -1e4


def move_bucket(name: str, n_buckets: int) -> int:
    """The hashed id of a move (or species) name of converter v1 / v2 (gmdata
    pokemon ``bucket``, the legacy scheme): 0 for the empty name, else crc32 of
    the lowercase alphanumerics mod (n_buckets − 1), plus 1.  Request move
    names ("Earthquake") and ids ("earthquake") give the same id."""
    if not name:
        return 0
    key = "".join(c for c in name.lower() if c.isalnum())
    return zlib.crc32(key.encode("utf-8")) % (n_buckets - 1) + 1


def n_move_ids(cfg) -> int:
    """Move id count of a config (the move embedding / identity-head size):
    its exact vocabulary (``pokemon_move_vocab``), else its hash buckets."""
    return getattr(cfg, "pokemon_move_vocab", 0) or cfg.pokemon_move_buckets


def n_species_ids(cfg) -> int:
    """Species id count of a config (the species embedding size): its exact
    vocabulary (``pokemon_species_vocab``), else its hash buckets."""
    return getattr(cfg, "pokemon_species_vocab", 0) or cfg.pokemon_species_buckets


def _check_cfg(cfg) -> None:
    if not hasattr(cfg, "pokemon_move_buckets"):
        raise TypeError(f"expected the network's pokemon config (its id scheme: exact "
                        f"vocabulary or hash buckets), got {cfg!r}")


def move_id(name: str, cfg) -> int:
    """The id ``cfg``'s network gives a move name (display name or Showdown
    id): exact — the vocabulary id (1 = UNK for a name it does not know, and
    for a name newer than the vocabulary the net was built with: an id past
    its table) — or the legacy hash bucket (:func:`move_bucket`)."""
    _check_cfg(cfg)
    n = getattr(cfg, "pokemon_move_vocab", 0)
    if not n:
        return move_bucket(name, cfg.pokemon_move_buckets)
    i = VOCAB.move_id(name)
    return i if i < n else ID_UNK


def species_id(name: str, cfg) -> int:
    """The id ``cfg``'s network gives a species name as the battle log spells
    it — the species part of a details string ("Urshifu-Rapid-Strike", the
    team-preview "Urshifu-*"): exact (UNK past the net's table) or the legacy
    hash bucket."""
    _check_cfg(cfg)
    n = getattr(cfg, "pokemon_species_vocab", 0)
    if not n:
        return move_bucket(name, cfg.pokemon_species_buckets)
    i = VOCAB.species_id(name)
    return i if i < n else ID_UNK


@dataclass
class PokemonConfig(ModelConfig):
    """Core config + Pokémon-specific dimensions (all defaulted so the
    dataclass inheritance stays valid)."""

    team_slots: int = TEAM_SLOTS
    #: numeric per-slot features (see module docstring feature table)
    pokemon_feature_dim: int = POKE_FEATURE_DIM
    #: exact species vocabulary: id count (reserved 0 none / 1 UNK included) of
    #: data_processing/games/pokemon's table the shards were converted with
    #: (converter v3 shard meta "species_vocab"; default: the vendored table);
    #: 0 = the legacy hash buckets below (model.config.LEGACY_CONFIG_DEFAULTS:
    #: configs saved before the field)
    pokemon_species_vocab: int = SPECIES_VOCAB_SIZE
    #: exact move vocabulary: id count (shard meta "move_vocab") — embeds the
    #: revealed moves (obs move_ids) and, with the identity head, is its output
    #: space; 0 = the legacy hash buckets below
    pokemon_move_vocab: int = MOVE_VOCAB_SIZE
    #: legacy (converter v1 / v2) hash buckets of the species names (id 0 =
    #: unknown); unused with an exact species vocabulary
    pokemon_species_buckets: int = 512
    #: legacy hash buckets of the move names (id 0 = no move; converter v2
    #: shard meta: 1024); unused with an exact move vocabulary
    pokemon_move_buckets: int = 256
    #: field-token features (weather/terrain/turn/side conditions [+ the v2
    #: request flags]: 20 = converter v1 / legacy, 22 = v2 shard meta)
    pokemon_field_features: int = FIELD_FEATURE_DIM
    #: 0 = move, 1 = switch
    n_action_types: int = 2
    n_move_slots: int = N_MOVE_SLOTS
    #: move argument head: "identity" = categorical over the move vocabulary
    #: (target tgt.move_id, deploy mask legal["move_id"]); "slot" = the legacy
    #: pointer over the 4 revealed-move slots (tgt.move_idx).  Configs saved
    #: before the field existed load as "slot" (LEGACY_CONFIG_DEFAULTS).
    pokemon_move_head: str = "identity"
    #: every slot's revealed information (converter v4, shard meta item_vocab /
    #: ability_vocab / type_vocab: the id counts of the vendored tables): its revealed
    #: moves, held item (or none), ability and tera type, both sides.  0 = off (the
    #: default: converter v3 shards and the configs before the fields)
    pokemon_item_vocab: int = 0
    pokemon_ability_vocab: int = 0
    pokemon_type_vocab: int = 0
    #: the battle's format (converter v5, shard meta format_vocab: the id count of
    #: data_processing/games/pokemon.FORMATS — Gen 9 OU, UU, ..., the random battles),
    #: embedded into the field token; 0 = off (the default: older shards and configs)
    pokemon_format_vocab: int = 0


class PokemonEncoder(ObservationEncoder):
    """12 Pokémon slot tokens + 1 field token → [B, T, 13, d].

    Slot layout attributes (for feature/env code):
        own_slice   slots 0-5
        opp_slice   slots 6-11
        field_idx   12
    """

    def __init__(self, cfg: PokemonConfig):
        super().__init__()
        self.cfg = cfg
        k = cfg.team_slots
        self.own_slice = slice(0, k)
        self.opp_slice = slice(k, 2 * k)
        self.field_idx = 2 * k
        self.n_tokens = 2 * k + 1

        self.feat_proj = nn.Linear(cfg.pokemon_feature_dim, cfg.d_model)
        self.species_embed = nn.Embedding(n_species_ids(cfg), cfg.d_model)
        # Unknown slots keep their token (fixed layout) but carry a
        # dedicated learnable embedding instead of stale features.
        self.slot_empty = nn.Parameter(torch.randn(1, 1, 1, cfg.d_model) * 0.02)

        self.field_proj = nn.Sequential(
            nn.Linear(cfg.pokemon_field_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # Per-slot embedding delivered via positional_embedding() — the
        # core adds it once, so the encoder must not add it again.
        self.slot_embed = nn.Parameter(torch.randn(1, self.n_tokens, cfg.d_model) * 0.02)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.species_embed.weight, std=0.02)

        # converter v4: every slot's revealed moves / item / ability / tera type (id 0 =
        # not revealed: the zero row, padding_idx — it adds nothing)
        self.slot_info = getattr(cfg, "pokemon_item_vocab", 0) > 0
        if self.slot_info:
            d = cfg.d_model
            self.slot_move_embed = nn.Embedding(n_move_ids(cfg), d, padding_idx=0)
            self.item_embed = nn.Embedding(cfg.pokemon_item_vocab, d, padding_idx=0)
            self.ability_embed = nn.Embedding(cfg.pokemon_ability_vocab, d, padding_idx=0)
            self.tera_type_embed = nn.Embedding(cfg.pokemon_type_vocab, d, padding_idx=0)
            self.no_item = nn.Parameter(torch.randn(d) * 0.02)
            for e in (self.slot_move_embed, self.item_embed, self.ability_embed, self.tera_type_embed):
                nn.init.trunc_normal_(e.weight, std=0.02)
                with torch.no_grad():
                    e.weight[0].zero_()
        # converter v5: the format (tier) of the battle, on the field token (id 0 = a
        # format outside the table: the zero row)
        self.format_vocab = getattr(cfg, "pokemon_format_vocab", 0)
        if self.format_vocab > 0:
            self.format_embed = nn.Embedding(self.format_vocab, cfg.d_model, padding_idx=0)
            nn.init.trunc_normal_(self.format_embed.weight, std=0.02)
            with torch.no_grad():
                self.format_embed.weight[0].zero_()

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # frame_mask is a documented no-op: no batch statistics here.
        present = obs["slot_mask"].unsqueeze(-1)                        # [B,T,12,1]
        x = self.feat_proj(obs["poke_feats"]) + self.species_embed(obs["species_ids"])
        if self.slot_info:
            x = (x + self.slot_move_embed(obs["slot_moves"].long()).sum(-2)
                 + self.item_embed(obs["slot_item"].long())
                 + self.ability_embed(obs["slot_ability"].long())
                 + self.tera_type_embed(obs["slot_tera_type"].long())
                 + obs["slot_no_item"].unsqueeze(-1).to(x.dtype) * self.no_item.to(x.dtype))
        # Absent slots: zero the content, substitute the "empty" embedding
        # (zeroing also makes the output invariant to padded-slot garbage).
        x = x * present + self.slot_empty * (~present)
        field = self.field_proj(obs["field_seq"])                       # [B,T,d]
        if self.format_vocab > 0:
            field = field + self.format_embed(obs["format_id"].long())
        return torch.cat([x, field.unsqueeze(2)], dim=2)               # [B,T,13,d]

    def positional_embedding(self) -> torch.Tensor:
        return self.slot_embed


class PokemonDecoder(ActionDecoder):
    """Hierarchical Type → (Move + Tera | Switch target) decoder.

    legal keys (all optional):
        action_type [B, T, 2] bool
        move_id     [B, T, V] bool      identity head: usable moves (the
                                        request's, :func:`request_move_mask`)
        move        [B, T, 4] bool      slot head: legal revealed-move slots
        switch      [B, T, 6] bool      own bench slots (alive, not active)
        tera        [B, T] bool         may still terastallize
        active_idx  [B, T] long         own active slot (NOT a mask — the
                                        acting Pokémon is state)
        move_ids    [B, T, 4] long      revealed-move ids (state, not a mask)
    labels keys (teacher forcing, −1 → sample):
        action_type [B, T] long
        active_idx  [B, T] long         conditioning state, never sampled

    decode() echoes the masks it was given — ``action_type_mask``,
    ``move_mask`` (the head's move mask), ``switch_mask`` — so il_loss
    scores the legal-masked distributions under the trainers'
    ``--il_legal_mask`` (converter v2 ``tgt.legal_action_type`` /
    ``legal_switch``); without masks il_loss is the plain CE.

    Hierarchical log-prob: type_lp plus the move head + tera Bernoulli
    only on move frames, the switch pointer only on switch frames (the
    ARG_USES table, Hearthstone discipline).  The tera Bernoulli is
    additionally gated by ``legal["tera"]`` when provided: once tera is
    spent the flag is deterministic and contributes no log-prob/entropy.
    Sampling, replay and entropy all gather the same flags.  ``greedy``
    (evaluation) takes argmaxes instead of samples.
    """

    #: (uses_move, uses_switch, uses_tera) per action type
    ARG_USES_DEFAULT = [
        (True, False, True),    # 0 move:   move + tera flag
        (False, True, False),   # 1 switch: target own slot
    ]

    def __init__(self, cfg: PokemonConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.n_action_types != len(self.ARG_USES_DEFAULT):
            raise ValueError(
                f"n_action_types={cfg.n_action_types} but the ARG_USES table "
                f"has {len(self.ARG_USES_DEFAULT)} rows"
            )
        self.move_head_kind = getattr(cfg, "pokemon_move_head", "slot")
        if self.move_head_kind not in MOVE_HEADS:
            raise ValueError(f"pokemon_move_head must be one of {MOVE_HEADS}, "
                             f"got {self.move_head_kind!r}")
        identity = self.move_head_kind == "identity"
        #: move id count (exact vocabulary or hash buckets)
        self.n_move_ids = n_move_ids(cfg)
        #: the action key / legal key / class count of the move argument
        self.move_action_key = "move_id" if identity else "move_idx"
        self.move_legal_key = "move_id" if identity else "move"
        self.n_move_classes = self.n_move_ids if identity else cfg.n_move_slots
        self.register_buffer(
            "ARG_USES", torch.tensor(self.ARG_USES_DEFAULT, dtype=torch.bool)
        )

        d = cfg.d_model
        V = self.n_move_ids
        self.type_head = MLPHead(d, cfg.n_action_types)
        self.type_embed = nn.Embedding(cfg.n_action_types, d)
        # One move embedding table: it embeds the revealed moves (move_ids,
        # 0 = none) and, for the identity head, the candidate moves.
        self.move_embed = nn.Embedding(V, d)
        if identity:
            # query (active_embed, type, ll, mean revealed-move embedding) ·
            # key(move embedding) + per-move bias + revealed-move bonus
            self.move_query = MLPHead(4 * d, d, hidden=d)
            self.move_key = nn.Linear(d, d, bias=False)
            # −log(V − 1): the initial log-partition over the real moves is ≈ 0
            # (softmax-neutral; keeps the z-loss of a large vocabulary small)
            self.move_bias = nn.Parameter(torch.full((V,), -math.log(max(V - 1, 1))))
            self.move_revealed = nn.Parameter(torch.zeros(()))
            no_move = torch.zeros(V, dtype=torch.bool)
            no_move[0] = True
            self.register_buffer("no_move", no_move, persistent=False)
        else:
            # Legacy slot head: slot k is scored by its revealed-move identity
            # (move_ids[k]) against (active_embed, type, ll).
            self.move_slot_embed = nn.Parameter(
                torch.randn(1, cfg.n_move_slots, d) * 0.02,
            )
            self.move_head = PointerHead(d, n_queries=3)  # move slots ⊗ (active, type, ll)
        # Switch target: pointer over the 6 own slot tokens ⊗ (type, ll).
        self.switch_head = PointerHead(d, n_queries=2)
        self.tera_head = MLPHead(d * 2, 1, hidden=d)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.type_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.move_embed.weight, std=0.02)

    # ------------------------------------------------------------------
    # Decode: tokens → logits
    # ------------------------------------------------------------------

    def _sample(self, logits, mask):
        """One index per row: masked sample, or the masked argmax when greedy."""
        if self.greedy:
            with torch.no_grad():
                return safe_masked_logits(logits, mask).argmax(-1)
        return masked_sample(logits, mask)

    def _pick(self, logits, label, mask, BT, n):
        """Teacher-forced (−1 rows → sampled) or fully sampled index [BT]."""
        if label is not None:
            gt = label.reshape(BT).long().clamp(0, n - 1)
            needs_sample = label.reshape(BT) < 0
            if needs_sample.any():
                return torch.where(needs_sample, self._sample(logits, mask), gt)
            return gt
        return self._sample(logits, mask)

    def _identity_logits(self, move_ids, active_embed, te, ll):
        """[BT, V] identity-head logits (id 0 at NO_MOVE_LOGIT)."""
        present = move_ids > 0                                         # [BT, 4]
        emb = self.move_embed(move_ids) * present.unsqueeze(-1)
        revealed_ctx = emb.sum(1) / present.sum(1, keepdim=True).clamp(min=1)
        q = self.move_query(torch.cat([active_embed, te, ll, revealed_ctx], dim=-1))
        keys = self.move_key(self.move_embed.weight)                   # [V, d]
        logits = (q @ keys.t()) * (q.shape[-1] ** -0.5) + self.move_bias
        revealed = torch.zeros_like(logits).scatter(1, move_ids, present.to(logits.dtype))
        logits = logits + self.move_revealed * revealed
        return logits.masked_fill(self.no_move, NO_MOVE_LOGIT)

    def decode(
        self,
        tokens: torch.Tensor,
        hl: torch.Tensor,
        ll: torch.Tensor,
        B: int,
        T: int,
        t0: int = 0,
        legal: Optional[Dict[str, torch.Tensor]] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        BT = B * T
        device = tokens.device
        own_tokens = tokens[:, : cfg.team_slots]                       # [BT, 6, d]

        def _legal(key, n):
            m = legal.get(key) if legal else None
            return m.reshape(BT, n).bool() if m is not None else None

        type_mask = _legal("action_type", cfg.n_action_types)
        type_logits = self.type_head(ll)                               # [BT, 2] raw
        action_type = self._pick(
            type_logits, labels.get("action_type") if labels else None,
            type_mask, BT, cfg.n_action_types,
        )
        te = self.type_embed(action_type)

        # The acting Pokémon is state, not a decision: read its own-slot
        # index from labels (IL) or legal (rollout/replay), default slot 0.
        active = None
        if labels is not None and labels.get("active_idx") is not None:
            active = labels["active_idx"]
        elif legal is not None and legal.get("active_idx") is not None:
            active = legal["active_idx"]
        if active is not None:
            active_idx = active.reshape(BT).long().clamp(0, cfg.team_slots - 1)
        else:
            active_idx = torch.zeros(BT, dtype=torch.long, device=device)
        active_embed = own_tokens[torch.arange(BT, device=device), active_idx]

        # Revealed-move ids of the active mon (state, like active_idx):
        # labels (IL, via LEGAL_FROM_OBS→legal in the trainer) or legal
        # (rollout/replay); absent → all slots id 0 ("unrevealed").
        move_ids = None
        if labels is not None and labels.get("move_ids") is not None:
            move_ids = labels["move_ids"]
        elif legal is not None and legal.get("move_ids") is not None:
            move_ids = legal["move_ids"]
        if move_ids is not None:
            move_ids = move_ids.reshape(BT, cfg.n_move_slots).long().clamp(
                0, self.n_move_ids - 1)
        else:
            move_ids = torch.zeros(BT, cfg.n_move_slots, dtype=torch.long,
                                   device=device)
        if self.move_head_kind == "identity":
            move_logits = self._identity_logits(move_ids, active_embed, te, ll)
        else:
            move_tokens = self.move_embed(move_ids) + self.move_slot_embed
            move_logits = self.move_head(move_tokens, active_embed, te, ll)
        switch_logits = self.switch_head(own_tokens, te, ll)           # [BT, 6]
        tera_logit = self.tera_head(
            torch.cat([active_embed, ll], dim=-1)
        ).squeeze(-1)                                                  # [BT]

        out = {
            "action_type_logits": type_logits.reshape(B, T, cfg.n_action_types),
            "move_logits": move_logits.reshape(B, T, self.n_move_classes),
            "switch_logits": switch_logits.reshape(B, T, cfg.team_slots),
            "tera_logit": tera_logit.reshape(B, T),
            "action_type_used": action_type.reshape(B, T),
            "active_idx_used": active_idx.reshape(B, T),
        }
        # echo the masks decode() was given (il_loss has no legal dict)
        for key, legal_key, n in (("action_type_mask", "action_type", cfg.n_action_types),
                                  ("move_mask", self.move_legal_key, self.n_move_classes),
                                  ("switch_mask", "switch", cfg.team_slots)):
            m = type_mask if legal_key == "action_type" else _legal(legal_key, n)
            if m is not None:
                out[key] = m.reshape(B, T, n)
        return out

    # ------------------------------------------------------------------
    # Shared lp/entropy core — one code path for sample() and
    # log_prob_entropy() is what makes the PPO ratio exact.
    # ------------------------------------------------------------------

    @staticmethod
    def _mask_logits(logits, mask):
        if mask is None:
            return logits
        safe = logits.masked_fill(~mask, float("-inf"))
        all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
        if all_masked.any():
            # Column-slice write — see base.safe_masked_logits.
            safe = safe.clone()
            safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
        return safe

    def _masks(self, pred, legal):
        """(type, move, switch) masks: the ``legal`` dict's, else decode()'s echoes."""
        def get(legal_key, echo):
            m = legal.get(legal_key) if legal else None
            if m is None:
                m = pred.get(echo)
            return m.bool() if m is not None else None
        return (get("action_type", "action_type_mask"), get(self.move_legal_key, "move_mask"),
                get("switch", "switch_mask"))

    def _lp_ent(self, pred, actions, legal):
        """Hierarchical (log_prob, entropy, per-head entropies) for any
        leading shape ([B] rollout or [B, T] replay)."""
        type_mask, move_mask, switch_mask = self._masks(pred, legal)

        type_masked = self._mask_logits(pred["action_type_logits"], type_mask)
        type_lp = F.log_softmax(type_masked, dim=-1).gather(
            -1, actions["action_type"].unsqueeze(-1)).squeeze(-1)
        type_ent = categorical_entropy(type_masked)

        move_masked = self._mask_logits(pred["move_logits"], move_mask)
        move_lp = F.log_softmax(move_masked, dim=-1).gather(
            -1, actions[self.move_action_key].unsqueeze(-1)).squeeze(-1)
        move_ent = categorical_entropy(move_masked)

        switch_masked = self._mask_logits(pred["switch_logits"], switch_mask)
        switch_lp = F.log_softmax(switch_masked, dim=-1).gather(
            -1, actions["switch_idx"].unsqueeze(-1)).squeeze(-1)
        switch_ent = categorical_entropy(switch_masked)

        tera_lp, tera_ent = bernoulli_log_prob_entropy(
            pred["tera_logit"], actions["tera"].bool(),
        )

        uses = self.ARG_USES[actions["action_type"]]                   # [..., 3]
        tera_gate = legal.get("tera") if legal else None
        use_move = uses[..., 0]
        use_switch = uses[..., 1]
        use_tera = uses[..., 2] if tera_gate is None else uses[..., 2] & tera_gate

        zero = torch.zeros_like(type_lp)
        total_lp = (
            type_lp
            + torch.where(use_move, move_lp, zero)
            + torch.where(use_switch, switch_lp, zero)
            + torch.where(use_tera, tera_lp, zero)
        )
        total_ent = (
            type_ent
            + torch.where(use_move, move_ent, zero)
            + torch.where(use_switch, switch_ent, zero)
            + torch.where(use_tera, tera_ent, zero)
        )

        with torch.no_grad():
            def _mean(ent, flag):
                f = flag.float()
                return (ent.detach() * f).sum() / f.sum().clamp(min=1)
            head_entropy = {
                "ent_type": type_ent.detach().mean(),
                "ent_move": _mean(move_ent, use_move),
                "ent_switch": _mean(switch_ent, use_switch),
                "ent_tera": _mean(tera_ent, use_tera),
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------
    # Rollout sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample move/switch/tera (the type was sampled in decode()).

        All argument heads are sampled even when the type doesn't use
        them so the replay buffer always holds valid indices; the
        ARG_USES gather zeroes their log-prob contribution.  The move
        argument is ``move_id`` (identity head) or ``move_idx`` (slot head).
        A tera flag sampled while ``legal["tera"]`` is False is forced to
        0 — its log-prob contribution is gated to zero on both the rollout
        and replay paths, so the PPO ratio stays exact."""
        with torch.no_grad():
            _, move_mask, switch_mask = self._masks(pred, legal)
            move = self._sample(pred["move_logits"], move_mask)
            switch_idx = self._sample(pred["switch_logits"], switch_mask)
            if self.greedy:
                tera = (pred["tera_logit"] > 0).long()
            else:
                tera = torch.bernoulli(torch.sigmoid(pred["tera_logit"])).long()
            tera_gate = legal.get("tera") if legal else None
            if tera_gate is not None:
                tera = tera * tera_gate.long()
            actions = {
                "action_type": pred["action_type_used"],
                self.move_action_key: move,
                "switch_idx": switch_idx,
                "tera": tera,
            }
            total_lp, _, _ = self._lp_ent(pred, actions, legal)
        return {**actions, "action_log_prob": total_lp}

    # ------------------------------------------------------------------
    # PPO log-prob replay
    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        return self._lp_ent(pred, actions, legal)

    # ------------------------------------------------------------------
    # Imitation loss
    # ------------------------------------------------------------------

    def _il_parts(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """Per-head targets, weights, denominators and per-element losses —
        the one code path :meth:`il_loss` and :meth:`il_head_terms` share.

        Frames without an observed choice (``tgt.action_type`` −1 /
        ``tgt.action_known`` 0, converter v2) carry no policy loss and no
        q_halt answer.  With decode() mask echoes (``--il_legal_mask``)
        each head scores its masked logits; a target outside its mask gets
        weight 0 (its CE would be +inf) and is counted in
        ``illegal_target_frac``; the normalisers are unchanged.  Without
        -1 labels, ``action_known`` and masks this is the historical loss,
        bit for bit."""
        cfg = self.cfg
        B, T = pred["action_type_used"].shape
        seq_f = seq_mask.float()

        known = tgt["action_type"] >= 0
        if "action_known" in tgt:
            known = known & tgt["action_known"].bool()
        tgt_type = tgt["action_type"].long().clamp(0, cfg.n_action_types - 1)
        uses = self.ARG_USES[tgt_type]                                 # [B, T, 3]
        if self.move_head_kind == "identity":
            if "move_id" not in tgt:
                raise ValueError(
                    "the identity move head (pokemon_move_head='identity') trains on "
                    "tgt['move_id'], which pokemon shards have from gmdata converter v2 on: "
                    "re-convert the data, or build pokemon_move_head='slot'")
            move_tgt, move_ok = tgt["move_id"], tgt["move_id"] >= 1
        else:
            # move_idx −1 on a move frame is unsupervisable (the used move
            # has no observed slot); _ce's clamp would otherwise silently
            # train slot 0.
            move_tgt, move_ok = tgt["move_idx"], tgt["move_idx"] >= 0
        type_mask = pred.get("action_type_mask")
        move_mask = pred.get("move_mask")
        switch_mask = pred.get("switch_mask")

        with torch.no_grad():
            known_f = known.float()
            type_match = (pred["action_type_used"] == tgt_type).float()
            w_type = seq_f * known_f
            w_move = uses[..., 0].float() * move_ok.float() * type_match * w_type
            w_switch = uses[..., 1].float() * type_match * w_type
            w_tera = uses[..., 2].float() * w_type
            n_type = w_type.sum().clamp(min=1)
            n_move = w_move.sum().clamp(min=1)
            n_switch = w_switch.sum().clamp(min=1)
            n_tera = w_tera.sum().clamp(min=1)

        def _ce(logits, target, n_classes):
            safe_tgt = target.long().clamp(0, n_classes - 1)
            return F.cross_entropy(
                logits.reshape(-1, n_classes), safe_tgt.reshape(-1),
                reduction="none",
            ).reshape(B, T).nan_to_num(0.0), safe_tgt

        type_scored = self._mask_logits(pred["action_type_logits"], type_mask)
        move_scored = self._mask_logits(pred["move_logits"], move_mask)
        switch_scored = self._mask_logits(pred["switch_logits"], switch_mask)
        type_loss_all, _ = _ce(type_scored, tgt_type, cfg.n_action_types)
        move_loss_all, safe_move = _ce(move_scored, move_tgt, self.n_move_classes)
        switch_loss_all, safe_switch = _ce(switch_scored, tgt["switch_idx"], cfg.team_slots)
        tgt_tera = tgt["tera"].float().clamp(0.0, 1.0)
        tera_loss_all = F.binary_cross_entropy_with_logits(
            pred["tera_logit"], tgt_tera, reduction="none",
        ).nan_to_num(0.0)

        def _legal_only(mask, target, ce, w):
            """(ce, weight, illegal) with targets outside ``mask`` unsupervised."""
            if mask is None:
                return ce, w, None
            ok = mask.bool().gather(-1, target.unsqueeze(-1)).squeeze(-1)
            return torch.where(ok, ce, torch.zeros_like(ce)), w * ok.float(), ~ok & (w > 0)

        type_loss_all, w_type, bad_type = _legal_only(type_mask, tgt_type, type_loss_all, w_type)
        move_loss_all, w_move, bad_move = _legal_only(move_mask, safe_move, move_loss_all, w_move)
        switch_loss_all, w_switch, bad_switch = _legal_only(switch_mask, safe_switch,
                                                            switch_loss_all, w_switch)
        illegal = None
        for bad in (bad_type, bad_move, bad_switch):
            if bad is not None:
                illegal = bad if illegal is None else illegal | bad
        return dict(seq_f=seq_f, known=known, tgt_type=tgt_type, uses=uses, move_ok=move_ok,
                    w_type=w_type, n_type=n_type, w_move=w_move, n_move=n_move,
                    w_switch=w_switch, n_switch=n_switch, w_tera=w_tera, n_tera=n_tera,
                    type_scored=type_scored, move_scored=move_scored,
                    switch_scored=switch_scored, type_mask=type_mask, move_mask=move_mask,
                    switch_mask=switch_mask, type_loss_all=type_loss_all,
                    move_loss_all=move_loss_all, safe_move=safe_move,
                    switch_loss_all=switch_loss_all, safe_switch=safe_switch,
                    tgt_tera=tgt_tera, tera_loss_all=tera_loss_all, illegal=illegal)

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict]:
        """:meth:`il_loss` per head (``ActionDecoder.il_head_terms``):
        ``action_type`` / ``move`` / ``switch`` categorical, ``tera``
        Bernoulli; weights include il_loss's label mask, ARG_USES gating and
        type-match condition weights (soft targets: ``tgt["move_probs"]``
        over the move head's classes — the vocabulary for the identity
        head — etc.)."""
        p = self._il_parts(pred, tgt, seq_mask)
        return {
            "action_type": {"logits": pred["action_type_logits"], "ce": p["type_loss_all"],
                            "weight": p["w_type"], "norm": p["n_type"], "mask": p["type_mask"]},
            "move": {"logits": pred["move_logits"], "ce": p["move_loss_all"],
                     "weight": p["w_move"], "norm": p["n_move"], "mask": p["move_mask"]},
            "switch": {"logits": pred["switch_logits"], "ce": p["switch_loss_all"],
                       "weight": p["w_switch"], "norm": p["n_switch"],
                       "mask": p["switch_mask"]},
            "tera": {"logits": pred["tera_logit"], "kind": "bernoulli",
                     "ce": p["tera_loss_all"], "weight": p["w_tera"], "norm": p["n_tera"]},
        }

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE per head, restricted to frames whose GT action type
        actually uses the argument, and condition-weighted by whether
        decode()'s sampled type matched the ground truth — the move and
        switch heads condition on the type embedding, so logits computed
        from the wrong type embedding are not a learning signal.  The
        tera head conditions only on state (active embed + ll), so it
        needs no condition weight; its BCE runs on all GT move frames.
        Frames without an observed choice are left out entirely (and are
        not q_halt answers); decode() mask echoes switch to masked CE
        (:meth:`_il_parts`).

        tgt keys: action_type / switch_idx / tera [B, T] long (unused
        arguments may be −1; action_type −1 = no observed choice), the
        move target of the head — ``move_id`` (identity: the used move's
        vocabulary id, ≥ 1) or ``move_idx`` (slot: −1 = the used move has
        no observed slot, contributes no move loss) — and optional
        ``action_known`` [B, T] bool; extra keys like value / active_idx /
        legal_* are ignored.
        """
        p = self._il_parts(pred, tgt, seq_mask)
        w_type, n_type = p["w_type"], p["n_type"]
        w_move, n_move = p["w_move"], p["n_move"]
        w_switch, n_switch = p["w_switch"], p["n_switch"]
        w_tera, n_tera = p["w_tera"], p["n_tera"]
        tgt_type, uses = p["tgt_type"], p["uses"]
        safe_move, safe_switch, tgt_tera = p["safe_move"], p["safe_switch"], p["tgt_tera"]

        type_loss = (p["type_loss_all"] * w_type).sum() / n_type
        move_loss = (p["move_loss_all"] * w_move).sum() / n_move
        switch_loss = (p["switch_loss_all"] * w_switch).sum() / n_switch
        tera_loss = (p["tera_loss_all"] * w_tera).sum() / n_tera

        loss = type_loss + move_loss + switch_loss + tera_loss

        with torch.no_grad():
            type_hit = p["type_scored"].argmax(-1) == tgt_type
            move_hit = p["move_scored"].argmax(-1) == safe_move
            switch_hit = p["switch_scored"].argmax(-1) == safe_switch
            tera_hit = (pred["tera_logit"] > 0).float() == tgt_tera
            type_acc = (type_hit.float() * w_type).sum() / n_type
            move_acc = (move_hit.float() * w_move).sum() / n_move
            switch_acc = (switch_hit.float() * w_switch).sum() / n_switch
            tera_acc = (tera_hit.float() * w_tera).sum() / n_tera
            u = uses.bool()
            answer_ok = (type_hit
                         & (move_hit | ~u[..., 0] | ~p["move_ok"])
                         & (switch_hit | ~u[..., 1])
                         & (tera_hit | ~u[..., 2]))

        out = {
            "answer_correct": answer_ok,
            "answer_mask": seq_mask.bool() & p["known"],
            "loss": loss,
            "type_loss": type_loss.detach(),
            "move_loss": move_loss.detach(),
            "switch_loss": switch_loss.detach(),
            "tera_loss": tera_loss.detach(),
            "type_acc": type_acc,
            "move_acc": move_acc,
            "switch_acc": switch_acc,
            "tera_acc": tera_acc,
        }
        if p["illegal"] is not None:
            n_seq = p["seq_f"].sum().clamp(min=1)
            out["illegal_target_frac"] = (p["illegal"].float() * p["seq_f"]).sum() / n_seq
        return out


# ----------------------------------------------------------------------
# Deploy: the legal dict of a live Showdown request
# ----------------------------------------------------------------------

def request_move_mask(moves: Sequence[Sequence[str]], cfg,
                      device: Optional[torch.device] = None) -> torch.Tensor:
    """[B, n_move_ids(cfg)] bool identity-head mask of B requests: row b = the
    usable moves of request b's active Pokémon (display names or Showdown
    ids, e.g. ``[m["id"] for m in request["active"][0]["moves"] if not
    m.get("disabled")]``) under ``cfg``'s id scheme, exactly as the converter
    maps them (:func:`move_id`: the exact vocabulary, or the legacy hash
    buckets of a hashed net).  A request move the vocabulary does not know is
    offered as UNK (id 1) — choosing it means that move (with several unknown
    moves in one request, any of them).  A row without moves (a switch-only
    request) is all False."""
    _check_cfg(cfg)
    mask = torch.zeros(len(moves), n_move_ids(cfg), dtype=torch.bool, device=device)
    for b, names in enumerate(moves):
        for name in names:
            v = move_id(name, cfg)
            if v:
                mask[b, v] = True
    return mask


def legal_from_request(
    request: Mapping[str, Any],
    slot_of: Mapping[str, int],
    cfg,
    move_ids: Optional[torch.Tensor] = None,
    n_slots: int = TEAM_SLOTS,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """The decoder ``legal`` dict (a batch of one: [1, ...]) for a Showdown
    request JSON (the ``|request|`` message of a battle), for
    ``GameNetwork.step(obs, legal=...)`` — the deploy-time mask hook of the
    identity move head.  ``cfg``: the network's config (``net.cfg``), whose
    id scheme maps the request's moves (:func:`request_move_mask`).

    ``slot_of`` maps a request Pokémon's ``ident`` ("p1: Garchomp") to its
    model slot, fixed at team preview: the team-preview request lists
    ``side.pokemon`` in the ``|poke|`` order the converter's slots follow
    (Showdown reorders ``side.pokemon`` on every switch later).
    ``move_ids`` [4] (or [1, 4]): the revealed-move ids of the active
    Pokémon from the bot's own battle-log tracker — the same causal state
    as the converter's ``obs.move_ids`` (names mapped by :func:`move_id`,
    species by :func:`species_id`), NOT the request's moveset.

    Keys: ``action_type`` [move, switch] (team preview / ``forceSwitch``:
    switch only), ``move_id`` (the request's usable moves; empty on switch
    requests), ``switch`` (alive, not active; nothing while ``trapped``
    outside a forced switch; the FAINTED slots on the ``forceSwitch`` a
    Revival Blessing issues — its active is ``reviving`` and Showdown only
    accepts a fainted Pokémon), ``tera`` (``canTerastallize``),
    ``active_idx`` (0 at team preview) and ``move_ids`` when given.  The obs
    of the same request must carry field_seq[FIELD_FORCED_SWITCH] = 1 for a
    ``forceSwitch`` request and field_seq[FIELD_TEAM_PREVIEW] = 1 at team
    preview.  ``wait`` requests are no decision.  (The converter has no
    frame for the Revival Blessing choice — it logs no ``|switch|`` — so that
    request is outside the IL data.)"""
    preview = bool(request.get("teamPreview"))
    forced = any(bool(x) for x in (request.get("forceSwitch") or []))
    alive = [False] * n_slots
    fainted = [False] * n_slots
    active: Optional[int] = None
    reviving = False
    for p in (request.get("side") or {}).get("pokemon", []):
        k = slot_of.get(p.get("ident", ""))
        if k is None or not 0 <= k < n_slots:
            continue
        fainted[k] = str(p.get("condition", "")).strip().endswith("fnt")
        alive[k] = not fainted[k]
        if p.get("active") and not preview:
            active = k
            reviving = reviving or bool(p.get("reviving"))
    act = (request.get("active") or [{}])[0] if not (preview or forced) else {}
    if forced and reviving:
        switch = list(fainted)
    else:
        switch = [alive[k] and (preview or k != active) for k in range(n_slots)]
    if act.get("trapped"):
        switch = [False] * n_slots
    moves = [m.get("id") or m.get("move", "") for m in act.get("moves", [])
             if not m.get("disabled")]
    legal = {
        "action_type": torch.tensor([[bool(moves), any(switch)]], device=device),
        "move_id": request_move_mask([moves], cfg, device=device),
        "switch": torch.tensor([switch], device=device),
        "tera": torch.tensor([bool(act.get("canTerastallize")) and bool(moves)], device=device),
        "active_idx": torch.tensor([active if active is not None else 0], device=device),
    }
    if move_ids is not None:
        legal["move_ids"] = torch.as_tensor(move_ids, device=device).long().reshape(1, -1)
    return legal
