"""Clash Royale game adapter — observation encoder and action decoder.

Observations combine three modalities:
  - an arena grid (unit presence / side / hp / tower planes) encoded by the
    same conv stack shape as the generals ``SpatialEncoder``,
  - card tokens (fixed padded slots with a presence mask, per the
    ``ObservationEncoder`` contract), see ``cr_card_tokens`` below,
  - a global scalar vector (elixir, time, tower counts, ...).

The action space is hierarchical: first decide whether to WAIT this frame
(elixir management is mostly about *not* playing), and if playing, pick a
card token and then a placement cell conditioned on that card.
Sampling, log-prob replay and teacher forcing all share one code path so
PPO ratios are exact (same discipline as the generals adapter).

Card tokens (``ClashRoyaleConfig.cr_card_tokens``):
  - ``"deck"`` (default; crdata converter_version >= 3): the 8 own deck
    cards, each carrying the causal hand belief P(in hand | own plays so far)
    and P(next | own plays so far) among its features
    (``data_processing/games/clash_royale/expand.py``).  The card head points
    over the deck tokens; ``card_slot`` (label / action) is the deck index,
    ``legal["card"]`` [.., 8] = P(in hand) > 0 and affordable.  Every deck
    token shares one positional embedding, so the adapter gives the token
    order (by card id, a function of the deck alone) no meaning: with the
    transformer / TRM cores the network is equivariant to it.  The
    recurrent-depth core is not exactly: its fixed inference state s_0 is
    one draw per token index (training draws it afresh), so the order is
    kept canonical — crdata and ``belief.deck_token_order`` sort the same way.
  - ``"hand"`` (legacy checkpoints; converter_version <= 2 shards): the 4
    reconstructed hand slots with per-slot embeddings.  Those shards inferred
    the hand from the whole match (the coming play marked "certainly in
    hand"), so this variant exists only to keep old checkpoints loading and
    computing what they computed.

Deployment: the live bot builds the deck features with the same causal
posterior from its own plays (``data_processing/games/clash_royale/belief.py``),
never from the client's true hand — that would be a different input
distribution than training.  The true hand (and elixir) may restrict
``legal["card"]``: it always lies inside the belief's support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.components.encoder import conv_stack, encode_frames, encoder_norm_of, uses_batch_stats
from model.components.transformer import _pe_from_positions, sinusoidal_pe_rows
from model.config import ModelConfig
from model.games.base import (
    ActionDecoder,
    MLPHead,
    ObservationEncoder,
    PointerHead,
    bernoulli_log_prob_entropy,
    categorical_entropy,
    masked_log_prob,
    masked_sample,
)


@dataclass
class ClashRoyaleConfig(ModelConfig):
    """Core config + Clash-Royale-specific dimensions (all defaulted so the
    dataclass inheritance stays valid)."""

    arena_h: int = 32
    arena_w: int = 18
    #: input planes: unit presence / side / hp / tower planes etc.
    arena_channels: int = 16
    hand_size: int = 4
    #: card tokens: "deck" = the deck_size own deck cards with the causal hand
    #: belief, card head over deck indices (crdata converter_version >= 3);
    #: "hand" (legacy checkpoints) = hand_size reconstructed hand slots — see
    #: the module docs.  Shards record it (meta "card_tokens").
    cr_card_tokens: str = "deck"
    deck_size: int = 8
    #: card-id vocabulary; index 0 = empty slot (catalog cr-card-catalog.v1:
    #: 124 ids; converted shards record it in their meta)
    n_cards: int = 124
    #: per-card scalar features (deck tokens: cost, kind, form, affordable,
    #: P(in hand), P(next) = 9; legacy hand slots: 8)
    card_feature_dim: int = 9
    #: own/enemy elixir, match time, tower counts, ...
    cr_global_features: int = 6
    #: time feature of the wait head's sinusoidal PE: "match" = the match
    #: clock in seconds read from the observation (globals_seq column
    #: CR_GLOBAL_TIME = time/300 s; the decode context), so any window of a
    #: game sees its true phase and clash_royale can be cropped like the other
    #: games; "frame" (legacy checkpoints) = the absolute frame index
    #: t0 .. t0+T-1, only correct for sequences decoded from their first frame
    cr_wait_time: str = "match"
    #: arena tokens: 1 (the default, and legacy checkpoints) = one token per cell and a
    #: pointer over the cells; p > 1 = p x p cells per token (the conv stack at full
    #: resolution, then a stride-p patch projection) and the placement a patch pointer +
    #: the cell inside the patch — p = 2: 144 arena tokens instead of 576, but the cell
    #: choice then comes from one patch token and a small head, not from 576 tokens the
    #: core reasons over one by one.  Placement precision is most of Clash Royale's skill,
    #: so the per-cell layout stays the default until an A/B shows the patches lose
    #: nothing (they only save compute)
    cr_arena_patch: int = 1
    #: recent-play tokens (expand.recent_tokens: which card each of the K latest
    #: plays of both sides was, where and how long ago); 0 = none (legacy)
    cr_recent_tokens: int = 8
    cr_recent_features: int = 7
    #: opponent-deck tokens (expand.opponent_tokens: the opponent's revealed cards
    #: and their place in its hand cycle); 0 = none (legacy)
    cr_opp_tokens: int = 8
    cr_opp_features: int = 10


#: globals_seq column holding the match clock / CR_TIME_SCALE seconds
#: (data_processing/games/clash_royale/expand.py: time/300 s)
CR_GLOBAL_TIME = 2
CR_TIME_SCALE = 300.0


def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Apply a legality mask with the same degenerate-row guard as
    ``masked_log_prob`` so entropy is evaluated under the same distribution
    sampling and log-prob bookkeeping use (callers cast to fp32 to match
    the fp32 log-prob path)."""
    if mask is None:
        return logits
    safe = logits.masked_fill(~mask, float("-inf"))
    all_masked = (safe.isinf() & (safe < 0)).all(dim=-1)
    if all_masked.any():
        # Column-slice write — see base.safe_masked_logits.
        safe = safe.clone()
        safe[..., 0] = safe[..., 0].masked_fill(all_masked, 0.0)
    return safe


def card_token_mode(cfg) -> str:
    """``cfg.cr_card_tokens`` ("deck" | "hand"; configs without the field: "hand")."""
    mode = getattr(cfg, "cr_card_tokens", "hand")
    if mode not in ("deck", "hand"):
        raise ValueError(f"cr_card_tokens must be 'deck' or 'hand', got {mode!r}")
    return mode


def n_card_tokens(cfg) -> int:
    """Card tokens of the encoder = classes of the card head."""
    return cfg.deck_size if card_token_mode(cfg) == "deck" else cfg.hand_size


def cr_legal_from_obs(obs: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
    """The decoder's ``legal`` dict from a deck-token observation (expand.py:
    a deck card is playable iff P(in hand) > 0 and affordable; placements on the
    own half, the whole arena when a playable card is a spell).  None for the
    legacy hand observation, whose trainers never passed masks."""
    if "deck_ids" not in obs or "legal_card" not in obs:
        return None
    return {"card": obs["legal_card"].bool(), "placement": obs["legal_place"].bool()}


class ClashRoyaleEncoder(ObservationEncoder):
    """Arena + card tokens + the play trackers + globals → tokens.

    obs keys (``cr_card_tokens="deck"``, D = deck_size):
        arena_seq    [B, T, C, H, W]      arena planes
        deck_ids     [B, T, D] long       own deck card ids (0 = empty)
        deck_feats   [B, T, D, F]         per-card features incl. the hand belief
                                          P(in hand), P(next)
        deck_mask    [B, T, D] bool       True = real card in slot
        globals_seq  [B, T, G]            global scalars
        recent_ids / recent_feats / recent_mask / recent_cell   (cr_recent_tokens K)
                                          the K latest plays of both sides, a ring slot
                                          per play (expand.recent_tokens)
        opp_ids / opp_feats / opp_mask    (cr_opp_tokens) the opponent's revealed cards
                                          and their hand cycle (expand.opponent_tokens)
    legacy ``"hand"``: hand_ids / hand_feats / hand_mask over hand_size slots.

    Token order: arena tokens ((H / p) x (W / p) patches, p = ``cr_arena_patch``),
    card tokens, recent-play tokens, opponent-deck tokens — the decoder relies on the
    first two (``n_arena_tokens``, ``n_card_tokens``).  A play token carries its card
    (the deck tokens' card embedding), its features and the arena position code of its
    cell (the patch's row / column embedding + the cell inside the patch), so plays
    and arena tokens share one spatial code; an ability has no cell.
    """

    def __init__(self, cfg: ClashRoyaleConfig):
        super().__init__()
        self.cfg = cfg
        self.card_tokens = card_token_mode(cfg)
        self.n_card_tokens = n_card_tokens(cfg)
        p = self.patch = getattr(cfg, "cr_arena_patch", 1)
        if cfg.arena_h % p or cfg.arena_w % p:
            raise ValueError(f"cr_arena_patch {p} must divide the arena {cfg.arena_h}x{cfg.arena_w}")
        self.grid_h, self.grid_w = cfg.arena_h // p, cfg.arena_w // p
        self.n_arena_tokens = self.grid_h * self.grid_w
        self.n_recent = getattr(cfg, "cr_recent_tokens", 0)
        self.n_opp = getattr(cfg, "cr_opp_tokens", 0)
        if (self.n_recent or self.n_opp) and self.card_tokens != "deck":
            raise ValueError("recent-play / opponent-deck tokens need cr_card_tokens='deck'")
        self.n_tokens = self.n_arena_tokens + self.n_card_tokens + self.n_recent + self.n_opp

        # Arena conv encoder: SpatialEncoder's Conv→Norm→ReLU→Res stack
        # (model.components.encoder.conv_stack, cfg.encoder_norm) at full resolution;
        # patches: to encoder_channels, then a stride-p projection to d_model.
        out_ch = cfg.d_model if p == 1 else cfg.encoder_channels
        self.arena_encoder = conv_stack(cfg.arena_channels, cfg.encoder_channels, out_ch,
                                        encoder_norm_of(cfg))
        self.batch_stats = uses_batch_stats(cfg)

        # Card tokens: card identity + card scalars, summed.  (The legacy hand
        # modules keep their names and construction order: old checkpoints
        # load, and a seeded legacy net initialises exactly as before.)
        self.card_embed = nn.Embedding(cfg.n_cards, cfg.d_model)
        if self.card_tokens == "hand":
            self.hand_feat_proj = nn.Linear(cfg.card_feature_dim, cfg.d_model)
            # Empty hand slots keep their token (fixed layout) but carry a
            # dedicated learnable embedding instead of stale card features.
            self.hand_empty = nn.Parameter(torch.randn(1, 1, 1, cfg.d_model) * 0.02)
        else:
            self.deck_feat_proj = nn.Linear(cfg.card_feature_dim, cfg.d_model)
            self.deck_empty = nn.Parameter(torch.randn(1, 1, 1, cfg.d_model) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(cfg.cr_global_features, cfg.d_model),
            nn.ReLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # 2-D arena PE (learnable row + column, summed) and the card-token
        # embedding.  Both are delivered through positional_embedding() —
        # the core adds it once, so the encoder must not add them again.
        self.row_embed = nn.Parameter(torch.randn(1, self.grid_h, 1, cfg.d_model) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(1, 1, self.grid_w, cfg.d_model) * 0.02)
        if self.card_tokens == "hand":
            self.hand_slot_embed = nn.Parameter(torch.randn(1, cfg.hand_size, cfg.d_model) * 0.02)
        else:
            # ONE embedding shared by every deck token: the token order (by
            # card id) carries no meaning, so nothing may depend on it
            self.deck_token_embed = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # The core only re-inits Linear/Conv/Norm; Embedding is ours to init.
        nn.init.trunc_normal_(self.card_embed.weight, std=0.02)

        d = cfg.d_model
        if p > 1:
            self.arena_patchify = nn.Conv2d(cfg.encoder_channels, d, p, stride=p)
            # the cell inside its patch (play tokens' position code)
            self.sub_embed = nn.Parameter(torch.randn(p * p, d) * 0.02)
        if self.n_recent:
            self.recent_feat_proj = nn.Linear(cfg.cr_recent_features, d)
            self.recent_empty = nn.Parameter(torch.randn(d) * 0.02)
            self.recent_nocell = nn.Parameter(torch.randn(d) * 0.02)
            # ring slots carry no order: one shared embedding (the token type)
            self.recent_token_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        if self.n_opp:
            self.opp_feat_proj = nn.Linear(cfg.cr_opp_features, d)
            self.opp_empty = nn.Parameter(torch.randn(d) * 0.02)
            self.opp_token_embed = nn.Parameter(torch.randn(1, 1, d) * 0.02)

    # ------------------------------------------------------------------

    def _encode_frames(self, arena_seq, frame_mask):
        """Run the arena conv encoder over every frame (a BatchNorm stack
        keeps padded frames out of its statistics:
        ``model.components.encoder.encode_frames``), then the patch projection."""
        B, T, C, H, W = arena_seq.shape
        feat = encode_frames(self.arena_encoder, arena_seq.reshape(B * T, C, H, W), B, T,
                             frame_mask, self.batch_stats)
        if self.patch == 1:
            return feat
        feat = self.arena_patchify(feat.reshape(B * T, *feat.shape[2:]))
        return feat.reshape(B, T, *feat.shape[1:])

    def _cell_code(self, cell: torch.Tensor) -> torch.Tensor:
        """Arena position code [..., d] of cells y * arena_w + x (the patch's row +
        column embedding, + the cell inside the patch); a negative cell (an ability, an
        empty slot) gets ``recent_nocell``."""
        cfg, p = self.cfg, self.patch
        has = cell >= 0
        c = cell.clamp(min=0)
        y, x = c // cfg.arena_w, c % cfg.arena_w
        code = self.row_embed[0, :, 0][y // p] + self.col_embed[0, 0][x // p]
        if p > 1:
            code = code + self.sub_embed[(y % p) * p + x % p]
        return torch.where(has.unsqueeze(-1), code, self.recent_nocell.to(code.dtype))

    def forward(
        self,
        obs: Dict[str, torch.Tensor],
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        arena_seq = obs["arena_seq"]
        B, T = arena_seq.shape[:2]
        d = self.cfg.d_model

        spatial = self._encode_frames(arena_seq, frame_mask)          # [B,T,d,h,w]
        spatial = spatial + self.global_proj(obs["globals_seq"]).unsqueeze(-1).unsqueeze(-1)
        arena_tokens = spatial.reshape(B, T, d, self.n_arena_tokens).permute(0, 1, 3, 2)

        if self.card_tokens == "deck":
            cards = self.card_embed(obs["deck_ids"]) + self.deck_feat_proj(obs["deck_feats"])
            present = obs["deck_mask"].unsqueeze(-1)                  # [B,T,deck,1]
            cards = cards * present + self.deck_empty * (~present)
            parts = [arena_tokens, cards]
            if self.n_recent:
                rec = (self.card_embed(obs["recent_ids"].long())
                       + self.recent_feat_proj(obs["recent_feats"].float())
                       + self._cell_code(obs["recent_cell"].long()))
                present = obs["recent_mask"].bool().unsqueeze(-1)
                parts.append(torch.where(present, rec, self.recent_empty.to(rec.dtype)))
            if self.n_opp:
                opp = self.card_embed(obs["opp_ids"].long()) + self.opp_feat_proj(obs["opp_feats"].float())
                present = obs["opp_mask"].bool().unsqueeze(-1)
                parts.append(torch.where(present, opp, self.opp_empty.to(opp.dtype)))
            return torch.cat([t.to(arena_tokens.dtype) for t in parts], dim=2)   # [B,T,S,d]

        hand = self.card_embed(obs["hand_ids"]) + self.hand_feat_proj(obs["hand_feats"])
        present = obs["hand_mask"].unsqueeze(-1)                      # [B,T,hand,1]
        # Absent slots: zero the card features, substitute the learnable
        # "empty" embedding (fixed token layout — never shrink the sequence).
        hand = hand * present + self.hand_empty * (~present)

        return torch.cat([arena_tokens, hand], dim=2)                 # [B,T,S,d]

    def decode_context(self, obs: Dict[str, torch.Tensor]) -> Optional[Dict[str, torch.Tensor]]:
        """``{"match_time": float [B, T]}`` — the match clock in seconds
        (``cr_wait_time="match"``, :class:`CardPlacementDecoder`'s wait PE);
        None for legacy ``"frame"`` configs, whose decoder takes no context."""
        if getattr(self.cfg, "cr_wait_time", "frame") != "match":
            return None
        return {"match_time": obs["globals_seq"][..., CR_GLOBAL_TIME].float() * CR_TIME_SCALE}

    def positional_embedding(self) -> torch.Tensor:
        d = self.cfg.d_model
        arena_pe = (self.row_embed + self.col_embed).reshape(1, -1, d)
        if self.card_tokens == "deck":
            card_pe = self.deck_token_embed.expand(1, self.n_card_tokens, d)
        else:
            card_pe = self.hand_slot_embed
        parts = [arena_pe, card_pe]
        if self.n_recent:
            parts.append(self.recent_token_embed.expand(1, self.n_recent, d))
        if self.n_opp:
            parts.append(self.opp_token_embed.expand(1, self.n_opp, d))
        return torch.cat(parts, dim=1)


class CardPlacementDecoder(ActionDecoder):
    """Hierarchical Wait → Card → Placement decoder.

    The card head points over the encoder's card tokens: the deck_size deck
    cards (``cr_card_tokens="deck"``; ``card_slot`` = deck index) or the
    legacy hand_size hand slots (``"hand"``); N = that token count.

    Placement, ``cr_arena_patch`` p:
      p = 1 (legacy): a pointer over the arena cells.
      p > 1: a pointer over the (H / p) x (W / p) arena patch tokens, then the cell
        inside the committed patch (an MLP of the patch token, the card and the LL
        token; the arc3 patch / sub pattern).  The patch is teacher-forced from
        ``labels["place_idx"]`` (the cell) like the card, else sampled; its log-prob
        is log π(patch) + log π(cell | patch).  IL scores every head under the
        legality masks (a target outside them gets no weight: il_legal_mask
        semantics), as sampling and replay do.

    legal keys:   card      [B, T, N]      bool
                  placement [B, T, H * W]  bool (cells; p > 1 derives the patch and
                                           cell masks from it)
    labels keys:  card_slot [B, T]  teacher-forcing card choice, place_idx [B, T]
                  the cell (p > 1) — −1 → sample from the model's own distribution
    actions keys: waited, card_slot, place_idx (the cell)

    Hierarchical log-prob (must match between sample() and
    log_prob_entropy() — the PPO-exactness invariant):
        wait → wait_lp
        play → wait_lp (not-wait side) + card_lp + placement_lp
    """

    def __init__(self, cfg: ClashRoyaleConfig):
        super().__init__()
        self.cfg = cfg
        self.n_card_tokens = n_card_tokens(cfg)
        p = self.patch = getattr(cfg, "cr_arena_patch", 1)
        self.n_cells = cfg.arena_h * cfg.arena_w
        self.grid_h, self.grid_w = cfg.arena_h // p, cfg.arena_w // p
        self.n_arena_tokens = self.grid_h * self.grid_w
        self.n_sub = p * p
        # Wait uses a sinusoidal time PE like generals' PassHead: whether
        # to hold elixir depends on match phase, not just the board.
        self.time_pe_dim = 32
        self.wait_time = getattr(cfg, "cr_wait_time", "frame")
        if self.wait_time not in ("match", "frame"):
            raise ValueError(f"cr_wait_time must be 'match' or 'frame', got {self.wait_time!r}")
        self.wait_head = MLPHead(cfg.d_model + self.time_pe_dim, 1, hidden=cfg.d_model)
        self.card_head = PointerHead(cfg.d_model, n_queries=1)       # card tokens ⊗ ll
        self.place_head = PointerHead(cfg.d_model, n_queries=2)      # arena ⊗ (card, ll)
        if p > 1:
            self.sub_head = MLPHead(3 * cfg.d_model, self.n_sub)     # (patch, card, ll)

    # -- cells <-> (patch, cell inside it) ---------------------------------

    def patch_of(self, cell: torch.Tensor) -> torch.Tensor:
        p, W = self.patch, self.cfg.arena_w
        return (cell // W // p) * self.grid_w + (cell % W) // p

    def sub_of(self, cell: torch.Tensor) -> torch.Tensor:
        p, W = self.patch, self.cfg.arena_w
        return (cell // W % p) * p + (cell % W) % p

    def cell_of(self, patch_idx: torch.Tensor, sub: torch.Tensor) -> torch.Tensor:
        p, W = self.patch, self.cfg.arena_w
        y = (patch_idx // self.grid_w) * p + sub // p
        x = (patch_idx % self.grid_w) * p + sub % p
        return y * W + x

    def _cell_table(self, cells: torch.Tensor) -> torch.Tensor:
        """[..., H * W] cell mask → [..., patches, p²] (the cells of every patch)."""
        p = self.patch
        m = cells.reshape(*cells.shape[:-1], self.grid_h, p, self.grid_w, p)
        n = m.dim()
        m = m.permute(*range(n - 4), n - 4, n - 2, n - 3, n - 1)
        return m.reshape(*cells.shape[:-1], self.n_arena_tokens, self.n_sub)

    # ------------------------------------------------------------------
    # Decode: tokens → logits
    # ------------------------------------------------------------------

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
        ctx: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        BT = B * T
        device = tokens.device
        A, N = self.n_arena_tokens, self.n_card_tokens
        arange = torch.arange(BT, device=device)

        arena_tok = tokens[:, :A]                                     # [BT, A, d]
        card_tok = tokens[:, A:A + N]                                 # [BT, N, d] card tokens

        card_logits = self.card_head(card_tok, ll)                   # [BT, N] raw

        # Card conditioning: teacher-forced when labels are given (−1 rows
        # fall back to the model's own masked sample), else sampled — the
        # chosen index is returned so losses can condition-weight on it.
        card_mask = legal.get("card") if legal else None
        cm_flat = card_mask.reshape(BT, N).bool() if card_mask is not None else None
        card_label = labels.get("card_slot") if labels else None
        if card_label is not None:
            gt_slot = card_label.reshape(BT).long().clamp(0, N - 1)
            needs_sample = card_label.reshape(BT) < 0
            if needs_sample.any():
                pred_slot = masked_sample(card_logits, cm_flat)
                card_slot = torch.where(needs_sample, pred_slot, gt_slot)
            else:
                card_slot = gt_slot
        else:
            card_slot = masked_sample(card_logits, cm_flat)

        card_embed = card_tok[arange, card_slot]
        placement_logits = self.place_head(arena_tok, card_embed, ll)  # [BT, A]

        # Wait PE of the match phase.  "match": the sinusoidal code of the
        # match clock (seconds) from the observation — identical in step()
        # and forward() and for any crop window.  "frame" (legacy): absolute
        # frame indices t0 .. t0+T-1; GameNetwork.forward decodes at t0=0,
        # so such a net must never be trained on random crops (the trainers
        # refuse it, see training/train_il_game.check_crop_policy).
        if self.wait_time == "match":
            if ctx is None or "match_time" not in ctx:
                raise KeyError("cr_wait_time='match' needs the decode context "
                               "ctx['match_time'] (ClashRoyaleEncoder.decode_context)")
            pos = ctx["match_time"].reshape(BT, 1).float()
            pe = _pe_from_positions(pos, self.time_pe_dim)
        else:
            pe = sinusoidal_pe_rows(t0, T, self.time_pe_dim, device)
            pe = pe.unsqueeze(0).expand(B, T, self.time_pe_dim).reshape(BT, self.time_pe_dim)
        wait_logits = self.wait_head(torch.cat([ll, pe], dim=-1)).squeeze(-1)

        out = {
            "wait_logits": wait_logits.reshape(B, T),
            "card_logits": card_logits.reshape(B, T, N),
            "placement_logits": placement_logits.reshape(B, T, A),
            "card_slot": card_slot.reshape(B, T),
        }
        if self.patch == 1:
            return out

        # patch → the cell inside it; masks from the cell mask
        cells = legal.get("placement") if legal else None
        table = self._cell_table(cells.reshape(BT, self.n_cells).bool()) if cells is not None else None
        patch_mask = table.any(-1) if table is not None else None
        place_label = labels.get("place_idx") if labels else None
        if place_label is not None:
            lbl = place_label.reshape(BT).long()
            gt_patch = self.patch_of(lbl.clamp(0, self.n_cells - 1))
            sampled = masked_sample(placement_logits, patch_mask)
            patch_idx = torch.where(lbl >= 0, gt_patch, sampled)
        else:
            patch_idx = masked_sample(placement_logits, patch_mask)
        patch_tok = arena_tok[arange, patch_idx]
        sub_logits = self.sub_head(torch.cat([patch_tok, card_embed, ll], dim=-1))  # [BT, p²]
        out["sub_logits"] = sub_logits.reshape(B, T, self.n_sub)
        out["patch_idx"] = patch_idx.reshape(B, T)
        if cm_flat is not None:
            out["card_mask"] = cm_flat.reshape(B, T, N)
        if table is not None:
            out["patch_mask"] = patch_mask.reshape(B, T, A)
            out["sub_mask"] = table[arange, patch_idx].reshape(B, T, self.n_sub)
        return out

    # ------------------------------------------------------------------
    # Rollout sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        pred: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample wait/placement (card was already sampled in decode()) and
        compute the hierarchical log-prob exactly as log_prob_entropy would.

        Card and placement are always sampled, even on wait frames, so the
        replay buffer has valid conditioning indices; the hierarchy zeroes
        their contribution when waited (same as generals' pass)."""
        wait_logits = pred["wait_logits"]
        card_slot = pred["card_slot"]
        card_mask = legal.get("card") if legal else None
        place_mask = legal.get("placement") if legal else None

        with torch.no_grad():
            waited = torch.bernoulli(torch.sigmoid(wait_logits)).bool()
            wait_lp, _ = bernoulli_log_prob_entropy(wait_logits, waited)

            card_lp = masked_log_prob(pred["card_logits"], card_slot, card_mask)

            if self.patch == 1:
                place_idx = masked_sample(pred["placement_logits"], place_mask)
                place_lp = masked_log_prob(pred["placement_logits"], place_idx, place_mask)
            else:
                pm, sm = pred.get("patch_mask"), pred.get("sub_mask")
                patch_idx = pred["patch_idx"]
                sub = masked_sample(pred["sub_logits"], sm)
                place_idx = self.cell_of(patch_idx, sub)
                place_lp = (masked_log_prob(pred["placement_logits"], patch_idx, pm)
                            + masked_log_prob(pred["sub_logits"], sub, sm))

            total_lp = torch.where(waited, wait_lp, wait_lp + card_lp + place_lp)

        return {
            "card_slot": card_slot,
            "place_idx": place_idx,
            "waited": waited,
            "action_log_prob": total_lp,
        }

    # ------------------------------------------------------------------
    # PPO log-prob replay
    # ------------------------------------------------------------------

    def log_prob_entropy(
        self,
        pred: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        legal: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        waited = actions["waited"]                                    # [B, T] bool
        card_mask = legal.get("card") if legal else None
        place_mask = legal.get("placement") if legal else None

        wait_lp, wait_ent = bernoulli_log_prob_entropy(pred["wait_logits"], waited)

        # Log-probs go through masked_log_prob — the exact fp32 path
        # sample() used for rollout bookkeeping — so bf16 rollout and PPO
        # replay agree bit-for-bit; entropy sees the same fp32 distribution.
        card_lp = masked_log_prob(
            pred["card_logits"], actions["card_slot"], card_mask,
        )
        card_ent = categorical_entropy(
            _mask_logits(pred["card_logits"].float(), card_mask),
        )

        if self.patch == 1:
            place_lp = masked_log_prob(
                pred["placement_logits"], actions["place_idx"], place_mask,
            )
            place_ent = categorical_entropy(
                _mask_logits(pred["placement_logits"].float(), place_mask),
            )
        else:
            # replay teacher-forces the stored cell's patch (labels["place_idx"]), so
            # the sub mask is the stored patch's
            pm, sm = pred.get("patch_mask"), pred.get("sub_mask")
            cell = actions["place_idx"].long().clamp(0, self.n_cells - 1)
            place_lp = (masked_log_prob(pred["placement_logits"], self.patch_of(cell), pm)
                        + masked_log_prob(pred["sub_logits"], self.sub_of(cell), sm))
            place_ent = (categorical_entropy(_mask_logits(pred["placement_logits"].float(), pm))
                         + categorical_entropy(_mask_logits(pred["sub_logits"].float(), sm)))

        total_lp = torch.where(waited, wait_lp, wait_lp + card_lp + place_lp)
        total_ent = torch.where(waited, wait_ent, wait_ent + card_ent + place_ent)

        with torch.no_grad():
            play_f = (~waited).float()
            n_play = play_f.sum().clamp(min=1)
            head_entropy = {
                "ent_wait": wait_ent.detach().mean(),
                "ent_card": (card_ent.detach() * play_f).sum() / n_play,
                "ent_place": (place_ent.detach() * play_f).sum() / n_play,
            }
        return total_lp, total_ent, head_entropy

    # ------------------------------------------------------------------
    # Imitation loss
    # ------------------------------------------------------------------

    def _il_parts(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Per-head targets, weights, denominators and per-element losses —
        the one code path :meth:`il_loss` and :meth:`il_head_terms` share."""
        B, T, hand = pred["card_logits"].shape
        n_arena = pred["placement_logits"].shape[-1]
        seq_f = seq_mask.float()
        n_seq = seq_f.sum().clamp(min=1)

        waited = tgt["waited"]
        play = seq_mask & ~waited
        play_f = play.float()
        n_play = play_f.sum().clamp(min=1)

        wait_loss_all = F.binary_cross_entropy_with_logits(
            pred["wait_logits"], waited.float(), reduction="none",
        ).nan_to_num(0.0)
        parts = dict(seq_f=seq_f, n_seq=n_seq, waited=waited, play_f=play_f, n_play=n_play,
                     wait_loss_all=wait_loss_all)
        if self.patch > 1:
            return {**parts, **self._il_patch_parts(pred, tgt, play)}

        tgt_card = torch.where(play, tgt["card_slot"], 0).long().clamp(0, hand - 1)
        tgt_place = torch.where(play, tgt["place_idx"], 0).long().clamp(0, n_arena - 1)

        card_loss_all = F.cross_entropy(
            pred["card_logits"].reshape(-1, hand), tgt_card.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)

        with torch.no_grad():
            cond = (pred["card_slot"] == tgt_card).float() * play_f
            n_cond = cond.sum().clamp(min=1.0)

        place_loss_all = F.cross_entropy(
            pred["placement_logits"].reshape(-1, n_arena), tgt_place.reshape(-1),
            reduction="none",
        ).reshape(B, T).nan_to_num(0.0)
        return dict(parts, tgt_card=tgt_card, tgt_place=tgt_place, cond=cond, n_cond=n_cond,
                    card_loss_all=card_loss_all, place_loss_all=place_loss_all)

    def _il_patch_parts(self, pred, tgt, play) -> Dict[str, object]:
        """The patch decoder's card / patch / cell heads under the legality masks
        (:func:`model.games.board._categorical_head`: a target outside its mask gets
        weight 0, like the other decoders' il_legal_mask semantics)."""
        from model.games.board import _categorical_head

        N = self.n_card_tokens
        play_f = play.float()
        tgt_card = torch.where(play, tgt["card_slot"], 0).long().clamp(0, N - 1)
        cell = torch.where(play, tgt["place_idx"], 0).long().clamp(0, self.n_cells - 1)
        tgt_patch, tgt_sub = self.patch_of(cell), self.sub_of(cell)
        terms, ok, illegal = {}, {}, {}
        n_play = play_f.sum().clamp(min=1.0)
        terms["card"], ok["card"], illegal["card"] = _categorical_head(
            pred["card_logits"], tgt_card, pred.get("card_mask"), play_f, n_play)
        with torch.no_grad():
            cond = (pred["card_slot"] == tgt_card).float() * play_f
            if illegal["card"] is not None:
                cond = cond * (~illegal["card"]).float()
        n_cond = cond.sum().clamp(min=1.0)
        terms["patch"], ok["patch"], illegal["patch"] = _categorical_head(
            pred["placement_logits"], tgt_patch, pred.get("patch_mask"), cond, n_cond)
        with torch.no_grad():
            cond_sub = cond * (pred["patch_idx"] == tgt_patch).float()
        n_sub = cond_sub.sum().clamp(min=1.0)
        terms["sub"], ok["sub"], illegal["sub"] = _categorical_head(
            pred["sub_logits"], tgt_sub, pred.get("sub_mask"), cond_sub, n_sub)
        return dict(terms=terms, ok=ok, illegal=illegal, tgt_card=tgt_card)

    def il_head_terms(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, Dict]:
        """:meth:`il_loss` per head (``ActionDecoder.il_head_terms``):
        ``wait`` Bernoulli on every frame, ``card`` on play frames,
        ``placement`` on play frames whose conditioning card matched
        (soft targets: ``tgt["placement_probs"]`` etc.); the patch decoder
        (``cr_arena_patch`` > 1): ``card``, ``patch`` (the placement's patch, frames
        whose card matched) and ``sub`` (the cell inside it, frames whose patch
        matched too), each with its legality mask."""
        p = self._il_parts(pred, tgt, seq_mask)
        wait = {"logits": pred["wait_logits"], "kind": "bernoulli",
                "ce": p["wait_loss_all"], "weight": p["seq_f"], "norm": p["n_seq"]}
        if self.patch > 1:
            return {"wait": wait, **p["terms"]}
        return {
            "wait": wait,
            "card": {"logits": pred["card_logits"], "ce": p["card_loss_all"],
                     "weight": p["play_f"], "norm": p["n_play"]},
            "placement": {"logits": pred["placement_logits"], "ce": p["place_loss_all"],
                          "weight": p["cond"], "norm": p["n_cond"]},
        }

    def il_loss(
        self,
        pred: Dict[str, torch.Tensor],
        tgt: Dict[str, torch.Tensor],
        seq_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Masked CE per head.  Placement conditions on the chosen card, so
        its loss is weighted by (decode's conditioning card == GT card) —
        the same discipline as ImitationLoss's cond_weight: a placement
        computed from the wrong card embedding is not a learning signal.

        tgt keys: waited [B,T] bool, card_slot [B,T] long, place_idx [B,T]
        long (card/place ignored — may be −1 — on waited frames).
        """
        p = self._il_parts(pred, tgt, seq_mask)
        seq_f, n_seq = p["seq_f"], p["n_seq"]
        waited = p["waited"]
        wait_loss = (p["wait_loss_all"] * seq_f).sum() / n_seq
        with torch.no_grad():
            wait_acc = (((pred["wait_logits"] > 0) == waited).float() * seq_f).sum() / n_seq
            pred_wait = pred["wait_logits"] > 0
        if self.patch > 1:
            terms, ok = p["terms"], p["ok"]
            head = {k: (t["ce"] * t["weight"]).sum() / t["norm"] for k, t in terms.items()}
            with torch.no_grad():
                acc = {f"{k}_acc": (ok[k].float() * terms[k]["weight"]).sum() / terms[k]["norm"]
                       for k in terms}
                answer_ok = torch.where(waited, pred_wait,
                                        ~pred_wait & ok["card"] & ok["patch"] & ok["sub"])
                valid = seq_mask.bool() & ~waited
                bad = [m for m in p["illegal"].values() if m is not None]
                extra = {}
                if bad:
                    any_bad = torch.stack(bad).any(0) & valid
                    extra["illegal_target_frac"] = (any_bad.float().sum()
                                                    / valid.float().sum().clamp(min=1))
            return {"answer_correct": answer_ok, "answer_mask": seq_mask.bool(),
                    "loss": wait_loss + head["card"] + head["patch"] + head["sub"],
                    "wait_loss": wait_loss.detach(), "card_loss": head["card"].detach(),
                    "place_loss": (head["patch"] + head["sub"]).detach(),
                    "patch_loss": head["patch"].detach(), "sub_loss": head["sub"].detach(),
                    "wait_acc": wait_acc, **acc, **extra}

        play_f, n_play = p["play_f"], p["n_play"]
        cond, n_cond = p["cond"], p["n_cond"]
        tgt_card, tgt_place = p["tgt_card"], p["tgt_place"]
        card_loss = (p["card_loss_all"] * play_f).sum() / n_play
        place_loss = (p["place_loss_all"] * cond).sum() / n_cond

        loss = wait_loss + card_loss + place_loss

        with torch.no_grad():
            card_acc = ((pred["card_logits"].argmax(-1) == tgt_card).float() * play_f).sum() / n_play
            place_acc = ((pred["placement_logits"].argmax(-1) == tgt_place).float() * cond).sum() / n_cond
            answer_ok = torch.where(
                waited, pred_wait,
                ~pred_wait & (pred["card_logits"].argmax(-1) == tgt_card)
                & (pred["placement_logits"].argmax(-1) == tgt_place))

        return {
            "answer_correct": answer_ok,
            "answer_mask": seq_mask.bool(),
            "loss": loss,
            "wait_loss": wait_loss.detach(),
            "card_loss": card_loss.detach(),
            "place_loss": place_loss.detach(),
            "wait_acc": wait_acc,
            "card_acc": card_acc,
            "place_acc": place_acc,
        }
