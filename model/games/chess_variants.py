"""Chess variants adapter: the Lichess variants chess960, crazyhouse, atomic, antichess,
three-check, king of the hill, horde and racing kings as ONE game, ``chess_variants``.

One registry game, not one per variant.  All eight are played with the same pieces on
the same 8x8 board and share almost all of their move patterns and tactics; only a few
rules differ (explosions, drops, forced captures, extra ways to win).  In a
MultiGameNetwork only the core is shared across registry games — every game brings its
own encoder, decoder and value head — so eight entries would learn eight separate board
encoders, from-to decoders and value heads for nearly the same game.  One entry shares
all of them, and a variant one-hot in the globals (plus the goal-square, promoted-piece
and pocket inputs) tells the network which rules apply.  The small variants (horde,
racing kings, king of the hill: 6.6-8.5 M lichess games each) then learn board
perception from the 160 M games of all eight instead of their own few, and a new
variant is a new one-hot slot, not a new adapter.  Standard chess stays the separate
``chess`` game (its converter, labels and checkpoints are untouched); chess960 from
Scharnagl position 518 is standard chess with king -> rook castling.  The cost is
clarity and control: one IL mixture weight covers all variants (their mix is the
corpus's — the fetch script's per-variant months / filters set it; the RL env's
``variants`` / ``weights`` set the RL mix), and diagnostics split by variant through the
shard sample meta ``variant`` (IL) or the env config (RL), not by game name.

Rules, frames and data live in Rust (``data_processing/rs/gmrules/src/games/
chess_variants.rs``, shakmaty's variant rules + lichess' automatic draws; converter
``gmdata chess-variants``, fetch ``scripts/fetch_chess_variants.sh``), simulated by
gmsim (``env/vec_gmsim.py``); the registration is ``model/games/gmrules/
chess_variants.py``.

Tokens (74): the 64 board squares (mover's view, ranks mirrored for Black) + 10 pocket
slots (own P N B R Q, opponent P N B R Q; crazyhouse drops, zero elsewhere) — the
shogi adapter's layout (:class:`model.games.shogi.ShogiEncoder`: board conv stack,
per-slot hand embeddings, globals broadcast to the board).  The action is the board
adapter's :class:`~model.games.board.FromToDecoder` over those tokens: from-token → to-
square → promotion (none, N, B, R, Q, K).  A drop is "from the own pocket slot to the
square", so moves and drops are ONE hierarchical categorical, with no mode head.
Castling is king → own rook square (chess960-safe).  Legality: the env's exact tables
``from`` [74] / ``to_cond`` [74, 74] / ``promotion_cond`` [74, 74, 6] in RL, the
converter's ``tgt.legal_from`` / ``legal_to_of`` / ``legal_promotion_of`` in IL
(``--il_legal_mask``).  RL: train_rl_joint's generic ``FromToSpec``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn

from model.games.board import BoardConfig, FromToDecoder
from model.games.shogi import ShogiEncoder

#: variant one-hot order of ``globals_seq[:8]`` (= gmrules ChessVariant, shard meta ``variants``)
VARIANTS = ("chess960", "crazyhouse", "atomic", "antichess", "threecheck", "kingofthehill",
            "horde", "racingkings")


@dataclass
class ChessVariantsConfig(BoardConfig):
    """The chessvar.v1 frame dims (defaults = the gmrules spec; a fresh IL run takes them
    from the shard meta, train_il_game META_ARCH_KEYS):

    board_seq [piece_channels=17, 8, 8], hand_seq [2 * n_hand_piece_types=10,
    hand_feature_dim=8], globals_seq [board_global_features=24]; promotions: none, N, B,
    R, Q, K (n_promotions=6)."""

    board_h: int = 8
    board_w: int = 8
    piece_channels: int = 17
    board_global_features: int = 24
    n_promotions: int = 6
    n_hand_piece_types: int = 5
    hand_feature_dim: int = 8

    @property
    def shogi_global_features(self) -> int:
        """The globals width under the name :class:`ShogiEncoder` reads."""
        return self.board_global_features


class ChessVariantsEncoder(ShogiEncoder):
    """Board planes → 64 square tokens (+ the globals, broadcast), 10 pocket-slot tokens
    (a shared Linear of the slot's count thermometer + a per-slot embedding).

    obs keys: ``board_seq`` [B, T, 17, 8, 8], ``hand_seq`` [B, T, 10, 8],
    ``globals_seq`` [B, T, 24].  BatchNorm excludes padded frames (``frame_mask``)."""


def build_chess_variants_network(**overrides) -> nn.Module:
    """ChessVariantsEncoder + FromToDecoder on a :class:`ChessVariantsConfig` (the
    registry builder of ``chess_variants``)."""
    from model.game_network import GameNetwork

    cfg = ChessVariantsConfig(**overrides)
    return GameNetwork(cfg, ChessVariantsEncoder(cfg), FromToDecoder(cfg))
