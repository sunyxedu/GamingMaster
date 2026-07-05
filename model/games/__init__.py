"""Game adapter registry.

The core network (``model.game_network.GameNetwork``) is game-agnostic:
Stage 1 (spatiotemporal history encoding) and Stage 2 (TRM pondering)
never change.  Each game plugs in an :class:`ObservationEncoder` (raw
observations → tokens) and an :class:`ActionDecoder` (pondered tokens →
action logits / sampling / log-probs).

Use :func:`build_game_network` to construct a ready-to-train network for
a registered game.
"""

from model.games.base import ActionDecoder, ObservationEncoder
from model.games.registry import GAME_REGISTRY, build_game_network

__all__ = [
    "ActionDecoder",
    "ObservationEncoder",
    "GAME_REGISTRY",
    "build_game_network",
]
