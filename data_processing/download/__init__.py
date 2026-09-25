"""generals.io replay download + .gior decoding (reference implementation for viewers)."""

from .deserializer import deserialize_replay
from .replay_fetcher import download_replays

__all__ = ['download_replays', 'deserialize_replay']
