"""Game registry.

Two sources of games:
* offline test beds (``metaharness.testbeds``), used by tests and dry runs;
* generated adaptors: a directory written by the ``adapt`` phase containing
  ``game.py`` with ``make_game(**kwargs)`` (see ``metaharness/games/adaptor.py``).
  Register one with ``register_adaptor(path)``; the CLI does this for ``--adaptor``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

GAMES: dict[str, tuple[str, Callable[..., Any]]] = {}


def register(name: str, tier: str):
    def deco(factory):
        GAMES[name] = (tier, factory)
        return factory
    return deco


def _populate() -> None:
    import importlib
    for mod in ("rulepuzzle", "snake", "lanerunner", "arc3"):     # registration side effect
        try:
            importlib.import_module(f"metaharness.testbeds.{mod}")
        except ImportError:
            pass                                    # optional dependency missing


def register_adaptor(path: Path | str, name: str | None = None) -> str:
    """Register a generated adaptor directory (``game.py`` exposing ``make_game``)."""
    from .adaptor import load_adaptor
    spec = load_adaptor(path)
    key = name or spec.name
    GAMES[key] = (spec.tier, spec.make_game)
    return key


def make_game(name: str, **kw: Any) -> Any:
    _populate()
    if name not in GAMES:
        raise KeyError(f"unknown game {name!r}; known: {sorted(GAMES)}")
    return GAMES[name][1](**kw)


def game_tier(name: str) -> str:
    _populate()
    return GAMES[name][0]
