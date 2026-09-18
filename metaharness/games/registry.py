from __future__ import annotations

from typing import Any, Callable

GAMES: dict[str, tuple[str, Callable[..., Any]]] = {}


def register(name: str, tier: str):
    def deco(factory):
        GAMES[name] = (tier, factory)
        return factory
    return deco


def _populate() -> None:
    import importlib
    for mod in ("rulepuzzle", "snake", "arc3"):     # registration side effect
        try:
            importlib.import_module(f"{__package__}.{mod}")
        except ImportError:
            pass                                    # optional / not on this branch


def make_game(name: str, **kw: Any) -> Any:
    _populate()
    if name not in GAMES:
        raise KeyError(f"unknown game {name!r}; known: {sorted(GAMES)}")
    return GAMES[name][1](**kw)


def game_tier(name: str) -> str:
    _populate()
    return GAMES[name][0]
