"""Game registry.

Two sources of games:
* offline test beds (``metaharness.testbeds``), used by tests and dry runs;
* generated adaptors: a directory written by the ``adapt`` phase containing
  ``game.py`` with ``make_game(**kwargs)`` (see ``metaharness/games/adaptor.py``).
  Register one with ``register_adaptor(path)``; the CLI does this for ``--adaptor``.
  A registered adaptor is never replaced: a name that is already taken gets
  ``adaptor:<name>`` (the key ``register_adaptor`` returns), and ``adaptor_spec(key)``
  gives its seeds, eval kwargs, tier and source hash.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .adaptor import AdaptorSpec

GAMES: dict[str, tuple[str, Callable[..., Any]]] = {}
_ADAPTORS: dict[str, AdaptorSpec] = {}          # registry key -> spec of a generated adaptor


def register(name: str, tier: str):
    def deco(factory):
        if name not in _ADAPTORS:               # a late test-bed import never replaces an adaptor
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


def register_adaptor(path: Path | str, name: str | None = None, *, kwargs: dict[str, Any] | None = None) -> str:
    """Register a generated adaptor directory (``game.py`` exposing ``make_game``); returns its key.

    The test beds are registered first (registering them later used to replace an adaptor named
    ``snake`` on the first ``make_game``). A name held by a different game — a test bed or another
    adaptor directory — gives ``adaptor:<name>`` (then ``adaptor:<name>:2``, ...); registering the
    same directory again reuses its key. ``kwargs`` go to the probe instance (see ``load_adaptor``)."""
    from .adaptor import load_adaptor
    _populate()
    spec = load_adaptor(path, kwargs=kwargs)
    base = name or spec.name
    key, n = base, 1
    while key in GAMES and not (key in _ADAPTORS and _ADAPTORS[key].path == spec.path):
        n += 1
        key = f"adaptor:{base}" if n == 2 else f"adaptor:{base}:{n - 1}"
    GAMES[key] = (spec.tier, spec.make_game)
    _ADAPTORS[key] = spec
    return key


def adaptor_spec(key: str) -> AdaptorSpec | None:
    """The ``AdaptorSpec`` registered under ``key`` (seeds, eval_kwargs, tier, source_sha256), or
    None for a test bed / unknown name."""
    return _ADAPTORS.get(key)


def make_game(name: str, **kw: Any) -> Any:
    _populate()
    if name not in GAMES:
        raise KeyError(f"unknown game {name!r}; known: {sorted(GAMES)}")
    return GAMES[name][1](**kw)


def game_tier(name: str) -> str:
    _populate()
    return GAMES[name][0]
