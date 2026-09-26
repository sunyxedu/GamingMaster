"""Registry mapping game names to network builders.

Each builder returns a fully assembled network — the game-agnostic core
(``GameNetwork``: Stage-1 spatiotemporal encoder + Stage-2 TRM) wired to
the game's ObservationEncoder / ActionDecoder.  Builders import lazily so
importing one game never pays for the others.

Register new games here (or call :func:`register_game` from plugin code).
gmrules games register themselves: every module of model/games/gmrules/ becomes
an entry (:func:`_register_rules_games`) — the candidate-action adapter, or the
entry's own ``builder`` for a custom schema.
"""

from __future__ import annotations

import functools
from typing import Callable, Dict

import torch.nn as nn


def _build_generals(**overrides) -> nn.Module:
    from model.config import ModelConfig
    from model.network import GeneralsNetwork

    return GeneralsNetwork(ModelConfig(**overrides))


def _build_chess(**overrides) -> nn.Module:
    """Chess-like board games: from-square → to-square (+promotion) moves."""
    from model.game_network import GameNetwork
    from model.games.board import BoardConfig, BoardEncoder, FromToDecoder

    overrides.setdefault("board_h", 8)
    overrides.setdefault("board_w", 8)
    cfg = BoardConfig(**overrides)
    return GameNetwork(cfg, BoardEncoder(cfg), FromToDecoder(cfg))


def _build_go(**overrides) -> nn.Module:
    """Placement board games (Go/Gomoku/Othello): point + pass moves."""
    from model.game_network import GameNetwork
    from model.games.board import BoardConfig, BoardEncoder, PlacementDecoder

    overrides.setdefault("board_h", 19)
    overrides.setdefault("board_w", 19)
    # own stones / opponent stones / empty / simple-ko forbidden point —
    # matches the SGF converter plane layout (gmdata go convert).
    overrides.setdefault("piece_channels", 4)
    # Converter v2 also writes obs.globals_seq [T,6] (mover is black, komi
    # from the mover's view, komi known, area scoring, handicap, previous
    # move was a pass) and shard meta {global_features: 6, board_size: N}.
    # The default stays 0 = the legacy (v1, planes-only) width: checkpoints
    # saved board_global_features=0 explicitly, and fresh runs on v2 shards
    # take 6 from the shard meta (train_il_game META_ARCH_KEYS); a net with
    # 0 ignores globals_seq.
    overrides.setdefault("board_global_features", 0)
    overrides.setdefault("n_promotions", 0)
    cfg = BoardConfig(**overrides)
    return GameNetwork(cfg, BoardEncoder(cfg), PlacementDecoder(cfg))


def _build_othello(**overrides) -> nn.Module:
    """Othello/Reversi: 8×8 placement + pass.  3 planes: own discs /
    opponent discs / empty, from the side-to-move perspective (channel
    swap, no geometric flip) — matches
    ``data_processing.games.othello.rules.board_planes`` (same layout as ``gmdata othello convert``).

    The cheapest full-pipeline smoke-test game (WTHOR human data, trivial
    exact rules) — see docs/game-data-sources.md.
    """
    from model.game_network import GameNetwork
    from model.games.board import BoardConfig, BoardEncoder, PlacementDecoder

    overrides.setdefault("board_h", 8)
    overrides.setdefault("board_w", 8)
    overrides.setdefault("piece_channels", 3)
    overrides.setdefault("board_global_features", 0)
    overrides.setdefault("n_promotions", 0)
    cfg = BoardConfig(**overrides)
    return GameNetwork(cfg, BoardEncoder(cfg), PlacementDecoder(cfg))


def _build_gomoku(**overrides) -> nn.Module:
    """Gomoku: 15×15 placement.  Gomoku has no pass move, and the placement
    legal mask cannot suppress the pass Bernoulli — ``allow_pass=False``
    removes the pass head so the action space is pure placement."""
    from model.game_network import GameNetwork
    from model.games.board import BoardConfig, BoardEncoder, PlacementDecoder

    overrides.setdefault("board_h", 15)
    overrides.setdefault("board_w", 15)
    overrides.setdefault("piece_channels", 3)
    overrides.setdefault("board_global_features", 0)
    overrides.setdefault("n_promotions", 0)
    overrides.setdefault("allow_pass", False)
    cfg = BoardConfig(**overrides)
    return GameNetwork(cfg, BoardEncoder(cfg), PlacementDecoder(cfg))


def _build_shogi(**overrides) -> nn.Module:
    """Shogi: 9×9 board + pieces in hand (move-or-drop action space)."""
    from model.game_network import GameNetwork
    from model.games.shogi import ShogiConfig, ShogiDecoder, ShogiEncoder

    cfg = ShogiConfig(**overrides)
    return GameNetwork(cfg, ShogiEncoder(cfg), ShogiDecoder(cfg))


def _build_lux(**overrides) -> nn.Module:
    """Lux AI–style: resource grid + units, all units act simultaneously
    (factored per-unit policy)."""
    from model.game_network import GameNetwork
    from model.games.lux import LuxConfig, LuxDecoder, LuxEncoder

    cfg = LuxConfig(**overrides)
    return GameNetwork(cfg, LuxEncoder(cfg), LuxDecoder(cfg))


def _build_mahjong(**overrides) -> nn.Module:
    """Riichi mahjong: 34 tile-kind + 4 player + 1 table tokens; typed
    action decoder (discard/riichi/call/hora) with a tile pointer."""
    from model.game_network import GameNetwork
    from model.games.mahjong import MahjongConfig, MahjongDecoder, MahjongEncoder

    cfg = MahjongConfig(**overrides)
    return GameNetwork(cfg, MahjongEncoder(cfg), MahjongDecoder(cfg))


def _build_pokemon(**overrides) -> nn.Module:
    """Pokémon singles: 12 pokemon slot tokens + field token; move-or-switch
    decoder with hash-bucket species embeddings."""
    from model.game_network import GameNetwork
    from model.games.pokemon import PokemonConfig, PokemonDecoder, PokemonEncoder

    cfg = PokemonConfig(**overrides)
    return GameNetwork(cfg, PokemonEncoder(cfg), PokemonDecoder(cfg))


def _build_clash_royale(**overrides) -> nn.Module:
    from model.game_network import GameNetwork
    from model.games.clash_royale import (
        CardPlacementDecoder,
        ClashRoyaleConfig,
        ClashRoyaleEncoder,
    )

    cfg = ClashRoyaleConfig(**overrides)
    return GameNetwork(cfg, ClashRoyaleEncoder(cfg), CardPlacementDecoder(cfg))


def _build_hearthstone(**overrides) -> nn.Module:
    from model.game_network import GameNetwork
    from model.games.hearthstone import (
        EntityActionDecoder,
        HearthstoneConfig,
        HearthstoneEncoder,
    )

    cfg = HearthstoneConfig(**overrides)
    return GameNetwork(cfg, HearthstoneEncoder(cfg), EntityActionDecoder(cfg))


def _build_starcraft(**overrides) -> nn.Module:
    from model.game_network import GameNetwork
    from model.games.starcraft import RTSDecoder, StarCraftConfig, StarCraftEncoder

    cfg = StarCraftConfig(**overrides)
    return GameNetwork(cfg, StarCraftEncoder(cfg), RTSDecoder(cfg))


GAME_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "generals": _build_generals,
    "chess": _build_chess,
    "go": _build_go,
    "othello": _build_othello,
    "gomoku": _build_gomoku,
    "shogi": _build_shogi,
    "lux": _build_lux,
    "mahjong": _build_mahjong,
    "pokemon": _build_pokemon,
    "clash_royale": _build_clash_royale,
    "hearthstone": _build_hearthstone,
    "starcraft": _build_starcraft,
}


def _build_rules_game(game: str, **overrides) -> nn.Module:
    """A gmrules game (model/games/gmrules/<game>.py): the generic candidate-action
    adapter (model/games/candidate.py), or the entry's own builder."""
    from model.games.gmrules import resolve, rules_games

    entry = rules_games()[game]
    if entry.is_candidate:
        from model.games.candidate import build_candidate_network

        return build_candidate_network(game, **overrides)
    return resolve(entry.builder)(**{**entry.config, **overrides})


def _register_rules_games() -> None:
    """Every game of model/games/gmrules/ (discovered: no edit here per game)."""
    from model.games.gmrules import rules_games

    for name in rules_games():
        if name in GAME_REGISTRY:
            raise ValueError(f"gmrules game {name!r} clashes with a registered game")
        GAME_REGISTRY[name] = functools.partial(_build_rules_game, name)


_register_rules_games()


def register_game(name: str, builder: Callable[..., nn.Module]) -> None:
    if name in GAME_REGISTRY:
        raise ValueError(f"game {name!r} already registered")
    GAME_REGISTRY[name] = builder


def build_game_network(game: str, **config_overrides) -> nn.Module:
    """Build a ready-to-train network for a registered game.

    ``config_overrides`` are forwarded to the game's config dataclass
    (e.g. ``build_game_network("chess", d_model=256)``, or
    ``build_game_network("go", board_h=9, board_w=9)`` for 9×9 Go).
    """
    try:
        builder = GAME_REGISTRY[game]
    except KeyError:
        raise KeyError(
            f"unknown game {game!r}; registered: {sorted(GAME_REGISTRY)}"
        ) from None
    return builder(**config_overrides)
