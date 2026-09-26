"""Registry of the candidate-action games (gmrules games played through
model/games/candidate.py).

One module per game in this package, defining ``GAME = CandidateGame(...)``.  The
modules are discovered automatically, and every consumer includes every game found
here: the model registry (``build_game_network``), training/train_il_game.py
(LABEL_KEYS, META_ARCH_KEYS), env/vec_game.py (VEC_ENV_REGISTRY, through
env/vec_candidate.py) and training/train_rl_joint.py (GAME_SPECS).  Adding a game
on the Python side = adding its module; no shared file changes.

    # model/games/candidates/kuhn_poker.py
    from model.games.candidates import CandidateGame
    GAME = CandidateGame(name="kuhn_poker", sim_config={"players": 2})

``name`` is the gmrules registry name (= the gm-shard ``game``).  The frame
dimensions default to the simulator's FrameSpec for ``sim_config`` (gmsim), and a
fresh IL run takes them from the shard meta (:data:`CANDIDATE_META_ARCH`), so an
entry only states what differs from :class:`~model.games.candidate.CandidateConfig`
and the generic classes.  This module is torch-free (imported by the registries).
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

#: shard meta key → CandidateConfig field(s) (train_il_game META_ARCH_KEYS of every
#: candidate game; written by gmdata cand.rs from the game's FrameSpec)
CANDIDATE_META_ARCH: Dict[str, tuple] = {
    "max_entities": ("cand_max_entities",),
    "entity_features": ("cand_entity_features",),
    "entity_types": ("cand_entity_types",),
    "global_features": ("cand_global_features",),
    "max_actions": ("cand_max_actions",),
    "action_features": ("cand_action_features",),
}


@dataclass(frozen=True)
class CandidateGame:
    """One candidate-action game's Python-side registration.

    name          gmrules registry name (gm-shard ``game``, gmsim game)
    config        CandidateConfig defaults for this game (d_model etc. come from
                  the run; frame dims default to the simulator's spec)
    sim           gmsim simulates it (RL env, ``make_vec_env(name)``)
    sim_config    default gmrules config of the env / the dims (JSON-able dict);
                  ``--env_kw <game>:key=value`` overrides it per run
    config_class  a CandidateConfig subclass (extra fields of custom classes)
    encoder       ObservationEncoder class (default CandidateEncoder), called with cfg
    decoder       ActionDecoder class (default CandidateDecoder), called with cfg
    rl_spec       zero-argument factory of the train_rl_joint spec (default
                  CandidateSpec)
    """

    name: str
    config: Mapping[str, Any] = field(default_factory=dict)
    sim: bool = True
    sim_config: Mapping[str, Any] = field(default_factory=dict)
    config_class: Optional[type] = None
    encoder: Optional[Callable[..., Any]] = None
    decoder: Optional[Callable[..., Any]] = None
    rl_spec: Optional[Callable[[], Any]] = None
    description: str = ""

    def sim_dims(self) -> Dict[str, Any]:
        """CandidateConfig frame dims of the simulator's FrameSpec for
        ``sim_config`` ({} without a simulator)."""
        if not self.sim:
            return {}
        import gmsim

        spec = gmsim.game_info(self.name, dict(self.sim_config))["spec"]
        return {cfg_keys[0]: spec[meta_key] for meta_key, cfg_keys in CANDIDATE_META_ARCH.items()}


_GAMES: Optional[Dict[str, CandidateGame]] = None


def candidate_games() -> Dict[str, CandidateGame]:
    """Every registered candidate game (discovered once from this package)."""
    global _GAMES
    if _GAMES is None:
        found: Dict[str, CandidateGame] = {}
        for mod in sorted(m.name for m in pkgutil.iter_modules(__path__)):
            game = getattr(importlib.import_module(f"{__name__}.{mod}"), "GAME", None)
            if game is None:
                continue
            if not isinstance(game, CandidateGame):
                raise TypeError(f"{__name__}.{mod}.GAME is not a CandidateGame")
            if game.name in found:
                raise ValueError(f"candidate game {game.name!r} registered twice")
            found[game.name] = game
        _GAMES = found
    return _GAMES


def is_candidate_game(name: str) -> bool:
    return name in candidate_games()
