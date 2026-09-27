"""Registry of the gmrules games on the Python side (games whose rules and frame
encoders live in the Rust crate data_processing/rs/gmrules; simulated by gmsim,
converted by gmdata).

One module per game in this package, defining ``GAME = RulesGame(...)``.  The
modules are discovered automatically, and every consumer includes every game found
here: the model registry (``build_game_network``), training/train_il_game.py
(LABEL_KEYS, META_ARCH_KEYS, LEGAL_FROM_OBS), env/vec_game.py (VEC_ENV_REGISTRY,
through env/vec_gmsim.py) and training/train_rl_joint.py (GAME_SPECS).  Adding a
game on the Python side = adding its module; no shared file changes.

Candidate-action games (the Rust spec is ``FrameSpec::candidate``) need no more
than a name — the generic adapter (model/games/candidate.py), the frame dims from
the simulator's spec / the shard meta (:data:`CANDIDATE_META_ARCH`), the generic
RL spec::

    # model/games/gmrules/kuhn_poker.py
    from model.games.gmrules import RulesGame
    GAME = RulesGame(name="kuhn_poker", sim_config={"players": 2})

Custom-schema games (their own obs / legal / target tensors) bring their model
``builder`` and the decoder's plumbing (e.g. a board game on the from-to decoder,
model/games/gmrules/breakthrough.py)::

    GAME = RulesGame(name="breakthrough", builder=build_breakthrough,
                     label_keys=("from_idx", "to_idx"),
                     meta_arch={"board_size": ("board_h", "board_w")},
                     sample_keys={"action_from": "from_idx", "action_to": "to_idx"},
                     rl_spec=FromToSpec)

``name`` is the gmrules registry name (= the gm-shard ``game``).  This module is
torch-free (imported by the registries); builders / specs are resolved lazily
(``"module:attr"`` strings are accepted for every callable field).
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

#: shard meta key → CandidateConfig field(s) (train_il_game META_ARCH_KEYS of every
#: candidate game; written by gmdata traj.rs from the game's FrameSpec meta)
CANDIDATE_META_ARCH: Dict[str, tuple] = {
    "max_entities": ("cand_max_entities",),
    "entity_features": ("cand_entity_features",),
    "entity_types": ("cand_entity_types",),
    "global_features": ("cand_global_features",),
    "max_actions": ("cand_max_actions",),
    "action_features": ("cand_action_features",),
}

#: decoder ``sample()`` key → action key of the candidate decoder
CANDIDATE_SAMPLE_KEYS: Dict[str, str] = {"action_idx": "action_idx"}

Ref = Union[None, str, Callable[..., Any]]


def resolve(ref: Ref) -> Optional[Callable[..., Any]]:
    """A callable, or a ``"package.module:attr"`` reference to one (lazy imports keep
    this registry torch-free)."""
    if ref is None or callable(ref):
        return ref
    mod, _, attr = str(ref).partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass(frozen=True)
class RulesGame:
    """One gmrules game's Python-side registration.

    name            gmrules registry name (gm-shard ``game``, gmsim game)
    sim             gmsim simulates it (RL env, ``make_vec_env(name)``)
    sim_config      default gmrules config of the env / the dims (JSON-able dict);
                    ``--env_kw <game>:key=value`` overrides it per run
    description     one line for listings

    Model — candidate schema (``builder`` None): the generic CandidateEncoder /
    CandidateDecoder, optionally customised by
      config        CandidateConfig defaults for this game (frame dims default to
                    the simulator's spec for ``sim_config``, a fresh IL run takes
                    them from the shard meta)
      config_class  a CandidateConfig subclass (extra fields of custom classes)
      encoder / decoder   ObservationEncoder / ActionDecoder classes called with cfg
    custom schema:
      builder       ``(**overrides) -> GameNetwork`` (the model registry entry)

    Trainer / env plumbing (defaults = the candidate adapter's):
      label_keys      train_il_game LABEL_KEYS (teacher-forcing labels, tgt keys)
      meta_arch       train_il_game META_ARCH_KEYS (shard meta key → config fields);
                      default CANDIDATE_META_ARCH for candidate games, {} otherwise
      legal_from_obs  train_il_game LEGAL_FROM_OBS (decoder legal dict from obs)
      sample_keys     decoder ``sample()`` key → the game's action key (the env's
                      ACTION_KEYS); default: the candidate decoder's
      rl_spec         zero-argument factory of the train_rl_joint spec (default:
                      CandidateSpec for candidate games; custom schemas whose
                      decoder teacher-forces labels need their own)
    """

    name: str
    sim: bool = True
    sim_config: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""
    config: Mapping[str, Any] = field(default_factory=dict)
    config_class: Ref = None
    encoder: Ref = None
    decoder: Ref = None
    builder: Ref = None
    label_keys: Tuple[str, ...] = ()
    meta_arch: Optional[Mapping[str, Tuple[str, ...]]] = None
    legal_from_obs: Ref = None
    sample_keys: Optional[Mapping[str, str]] = None
    rl_spec: Ref = None

    @property
    def is_candidate(self) -> bool:
        """The game plays through the generic candidate-action adapter."""
        return self.builder is None

    def meta_arch_keys(self) -> Dict[str, Tuple[str, ...]]:
        if self.meta_arch is not None:
            return dict(self.meta_arch)
        return dict(CANDIDATE_META_ARCH) if self.is_candidate else {}

    def action_keys_map(self) -> Dict[str, str]:
        """decoder sample() key → action key (the env's ACTION_KEYS)."""
        if self.sample_keys is not None:
            return dict(self.sample_keys)
        return dict(CANDIDATE_SAMPLE_KEYS)

    def sim_spec(self, sim_config: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """The simulator's frame spec for ``sim_config`` (default: this entry's)."""
        import gmsim

        cfg = dict(self.sim_config) if sim_config is None else dict(sim_config)
        return gmsim.game_info(self.name, cfg)["spec"]

    def sim_dims(self) -> Dict[str, Any]:
        """Config values of the meta-arch keys from the simulator's spec meta for
        ``sim_config`` ({} without a simulator)."""
        if not self.sim:
            return {}
        meta = self.sim_spec()["meta"]
        out: Dict[str, Any] = {}
        for meta_key, cfg_keys in self.meta_arch_keys().items():
            if meta_key in meta:
                for k in cfg_keys:
                    out[k] = meta[meta_key]
        return out


_GAMES: Optional[Dict[str, RulesGame]] = None


def rules_games() -> Dict[str, RulesGame]:
    """Every registered gmrules game (discovered once from this package)."""
    global _GAMES
    if _GAMES is None:
        found: Dict[str, RulesGame] = {}
        for mod in sorted(m.name for m in pkgutil.iter_modules(__path__)):
            game = getattr(importlib.import_module(f"{__name__}.{mod}"), "GAME", None)
            if game is None:
                continue
            if not isinstance(game, RulesGame):
                raise TypeError(f"{__name__}.{mod}.GAME is not a RulesGame")
            if game.name in found:
                raise ValueError(f"gmrules game {game.name!r} registered twice")
            found[game.name] = game
        _GAMES = found
    return _GAMES


def candidate_games() -> Dict[str, RulesGame]:
    """The registered games that use the candidate-action adapter."""
    return {k: g for k, g in rules_games().items() if g.is_candidate}


def is_rules_game(name: str) -> bool:
    return name in rules_games()
