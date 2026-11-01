"""Checkpoint schema: the version stamp the trainers write, and the rules that read the
checkpoints written before it existed.

Every checkpoint is a plain dict (torch.save).  Those written by code at or after this
module carry ``schema_version`` (:data:`SCHEMA_VERSION_KEY`); older ones have no such key
and read as version :data:`LEGACY_SCHEMA_VERSION` (0).

Versions
    0  legacy — no stamp.  Config dicts may predate fields whose defaults have since
       changed (:data:`LEGACY_CONFIG_DEFAULTS`), and a generals config may lack
       ``memory_vision``, which only the whole checkpoint tells
       (:func:`legacy_memory_vision`).
    1  the first stamped version: no format change.  Trainers write their current config
       dicts, but exports of older checkpoints can retain incomplete legacy configs.
       The stamp identifies the writer's schema, not the age or completeness of a config.

Readers apply the legacy rules to every checkpoint whatever its version — they are the
identity on a complete config dict, and the version is recorded so a later change can be
keyed on it.  Adding a field with a new default still needs a
:data:`LEGACY_CONFIG_DEFAULTS` entry; a version stamp must not bypass those rules.

Older code reading a stamped checkpoint: the stamp is one more int-valued key in the
dict, which every loader ignores (none enumerates the top-level keys).
"""

import ast
from typing import Any, Mapping, MutableMapping, Optional, TypeVar

SCHEMA_VERSION_KEY = "schema_version"
#: what the trainers write today
CHECKPOINT_SCHEMA_VERSION = 1
#: what an unstamped checkpoint reads as
LEGACY_SCHEMA_VERSION = 0

_M = TypeVar("_M", bound=MutableMapping[str, Any])


def stamp_schema_version(ckpt: _M) -> _M:
    """Set :data:`SCHEMA_VERSION_KEY` to :data:`CHECKPOINT_SCHEMA_VERSION` on the checkpoint
    dict about to be saved (in place; returned, for
    ``torch.save(stamp_schema_version(ck), path)``)."""
    ckpt[SCHEMA_VERSION_KEY] = CHECKPOINT_SCHEMA_VERSION
    return ckpt


def schema_version(ckpt: Mapping[str, Any]) -> int:
    """The version a loaded checkpoint dict was written at: its stamp, or
    :data:`LEGACY_SCHEMA_VERSION` for a checkpoint from before the stamp existed."""
    v = ckpt.get(SCHEMA_VERSION_KEY, LEGACY_SCHEMA_VERSION)
    if isinstance(v, bool) or not isinstance(v, int) or v < 0:
        raise ValueError(f"checkpoint {SCHEMA_VERSION_KEY} must be a non-negative int, got {v!r}")
    return v


# ----------------------------------------------------------------------
# Legacy config dicts: fields that predate a default
# ----------------------------------------------------------------------

#: values a checkpoint config written before a field existed must get so the
#: rebuilt network matches the saved weights (current defaults differ).
LEGACY_CONFIG_DEFAULTS = {
    "core_type": "trm",
    "trm_arch": "legacy",
    "halt_exploration_prob": 0.0,
    "qk_norm": False,
    # Generals sighted-mountain memory: off for a config dict on its own —
    # the IL records have none and GeneralsVecEnv / train_rl_joint defaulted
    # off.  The training/train_rl.py checkpoints that played with it on are
    # recognised from the whole checkpoint (legacy_memory_vision, applied by
    # checkpoint_config / checkpoint_config_dict).
    "memory_vision": False,
    # conv observation encoders: BatchNorm2d (model/components/encoder.py)
    "encoder_norm": "batch",
    # Lux adapter (model/games/lux.py): no unit-cell PE, the MLP sap head
    "lux_unit_cell_pe": False,
    "lux_sap_head": "mlp",
    # Clash Royale adapter (model/games/clash_royale.py): frame-index wait PE;
    # the 4 reconstructed hand slots (8 features each) instead of the 8 deck
    # tokens with the causal hand belief (crdata converter_version 3)
    "cr_wait_time": "frame",
    "cr_card_tokens": "hand",
    "card_feature_dim": 8,
    # ... and one token per arena cell, no recent-play / opponent-deck tokens
    "cr_arena_patch": 1,
    "cr_recent_tokens": 0,
    "cr_opp_tokens": 0,
    # Pokémon adapter (model/games/pokemon.py): the revealed-move slot head; crc32
    # hash buckets of the move / species names (converter v1 / v2) instead of the
    # exact Showdown vocabularies (converter v3)
    "pokemon_move_head": "slot",
    "pokemon_move_vocab": 0,
    "pokemon_species_vocab": 0,
}

#: LEGACY_CONFIG_DEFAULTS keys of ONE game's config class → a marker field
#: every config of that game ever saved carries.  They are applied only to
#: dicts that have the marker, so other games' configs do not grow keys
#: their class rejects (build_net_tolerant would drop them, loudly).
LEGACY_CONFIG_SCOPE = {
    "lux_unit_cell_pe": "n_unit_actions",
    "lux_sap_head": "n_unit_actions",
    "cr_wait_time": "cr_global_features",
    "cr_card_tokens": "cr_global_features",
    "card_feature_dim": "cr_global_features",
    "cr_arena_patch": "cr_global_features",
    "cr_recent_tokens": "cr_global_features",
    "cr_opp_tokens": "cr_global_features",
    # every loadable pokemon config: the slot head needs pokemon_move_buckets
    "pokemon_move_head": "pokemon_move_buckets",
    "pokemon_move_vocab": "pokemon_move_buckets",
    "pokemon_species_vocab": "pokemon_move_buckets",
}


def upgrade_config_dict(d: dict) -> dict:
    """Checkpoint config dict → kwargs for the current config class: fields
    missing from an old checkpoint take their LEGACY value, not the new
    default (e.g. the TRM block layout)."""
    out = dict(d)
    for k, v in LEGACY_CONFIG_DEFAULTS.items():
        marker = LEGACY_CONFIG_SCOPE.get(k)
        if marker is None or marker in d:
            out.setdefault(k, v)
    return out


# ----------------------------------------------------------------------
# Legacy rules that need the whole checkpoint, not just its config dict
# ----------------------------------------------------------------------

#: the config field that landed 24 minutes after GeneralsEnv's sighted-
#: mountain memory went in, ON by default (2026-08-11: memory c56c5e2 at
#: 21:07, joint_decoder 164e37d at 21:32).  training/train_rl.py never set
#: the env's memory_vision, so a train_rl config dict carrying this key was
#: written by code whose env had the memory on; one without it was written
#: by older code — which, but for that 24-minute window, had no memory
#: (phases 1-7, run-4/4b/4c, run5b; see legacy_memory_vision's caveat).
MEMORY_VISION_ERA_FIELD = "joint_decoder"


def is_train_rl_checkpoint(ckpt: Mapping[str, Any]) -> bool:
    """A training/train_rl.py checkpoint (``rl_checkpoint_iter*.pt``,
    ``rl_ema_iter*.pt``) or a scripts/distill_joint_head.py seed (distilled
    in the same env): one ``config`` next to an ``iteration`` counter — IL
    checkpoints count ``epoch`` / ``step``, train_rl_game ``iter``, the
    multi-game trainers write ``configs`` and single-game exports ``game``."""
    return ("config" in ckpt and "iteration" in ckpt
            and "game" not in ckpt and "games" not in ckpt)


def legacy_memory_vision(ckpt: Mapping[str, Any]) -> bool:
    """cfg.memory_vision of a generals checkpoint written before the field
    existed: the observation it was actually trained on.

    * training/train_rl.py checkpoints (:func:`is_train_rl_checkpoint`)
      whose config carries :data:`MEMORY_VISION_ERA_FIELD`: True —
      GeneralsEnv defaulted to the memory from 2026-08-11 until this field
      existed, and train_rl.py never switched it off.  Older train_rl
      configs predate the memory: False.
    * train_rl_joint checkpoints: what ``--env_kw generals:memory_vision=``
      gave the env (parsed as the trainer parsed it), else False —
      GeneralsVecEnv defaulted off and GeneralsSpec never set it.
    * everything else — the IL checkpoints (train_il, train_il_game,
      train_multi_game and their exports): False, the IL records have no
      memory.

    A blanket legacy True (GeneralsEnv's old default) would feed every
    IL-only network an observation it never saw; a blanket False would take
    the memory away from the train_rl lineages that trained with it.

    Caveat: the marker dates the CODE that wrote a config, not the run.  A
    train_rl process started from code between c56c5e2 and 164e37d
    (2026-08-11 21:07-21:31) played with the memory but wrote no
    joint_decoder field — most likely the run5c lineage (job 45756, launched
    from 34f8b07) — and resolves to False here, as does any
    checkpoint such a process wrote before a requeue onto newer code.  Its
    checkpoints cannot be told from the pre-memory run5b ones by content:
    record the setting before resuming or playing them, e.g.
    ``ck = torch.load(p, weights_only=False); ck["config"]["memory_vision"]
    = True; torch.save(ck, p)`` (or pass training/train_rl.py
    --memory_vision 1)."""
    if is_train_rl_checkpoint(ckpt):
        return MEMORY_VISION_ERA_FIELD in ckpt["config"]
    if ckpt.get("stage") == "rl_joint":
        on = False
        for item in (ckpt.get("args") or {}).get("env_kw") or ():
            game, _, kv = str(item).partition(":")
            key, sep, raw = kv.partition("=")
            if game == "generals" and key == "memory_vision" and sep:
                try:
                    val = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    val = raw                    # the trainer kept the string
                on = bool(val)                   # the last one wins, as there
        return on
    return False


def checkpoint_config_dict(ckpt: Mapping[str, Any], game: Optional[str] = None) -> dict:
    """A checkpoint's config dict — ``config`` of a single-game checkpoint,
    ``configs[game]`` of a multi-game one — upgraded like
    :func:`upgrade_config_dict`, plus the legacy rules that need the whole
    checkpoint (a generals network's memory_vision:
    :func:`legacy_memory_vision`)."""
    d = dict(ckpt["config"] if game is None else ckpt["configs"][game])
    if "memory_vision" not in d and game in (None, "generals"):
        d["memory_vision"] = legacy_memory_vision(ckpt)
    return upgrade_config_dict(d)
