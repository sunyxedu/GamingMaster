"""Generic imitation-learning trainer for any registered game adapter.

Trains ``build_game_network(<game>)`` on shards produced by the
per-game converters (see data_processing/games/), using the decoder's
own ``il_loss`` plus the value loss (MSE for the scalar head, HL-Gauss
cross-entropy with ``--override hl_gauss_value=True``; see value_losses).

    uv run python -m training.train_il_game --game othello \
        --data_dir data/othello/train --epochs 10 --batch_size 8

generals runs here too (gio record directories or ``gmdata generals shard``
gm-shards; its decoder's il_loss is train_il.py's ImitationLoss, so the
objective is the same), and training/train_il.py stays its dedicated
trainer (world-model objective, multi-node plumbing, legacy checkpoints).

Single-process on purpose: the multi-node story (manual grad allreduce,
NaN voting) stays in training/train_il.py until a non-generals game actually
trains at that scale.

Engine supervision is used wherever the shards carry it (tgt naming in
data_processing/games/dataset.py): soft policy targets ``<head>_probs``
(teacher KL through the decoder's ``il_head_terms``, --soft_target_weight)
and ``engine_value`` targets (--engine_value_weight); --il_legal_mask feeds
the shards' ``legal_<key>`` masks to the decoder.  Shards older than the
frozen pre-training schema (SCHEMA_FREEZE_CONVERTER_VERSION) train with a
warning, or not at all under --require_frozen_schema.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import os
import random
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.games.dataset import (
    collate_game_sequences,
    open_game_dataset,
    training_sampler,
)
from data_processing.games.generals import GeneralsGioDataset, gio_dataset_advice
from model.components.core import update_moe_balance
from model.components.encoder import describe_encoder_norm
from model.config import checkpoint_config_dict, upgrade_config_dict
from model.games import build_game_network
from model.games import generals as generals_adapter
from model.games import lux as lux_adapter
from model.games.base import densify_soft_target, soft_cross_entropy
from model.games.clash_royale import cr_legal_from_obs
from model.games.gmrules import resolve, rules_games
from model.losses import hl_gauss_cross_entropy
from model.optim import (
    TRM_BETAS,
    TRM_EMA_RATE,
    TRM_LR,
    TRM_LR_MIN_RATIO,
    TRM_WARMUP_STEPS,
    TRM_WEIGHT_DECAY,
    AdamATan2,
    ModelEMA,
    trm_lr_lambda,
)
from model.rl_utils import load_model_state, optim_param_groups
from training.deep_supervision import Q_HALT_WEIGHT, BatchCarry, split_rows
from training.diagnostics import attention_probe, format_attention, logit_lse_absmax, qk_caps

# Teacher-forcing conditioning labels per game: the tgt keys the decoder
# reads from ``labels`` in decode() (−1 → sample from its own policy).
LABEL_KEYS = {
    # derived from count / src_y / src_x (LABELS_FROM_TGT), not stored
    "generals": ("source_y", "source_x"),
    "chess": ("from_idx", "to_idx"),
    "othello": (),
    "gomoku": (),
    "go": (),
    "shogi": ("dropped", "from_idx", "to_idx", "piece_idx"),
    "lux": (),
    "starcraft": ("action_type", "actor_idx"),
    "hearthstone": ("action_type", "actor_idx"),
    # the patch placement decoder (cr_arena_patch > 1) also conditions on the cell's patch
    "clash_royale": ("card_slot", "place_idx"),
    "mahjong": ("action_type",),
    # active_idx is state (which own slot is on the field), delivered via
    # labels because the observer-view converter stores it in tgt.
    "pokemon": ("action_type", "active_idx"),
    # the cell head conditions on the clicked patch (model/games/arc3.py)
    "arc3": ("click_idx",),
    # the argument heads condition on the button (model/games/hok.py)
    "hok": ("button_idx",),
}

# STATE keys are facts about the position, not decisions: the decoder
# must ALWAYS receive them.  build_labels never drops these to −1 under
# the teacher-forcing schedule — pokemon's decode() would clamp −1 to
# slot 0 and condition the move/tera heads on the wrong Pokémon.
STATE_LABEL_KEYS = {
    "pokemon": ("active_idx",),
}

# Conditioning labels computed from the stored targets instead of read from
# tgt keys of the same name: generals teacher-forces the source cell of move
# frames only — pass / dropped-move frames get −1 and the decoder samples
# its own source there (training/train_il.py's rule; forcing a label onto a
# pass frame would feed its pass head a conditioning it never sees in play).
LABELS_FROM_TGT = {
    "generals": generals_adapter.source_labels,
}

# Per-game legality derived from the observation itself: some decoders
# need ``legal`` at decode time even during IL (lux's il_loss masks by
# pred["unit_present"], which decode() reads from legal["unit"]; the
# starcraft actor mask keeps tf_prob<1 self-sampling off empty unit
# slots — at tf_prob=1.0 it is never consulted).
# Applied inside run_batch, so training and evaluate() share the mask.
LEGAL_FROM_OBS = {
    # source mask (+ direction mask when present), as train_il.py passes it
    "generals": generals_adapter.legal_from_obs,
    # occupied slots + the episode's sap range (model.games.lux.lux_legal_from_obs)
    "lux": lux_adapter.lux_legal_from_obs,
    "starcraft": lambda obs: {"actor": obs["unit_mask"].bool()},
    # entity_mask (True = slot occupied) keeps tf_prob<1 actor
    # self-sampling off empty entity slots, same as starcraft above.
    "hearthstone": lambda obs: {"actor": obs["entity_mask"].bool()},
    # move_ids is state (revealed moves of the active mon) consumed by
    # the move head at decode time; guarded for pre-move_ids shards.
    "pokemon": lambda obs: (
        {"move_ids": obs["move_ids"]} if "move_ids" in obs else None
    ),
    # deck-token shards: a deck card is playable iff P(in hand) > 0 and
    # affordable (expand.py); keeps self-sampled cards (tf_prob < 1, waited
    # frames) on cards the player may hold.  Legacy hand shards: None.
    "clash_royale": cr_legal_from_obs,
}

# Shard-level meta facts (written by the converters) that must match the
# model config: {game: {meta_key: (config_field, ...)}}.  Fresh runs
# default the config from the data; every run hard-fails on a mismatch.
META_ARCH_KEYS = {
    "gomoku": {"board_size": ("board_h", "board_w")},
    # crdata writes the arena geometry and the card vocabulary of the
    # catalog it converted with (n_cards grows when cards are added), and
    # (converter_version >= 3) the card tokens: 8 deck tokens with the causal
    # hand belief ("deck") — older shards imply the legacy "hand" slots
    "clash_royale": {**{k: (k,) for k in (
        "arena_h", "arena_w", "arena_channels", "hand_size", "n_cards",
        "card_feature_dim", "cr_global_features", "deck_size")},
        "card_tokens": ("cr_card_tokens",)},
    # converter v2 records the field-token width (v1: 20; v2 appends the
    # forced-switch / team-preview request flags → 22); v1 / v2 the hash
    # buckets of the move / species names, v3 instead the sizes of the exact
    # Showdown vocabularies (data_processing/games/pokemon; 0 = hashed)
    "pokemon": {"move_buckets": ("pokemon_move_buckets",),
                "species_buckets": ("pokemon_species_buckets",),
                "field_features": ("pokemon_field_features",),
                "move_vocab": ("pokemon_move_vocab",),
                "species_vocab": ("pokemon_species_vocab",),
                # converter v4: every slot's revealed item / ability / tera type
                "item_vocab": ("pokemon_item_vocab",),
                "ability_vocab": ("pokemon_ability_vocab",),
                "type_vocab": ("pokemon_type_vocab",),
                # converter v5: the per-slot level, the battle's format
                "poke_features": ("pokemon_feature_dim",),
                "format_vocab": ("pokemon_format_vocab",)},
    # gio records (GeneralsGioDataset.shard_meta / `gmdata generals shard`);
    # with turn_globals the net has TURN_GLOBAL_FEATURES more globals than
    # the data (meta_arch_value), appended at batch time (OBS_HOOKS)
    "generals": {k: (k,) for k in (
        "grid_h", "grid_w", "grid_channels", "global_features")},
    # frozen pre-training schema: the converters record the width of their
    # obs.globals_seq (go: side-to-move / komi; chess: + repetition flags;
    # shogi: + repetition counts; lux: + env params) and lux its sap range
    # (the sap-offset head's (2r+1)² classes)
    "go": {"global_features": ("board_global_features",),
           "board_size": ("board_h", "board_w")},
    "chess": {"global_features": ("board_global_features",)},
    "shogi": {"global_features": ("shogi_global_features",)},
    "lux": {"global_features": ("lux_global_features",),
            "sap_range": ("lux_sap_range",)},
    # ARC-AGI-3 trajectories (scripts/arc3_solve.py): the globals width and the grid
    "arc3": {"global_features": ("arc_global_features",), "grid": ("arc_grid",)},
    # Honor of Kings 1v1 (gmdata hok convert): the research environment's feature width
    "hok": {"features": ("hok_features",)},
    # riichi mahjong (gmdata mahjong convert v4): the tile planes' width (v3: 16; v4 adds
    # each opponent's public discard history → 37)
    "mahjong": {"tile_features": ("mahjong_tile_feature_dim",)},
}

# gmrules games (one module each in model/games/gmrules/): candidate-action games
# are one categorical over the frame's candidates (nothing to teacher-force) with
# the frame dimensions from the shard meta (written by gmdata traj.rs from the
# FrameSpec); custom-schema games declare their own tables in their entry.
for _game, _entry in rules_games().items():
    LABEL_KEYS.setdefault(_game, tuple(_entry.label_keys))
    META_ARCH_KEYS.setdefault(_game, _entry.meta_arch_keys())
    if _entry.legal_from_obs is not None:
        LEGAL_FROM_OBS.setdefault(_game, resolve(_entry.legal_from_obs))

# The value a META_ARCH_KEYS meta key implies for shards written BEFORE the
# converter recorded it (their meta lacks the key): check_meta_arch holds an
# old-generation shard to the architecture it was made for, so a new-schema
# checkpoint on old shards (or an old one on new shards) fails loudly.
# These equal the config dataclass / registry defaults, so a fresh run on
# old shards builds the legacy architecture without injecting anything
# (apply_meta_arch_defaults only reads keys the meta actually has) — except
# (1) clash_royale: its converter_version <= 2 shards carry a hindsight hand (a
# future leak), so a fresh default ("deck") net on them fails loudly instead;
# --override cr_card_tokens=hand trains the legacy net on them; (2) pokemon:
# converter v1 / v2 shards hash the move / species names into crc32 buckets
# (distinct names share ids and labels), so a fresh default (exact-vocabulary)
# net on them fails loudly (META_MISMATCH_HINTS); --override
# pokemon_move_vocab=0 pokemon_species_vocab=0 trains the legacy hashed net;
# (3) config defaults that changed with the new generation, which
# META_LEGACY_IMPLIES injects (pokemon's move head).
META_LEGACY = {
    "go": {"global_features": 0},
    "chess": {"global_features": 8},
    "shogi": {"global_features": 4},
    "lux": {"global_features": 6, "sap_range": 0},
    "clash_royale": {"card_tokens": "hand"},
    "pokemon": {"field_features": 20, "move_vocab": 0, "species_vocab": 0,
                "item_vocab": 0, "ability_vocab": 0, "type_vocab": 0,
                "poke_features": 17, "format_vocab": 0},
    "mahjong": {"tile_features": 16},
}

_POKEMON_VOCAB_HINT = (
    "pokemon move / species ids: 0 = the crc32 hash buckets of converter v1 / v2 (distinct "
    "names share an id), else the size of the exact Showdown vocabulary of converter v3 "
    "(data_processing/games/pokemon).  Re-convert old shards with the current gmdata "
    "(`gmdata pokemon convert`, module docs of data_processing/rs/gmdata/src/pokemon.rs); "
    "train the legacy hashed net on them only on purpose (--override pokemon_move_vocab=0 "
    "pokemon_species_vocab=0).  A hashed checkpoint needs a fresh pokemon adapter for v3 "
    "shards; an exact net and shards of different sizes come from different vocabulary "
    "versions (append-only): convert with the vocabulary the net was built with")

# What a META mismatch means and the ways out, where "re-convert or use a
# matching checkpoint" is not the whole story: {game: {meta_key: hint}}
# (check_meta_arch appends the hint to its error).
META_MISMATCH_HINTS = {
    "pokemon": {"move_vocab": _POKEMON_VOCAB_HINT, "species_vocab": _POKEMON_VOCAB_HINT},
}

# First converter generation of the frozen pre-training schema per game:
# older shards still train (their META_LEGACY architecture), but they lack
# the frozen inputs / targets, so the trainers warn (or, with
# --require_frozen_schema, refuse).
SCHEMA_FREEZE_CONVERTER_VERSION = {"chess": 3, "go": 2, "shogi": 2, "mahjong": 4, "lux": 3,
                                   "clash_royale": 3, "pokemon": 5}
_SCHEMA_FREEZE_CONTENT = {
    "chess": "repetition flags + legal masks",
    "go": "side-to-move/komi globals",
    "shogi": "repetition + legal masks",
    "mahjong": "exact legal masks, leak-free claim frames + il_weight; each opponent's "
               "public discard history on the tile planes (older shards count the discards "
               "per kind, losing their order, tsumogiri and the riichi safety)",
    "lux": "env-param globals + sap targets; one sample per game (older shards cut it into "
           "matches, restarting the memory the game keeps) with team 1's reflected view",
    "clash_royale": "deck tokens with the causal hand belief; older shards infer the hand "
                    "from the whole match",
    "pokemon": "exact Showdown move / species ids (older shards hash the names into "
               "colliding buckets), move-identity targets, forced-switch / lead / "
               "unlabelled-turn frames, request-phase flags + legal masks; every slot's "
               "revealed moves / item / ability / tera type (older shards show the own "
               "active's moves only); every slot's level and the battle's format (older "
               "shards: level-100 Gen 9 OU only)",
}


def schema_freeze_warning(game: str, dataset) -> Optional[str]:
    """Why ``game``'s ``dataset`` predates the frozen pre-training schema
    (SCHEMA_FREEZE_CONVERTER_VERSION), or None.  Games without an entry
    (generals' GeneralsGioDataset included) are never flagged."""
    need = SCHEMA_FREEZE_CONVERTER_VERSION.get(game)
    if need is None:
        return None
    have = getattr(dataset, "converter_version", None) or 1
    if have >= need:
        return None
    return (f"{game} shards have converter_version {have} < {need}: they predate the "
            f"frozen pre-training schema ({game}: {_SCHEMA_FREEZE_CONTENT[game]}); "
            f"re-convert with the current converter")

# Teacher-forcing probability a game trains at unless the command line says
# otherwise (train_il_game --tf_start/--tf_end, train_multi_game --tf_prob /
# --game_tf); every other game: 1.0.  generals: 0, training/train_il.py's
# default — its direction / split / pass heads then learn on the source the
# policy itself picks, as in play (teacher-forced sources are an exposure
# bias the generals recipe anneals away, e.g. --tf_start 0.4 --tf_end 0).
DEFAULT_TF = {"generals": 0.0}


def default_tf(game: str) -> float:
    return DEFAULT_TF.get(game, 1.0)


# Games the GLOBAL --max_len never crops (their full sequences are the
# default); an explicit per-game max_len (train_multi_game --game_max_len,
# eval_multi_game --game_max_len, train_il_game --max_len) still crops them
# when the network is crop-safe — :func:`check_crop_policy`.
UNCROPPABLE = frozenset({"clash_royale"})

# Networks that read ABSOLUTE frame time and so must see sequences from
# their first frame: game → predicate on the config.  A legacy clash_royale
# wait head (cr_wait_time="frame") encodes the frame index t0 .. t0+T-1 with
# t0 = 0 in GameNetwork.forward; a random window would feed late-game
# frames early-game time.  cr_wait_time="match" reads the match clock from
# the observation instead and is crop-safe.
CROP_UNSAFE = {
    "clash_royale": lambda cfg: getattr(cfg, "cr_wait_time", "frame") == "frame",
}


def check_crop_policy(game: str, cfg, max_len: Optional[int]) -> None:
    """Refuse a random-cropped dataset (``max_len`` not None) for a network
    whose decoder reads absolute frame time (:data:`CROP_UNSAFE`)."""
    unsafe = CROP_UNSAFE.get(game)
    if max_len is not None and game not in PREFIX_CROP and unsafe is not None and unsafe(cfg):
        raise SystemExit(f"{game}: this network reads absolute frame time (e.g. a legacy "
                         f"cr_wait_time='frame' clash_royale wait head) and cannot train "
                         f"or evaluate on random {max_len}-frame windows — drop the crop "
                         f"or use a crop-safe config (cr_wait_time='match')")

# Games whose ``max_len`` keeps the FIRST frames instead of a random window:
# the generals pass head reads the absolute frame index (decoded at t0=0),
# turn_globals derive the turn from it, and the fog-of-war memory needs the
# game from its start (data_processing/games/generals.py).
PREFIX_CROP = frozenset({"generals"})


def open_dataset(game: str, data_dir: str, max_len: Optional[int] = None,
                 whole_records: bool = False):
    """``game``'s dataset under ``data_dir`` with its crop policy (edge-balanced
    random windows — every frame of a long record drawn about as often, see
    GameShardDataset — or the first frames for PREFIX_CROP games); generals also reads
    gio record directories (data_processing.games.dataset.open_game_dataset).
    ``whole_records``: every record from its first frame (``max_len`` = a cap, kept as
    a prefix), for WholeRecordBatchSampler batches.  UNCROPPABLE games are the
    caller's business (max_len=None)."""
    prefix = whole_records or game in PREFIX_CROP
    return open_game_dataset(data_dir, game, max_len=max_len,
                             crop="prefix" if prefix else "balanced")


def _generals_obs(obs: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    """The recorded frames as the network's config expects them (the same
    transforms training/train_rl.py applies to its IL batches; records are
    prefix-truncated, so frame t is turn t): memory_vision nets see the
    sighted-mountain memory, derived from the frames so far
    (model.games.generals.apply_memory_vision); turn_globals nets expect 3
    more globals than the records carry, derived from the frame index."""
    if cfg is None:
        return obs
    if getattr(cfg, "memory_vision", False):
        obs = {**obs, "grid_seq": generals_adapter.apply_memory_vision(obs["grid_seq"])}
    if not getattr(cfg, "turn_globals", False):
        return obs
    g = obs["globals_seq"]
    if g.shape[-1] + generals_adapter.TURN_GLOBAL_FEATURES != cfg.global_features:
        return obs
    return {**obs, "globals_seq": generals_adapter.append_turn_globals(g)}


# Per-game observation transforms that depend on the model config, applied
# in prepare_batch (training and evaluation alike).
OBS_HOOKS = {
    "generals": _generals_obs,
}


def meta_arch_value(game: str, meta_key: str, value, cfg_get):
    """The config value implied by shard meta ``meta_key`` = ``value``
    (``cfg_get(field, default)`` reads the run's config / overrides)."""
    if (game == "generals" and meta_key == "global_features"
            and cfg_get("turn_globals", False)):
        return value + generals_adapter.TURN_GLOBAL_FEATURES
    return value


# Config fields an old converter generation implies without a meta key of
# their own: when the shard meta lacks META key K (the converter predates it),
# a fresh run also defaults these fields, so pre-freeze shards keep training
# the architecture they were made for.  pokemon: converter v1 shards (no
# "field_features") carry only the revealed-move slot target tgt.move_idx —
# the identity move head (the config default) trains on v2's tgt.move_id.
META_LEGACY_IMPLIES = {"pokemon": {"field_features": {"pokemon_move_head": "slot"}}}


def apply_meta_arch_defaults(game: str, kw: Dict, shard_meta: Dict) -> Dict:
    """Fresh runs: default ``game``'s data-dependent config fields
    (META_ARCH_KEYS) from the shard meta, plus META_LEGACY_IMPLIES when the
    meta predates a key; explicit values in ``kw`` win (check_meta_arch then
    rejects a contradiction)."""
    for meta_key, cfg_keys in META_ARCH_KEYS.get(game, {}).items():
        if meta_key in shard_meta:
            v = meta_arch_value(game, meta_key, shard_meta[meta_key], kw.get)
            for ck in cfg_keys:
                kw.setdefault(ck, v)
    for meta_key, implied in META_LEGACY_IMPLIES.get(game, {}).items():
        if meta_key not in shard_meta:
            for ck, v in implied.items():
                kw.setdefault(ck, v)
    return kw

# Converter generations with semantic changes: shards older than this
# train, but against re-converted data they are a different game (chess
# v2 = rank-mirror a7f5242; v1 frames are 180°-rotated for black).
MIN_CONVERTER_VERSION = {"chess": 2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, choices=sorted(LABEL_KEYS))
    p.add_argument("--data_dir", required=True)
    p.add_argument("--val_dir", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=1)
    # Optimisation defaults = reference TRM recipe (model/optim.py).
    p.add_argument("--lr", type=float, default=TRM_LR)
    p.add_argument("--weight_decay", type=float, default=TRM_WEIGHT_DECAY)
    p.add_argument("--warmup_steps", type=int, default=TRM_WARMUP_STEPS)
    p.add_argument("--beta1", type=float, default=TRM_BETAS[0])
    p.add_argument("--beta2", type=float, default=TRM_BETAS[1])
    p.add_argument("--lr_min_ratio", type=float, default=TRM_LR_MIN_RATIO,
                   help="cosine floor after warmup; 1.0 = constant LR (reference)")
    p.add_argument("--optimizer", choices=["adam_atan2", "adamw"], default="adam_atan2")
    p.add_argument("--moe_balance_rate", type=float, default=1e-3,
                   help="γ of the MoE aux-loss-free balance-bias update "
                        "(DeepSeek-V3), applied once per optimizer step")
    p.add_argument("--ema_rate", type=float, default=TRM_EMA_RATE,
                   help="EMA of the weights (evaluation + saved model_state); 0 = off")
    p.add_argument("--max_len", type=int, default=None,
                   help="crop sequences longer than this: a random window "
                        "(generals: its first max_len frames, see PREFIX_CROP)")
    p.add_argument("--halt_max_steps", type=int, default=0,
                   help="N_sup: max deep-supervision steps per frame "
                        "(0 = model config, reference 16); inference runs all")
    p.add_argument("--q_halt_weight", type=float, default=Q_HALT_WEIGHT)
    p.add_argument("--value_weight", type=float, default=0.2)
    p.add_argument("--logit_z_loss", type=float, default=1e-4,
                   help="z-loss weight on every categorical action head's log-partition "
                        "(PaLM; prevents output-logit divergence, seen on go at lr 3e-4); "
                        "0 = off")
    p.add_argument("--soft_target_weight", type=float, default=1.0,
                   help="alpha of the engine soft targets (tgt <head>_probs, when the "
                        "shards have them): 1 = train the teacher distribution instead "
                        "of the one-hot label on teacher frames, 0 = hard labels only")
    p.add_argument("--engine_value_weight", type=float, default=1.0,
                   help="beta of the engine value targets (tgt engine_value, when the "
                        "shards have them): value target = beta*engine + (1-beta)*outcome")
    p.add_argument("--il_legal_mask", action="store_true",
                   help="feed the shards' tgt legal_<key> masks to the decoder as "
                        "legal[<key>] (legal-masked IL loss / metrics where the decoder "
                        "supports them)")
    p.add_argument("--require_frozen_schema", action="store_true",
                   help="refuse shards older than the frozen pre-training schema "
                        "(SCHEMA_FREEZE_CONVERTER_VERSION) instead of warning")
    p.add_argument("--tf_start", type=float, default=None,
                   help="teacher-forcing prob at epoch 0 (conditioning labels); "
                        "default: the game's DEFAULT_TF (generals 0 = train_il.py, "
                        "the others 1.0)")
    p.add_argument("--tf_end", type=float, default=None,
                   help="teacher-forcing prob at the last epoch (linear in between); "
                        "default: the game's DEFAULT_TF")
    p.add_argument("--clip_grad", type=float, default=0.0,
                   help="max grad norm; 0 = no clipping (reference TRM)")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--save_dir", default="checkpoints/il_game")
    p.add_argument("--resume", default=None)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", nargs="*", default=[],
                   metavar="KEY=VALUE",
                   help="config overrides, e.g. d_model=256 board_h=9")
    return p.parse_args()


def parse_overrides(pairs) -> Dict:
    from model.config import ModelConfig
    str_fields = {k for k, f in ModelConfig.__dataclass_fields__.items() if f.type in ("str", str)}
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"--override expects KEY=VALUE, got {pair!r}")
        if key in str_fields:
            # a string field stays a string (core_gated_memory=2,5 is a spec, not a tuple)
            out[key] = raw
            continue
        if raw.lower() in ("true", "false"):
            # any spelling: the string 'false' would be truthy (qk_norm=false
            # silently building QK-norm)
            out[key] = raw.lower() == "true"
            continue
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw  # plain string (e.g. temporal_mixer=recurrent)
    return out


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resume_arch_overrides(resume_ckpt: Dict, cli_overrides: Dict) -> Dict:
    """Architecture kwargs for build_game_network on resume.

    The checkpoint's full saved config wins — rebuilding from saved
    *overrides* + current registry defaults would silently reshape the
    net whenever a registry default drifts between save and resume.
    Explicit CLI overrides still apply on top (loudly): they are the
    escape hatch when the saved config itself must be corrected.
    """
    if resume_ckpt.get("config") is not None:
        # Fields newer than the checkpoint take their legacy value (e.g. the
        # TRM block layout, or the observation a train_rl lineage played
        # with), so the rebuilt net matches the saved weights and inputs.
        arch = checkpoint_config_dict(resume_ckpt)
    else:
        arch = resume_ckpt.get("config_overrides", {})
        print(f"[warn] checkpoint predates full-config saving; rebuilding "
              f"from saved overrides + current defaults: {arch}")
        arch = upgrade_config_dict(arch)
    if cli_overrides:
        changed = {k: v for k, v in cli_overrides.items() if arch.get(k) != v}
        if changed:
            print(f"[warn] --override replaces checkpoint arch values: {changed}")
        arch.update(cli_overrides)
    return arch


def check_meta_arch(game: str, cfg, shard_meta: Dict) -> None:
    """Hard-fail when shard-level converter meta contradicts the model
    config (e.g. 20×20 gomoku shards vs a 15×15 net).  A key the shard
    meta lacks is held to its META_LEGACY value (the pre-key converter
    output) when one is defined, else skipped; a config field the config
    object does not have (a config class older than the field) is skipped.
    The error carries the key's META_MISMATCH_HINTS entry, if any."""
    legacy = META_LEGACY.get(game, {})
    hints = META_MISMATCH_HINTS.get(game, {})
    for meta_key, cfg_keys in META_ARCH_KEYS.get(game, {}).items():
        if meta_key in shard_meta:
            have, note = shard_meta[meta_key], ""
        elif meta_key in legacy:
            have = legacy[meta_key]
            note = (f" (implied: the shard meta has no {meta_key!r}, i.e. the "
                    f"converter predates it)")
        else:
            continue
        want = meta_arch_value(game, meta_key, have,
                               lambda f, d=None: getattr(cfg, f, d))
        for ck in cfg_keys:
            if not hasattr(cfg, ck):
                continue
            if getattr(cfg, ck) != want:
                hint = f". {hints[meta_key]}" if meta_key in hints else ""
                raise ValueError(
                    f"shards were converted with {meta_key}={have}{note} but the "
                    f"model has {ck}={getattr(cfg, ck)}; re-convert the data or "
                    f"use a matching checkpoint/--override{hint}"
                )


_UNEXPECTED_KW = re.compile(r"unexpected keyword argument '([^']+)'")


def build_net_tolerant(game: str, arch: Dict) -> Tuple[torch.nn.Module, List[str]]:
    """build_game_network, dropping saved-config keys the current config
    class no longer accepts (checkpoints carry fields from the code that
    wrote them).  Returns (net, dropped_keys)."""
    arch = dict(arch)
    dropped: List[str] = []
    while True:
        try:
            return build_game_network(game, **arch), dropped
        except TypeError as e:
            m = _UNEXPECTED_KW.search(str(e))
            if m is None or m.group(1) not in arch:
                raise
            dropped.append(m.group(1))
            arch.pop(m.group(1))


def build_labels(
    game: str, tgt: Dict[str, torch.Tensor], tf_prob: float,
    frame_mask: torch.Tensor,
) -> Optional[Dict[str, torch.Tensor]]:
    """Teacher-forcing labels: keep the ground-truth conditioning with
    probability ``tf_prob`` per frame, else −1 (decoder self-samples).
    STATE_LABEL_KEYS entries are copied unconditionally — they are game
    state, not decisions, and must never be dropped to −1.

    ``tf_prob <= 0`` without state keys returns None (no labels at all, no
    RNG draw): what training/train_il.py passes at tf 0, and what every
    decoder treats like all-−1 labels."""
    state_keys = set(STATE_LABEL_KEYS.get(game, ()))
    if tf_prob <= 0 and not state_keys:
        return None
    derive = LABELS_FROM_TGT.get(game)
    src = derive(tgt) if derive is not None else tgt
    keys = [k for k in LABEL_KEYS[game] if k in src]
    if not keys:
        return None
    labels = {}
    if tf_prob >= 1.0:
        for k in keys:
            labels[k] = src[k]
        return labels
    keep = (torch.rand(frame_mask.shape, device=frame_mask.device) < tf_prob)
    for k in keys:
        if k in state_keys:
            labels[k] = src[k]
        else:
            labels[k] = torch.where(keep, src[k], torch.full_like(src[k], -1))
    return labels


def legal_from_tgt(tgt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """The converters' IL legality masks: ``tgt["legal_<key>"]`` (bool, or
    packed-bits float {0, 1}) → ``{"<key>": bool mask}`` (the decoder's
    ``legal`` keys; see ``ActionDecoder.il_head_terms``)."""
    return {k[len("legal_"):]: v.bool() for k, v in tgt.items() if k.startswith("legal_")}


def batch_legal(game: str, obs: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                il_legal_mask: bool = False) -> Optional[Dict[str, torch.Tensor]]:
    """The decoder ``legal`` dict of an IL batch: LEGAL_FROM_OBS (always),
    plus the shards' ``legal_*`` targets under ``il_legal_mask`` (they win
    on a shared key); None when empty."""
    legal_fn = LEGAL_FROM_OBS.get(game)
    legal = dict((legal_fn(obs) if legal_fn is not None else None) or {})
    if il_legal_mask:
        legal.update(legal_from_tgt(tgt))
    return legal or None


def prepare_batch(game: str, batch: Dict, device: torch.device,
                  tf_prob: float, cfg=None, il_legal_mask: bool = False) -> Dict:
    """Move a batch to ``device`` and draw its teacher-forcing labels once
    (kept for every supervision step of the batch).  ``cfg`` (the game
    network's config) feeds the config-dependent OBS_HOOKS;
    ``il_legal_mask`` adds the shards' ``tgt["legal_*"]`` masks to the
    decoder's ``legal`` (:func:`batch_legal`)."""
    obs = {k: v.to(device) for k, v in batch["obs"].items()}
    hook = OBS_HOOKS.get(game)
    if hook is not None:
        obs = hook(obs, cfg)
    tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
    frame_mask = batch["meta"]["frame_mask"].to(device)
    labels = build_labels(game, tgt, tf_prob, frame_mask)
    legal = batch_legal(game, obs, tgt, il_legal_mask)
    return dict(obs=obs, tgt=tgt, frame_mask=frame_mask, labels=labels, legal=legal)


def prepare_rows(game: str, batch: Dict, device: torch.device,
                 tf_prob: float, cfg=None, il_legal_mask: bool = False) -> List[Dict]:
    """Batch → per-row data for training.deep_supervision.BatchCarry; the
    teacher-forcing draw belongs to the example (kept while it ponders).
    ``legal`` is not carried: :func:`mb_from_merged` rebuilds it from the
    merged obs / tgt (the ``legal_*`` targets travel in ``tgt``)."""
    mb = prepare_batch(game, batch, device, tf_prob, cfg=cfg, il_legal_mask=il_legal_mask)
    aux = {f"label:{k}": v for k, v in (mb["labels"] or {}).items()}
    return split_rows({"obs": mb["obs"], "tgt": mb["tgt"], "aux": aux,
                       "meta": {"lengths": batch["meta"]["lengths"]}})


def mb_from_merged(game: str, merged: Dict, il_legal_mask: bool = False) -> Dict:
    """BatchCarry's merged batch → the ``mb`` dict batch_loss consumes."""
    aux = merged.get("aux", {})
    labels = {k[len("label:"):]: v for k, v in aux.items() if k.startswith("label:")}
    obs = merged["obs"]
    return dict(obs=obs, tgt=merged["tgt"], frame_mask=merged["meta"]["frame_mask"],
                labels=labels or None,
                legal=batch_legal(game, obs, merged["tgt"], il_legal_mask))


def describe_value_head(cfg) -> str:
    """Short summary of the value head a config builds (for logs)."""
    if getattr(cfg, "hl_gauss_value", False):
        return (f"hl_gauss(bins={cfg.hl_gauss_bins},vmax={cfg.hl_gauss_vmax},"
                f"sigma={cfg.hl_gauss_sigma})")
    return "scalar_mse"


def check_value_support(target: torch.Tensor, mask: torch.Tensor,
                        vmax: float) -> None:
    """Raise unless every supervised (``mask``) value target is finite and
    inside the HL-Gauss support [-vmax, vmax].

    The Gaussian-histogram target construction clamps to the support, so an
    out-of-range target would otherwise train silently toward the edge.
    Every converter emits final outcomes in [-1, 1] (see the ``tgt.value``
    contract in data_processing/games/dataset.py), which the default support
    (vmax 1.5) covers with margin; padded frames (target 0) are masked out."""
    bad = mask.bool() & ~(target.abs() <= vmax)          # NaN compares False
    if bool(bad.any()):
        vals = target[bad]
        raise ValueError(
            f"{int(bad.sum())} value target(s) outside the HL-Gauss support "
            f"[-{vmax}, {vmax}] (e.g. {vals[:4].tolist()}, range "
            f"[{float(vals.min())}, {float(vals.max())}]): IL value targets are "
            f"final outcomes in [-1, 1] from the frame's player's perspective. "
            f"Fix the converter, or raise hl_gauss_vmax if the game's value "
            f"scale is genuinely wider")


def value_losses(net, pred: Dict[str, torch.Tensor], target: torch.Tensor,
                 mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(value_loss, value_mse), per-frame means over the ``mask`` frames.

    ``value_loss`` trains the head: the HL-Gauss cross-entropy against the
    Gaussian-smeared target (Farebrother et al. 2024, "Stop Regressing")
    when the network has an HLGaussValueHead (``pred["value_logits"]``),
    plain MSE for the scalar head (unchanged).  ``value_mse`` is the MSE of
    the scalar prediction (the HL-Gauss head's expectation) and so is
    comparable across head types; for the scalar head it IS value_loss.
    """
    mask_f = mask.float()
    n = mask_f.sum().clamp(min=1)
    target = target.float()
    logits = pred.get("value_logits")
    # per-frame terms are nan_to_num'ed BEFORE masking, as ImitationLoss
    # does: a non-finite prediction on a padded / halted frame (NaN·0 = NaN)
    # must not turn the whole micro-batch's loss into NaN
    if logits is None:
        mse = (F.mse_loss(pred["value"].float(), target,
                          reduction="none").nan_to_num(0.0) * mask_f).sum() / n
        return mse, mse
    # support = the head's own (vmax fixed its bin centres); sigma = config,
    # the same source train_rl's RLLoss reads
    vmax = float(net.value_head.vmax)
    check_value_support(target, mask, vmax)
    ce = hl_gauss_cross_entropy(logits, target, vmax, float(net.cfg.hl_gauss_sigma))
    v_loss = (ce.nan_to_num(0.0) * mask_f).sum() / n
    with torch.no_grad():
        mse = (F.mse_loss(pred["value"].float(), target,
                          reduction="none").nan_to_num(0.0) * mask_f).sum() / n
    return v_loss, mse


def logit_z_loss(pred: Dict[str, torch.Tensor], mask: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """z-loss (PaLM; Wortsman et al. 2023 'output logit divergence'):
    Σ_heads mean_frames (log Z)² over every categorical action head
    (``*_logits`` tensors of rank ≥ 3, softmax over the last dim), on the
    ``mask`` [B, T] frames.  Rows that are entirely −inf (fully masked) are
    excluded; extra per-frame dims (e.g. per-unit heads) are averaged.
    Returns (z_loss, max |log Z| for monitoring)."""
    total = mask.new_zeros((), dtype=torch.float32)
    lse_max = mask.new_zeros((), dtype=torch.float32)
    m = mask.float()
    for k, v in pred.items():
        if not (k.endswith("_logits") and torch.is_tensor(v) and v.dim() >= 3
                and v.shape[:2] == mask.shape and v.shape[-1] > 1):
            continue
        lse = torch.logsumexp(v.float(), dim=-1)                  # [B, T, ...]
        ok = torch.isfinite(lse)
        lse = torch.where(ok, lse, torch.zeros_like(lse))
        w = ok.float() * m.view(*m.shape, *([1] * (lse.dim() - 2)))
        total = total + (lse.square() * w).sum() / w.sum().clamp(min=1)
        with torch.no_grad():
            lse_max = torch.maximum(lse_max, (lse.abs() * w).max())
    return total, lse_max


def soft_target_heads(tgt: Dict[str, torch.Tensor]) -> List[str]:
    """Sorted head names with a soft target in ``tgt`` (``<h>_probs`` dense
    or ``<h>_probs_idx`` sparse; ``legal_*`` keys never count)."""
    heads = set()
    for k in tgt:
        if k.startswith("legal_"):
            continue
        for suffix in ("_probs", "_probs_idx"):
            if k.endswith(suffix) and len(k) > len(suffix):
                heads.add(k[: -len(suffix)])
    return sorted(heads)


def _teacher_probs(h: str, term: Dict, tgt: Dict[str, torch.Tensor],
                   kind: str) -> torch.Tensor:
    """Head ``h``'s teacher target as float32 of the term logits' shape
    (categorical [..., N] / bernoulli [...]); ValueError on a malformed one."""
    logits = term["logits"]
    dense, sparse = f"{h}_probs" in tgt, f"{h}_probs_idx" in tgt
    if dense and sparse:
        raise ValueError(f"soft target {h!r}: both {h}_probs and {h}_probs_idx given")
    if sparse:
        if kind != "categorical":
            raise ValueError(f"soft target {h!r}: sparse {h}_probs_idx needs a categorical "
                             f"head, {h!r} is {kind}")
        if f"{h}_probs_val" not in tgt:
            raise ValueError(f"soft target {h!r}: {h}_probs_idx without {h}_probs_val")
        p = densify_soft_target(tgt[f"{h}_probs_idx"], tgt[f"{h}_probs_val"],
                                logits.shape[-1])
    else:
        p = tgt[f"{h}_probs"].float()
    if p.shape != logits.shape:
        raise ValueError(f"soft target {h!r} has shape {tuple(p.shape)}, the {kind} head's "
                         f"logits {tuple(logits.shape)}")
    bad = ~(p >= 0)                                     # NaN compares False
    if kind == "bernoulli":
        bad = bad | ~(p <= 1)
    if bool((bad | ~torch.isfinite(p)).any()):          # one host sync per head
        raise ValueError(f"soft target {h!r} has non-finite or negative probabilities"
                         + (" or probabilities > 1" if kind == "bernoulli" else ""))
    return p


def soft_target_losses(net, pred: Dict[str, torch.Tensor], tgt: Dict[str, torch.Tensor],
                       mask: torch.Tensor, alpha: float,
                       stats: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    """The soft-target (teacher-distribution) correction to il_loss, or None
    when ``tgt`` has no soft target (nothing computed).

    Per head h of ``net.action_decoder.il_head_terms`` with a soft target
    (naming in ``ActionDecoder.il_head_terms``), on the teacher elements
    (presence × il_loss's own weight w):

        + α · scale · Σ (soft_ce − ce) · w / norm   (ce given: the soft term
                                                     replaces the hard one)
        + α · scale · Σ soft_ce · w / norm          (ce None: additive)

    — α = 1 trains the teacher distribution instead of the one-hot label
    where a teacher exists, α ∈ (0, 1) blends the two, 0 only reports.
    Categorical teacher rows are renormalised; with a term ``mask`` the
    teacher's illegal mass is reported (``<h>_teacher_illegal_mass``) and
    dropped first (a row with no legal mass left keeps its hard label).
    Stats (detached, into ``stats``): ``<h>_teacher_kl`` = mean KL(p ‖ q)
    over the teacher elements, ``<h>_teacher_frac`` = their share of the
    head's supervised weight."""
    heads = soft_target_heads(tgt)
    if not heads:
        return None
    dec = net.action_decoder
    terms = dec.il_head_terms(pred, tgt, mask)
    total = None
    for h in heads:
        if h not in terms:
            raise ValueError(
                f"tgt has a soft target for head {h!r}, but {type(dec).__name__}."
                f"il_head_terms exposes {sorted(terms)}")
        term = terms[h]
        kind = term.get("kind", "categorical")
        if kind not in ("categorical", "bernoulli"):
            raise ValueError(f"head {h!r}: unknown term kind {kind!r}")
        logits = term["logits"]
        weight = term["weight"].detach().float()
        p = _teacher_probs(h, term, tgt, kind)
        if f"{h}_probs_mask" in tgt:
            has = tgt[f"{h}_probs_mask"].bool()
        elif kind == "categorical":
            has = p.sum(-1) > 0
        else:
            has = torch.ones_like(p, dtype=torch.bool)
        if has.shape != weight.shape:
            raise ValueError(f"soft target {h!r}: presence {tuple(has.shape)} vs head "
                             f"weight {tuple(weight.shape)}")
        tmask = term.get("mask")
        if kind == "categorical":
            with torch.no_grad():
                if tmask is not None:
                    tmask = tmask.bool()
                    row = p.sum(-1)
                    rows = has & (weight > 0) & (row > 0)
                    frac = (p * ~tmask).sum(-1) / row.clamp_min(1e-12)
                    stats[f"{h}_teacher_illegal_mass"] = (
                        (frac * rows).sum() / rows.sum().clamp(min=1))
                    p = p * tmask
                row = p.sum(-1, keepdim=True)
                p = torch.where(row > 0, p / row.clamp_min(1e-12), p)
                has = has & (row.squeeze(-1) > 0)
            soft_ce, ent = soft_cross_entropy(logits.float(), p, tmask)
        else:
            soft_ce = F.binary_cross_entropy_with_logits(logits.float(), p, reduction="none")
            with torch.no_grad():
                ent = -(torch.special.xlogy(p, p) + torch.special.xlogy(1 - p, 1 - p))
        soft_ce = soft_ce.nan_to_num(0.0)
        w = weight * has.float()
        if alpha > 0:
            scale = float(term.get("scale", 1.0))
            ce = term.get("ce")
            diff = soft_ce if ce is None else soft_ce - ce.float()
            add = alpha * scale * (diff * w).sum() / term["norm"]
            total = add if total is None else total + add
        with torch.no_grad():
            n_w = w.sum()
            stats[f"{h}_teacher_kl"] = ((soft_ce - ent) * w).sum() / n_w.clamp(min=1)
            stats[f"{h}_teacher_frac"] = n_w / weight.sum().clamp(min=1)
    return total


def value_target(tgt: Dict[str, torch.Tensor], mask: torch.Tensor, beta: float
                 ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
                            Optional[torch.Tensor]]:
    """(target, supervised frames, engine value, engine mask) of the value
    head when ``tgt`` has ``engine_value`` and/or ``value_mask``: frames
    with both a valid outcome and an engine value train on
    β · engine + (1 − β) · outcome, frames with one of them on that one."""
    ones = torch.ones_like(mask, dtype=torch.bool)
    if "value" in tgt:
        val = tgt["value"].float()
        val_m = tgt["value_mask"].bool() if "value_mask" in tgt else ones
    else:
        val, val_m = torch.zeros_like(mask, dtype=torch.float32), ~ones
    eng = eng_m = None
    if "engine_value" in tgt:
        eng = tgt["engine_value"].float()
        eng_m = tgt["engine_value_mask"].bool() if "engine_value_mask" in tgt else ones
        target = torch.where(eng_m & val_m, beta * eng + (1.0 - beta) * val,
                             torch.where(eng_m, eng, val))
        sup = mask & (eng_m | val_m)
    else:
        target, sup = val, mask & val_m
    # unsupervised frames may hold anything (NaN for a missing engine eval):
    # keep them out of the loss arithmetic altogether
    target = torch.where(sup, target, torch.zeros_like(target))
    return target, sup, eng, eng_m


def batch_loss(
    net, mb: Dict, device: torch.device, value_weight: float,
    q_halt_weight: float, bf16: bool,
    carry=None, active: Optional[torch.Tensor] = None,
    act_steps: Optional[int] = None, reset: Optional[torch.Tensor] = None,
    z_loss_weight: float = 0.0,
    soft_target_weight: float = 1.0, engine_value_weight: float = 1.0,
    init_cache=None, t0: int = 0, return_cache: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """One TRM supervision step: decoder IL loss + value loss (value_losses:
    MSE, or HL-Gauss CE for an HL-Gauss head) + q_halt BCE (target = whole
    action exactly right, reference ACTLossHead), all on the ``active``
    frames.  Stats report both ``value_loss`` (the value term's training
    loss) and ``value_mse`` (comparable across head types).
    ``init_cache`` / ``t0`` / ``return_cache``: a chunk of a longer batch
    (GameNetwork.forward; :func:`chunked_batch_loss`).

    Engine supervision, used only when the shards carry it (absent keys =
    the plain objective above, bit for bit): soft policy targets
    ``tgt["<head>_probs"]`` (:func:`soft_target_losses`, weight α =
    ``soft_target_weight``) and engine values ``tgt["engine_value"]``
    blended into the value target with β = ``engine_value_weight``
    (:func:`value_target`; ``tgt["value_mask"]`` marks valid outcomes).
    Returns (loss, stats, pred)."""
    mask = mb["frame_mask"] if active is None else active
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    chunk_kw = ({} if init_cache is None and not t0 and not return_cache
                else dict(init_cache=init_cache, t0=t0, return_cache=return_cache))
    with autocast:
        pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"],
                   act_steps=act_steps, frame_mask=mb["frame_mask"],
                   carry=carry, active=active, reset=reset, return_carry=True, **chunk_kw)
        losses = net.action_decoder.il_loss(pred, mb["tgt"], mask)
        loss = losses["loss"]
        tgt = mb["tgt"]
        soft = soft_target_losses(net, pred, tgt, mask, soft_target_weight, losses)
        if soft is not None:
            loss = loss + soft
        z_loss, lse_max = logit_z_loss(pred, mask)
        losses["logit_lse_max"] = lse_max
        if z_loss_weight > 0:
            loss = loss + z_loss_weight * z_loss
            losses["z_loss"] = z_loss.detach()
        if "engine_value" in tgt or "value_mask" in tgt:
            v_tgt, v_mask, eng, eng_m = value_target(tgt, mask, engine_value_weight)
            v_loss, v_mse = value_losses(net, pred, v_tgt, v_mask)
            loss = loss + value_weight * v_loss
            losses["value_loss"] = v_loss.detach()
            losses["value_mse"] = v_mse.detach()
            if eng is not None:
                with torch.no_grad():
                    m = (mask & eng_m).float()
                    v = pred["value"].float()
                    losses["engine_value_mse"] = (
                        ((v - eng).square().nan_to_num(0.0) * m).sum() / m.sum().clamp(min=1))
                    nz = m * (eng != 0).float()
                    losses["engine_value_sign_acc"] = (
                        ((torch.sign(v) == torch.sign(eng)).float() * nz).sum()
                        / nz.sum().clamp(min=1))
                    losses["engine_value_frac"] = m.sum() / mask.float().sum().clamp(min=1)
        elif "value" in tgt:
            v_loss, v_mse = value_losses(net, pred, tgt["value"], mask)
            loss = loss + value_weight * v_loss
            losses["value_loss"] = v_loss.detach()
            losses["value_mse"] = v_mse.detach()
        correct = losses.pop("answer_correct", None)
        answer_mask = losses.pop("answer_mask", None)
        if correct is not None:
            q_mask = (answer_mask if answer_mask is not None else mask) & mask
            q_f = q_mask.float()
            n_q = q_f.sum().clamp(min=1)
            with torch.no_grad():
                # whole action exactly right (all heads), every core
                losses["exact_acc"] = (correct.float() * q_f).sum() / n_q
            # q_halt exists only for the legacy TRM core (its halting head);
            # other cores emit a constant 0, which would only add a constant
            if "q_halt" in pred and getattr(net, "core_type", "trm") == "trm":
                q_bce = F.binary_cross_entropy_with_logits(
                    pred["q_halt"].float(), correct.float(), reduction="none").nan_to_num(0.0)
                q_loss = (q_bce * q_f).sum() / n_q
                loss = loss + q_halt_weight * q_loss
                losses["q_halt_loss"] = q_loss.detach()
                with torch.no_grad():
                    losses["q_halt_acc"] = (((pred["q_halt"] > 0) == correct).float()
                                            * q_f).sum() / n_q

    stats = {k: float(v) for k, v in losses.items() if k != "loss"}
    stats["loss"] = float(loss.detach())
    return loss, stats, pred


def _time_slice(x, s: int, e: int, B: int, T: int):
    """Frames [s, e) of every [B, T, ...] tensor in a (nested) batch dict."""
    if torch.is_tensor(x):
        return x[:, s:e] if x.dim() >= 2 and x.shape[0] == B and x.shape[1] == T else x
    if isinstance(x, dict):
        return {k: _time_slice(v, s, e, B, T) for k, v in x.items()}
    return x


def chunked_batch_loss(net, mb: Dict, device: torch.device, chunk: int,
                       backward: Callable[[torch.Tensor], None], *loss_args,
                       active: Optional[torch.Tensor] = None,
                       reset: Optional[torch.Tensor] = None, carry=None, **loss_kw
                       ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    """:func:`batch_loss` and ``backward(loss)`` for a batch; one longer than ``chunk``
    frames (> 0) goes chunk by chunk — truncated backpropagation through time: chunk
    c covers frames [c·chunk, (c+1)·chunk) and starts from the memory state the
    previous chunk ended in (detached: the gradient stops at chunk boundaries), its
    loss scaled by its share of the batch's supervised frames, so the chunks together
    weigh like the whole batch while holding one chunk's activations.  The forward is
    exactly the whole batch's (GameNetwork.forward ``init_cache`` / ``t0``); frames
    are a prefix of each row (tail padding), so the chunks stop at the last frame.
    Returns (stats, pred) as batch_loss does — stats frame-weighted over the chunks,
    pred only ``q_halt`` [B, T] (the chunked path is for cores without deep
    supervision, whose q_halt is 0)."""
    fm = mb["frame_mask"]
    B, T = fm.shape
    if chunk <= 0 or T <= chunk:
        loss, stats, pred = batch_loss(net, mb, device, *loss_args, active=active,
                                       reset=reset, carry=carry, **loss_kw)
        backward(loss)
        return stats, pred
    if carry is not None or net.supervision_steps > 1:
        raise ValueError("chunked training carries the memory state, not TRM latents: "
                         "it needs a core without deep supervision")
    mask = fm if active is None else active
    n_all = float(mask.sum())
    cache, agg = None, {}
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        sub = _time_slice(mb, s, e, B, T)
        sub_active = None if active is None else active[:, s:e]
        n = float(mask[:, s:e].sum())
        if n == 0:
            break
        loss, stats, pred = batch_loss(
            net, sub, device, *loss_args, active=sub_active,
            reset=None if reset is None else reset[:, s:e], init_cache=cache, t0=s,
            return_cache=True, **loss_kw)
        backward(loss * (n / n_all))
        cache = [c.detach() for c in pred["st_cache"]]
        for k, v in stats.items():
            agg[k] = agg.get(k, 0.0) + v * n / n_all
    return agg, {"q_halt": torch.zeros(B, T, device=fm.device)}


@contextlib.contextmanager
def greedy_decoding(net):
    """Evaluation: a decoder that self-samples a conditioning index where the
    data has no label (generals' source on pass / dropped-move frames) takes
    the argmax instead (``ActionDecoder.greedy``), so validation metrics do
    not depend on the RNG state — the same on any rank split.  Training
    keeps sampling."""
    dec = net.action_decoder
    prev = dec.greedy
    dec.greedy = True
    try:
        yield
    finally:
        dec.greedy = prev


def evaluate(net, game, loader, device, value_weight, q_halt_weight, bf16,
             ema: Optional[ModelEMA] = None,
             diag: Optional[List[str]] = None, il_legal_mask: bool = False,
             soft_target_weight: float = 1.0,
             engine_value_weight: float = 1.0) -> Dict[str, float]:
    """Validation at full depth (all halt_max_steps supervision steps, as
    the reference evaluates), on the EMA weights when enabled, with
    deterministic conditioning (greedy_decoding).  With ``diag`` (a list)
    the first batch also runs under training.diagnostics.attention_probe
    and its line is appended.  The remaining keywords are the training
    batch_loss / prepare_batch settings."""
    model = ema.ema_copy(net) if ema is not None else net
    model.eval()
    agg: Dict[str, float] = {}
    n = 0
    with torch.no_grad(), greedy_decoding(model):
        for i, batch in enumerate(loader):
            mb = prepare_batch(game, batch, device, tf_prob=1.0, cfg=model.cfg,
                               il_legal_mask=il_legal_mask)
            probe = (attention_probe(model) if diag is not None and i == 0
                     else contextlib.nullcontext())
            with probe as attn:
                _, stats, pred = batch_loss(model, mb, device, value_weight,
                                            q_halt_weight, bf16,
                                            act_steps=model.default_depth,
                                            soft_target_weight=soft_target_weight,
                                            engine_value_weight=engine_value_weight)
            if attn is not None:
                diag.append(format_attention(attn, logit_lse_absmax(pred, mb["frame_mask"])))
            for k, v in stats.items():
                agg[k] = agg.get(k, 0.0) + v
            n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in agg.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    overrides = parse_overrides(args.override)

    if args.game == "clash_royale" and args.max_len is not None:
        raise SystemExit(
            "--max_len random-crops sequences, but the clash_royale wait "
            "head derives absolute time from t0=0 (model/games/"
            "clash_royale.py decode), so cropped frames get wrong time "
            "features; train clash_royale without --max_len"
        )

    resume_ckpt = None
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_ckpt.get("game") != args.game:
            raise ValueError(
                f"checkpoint is for game {resume_ckpt.get('game')!r}, not {args.game!r}"
            )

    dataset = open_dataset(args.game, args.data_dir, args.max_len)
    val_dataset = None
    if args.val_dir:
        val_dataset = open_dataset(args.game, args.val_dir, args.max_len)
        if (val_dataset.converter_version != dataset.converter_version
                or val_dataset.shard_meta != dataset.shard_meta):
            raise ValueError(
                f"val shards (converter_version="
                f"{val_dataset.converter_version}, meta={val_dataset.shard_meta}) "
                f"were converted differently from train shards "
                f"(converter_version={dataset.converter_version}, "
                f"meta={dataset.shard_meta}); re-convert both together"
            )
    if isinstance(dataset, GeneralsGioDataset):
        advice = gio_dataset_advice(len(dataset), args.num_workers)
        if advice:
            print(f"[warn] {advice}")
    min_cv = MIN_CONVERTER_VERSION.get(args.game)
    if min_cv is not None and (dataset.converter_version or 1) < min_cv:
        print(f"[WARN] {args.game} shards have converter_version "
              f"{dataset.converter_version} < {min_cv}: they predate a "
              f"semantic converter fix — re-convert with the current "
              f"converter, and never mix generations in one directory")
    frozen = schema_freeze_warning(args.game, dataset)
    if frozen is not None:
        if args.require_frozen_schema:
            raise SystemExit(f"--require_frozen_schema: {frozen}")
        print(f"[WARN] {frozen}")

    # A resume checkpoint defines the architecture (same invariant as
    # train_il/train_rl): rebuilding from CLI overrides would silently
    # change the model shape and crash — or worse, drift — on load.
    if resume_ckpt is not None:
        arch = resume_arch_overrides(resume_ckpt, overrides)
    else:
        arch = apply_meta_arch_defaults(args.game, dict(overrides), dataset.shard_meta)
    net, dropped = build_net_tolerant(args.game, arch)
    if dropped:
        print(f"[warn] checkpoint config fields unknown to current code, "
              f"ignored: {dropped}")
    net = net.to(device)
    cfg = net.cfg
    check_crop_policy(args.game, cfg, args.max_len)
    check_meta_arch(args.game, cfg, dataset.shard_meta)
    overrides = arch
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[init] game={args.game} device={device.type} params={n_params:,}")

    # Shard-locality shuffling: plain shuffle=True would reload a whole
    # shard from disk for nearly every sample (the dataset caches one
    # shard); this sampler shuffles shard order + within-shard order.
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        sampler=training_sampler(dataset),
                        num_workers=args.num_workers,
                        collate_fn=collate_game_sequences, drop_last=False)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size, collate_fn=collate_game_sequences,
        )
    print(f"[data] {len(dataset)} games, {len(loader)} batches/epoch")

    if args.halt_max_steps > 0:
        cfg.halt_max_steps = args.halt_max_steps
    print(f"[init] core={cfg.core_type} moe_experts={cfg.moe_experts} qk_norm={cfg.qk_norm} "
          f"encoder_norm={describe_encoder_norm(cfg)} "
          f"supervision_steps={net.supervision_steps} inference_depth={net.default_depth} "
          f"value_head={describe_value_head(cfg)}")
    if args.resume and not cfg.qk_norm:
        print("[note] qk_norm=False comes from the resume checkpoint's config "
              "(a pre-QK-norm lineage cannot switch it on)")

    groups = optim_param_groups(net, args.weight_decay)
    if args.optimizer == "adam_atan2":
        optimizer = AdamATan2(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                              weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2))
    # reference total_steps = epochs * N / global_batch: one optimizer step
    # per accumulation window of loader batches
    total_steps = max(1, args.epochs * len(loader) // args.grad_accum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, trm_lr_lambda(args.warmup_steps, total_steps, args.lr_min_ratio))

    ema = ModelEMA(net, mu=args.ema_rate) if args.ema_rate > 0 else None
    start_epoch = 0
    if resume_ckpt is not None:
        # model_state holds the EMA weights; train_model_state the raw ones
        load_model_state(net, resume_ckpt.get("train_model_state",
                                              resume_ckpt["model_state"]), ctx=args.resume)
        if ema is not None:
            if "train_model_state" in resume_ckpt:
                ema.load_state_dict({k: v.to(device)
                                     for k, v in resume_ckpt["model_state"].items()})
            else:
                ema = ModelEMA(net, mu=args.ema_rate)
        same_opt = resume_ckpt.get("optimizer_name", "adamw") == args.optimizer
        if "optimizer_state" in resume_ckpt and same_opt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state"])
        elif "optimizer_state" in resume_ckpt:
            print(f"[resume] optimizer state skipped (saved by "
                  f"{resume_ckpt.get('optimizer_name', 'adamw')}, now {args.optimizer})")
        if "scheduler_state" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state"])
        start_epoch = resume_ckpt.get("epoch", -1) + 1
        print(f"[resume] {args.resume} → epoch {start_epoch}")
        resume_ckpt = None  # release checkpoint tensors

    os.makedirs(args.save_dir, exist_ok=True)
    net.train()
    opt_step = 0

    def optimizer_step():
        nonlocal opt_step
        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), args.clip_grad if args.clip_grad > 0 else float("inf"))
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            print(f"  [NaN] dropped a supervision step (grad_norm={grad_norm.item():.4f})")
            return
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(net)
        # MoE aux-loss-free balancing over the whole step (all micro-batches)
        update_moe_balance(net, args.moe_balance_rate)
        opt_step += 1

    # Reference train_state.carry: one per accumulation slot, persisting
    # across batches and epochs (training/deep_supervision.py).
    store = (torch.device("cpu")
             if args.grad_accum > 1 and device.type == "cuda" else None)
    carries = [BatchCarry(net.supervision_steps, cfg.halt_exploration_prob, store=store)
               for _ in range(max(1, args.grad_accum))]

    tf_start = args.tf_start if args.tf_start is not None else default_tf(args.game)
    tf_end = args.tf_end if args.tf_end is not None else default_tf(args.game)
    print(f"[init] teacher forcing {tf_start:g} → {tf_end:g}")
    for epoch in range(start_epoch, args.epochs):
        frac = epoch / max(1, args.epochs - 1)
        tf_prob = tf_start + (tf_end - tf_start) * frac
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        slot = 0
        n_slots = max(1, args.grad_accum)
        for i, batch in enumerate(loader):
            # one loader batch = ONE supervision (ACT) step of the slot's
            # carried batch; halted rows are replaced by this batch's rows
            def forward_backward(merged, carry, active, reset, weight):
                loss, stats, pred = batch_loss(
                    net, mb_from_merged(args.game, merged, args.il_legal_mask), device,
                    args.value_weight, args.q_halt_weight, args.bf16,
                    carry=carry, active=active, reset=reset,
                    z_loss_weight=args.logit_z_loss,
                    soft_target_weight=args.soft_target_weight,
                    engine_value_weight=args.engine_value_weight)
                # weight = n_active / n_frames (reference fixed-batch
                # normalisation)
                (loss * (weight / n_slots)).backward()
                return pred, stats

            stats = carries[slot].step(
                prepare_rows(args.game, batch, device, tf_prob, cfg=cfg,
                             il_legal_mask=args.il_legal_mask), forward_backward)
            slot += 1
            if slot < n_slots and (i + 1) < len(loader):
                continue
            slot = 0
            optimizer_step()
            if stats is not None and (opt_step - 1) % args.log_every == 0:
                head = " ".join(f"{k}={v:.4f}" for k, v in sorted(stats.items()))
                caps = qk_caps(net)
                if caps:
                    head += f" qk_cap_max={max(caps.values()):.2f}"
                print(f"[e{epoch} b{i}/{len(loader)}] tf={tf_prob:.2f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} {head}", flush=True)

        if val_loader is not None:
            diag: List[str] = []
            val = evaluate(net, args.game, val_loader, device,
                           args.value_weight, args.q_halt_weight, args.bf16, ema=ema,
                           diag=diag, il_legal_mask=args.il_legal_mask,
                           soft_target_weight=args.soft_target_weight,
                           engine_value_weight=args.engine_value_weight)
            print(f"[e{epoch} val] " + " ".join(
                f"{k}={v:.4f}" for k, v in sorted(val.items())))
            for line in diag:
                print(f"[e{epoch} attn] {line}", flush=True)

        ckpt_path = os.path.join(args.save_dir, f"{args.game}_epoch{epoch}.pt")
        ckpt = {
            "game": args.game,
            "epoch": epoch,
            # deployment weights = EMA (the reference saves the EMA copy)
            "model_state": (ema.ema_model_state(net) if ema is not None
                            else net.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "optimizer_name": args.optimizer,
            "scheduler_state": scheduler.state_dict(),
            "config": cfg.__dict__,
            "config_overrides": overrides,
        }
        if ema is not None:
            ckpt["train_model_state"] = net.state_dict()
        torch.save(ckpt, ckpt_path)
        print(f"[e{epoch}] saved {ckpt_path} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
