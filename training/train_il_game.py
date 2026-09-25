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
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import os
import random
import re
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_processing.games.dataset import (
    collate_game_sequences,
    open_game_dataset,
    training_sampler,
)
from model.components.core import update_moe_balance
from model.config import upgrade_config_dict
from model.games import build_game_network
from model.games import generals as generals_adapter
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
    "clash_royale": ("card_slot",),
    "mahjong": ("action_type",),
    # active_idx is state (which own slot is on the field), delivered via
    # labels because the observer-view converter stores it in tgt.
    "pokemon": ("action_type", "active_idx"),
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
    "lux": lambda obs: {"unit": obs["unit_mask"].bool()},
    "starcraft": lambda obs: {"actor": obs["unit_mask"].bool()},
    # entity_mask (True = slot occupied) keeps tf_prob<1 actor
    # self-sampling off empty entity slots, same as starcraft above.
    "hearthstone": lambda obs: {"actor": obs["entity_mask"].bool()},
    # move_ids is state (revealed moves of the active mon) consumed by
    # the move head at decode time; guarded for pre-move_ids shards.
    "pokemon": lambda obs: (
        {"move_ids": obs["move_ids"]} if "move_ids" in obs else None
    ),
}

# Shard-level meta facts (written by the converters) that must match the
# model config: {game: {meta_key: (config_field, ...)}}.  Fresh runs
# default the config from the data; every run hard-fails on a mismatch.
META_ARCH_KEYS = {
    "gomoku": {"board_size": ("board_h", "board_w")},
    # crdata writes the arena geometry and the card vocabulary of the
    # catalog it converted with (n_cards grows when cards are added)
    "clash_royale": {k: (k,) for k in (
        "arena_h", "arena_w", "arena_channels", "hand_size", "n_cards",
        "card_feature_dim", "cr_global_features")},
    "pokemon": {"move_buckets": ("pokemon_move_buckets",),
                "species_buckets": ("pokemon_species_buckets",)},
    # gio records (GeneralsGioDataset.shard_meta / `gmdata generals shard`);
    # with turn_globals the net has TURN_GLOBAL_FEATURES more globals than
    # the data (meta_arch_value), appended at batch time (OBS_HOOKS)
    "generals": {k: (k,) for k in (
        "grid_h", "grid_w", "grid_channels", "global_features")},
}

# Games whose sequences must not be random-cropped: the clash_royale wait
# head derives absolute time from t0=0 (model/games/clash_royale.py decode).
UNCROPPABLE = frozenset({"clash_royale"})

# Games whose ``max_len`` keeps the FIRST frames instead of a random window:
# the generals pass head reads the absolute frame index (decoded at t0=0),
# turn_globals derive the turn from it, and the fog-of-war memory needs the
# game from its start (data_processing/games/generals.py).
PREFIX_CROP = frozenset({"generals"})


def open_dataset(game: str, data_dir: str, max_len: Optional[int] = None):
    """``game``'s dataset under ``data_dir`` with its crop policy (random
    windows, or the first frames for PREFIX_CROP games); generals also reads
    gio record directories (data_processing.games.dataset.open_game_dataset).
    UNCROPPABLE games are the caller's business (max_len=None)."""
    return open_game_dataset(data_dir, game, max_len=max_len,
                             crop="prefix" if game in PREFIX_CROP else "random")


def _generals_obs(obs: Dict[str, torch.Tensor], cfg) -> Dict[str, torch.Tensor]:
    """turn_globals nets expect 3 more globals than the records carry:
    derive them from the frame index (records are prefix-truncated, so
    frame t is turn t) — the same features training/train_rl.py appends to
    its IL batches."""
    if cfg is None or not getattr(cfg, "turn_globals", False):
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


def apply_meta_arch_defaults(game: str, kw: Dict, shard_meta: Dict) -> Dict:
    """Fresh runs: default ``game``'s data-dependent config fields
    (META_ARCH_KEYS) from the shard meta; explicit values in ``kw`` win
    (check_meta_arch then rejects a contradiction)."""
    for meta_key, cfg_keys in META_ARCH_KEYS.get(game, {}).items():
        if meta_key in shard_meta:
            v = meta_arch_value(game, meta_key, shard_meta[meta_key], kw.get)
            for ck in cfg_keys:
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
                   help="random-crop sequences longer than this")
    p.add_argument("--halt_max_steps", type=int, default=0,
                   help="N_sup: max deep-supervision steps per frame "
                        "(0 = model config, reference 16); inference runs all")
    p.add_argument("--q_halt_weight", type=float, default=Q_HALT_WEIGHT)
    p.add_argument("--value_weight", type=float, default=0.2)
    p.add_argument("--tf_start", type=float, default=1.0,
                   help="teacher-forcing prob at epoch 0 (conditioning labels)")
    p.add_argument("--tf_end", type=float, default=1.0)
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
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"--override expects KEY=VALUE, got {pair!r}")
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
    arch = resume_ckpt.get("config")
    if arch is None:
        arch = resume_ckpt.get("config_overrides", {})
        print(f"[warn] checkpoint predates full-config saving; rebuilding "
              f"from saved overrides + current defaults: {arch}")
    # Fields newer than the checkpoint take their legacy value (e.g. the
    # TRM block layout), so the rebuilt net matches the saved weights.
    arch = upgrade_config_dict(arch)
    if cli_overrides:
        changed = {k: v for k, v in cli_overrides.items() if arch.get(k) != v}
        if changed:
            print(f"[warn] --override replaces checkpoint arch values: {changed}")
        arch.update(cli_overrides)
    return arch


def check_meta_arch(game: str, cfg, shard_meta: Dict) -> None:
    """Hard-fail when shard-level converter meta contradicts the model
    config (e.g. 20×20 gomoku shards vs a 15×15 net)."""
    for meta_key, cfg_keys in META_ARCH_KEYS.get(game, {}).items():
        if meta_key not in shard_meta:
            continue
        want = meta_arch_value(game, meta_key, shard_meta[meta_key],
                               lambda f, d=None: getattr(cfg, f, d))
        for ck in cfg_keys:
            if getattr(cfg, ck) != want:
                raise ValueError(
                    f"shards were converted with {meta_key}="
                    f"{shard_meta[meta_key]} but the model has "
                    f"{ck}={getattr(cfg, ck)}; re-convert the data or "
                    f"use a matching checkpoint/--override"
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
    state, not decisions, and must never be dropped to −1."""
    derive = LABELS_FROM_TGT.get(game)
    src = derive(tgt) if derive is not None else tgt
    keys = [k for k in LABEL_KEYS[game] if k in src]
    if not keys:
        return None
    state_keys = set(STATE_LABEL_KEYS.get(game, ()))
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


def prepare_batch(game: str, batch: Dict, device: torch.device,
                  tf_prob: float, cfg=None) -> Dict:
    """Move a batch to ``device`` and draw its teacher-forcing labels once
    (kept for every supervision step of the batch).  ``cfg`` (the game
    network's config) feeds the config-dependent OBS_HOOKS."""
    obs = {k: v.to(device) for k, v in batch["obs"].items()}
    hook = OBS_HOOKS.get(game)
    if hook is not None:
        obs = hook(obs, cfg)
    tgt = {k: v.to(device) for k, v in batch["tgt"].items()}
    frame_mask = batch["meta"]["frame_mask"].to(device)
    labels = build_labels(game, tgt, tf_prob, frame_mask)
    legal_fn = LEGAL_FROM_OBS.get(game)
    legal = legal_fn(obs) if legal_fn is not None else None
    return dict(obs=obs, tgt=tgt, frame_mask=frame_mask, labels=labels, legal=legal)


def prepare_rows(game: str, batch: Dict, device: torch.device,
                 tf_prob: float, cfg=None) -> List[Dict]:
    """Batch → per-row data for training.deep_supervision.BatchCarry; the
    teacher-forcing draw belongs to the example (kept while it ponders)."""
    mb = prepare_batch(game, batch, device, tf_prob, cfg=cfg)
    aux = {f"label:{k}": v for k, v in (mb["labels"] or {}).items()}
    return split_rows({"obs": mb["obs"], "tgt": mb["tgt"], "aux": aux,
                       "meta": {"lengths": batch["meta"]["lengths"]}})


def mb_from_merged(game: str, merged: Dict) -> Dict:
    """BatchCarry's merged batch → the ``mb`` dict batch_loss consumes."""
    aux = merged.get("aux", {})
    labels = {k[len("label:"):]: v for k, v in aux.items() if k.startswith("label:")}
    obs = merged["obs"]
    legal_fn = LEGAL_FROM_OBS.get(game)
    return dict(obs=obs, tgt=merged["tgt"], frame_mask=merged["meta"]["frame_mask"],
                labels=labels or None,
                legal=legal_fn(obs) if legal_fn is not None else None)


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
    if logits is None:
        mse = (F.mse_loss(pred["value"].float(), target,
                          reduction="none") * mask_f).sum() / n
        return mse, mse
    # support = the head's own (vmax fixed its bin centres); sigma = config,
    # the same source train_rl's RLLoss reads
    vmax = float(net.value_head.vmax)
    check_value_support(target, mask, vmax)
    ce = hl_gauss_cross_entropy(logits, target, vmax, float(net.cfg.hl_gauss_sigma))
    v_loss = (ce * mask_f).sum() / n
    with torch.no_grad():
        mse = (F.mse_loss(pred["value"].float(), target,
                          reduction="none") * mask_f).sum() / n
    return v_loss, mse


def batch_loss(
    net, mb: Dict, device: torch.device, value_weight: float,
    q_halt_weight: float, bf16: bool,
    carry=None, active: Optional[torch.Tensor] = None,
    act_steps: Optional[int] = None, reset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """One TRM supervision step: decoder IL loss + value loss (value_losses:
    MSE, or HL-Gauss CE for an HL-Gauss head) + q_halt BCE (target = whole
    action exactly right, reference ACTLossHead), all on the ``active``
    frames.  Stats report both ``value_loss`` (the value term's training
    loss) and ``value_mse`` (comparable across head types).  Returns
    (loss, stats, pred)."""
    mask = mb["frame_mask"] if active is None else active
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16)
    with autocast:
        pred = net(mb["obs"], legal=mb["legal"], labels=mb["labels"],
                   act_steps=act_steps, frame_mask=mb["frame_mask"],
                   carry=carry, active=active, reset=reset, return_carry=True)
        losses = net.action_decoder.il_loss(pred, mb["tgt"], mask)
        loss = losses["loss"]
        tgt = mb["tgt"]
        if "value" in tgt:
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
             diag: Optional[List[str]] = None) -> Dict[str, float]:
    """Validation at full depth (all halt_max_steps supervision steps, as
    the reference evaluates), on the EMA weights when enabled, with
    deterministic conditioning (greedy_decoding).  With ``diag`` (a list)
    the first batch also runs under training.diagnostics.attention_probe
    and its line is appended."""
    model = ema.ema_copy(net) if ema is not None else net
    model.eval()
    agg: Dict[str, float] = {}
    n = 0
    with torch.no_grad(), greedy_decoding(model):
        for i, batch in enumerate(loader):
            mb = prepare_batch(game, batch, device, tf_prob=1.0, cfg=model.cfg)
            probe = (attention_probe(model) if diag is not None and i == 0
                     else contextlib.nullcontext())
            with probe as attn:
                _, stats, pred = batch_loss(model, mb, device, value_weight,
                                            q_halt_weight, bf16,
                                            act_steps=model.default_depth)
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
    min_cv = MIN_CONVERTER_VERSION.get(args.game)
    if min_cv is not None and (dataset.converter_version or 1) < min_cv:
        print(f"[WARN] {args.game} shards have converter_version "
              f"{dataset.converter_version} < {min_cv}: they predate a "
              f"semantic converter fix — re-convert with the current "
              f"converter, and never mix generations in one directory")

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

    for epoch in range(start_epoch, args.epochs):
        frac = epoch / max(1, args.epochs - 1)
        tf_prob = args.tf_start + (args.tf_end - args.tf_start) * frac
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        slot = 0
        n_slots = max(1, args.grad_accum)
        for i, batch in enumerate(loader):
            # one loader batch = ONE supervision (ACT) step of the slot's
            # carried batch; halted rows are replaced by this batch's rows
            def forward_backward(merged, carry, active, reset, weight):
                loss, stats, pred = batch_loss(
                    net, mb_from_merged(args.game, merged), device,
                    args.value_weight, args.q_halt_weight, args.bf16,
                    carry=carry, active=active, reset=reset)
                # weight = n_active / n_frames (reference fixed-batch
                # normalisation)
                (loss * (weight / n_slots)).backward()
                return pred, stats

            stats = carries[slot].step(
                prepare_rows(args.game, batch, device, tf_prob, cfg=cfg), forward_backward)
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
                           diag=diag)
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
