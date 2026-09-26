"""Model configuration for the Generals.io neural network."""

import ast
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass
class ModelConfig:
    # Grid dimensions
    grid_h: int = 25
    grid_w: int = 25

    # Input channels: 11 one-hot state types + 1 army channel
    grid_channels: int = 12
    n_grid_types: int = 11
    global_features: int = 4  # myArmy, myLand, opArmy, opLand
                              # (7 with turn_globals: + production-cycle
                              #  phase, half-turn parity, scaled turn)

    # Generals-adapter input features (both default OFF for byte-compatible
    # behavior with existing checkpoints; enabling them zero-pads the first
    # conv / global projection — function-preserving — via the
    # shape-adaptive loader in GeneralsNetwork.load_state_dict):
    #   derive_army_channels — GeneralsEncoder derives 3 LINEAR army
    #     channels (own/enemy/neutral-city, army/50 clipped) from the grid
    #     before the spatial encoder. The single log1p(army)/10 channel
    #     makes exact army comparisons nearly invisible in input space
    #     (200-vs-250 differs by ~0.02); combat arithmetic is linear.
    #   turn_globals — the env appends (turn%50)/50, turn%2, turn/500 to
    #     the globals (set global_features=7 together with this): the
    #     temporal attention has no positional encoding, so the policy
    #     otherwise cannot see the 50-tick production cycle at all.
    derive_army_channels: bool = False
    turn_globals: bool = False
    # Generals observation, sighted-mountain memory: True = a mountain the
    # player has seen stays encoded as a mountain (state 4) after vision
    # moves on (env.generals_env memory_vision; recorded games get it
    # causally from their frames, model.games.generals.apply_memory_vision);
    # False = pure instantaneous fog, the IL records' own encoding.  Which
    # observation the network was trained on — sticky in checkpoints like
    # turn_globals, and every env builder sets the env from it.  Configs
    # saved before this field existed: legacy_memory_vision below.
    memory_vision: bool = False

    # Core model dimensions
    d_model: int = 384
    n_heads: int = 12
    d_ff: int = 1024
    dropout: float = 0.1

    # Stage 1: Spatiotemporal encoding (separate weights per layer)
    n_st_layers: int = 3

    # Stage 2 core (model/components/core.py):
    #   "recurrent_depth" — prelude → r × recurrent blocks → coda, Huginn-style
    #                       (arXiv 2502.05171); the default
    #   "transformer"     — core_layers untied blocks, one pass
    #   "trm"             — Tiny Recursive Model (legacy; checkpoints saved
    #                       before core_type existed load as "trm")
    core_type: str = "recurrent_depth"
    core_expansion: float = 4.0          # SwiGLU width of dense core FFNs
    core_layers: int = 6                 # "transformer" core depth
    rd_prelude_layers: int = 2           # l_P
    rd_recurrent_layers: int = 2         # l_R (the looped block)
    rd_coda_layers: int = 2              # l_C
    rd_mean_iters: int = 8               # r̄ of the training log-normal Poisson
    rd_backprop_iters: int = 4           # k: truncated BPTT depth
    rd_max_iters: int = 32               # cap on the sampled r (0 = none)
    rd_eval_iters: int = 0               # fixed inference r (0 = rd_mean_iters)
    # Mixture of Experts in the core FFNs (0 = dense). DeepSeek-V3 routing:
    # sigmoid affinities, top-k, shared experts, aux-loss-free balance bias.
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_shared_experts: int = 1
    moe_expansion: float = 2.0           # per-expert SwiGLU width
    moe_bias_clip: float = 0.0           # bound on |balance bias| (0 = unbounded)
    # QK-normalisation: per-head RMSNorm (learnable scales over head_dim,
    # initial cap √head_dim·g_q·g_k ≥ 8, see core.qk_norm_init_scale) of
    # queries and keys before the dot product, in the Stage-2
    # transformer / recurrent_depth core blocks (not the TRM cores) and the
    # Stage-1 spatial / temporal attention.  Bounds every attention logit
    # by √head_dim·max|g_q|·max|g_k|, so logits cannot grow
    # with the activations or the q/k weights — the "attention logit growth"
    # divergence of Wortsman et al. 2023 (arXiv 2309.14322).  Per-head form
    # as in ViT-22B (Dehghani et al. 2023, arXiv 2302.05442) / Qwen3 (OLMo 2
    # normalises the full-width projection instead).
    # Checkpoints saved before this field existed load with it OFF.
    qk_norm: bool = True

    # Stage 2: TRM (Tiny Recursive Model, arXiv 2510.04871; defaults follow
    # the reference implementation SamsungSAILMontreal/TinyRecursiveModels,
    # config/arch/trm.yaml)
    trm_H_cycles: int = 3    # T: outer loop — refine answer z_H
    trm_L_cycles: int = 6    # n: inner loop — refine reasoning z_L
    trm_block_layers: int = 2  # layers inside the single shared network
    # Deep supervision: at most N_sup = halt_max_steps supervision steps per
    # example in training (each its own optimizer step, latents carried
    # detached); inference ALWAYS runs all halt_max_steps steps (the
    # reference halts early only during training).
    halt_max_steps: int = 16
    halt_exploration_prob: float = 0.1
    # Block layout. "trm" = reference block: post-norm RMSNorm (no weight),
    # bias-free attention + SwiGLU, no dropout, truncated LeCun init.
    # "legacy" = the pre-2026-09 LayerNorm/GELU/dropout block; checkpoints
    # saved before this field existed load as legacy (upgrade_config_dict).
    trm_arch: str = "trm"
    trm_expansion: float = 4.0
    trm_rms_eps: float = 1e-5

    # Temporal modelling ("spatiotemporal" is the only supported value; the
    # field is kept so old checkpoints' configs load unchanged)
    temporal_type: str = "spatiotemporal"

    # Temporal mixer inside SpatiotemporalBlock:
    #   "attention"  – causal temporal self-attention over the full history
    #                  (KV cache grows O(T) at inference)
    #   "recurrent"  – token-wise gated linear recurrence: episode memory
    #                  lives in a fixed-size state ("in the neurons"),
    #                  O(1) per step, full BPTT during training
    temporal_mixer: str = "attention"

    # Spatial encoder
    encoder_channels: int = 192

    # Action space
    n_directions: int = 4
    n_split: int = 2  # 0=all-but-1, 1=50%
    # AverageJoe-style flat joint-action decoder: ONE categorical over
    # (2 * n_directions * N + 1) arrows instead of the factorized
    # Source→Direction hierarchy. Sticky in checkpoints like
    # hl_gauss_value — a resume keeps whichever decoder trained.
    joint_decoder: bool = False

    # Self-supervised next-observation prediction from the Stage-1 belief
    # tokens (game-agnostic memory pressure: the belief must retain whatever
    # is needed to predict the next frame — e.g. fogged tiles re-entering
    # vision). Spatiotemporal path only.
    predict_next_obs: bool = False

    # HL-Gauss distributional value head (phase-6, AverageJoe-style):
    # the critic outputs a categorical distribution over hl_gauss_bins
    # support points in [-vmax, vmax] and is trained with cross-entropy
    # against a Gaussian-smeared target (histogram loss). The scalar value
    # exposed to GAE/eval is the distribution's expectation, so every
    # consumer of out["value"] is unchanged. vmax=1.5 covers the sparse
    # ±1 terminal plus the ±0.35-scaled truncation score with margin;
    # sigma≈2.5 bin widths mirrors AverageJoe's 128-bin/σ0.04 on [-1,1].
    # Sticky via checkpoint config like the input-feature flags.
    #
    # The same defaults suit IL (training/train_il_game.py value_losses,
    # targets = final outcomes in [-1, 1]: W/D/L, or mahjong placements
    # ±1, ±1/3):
    #   - vmax 1.5 > 1 is deliberate: a ±1 target's Gaussian (±4σ = ±0.24)
    #     stays inside the support, so its smeared mean is exactly ±1.  With
    #     vmax = 1 half of it would be cut at the edge and renormalised,
    #     pulling the CE-optimal expectation in to ≈ ±(1 − 0.8σ) ≈ ±0.95.
    #   - 128 bins → resolution 3/128 ≈ 0.023, far below the closest target
    #     levels (mahjong, 2/3 apart); σ ≈ 2.6 bins smooths neighbouring
    #     bins (Farebrother et al. 2024 recommend σ ≈ 0.75 bin widths;
    #     either works — the IL target noise is the outcome itself, not σ).
    # Changing any of the three would change what existing HL-Gauss
    # checkpoints compute; a new default needs a LEGACY_CONFIG_DEFAULTS entry.
    hl_gauss_value: bool = False
    hl_gauss_bins: int = 128
    hl_gauss_vmax: float = 1.5
    hl_gauss_sigma: float = 0.06



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
    # Lux adapter (model/games/lux.py): no unit-cell PE, the MLP sap head
    "lux_unit_cell_pe": False,
    "lux_sap_head": "mlp",
    # Clash Royale adapter (model/games/clash_royale.py): frame-index wait PE
    "cr_wait_time": "frame",
}

#: LEGACY_CONFIG_DEFAULTS keys of ONE game's config class → a marker field
#: every config of that game ever saved carries.  They are applied only to
#: dicts that have the marker, so other games' configs do not grow keys
#: their class rejects (build_net_tolerant would drop them, loudly).
LEGACY_CONFIG_SCOPE = {
    "lux_unit_cell_pe": "n_unit_actions",
    "lux_sap_head": "n_unit_actions",
    "cr_wait_time": "cr_global_features",
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


def config_from_dict(d: dict, cls=None):
    """Build ``cls`` (default ModelConfig) from a checkpoint config dict,
    dropping unknown keys and applying :func:`upgrade_config_dict`."""
    cls = cls or ModelConfig
    d = upgrade_config_dict(d)
    return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------
# Legacy rules that need the whole checkpoint, not just its config dict
# ----------------------------------------------------------------------

#: the config field that landed 24 minutes after GeneralsEnv's sighted-
#: mountain memory went in, ON by default (2026-08-11: memory c56c5e2 at
#: 21:07, joint_decoder 164e37d at 21:32).  training/train_rl.py never set
#: the env's memory_vision, so a train_rl config dict carrying this key was
#: written by code whose env had the memory on; one without it predates the
#: memory (phases 1-7).
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
    the memory away from the train_rl lineages that trained with it."""
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


def checkpoint_config(ckpt: Mapping[str, Any], cls=None):
    """``cls`` (default ModelConfig) from a single-game checkpoint dict:
    :func:`config_from_dict` of its ``config`` with the checkpoint-level
    legacy rules (:func:`checkpoint_config_dict`) — what every loader of a
    network that will play in an env must use."""
    return config_from_dict(checkpoint_config_dict(ckpt), cls)
