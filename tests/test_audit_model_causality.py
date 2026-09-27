"""Model-level causality / information-flow audit of every registered game
network (``model.games.registry``), as the trainers build it.

At decision time frame t may only use frames <= t of its own stream.  IL
trains all frames of a sequence in one forward (and several sequences in
one batch), so the network itself must not let frame t read

  (a) later frames of its stream: perturbing obs / legal / labels of frames
      > t leaves every [B, T, ...] output at frames <= t unchanged — both
      temporal mixers (recurrent, attention), both core families (legacy
      TRM, recurrent-depth + MoE), eval() and train() (dropout on);
  (b) other samples of the batch (eval: rows are independent — also what
      makes the RL learner's batched seat streams private);
  (c) another frame's teacher-forcing labels: perturbing the labels of every
      frame != t leaves frame t unchanged (a later head conditions on the
      ground truth of its own frame only);
  (d) another frame's legal masks (same test on the ``legal`` dict);
  (e) the decode context (lux unit cells, clash royale match clock) is a
      per-frame function of that frame's own obs.

Conv-encoder normalisation (``cfg.encoder_norm``, model/components/encoder.py).
The default, a per-frame GroupNorm ("group"; one group = LayerNorm over
(C, H, W) by default, 4 groups tested too), makes train mode exactly causal
and the samples of a batch independent (``test_forward_causal_train_mode``,
``test_samples_independent_train_mode``: every game, no exception).  LEGACY
LEAK, kept on purpose: checkpoints written before the field existed rebuild
their BatchNorm2d encoders bit-exactly ("batch"), whose train-mode batch
statistics span every frame of every sequence in the batch, so frame t sees
(aggregated) statistics of future frames and of the other samples.  The
``..._legacy_batchnorm`` tests pin that as strict xfails, and
``..._batchnorm_frozen`` shows it is the only train-mode coupling of the
legacy layout (train mode with just the BN layers in eval mode is exactly
causal).  Eval-mode tests run both layouts: legacy checkpoints are deployed
too.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional

import pytest
import torch
import torch.nn as nn

from model.games import GAME_REGISTRY, build_game_network
from model.games.gmrules import rules_games
from training.train_il_game import LABEL_KEYS, STATE_LABEL_KEYS

B, T = 3, 7
T_CUT = 3            # frames 0..T_CUT are the "past", T_CUT+1.. the "future"

BASE = dict(d_model=32, n_heads=4, d_ff=64, n_st_layers=2, encoder_channels=8, dropout=0.1)
CORES = {
    "trm": dict(core_type="trm", trm_H_cycles=2, trm_L_cycles=2, trm_block_layers=1,
                halt_max_steps=2),
    "rd_moe": dict(core_type="recurrent_depth", rd_prelude_layers=1, rd_recurrent_layers=1,
                   rd_coda_layers=1, rd_mean_iters=2, rd_backprop_iters=1, core_layers=1,
                   moe_experts=4, moe_top_k=2),
}
MIXERS = ("recurrent", "attention")

GAME_OVERRIDES = {
    "generals": dict(grid_h=5, grid_w=5),
    "chess": dict(board_h=6, board_w=6, board_global_features=10),
    "go": dict(board_h=7, board_w=7, board_global_features=6),
    "othello": {},
    "gomoku": dict(board_h=9, board_w=9),
    "shogi": dict(board_h=5, board_w=5, shogi_global_features=7),
    "lux": dict(map_h=6, map_w=6, max_units=5, lux_sap_range=2, lux_sap_head="pointer"),
    "mahjong": {},
    "pokemon": dict(pokemon_species_buckets=64),
    "clash_royale": dict(arena_h=8, arena_w=6, n_cards=20),
    "hearthstone": dict(hand_slots=5, board_slots=3),
    "starcraft": dict(minimap_h=16, minimap_w=16, max_units=6),
}

#: games with a normalised conv encoder (cfg.encoder_norm): per-frame GroupNorm
#: by default, BatchNorm2d on batch statistics in train mode under "batch"
CONV_ENCODER_GAMES = frozenset({"generals", "chess", "go", "othello", "gomoku", "shogi", "lux",
                                "clash_royale", "starcraft", "breakthrough",
                                "chess_variants", "xiangqi"})
#: cfg.encoder_norm values: the default, and the legacy layout
NORMS = ("group", "batch")

# gmrules games (model/games/gmrules/): the candidate-action ones go through the
# generic candidate branches below (dims = their simulator's spec); custom schemas
# need their own entry (breakthrough, xiangqi: the from-to board branch, like chess)
GAME_OVERRIDES.update({g: {} for g in rules_games() if g not in GAME_OVERRIDES})
GAME_OVERRIDES["breakthrough"] = dict(board_h=6, board_w=6)

#: from-to decoder games (legal from / to_of, labels from_idx / to_idx)
FROMTO_GAMES = frozenset({"chess", "breakthrough", "chess_variants", "xiangqi"})

def _fromto_tokens(game: str, cfg) -> int:
    """Decoder tokens of a from-to game: the board squares (+ chess_variants' pocket
    slots, the from-token of a drop)."""
    extra = 2 * cfg.n_hand_piece_types if game == "chess_variants" else 0
    return cfg.board_h * cfg.board_w + extra


def _is_candidate(game: str) -> bool:
    entry = rules_games().get(game)
    return entry is not None and entry.is_candidate


# ---------------------------------------------------------------------------
# random, schema-correct batches per game
# ---------------------------------------------------------------------------

def _mask(g, *shape, p=0.5):
    m = torch.rand(*shape, generator=g) < p
    m[..., 0] |= ~m.any(-1)                                   # never an empty row
    return m


def _obs(game: str, cfg, g: torch.Generator) -> Dict[str, torch.Tensor]:
    r = lambda *s: torch.randn(B, T, *s, generator=g)            # noqa: E731
    bits = lambda *s: (torch.rand(B, T, *s, generator=g) < 0.3).float()   # noqa: E731
    if _is_candidate(game):
        E, A = cfg.cand_max_entities, cfg.cand_max_actions
        em = _mask(g, B, T, E, p=0.7)
        am = torch.arange(A) < torch.randint(1, A + 1, (B, T, 1), generator=g)
        o = {"entities": r(E, cfg.cand_entity_features) * em.unsqueeze(-1), "entity_mask": em,
             "globals": r(cfg.cand_global_features),
             "actions": r(A, cfg.cand_action_features) * am.unsqueeze(-1), "action_mask": am}
        if cfg.cand_entity_types > 0:
            o["entity_type"] = torch.randint(0, cfg.cand_entity_types, (B, T, E),
                                             generator=g).int() * em
        return o
    if game == "generals":
        grid = torch.rand(B, T, cfg.grid_channels, cfg.grid_h, cfg.grid_w, generator=g)
        return {"grid_seq": grid, "globals_seq": r(cfg.global_features)}
    if game == "chess_variants":
        return {"board_seq": bits(cfg.piece_channels, cfg.board_h, cfg.board_w),
                "hand_seq": bits(2 * cfg.n_hand_piece_types, cfg.hand_feature_dim),
                "globals_seq": r(cfg.board_global_features)}
    if game in ("chess", "go", "othello", "gomoku", "breakthrough", "xiangqi"):
        o = {"board_seq": bits(cfg.piece_channels, cfg.board_h, cfg.board_w)}
        if cfg.board_global_features:
            o["globals_seq"] = r(cfg.board_global_features)
        return o
    if game == "shogi":
        return {"board_seq": bits(cfg.piece_channels, cfg.board_h, cfg.board_w),
                "hand_seq": r(2 * cfg.n_hand_piece_types, cfg.hand_feature_dim),
                "globals_seq": r(cfg.shogi_global_features)}
    if game == "lux":
        U = cfg.max_units
        mask = torch.rand(B, T, U, generator=g) < 0.6
        mask[..., 0] = True
        units = r(U, cfg.unit_feature_dim)
        units[..., 0] = torch.randint(0, cfg.map_w, (B, T, U), generator=g) / cfg.map_w
        units[..., 1] = torch.randint(0, cfg.map_h, (B, T, U), generator=g) / cfg.map_h
        return {"map_seq": r(cfg.map_channels, cfg.map_h, cfg.map_w), "units_seq": units,
                "unit_mask": mask, "globals_seq": r(cfg.lux_global_features)}
    if game == "mahjong":
        from model.games.mahjong import N_MJ_PLAYERS, N_TILE_KINDS
        return {"tiles_seq": r(N_TILE_KINDS, cfg.mahjong_tile_feature_dim),
                "players_seq": r(N_MJ_PLAYERS, cfg.mahjong_player_feature_dim),
                "table_seq": r(cfg.mahjong_table_feature_dim)}
    if game == "pokemon":
        from model.games.pokemon import POKE_FEAT_ACTIVE, n_species_ids
        S = 2 * cfg.team_slots
        feats = r(S, cfg.pokemon_feature_dim)
        feats[..., POKE_FEAT_ACTIVE] = 0.0
        feats[..., 0, POKE_FEAT_ACTIVE] = 1.0
        mask = torch.rand(B, T, S, generator=g) < 0.7
        mask[..., :cfg.team_slots] = True
        ids = torch.randint(1, n_species_ids(cfg), (B, T, S), generator=g)
        return {"species_ids": ids, "poke_feats": feats, "slot_mask": mask,
                "field_seq": r(cfg.pokemon_field_features)}
    if game == "clash_royale":
        from model.games.clash_royale import n_card_tokens
        N, pre = n_card_tokens(cfg), cfg.cr_card_tokens        # deck tokens (or legacy hand slots)
        mask = torch.rand(B, T, N, generator=g) < 0.8
        mask[..., 0] = True
        return {"arena_seq": r(cfg.arena_channels, cfg.arena_h, cfg.arena_w),
                f"{pre}_ids": torch.randint(0, cfg.n_cards, (B, T, N), generator=g),
                f"{pre}_feats": r(N, cfg.card_feature_dim), f"{pre}_mask": mask,
                "globals_seq": torch.rand(B, T, cfg.cr_global_features, generator=g)}
    if game == "hearthstone":
        S = cfg.hand_slots + 2 * cfg.board_slots + 3
        mask = torch.rand(B, T, S, generator=g) < 0.7
        mask[..., -3:] = True
        return {"entities_seq": r(S, cfg.entity_feature_dim), "entity_mask": mask,
                "entity_types": torch.randint(0, cfg.n_entity_types, (B, T, S), generator=g),
                "globals_seq": r(cfg.hs_global_features)}
    if game == "starcraft":
        mask = torch.rand(B, T, cfg.max_units, generator=g) < 0.7
        mask[..., 0] = True
        return {"minimap_seq": r(cfg.minimap_channels, cfg.minimap_h, cfg.minimap_w),
                "units_seq": r(cfg.max_units, cfg.unit_feature_dim), "unit_mask": mask,
                "globals_seq": r(cfg.sc_global_features)}
    raise KeyError(game)


def _legal(game: str, cfg, obs, g) -> Optional[Dict[str, torch.Tensor]]:
    m = lambda *s: _mask(g, B, T, *s)                           # noqa: E731
    if _is_candidate(game):
        # ANDed with the frame's own candidate mask (slot 0 always legal)
        return {"action": obs["action_mask"] & m(cfg.cand_max_actions)}
    if game == "generals":
        return {"source": m(1, cfg.grid_h * cfg.grid_w).reshape(B, T, 1, cfg.grid_h, cfg.grid_w),
                "direction": _mask(g, B, T, cfg.grid_h, cfg.grid_w, 4).permute(0, 1, 4, 2, 3)}
    if game in ("chess", "go", "othello", "gomoku", "shogi", "breakthrough", "chess_variants", "xiangqi"):
        N = cfg.board_h * cfg.board_w
        if game in FROMTO_GAMES:
            N = _fromto_tokens(game, cfg)
            lg = {"from": m(N), "to_of": m(N)}
            if cfg.n_promotions > 1:
                lg["promotion_of"] = m(cfg.n_promotions)
            return lg
        if game == "shogi":
            return {"mode": m(2), "from": m(N), "to_of": m(N), "promo_of": m(2),
                    "drop_piece": m(2 * cfg.n_hand_piece_types), "drop_to_of": m(N)}
        return {"place": m(N)}
    if game == "lux":
        from model.games.lux import lux_sap_legal
        xy = torch.stack([torch.round(obs["units_seq"][..., 0] * cfg.map_w),
                          torch.round(obs["units_seq"][..., 1] * cfg.map_h)], -1).long()
        return {"unit_action": m(cfg.max_units, cfg.n_unit_actions),
                "unit": obs["unit_mask"].clone(),
                "sap": lux_sap_legal(xy, obs["unit_mask"], cfg.lux_sap_range, cfg.lux_sap_range,
                                     cfg.map_w, cfg.map_h)}
    if game == "mahjong":
        return {"action_type": m(8), "tile_of": m(34)}
    if game == "pokemon":
        # both move heads' masks (identity: move_id over the vocabulary) and the
        # revealed-move ids the decoder conditions on (state, from obs)
        from model.games.pokemon import n_move_ids
        return {"action_type": m(cfg.n_action_types), "move": m(cfg.n_move_slots),
                "move_id": m(n_move_ids(cfg)),
                "switch": m(cfg.team_slots), "tera": torch.rand(B, T, generator=g) < 0.5,
                "active_idx": torch.randint(0, cfg.team_slots, (B, T), generator=g),
                "move_ids": torch.randint(0, n_move_ids(cfg), (B, T, cfg.n_move_slots),
                                          generator=g)}
    if game == "clash_royale":
        from model.games.clash_royale import n_card_tokens
        return {"card": m(n_card_tokens(cfg)), "placement": m(cfg.arena_h * cfg.arena_w)}
    if game == "hearthstone":
        S = cfg.hand_slots + 2 * cfg.board_slots + 3
        return {"action_type": m(cfg.n_action_types), "actor": m(S), "target": m(S)}
    if game == "starcraft":
        d = cfg.spatial_downsample
        return {"action_type": m(cfg.n_action_types), "actor": m(cfg.max_units),
                "target_unit": m(cfg.max_units),
                "location": m((cfg.minimap_h // d) * (cfg.minimap_w // d))}
    raise KeyError(game)


def _label_highs(game: str, cfg) -> Dict[str, int]:
    """Exclusive upper bound of every teacher-forcing label of LABEL_KEYS[game]."""
    if game == "generals":
        return {"source_y": cfg.grid_h, "source_x": cfg.grid_w}
    if game in FROMTO_GAMES:
        N = _fromto_tokens(game, cfg)
        return {"from_idx": N, "to_idx": N}
    if game == "shogi":
        N = cfg.board_h * cfg.board_w
        return {"dropped": 2, "from_idx": N, "to_idx": N, "piece_idx": cfg.n_hand_piece_types}
    if game == "starcraft":
        return {"action_type": cfg.n_action_types, "actor_idx": cfg.max_units}
    if game == "hearthstone":
        return {"action_type": cfg.n_action_types,
                "actor_idx": cfg.hand_slots + 2 * cfg.board_slots + 3}
    if game == "clash_royale":
        from model.games.clash_royale import n_card_tokens
        return {"card_slot": n_card_tokens(cfg)}
    if game == "mahjong":
        return {"action_type": 8}
    if game == "pokemon":
        return {"action_type": cfg.n_action_types, "active_idx": cfg.team_slots}
    return {}


def _labels(game: str, cfg, g, self_sample: bool = True) -> Optional[Dict[str, torch.Tensor]]:
    """Ground-truth conditioning labels; with ``self_sample`` some frames of
    the PAST carry −1 (the decoder samples its own conditioning there —
    the tf_prob < 1 path), identical in both runs of a comparison."""
    highs = _label_highs(game, cfg)
    if not highs:
        return None
    out = {}
    for k, hi in highs.items():
        v = torch.randint(0, hi, (B, T), generator=g)
        if self_sample and k not in STATE_LABEL_KEYS.get(game, ()):
            v[:, 0] = -1
            v[0, 1] = -1
        out[k] = v
    return out


def _splice(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], keep: torch.Tensor):
    """``a`` on frames where ``keep`` [T] bool, ``b`` elsewhere (every tensor)."""
    return {k: torch.where(keep.view(1, T, *([1] * (v.dim() - 2))), v, b[k]) for k, v in a.items()}


def _future_labels(labels, g_hi):
    """Replacement labels for the future frames: fresh draws, never −1 (the
    −1 frames are in the shared past, so the decoders' 'any row to sample'
    branch — and its RNG use — is the same in both runs)."""
    return {k: torch.randint(0, int(v.max()) + 1, v.shape, generator=g_hi)
            for k, v in labels.items()}


# ---------------------------------------------------------------------------
# network + forward helpers
# ---------------------------------------------------------------------------

_NETS: Dict = {}


def _net(game: str, mixer: str = "recurrent", core: str = "trm",
         norm: str = "group", groups: int = 1) -> nn.Module:
    """The game's network (cached); ``norm`` = cfg.encoder_norm ("group" is the
    default config, "batch" the legacy-checkpoint layout), ``groups`` =
    cfg.encoder_norm_groups."""
    key = (game, mixer, core, norm, groups)
    if key not in _NETS:
        torch.manual_seed(0)
        net = build_game_network(game, **BASE, **CORES[core], temporal_mixer=mixer,
                                 encoder_norm=norm, encoder_norm_groups=groups,
                                 **GAME_OVERRIDES[game])
        with torch.no_grad():
            for m in net.modules():
                # non-trivial BatchNorm running statistics for the eval-mode runs
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.running_mean.uniform_(-0.5, 0.5)
                    m.running_var.uniform_(0.5, 2.0)
                # and non-trivial GroupNorm affine parameters (init: 1 / 0)
                if isinstance(m, nn.GroupNorm):
                    m.weight.uniform_(0.5, 1.5)
                    m.bias.uniform_(-0.5, 0.5)
        _NETS[key] = net
    return _NETS[key]


@contextlib.contextmanager
def _mode(net: nn.Module, mode: str):
    """eval | train | train_bn_frozen (train mode, BatchNorm on running stats)."""
    was = net.training
    net.train(mode != "eval")
    if mode == "train_bn_frozen":
        for m in net.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    try:
        yield
    finally:
        net.train(was)


def _run(net, obs, legal, labels, seed: int = 1234) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)        # same dropout masks / recurrence draws / samples in both runs
    with torch.no_grad():
        out = net(obs, legal=legal, labels=labels, act_steps=2)
    return {k: v for k, v in out.items()
            if torch.is_tensor(v) and v.dim() >= 2 and tuple(v.shape[:2]) == (B, T)}


def _assert_frames_equal(a: Dict, b: Dict, frames, rows, what: str, exact: bool) -> None:
    assert set(a) == set(b) and a
    for k in sorted(a):
        x, y = a[k][rows][:, frames], b[k][rows][:, frames]
        if exact:
            ok = torch.equal(x, y)
        else:          # MoE: per-expert row counts change → summation-order ulps only
            ok = torch.allclose(x.float(), y.float(), rtol=1e-5, atol=1e-6)
        if not ok:
            diff = (x.float() - y.float()).abs().max().item()
            raise AssertionError(f"{what}: output {k!r} changed (max |Δ| = {diff:.3g})")


def _batch(game, net, seed=0):
    cfg = net.cfg
    g = torch.Generator().manual_seed(seed)
    obs = _obs(game, cfg, g)
    return obs, _legal(game, cfg, obs, g), _labels(game, cfg, g), g


def _perturb_future(game, net, obs, legal, labels):
    """(obs, legal, labels) with every frame > T_CUT redrawn."""
    cfg = net.cfg
    g2 = torch.Generator().manual_seed(99)
    obs2 = _obs(game, cfg, g2)
    legal2 = _legal(game, cfg, obs2, g2)
    past = torch.arange(T) <= T_CUT
    o = _splice(obs, obs2, past)
    lg = _splice(legal, legal2, past) if legal is not None else None
    lb = _splice(labels, _future_labels(labels, g2), past) if labels is not None else None
    changed = any(not torch.equal(o[k], obs[k]) for k in obs)
    assert changed
    return o, lg, lb


def _norm_layers(net: nn.Module, kind) -> list:
    return [m for m in net.modules() if isinstance(m, kind)]


def test_registry_and_label_tables_cover_every_game():
    assert set(GAME_OVERRIDES) == set(GAME_REGISTRY)
    for game in GAME_REGISTRY:
        net = _net(game)
        assert set(_label_highs(game, net.cfg)) == set(LABEL_KEYS.get(game, ())), game
        # the default config: per-frame GroupNorm in every conv encoder, no
        # BatchNorm anywhere
        assert net.cfg.encoder_norm == "group"
        assert not _norm_layers(net, nn.modules.batchnorm._BatchNorm), game
        assert bool(_norm_layers(net, nn.GroupNorm)) == (game in CONV_ENCODER_GAMES), game
        # the legacy layout swaps exactly those layers for BatchNorm2d
        legacy = _net(game, norm="batch")
        assert not _norm_layers(legacy, nn.GroupNorm), game
        assert (len(_norm_layers(legacy, nn.modules.batchnorm._BatchNorm))
                == len(_norm_layers(net, nn.GroupNorm))), game


# ---------------------------------------------------------------------------
# (a) causality over frames
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("core", sorted(CORES))
@pytest.mark.parametrize("mixer", MIXERS)
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_forward_causal_eval(game, mixer, core, norm):
    net = _net(game, mixer, core, norm)
    obs, legal, labels, _ = _batch(game, net)
    o2, lg2, lb2 = _perturb_future(game, net, obs, legal, labels)
    with _mode(net, "eval"):
        a = _run(net, obs, legal, labels)
        b = _run(net, o2, lg2, lb2)
    _assert_frames_equal(a, b, slice(0, T_CUT + 1), slice(None),
                         f"{game}/{mixer}/{core}/{norm} eval: frames <= {T_CUT} see the future",
                         exact=core != "rd_moe")
    # and the perturbation is live: a later frame does change
    assert any(not torch.equal(a[k][:, T_CUT + 1:], b[k][:, T_CUT + 1:]) for k in a)


@pytest.mark.parametrize("mixer", MIXERS)
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_forward_causal_train_mode_batchnorm_frozen(game, mixer):
    """The legacy layout (encoder_norm="batch") in train mode (dropout,
    recurrence sampling, MoE load tally) with only the BatchNorm layers on
    running statistics: exactly causal, so BatchNorm is the ONLY train-mode
    coupling across frames of legacy checkpoints."""
    for core in CORES:
        net = _net(game, mixer, core, "batch")
        obs, legal, labels, _ = _batch(game, net)
        o2, lg2, lb2 = _perturb_future(game, net, obs, legal, labels)
        with _mode(net, "train_bn_frozen"):
            a = _run(net, obs, legal, labels)
            b = _run(net, o2, lg2, lb2)
        _assert_frames_equal(a, b, slice(0, T_CUT + 1), slice(None),
                             f"{game}/{mixer}/{core} train (BN frozen)", exact=core != "rd_moe")


_LEGACY_BN_LEAK = pytest.mark.xfail(
    strict=True, reason="LEGACY LEAK (encoder_norm='batch', checkpoints predating the field): "
                        "train-mode BatchNorm2d batch statistics span every frame (and sample) "
                        "of the batch — frame t sees future frames")


def _check_forward_causal_train_mode(net, game):
    obs, legal, labels, _ = _batch(game, net)
    o2, lg2, lb2 = _perturb_future(game, net, obs, legal, labels)
    with _mode(net, "train"):
        a = _run(net, obs, legal, labels)
        b = _run(net, o2, lg2, lb2)
    _assert_frames_equal(a, b, slice(0, T_CUT + 1), slice(None),
                         f"{game} train: frames <= {T_CUT} see the future", exact=True)


@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_forward_causal_train_mode(game, groups):
    """Train mode (the IL / PPO forward) is exactly causal for every game with
    the default per-frame encoder norm (one group, and 4 groups)."""
    _check_forward_causal_train_mode(_net(game, "recurrent", "trm", groups=groups), game)


@_LEGACY_BN_LEAK
@pytest.mark.parametrize("game", sorted(CONV_ENCODER_GAMES))
def test_forward_causal_train_mode_legacy_batchnorm(game):
    _check_forward_causal_train_mode(_net(game, "recurrent", "trm", "batch"), game)


# ---------------------------------------------------------------------------
# (b) independence of the samples of a batch
# ---------------------------------------------------------------------------

def _perturb_sample(game, net, obs, legal, labels, row: int):
    g2 = torch.Generator().manual_seed(7)
    obs2 = _obs(game, net.cfg, g2)
    legal2 = _legal(game, net.cfg, obs2, g2)
    sel = torch.zeros(B, dtype=torch.bool)
    sel[row] = True

    def mix(a, b):
        return {k: torch.where(sel.view(B, *([1] * (v.dim() - 1))), b[k], v) for k, v in a.items()}
    return (mix(obs, obs2), mix(legal, legal2) if legal is not None else None, labels)


@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("mixer", MIXERS)
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_samples_independent_eval(game, mixer, norm):
    """Eval mode: sample 0's outputs ignore sample 2 entirely — the RL learner
    batches both seats' streams (mirror self-play) and the opponents' rows in
    one forward / step, so a coupling would hand one seat the other's view."""
    for core in CORES:
        net = _net(game, mixer, core, norm)
        obs, legal, labels, _ = _batch(game, net)
        o2, lg2, lb2 = _perturb_sample(game, net, obs, legal, labels, row=2)
        with _mode(net, "eval"):
            a = _run(net, obs, legal, labels)
            b = _run(net, o2, lg2, lb2)
        _assert_frames_equal(a, b, slice(None), slice(0, 2),
                             f"{game}/{mixer}/{core}/{norm} eval rows", exact=core != "rd_moe")


def _check_samples_independent_train_mode(net, game):
    obs, legal, labels, _ = _batch(game, net)
    o2, lg2, lb2 = _perturb_sample(game, net, obs, legal, labels, row=2)
    with _mode(net, "train"):
        a = _run(net, obs, legal, labels)
        b = _run(net, o2, lg2, lb2)
    _assert_frames_equal(a, b, slice(None), slice(0, 2), f"{game} train rows", exact=True)


@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_samples_independent_train_mode(game, groups):
    _check_samples_independent_train_mode(_net(game, "recurrent", "trm", groups=groups), game)


@_LEGACY_BN_LEAK
@pytest.mark.parametrize("game", sorted(CONV_ENCODER_GAMES))
def test_samples_independent_train_mode_legacy_batchnorm(game):
    _check_samples_independent_train_mode(_net(game, "recurrent", "trm", "batch"), game)


# ---------------------------------------------------------------------------
# (c) teacher forcing and (d) legal masks: own frame only
# ---------------------------------------------------------------------------

def _other_frames(t: int) -> torch.Tensor:
    keep = torch.zeros(T, dtype=torch.bool)
    keep[t] = True
    return keep


@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("game", sorted(g for g in GAME_OVERRIDES if LABEL_KEYS.get(g)))
def test_labels_condition_their_own_frame_only(game, mode):
    """Perturb the labels of EVERY frame but t (past and future): frame t's
    logits / committed indices / masks are unchanged.  (Train mode included:
    labels never reach the encoder, whose legacy BatchNorm is the only
    train-mode coupling of a legacy layout.)"""
    net = _net(game, "recurrent", "trm")
    obs, legal, labels, _ = _batch(game, net)
    labels = _labels(game, net.cfg, torch.Generator().manual_seed(3), self_sample=False)
    other = _labels(game, net.cfg, torch.Generator().manual_seed(4), self_sample=False)
    for t in (0, T_CUT, T - 1):
        lb2 = _splice(labels, other, _other_frames(t))
        assert any(not torch.equal(lb2[k], labels[k]) for k in labels)
        with _mode(net, mode):
            a = _run(net, obs, legal, labels)
            b = _run(net, obs, legal, lb2)
        _assert_frames_equal(a, b, slice(t, t + 1), slice(None),
                             f"{game} {mode}: frame {t} reads another frame's labels", exact=True)
        # the labels do matter on their own frame (the test is live)
        lb3 = _splice(other, labels, _other_frames(t))
        with _mode(net, mode):
            c = _run(net, obs, legal, lb3)
        assert any(not torch.equal(a[k][:, t], c[k][:, t]) for k in a), (game, t)


@pytest.mark.parametrize("mode", ["eval", "train"])
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_legal_masks_condition_their_own_frame_only(game, mode):
    net = _net(game, "recurrent", "trm")
    obs, legal, labels, g = _batch(game, net)
    g2 = torch.Generator().manual_seed(5)
    legal2 = _legal(game, net.cfg, obs, g2)
    for t in (0, T_CUT, T - 1):
        lg = _splice(legal, legal2, _other_frames(t))
        with _mode(net, mode):
            a = _run(net, obs, legal, labels)
            b = _run(net, obs, lg, labels)
        _assert_frames_equal(a, b, slice(t, t + 1), slice(None),
                             f"{game} {mode}: frame {t} reads another frame's legal mask",
                             exact=True)


# ---------------------------------------------------------------------------
# (e) decode context: per-frame function of the frame's own obs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["lux", "clash_royale", "kuhn_poker"])
def test_decode_context_is_per_frame(game):
    net = _net(game)
    enc = net.obs_encoder
    obs, *_ = _batch(game, net)
    obs2 = _obs(game, net.cfg, torch.Generator().manual_seed(11))
    ctx = enc.decode_context(obs)
    assert ctx, game
    for t in range(T):
        o = _splice(obs, obs2, _other_frames(t))
        c2 = enc.decode_context(o)
        for k in ctx:
            assert torch.equal(ctx[k][:, t], c2[k][:, t]), (game, k, t)
    # which obs keys it reads: only the acting player's own view
    reads = {"lux": {"units_seq", "unit_mask"}, "clash_royale": {"globals_seq"},
             "kuhn_poker": {"action_mask"}}[game]
    for k in obs:
        o = dict(obs)
        o[k] = obs2[k]
        same = all(torch.equal(ctx[c], enc.decode_context(o)[c]) for c in ctx)
        assert same == (k not in reads), (game, k)


# ---------------------------------------------------------------------------
# step() == forward() prefix: the per-frame inference path sees the same past
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("mixer", MIXERS)
@pytest.mark.parametrize("game", sorted(GAME_OVERRIDES))
def test_step_chain_reproduces_forward_logits(game, mixer, norm):
    """Chaining GameNetwork.step over frames 0..T-1 (the deployed bot) gives
    the full-sequence forward's logits at every frame (eval, no labels, exact
    masks): IL trains exactly the function the bot runs — no information in
    training that the step path does not have."""
    net = _net(game, mixer, "trm", norm)
    obs, legal, _, _ = _batch(game, net)
    with _mode(net, "eval"), torch.no_grad():
        full = net(obs, legal=legal, labels=None, act_steps=2)
        cache = None
        for t in range(T):
            o = {k: v[:, t] for k, v in obs.items()}
            lg = {k: v[:, t] for k, v in legal.items()} if legal else None
            torch.manual_seed(t)
            out = net.step(o, legal=lg, st_cache=cache, step_idx=t, act_steps=2)
            cache = out["st_cache"]
            for k, v in out.items():
                if (k.endswith("_logits") or k.endswith("_logit")) and k in full \
                        and not _conditioned_on_sample(game, k):
                    torch.testing.assert_close(v.float(), full[k][:, t].float(), rtol=1e-4,
                                               atol=1e-5, msg=f"{game}/{mixer} frame {t} {k}")


def _conditioned_on_sample(game: str, key: str) -> bool:
    """Heads conditioned on an earlier head's SAMPLED index (differs between
    two independent draws, not an information difference)."""
    first = {"generals": ("source_logits",),
             "chess": ("from_logits",), "shogi": ("mode_logit", "mode_logits", "from_logits",
                                                  "piece_logits"),
             "breakthrough": ("from_logits",), "chess_variants": ("from_logits",),
             "xiangqi": ("from_logits",),
             "starcraft": ("type_logits",), "hearthstone": ("type_logits",),
             "clash_royale": ("wait_logits", "card_logits"), "mahjong": ("type_logits",),
             # the move / switch heads condition on the sampled type; tera does not
             "pokemon": ("action_type_logits", "tera_logit")}
    return game in first and key not in first[game]
