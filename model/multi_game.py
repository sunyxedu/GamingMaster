"""One shared core, many games.

A multi-game model is a set of GameNetworks (one per game) whose CORE
modules are the same Python objects:

    shared (the "game brain"):  Stage-1 spatiotemporal layers, the Stage-2
                                core, HL/LL decision tokens and their
                                positions, pre/mid/post norms
    per game:                   ObservationEncoder (obs → tokens, incl. the
                                token positional embedding), ActionDecoder,
                                value head

The value head is per game on purpose: value targets have game-specific
scale and meaning (outcome, placement, score), the same reason multi-game
Q-learning keeps one head per game (Kumar et al. 2022, arXiv 2211.15144).

Adapting the brain to a new game (the HPT / frozen-trunk recipe, arXiv
2409.20537; Lu et al. 2021, arXiv 2103.05247): build that game's network,
``load_core`` the brain, ``freeze_core`` it — optionally leaving the core's
normalisation parameters trainable, which Lu et al. found essential for a
frozen transformer — and train only the adapter.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import torch
import torch.nn as nn

from model.config import upgrade_config_dict

#: GameNetwork attributes that make up the shared core
SHARED_CORE_ATTRS = ("st_layers", "core", "trm", "hl_token", "ll_token",
                     "hl_pos", "ll_pos", "pre_norm", "mid_norm", "post_norm")

#: config fields that must agree for games to share one core
CORE_CONFIG_FIELDS = (
    "d_model", "n_heads", "d_ff", "dropout", "n_st_layers", "temporal_type",
    "temporal_mixer", "core_type", "core_expansion", "core_layers",
    "rd_prelude_layers", "rd_recurrent_layers", "rd_coda_layers",
    "rd_mean_iters", "rd_backprop_iters", "rd_max_iters", "rd_eval_iters",
    "moe_experts", "moe_top_k", "moe_shared_experts", "moe_expansion",
    "trm_arch", "trm_H_cycles", "trm_L_cycles", "trm_block_layers",
    "trm_expansion", "halt_max_steps", "qk_norm",
)

_NORM_TYPES = (nn.LayerNorm, nn.RMSNorm)


def is_core_key(name: str) -> bool:
    return any(name == a or name.startswith(a + ".") for a in SHARED_CORE_ATTRS)


def check_core_compatible(cfgs: Mapping[str, object]) -> None:
    """Raise if the games' configs disagree on any core field."""
    names = list(cfgs)
    ref = cfgs[names[0]]
    for g in names[1:]:
        for f in CORE_CONFIG_FIELDS:
            a, b = getattr(ref, f, None), getattr(cfgs[g], f, None)
            if a != b:
                raise ValueError(f"games {names[0]!r} and {g!r} disagree on core "
                                 f"field {f}: {a!r} vs {b!r}")


def share_core(nets: Mapping[str, nn.Module]) -> None:
    """Make every network use the first network's core modules."""
    names = list(nets)
    check_core_compatible({g: nets[g].cfg for g in names})
    src = nets[names[0]]
    for g in names[1:]:
        for a in SHARED_CORE_ATTRS:
            if hasattr(src, a):
                setattr(nets[g], a, getattr(src, a))


class MultiGameNetwork(nn.Module):
    """ModuleDict of per-game GameNetworks sharing one core.

    ``parameters()`` yields each shared tensor once (PyTorch de-duplicates),
    so one optimizer over ``self`` updates the core once per step no matter
    how many games contributed gradients.
    """

    def __init__(self, nets: Mapping[str, nn.Module]):
        super().__init__()
        share_core(nets)
        self.nets = nn.ModuleDict(dict(nets))

    @property
    def games(self) -> List[str]:
        return list(self.nets.keys())

    def __getitem__(self, game: str) -> nn.Module:
        return self.nets[game]

    def core_state_dict(self) -> Dict[str, torch.Tensor]:
        return core_state_dict(self.nets[self.games[0]])


def build_multi_game(games: Iterable[str], core_overrides: Optional[Dict] = None,
                     game_overrides: Optional[Mapping[str, Dict]] = None
                     ) -> MultiGameNetwork:
    """Build one network per game with the same core config, then tie."""
    from model.games import build_game_network

    core_overrides = dict(core_overrides or {})
    nets = {}
    for g in games:
        kw = {**core_overrides, **dict((game_overrides or {}).get(g, {}))}
        nets[g] = build_game_network(g, **kw)
    return MultiGameNetwork(nets)


def core_state_dict(net: nn.Module) -> Dict[str, torch.Tensor]:
    """The shared-core part of one game network's state dict."""
    return {k: v.detach().clone() for k, v in net.state_dict().items() if is_core_key(k)}


def load_core(net: nn.Module, core_state: Mapping[str, torch.Tensor]) -> None:
    """Load a core (from any game's network) into ``net``; strict on the
    core keys, adapter weights untouched."""
    own = {k for k in net.state_dict() if is_core_key(k)}
    missing = own - set(core_state)
    unexpected = set(core_state) - own
    if missing or unexpected:
        raise ValueError(f"core mismatch: missing {sorted(missing)[:5]}, "
                         f"unexpected {sorted(unexpected)[:5]}")
    net.load_state_dict(dict(core_state), strict=False)


def core_parameters(net: nn.Module):
    return [p for n, p in net.named_parameters() if is_core_key(n)]


def adapter_parameters(net: nn.Module):
    return [p for n, p in net.named_parameters() if not is_core_key(n)]


def freeze_core(net: nn.Module, train_norms: bool = False) -> int:
    """Freeze the shared core; with ``train_norms`` its normalisation
    parameters (and the decision-token positions) stay trainable.
    Returns the number of trainable parameters left."""
    keep = set()
    if train_norms:
        for a in SHARED_CORE_ATTRS:
            mod = getattr(net, a, None)
            if isinstance(mod, nn.Module):
                for sub in mod.modules():
                    if isinstance(sub, _NORM_TYPES):
                        keep.update(id(p) for p in sub.parameters())
        for a in ("hl_pos", "ll_pos"):
            if hasattr(net, a):
                keep.add(id(getattr(net, a)))
    for n, p in net.named_parameters():
        if is_core_key(n):
            p.requires_grad_(id(p) in keep)
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def core_config_overrides(cfg) -> Dict:
    """The core fields of ``cfg`` (a config object or a checkpoint config
    dict) as build overrides — to build a new game's network around an
    existing brain."""
    d = upgrade_config_dict(dict(cfg if isinstance(cfg, Mapping) else cfg.__dict__))
    return {f: d[f] for f in CORE_CONFIG_FIELDS if f in d}
