"""Multi-task PopArt over the per-game value heads (Hessel et al. 2019,
"Multi-task Deep Reinforcement Learning with PopArt", arXiv 1809.04474;
van Hasselt et al. 2016, arXiv 1602.07714).

Every game ``g`` keeps running statistics (μ_g, σ_g) of its value targets.
Its value head's raw output is the NORMALISED value ñ; the value in reward
units is ``V = σ_g · ñ + μ_g``.

  Pop  (Preserving Outputs Precisely): whenever (μ, σ) → (μ', σ') the head's
       final linear layer is rescaled,
           w' = (σ / σ') · w,     b' = (σ · b + μ − μ') / σ',
       so V is unchanged for every input — the policy's advantages and the
       rollout values stay valid across the statistics update.
  Art  (Adaptively Rescaling Targets): the critic regresses ñ on the
       normalised target ``(G − μ') / σ'``, so every game's value loss (and
       its gradient into the shared core) has unit scale regardless of the
       game's reward scale.

The heads are this repo's :class:`model.components.heads.ValueHead` (final
layer ``net[-1]``, Linear(d, 1)); any module whose last ``nn.Linear`` has one
output and a bias works.  :class:`~model.components.heads.HLGaussValueHead`
games (bounded support, ±1 targets) run in PASSTHROUGH mode: normalise /
denormalise are the identity, the head is never touched, the statistics
are still tracked for logging.

Stage-1 heads were trained on [-1, 1] outcome targets; the statistics start
at (0, 1), where the transform is the identity, so RL starts from the IL
critic unchanged, and by default (``prior_weight=1``) they leave it as a
plain EMA — one degenerate batch (every game a 0-reward timeout) moves σ by
``1 − decay`` of the way, instead of snapping it to ``std_min`` and scaling
the head by 1/std_min.  Once they move, EVERY consumer of a PopArt head
speaks normalised units: other losses on it (the IL replay stream's value
loss) take ``normalize(g, target)``; exports and eval snapshots should go
through :meth:`PopArt.denormalized_state_dict`, which folds (μ, σ) back
into the heads so downstream code reads reward units without PopArt state.

Values fed to GAE must be converted with the statistics of the WEIGHTS THAT
PRODUCED THEM.  In the synchronous loop use the learner's π_old recompute
pass (current weights, current statistics, before :meth:`PopArt.update`);
values stored by actors that ran older weights need the snapshot taken when
those weights were published: ``denormalize(g, v, stats=snap)`` with
``snap = popart.snapshot()`` recorded alongside each weight version.

Weight copies outside the model (the EMA shadow) must receive the same
affine rescale to stay in the current normalisation: :meth:`PopArt.update`
takes ``mirrors`` (see :meth:`PopArt.mirror_tensors`).  Rescaling is affine
in (w, b), so applying it to an EMA shadow equals the EMA of rescaled
weights.  Adam moments are left as they are (as in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

from model.components.heads import HLGaussValueHead
from training.rl_joint.normalizers import RunningMoments
from training.rl_joint.stats import GameBatch, per_game_moments

POPART = "popart"
PASSTHROUGH = "passthrough"

#: a weight copy to keep consistent: a module (its last Linear) or (weight, bias)
Mirror = Union[nn.Module, Tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class PopArtTransform:
    """One statistics change (μ, σ) → (μ', σ') and its head rescale."""

    mu_old: float
    sigma_old: float
    mu_new: float
    sigma_new: float

    @property
    def scale(self) -> float:
        return self.sigma_old / self.sigma_new

    @property
    def shift(self) -> float:
        return (self.mu_old - self.mu_new) / self.sigma_new

    @property
    def is_identity(self) -> bool:
        return self.mu_old == self.mu_new and self.sigma_old == self.sigma_new


@torch.no_grad()
def popart_rescale_(weight: torch.Tensor, bias: torch.Tensor, tf: PopArtTransform) -> None:
    """In place: ``w ← scale·w``, ``b ← scale·b + shift`` (output-preserving)."""
    if tf.is_identity:
        return
    weight.mul_(tf.scale)
    bias.mul_(tf.scale).add_(tf.shift)


def final_linear(head: nn.Module) -> nn.Linear:
    """The output layer PopArt rescales: ``head`` itself if it is a Linear,
    else its last registered ``nn.Linear``; must have 1 output and a bias."""
    layer = head if isinstance(head, nn.Linear) else None
    if layer is None:
        for m in head.modules():
            if isinstance(m, nn.Linear):
                layer = m
    if layer is None:
        raise ValueError(f"{type(head).__name__} has no nn.Linear output layer")
    if layer.out_features != 1 or layer.bias is None:
        raise ValueError(f"PopArt needs a scalar output layer with a bias, got "
                         f"Linear({layer.in_features}, {layer.out_features}, "
                         f"bias={layer.bias is not None})")
    return layer


def _mirror_tensors(m: Mirror) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(m, nn.Module):
        layer = final_linear(m)
        return layer.weight, layer.bias
    return m


class PopArt:
    """Per-game PopArt statistics bound to per-game value heads.

    Args:
        heads:   game → value head module (:meth:`from_model` reads them off a
                 MultiGameNetwork)
        decay:   EMA decay of the target moments per update (float or per
                 game); ``1 − decay`` is the paper's step size β
        modes:   optional game → "popart" | "passthrough"; default: HL-Gauss
                 heads pass through, everything else is PopArt
        std_min / std_max: bounds on σ.  The paper clamps to [1e-4, 1e6]; the
                 default floor here is 1e-2 (rewards in this repo are O(1)):
                 degenerate targets (every game timed out at 0) would
                 otherwise scale the head by σ/σ' ~ 1e4 and send a huge
                 normalised loss into the SHARED core
        prior_weight: pseudo-count of the initial (0, 1) statistics (see
                 :class:`~training.rl_joint.normalizers.RunningMoments`);
                 1 = the paper's uncorrected EMA, 0 = the first update
                 jumps to that batch's moments
    """

    def __init__(self, heads: Mapping[str, nn.Module], decay: object = 0.99, *,
                 modes: Optional[Mapping[str, str]] = None,
                 std_min: float = 1e-2, std_max: float = 1e6, prior_weight: float = 1.0):
        self.games = list(heads)
        self.heads = dict(heads)
        self.moments = RunningMoments(self.games, decay, std_min=std_min, std_max=std_max,
                                      prior_weight=prior_weight)
        modes = dict(modes or {})
        unknown = set(modes) - set(self.games)
        if unknown:
            raise KeyError(f"modes for unknown games {sorted(unknown)}")
        self.modes: Dict[str, str] = {}
        self._layers: Dict[str, nn.Linear] = {}
        for g, head in self.heads.items():
            mode = modes.get(g, PASSTHROUGH if isinstance(head, HLGaussValueHead) else POPART)
            if mode not in (POPART, PASSTHROUGH):
                raise ValueError(f"unknown PopArt mode {mode!r} for {g!r}")
            if mode == POPART:
                if isinstance(head, HLGaussValueHead):
                    raise ValueError(f"{g!r}: an HL-Gauss head has a fixed categorical "
                                     f"support and cannot be PopArt-rescaled")
                self._layers[g] = final_linear(head)
            self.modes[g] = mode

    @classmethod
    def from_model(cls, model, games: Optional[Iterable[str]] = None, **kw) -> "PopArt":
        """PopArt over ``model[g].value_head`` for every game of a
        MultiGameNetwork (or any game → GameNetwork mapping)."""
        names = list(games) if games is not None else list(
            model.games if hasattr(model, "games") else model.keys())
        return cls({g: model[g].value_head for g in names}, **kw)

    # ---- the transform --------------------------------------------------

    def mode(self, game: str) -> str:
        return self.modes[game]

    def mean(self, game: str) -> float:
        return self.moments.mean(game) if self.modes[game] == POPART else 0.0

    def std(self, game: str) -> float:
        return self.moments.std(game) if self.modes[game] == POPART else 1.0

    def target_mean(self, game: str) -> float:
        """Running mean of the targets (tracked in both modes, for logs)."""
        return self.moments.mean(game)

    def target_std(self, game: str) -> float:
        return self.moments.std(game)

    def normalize(self, game: str, x: torch.Tensor) -> torch.Tensor:
        """Reward units → the head's normalised units (identity in
        passthrough): value targets, and old values for value clipping."""
        if self.modes[game] == PASSTHROUGH:
            return x
        return (x - self.mean(game)) / self.std(game)

    def denormalize(self, game: str, x: torch.Tensor,
                    stats: Optional[Mapping[str, Tuple[float, float]]] = None) -> torch.Tensor:
        """The head's raw output → reward units (identity in passthrough):
        the values GAE / bootstrapping consume.  ``stats``: a
        :meth:`snapshot` taken when the weights that produced ``x`` were
        current (stale actors); default the current statistics."""
        if self.modes[game] == PASSTHROUGH:
            return x
        mu, sigma = stats[game] if stats is not None else (self.mean(game), self.std(game))
        return x * sigma + mu

    def snapshot(self) -> Dict[str, Tuple[float, float]]:
        """The current (μ, σ) of every game — record it with each published
        weight version to :meth:`denormalize` that version's outputs after
        later updates (identity (0, 1) for passthrough games)."""
        return {g: (self.mean(g), self.std(g)) for g in self.games}

    # ---- updates --------------------------------------------------------

    def update(self, batches: GameBatch, *, sync: bool = True, group=None,
               mirrors: Iterable[Mapping[str, Mirror]] = ()) -> Dict[str, PopArtTransform]:
        """Fold this iteration's value targets (reward units, ``game → G`` or
        ``(G, mask)``) into the statistics and rescale every PopArt head —
        and the matching entries of each ``mirrors`` mapping — so that
        outputs in reward units are preserved.  Call before the learning
        epochs, on every rank (one all-reduce).  Returns the transform of
        every PopArt game (identity if the game had no data)."""
        moments = per_game_moments(batches, self.games, sync=sync, group=group)
        before = {g: (self.mean(g), self.std(g)) for g in self.games}
        self.moments.update_from_moments(moments)
        out = {}
        for g in self.games:
            if self.modes[g] != POPART:
                continue
            tf = PopArtTransform(*before[g], self.mean(g), self.std(g))
            self._apply(g, tf, mirrors)
            out[g] = tf
        return out

    def _apply(self, g: str, tf: PopArtTransform, mirrors: Iterable[Mapping[str, Mirror]]) -> None:
        layer = self._layers[g]
        popart_rescale_(layer.weight, layer.bias, tf)
        for mirror in mirrors:
            if g in mirror:
                w, b = _mirror_tensors(mirror[g])
                if w is layer.weight or b is layer.bias:
                    raise ValueError(f"{g!r}: a mirror aliases the live head "
                                     f"(it would be rescaled twice)")
                popart_rescale_(w, b, tf)

    def _head_names(self, model: nn.Module, g: str) -> Tuple[List[str], List[str]]:
        """Every name (aliases included) of game ``g``'s final-layer weight
        and bias in ``model``, canonical (first) name first."""
        layer = self._layers[g]
        wn: List[str] = []
        bn: List[str] = []
        for n, p in model.named_parameters(remove_duplicate=False):
            if p is layer.weight:
                wn.append(n)
            elif p is layer.bias:
                bn.append(n)
        if not wn or not bn:
            raise KeyError(f"{g!r}: its value head is not part of the given model")
        return wn, bn

    def mirror_tensors(self, model: nn.Module, named: Mapping[str, torch.Tensor]
                       ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Map each PopArt game to its (weight, bias) entries in ``named``
        — a name → tensor dict over ``model``'s parameter names, e.g.
        ``ModelEMA.shadow`` — for use as an :meth:`update` mirror.  Wrapper
        prefixes (DDP ``module.``, torch.compile ``_orig_mod.``) on either
        side are ignored.  Only a FROZEN head (``requires_grad=False``, which
        ModelEMA does not shadow) may be absent; any other missing head
        raises ``KeyError`` — silently skipping it would leave the copy's
        head in stale normalisation units."""
        lookup = _canonical_index(named)
        out = {}
        for g, layer in self._layers.items():
            wn, bn = self._head_names(model, g)
            wk = next((lookup[k] for k in map(_canonical, wn) if k in lookup), None)
            bk = next((lookup[k] for k in map(_canonical, bn) if k in lookup), None)
            if wk is None or bk is None:
                if not (layer.weight.requires_grad or layer.bias.requires_grad):
                    continue
                raise KeyError(f"{g!r}: value head parameters {wn[0]!r} / {bn[0]!r} are "
                               f"missing from the mirror mapping")
            out[g] = (named[wk], named[bk])
        return out

    @torch.no_grad()
    def denormalized_state_dict(self, model: nn.Module,
                                state: Optional[Mapping[str, torch.Tensor]] = None
                                ) -> Dict[str, torch.Tensor]:
        """``state`` (default ``model.state_dict()``; e.g.
        ``ema.ema_model_state(model)`` for the EMA) with every PopArt head's
        final layer folded back into reward units, ``w·σ`` and ``b·σ + μ``:
        a model loading it outputs ``denormalize(g, head)`` directly, i.e.
        the transform (0, 1).  Use for exports, eval / league snapshots and
        anything that reads ``out["value"]`` without PopArt state.  Returns
        a new dict (folded tensors cloned, the rest shared)."""
        out = dict(model.state_dict() if state is None else state)
        lookup = _canonical_index(out)
        for g in self._layers:
            wn, bn = self._head_names(model, g)
            mu, sigma = self.mean(g), self.std(g)
            wkeys = {lookup[k] for k in map(_canonical, wn) if k in lookup}
            bkeys = {lookup[k] for k in map(_canonical, bn) if k in lookup}
            if not wkeys or not bkeys:
                raise KeyError(f"{g!r}: value head {wn[0]!r} / {bn[0]!r} not in the state dict")
            for k in wkeys:
                out[k] = out[k].detach().clone().mul_(sigma)
            for k in bkeys:
                out[k] = out[k].detach().clone().mul_(sigma).add_(mu)
        return out

    # ---- persistence ----------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {"modes": dict(self.modes), "moments": self.moments.state_dict()}

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True, *,
                        mirrors: Iterable[Mapping[str, Mirror]] = ()
                        ) -> Dict[str, PopArtTransform]:
        """Restore the statistics.  The head weights live in the model
        checkpoint, saved in the same normalisation as these statistics —
        always save and restore the two together, and load the model (and
        EMA) weights FIRST.

        σ = clamp(std, std_min, std_max) is part of the transform the saved
        heads assume.  If this instance's bounds differ from the
        checkpoint's (a retuned resume) and move some game's σ, that game's
        head — and its entries in ``mirrors`` (the EMA shadow, see
        :meth:`mirror_tensors`) — is Pop-rescaled so outputs in reward units
        are unchanged.  Returns those transforms (empty when nothing moved).
        """
        mirrors = list(mirrors)
        for g, mode in state["modes"].items():
            if g in self.modes and self.modes[g] != mode:
                raise ValueError(f"{g!r}: checkpoint PopArt mode {mode!r} != {self.modes[g]!r}")
        saved = state["moments"]
        self.moments.load_state_dict(saved, strict=strict)
        lo = float(saved.get("std_min", self.moments.std_min))
        hi = float(saved.get("std_max", self.moments.std_max))
        out = {}
        for g in self._layers:
            if g not in saved["m"]:
                continue
            old, new = self.moments.std(g, lo, hi), self.std(g)
            if old != new:
                mu = self.mean(g)
                tf = PopArtTransform(mu, old, mu, new)
                self._apply(g, tf, mirrors)
                out[g] = tf
        return out


_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def _canonical(name: str) -> str:
    """A parameter name without leading wrapper prefixes (DDP, compile)."""
    stripped = True
    while stripped:
        stripped = False
        for pre in _WRAPPER_PREFIXES:
            if name.startswith(pre):
                name, stripped = name[len(pre):], True
    return name


def _canonical_index(named: Mapping[str, object]) -> Dict[str, str]:
    return {_canonical(k): k for k in named}

