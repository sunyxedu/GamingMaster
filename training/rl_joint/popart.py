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
critic unchanged.  Once they move, EVERY consumer of a PopArt head speaks
normalised units: other losses on it (the IL replay stream's value loss)
take ``normalize(g, target)``, and whoever reads values off a checkpoint
(eval, play scripts) needs the PopArt state saved with it.

Weight copies outside the model (the EMA shadow) must receive the same
affine rescale to stay in the current normalisation: :meth:`PopArt.update`
takes ``mirrors`` (see :meth:`PopArt.mirror_tensors`).  Rescaling is affine
in (w, b), so applying it to an EMA shadow equals the EMA of rescaled
weights.  Adam moments are left as they are (as in the paper).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

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
                 degenerate early targets (every game timed out at 0) would
                 otherwise scale the head by σ/σ' ~ 1e4 and send a huge
                 normalised loss into the SHARED core
    """

    def __init__(self, heads: Mapping[str, nn.Module], decay: object = 0.99, *,
                 modes: Optional[Mapping[str, str]] = None,
                 std_min: float = 1e-2, std_max: float = 1e6):
        self.games = list(heads)
        self.heads = dict(heads)
        self.moments = RunningMoments(self.games, decay, std_min=std_min, std_max=std_max)
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

    def denormalize(self, game: str, x: torch.Tensor) -> torch.Tensor:
        """The head's raw output → reward units (identity in passthrough):
        the values GAE / bootstrapping consume."""
        if self.modes[game] == PASSTHROUGH:
            return x
        return x * self.std(game) + self.mean(game)

    # ---- updates --------------------------------------------------------

    def update(self, batches: GameBatch, *, sync: bool = True,
               mirrors: Iterable[Mapping[str, Mirror]] = ()) -> Dict[str, PopArtTransform]:
        """Fold this iteration's value targets (reward units, ``game → G`` or
        ``(G, mask)``) into the statistics and rescale every PopArt head —
        and the matching entries of each ``mirrors`` mapping — so that
        outputs in reward units are preserved.  Call before the learning
        epochs, on every rank (one all-reduce).  Returns the transform of
        every PopArt game (identity if the game had no data)."""
        moments = per_game_moments(batches, self.games, sync=sync)
        before = {g: (self.mean(g), self.std(g)) for g in self.games}
        self.moments.update_from_moments(moments)
        mirrors = list(mirrors)
        out = {}
        for g in self.games:
            if self.modes[g] != POPART:
                continue
            tf = PopArtTransform(*before[g], self.mean(g), self.std(g))
            layer = self._layers[g]
            popart_rescale_(layer.weight, layer.bias, tf)
            for mirror in mirrors:
                if g in mirror:
                    w, b = _mirror_tensors(mirror[g])
                    if w is layer.weight or b is layer.bias:
                        raise ValueError(f"{g!r}: a mirror aliases the live head "
                                         f"(it would be rescaled twice)")
                    popart_rescale_(w, b, tf)
            out[g] = tf
        return out

    def mirror_tensors(self, model: nn.Module, named: Mapping[str, torch.Tensor]
                       ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Map each PopArt game to its (weight, bias) entries in ``named``
        — a name → tensor dict over ``model``'s parameter names, e.g.
        ``ModelEMA.shadow`` — for use as an :meth:`update` mirror.  Games
        whose head is absent from ``named`` (a frozen head outside the EMA)
        are skipped."""
        # canonical (first) names, the ones ModelEMA keys its shadow by
        names = {id(p): n for n, p in model.named_parameters()}
        out = {}
        for g, layer in self._layers.items():
            wn, bn = names.get(id(layer.weight)), names.get(id(layer.bias))
            if wn is None or bn is None:
                raise KeyError(f"{g!r}: its value head is not part of the given model")
            if wn in named and bn in named:
                out[g] = (named[wn], named[bn])
        return out

    # ---- persistence ----------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {"modes": dict(self.modes), "moments": self.moments.state_dict()}

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True) -> None:
        """Restore the statistics.  The head weights are NOT touched: they
        live in the model checkpoint, saved in the same normalisation as
        these statistics — always save and restore the two together."""
        for g, mode in state["modes"].items():
            if g in self.modes and self.modes[g] != mode:
                raise ValueError(f"{g!r}: checkpoint PopArt mode {mode!r} != {self.modes[g]!r}")
        self.moments.load_state_dict(state["moments"], strict=strict)

