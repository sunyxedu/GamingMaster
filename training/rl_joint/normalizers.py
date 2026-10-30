"""Per-game running statistics and the per-game advantage normaliser.

Mixed-game PPO must not normalise advantages per mixed minibatch: the
games' advantage scales differ by orders of magnitude (a ±1 board game vs a
shaped-reward RTS), so a shared mean/std lets the loudest game set every
game's step size, and the mixture — hence the scale — changes from batch to
batch.  Each game instead keeps its own exponentially-averaged first and
second moment, updated once per iteration from that game's advantages
(summed over all ranks), and its advantages are standardised with those.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from training.rl_joint.stats import GameBatch, per_game_moments


def _check_games(own: Sequence[str], saved: Iterable[str], strict: bool, what: str) -> None:
    missing, unexpected = set(own) - set(saved), set(saved) - set(own)
    if strict and (missing or unexpected):
        raise KeyError(f"{what} state mismatch: missing {sorted(missing)}, "
                       f"unexpected {sorted(unexpected)}")


class RunningMoments:
    """Per-game EMA of the mean and the mean of squares, bias-corrected.

    Raw averages are divided by their accumulated weight (Adam's bias
    correction).  With ``prior_weight=0`` they start at zero and the first
    update yields that batch's exact moments; ``prior_weight=w > 0`` starts
    them at a (mean 0, std 1) prior worth ``w`` updates — ``w = 1`` is the
    plain, uncorrected EMA started at (0, 1) (PopArt's default: one
    degenerate batch cannot snap σ to its floor).  The weight is tracked,
    not recomputed from a step count, so a resumed run may change
    ``decay``.  Before any update a game reports mean 0 / std 1 (the
    identity transform).  ``decay`` may be a float or a per-game mapping.
    Statistics are Python floats (float64).
    """

    def __init__(self, games: Sequence[str], decay: object = 0.99,
                 std_min: float = 1e-4, std_max: float = 1e6, *,
                 prior_weight: float = 0.0):
        if not games or len(set(games)) != len(games):
            raise ValueError(f"games must be a non-empty list of distinct names, got {games!r}")
        self.games = list(games)
        self.decay = self._per_game(decay, "decay")
        for g, d in self.decay.items():
            if not 0.0 <= d < 1.0:
                raise ValueError(f"decay for {g!r} must be in [0, 1), got {d}")
        if not 0.0 < std_min <= std_max:
            raise ValueError(f"need 0 < std_min <= std_max, got {std_min}, {std_max}")
        if not prior_weight >= 0.0:
            raise ValueError(f"prior_weight must be >= 0, got {prior_weight}")
        self.std_min, self.std_max = float(std_min), float(std_max)
        self.prior_weight = float(prior_weight)
        self._m = {g: 0.0 for g in self.games}
        self._sq = {g: self.prior_weight for g in self.games}     # mean_sq prior 1
        self._w = {g: self.prior_weight for g in self.games}
        self._t = {g: 0 for g in self.games}

    def _per_game(self, value: object, name: str) -> Dict[str, float]:
        if isinstance(value, Mapping):
            missing = set(self.games) - set(value)
            if missing:
                raise KeyError(f"{name} missing for games {sorted(missing)}")
            return {g: float(value[g]) for g in self.games}
        return {g: float(value) for g in self.games}

    # ---- statistics -----------------------------------------------------

    def count(self, game: str) -> int:
        """Number of updates ``game`` has received."""
        return self._t[game]

    def mean(self, game: str) -> float:
        w = self._w[game]
        return self._m[game] / w if w > 0 else 0.0

    def mean_sq(self, game: str) -> float:
        w = self._w[game]
        return self._sq[game] / w if w > 0 else 1.0

    def std(self, game: str, std_min: Optional[float] = None,
            std_max: Optional[float] = None) -> float:
        """Running std clamped to [std_min, std_max] (default: this
        instance's bounds)."""
        lo = self.std_min if std_min is None else float(std_min)
        hi = self.std_max if std_max is None else float(std_max)
        var = max(self.mean_sq(game) - self.mean(game) ** 2, 0.0)
        return min(max(math.sqrt(var), lo), hi)

    # ---- updates --------------------------------------------------------

    def update_from_moments(self, moments: Mapping[str, Tuple[float, float, float]]
                            ) -> Dict[str, bool]:
        """Fold (count, mean, mean_sq) batch moments in; games with a zero
        count are left untouched.  Returns which games were updated."""
        done = {}
        for g in self.games:
            n, m, sq = moments.get(g, (0.0, 0.0, 0.0))
            if n <= 0:
                done[g] = False
                continue
            d = self.decay[g]
            self._m[g] = d * self._m[g] + (1.0 - d) * m
            self._sq[g] = d * self._sq[g] + (1.0 - d) * sq
            self._w[g] = d * self._w[g] + (1.0 - d)
            self._t[g] += 1
            done[g] = True
        return done

    def update(self, batches: GameBatch, *, sync: bool = True, group=None) -> Dict[str, bool]:
        """Update from per-game tensors (``game -> x`` or ``(x, mask)``);
        under torchrun call on every rank of ``group`` (one all-reduce)."""
        return self.update_from_moments(
            per_game_moments(batches, self.games, sync=sync, group=group))

    def reset(self, game: str) -> None:
        """Forget ``game``'s statistics (back to the constructor's state): its next update
        yields that batch's own moments (with ``prior_weight`` 0)."""
        self._m[game] = 0.0
        self._sq[game] = self.prior_weight
        self._w[game] = self.prior_weight
        self._t[game] = 0

    # ---- persistence ----------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {"games": list(self.games), "decay": dict(self.decay),
                "std_min": self.std_min, "std_max": self.std_max,
                "m": dict(self._m), "sq": dict(self._sq), "w": dict(self._w),
                "t": dict(self._t)}

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True) -> None:
        """Restore statistics (the configuration — decay, bounds — stays the
        constructor's).  ``strict=False`` loads the games both sides know
        and leaves new games at their initial state."""
        _check_games(self.games, state["games"], strict, type(self).__name__)
        for g in self.games:
            if g in state["m"]:
                self._m[g] = float(state["m"][g])
                self._sq[g] = float(state["sq"][g])
                self._w[g] = float(state["w"][g])
                self._t[g] = int(state["t"][g])


class AdvantageNormalizer:
    """Standardise each game's advantages with its own running moments.

    Per iteration::

        norm.update({g: (adv_g, learner_mask_g) for g in games_this_iter})
        adv_g = norm.normalize(g, adv_g)

    ``center=False`` only divides by the running std (keeps the sign of
    every advantage, e.g. before a top-|A| filter on sparse ±1 games).
    ``eps`` floors the std: a game whose advantages collapse (a solved or
    degenerate game, a perfect critic) must not have its noise amplified
    into the shared core's gradient.
    """

    def __init__(self, games: Sequence[str], decay: object = 0.99, *,
                 center: bool = True, eps: float = 1e-3, std_max: float = 1e6):
        self.moments = RunningMoments(games, decay, std_min=eps, std_max=std_max)
        self.center = center

    @property
    def games(self):
        return self.moments.games

    def update(self, batches: GameBatch, *, sync: bool = True, group=None) -> Dict[str, bool]:
        return self.moments.update(batches, sync=sync, group=group)

    def mean(self, game: str) -> float:
        return self.moments.mean(game)

    def std(self, game: str) -> float:
        return self.moments.std(game)

    def reset(self, game: str) -> None:
        """Restart ``game``'s moments: its advantages changed scale (e.g. another GAE
        lambda), and the running std would keep the old scale for ~1/(1 - decay) updates."""
        self.moments.reset(game)

    def normalize(self, game: str, adv: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(adv − mean) / std with ``game``'s running statistics; masked
        entries are zeroed."""
        out = adv - self.mean(game) if self.center else adv
        out = out / self.std(game)
        if mask is not None:
            out = torch.where(mask.bool(), out, torch.zeros((), dtype=out.dtype,
                                                            device=out.device))
        return out

    def state_dict(self) -> Dict[str, object]:
        return {"moments": self.moments.state_dict(), "center": self.center}

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True) -> None:
        self.moments.load_state_dict(state["moments"], strict=strict)
