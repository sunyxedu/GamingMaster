"""Per-game adaptive coefficients: entropy toward a target entropy, KL beta
toward a KL budget.

A single entropy weight / KL weight across games is wrong twice over: the
games' action spaces differ (log |A| from ~2 to ~9 nats), so one target
entropy fits none of them, and a shared beta lets the game that drifts
fastest from the Stage-1 reference dictate the anchoring of all the others.
Each game gets its own controller state; the trainer feeds each iteration's
measured per-game value (all-reduced, so every rank runs the same update)
and reads the coefficient for the next iteration.

Both controllers are deterministic functions of their inputs and state
(Python floats), with ``state_dict`` / ``load_state_dict`` for resume.  The
targets, rates and bounds are configuration (constructor arguments) and are
NOT restored from a checkpoint, so a resumed run can retune them.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence


def _per_game(games: Sequence[str], value: object, name: str) -> Dict[str, float]:
    if isinstance(value, Mapping):
        missing = set(games) - set(value)
        if missing:
            raise KeyError(f"{name} missing for games {sorted(missing)}")
        return {g: float(value[g]) for g in games}
    return {g: float(value) for g in games}


class _PerGameCoef:
    """Shared bookkeeping: per-game values, bounds, persistence."""

    def __init__(self, games: Sequence[str], init: object, lo: object, hi: object):
        if not games or len(set(games)) != len(games):
            raise ValueError(f"games must be a non-empty list of distinct names, got {games!r}")
        self.games = list(games)
        self.lo = _per_game(self.games, lo, "min")
        self.hi = _per_game(self.games, hi, "max")
        for g in self.games:
            if self.lo[g] > self.hi[g]:
                raise ValueError(f"{g!r}: min {self.lo[g]} > max {self.hi[g]}")
        self.values = {g: self._clamp(g, v) for g, v in
                       _per_game(self.games, init, "init").items()}

    def _clamp(self, game: str, v: float) -> float:
        return min(max(v, self.lo[game]), self.hi[game])

    def __getitem__(self, game: str) -> float:
        return self.values[game]

    def as_dict(self) -> Dict[str, float]:
        return dict(self.values)

    def _check_measured(self, measured: Mapping[str, float]) -> None:
        unknown = set(measured) - set(self.games)
        if unknown:
            raise KeyError(f"measurements for unknown games {sorted(unknown)}")
        for g, v in measured.items():
            if not math.isfinite(float(v)):
                raise ValueError(f"non-finite measurement for {g!r}: {v}")

    def state_dict(self) -> Dict[str, object]:
        return {"values": dict(self.values)}

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True) -> None:
        saved = state["values"]
        missing, unexpected = set(self.games) - set(saved), set(saved) - set(self.games)
        if strict and (missing or unexpected):
            raise KeyError(f"{type(self).__name__} state mismatch: missing "
                           f"{sorted(missing)}, unexpected {sorted(unexpected)}")
        for g in self.games:
            if g in saved:
                self.values[g] = self._clamp(g, float(saved[g]))


class AdaptiveEntropyCoef(_PerGameCoef):
    """Per-game entropy coefficient α_g driven toward a target entropy H*_g
    (the SAC temperature rule, Haarnoja et al. 2018, arXiv 1812.05905):

        mode="log":     log α ← log α + lr · (H* − H)      (α > 0)
        mode="linear":  α ← α + lr · (H* − H)              (α may go < 0)

    Entropy below target raises α (more exploration bonus), above lowers it.
    The linear mode with a negative ``min_coef`` also covers the generals
    trainer's entropy CEILING: past the target the coefficient turns into an
    active suppressor instead of stopping at zero.  Errors within
    ``tolerance`` nats leave α unchanged.
    """

    def __init__(self, games: Sequence[str], target: object, *, init: object = 0.01,
                 lr: float = 0.05, min_coef: object = 1e-5, max_coef: object = 0.1,
                 mode: str = "log", tolerance: float = 0.0):
        if mode not in ("log", "linear"):
            raise ValueError(f"mode must be 'log' or 'linear', got {mode!r}")
        super().__init__(games, init, min_coef, max_coef)
        if mode == "log" and any(v <= 0 for v in self.lo.values()):
            raise ValueError("mode='log' needs a positive min_coef")
        self.target = _per_game(self.games, target, "target")
        self.lr = float(lr)
        self.mode = mode
        self.tolerance = float(tolerance)

    def update(self, entropy: Mapping[str, float]) -> Dict[str, float]:
        """Step every game in ``entropy`` (mean policy entropy of its learner
        tokens this iteration); others are untouched.  Returns all coefs."""
        self._check_measured(entropy)
        for g, h in entropy.items():
            err = self.target[g] - float(h)
            if abs(err) <= self.tolerance:
                continue
            a = self.values[g]
            a = math.exp(math.log(a) + self.lr * err) if self.mode == "log" else a + self.lr * err
            self.values[g] = self._clamp(g, a)
        return self.as_dict()


class AdaptiveKLBeta(_PerGameCoef):
    """Per-game KL penalty coefficient β_g driven toward a KL budget.

      mode="proportional" (Ziegler et al. 2019, arXiv 1909.08593):
            e = clip(KL / budget − 1, ±err_clip),   β ← β · (1 + rate · e)
      mode="step" (Schulman et al. 2017, PPO adaptive-KL):
            KL > budget·tol → β ← β·factor;  KL < budget/tol → β ← β/factor

    A game whose measured KL to the frozen Stage-1 reference exceeds its
    budget gets anchored harder, one well inside its budget is let go.  The
    defaults move β by at most ±10% per update (one update per iteration;
    Ziegler et al. update per minibatch with rate 0.1).
    """

    def __init__(self, games: Sequence[str], budget: object, *, init: object = 0.1,
                 rate: float = 0.5, err_clip: float = 0.2, min_beta: object = 1e-4,
                 max_beta: object = 10.0, mode: str = "proportional",
                 step_tol: float = 1.5, step_factor: float = 2.0):
        if mode not in ("proportional", "step"):
            raise ValueError(f"mode must be 'proportional' or 'step', got {mode!r}")
        super().__init__(games, init, min_beta, max_beta)
        self.budget = _per_game(self.games, budget, "budget")
        if any(b <= 0 for b in self.budget.values()):
            raise ValueError(f"KL budgets must be positive, got {self.budget}")
        if any(v <= 0 for v in self.lo.values()):
            raise ValueError("multiplicative beta updates need a positive min_beta")
        if step_tol <= 1.0 or step_factor <= 1.0:
            raise ValueError("step_tol and step_factor must exceed 1")
        self.rate = float(rate)
        self.err_clip = float(err_clip)
        self.mode = mode
        self.step_tol = float(step_tol)
        self.step_factor = float(step_factor)

    def update(self, kl: Mapping[str, float]) -> Dict[str, float]:
        """Step every game in ``kl`` (its measured mean KL to the reference
        this iteration); others are untouched.  Returns all betas."""
        self._check_measured(kl)
        for g, k in kl.items():
            k, b, budget = float(k), self.values[g], self.budget[g]
            if self.mode == "proportional":
                e = min(max(k / budget - 1.0, -self.err_clip), self.err_clip)
                b = b * (1.0 + self.rate * e)
            elif k > budget * self.step_tol:
                b = b * self.step_factor
            elif k < budget / self.step_tol:
                b = b / self.step_factor
            self.values[g] = self._clamp(g, b)
        return self.as_dict()

