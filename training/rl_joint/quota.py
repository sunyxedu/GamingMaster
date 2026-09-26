"""Learner-token quotas across games.

Mixed-game RL spends a fixed learner budget (tokens per iteration) on the
games.  The scheduler turns per-game SHARES into integer per-iteration token
QUOTAS:

  shares   static weights (configuration), optionally mixed with a bandit's
           weights (the hook for a learning-progress curriculum), projected
           onto the simplex with a per-game floor and cap — every game keeps
           at least its floor (no game is forgotten), none exceeds its cap
           (no game crowds the others out);
  quotas   shares × tokens, rounded by largest remainder (the integers sum
           to the budget exactly).  With debt carrying, a game that received
           fewer tokens than its share in recent iterations (its episodes
           were long and had not finished, its actors lagged) is topped up
           next time, bounded by the same floor / cap.

Everything is deterministic (Python floats, game order breaks ties) and
checkpointable; random game draws take a caller-owned ``torch.Generator``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

import torch


def _per_game(games: Sequence[str], value: object, name: str) -> Dict[str, float]:
    if isinstance(value, Mapping):
        missing = set(games) - set(value)
        if missing:
            raise KeyError(f"{name} missing for games {sorted(missing)}")
        return {g: float(value[g]) for g in games}
    return {g: float(value) for g in games}


def project_shares(weights: Mapping[str, float], floor: object = 0.0,
                   cap: object = 1.0) -> Dict[str, float]:
    """Shares ``p_g = clip(λ·w_g, floor_g, cap_g)`` with Σ p = 1.

    The scale λ is found by bisection (Σ p is continuous and non-decreasing
    in λ), so unclamped games keep their relative weights exactly.  All-zero
    weights mean equal weights.  If the positive-weight games cannot absorb
    the mass (all capped), the rest goes to the zero-weight games in
    proportion to their headroom.  Requires Σ floor ≤ 1 ≤ Σ cap.
    """
    games = list(weights)
    if not games:
        raise ValueError("no games to share between")
    w = {g: float(weights[g]) for g in games}
    if any(v < 0 or not math.isfinite(v) for v in w.values()):
        raise ValueError(f"share weights must be finite and non-negative, got {w}")
    lo, hi = _per_game(games, floor, "floor"), _per_game(games, cap, "cap")
    for g in games:
        if not 0.0 <= lo[g] <= hi[g] <= 1.0:
            raise ValueError(f"{g!r}: need 0 <= floor <= cap <= 1, got {lo[g]}, {hi[g]}")
    if sum(lo.values()) > 1.0 + 1e-9 or sum(hi.values()) < 1.0 - 1e-9:
        raise ValueError(f"infeasible bounds: sum(floor)={sum(lo.values())}, "
                         f"sum(cap)={sum(hi.values())}")
    if sum(w.values()) == 0.0:
        w = {g: 1.0 for g in games}

    def shares(lam: float) -> Dict[str, float]:
        return {g: min(max(lam * w[g], lo[g]), hi[g]) for g in games}

    lam_hi = 1.0 / sum(w.values())
    for _ in range(200):
        if sum(shares(lam_hi).values()) >= 1.0:
            break
        lam_hi *= 2.0
    p = shares(lam_hi)
    if sum(p.values()) >= 1.0:
        lam_lo = 0.0
        for _ in range(200):
            mid = 0.5 * (lam_lo + lam_hi)
            if sum(shares(mid).values()) < 1.0:
                lam_lo = mid
            else:
                lam_hi = mid
        p = shares(lam_hi)
    else:
        rest = 1.0 - sum(p.values())
        room = {g: hi[g] - p[g] for g in games}
        total_room = sum(room.values())
        if total_room > 0:      # 0 only when Σ cap = 1 − rounding: renormalise below
            p = {g: p[g] + rest * room[g] / total_room for g in games}
    s = sum(p.values())
    return {g: v / s for g, v in p.items()}


def largest_remainder(shares: Mapping[str, float], total: int) -> Dict[str, int]:
    """Integers proportional to ``shares`` summing to ``total`` (Hamilton's
    method; ties by game order)."""
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    s = sum(shares.values())
    if s <= 0:
        raise ValueError("shares must have a positive sum")
    exact = {g: total * v / s for g, v in shares.items()}
    out = {g: int(math.floor(x)) for g, x in exact.items()}
    left = total - sum(out.values())
    order = sorted(shares, key=lambda g: (-(exact[g] - out[g]), list(shares).index(g)))
    for g in order[:left]:
        out[g] += 1
    return out


@runtime_checkable
class ShareProvider(Protocol):
    """The bandit hook: adaptive per-game weights mixed into the shares."""

    def weights(self) -> Mapping[str, float]: ...

    def update(self, signals: Mapping[str, float]) -> None: ...

    def state_dict(self) -> Dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class LearningProgressBandit:
    """Weights ∝ absolute learning progress (Kanitscheider et al. 2021,
    arXiv 2106.14876; Portelas et al. 2019, arXiv 1910.07224): a fast and a
    slow bias-corrected EMA of each game's score (win rate vs a fixed
    opponent, Elo, negative loss, ...); progress = |fast − slow|, positive
    for improvement and for forgetting alike.  ``eps`` of the mass stays
    uniform; a game with fewer than ``warmup`` scores is optimistic (it
    gets the largest progress seen)."""

    def __init__(self, games: Sequence[str], *, fast: float = 0.5, slow: float = 0.9,
                 eps: float = 0.1, warmup: int = 2):
        if not 0.0 <= fast < slow < 1.0:
            raise ValueError(f"need 0 <= fast < slow < 1, got {fast}, {slow}")
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1], got {eps}")
        if warmup < 1:
            raise ValueError(f"warmup must be >= 1, got {warmup}")
        self.games = list(games)
        self.fast, self.slow, self.eps, self.warmup = float(fast), float(slow), float(eps), warmup
        self._f = {g: 0.0 for g in self.games}
        self._s = {g: 0.0 for g in self.games}
        self._n = {g: 0 for g in self.games}

    def update(self, signals: Mapping[str, float]) -> None:
        unknown = set(signals) - set(self.games)
        if unknown:
            raise KeyError(f"signals for unknown games {sorted(unknown)}")
        for g, x in signals.items():
            x = float(x)
            if not math.isfinite(x):
                raise ValueError(f"non-finite signal for {g!r}: {x}")
            self._f[g] = self.fast * self._f[g] + (1.0 - self.fast) * x
            self._s[g] = self.slow * self._s[g] + (1.0 - self.slow) * x
            self._n[g] += 1

    def progress(self) -> Dict[str, float]:
        lp = {}
        for g in self.games:
            n = self._n[g]
            if n >= self.warmup:
                lp[g] = abs(self._f[g] / (1.0 - self.fast ** n)
                            - self._s[g] / (1.0 - self.slow ** n))
        top = max(lp.values(), default=1.0)
        return {g: lp.get(g, top) for g in self.games}

    def weights(self) -> Dict[str, float]:
        lp = self.progress()
        total = sum(lp.values())
        G = len(self.games)
        if total <= 0:
            return {g: 1.0 / G for g in self.games}
        return {g: (1.0 - self.eps) * lp[g] / total + self.eps / G for g in self.games}

    def state_dict(self) -> Dict[str, object]:
        return {"fast_ema": dict(self._f), "slow_ema": dict(self._s), "n": dict(self._n)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        for g in self.games:
            if g in state["n"]:
                self._f[g] = float(state["fast_ema"][g])
                self._s[g] = float(state["slow_ema"][g])
                self._n[g] = int(state["n"][g])


class TokenQuotaScheduler:
    """Per-game learner-token quotas from static shares + floor / cap, with
    optional debt carrying and an optional bandit.

    Args:
        shares:      game → static weight (normalised internally)
        floor / cap: per-game share bounds (float or mapping)
        bandit:      a :class:`ShareProvider`; its weights are mixed in as
                     ``(1 − bandit_mix)·static + bandit_mix·bandit``
        carry_debt:  top up games that got fewer tokens than their share
        debt_decay:  per-:meth:`record` decay of the token ledger (the debt
                     looks back ~1/(1 − decay) iterations; 1.0 = forever)

    Per iteration::

        q = sched.quotas(tokens_per_iter)     # game → learner tokens
        ... roll out / subsample, train ...
        sched.record(tokens_trained)          # game → tokens actually used
    """

    def __init__(self, shares: Mapping[str, float], *, floor: object = 0.0, cap: object = 1.0,
                 bandit: Optional[ShareProvider] = None, bandit_mix: float = 1.0,
                 carry_debt: bool = True, debt_decay: float = 0.9):
        self.games = list(shares)
        if not self.games:
            raise ValueError("no games")
        self.static = {g: float(v) for g, v in shares.items()}
        if not sum(self.static.values()) > 0:
            raise ValueError(f"static shares need a positive sum, got {self.static}")
        self.floor = _per_game(self.games, floor, "floor")
        self.cap = _per_game(self.games, cap, "cap")
        project_shares(self.static, self.floor, self.cap)   # validates the bounds
        if bandit is not None and not isinstance(bandit, ShareProvider):
            raise TypeError(f"bandit must implement ShareProvider, got {type(bandit).__name__}")
        if not 0.0 <= bandit_mix <= 1.0:
            raise ValueError(f"bandit_mix must be in [0, 1], got {bandit_mix}")
        if not 0.0 <= debt_decay <= 1.0:
            raise ValueError(f"debt_decay must be in [0, 1], got {debt_decay}")
        self.bandit = bandit
        self.bandit_mix = float(bandit_mix)
        self.carry_debt = carry_debt
        self.debt_decay = float(debt_decay)
        self.ledger = {g: 0.0 for g in self.games}
        self.total_tokens = {g: 0 for g in self.games}
        self.iterations = 0

    # ---- shares ---------------------------------------------------------

    def shares(self) -> Dict[str, float]:
        """Target token fractions (static ⊕ bandit, floor / cap applied)."""
        s = sum(self.static.values())
        raw = {g: self.static[g] / s for g in self.games}
        if self.bandit is not None and self.bandit_mix > 0:
            bw = dict(self.bandit.weights())
            missing = set(self.games) - set(bw)
            if missing:
                raise KeyError(f"bandit weights missing games {sorted(missing)}")
            bs = sum(max(float(bw[g]), 0.0) for g in self.games)
            if bs > 0:
                raw = {g: (1.0 - self.bandit_mix) * raw[g]
                       + self.bandit_mix * max(float(bw[g]), 0.0) / bs for g in self.games}
        return project_shares(raw, self.floor, self.cap)

    def debts(self) -> Dict[str, float]:
        """Tokens each game is owed (negative: ahead) vs its share of the
        decayed ledger."""
        s = self.shares()
        total = sum(self.ledger.values())
        return {g: s[g] * total - self.ledger[g] for g in self.games}

    # ---- quotas ---------------------------------------------------------

    def quotas(self, total: int) -> Dict[str, int]:
        """Integer learner-token quotas for the next iteration, summing to
        ``total``."""
        s = self.shares()
        if self.carry_debt and sum(self.ledger.values()) > 0:
            owed = self.debts()
            # ideal allotment brings each game to its share after this
            # iteration; the deficits sum to ``total`` exactly
            want = {g: max(s[g] * total + owed[g], 0.0) for g in self.games}
            s = project_shares(want, self.floor, self.cap)
        return largest_remainder(s, int(total))

    def allocate(self, n: int) -> Dict[str, int]:
        """Split ``n`` units (actor slots, episodes) by the current shares,
        no debt."""
        return largest_remainder(self.shares(), int(n))

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> List[str]:
        """``n`` i.i.d. game draws with the current shares."""
        s = self.shares()
        p = torch.tensor([s[g] for g in self.games], dtype=torch.float64)
        idx = torch.multinomial(p, n, replacement=True, generator=generator)
        return [self.games[i] for i in idx.tolist()]

    def record(self, consumed: Mapping[str, int]) -> None:
        """Book the tokens each game actually contributed this iteration."""
        unknown = set(consumed) - set(self.games)
        if unknown:
            raise KeyError(f"tokens for unknown games {sorted(unknown)}")
        for g in self.games:
            n = int(consumed.get(g, 0))
            if n < 0:
                raise ValueError(f"negative token count for {g!r}")
            self.ledger[g] = self.debt_decay * self.ledger[g] + n
            self.total_tokens[g] += n
        self.iterations += 1

    # ---- persistence ----------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        return {"ledger": dict(self.ledger), "total_tokens": dict(self.total_tokens),
                "iterations": self.iterations,
                "bandit": self.bandit.state_dict() if self.bandit is not None else None}

    def load_state_dict(self, state: Mapping[str, object], strict: bool = True) -> None:
        saved = state["ledger"]
        missing, unexpected = set(self.games) - set(saved), set(saved) - set(self.games)
        if strict and (missing or unexpected):
            raise KeyError(f"quota state mismatch: missing {sorted(missing)}, "
                           f"unexpected {sorted(unexpected)}")
        for g in self.games:
            if g in saved:
                self.ledger[g] = float(saved[g])
                self.total_tokens[g] = int(state["total_tokens"][g])
        self.iterations = int(state["iterations"])
        if self.bandit is not None and state.get("bandit") is not None:
            self.bandit.load_state_dict(state["bandit"])
