"""Constraint-first Minesweeper harness.

The adaptor exposes enough of the visible board to solve the usual local
equations exactly.  This harness keeps its own (conservative) mine/safe sets,
uses subset inference and small component enumeration, and only guesses when
the visible information leaves no safe move.  It deliberately makes no slow
System-2 calls: on this turn game a local proof is both faster and more
reliable than asking a model to repeat arithmetic.
"""
from __future__ import annotations

import math

def _neigh(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield (rr, cc)


def _cell_kind(x):
    if isinstance(x, str):
        if x.startswith("revealed:"):
            try:
                return "r", int(x.split(":", 1)[1])
            except Exception:
                return "r", None
        if x == "flagged":
            return "f", None
    return "u", None


def _choose_weighted(cells, equations, total_mines, known_mines, nonfrontier_count):
    """Return (certain_safe, certain_mines, risk map) for a frontier.

    Enumeration is intentionally capped.  Basic and subset rules still work
    on large boards; when a component is too large the caller falls back to a
    minimum-risk guess over the cells it can score.
    """
    if not cells:
        return set(), set(), {}
    cells = list(cells)
    if len(cells) > 22:
        return set(), set(), {}
    idx = {p: i for i, p in enumerate(cells)}
    eqs = [(tuple(idx[p] for p in s if p in idx), int(k))
           for s, k in equations if s]
    eqs = [(a, k) for a, k in eqs if a]
    if not eqs:
        return set(), set(), {}
    # Incidence order greatly prunes the backtracking tree.
    order = sorted(range(len(cells)),
                   key=lambda i: -sum(i in a for a, _ in eqs))
    assign = [-1] * len(cells)
    valid = []

    def rec(pos):
        # Check every equation whose assigned variables can already decide it.
        for inds, need in eqs:
            s = sum(assign[i] for i in inds if assign[i] != -1)
            rem = sum(assign[i] == -1 for i in inds)
            if s > need or s + rem < need:
                return
        if pos == len(order):
            valid.append(tuple(assign))
            return
        i = order[pos]
        for v in (0, 1):
            assign[i] = v
            rec(pos + 1)
        assign[i] = -1

    rec(0)
    if not valid:
        return set(), set(), {}

    # Couple each frontier assignment to the global mine count.  Assignments
    # with k mines have C(nonfrontier, remaining-k) completions elsewhere.
    # This is the useful global correction when a frontier is small and the
    # board still has many unconstrained cells.
    rem_total = int(total_mines) - int(known_mines)
    weights = []
    for a in valid:
        k = sum(a)
        left = rem_total - k
        weights.append(math.comb(nonfrontier_count, left)
                       if 0 <= left <= nonfrontier_count else 0)
    if not any(weights):
        weights = [1] * len(valid)
    risk = {p: 0.0 for p in cells}
    for a, wt in zip(valid, weights):
        for i, p in enumerate(cells):
            risk[p] += wt * a[i]
    den = float(sum(weights))
    for p in risk:
        risk[p] /= den
    certain_mines = {p for p, q in risk.items() if q >= 1.0 - 1e-12}
    certain_safe = {p for p, q in risk.items() if q <= 1e-12}
    return certain_safe, certain_mines, risk


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def _solve(self, obs):
        cells = obs.get("cells") or []
        h, w = int(obs.get("height", len(cells))), int(obs.get("width", 0))
        unknown, flagged, revealed = set(), set(), {}
        for r, row in enumerate(cells):
            for c, val in enumerate(row):
                k, n = _cell_kind(val)
                if k == "u":
                    unknown.add((r, c))
                elif k == "f":
                    flagged.add((r, c))
                else:
                    revealed[(r, c)] = n

        # Flags are annotations and may be wrong.  Treat them as constraints
        # only after the board itself proves them; this avoids the leader's
        # failure mode where a speculative flag poisoned every later chord.
        mines = set()
        safe = set()
        equations = []
        for (r, c), n in revealed.items():
            if n is None:
                continue
            ns = set(_neigh(r, c, h, w))
            u = ns & unknown
            if u:
                equations.append((u, int(n)))

        # Repeated residual-zero/full rules plus subset differences.
        changed = True
        while changed:
            changed = False
            reduced = []
            for s, k in equations:
                s2 = set(s) - mines - safe
                k2 = k - len(set(s) & mines)
                if k2 < 0 or k2 > len(s2):
                    continue
                if not s2:
                    continue
                reduced.append((frozenset(s2), k2))
                if k2 == 0:
                    before = len(safe); safe |= s2
                    changed |= len(safe) != before
                elif k2 == len(s2):
                    before = len(mines); mines |= s2
                    changed |= len(mines) != before
            equations = reduced
            # Subset subtraction supplies the common 1-2/edge deductions.
            additions = []
            # Pairwise subset inference is powerful on a frontier but can be
            # quadratic after a large flood-fill.  Basic residual rules remain
            # exact; cap the derived-equation work to keep every turn bounded.
            if len(equations) <= 400:
                for a, ka in equations:
                    for b, kb in equations:
                        if a != b and a < b:
                            d, kd = frozenset(b - a), kb - ka
                            if d and 0 <= kd <= len(d):
                                additions.append((d, kd))
                                if len(additions) >= 3000:
                                    break
                    if len(additions) >= 3000:
                        break
            for e in additions:
                if e not in equations:
                    equations.append(e); changed = True

        frontier = set().union(*(set(s) for s, _ in equations)) if equations else set()
        esafe, emines, risk = _choose_weighted(
            frontier, equations, int(obs.get("mines", 0)), len(mines),
            max(0, len(unknown - flagged) - len(frontier)))
        safe |= esafe
        mines |= emines
        # Any explicitly flagged cell is excluded from guesses; it is merely
        # an annotation from the game and we never create such annotations.
        candidates = unknown - flagged - mines
        return safe & candidates, mines, risk, candidates, (h, w)

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        first = True
        stalled = 0
        while not ctx.done:
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    return
            if obs.get("status") != "playing" or obs.get("won"):
                return
            cells = obs.get("cells") or []
            unknown_count = sum(1 for row in cells for x in row if x == "unknown")
            if unknown_count == 0:
                return
            if first and obs.get("safe_revealed", 0) == 0:
                h, w = int(obs.get("height", len(cells))), int(obs.get("width", 0))
                action = {"type": "reveal", "row": h // 2, "col": w // 2}
                first = False
            else:
                first = False
                safe, mines, risk, candidates, (h, w) = self._solve(obs)
                # Commit every currently proved safe cell.  Recompute after
                # each transition because a zero flood-fill can change many
                # equations at once.
                if safe:
                    r, c = sorted(safe)[0]
                    action = {"type": "reveal", "row": r, "col": c}
                else:
                    if not candidates:
                        return
                    # Lowest posterior mine probability; center-distance and
                    # degree are deterministic tie-breakers that favour a
                    # useful opening when probabilities are equal.
                    def key(p):
                        r, c = p
                        q = risk.get(p, float(obs.get("remaining_mines", obs.get("mines", 0))) /
                                  max(1, len(candidates)))
                        degree = sum(1 for rr, cc in _neigh(r, c, h, w)
                                     if (rr, cc) in candidates)
                        return (q, -degree, abs(r-h/2)+abs(c-w/2), r, c)
                    r, c = min(candidates, key=key)
                    action = {"type": "reveal", "row": r, "col": c}
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
                stalled = 0
            except Exception:
                # An illegal/repeated action should not kill an episode.  The
                # next observation lets the solver discard it and continue.
                stalled += 1
                try:
                    obs = game.observe()
                except Exception:
                    return
                if stalled > 8:
                    return


def build(ctx):
    return Harness(ctx)
