"""Constraint solver harness for the offline Minesweeper adaptor.

The game is turn based, so the useful currency is correct transitions rather
than model calls.  This module keeps no speculative flags: every equation is
rebuilt from the returned observation, and only assignments proved by those
equations are treated as facts.  Small frontier components are enumerated and
coupled through the board-wide mine count.  When a guess is unavoidable, the
posterior mine probability is primary; ties favour cells with a large chance
of a zero expansion (corners/edges) or useful frontier exposure.
"""
from __future__ import annotations

import math
from collections import defaultdict


def neighbors(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    yield rr, cc


def _revealed(v):
    if isinstance(v, str) and v.startswith("revealed:"):
        try:
            return int(v.split(":", 1)[1])
        except Exception:
            return None
    return None


def _read(obs):
    grid = obs.get("cells") or []
    h = int(obs.get("height", len(grid)) or len(grid))
    w = int(obs.get("width", len(grid[0]) if grid else 0) or 0)
    hidden, revealed = set(), {}
    for r, row in enumerate(grid[:h]):
        for c, value in enumerate((row or [])[:w]):
            n = _revealed(value)
            if n is None:
                # We never create flags.  Treat a visible flag as hidden so a
                # stale/foreign annotation cannot manufacture a false fact.
                if value in ("unknown", "flagged"):
                    hidden.add((r, c))
            else:
                revealed[(r, c)] = n
    return hidden, revealed, (h, w)


def _equations(hidden, revealed, h, w):
    out = []
    for (r, c), n in revealed.items():
        s = frozenset(neighbors(r, c, h, w)) & hidden
        if s and 0 <= int(n) <= len(s):
            out.append((s, int(n)))
    return out


def _closure(base):
    """Apply residual and subset rules until no new exact fact appears."""
    safe, mines = set(), set()
    work = list(dict.fromkeys(base))
    # A derived equation is retained, but the cap prevents a dense board from
    # spending unbounded time producing redundant pairwise differences.
    for _ in range(24):
        changed = False
        norm = []
        seen = set()
        for s, need0 in work:
            left = set(s) - safe - mines
            need = int(need0) - len(set(s) & mines)
            if not left or need < 0 or need > len(left):
                continue
            item = (frozenset(left), need)
            if item not in seen:
                seen.add(item)
                norm.append(item)
            if need == 0:
                old = len(safe)
                safe.update(left)
                changed |= len(safe) != old
            elif need == len(left):
                old = len(mines)
                mines.update(left)
                changed |= len(mines) != old
        additions = []
        if len(norm) <= 500:
            for a, na in norm:
                for b, nb in norm:
                    if a == b or not a < b:
                        continue
                    d, nd = frozenset(b - a), int(nb) - int(na)
                    if d and 0 <= nd <= len(d):
                        item = (d, nd)
                        if item not in seen and item not in additions:
                            additions.append(item)
                            if len(additions) >= 5000:
                                break
                if len(additions) >= 5000:
                    break
        if additions:
            work.extend(additions)
            changed = True
        if not changed:
            return safe - mines, mines, norm
    return safe - mines, mines, work


def _components(equations):
    by = defaultdict(list)
    for s, n in equations:
        for p in s:
            by[p].append((s, n))
    seen, result = set(), []
    for start in by:
        if start in seen:
            continue
        stack, cells, eqs = [start], set(), set()
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            cells.add(p)
            for e in by[p]:
                eqs.add(e)
                for q in e[0]:
                    if q not in seen:
                        stack.append(q)
        result.append((cells, list(eqs)))
    return result


def _enumerate(cells, equations, cap=24, node_cap=350000):
    """Return counts by component mine total, or None when too large."""
    ordered = sorted(cells)
    if len(ordered) > cap:
        return None
    index = {p: i for i, p in enumerate(ordered)}
    local = [([index[p] for p in s], int(n)) for s, n in equations]
    incidence = [[] for _ in ordered]
    for ei, (inds, _) in enumerate(local):
        for i in inds:
            incidence[i].append(ei)
    order = sorted(range(len(ordered)), key=lambda i: -len(incidence[i]))
    assignment = [-1] * len(ordered)
    sums, used = [0] * len(local), [0] * len(local)
    dist = defaultdict(int)
    counts = defaultdict(lambda: [0] * len(ordered))
    nodes = 0

    def rec(pos):
        nonlocal nodes
        nodes += 1
        if nodes > node_cap:
            return False
        if pos == len(order):
            k = sum(assignment)
            dist[k] += 1
            row = counts[k]
            for i, v in enumerate(assignment):
                row[i] += v
            return True
        i = order[pos]
        for value in (0, 1):
            ok, touched = True, []
            for ei in incidence[i]:
                sums[ei] += value
                used[ei] += 1
                touched.append(ei)
                inds, need = local[ei]
                left = len(inds) - used[ei]
                if sums[ei] > need or sums[ei] + left < need:
                    ok = False
            if ok and not rec(pos + 1):
                for ei in touched:
                    sums[ei] -= value
                    used[ei] -= 1
                return False
            for ei in touched:
                sums[ei] -= value
                used[ei] -= 1
        assignment[i] = -1
        return True

    if not rec(0) or not dist:
        return None
    return ordered, dict(dist), dict(counts)


def _convolve(a, b):
    out = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] += va * vb
    return dict(out)


def _rates(hidden, equations, total_mines, known_mines):
    """Posterior rates, with exact weighting for tractable components."""
    infos, covered = [], set()
    for cells, eqs in _components(equations):
        got = _enumerate(cells, eqs)
        if got is not None:
            infos.append(got)
            covered.update(cells)

    rem = max(0, int(total_mines) - len(known_mines))
    # Cells in an oversized component are put in the residual pool.  Their
    # individual rates are improved below from the equations, while this pool
    # keeps the global mine count from being overconfident.
    active = hidden - known_mines
    residual = len(active - covered)
    rates = {}
    if infos:
        prefix = [{0: 1}]
        for _, dist, _ in infos:
            prefix.append(_convolve(prefix[-1], dist))
        suffix = [{} for _ in range(len(infos) + 1)]
        suffix[-1] = {0: 1}
        for i in range(len(infos) - 1, -1, -1):
            suffix[i] = _convolve(infos[i][1], suffix[i + 1])
        total_weight = 0
        for k, ways in prefix[-1].items():
            left = rem - k
            if 0 <= left <= residual:
                total_weight += ways * math.comb(residual, left)
        if total_weight:
            for i, (ordered, dist, counts) in enumerate(infos):
                other = _convolve(prefix[i], suffix[i + 1])
                for j, p in enumerate(ordered):
                    numerator = 0
                    for k, ways in dist.items():
                        hit = counts[k][j]
                        if not hit:
                            continue
                        for ko, other_ways in other.items():
                            left = rem - k - ko
                            if 0 <= left <= residual:
                                numerator += hit * other_ways * math.comb(residual, left)
                    rates[p] = numerator / float(total_weight)
            if residual:
                expected = 0
                for k, ways in prefix[-1].items():
                    left = rem - k
                    if 0 <= left <= residual:
                        expected += ways * math.comb(residual, left) * left
                q = expected / float(total_weight * residual)
                for p in active - covered:
                    rates[p] = q
        else:
            infos = []
    if not infos:
        q = rem / float(max(1, len(hidden)))
        rates = {p: q for p in active}

    # Oversized components have no exact rate.  Use the tightest local
    # residual ratio involving each cell as a stable approximation.
    for s, need in equations:
        left = set(s) & hidden
        if not left:
            continue
        ratio = max(0.0, min(1.0, float(need) / len(left)))
        for p in left:
            if p not in covered:
                rates[p] = min(rates.get(p, 1.0), ratio)
    return rates


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx

    def run_episode(self, game) -> None:
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        first = True
        while not ctx.done:
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    return
            if obs.get("status") != "playing" or obs.get("won"):
                return
            hidden, revealed, (h, w) = _read(obs)
            if not hidden or h <= 0 or w <= 0:
                return
            if first and int(obs.get("safe_revealed", 0) or 0) == 0:
                first = False
                action = {"type": "reveal", "row": h // 2, "col": w // 2}
            else:
                first = False
                equations = _equations(hidden, revealed, h, w)
                safe, mines, reduced = _closure(equations)
                safe &= hidden - mines
                if safe:
                    # Safe cells bordering more hidden cells tend to expose
                    # more information; this also makes flood-fill follow-up
                    # actions deterministic.
                    def safe_key(p):
                        degree = sum(q in hidden for q in neighbors(p[0], p[1], h, w))
                        exposed = sum(q in revealed for q in neighbors(p[0], p[1], h, w))
                        return (-degree, -exposed, p[0], p[1])
                    r, c = min(safe, key=safe_key)
                else:
                    rates = _rates(hidden, reduced, int(obs.get("mines", 0) or 0), mines)
                    candidates = hidden - mines
                    if not candidates:
                        return
                    def key(p):
                        r0, c0 = p
                        q = float(rates.get(p, 1.0))
                        # At equal risk, a boundary square has fewer adjacent
                        # mines and a higher chance of a zero flood-fill.  A
                        # frontier square gets a small information bonus.
                        n_unknown = sum(qp in hidden for qp in neighbors(r0, c0, h, w))
                        exposed = sum(qp in revealed for qp in neighbors(r0, c0, h, w))
                        border = min(r0, c0, h - 1 - r0, w - 1 - c0)
                        return (round(q, 12), n_unknown if exposed == 0 else -exposed,
                                border, -exposed, r0, c0)
                    r, c = min(candidates, key=key)
                action = {"type": "reveal", "row": int(r), "col": int(c)}
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except Exception:
                # A reveal may legally end the game by hitting a mine; action
                # and observation failures are contained for pre-flight too.
                try:
                    obs = game.observe()
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
