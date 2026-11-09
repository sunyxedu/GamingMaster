"""Constraint solver with an expansion-aware opening policy.

All decisions come from the visible board.  The first click is put in a
corner, where the guaranteed-safe opening has only three neighbours and is
therefore more likely to be a zero flood fill than a central click.  After
that, exact residual/subset deductions take priority; small frontier
components are enumerated and coupled through the board-wide mine total.
"""
from __future__ import annotations

import math
from collections import defaultdict


def neighbours(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    yield rr, cc


def parse_cell(value):
    if isinstance(value, str):
        if value in ("unknown", "flagged"):
            return "hidden", None
        if value.startswith("revealed:"):
            try:
                return "revealed", int(value.split(":", 1)[1])
            except Exception:
                return "revealed", None
        if value == "mine":
            return "mine", None
    return "hidden", None


def board_view(obs):
    grid = obs.get("cells") if isinstance(obs, dict) else None
    if not isinstance(grid, list):
        grid = []
    h = int(obs.get("height", len(grid)) or len(grid)) if isinstance(obs, dict) else len(grid)
    w = int(obs.get("width", 0) or (len(grid[0]) if grid and isinstance(grid[0], list) else 0)) if isinstance(obs, dict) else 0
    h = max(0, min(h, len(grid)))
    hidden, revealed, shape = set(), {}, (h, w)
    for r in range(h):
        row = grid[r] if isinstance(grid[r], list) else []
        for c in range(min(w, len(row))):
            kind, number = parse_cell(row[c])
            if kind == "revealed" and number is not None:
                revealed[(r, c)] = number
            elif kind == "hidden":
                hidden.add((r, c))
    return grid, hidden, revealed, shape


def equations(obs, known):
    grid, hidden, revealed, (h, w) = board_view(obs)
    known.intersection_update(hidden)
    out = []
    for (r, c), number in revealed.items():
        around = set(neighbours(r, c, h, w))
        left = around & hidden
        residual = int(number) - len(around & known)
        left.difference_update(known)
        if left and 0 <= residual <= len(left):
            out.append((frozenset(left), residual))
    return grid, hidden, revealed, (h, w), list(dict.fromkeys(out))


def closure(base):
    """Apply residual zero/full and bounded subset inference to a fixed point."""
    safe, mines = set(), set()
    work = list(dict.fromkeys(base))
    for _ in range(16):
        changed = False
        normalized = []
        for cells, need0 in work:
            original = set(cells)
            left = original - safe - mines
            need = int(need0) - len(original & mines)
            if not left or need < 0 or need > len(left):
                continue
            item = (frozenset(left), need)
            if item not in normalized:
                normalized.append(item)
            if need == 0:
                before = len(safe)
                safe.update(left)
                changed |= before != len(safe)
            elif need == len(left):
                before = len(mines)
                mines.update(left)
                changed |= before != len(mines)

        # Subset differences expose the common 1-2 and edge patterns.  Keep
        # the closure bounded so a large flood fill cannot create quadratic
        # work forever.
        derived = []
        if len(normalized) <= 500:
            for a, na in normalized:
                for b, nb in normalized:
                    if a == b or not a < b:
                        continue
                    d, nd = frozenset(b - a), int(nb) - int(na)
                    if d and 0 <= nd <= len(d):
                        item = (d, nd)
                        if item not in work and item not in derived:
                            derived.append(item)
                            if len(derived) >= 4000:
                                break
                if len(derived) >= 4000:
                    break
        if derived:
            work.extend(derived)
            changed = True
        if not changed:
            break
    safe.difference_update(mines)
    return safe, mines, normalized if 'normalized' in locals() else []


def components(eqs):
    by_cell = defaultdict(list)
    for cells, need in eqs:
        for p in cells:
            by_cell[p].append((cells, need))
    seen, result = set(), []
    for start in by_cell:
        if start in seen:
            continue
        stack, cells, got = [start], set(), set()
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            cells.add(p)
            for item in by_cell.get(p, ()):
                got.add(item)
                stack.extend(q for q in item[0] if q not in seen)
        result.append((cells, list(got)))
    return result


def enumerate_component(cells, eqs, cap=22):
    ordered = sorted(cells)
    if len(ordered) > cap:
        return None
    index = {p: i for i, p in enumerate(ordered)}
    local = [([index[p] for p in s], int(n)) for s, n in eqs]
    incidence = [[] for _ in ordered]
    for ei, (inds, _) in enumerate(local):
        for i in inds:
            incidence[i].append(ei)
    order = sorted(range(len(ordered)), key=lambda i: -len(incidence[i]))
    assignment = [-1] * len(ordered)
    sums = [0] * len(local)
    used = [0] * len(local)
    distribution = defaultdict(int)
    counts = defaultdict(lambda: [0] * len(ordered))

    def visit(pos):
        if pos == len(order):
            k = sum(assignment)
            distribution[k] += 1
            row = counts[k]
            for i, value in enumerate(assignment):
                row[i] += value
            return
        i = order[pos]
        for value in (0, 1):
            touched = []
            valid = True
            for ei in incidence[i]:
                sums[ei] += value
                used[ei] += 1
                touched.append(ei)
                inds, need = local[ei]
                remaining = len(inds) - used[ei]
                if sums[ei] > need or sums[ei] + remaining < need:
                    valid = False
            if valid:
                assignment[i] = value
                visit(pos + 1)
            for ei in touched:
                sums[ei] -= value
                used[ei] -= 1
        assignment[i] = -1

    visit(0)
    if not distribution:
        return None
    return ordered, dict(distribution), dict(counts)


def combine(a, b):
    out = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] += va * vb
    return dict(out)


def global_rates(hidden, eqs, total_mines, known):
    infos, covered = [], set()
    for cells, ceqs in components(eqs):
        item = enumerate_component(cells, ceqs)
        if item is not None:
            infos.append(item)
            covered.update(cells)
    unknown = set(hidden) - set(known)
    remaining = max(0, int(total_mines) - len(known))
    unconstrained = len(unknown - covered)
    if not infos:
        q = remaining / max(1, len(unknown))
        return {p: q for p in unknown}

    distributions = [item[1] for item in infos]
    prefix = [{0: 1}]
    for dist in distributions:
        prefix.append(combine(prefix[-1], dist))
    suffix = [{} for _ in range(len(distributions) + 1)]
    suffix[-1] = {0: 1}
    for i in range(len(distributions) - 1, -1, -1):
        suffix[i] = combine(distributions[i], suffix[i + 1])
    total_weight = 0
    for k, count in prefix[-1].items():
        left = remaining - k
        if 0 <= left <= unconstrained:
            total_weight += count * math.comb(unconstrained, left)

    rates = {}
    if not total_weight:
        # In a contradictory/overconstrained view, retain local frequencies
        # rather than manufacturing a zero-risk move.
        for ordered, dist, counts in infos:
            denominator = max(1, sum(dist.values()))
            for j, p in enumerate(ordered):
                rates[p] = sum(row[j] for row in counts.values()) / denominator
        q = remaining / max(1, len(unknown))
        rates.update({p: q for p in unknown if p not in rates})
        return rates

    for i, (ordered, dist, counts) in enumerate(infos):
        others = combine(prefix[i], suffix[i + 1])
        for j, p in enumerate(ordered):
            numerator = 0
            for k, count in dist.items():
                mine_count = counts[k][j]
                if not mine_count:
                    continue
                for other_k, other_count in others.items():
                    left = remaining - k - other_k
                    if 0 <= left <= unconstrained:
                        numerator += mine_count * other_count * math.comb(unconstrained, left)
            rates[p] = numerator / total_weight
    if unconstrained:
        numerator = 0
        for k, count in prefix[-1].items():
            left = remaining - k
            if 0 <= left <= unconstrained:
                numerator += count * math.comb(unconstrained, left) * left
        q = numerator / (total_weight * unconstrained)
        for p in unknown - covered:
            rates[p] = q
    return rates


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.known = set()

    def run_episode(self, game):
        ctx = self.ctx
        self.known.clear()
        try:
            obs = game.observe()
        except Exception:
            return
        first = True
        while not getattr(ctx, "done", False):
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    return
            if obs.get("status") != "playing" or obs.get("won"):
                return
            try:
                grid, hidden, revealed, (h, w), base = equations(obs, self.known)
            except Exception:
                return
            if not hidden or h <= 0 or w <= 0:
                return

            if first and int(obs.get("safe_revealed", 0) or 0) == 0:
                # A corner has the smallest neighbourhood and hence the
                # greatest chance of opening a zero region on the safe first
                # click.  The choice is deterministic across episodes.
                action = {"type": "reveal", "row": 0, "col": 0}
                first = False
            else:
                first = False
                safe, mines, reduced = closure(base)
                self.known.update(mines)
                safe.intersection_update(hidden - self.known)
                if safe:
                    # Prefer a forced safe square bordering many unrevealed
                    # cells, since it is most likely to expose new equations.
                    def safe_key(p):
                        degree = sum(q in hidden for q in neighbours(p[0], p[1], h, w))
                        exposed = sum(q in revealed for q in neighbours(p[0], p[1], h, w))
                        return (-degree, -exposed, p[0], p[1])
                    r, c = min(safe, key=safe_key)
                else:
                    rates = global_rates(hidden, reduced, int(obs.get("mines", 0) or 0), self.known)
                    candidates = [p for p in hidden if p not in self.known and
                                  parse_cell(grid[p[0]][p[1]])[0] == "hidden"]
                    if not candidates:
                        return

                    # If risks tie, favour a low-degree square.  Such a
                    # square has fewer adjacent mines and a higher chance of
                    # being zero, which can reveal a whole region.
                    def guess_key(p):
                        risk = rates.get(p, 1.0)
                        degree = sum(1 for _ in neighbours(p[0], p[1], h, w))
                        frontier_touch = sum(q in revealed for q in neighbours(p[0], p[1], h, w))
                        return (risk, degree, -frontier_touch,
                                abs(p[0] - h / 2) + abs(p[1] - w / 2), p[0], p[1])
                    r, c = min(candidates, key=guess_key)
                action = {"type": "reveal", "row": int(r), "col": int(c)}

            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
            except Exception:
                try:
                    obs = game.observe()
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
