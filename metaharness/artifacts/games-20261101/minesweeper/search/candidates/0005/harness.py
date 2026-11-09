"""Minesweeper solver with posterior predictive reveal planning.

Exact constraints determine safe moves. When a guess is needed, sample
complete layouts consistent with the visible board and compare the expected
safe progress from flood fill and the next clue's deductions. Samples rank
ambiguous moves only; no sampled event becomes a safety proof.
"""
from __future__ import annotations

import math
import random
import time
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


def closure(base, stop=None):
    """Apply residual zero/full and bounded subset inference to a fixed point."""
    safe, mines = set(), set()
    work = list(dict.fromkeys(base))
    for _ in range(16):
        if stop and stop():
            raise SearchLimit()
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


class SearchLimit(Exception):
    """Search was incomplete; partial frequencies must never become facts."""


def enumerate_component(cells, eqs, cap=22, models=False, stop=None):
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
    assignments = defaultdict(list)
    nodes = 0
    deadline = time.monotonic() + 2.0

    def visit(pos):
        nonlocal nodes
        nodes += 1
        if nodes > 250000 or (nodes % 256 == 0 and
                (time.monotonic() > deadline or (stop and stop()))):
            raise SearchLimit()

        if pos == len(order):
            k = sum(assignment)
            distribution[k] += 1
            row = counts[k]
            if models:
                assignments[k].append(frozenset(ordered[i] for i, v in enumerate(assignment) if v))
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

    try:
        visit(0)
    except SearchLimit:
        return None
    if not distribution:
        return None
    result = ordered, dict(distribution), dict(counts)
    return result + (dict(assignments),) if models else result


def combine(a, b):
    out = defaultdict(int)
    for ka, va in a.items():
        for kb, vb in b.items():
            out[ka + kb] += va * vb
    return dict(out)


def global_rates(hidden, eqs, total_mines, known, stop=None, precomputed=None):
    infos, covered = [], set()
    if precomputed is not None:
        infos = [item[:3] for item in precomputed]
        covered = {p for item in infos for p in item[0]}
    else:
        for cells, ceqs in components(eqs):
            if stop and stop():
                raise SearchLimit()
            item = enumerate_component(cells, ceqs, stop=stop)
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


def sample_worlds(hidden, known, eqs, total, stop, count=384):
    """Uniform posterior completions; return None if any component is omitted.

    Conditional on the visible first-click-safe observation, layouts with the
    right total and all clues have equal prior weight. Draw component mine
    totals by completion count, then assignments and residual cells uniformly.
    """
    infos, covered = [], set()
    for cs, es in components(eqs):
        if stop():
            raise SearchLimit()
        got = enumerate_component(cs, es, cap=28, models=True, stop=stop)
        if got is None:
            return None
        infos.append(got)
        covered.update(cs)
    pool = sorted(hidden - known - covered)
    remaining = total - len(known)
    if remaining < 0:
        return None
    suffix = [None] * (len(infos) + 1)
    suffix[-1] = {k: math.comb(len(pool), k)
                  for k in range(min(len(pool), remaining) + 1)}
    for i in range(len(infos) - 1, -1, -1):
        suffix[i] = {k: v for k, v in combine(infos[i][1], suffix[i + 1]).items()
                     if k <= remaining}
    if not suffix[0].get(remaining, 0):
        return None
    # Fixed sampling randomness is independent of the environment's seed.
    rng = random.Random(271828)
    worlds = []
    for sample_i in range(count):
        if sample_i % 32 == 0 and stop():
            raise SearchLimit()
        left = remaining
        world = set(known)
        for i, (_, dist, _, models) in enumerate(infos):
            choices = [(k, ways * suffix[i + 1].get(left - k, 0))
                       for k, ways in sorted(dist.items())]
            ticket = rng.randrange(sum(weight for _, weight in choices))
            for k, weight in choices:
                ticket -= weight
                if ticket < 0:
                    world.update(rng.choice(models[k]))
                    left -= k
                    break
        world.update(rng.sample(pool, left))
        worlds.append(world)
    return worlds, infos


def plan_reveal(hidden, revealed, known, rates, total, h, w, baseline, ctx):
    """One observation lookahead: expected safe progress per terminal hazard.

    The sample board is used to predict only observable outcomes and zero
    floods. Extra safes come from deterministic deduction on that hypothetical
    observation; knowledge of sampled mines is never used as a proof.
    """
    started = time.monotonic()
    wall_left = float(ctx.budget.remaining().get("wall_s", 60))
    deadline = started + min(12.0, max(0.0, wall_left - 3.0))
    def stop():
        return ctx.done or time.monotonic() >= deadline

    active = hidden - known
    # Rebuild residuals from current facts; closure may have added mines
    # after an earlier equation in its final pass was normalized.
    eqs = []
    for p, n in revealed.items():
        around = set(neighbours(*p, h, w))
        s = frozenset(around & active)
        need = n - len(around & known)
        if not 0 <= need <= len(s):
            return None, {"reason": "inconsistent_clues"}
        if s:
            eqs.append((s, need))
    eqs = list(dict.fromkeys(eqs))
    sampled = sample_worlds(hidden, known, eqs, total, stop)
    if sampled is None:
        return None, {"reason": "incomplete_posterior", "fallback": list(baseline)}
    worlds, infos = sampled
    rates = global_rates(hidden, eqs, total, known, stop=stop, precomputed=infos)
    adjacency = {p: tuple(neighbours(*p, h, w))
                 for p in ((r, c) for r in range(h) for c in range(w))}
    frontier = {p for s, _ in eqs for p in s}
    minimum = min(rates.get(p, 1.0) for p in active)
    # Always include the safest move. Other candidates remain within a small
    # safety band; this prevents short-horizon gain from excusing large risks.
    eligible = {p for p in active if rates.get(p, 1.0) <= minimum + 0.06}
    # A newly completed large component can correct the fallback's rates.
    # Never exempt its old baseline from the exact safety band.
    if baseline not in eligible:
        baseline = min(eligible, key=lambda p: (rates.get(p, 1.0), p))
    def contact(p):
        return sum(q in revealed for q in adjacency[p])
    def risk_key(p):
        return (round(rates.get(p, 1.0), 12), -contact(p), len(adjacency[p]), p)
    shortlist = set(sorted(eligible & frontier, key=risk_key)[:8])
    pool = eligible - frontier
    # Include different geometric opportunities, not just a row-major strip.
    # Nearest points to corner, edge-midpoint and center anchors cover the
    # unconstrained region without evaluating hundreds of identical risks.
    anchors = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1),
               (0, w // 2), (h - 1, w // 2), (h // 2, 0),
               (h // 2, w - 1), (h // 2, w // 2)]
    if pool:
        for anchor in anchors:
            shortlist.add(min(pool, key=lambda p: (
                abs(p[0] - anchor[0]) + abs(p[1] - anchor[1]), risk_key(p))))
        # A square just beyond the existing frontier can reveal overlapping
        # equations while retaining the unconstrained marginal mine rate.
        shortlist.add(min(pool, key=lambda p: (
            min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for q in revealed),
            risk_key(p))))
    shortlist.add(baseline)
    ordered = sorted(shortlist)

    forced = {}
    for p in ordered:
        if stop():
            raise SearchLimit()
        around = set(adjacency[p])
        local = frozenset(around & active)
        fixed = len(around & known)
        # Only enumerate clue outcomes that actually occur in posterior draws.
        outcomes = {sum(q in world for q in adjacency[p])
                    for world in worlds if p not in world}
        for n in outcomes:
            safes, _, _ = closure(eqs + [(frozenset([p]), 0), (local, n - fixed)], stop=stop)
            forced[p, n] = safes - known

    gain = {p: [] for p in ordered}
    zero_events = defaultdict(int)
    for si, world in enumerate(worlds):
        if si % 8 == 0 and stop():
            raise SearchLimit()
        numbers = defaultdict(int)
        for mine in world:
            for q in adjacency[mine]:
                numbers[q] += 1
        # Identify each zero region and its safe border once per sample.
        zero_cache = {}
        for p in ordered:
            if p in world:
                continue
            n = numbers[p]
            if n:
                opened = {p}
            elif p in zero_cache:
                opened = zero_cache[p]
                zero_events[p] += 1
            else:
                stack, seen, opened = [p], set(), set()
                while stack:
                    q = stack.pop()
                    if q in seen:
                        continue
                    seen.add(q)
                    opened.add(q)
                    for t in adjacency[q]:
                        if t not in world:
                            opened.add(t)
                            if numbers[t] == 0 and t not in seen:
                                stack.append(t)
                for q in seen:
                    zero_cache[q] = opened
                zero_events[p] += 1
            gain[p].append(len((opened | forced[p, n]) & hidden))
    rows = []
    for p in ordered:
        values = gain[p]
        if len(values) < 24:
            continue
        mean = sum(values) / len(values)
        variance = max(0.0, sum(x * x for x in values) / len(values) - mean * mean)
        # A modest uncertainty discount reduces selecting lucky flood samples.
        conservative_gain = max(1.0, mean - 0.5 * math.sqrt(variance / len(values)))
        risk = rates.get(p, 1.0)
        utility = (1.0 - risk) * conservative_gain / max(risk, 1e-9)
        rows.append((utility, -risk, p, mean, zero_events[p] / len(worlds)))
    if not rows:
        return None, {"reason": "too_few_samples"}
    best = max(rows, key=lambda row: (row[0], row[1], tuple(-v for v in row[2])))
    return best[2], {
        "reason": "posterior_progress", "chosen": list(best[2]),
        "baseline": list(baseline), "samples": len(worlds),
        "candidates": [{"cell": list(row[2]), "risk": -row[1],
                        "conditional_gain": round(row[3], 3),
                        "p_zero_sample": round(row[4], 3),
                        "utility": round(row[0], 3)}
                       for row in sorted(rows, reverse=True)],
        "search_s": round(time.monotonic() - started, 3),
    }


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
        stalled = 0
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
                try:
                    safe, mines, reduced = closure(base, stop=lambda: ctx.done)
                except SearchLimit:
                    return
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
                    try:
                        rates = global_rates(hidden, reduced, int(obs.get("mines", 0) or 0),
                                             self.known, stop=lambda: ctx.done)
                    except SearchLimit:
                        return
                    candidates = [p for p in hidden if p not in self.known and
                                  parse_cell(grid[p[0]][p[1]])[0] == "hidden"]
                    if not candidates:
                        return

                    def guess_key(p):
                        risk = rates.get(p, 1.0)
                        degree = sum(1 for _ in neighbours(p[0], p[1], h, w))
                        frontier_touch = sum(q in revealed for q in neighbours(p[0], p[1], h, w))
                        return (round(risk, 12), degree, -frontier_touch,
                                abs(p[0] - h / 2) + abs(p[1] - w / 2), p[0], p[1])
                    chosen = min(candidates, key=guess_key)
                    # Exhaustive local certainty takes priority over planning.
                    if rates.get(chosen, 1.0) > 0.0:
                        try:
                            planned, diagnostics = plan_reveal(
                                hidden, revealed, self.known, rates,
                                int(obs.get("mines", 0)), h, w, chosen, ctx)
                            if planned is not None:
                                chosen = planned
                            ctx.log("guess_lookahead", **diagnostics)
                        except SearchLimit:
                            pass
                    r, c = chosen
                action = {"type": "reveal", "row": int(r), "col": int(c)}

            before = (obs.get("safe_revealed"), obs.get("flags"))
            try:
                nxt = ctx.act(action)
                obs = nxt if isinstance(nxt, dict) else game.observe()
                after = (obs.get("safe_revealed"), obs.get("flags"))
                stalled = stalled + 1 if after == before else 0
            except Exception:
                stalled += 1
                try:
                    obs = game.observe()
                except Exception:
                    return
            if stalled >= 3:
                return


def build(ctx):
    return Harness(ctx)
