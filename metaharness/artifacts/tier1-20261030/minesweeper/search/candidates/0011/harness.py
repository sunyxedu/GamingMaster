"""Minesweeper solver with exact posterior Bellman endgame search.

Exact constraints certify moves. Posterior completions are used only to
simulate feedback and value three-gamble continuations, never as live facts.
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


class SimWorld:
    """A posterior completion used only to produce observable simulator feedback."""
    def __init__(self, mines, adjacency):
        self.mines = mines
        self.adj = adjacency
        self.numbers = {p: sum(q in mines for q in ns) for p, ns in adjacency.items()}
        self.floods = {}

    def reveal(self, p):
        if p in self.mines:
            return None
        if self.numbers[p]:
            return {p: self.numbers[p]}
        if p not in self.floods:
            stack, zeros, opened = [p], set(), set()
            while stack:
                q = stack.pop()
                if q in zeros:
                    continue
                zeros.add(q)
                opened.add(q)
                for t in self.adj[q]:
                    if t not in self.mines:
                        opened.add(t)
                        if self.numbers[t] == 0 and t not in zeros:
                            stack.append(t)
            result = {q: self.numbers[q] for q in opened}
            for q in zeros:
                self.floods[q] = result
        return self.floods[p]


def exact_completions(hidden, revealed, known, total, h, w, stop,
                      max_cells=24, max_worlds=4096, max_nodes=1000000):
    """Complete, equally weighted posterior; a bound returns no partial facts."""
    active = sorted(hidden - known)
    if not active or len(active) > max_cells:
        return None
    index = {p: i for i, p in enumerate(active)}
    remaining = total - len(known)
    if not 0 <= remaining <= len(active):
        return None
    local = []
    for p, number in revealed.items():
        ns = set(neighbours(*p, h, w))
        cells = tuple(sorted(index[q] for q in ns if q in index))
        need = number - len(ns & known)
        if not 0 <= need <= len(cells):
            return None
        if cells:
            local.append((cells, need))
    local = list(dict.fromkeys(local))
    incidence = [[] for _ in active]
    for j, (cells, _) in enumerate(local):
        for i in cells:
            incidence[i].append(j)
    order = sorted(range(len(active)), key=lambda i: (-len(incidence[i]), i))
    sums, used = [0] * len(local), [0] * len(local)
    worlds, nodes = [], 0

    def visit(pos, count, mask):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or nodes % 256 == 0 and stop():
            raise SearchLimit()
        if count > remaining or count + len(active) - pos < remaining:
            return
        if pos == len(active):
            worlds.append(mask)
            if len(worlds) > max_worlds:
                raise SearchLimit()
            return
        i = order[pos]
        for value in (0, 1):
            valid = True
            for j in incidence[i]:
                sums[j] += value
                used[j] += 1
                cells, need = local[j]
                if sums[j] > need or sums[j] + len(cells) - used[j] < need:
                    valid = False
            if valid:
                visit(pos + 1, count + value, mask | ((1 << i) if value else 0))
            for j in incidence[i]:
                sums[j] -= value
                used[j] -= 1

    try:
        if stop():
            return None
        visit(0, 0, 0)
    except SearchLimit:
        return None
    return (active, worlds) if worlds else None


class ExactEndgame:
    """Bellman search on complete observable belief states, without hindsight.

    Values are integer sums of additional safe reveals over equally probable
    worlds.  A reveal's branches contain only worlds with identical feedback.
    Simulated mines generate feedback, never influence a branch's choice.
    """
    def __init__(self, cells, masks, known, h, w, stop, max_states=60000):
        self.cells, self.masks, self.stop = cells, masks, stop
        self.max_states, self.cache = max_states, {}
        self.index = {p: i for i, p in enumerate(cells)}
        self.full = (1 << len(cells)) - 1
        self.feedback = []
        adjacency = {p: tuple(neighbours(*p, h, w))
                     for p in ((r, c) for r in range(h) for c in range(w))}
        for mask in masks:
            self.check()
            mines = set(known) | {p for i, p in enumerate(cells) if mask >> i & 1}
            world = SimWorld(mines, adjacency)
            row = []
            for p in cells:
                response = world.reveal(p)
                if response is None:
                    row.append(None)
                else:
                    # One bitmask per clue value represents the complete
                    # feedback. Restriction to currently hidden cells avoids
                    # splitting identical observations by already-known clues.
                    by_number = [0] * 9
                    for q, number in response.items():
                        i = self.index.get(q)
                        if i is not None:
                            by_number[number] |= 1 << i
                    row.append(tuple(by_number))
            self.feedback.append(row)

    def check(self):
        if self.stop() or len(self.cache) >= self.max_states:
            raise SearchLimit()

    @staticmethod
    def bits(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def branches(self, ids, hidden, moves):
        """Reveal guaranteed-safe moves together; group by actual feedback."""
        groups = {}
        for wi in ids:
            signature = [0] * 9
            for i in moves:
                feedback = self.feedback[wi][i]
                if feedback is None:
                    # For a single risky move this is the terminal loss branch.
                    # Multiple moves are allowed only after universal proof.
                    if len(moves) != 1:
                        raise ValueError('unsafe exact batch')
                    break
                for number, mask in enumerate(feedback):
                    signature[number] |= mask & hidden
            else:
                key = tuple(signature)
                groups.setdefault(key, []).append(wi)
        return groups

    def action_value(self, ids, hidden, moves):
        score = 0
        for feedback, members in self.branches(ids, hidden, moves).items():
            opened = 0
            for mask in feedback:
                opened |= mask
            if not opened:
                raise ValueError('exact reveal did not progress')
            score += opened.bit_count() * len(members)
            score += self.solve(tuple(members), hidden & ~opened)
        return score

    def solve(self, ids, hidden):
        self.check()
        # Removing universally mined cells is only belief-state compression;
        # they still contribute to every simulated clue.
        possible_mines, certain_mines = 0, self.full
        for wi in ids:
            possible_mines |= self.masks[wi]
            certain_mines &= self.masks[wi]
        hidden &= ~certain_mines
        if not hidden:
            return 0
        key = (ids, hidden)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        safe = hidden & ~possible_mines
        if safe:
            # Revealing all guaranteed-safe cells before any gamble cannot
            # reduce expected progress. The observations still branch before
            # another choice is made, including any zero flood feedback.
            value = self.action_value(ids, hidden, tuple(self.bits(safe)))
        else:
            value = max(self.action_value(ids, hidden, (i,))
                        for i in self.bits(hidden))
        self.cache[key] = value
        return value

    def root_values(self):
        ids = tuple(range(len(self.masks)))
        values = {}
        for i, p in enumerate(self.cells):
            self.check()
            if all(mask >> i & 1 for mask in self.masks):
                continue
            values[p] = self.action_value(ids, self.full, (i,))
        return values


def exact_endgame(hidden, revealed, known, total, h, w, baseline, ctx):
    """Exact score optimization where the complete posterior fits the cap."""
    if len(hidden - known) > 24:
        return None, None
    started = time.monotonic()
    wall_left = float(ctx.budget.remaining().get('wall_s', 0))
    deadline = started + min(18.0, max(0.0, wall_left / 10.0 - 0.5))
    def stop():
        return ctx.done or time.monotonic() >= deadline
    complete = exact_completions(hidden, revealed, known, total, h, w, stop)
    if complete is None:
        return None, {'reason': 'exact_endgame_incomplete'}
    cells, masks = complete
    try:
        solver = ExactEndgame(cells, masks, known, h, w, stop)
        values = solver.root_values()
    except SearchLimit:
        return None, {'reason': 'exact_endgame_limit', 'worlds': len(masks)}
    if not values:
        return None, {'reason': 'exact_endgame_empty'}
    # Do not manufacture preferences from finite sample noise in exact ties.
    # Retain the pre-existing geometric/risk baseline whenever it is optimal.
    best_value = max(values.values())
    best = [p for p in values if values[p] == best_value]
    chosen = baseline if baseline in best else min(best, key=lambda p: (
        sum(mask >> cells.index(p) & 1 for mask in masks),
        len(tuple(neighbours(*p, h, w))), p))
    return chosen, {'reason': 'exact_endgame', 'chosen': list(chosen),
        'baseline': list(baseline), 'worlds': len(masks), 'states': len(solver.cache),
        'candidates': [{'cell': list(p), 'gain_numerator': values[p],
                        'expected_gain': values[p] / len(masks)} for p in sorted(values)],
        'search_s': round(time.monotonic() - started, 6)}


def residuals(hidden, revealed, known, adjacency):
    eqs = []
    active = hidden - known
    for p, number in revealed.items():
        around = set(adjacency[p])
        cells = frozenset(around & active)
        need = number - len(around & known)
        if not 0 <= need <= len(cells):
            raise ValueError("contradictory visible clues")
        if cells:
            eqs.append((cells, need))
    return list(dict.fromkeys(eqs))


def rollout(world, first, initial_hidden, initial_revealed, initial_known,
            total, adjacency, stop, cache, horizon=3):
    """Evaluate a fixed observation-only continuation policy, without hindsight.

    Safe moves cost no gamble. The sampled world is consulted only by reveal,
    never by the policy that selects the next action or declares a cell safe.
    """
    hidden = set(initial_hidden)
    revealed = dict(initial_revealed)
    known = set(initial_known)
    action, gambles = first, 0
    center_r = (max(p[0] for p in adjacency) + 1) / 2
    center_c = (max(p[1] for p in adjacency) + 1) / 2
    while True:
        if stop():
            raise SearchLimit()
        feedback = world.reveal(action)
        gambles += 1
        if feedback is None:
            return len(initial_hidden) - len(hidden)
        revealed.update(feedback)
        hidden.difference_update(feedback)
        while hidden - known:
            if stop():
                raise SearchLimit()
            key = (tuple(sorted(revealed.items())), tuple(sorted(known)))
            decision = cache.get(key)
            if decision is None:
                eqs = residuals(hidden, revealed, known, adjacency)
                safe, mines, _ = closure(eqs, stop=stop)
                more_known = known | mines
                safe.intersection_update(hidden - more_known)
                if safe:
                    decision = (frozenset(safe), frozenset(more_known), None)
                else:
                    eqs = residuals(hidden, revealed, more_known, adjacency)
                    rates = global_rates(hidden, eqs, total, more_known, stop=stop)
                    active = hidden - more_known
                    if not active:
                        decision = (frozenset(), frozenset(more_known), None)
                    else:
                        def key_move(p):
                            return (round(rates.get(p, 1.0), 12), len(adjacency[p]),
                                    -sum(q in revealed for q in adjacency[p]),
                                    abs(p[0]-center_r)+abs(p[1]-center_c), p)
                        next_action = min(active, key=key_move)
                        if rates.get(next_action, 1.0) == 0.0:
                            decision = (frozenset([next_action]), frozenset(more_known), None)
                        else:
                            decision = (frozenset(), frozenset(more_known), next_action)
                if len(cache) < 12000:
                    cache[key] = decision
            safe, facts, next_action = decision
            known = set(facts)
            if not safe:
                break
            for p in sorted(safe):
                if p not in hidden:
                    continue
                feedback = world.reveal(p)
                if feedback is None:
                    # A proof failing in a consistent world invalidates this
                    # simulated branch; never conceal it as a useful sample.
                    raise ValueError("simulated safety proof contradicted")
                revealed.update(feedback)
                hidden.difference_update(feedback)
        if not hidden - known or gambles >= horizon or next_action is None:
            return len(initial_hidden) - len(hidden)
        action = next_action


def plan_reveal(hidden, revealed, known, rates, total, h, w, baseline, ctx):
    """Posterior policy evaluation for three successive risky reveals."""
    started = time.monotonic()
    wall_left = float(ctx.budget.remaining().get("wall_s", 60))
    deadline = started + min(45.0, max(0.0, wall_left / 8.0 - 1.0))
    def stop():
        return ctx.done or time.monotonic() >= deadline
    adjacency = {p: tuple(neighbours(*p, h, w))
                 for p in ((r, c) for r in range(h) for c in range(w))}
    eqs = residuals(hidden, revealed, known, adjacency)
    sampled = sample_worlds(hidden, known, eqs, total, stop, count=384)
    if sampled is None:
        return None, {"reason": "incomplete_posterior", "fallback": list(baseline)}
    layouts, infos = sampled
    rates = global_rates(hidden, eqs, total, known, stop=stop, precomputed=infos)
    active = hidden - known
    minimum = min(rates.get(p, 1.0) for p in active)
    eligible = {p for p in active if rates.get(p, 1.0) <= minimum + 0.06}
    if minimum == 0.0:
        p = min(eligible, key=lambda p: (rates[p], p))
        return p, {"reason": "exact_safe", "chosen": list(p)}
    if baseline not in eligible:
        baseline = min(eligible, key=lambda p: (rates.get(p, 1.0), p))
    frontier = {p for cells, _ in eqs for p in cells}
    def risk_key(p):
        return (round(rates.get(p, 1.0), 12),
                -sum(q in revealed for q in adjacency[p]), len(adjacency[p]), p)
    candidates = set(sorted(eligible & frontier, key=risk_key)[:8])
    pool = eligible - frontier
    anchors = [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1),
               (0, w//2), (h-1, w//2), (h//2, 0), (h//2, w-1), (h//2, w//2)]
    if pool:
        for a in anchors:
            candidates.add(min(pool, key=lambda p: (
                abs(p[0]-a[0])+abs(p[1]-a[1]), risk_key(p))))
        candidates.add(min(pool, key=lambda p: (
            -sum(q in frontier for q in adjacency[p]), risk_key(p))))
    candidates.add(baseline)
    worlds = []
    for layout in layouts:
        if stop():
            raise SearchLimit()
        worlds.append(SimWorld(layout, adjacency))
    # Cheap broad screen preserves both flood opportunities and cells whose
    # clues intersect existing frontier hypotheses. This is not the final
    # objective; subsequent policy progress is evaluated for every finalist.
    pilot = {}
    for p in sorted(candidates):
        if stop():
            raise SearchLimit()
        values = [len(set(feedback) & hidden)
                  for world in worlds if (feedback := world.reveal(p)) is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        risk = rates[p]
        pilot[p] = (1-risk)*mean/max(risk, 1e-9)
    if not pilot:
        return None, {"reason": "empty_pilot"}
    fallback = max(pilot, key=lambda p: (pilot[p], -rates[p], tuple(-x for x in p)))
    finalists = set(sorted(pilot, key=lambda p: (-pilot[p], rates[p], p))[:5])
    # Do not discard low-immediate-gain information probes during screening.
    finalists.update(sorted(candidates, key=lambda p: (
        -sum(q in frontier for q in adjacency[p]), risk_key(p)))[:2])
    finalists.add(baseline)
    ordered = sorted(finalists)
    gains = {p: [] for p in ordered}
    rounds, cache = 0, {}
    # Complete balanced rounds only: slow branches cannot bias the sample set.
    for world in worlds[:256]:
        if stop():
            break
        row = {}
        try:
            for p in ordered:
                row[p] = None if p in world.mines else rollout(
                    world, p, hidden, revealed, known, total, adjacency, stop, cache)
        except SearchLimit:
            break
        except ValueError:
            return None, {"reason": "inconsistent_rollout"}
        for p, gain in row.items():
            if gain is not None:
                gains[p].append(gain)
        rounds += 1
    if rounds < 12:
        return fallback, {"reason": "rollout_limit", "rounds": rounds,
                          "chosen": list(fallback), "search_s": round(time.monotonic()-started, 3)}
    rows = []
    for p in ordered:
        values = gains[p]
        if len(values) < 8:
            continue
        mean = sum(values)/len(values)
        var = max(0.0, sum(x*x for x in values)/len(values)-mean*mean)
        expected = (1-rates[p])*mean
        uncertainty = (1-rates[p])*math.sqrt(var/len(values))
        utility = expected - 0.5*uncertainty
        rows.append({"cell": list(p), "risk": rates[p], "expected_gain": round(expected, 4),
                     "standard_error": round(uncertainty, 4), "utility": round(utility, 4),
                     "safe_samples": len(values), "pilot": round(pilot.get(p, 0), 4)})
    if not rows:
        return fallback, {"reason": "insufficient_safe_samples", "rounds": rounds}
    best = max(rows, key=lambda r: (r["utility"], -r["risk"], tuple(-v for v in r["cell"])))
    return tuple(best["cell"]), {"reason": "posterior_continuation", "chosen": best["cell"],
            "baseline": list(baseline), "pilot_choice": list(fallback), "rounds": rounds,
            "horizon": 3, "candidates": rows, "cache_states": len(cache),
            "search_s": round(time.monotonic()-started, 3)}


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
                        # Rebuild after closure added mines: no stale residuals.
                        _, _, _, _, reduced = equations(obs, self.known)
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
                            planned, diagnostics = exact_endgame(
                                hidden, revealed, self.known,
                                int(obs.get("mines", 0)), h, w, chosen, ctx)
                            if diagnostics is not None:
                                ctx.log("exact_endgame_search", **diagnostics)
                            if planned is None:
                                planned, diagnostics = plan_reveal(
                                    hidden, revealed, self.known, rates,
                                    int(obs.get("mines", 0)), h, w, chosen, ctx)
                            if planned is not None:
                                chosen = planned
                            ctx.log("guess_lookahead", **diagnostics)
                        except (SearchLimit, ValueError, OverflowError):
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
