"""Deterministic Minesweeper constraint harness.

The previous candidate delegated every turn to Astra.  This harness keeps the
visible board as the source of truth, derives exact local constraints, and
uses enumeration only when no forced move remains.  Astra is deliberately not
needed: a turn game has no benefit from paying tens of seconds for arithmetic.
"""
from __future__ import annotations

from collections import defaultdict

from metaharness.core.contract import BudgetExceeded, EpisodeOver, ModelUnavailable


def _neighbors(r, c, h, w):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    yield rr, cc


def _cell_kind(v):
    if isinstance(v, str):
        if v == "unknown":
            return "u", None
        if v == "flagged":
            return "f", None
        if v.startswith("revealed:"):
            try:
                return "r", int(v.split(":", 1)[1])
            except Exception:
                return "r", None
        if v == "mine":
            return "m", None
    return "u", None


def _equations(obs, confirmed):
    cells = obs.get("cells") or []
    h, w = len(cells), (len(cells[0]) if cells else 0)
    eqs = []
    frontier = set()
    for r in range(h):
        for c in range(w):
            kind, value = _cell_kind(cells[r][c])
            if kind != "r" or value is None:
                continue
            unknown = set()
            known_mines = 0
            for q in _neighbors(r, c, h, w):
                k, _ = _cell_kind(cells[q[0]][q[1]])
                if q in confirmed:
                    known_mines += 1
                elif k in ("u", "f"):
                    unknown.add(q)
            residual = value - known_mines
            if unknown and 0 <= residual <= len(unknown):
                fs = frozenset(unknown)
                eqs.append((fs, residual))
                frontier.update(fs)
            elif not unknown and residual != 0:
                # A contradiction means a stale deduction; caller will fall
                # back to a fresh probabilistic choice.
                eqs.append((frozenset(), residual))
    # Remove duplicate equations while retaining deterministic order.
    return list(dict.fromkeys(eqs)), frontier


def _forced(eqs):
    """Return (safe, mines) from zero/full and subset constraints."""
    safe, mines = set(), set()
    work = list(eqs)
    for _ in range(8):
        changed = False
        normalized = []
        for s, n in work:
            original = set(s)
            n = n - len(mines & original)
            s = original - safe - mines
            # Newly forced mines are removed from a constraint and its count
            # is reduced at the same time; otherwise a stale residual could
            # incorrectly turn every remaining cell into a mine.
            if not s:
                continue
            if 0 <= n <= len(s):
                normalized.append((frozenset(s), n))
                if n == 0:
                    before = len(safe)
                    safe.update(s)
                    changed |= len(safe) != before
                elif n == len(s):
                    before = len(mines)
                    mines.update(s)
                    changed |= len(mines) != before
        # Subset subtraction is exact and often exposes the next move.
        all_eq = normalized[:]
        for a, na in normalized:
            for b, nb in normalized:
                if a == b or not a < b:
                    continue
                d = b - a
                dn = nb - na
                if d and 0 <= dn <= len(d):
                    item = (frozenset(d), dn)
                    if item not in all_eq:
                        all_eq.append(item)
                        changed = True
                        if dn == 0:
                            before = len(safe)
                            safe.update(d)
                            changed |= len(safe) != before
                        elif dn == len(d):
                            before = len(mines)
                            mines.update(d)
                            changed |= len(mines) != before
        work = all_eq
        if not changed:
            break
    safe.difference_update(mines)
    return safe, mines


def _components(eqs):
    """Partition frontier equations into independent cell components."""
    by_cell = defaultdict(list)
    for s, n in eqs:
        for cell in s:
            by_cell[cell].append((s, n))
    seen, out = set(), []
    for start in by_cell:
        if start in seen:
            continue
        stack, comp_cells, comp_eq = [start], set(), set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp_cells.add(x)
            for s, n in by_cell.get(x, ()):
                key = (s, n); comp_eq.add(key)
                for y in s:
                    if y not in seen:
                        stack.append(y)
        out.append((comp_cells, list(comp_eq)))
    return out


def _probabilities(eqs, limit=21):
    """Enumerate small constraint components and return cell mine rates."""
    rates = {}
    for cells, ceqs in _components(eqs):
        ordered = sorted(cells)
        if len(ordered) > limit:
            continue
        index = {x: i for i, x in enumerate(ordered)}
        local = [([index[x] for x in s], n) for s, n in ceqs]
        counts = [0] * len(ordered)
        total = 0
        # Backtracking with early equation bounds keeps 21-cell components
        # cheap while avoiding a huge 2**N flat loop.
        assign = [-1] * len(ordered)
        incidence = [[] for _ in ordered]
        for ei, (inds, n) in enumerate(local):
            for i in inds:
                incidence[i].append(ei)
        sums = [0] * len(local); seen = [0] * len(local)
        def rec(i):
            nonlocal total
            if i == len(ordered):
                if all(sums[j] == n for j, (_, n) in enumerate(local)):
                    total += 1
                    for k, v in enumerate(assign):
                        counts[k] += v
                return
            for v in (0, 1):
                ok = True
                touched = []
                for ei in incidence[i]:
                    sums[ei] += v; seen[ei] += 1; touched.append(ei)
                    inds, n = local[ei]
                    rem = len(inds) - seen[ei]
                    if sums[ei] > n or sums[ei] + rem < n:
                        ok = False
                if ok:
                    assign[i] = v; rec(i + 1)
                for ei in touched:
                    sums[ei] -= v; seen[ei] -= 1
            assign[i] = -1
        rec(0)
        if total:
            for i, cell in enumerate(ordered):
                rates[cell] = counts[i] / total
    return rates


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.confirmed = set()
        self.flagged = set()

    def _act(self, game, action):
        try:
            return self.ctx.act(action)
        except (BudgetExceeded, EpisodeOver, ModelUnavailable, TimeoutError, RuntimeError):
            return None
        except Exception:
            # Illegal actions should not make a pre-flight garbage episode
            # fail; the next observation is enough to recover.
            return None

    def run_episode(self, game):
        ctx = self.ctx
        try:
            obs = game.observe()
        except Exception:
            return
        while not ctx.done and not getattr(game, "done", False):
            if not isinstance(obs, dict):
                try:
                    obs = game.observe()
                except Exception:
                    break
            status = obs.get("status")
            if status in ("won", "lost") or obs.get("won"):
                break
            cells = obs.get("cells") or []
            h, w = len(cells), (len(cells[0]) if cells else 0)
            if not h or not w:
                break
            # The guaranteed-safe opening is immediate and deterministic.
            if int(obs.get("safe_revealed", 0) or 0) == 0:
                action = f"reveal {h // 2} {w // 2}"
                new = self._act(game, action)
                if new is None:
                    try: new = game.observe()
                    except Exception: break
                obs = new
                continue

            hidden = set()
            visible_flags = set()
            for r in range(h):
                for c in range(w):
                    k, _ = _cell_kind(cells[r][c])
                    if k in ("u", "f"):
                        hidden.add((r, c))
                        if k == "f": visible_flags.add((r, c))
            # Keep only coordinates still hidden; stale entries cannot affect
            # future equations after flood fill.
            self.confirmed.intersection_update(hidden)
            eqs, frontier = _equations(obs, self.confirmed)
            safe, mines = _forced(eqs)
            self.confirmed.update(mines)
            # Flag only code-proven mines, then use valid chords for bulk gain.
            pending = sorted(self.confirmed - visible_flags)
            if pending:
                action = f"flag {pending[0][0]} {pending[0][1]}"
            else:
                chord = None
                for r in range(h):
                    for c in range(w):
                        k, n = _cell_kind(cells[r][c])
                        if k != "r" or n is None:
                            continue
                        ns = list(_neighbors(r, c, h, w))
                        fs = sum(1 for q in ns if q in visible_flags)
                        hu = [q for q in ns if q in hidden and q not in visible_flags]
                        if hu and fs == n:
                            chord = (r, c); break
                    if chord: break
                if chord:
                    action = f"chord {chord[0]} {chord[1]}"
                else:
                    candidates = sorted(safe & hidden)
                    if not candidates:
                        rates = _probabilities(eqs)
                        # Unconstrained cells get the global remaining-mine
                        # rate; this is safer than Astra's arbitrary guesses.
                        rem = max(0, int(obs.get("remaining_mines", obs.get("mines", 0)) or 0)
                                   - len(self.confirmed - visible_flags))
                        uncon = hidden - frontier
                        base = rem / max(1, len(hidden))
                        candidates = sorted(hidden, key=lambda q: (rates.get(q, base), q))
                    if not candidates:
                        break
                    q = candidates[0]
                    action = f"reveal {q[0]} {q[1]}"
            new = self._act(game, action)
            if new is None:
                try: new = game.observe()
                except Exception: break
            obs = new


def build(ctx):
    return Harness(ctx)
