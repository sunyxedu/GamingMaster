"""Public-range turn-to-river planning against heuristic-v1; stdlib only."""
from __future__ import annotations

import bisect
import hashlib
import itertools
import math
import random
from collections import Counter

from metaharness.core.contract import BudgetExceeded

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = tuple(range(52))
COMBOS = tuple(itertools.combinations(DECK, 2))
STREET_N = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
STRAIGHT = [0] * 8192
for _mask in range(8192):
    for _high in range(12, 3, -1):
        if (_mask >> (_high - 4)) & 31 == 31:
            STRAIGHT[_mask] = _high + 2
            break
    if not STRAIGHT[_mask] and _mask & 4111 == 4111:
        STRAIGHT[_mask] = 5


def card(s):
    return 4 * RANKS.index(s[0]) + SUITS.index(s[1])


def score(cards):
    """Exact best five of 5--7, packed lexicographic rank (larger wins)."""
    counts = [0] * 13
    suits = [0] * 4
    mask = 0
    for c in cards:
        r, s = c // 4, c % 4
        counts[r] += 1
        suits[s] |= 1 << r
        mask |= 1 << r
    flush = next((m for m in suits if m.bit_count() >= 5), 0)
    if flush and STRAIGHT[flush]:
        return (8 * 15 + STRAIGHT[flush]) * 15 ** 4
    quads = [r + 2 for r in range(12, -1, -1) if counts[r] == 4]
    trips = [r + 2 for r in range(12, -1, -1) if counts[r] == 3]
    pairs = [r + 2 for r in range(12, -1, -1) if counts[r] >= 2]
    high = [r + 2 for r in range(12, -1, -1) if counts[r]]
    if quads:
        values = [7, quads[0], next(r for r in high if r != quads[0])]
    elif trips and len(pairs) >= 2:
        values = [6, trips[0], next(r for r in pairs if r != trips[0])]
    elif flush:
        values = [5] + [r + 2 for r in range(12, -1, -1) if flush & (1 << r)][:5]
    elif STRAIGHT[mask]:
        values = [4, STRAIGHT[mask]]
    elif trips:
        values = [3, trips[0]] + [r for r in high if r != trips[0]][:2]
    elif len(pairs) >= 2:
        values = [2] + pairs[:2] + [next(r for r in high if r not in pairs[:2])]
    elif pairs:
        values = [1, pairs[0]] + [r for r in high if r != pairs[0]][:3]
    else:
        values = [0] + high[:5]
    out = 0
    for v in values + [0] * (6 - len(values)):
        out = out * 15 + v
    return out


def klass(h):
    a, b = sorted((h[0] // 4 + 2, h[1] // 4 + 2), reverse=True)
    return (a, b, h[0] % 4 == h[1] % 4)


def premium(h):
    a, b, s = klass(h)
    return a == b and a >= 12 or a == 14 and b == 13


def opens(h):
    a, b, suited = klass(h)
    return (a == b or suited or a >= 13 or
            (a == 12 and b >= 7) or (a == 11 and b >= 8) or
            (a == 10 and b >= 8) or (a, b) in ((9, 8), (8, 7)))


def seeded(label):
    return random.Random(int.from_bytes(hashlib.sha256(str(label).encode()).digest()[:8], 'big'))


def smooth(x, cutoff, width):
    z = max(-30.0, min(30.0, (x - cutoff) / width))
    return 1.0 / (1.0 + math.exp(-z))


def legal_action(o, action, amount=None):
    legal = o.get('legal', {})
    if action == 'raise_to' and legal.get('can_raise'):
        lo, hi = legal.get('min_raise_to'), legal.get('max_raise_to')
        if lo is not None and hi is not None:
            return 'raise_to ' + str(max(int(lo), min(int(hi), int(amount))))
    if legal.get('can_' + action):
        return action
    for a in ('check', 'fold', 'call'):
        if legal.get('can_' + a):
            return a
    choices = o.get('candidates', [])
    return choices[0]['id'] if choices else 'wait'


class _StopEpisode(Exception):
    """Cooperative cancellation of CPU work, caught at the episode boundary."""


class RiverDistribution:
    """Prefix moments on exact opponent random-hand equity; weights need not sum to 1."""
    def __init__(self, rows):
        self.q = []
        self.mass = [0.0]
        self.wins = [0.0]
        for q, h, w in rows:  # already sorted by q
            if w > 0:
                self.q.append(q)
                self.mass.append(self.mass[-1] + w)
                self.wins.append(self.wins[-1] + w * h)

    def moments(self, low=0.0, high=1.00001):
        a = bisect.bisect_left(self.q, low)
        b = bisect.bisect_left(self.q, high)
        return self.mass[b] - self.mass[a], self.wins[b] - self.wins[a]


def size_menu(pot, call, hbet, vbet, hstack, vstack, minimum=None, maximum=None):
    maximum = min(hbet + hstack, vbet + vstack) if maximum is None else maximum
    minimum = (vbet + max(100, vbet - hbet) if call else hbet + 100) if minimum is None else minimum
    if hstack <= call or vstack <= 0 or maximum <= max(hbet, vbet):
        return ()
    minimum = min(minimum, maximum)  # legal short all-in
    targets = {int(minimum), int(maximum)}
    for f in (0.25, 0.50, 0.75, 1.0, 1.5, 2.5, 4.0):
        targets.add(int(max(minimum, min(maximum, int(max(hbet, vbet) + f * (pot + call) + 0.5)))))
    return tuple(sorted(targets))


class RiverSolver:
    """Exact terminal response tree against the documented deterministic river policy.

    One action is selected for an entire public-information branch. The hidden
    holding only enters distribution moments, never the choice of a hero action.
    Returns (expected chips from this node, action, selection utility).
    """
    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def node(self, d, pot, hstack, vstack, hbet=0, vbet=0,
             raised=False, low=0.0, high=1.00001, bounds=None, check_value=None):
        self.checkpoint()
        mass, wins = d.moments(low, high)
        if mass <= 1e-12:
            return 0.0, 'check' if hbet >= vbet else 'fold', 0.0
        call = min(hstack, max(0, vbet - hbet))
        passive = wins * (pot + call) - mass * call
        if not call and check_value is not None:
            passive = check_value
        best_ev = max(0.0, passive) if call else passive
        best = 'call' if call and passive > 0 else 'fold' if call else 'check'
        utility = best_ev
        kwargs = {} if bounds is None else {'minimum': bounds[0], 'maximum': bounds[1]}
        for target in size_menu(pot, call, hbet, vbet, hstack, vstack, **kwargs):
            cost = min(target - hbet, call + vstack)
            vc = min(vstack, target - vbet)
            matched = pot + cost + vc
            cutoff = vc / max(1, matched)
            # A raising opportunity exists only with money behind on both sides.
            can_reraise = not raised and min(hstack - cost, vstack - vc) > 0
            rlow = max(low, 0.80) if can_reraise else high
            rlow = min(high, rlow)
            cm, cw = d.moments(max(low, cutoff), rlow) if max(low, cutoff) < rlow else (0.0, 0.0)
            rm, rw = d.moments(rlow, high)
            fm = max(0.0, mass - cm - rm)
            ev = fm * pot + cw * matched - cm * cost
            if rm > 1e-12:
                extra = min(hstack - cost, vstack - vc,
                            max(target - max(hbet, vbet), int(0.75 * matched + 0.5)))
                # The opponent has now used its one raise. Reopening is legal
                # unless it made a short all-in, which leaves no money to raise.
                continuation, _, _ = self.node(
                    d, pot + cost + vc + extra, hstack - cost, vstack - vc - extra,
                    target, target + extra, True, rlow, high)
                ev += continuation - rm * cost
            adjusted = ev - 0.025 * cost * (cm + rm)
            if cost > 3 * pot:
                cmass = cm + rm
                ceq = (cw + rw) / max(cmass, 1e-12)
                if ceq < 0.56:
                    continue
            if adjusted > utility + max(8.0, 0.025 * pot) * mass:
                utility, best_ev, best = adjusted, ev, 'raise_to ' + str(target)
        return best_ev, best, utility

    def after_check(self, d, pot, stack):
        """Hero checks OOP; opponent checks or selects one of three public bets."""
        self.checkpoint()
        weak_mass, weak_wins = d.moments(0.0, 0.62)
        ev = 0.92 * weak_wins * pot
        branches = ((0.0, 0.62, 0.08, 0.5),
                    (0.62, 0.85, 1.0, 0.75), (0.85, 1.00001, 1.0, 1.0))
        # Clipping can make distinct equity branches produce the same public
        # size. Merge their distributions before taking a maximum.
        groups = {}
        for low, high, prob, frac in branches:
            amount = min(stack, max(100, int(frac * pot + 0.5)))
            groups.setdefault(amount, []).append((low, high, prob))
        for amount, branches in groups.items():
            if amount <= 0:
                continue
            if len(branches) == 1:
                low, high, prob = branches[0]
                value, _, _ = self.node(d, pot + amount, stack, stack - amount,
                                       0, amount, True, low, high)
                ev += prob * value
            else:
                rows = []
                for i, q in enumerate(d.q):
                    factor = sum(prob for low, high, prob in branches if low <= q < high)
                    w = (d.mass[i + 1] - d.mass[i]) * factor
                    wh = (d.wins[i + 1] - d.wins[i]) * factor
                    if w > 0:
                        rows.append((q, wh / w, w))
                mixed = RiverDistribution(rows)
                value, _, _ = self.node(mixed, pot + amount, stack, stack - amount,
                                       0, amount, True)
                ev += value
        return ev

    def start(self, d, pot, stack, in_position):
        if stack <= 0:
            return d.wins[-1] * pot
        if in_position:
            # Opponent acts first. A checked-to hero can bet with weak-range
            # information; the same visible bet-size branches as after_check.
            ev, _, _ = self.node(d, pot, stack, stack, low=0.0, high=0.62)
            checked_value = 0.92 * ev
            weak_wins = d.moments(0.0, 0.62)[1]
            return checked_value + self.after_check(d, pot, stack) - 0.92 * weak_wins * pot
        checked = self.after_check(d, pot, stack)
        ev, action, utility = self.node(d, pot, stack, stack, check_value=checked)
        return ev


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.pre = {}
        self.hand = None
        self.weights = {}
        self.tables = {}
        self.rivers = {}
        self.river_solver = RiverSolver(self.checkpoint)
        self.seen = 0
        self.diagnostics = Counter()

    def checkpoint(self):
        # ctx.done alone is not enough if the deadline monitor has not run yet.
        # Leave a small margin for unwinding before calls begin being refused.
        if self.ctx.done or self.ctx.budget.remaining()['wall_s'] <= 0.5:
            raise _StopEpisode()

    def pre_equity(self, h):
        self.checkpoint()
        key = klass(h)
        if key not in self.pre:
            # Canonical class simulation: independent of match/deal seeds.
            a, b, suited = key
            hole = ((a - 2) * 4, (b - 2) * 4 + (0 if suited else 1))
            deck = [c for c in DECK if c not in hole]
            rng = seeded(('preflop-equity', key))
            wins = 0.0
            for _ in range(2048):
                if _ % 32 == 0:
                    self.checkpoint()
                draw = rng.sample(deck, 7)
                s1 = score(hole + tuple(draw[2:]))
                s2 = score(draw)
                wins += (s1 > s2) + 0.5 * (s1 == s2)
            self.pre[key] = wins / 2048
        return self.pre[key]

    def table(self, board, hole):
        """All combo equities using common public runouts and exact river CDFs.

        The opponent's random-opponent equity does NOT remove hero's cards.
        Hero-versus-combo equity does remove both players' actual cards.
        """
        self.checkpoint()
        key = tuple(board)
        if key in self.tables:
            return self.tables[key]
        bset = set(board)
        available = [c for c in DECK if c not in bset]
        pairs = [p for p in COMBOS if not bset.intersection(p)]
        accum = {p: [0.0, 0, 0.0, 0] for p in pairs}
        rng = seeded(('public-board', board, hole))
        if len(board) == 5:
            runs = [()]
        elif len(board) == 4:
            runs = [(c,) for c in available]
        else:
            runs = [tuple(rng.sample(available, 2)) for _ in range(160)]
        for run in runs:
            self.checkpoint()
            dead = set(run)
            complete = tuple(board) + tuple(run)
            ranked = []
            for index, p in enumerate(pairs):
                if index % 32 == 0:
                    self.checkpoint()
                if not dead.intersection(p):
                    ranked.append((p, score(p + complete)))
            allranks = sorted(s for p, s in ranked)
            percard = [[] for _ in DECK]
            for index, ((a, b), s) in enumerate(ranked):
                if index % 32 == 0:
                    self.checkpoint()
                percard[a].append(s)
                percard[b].append(s)
            for values in percard:
                values.sort()
            hero_ok = not dead.intersection(hole)
            hs = score(tuple(hole) + complete) if hero_ok else 0
            river_slice = []
            for index, ((a, b), s) in enumerate(ranked):
                if index % 32 == 0:
                    self.checkpoint()
                # Remove both hole cards from the opposing combo distribution.
                below = bisect.bisect_left(allranks, s)
                below -= bisect.bisect_left(percard[a], s) + bisect.bisect_left(percard[b], s)
                upto = bisect.bisect_right(allranks, s)
                upto -= bisect.bisect_right(percard[a], s) + bisect.bisect_right(percard[b], s)
                upto += 1  # own combo was subtracted twice
                v = accum[(a, b)]
                v[0] += (below + upto) / 1980.0
                v[1] += 1
                if hero_ok and a not in hole and b not in hole:
                    v[2] += (hs > s) + 0.5 * (hs == s)
                    v[3] += 1
                    if len(board) == 4:
                        river_slice.append(((below + upto) / 1980.0, (a, b), (hs > s) + 0.5 * (hs == s)))
            if len(board) == 4 and hero_ok:
                self.rivers[run[0]] = tuple(sorted(river_slice))
        result = {p: (v[0] / max(1, v[1]), v[2] / max(1, v[3])) for p, v in accum.items()}
        self.tables[key] = result
        return result

    def update_range(self, o, hole, board):
        self.checkpoint()
        if self.hand != o.get('hand_id'):
            self.hand = o.get('hand_id')
            self.weights = {p: 1.0 for p in COMBOS if not set(hole).intersection(p)}
            self.tables = {}
            self.rivers = {}
            self.seen = 0
        history = o.get('history') or []
        hero = o['hero_seat']
        bets = [0, 0]
        last_street = None
        raised = [False, False]
        for i, event in enumerate(history):
            self.checkpoint()
            street = event.get('street', 'preflop')
            if street != last_street:
                bets = [int(o.get('big_blind', 100)), int(o.get('small_blind', 50))] if street == 'preflop' else [0, 0]
                raised = [False, False]
                last_street = street
            seat = event.get('seat')
            action = event.get('action')
            call = max(bets) - bets[seat]
            pot = max(1, event.get('pot_before', 1))
            if i >= self.seen and seat != hero:
                n = STREET_N.get(street, len(board))
                public = tuple(board[:n])
                table = self.table(public, hole) if n else None
                width = 0.045 if n == 3 else 0.028 if n == 4 else 0.012
                for index, p in enumerate(list(self.weights)):
                    if index % 32 == 0:
                        self.checkpoint()
                    if set(public).intersection(p):
                        del self.weights[p]
                        continue
                    eq = table[p][0] if table else self.pre_equity(p)
                    if n == 0:
                        if seat == 1 and not any(e.get('action') == 'raise_to' for e in history[:i]):
                            # Exact documented class set, with a small model-error floor.
                            like = 0.997 if opens(p) else 0.003
                        elif call == 0:
                            r = smooth(eq, 0.60, 0.022)
                            like = r if action == 'raise_to' else 1 - r
                        elif premium(p):
                            like = 0.995 if action == 'raise_to' else 0.005
                        else:
                            c = smooth(eq, max(0.44, call / (pot + call) + 0.08), 0.024)
                            like = c if action == 'call' else (1 - c) if action == 'fold' else 0.002
                    elif call == 0:
                        value = smooth(eq, 0.62, width)
                        if action == 'check':
                            like = 0.92 * (1 - value)
                        elif action == 'raise_to':
                            frac = float(event.get('cost') or 0) / pot
                            if frac >= 0.88:
                                like = smooth(eq, 0.85, width)
                            elif frac >= 0.61:
                                like = value * (1 - smooth(eq, 0.85, width))
                            else:
                                like = 0.08 * (1 - value)
                        else:
                            like = 0.002
                    else:
                        r = smooth(eq, 0.80, width) if not raised[seat] else 0.0
                        c = smooth(eq, call / (pot + call) + (0.0 if n == 5 else 0.05), width)
                        like = r if action == 'raise_to' else (1 - r) * c if action == 'call' else (1 - r) * (1 - c)
                    self.weights[p] *= max(0.003, like)
                total = sum(self.weights.values())
                if total:
                    self.weights = {p: w / total for p, w in self.weights.items()}
            if action == 'raise_to':
                bets[seat] = int(event.get('raise_to') or bets[seat] + event.get('cost', 0))
                raised[seat] = True
            elif action == 'call':
                bets[seat] += int(event.get('cost') or call)
        self.seen = len(history)
        self.weights = {p: w for p, w in self.weights.items() if not set(board).intersection(p)}

    def preflop(self, o, hole):
        self.checkpoint()
        q = self.pre_equity(hole)
        hero = o['hero_seat']
        call = int(o['to_call'])
        pot = int(o['pot'])
        bets = o['street_bets']
        history = o.get('history', [])
        other_raises = [e for e in history if e.get('seat') != hero and e.get('action') == 'raise_to']
        own_raises = [e for e in history if e.get('seat') == hero and e.get('action') == 'raise_to']
        # The opponent judges huge calls against random hands. Test deep value jams.
        if q >= 0.60 and o['legal'].get('can_raise'):
            target = o['legal']['max_raise_to']
            cost = target - bets[hero]
            opp_call = min(o['stacks'][1 - hero], target - bets[1 - hero])
            threshold = max(0.44, opp_call / (pot + cost + opp_call) + 0.08)
            pool, weights = [], []
            for index, (p, w) in enumerate(self.weights.items()):
                if index % 32 == 0:
                    self.checkpoint()
                prob = 1.0 if premium(p) else smooth(self.pre_equity(p), threshold, 0.024)
                pool.append(p)
                weights.append(w * prob)
            total = sum(self.weights.values()) or 1
            continue_p = sum(weights) / total
            rng = seeded(('preflop-jam', hole, tuple((e.get('seat'), e.get('raise_to')) for e in history)))
            wins = 0.0
            trials = 1400
            for index, p in enumerate(rng.choices(pool, weights=weights, k=trials)):
                if index % 32 == 0:
                    self.checkpoint()
                deck = [c for c in DECK if c not in hole and c not in p]
                b = tuple(rng.sample(deck, 5))
                a, z = score(tuple(hole) + b), score(p + b)
                wins += (a > z) + 0.5 * (a == z)
            equity = wins / trials
            ev = (1 - continue_p) * pot + continue_p * (equity * (pot + cost + opp_call) - cost)
            if equity >= 0.55 and ev > max(100, 0.18 * pot):
                self.diagnostics['preflop_value_jams'] += 1
                return legal_action(o, 'raise_to', target), {'q': q, 'jam_equity': equity, 'ev': ev}
        if other_raises and (own_raises or len(other_raises) > 1 or max(bets) > 600):
            # Premium-only reraise branch: preserve implied odds only at small prices.
            a, b, suited = klass(hole)
            if a == b and a == 14:
                return legal_action(o, 'raise_to', o['legal'].get('max_raise_to') or 20000), {'q': q}
            if call <= 650 and (a == b or suited and a >= 12):
                return legal_action(o, 'call'), {'q': q}
            return legal_action(o, 'fold'), {'q': q}
        if hero == o['button_seat'] and not own_raises and max(bets) <= 100:
            return legal_action(o, 'raise_to', 200) if q >= 0.365 else legal_action(o, 'fold'), {'q': q}
        if call == 0:
            return legal_action(o, 'raise_to', 400) if q >= 0.57 else 'check', {'q': q}
        odds = call / (pot + call)
        return legal_action(o, 'call' if q >= max(0.405, odds + 0.10) else 'fold'), {'q': q}

    def river_value(self, weights, pot, stack, in_position):
        """Expected future pot receipts, with one policy per exposed river.

        Each hypothetical holding leaves 44 rivers after hero+turn+villain card
        removal. Summing unnormalized branch mass / 44 handles card information
        correctly, including the river's effect on the opponent posterior.
        """
        self.checkpoint()
        if not weights or sum(weights.values()) <= 1e-12:
            return 0.0
        total = 0.0
        for river, values in self.rivers.items():
            self.checkpoint()
            d = RiverDistribution((q, h, weights.get(p, 0.0)) for q, p, h in values)
            if not d.q:
                continue
            total += self.river_solver.start(d, pot, stack, in_position)
            self.diagnostics['river_nodes'] += 1
        return total / 44.0

    def turn_response(self, rows, pot, call, hstack, vstack, hbet, vbet, in_position):
        """After an opponent turn bet/raise: one shared fold/call/jam decision.

        Opponent has used its single voluntary raise on this street, so a jam
        gets only a call/fold response. Rows have unnormalized branch weights.
        """
        self.checkpoint()
        mass = sum(w for p, w, e, h in rows)
        if mass <= 1e-12:
            return 0.0
        call = min(call, hstack)
        weights = {p: w for p, w, e, h in rows}
        remaining = max(0, min(hstack - call, vstack))
        value_call = self.river_value(weights, pot + call, remaining, in_position) - mass * call
        best_ev, utility = max(0.0, value_call), max(0.0, value_call)
        if hstack > call and vstack > 0:
            cost = min(hstack, call + vstack)
            vc = min(vstack, hbet + cost - vbet)
            odds = vc / (pot + cost + vc)
            ev = called = winning = 0.0
            for p, w, e, h in rows:
                c = smooth(e, odds + 0.05, 0.028)
                ev += w * ((1 - c) * pot + c * (h * (pot + cost + vc) - cost))
                called += w * c
                winning += w * c * h
            adjusted = ev - 0.055 * cost * called
            if (cost <= 3 * pot or winning / max(called, 1e-12) >= 0.61) and adjusted > utility:
                best_ev = ev
        return best_ev

    def turn_check(self, rows, pot, stack, in_position):
        if in_position:
            return self.river_value({p: w for p, w, e, h in rows}, pot, stack, True)
        check_weights = {}
        groups = {}
        for p, w, e, h in rows:
            low = 1 - smooth(e, 0.62, 0.028)
            high = smooth(e, 0.85, 0.028)
            mid = max(0.0, 1 - low - high)
            check_weights[p] = w * 0.92 * low
            for prob, fraction in ((0.08 * low, 0.5), (mid, 0.75), (high, 1.0)):
                bet = min(stack, max(100, int(fraction * pot + 0.5)))
                by_hand = groups.setdefault(bet, {})
                by_hand[p] = by_hand.get(p, 0.0) + w * prob
        ev = self.river_value(check_weights, pot, stack, False)
        for bet, weights in groups.items():
            if bet <= 0:
                continue
            branch = [(p, weights.get(p, 0.0), e, h) for p, w, e, h in rows]
            ev += self.turn_response(branch, pot + bet, bet, stack, stack - bet, 0, bet, False)
        return ev

    def turn_plan(self, o, hole, board):
        self.checkpoint()
        table = self.table(tuple(board), hole)
        total = sum(w for p, w in self.weights.items() if p in table) or 1.0
        rows = [(p, w / total, table[p][0], table[p][1]) for p, w in self.weights.items() if p in table]
        hero = o['hero_seat']
        ip = hero == o['button_seat']
        pot, call = int(o['pot']), int(o['to_call'])
        hb, vb = o['street_bets'][hero], o['street_bets'][1 - hero]
        hs, vs = o['stacks'][hero], o['stacks'][1 - hero]
        if call:
            passive = self.river_value({p: w for p, w, e, h in rows}, pot + call, min(hs - call, vs), ip) - call
            best_ev = max(0.0, passive)
            action = 'call' if passive > 0 else 'fold'
        else:
            passive = self.turn_check(rows, pot, min(hs, vs), ip)
            best_ev, action = passive, 'check'
        best_utility = best_ev
        details = {'equity': round(sum(w * h for p, w, e, h in rows), 4),
                   'future_passive_ev': round(passive, 2), 'horizon': 'river',
                   'range_combos': len(rows)}
        legal = o['legal']
        if legal.get('can_raise'):
            opp_raised = any(e.get('seat') != hero and e.get('street') == 'turn' and e.get('action') == 'raise_to' for e in o.get('history', []))
            for target in size_menu(pot, call, hb, vb, hs, vs, legal['min_raise_to'], legal['max_raise_to']):
                self.checkpoint()
                cost = min(target - hb, call + vs)
                vc = min(vs, target - vb)
                matched = pot + cost + vc
                odds = vc / max(1, matched)
                callers, raisers = {}, []
                fold_mass = called_mass = raise_mass = win_cont = 0.0
                for p, w, e, h in rows:
                    r = smooth(e, 0.80, 0.028) if not opp_raised and min(hs - cost, vs - vc) > 0 else 0.0
                    c = smooth(e, odds + 0.05, 0.028)
                    wc, wr = w * (1 - r) * c, w * r
                    callers[p] = wc
                    raisers.append((p, wr, e, h))
                    called_mass += wc
                    raise_mass += wr
                    win_cont += (wc + wr) * h
                    fold_mass += w - wc - wr
                ev = fold_mass * pot
                remaining = min(hs - cost, vs - vc)
                ev += self.river_value(callers, matched, remaining, ip) - cost * called_mass
                if raise_mass > 1e-12:
                    extra = min(remaining, max(target - max(hb, vb), int(0.75 * matched + 0.5)))
                    response = self.turn_response(raisers, matched + extra, extra, hs - cost, vs - vc - extra,
                                                  target, target + extra, ip)
                    ev += response - cost * raise_mass
                continued = called_mass + raise_mass
                ceq = win_cont / max(continued, 1e-12)
                utility = ev - 0.055 * cost * continued
                if cost > 3 * pot and ceq < 0.61:
                    continue
                if utility > best_utility + max(8.0, 0.025 * pot):
                    action, best_ev, best_utility = 'raise_to ' + str(target), ev, utility
                    details.update(future_bet_ev=round(ev, 2), called_equity=round(ceq, 4), continue_p=round(continued, 4))
        details['selected_ev'] = round(best_ev, 2)
        return legal_action(o, 'raise_to', int(action.split()[1])) if action.startswith('raise_to') else legal_action(o, action), details

    def river_plan(self, o, hole, board):
        table = self.table(tuple(board), hole)
        total = sum(w for p, w in self.weights.items() if p in table) or 1.0
        d = RiverDistribution(sorted((table[p][0], table[p][1], w / total) for p, w in self.weights.items() if p in table))
        hero = o['hero_seat']
        hb, vb = o['street_bets'][hero], o['street_bets'][1 - hero]
        hs, vs = o['stacks'][hero], o['stacks'][1 - hero]
        pot = o['pot']
        raised = any(e.get('seat') != hero and e.get('street') == 'river' and e.get('action') == 'raise_to' for e in o.get('history', []))
        check_value = None
        if not o['to_call'] and hero != o['button_seat']:
            check_value = self.river_solver.after_check(d, pot, min(hs, vs))
        bounds = (o['legal']['min_raise_to'], o['legal']['max_raise_to']) if o['legal'].get('can_raise') else None
        value, action, utility = self.river_solver.node(d, pot, hs, vs, hb, vb, raised, bounds=bounds,
                                                        check_value=check_value)
        return legal_action(o, 'raise_to', int(action.split()[1])) if action.startswith('raise_to') else legal_action(o, action), {
            'equity': round(d.wins[-1], 4), 'river_ev': round(value, 2), 'river_utility': round(utility, 2), 'horizon': 'terminal'}

    def postflop(self, o, hole, board):
        self.checkpoint()
        table = self.table(tuple(board), hole)
        rows = [(p, w, table[p][0], table[p][1]) for p, w in self.weights.items() if p in table]
        total = sum(w for p, w, e, h in rows) or 1.0
        rows = [(p, w / total, e, h) for p, w, e, h in rows]
        equity = sum(w * h for p, w, e, h in rows)
        pot, call = int(o['pot']), int(o['to_call'])
        hero = o['hero_seat']
        bets = o['street_bets']
        n = len(board)
        # Future-street realization haircut; no haircut on the terminal river.
        realization = 1.0 if n == 5 else 0.95 if hero == 1 else 0.91
        effective_eq = max(0.0, equity * realization - (0.018 if n < 5 else 0.0))
        passive_ev = effective_eq * (pot + call) - call
        baseline = max(0.0, passive_ev) if call else effective_eq * pot
        best = legal_action(o, 'call' if passive_ev > max(0, 0.012 * pot) else 'fold') if call else 'check'
        best_ev = baseline
        legal = o['legal']
        details = {'equity': round(equity, 4), 'passive_ev': round(passive_ev, 1)}
        if legal.get('can_raise'):
            lo, hi = legal['min_raise_to'], legal['max_raise_to']
            targets = {lo, hi}
            for frac in (0.25, 0.50, 0.75, 1.0, 1.5, 2.5, 4.0):
                targets.add(max(lo, min(hi, round(max(bets) + frac * (pot + call)))))
            width = 0.045 if n == 3 else 0.028 if n == 4 else 0.010
            opp_raised = any(e.get('seat') != hero and e.get('street') == o['street'] and e.get('action') == 'raise_to' for e in o.get('history', []))
            for target in sorted(targets):
                self.checkpoint()
                cost = target - bets[hero]
                oppcall = min(o['stacks'][1 - hero], max(0, target - bets[1 - hero]))
                # Refunded chips never belong in a betting EV.
                cost = min(cost, call + o['stacks'][1 - hero])
                odds = oppcall / max(1, pot + cost + oppcall)
                ev, called, win_called = 0.0, 0.0, 0.0
                raised_p, raised_wins = 0.0, 0.0
                for index, (p, w, e, h) in enumerate(rows):
                    if index % 32 == 0:
                        self.checkpoint()
                    r = smooth(e, 0.8, width) if not opp_raised and oppcall < o['stacks'][1 - hero] else 0.0
                    c = smooth(e, odds + (0.0 if n == 5 else 0.05), width)
                    cont = r + (1 - r) * c
                    response_ev = h * (pot + cost + oppcall) - cost
                    if n < 5:
                        response_ev -= (1 - h) * 0.035 * (pot + cost + oppcall)
                    ev += w * ((1 - cont) * pot + (1 - r) * c * response_ev)
                    raised_p += w * r
                    raised_wins += w * r * h
                    called += w * cont
                    win_called += w * cont * h
                if raised_p:
                    # One common future decision against the entire raising range.
                    # Never choose a different response for each hypothetical hole pair.
                    raise_equity = raised_wins / raised_p
                    extra = min(max(0, o['stacks'][hero] - cost),
                                max(0, o['stacks'][1 - hero] - oppcall),
                                int(0.75 * (pot + cost + oppcall) + 0.5))
                    raised_call = raise_equity * (pot + cost + oppcall + 2 * extra) - cost - extra
                    if n < 5:
                        raised_call -= (1 - raise_equity) * 0.035 * (pot + cost + oppcall + 2 * extra)
                    ev += raised_p * max(-cost, raised_call)
                eq_called = win_called / max(called, 1e-9)
                # Avoid committing 200bb on tiny MC edges; larger bets need clearer value.
                safety = (0.025 if n == 5 else 0.055) * cost * called
                if cost > 3 * pot and eq_called < (0.61 if n < 5 else 0.56):
                    continue
                adjusted = ev - safety
                if adjusted > best_ev + max(8, 0.025 * pot):
                    best_ev = adjusted
                    best = legal_action(o, 'raise_to', target)
                    details.update({'bet_ev': round(ev, 1), 'called_equity': round(eq_called, 4), 'continue_p': round(called, 4)})
        details['range_combos'] = len(rows)
        details['action'] = best
        return best, details

    def run_episode(self, game):
        try:
            self._run_episode(game)
        except (_StopEpisode, BudgetExceeded):
            # Includes EpisodeOver and a deadline racing with observe/log/act.
            return
        finally:
            self.ctx.memory['diagnostics'] = dict(self.diagnostics)

    def _run_episode(self, game):
        self.checkpoint()
        obs = self.ctx.observe()
        while not self.ctx.done and not obs.get('done'):
            self.checkpoint()
            hole = tuple(card(s) for s in obs['hole_cards'])
            board = tuple(card(s) for s in obs.get('board', []))
            self.update_range(obs, hole, board)
            self.checkpoint()
            if not board:
                action, details = self.preflop(obs, hole)
            elif len(board) == 4:
                action, details = self.turn_plan(obs, hole, board)
            elif len(board) == 5:
                action, details = self.river_plan(obs, hole, board)
            else:
                action, details = self.postflop(obs, hole, board)
            self.checkpoint()
            self.ctx.log('range_decision', hand=obs.get('hand_id'), street=obs.get('street'), action=action, **{k:v for k,v in details.items() if k != 'action'})
            obs = self.ctx.act(action)
            if obs is None:
                obs = self.ctx.observe()


def build(ctx):
    return Harness(ctx)
