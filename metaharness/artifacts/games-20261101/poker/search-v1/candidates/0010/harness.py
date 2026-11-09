"""Cached-equity posterior and public-action exploitative value sizing; stdlib only."""
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


# Rank-mask tables: exact base-15 kicker encodings, shared by every board.
TOP5 = [0] * 8192
TOP3 = [0] * 8192
TOP2 = [0] * 8192
for _mask in range(8192):
    _rs = [r + 2 for r in range(12, -1, -1) if _mask & (1 << r)]
    for _n, _table in ((2, TOP2), (3, TOP3), (5, TOP5)):
        _v = 0
        for _r in (_rs[:_n] + [0] * _n)[:_n]:
            _v = _v * 15 + _r
        _table[_mask] = _v
CARD_BITS = tuple(1 << (c // 4) for c in DECK)


def mask_score(a, b, c, d):
    """Exact best-five rank from four suit rank masks; larger wins."""
    ranks = a | b | c | d
    flush = a if a.bit_count() >= 5 else b if b.bit_count() >= 5 else c if c.bit_count() >= 5 else d if d.bit_count() >= 5 else 0
    if flush and STRAIGHT[flush]:
        return 8 * 759375 + STRAIGHT[flush] * 50625
    quad = a & b & c & d
    if quad:
        q = quad.bit_length() + 1
        return 7 * 759375 + q * 50625 + ((ranks ^ quad).bit_length() + 1) * 3375
    trips = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d)
    pairs = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d)
    if trips:
        tbit = 1 << (trips.bit_length() - 1)
        other = pairs ^ tbit
        if other:
            return 6 * 759375 + (trips.bit_length() + 1) * 50625 + (other.bit_length() + 1) * 3375
    if flush:
        return 5 * 759375 + TOP5[flush]
    straight = STRAIGHT[ranks]
    if straight:
        return 4 * 759375 + straight * 50625
    if trips:
        return 3 * 759375 + (trips.bit_length() + 1) * 50625 + TOP2[ranks ^ tbit] * 225
    if pairs:
        pbit = 1 << (pairs.bit_length() - 1)
        rest = pairs ^ pbit
        if rest:
            p2bit = 1 << (rest.bit_length() - 1)
            return 2 * 759375 + (pairs.bit_length() + 1) * 50625 + (rest.bit_length() + 1) * 3375 + ((ranks ^ pbit ^ p2bit).bit_length() + 1) * 225
        return 759375 + (pairs.bit_length() + 1) * 50625 + TOP3[ranks ^ pbit] * 15
    return TOP5[ranks]


def score(cards):
    suits = [0, 0, 0, 0]
    for c in cards:
        suits[c & 3] |= CARD_BITS[c]
    return mask_score(*suits)


def board_scorer(board):
    """Amortize the five common board cards across all 1081 hole pairs."""
    suits = [0, 0, 0, 0]
    for c in board:
        suits[c & 3] |= CARD_BITS[c]
    a, b, c, d = suits
    def rank_pair(pair):
        x, y = pair
        sx, sy = x & 3, y & 3
        bx, by = CARD_BITS[x], CARD_BITS[y]
        return mask_score(a | (bx if sx == 0 else 0) | (by if sy == 0 else 0),
                          b | (bx if sx == 1 else 0) | (by if sy == 1 else 0),
                          c | (bx if sx == 2 else 0) | (by if sy == 2 else 0),
                          d | (bx if sx == 3 else 0) | (by if sy == 3 else 0))
    return rank_pair


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


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.pre = {}
        self.hand = None
        self.weights = {}
        self.tables = {}
        self.table_variance = {}
        self.latent_board = None
        self.latent = {}
        self.segments = [(0.0, 1.000001, 1.0)]
        self.seen = 0
        self.diagnostics = Counter()
        self.step_reserve = 20.0

    def checkpoint(self):
        # ctx.done alone is not enough if the deadline monitor has not run yet.
        # Leave a small margin for unwinding before calls begin being refused.
        if self.ctx.done or self.ctx.budget.remaining()['wall_s'] <= 1.0:
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
        accum = {p: [0.0, 0, 0.0, 0, 0.0] for p in pairs}
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
            rank_pair = board_scorer(complete)
            ranked = []
            for index, p in enumerate(pairs):
                if index % 32 == 0:
                    self.checkpoint()
                if not dead.intersection(p):
                    ranked.append((p, rank_pair(p)))
            # One strength-ordered sweep supplies all blocker-corrected CDFs.
            # The parent independently bisected 53 sorted lists for every pair.
            ranked.sort(key=lambda row: row[1])
            below_card = [0] * 52
            hero_ok = not dead.intersection(hole)
            hs = rank_pair(hole) if hero_ok else 0
            i = 0
            while i < len(ranked):
                self.checkpoint()
                j = i + 1
                strength = ranked[i][1]
                while j < len(ranked) and ranked[j][1] == strength:
                    j += 1
                tied_card = [0] * 52
                for k in range(i, j):
                    a, b = ranked[k][0]
                    tied_card[a] += 1
                    tied_card[b] += 1
                for k in range(i, j):
                    (a, b), strength = ranked[k]
                    below = i - below_card[a] - below_card[b]
                    tied = j - i - tied_card[a] - tied_card[b] + 1
                    q = (2 * below + tied) / 1980.0
                    v = accum[(a, b)]
                    v[4] += q * q
                    v[0] += q
                    v[1] += 1
                    if hero_ok and a not in hole and b not in hole:
                        h = (hs > strength) + 0.5 * (hs == strength)
                        v[2] += h
                        v[3] += 1
                for k in range(i, j):
                    a, b = ranked[k][0]
                    below_card[a] += 1
                    below_card[b] += 1
                i = j
        result = {p: (v[0] / max(1, v[1]), v[2] / max(1, v[3])) for p, v in accum.items()}
        self.table_variance[key] = {p: max(0.0, v[4] / max(1, v[1]) - result[p][0] ** 2) / max(1, v[1] - 1)
                                    if len(board) == 3 else 0.0 for p, v in accum.items()}
        self.tables[key] = result
        return result

    def ensure_latent(self, board, hole):
        """One noisy equity draw per holding and street, never per decision."""
        board = tuple(board)
        if self.latent_board == board:
            return
        table = self.table(board, hole)
        variance = self.table_variance[board]
        self.latent_board = board
        self.segments = [(0.0, 1.000001, 1.0)]
        self.latent = {}
        for p, (q, _) in table.items():
            # Sampling variance bound (ties reduce it) plus our flop integration
            # error. Turn integrates all rivers, and river equity is exact.
            sd = math.sqrt(q * (1.0 - q) / 512.0 + variance[p]) if len(board) < 5 else 0.0
            self.latent[p] = (q, sd)
        self.normalizers = {p: self.segment_mass(p, self.segments) for p in table}

    def interval_mass(self, p, low, high):
        if high <= low:
            return 0.0
        q, sd = self.latent[p]
        if sd < 1e-12:
            return float(low <= q < high)
        scale = sd * 1.4142135623730951
        a, b = (low - q) / scale, (high - q) / scale
        # erfc avoids cancellation in either tail of a normal distribution.
        if a >= 0:
            return max(0.0, 0.5 * (math.erfc(a) - math.erfc(b)))
        return max(0.0, 0.5 * (math.erfc(-b) - math.erfc(-a)))

    def segment_mass(self, p, segments, low=0.0, high=1.000001):
        return sum(f * self.interval_mass(p, max(a, low), min(b, high))
                   for a, b, f in segments if min(b, high) > max(a, low))

    def latent_mass(self, p, low, high):
        denom = self.normalizers.get(p, 0.0)
        if denom <= 1e-200:
            return 0.0
        return min(1.0, self.segment_mass(p, self.segments, low, high) / denom)

    def condition_latent(self, support):
        """Intersect public-action intervals with the previous same-street belief."""
        new = [(max(a, x), min(b, y), f * z)
               for a, b, f in self.segments for x, y, z in support
               if min(b, y) > max(a, x) and f * z > 0]
        if not new:
            self.diagnostics['latent_support_conflicts'] += 1
            # Model mismatch: discard stale latent constraints, never crash or
            # manufacture a private holding. Keep the last holding posterior.
            new = support or [(0.0, 1.000001, 1.0)]
            self.normalizers = {p: self.interval_mass(p, 0.0, 1.000001) for p in self.latent}
        updated = {}
        norms = {}
        for index, (p, w) in enumerate(self.weights.items()):
            if index % 64 == 0:
                self.checkpoint()
            if p not in self.latent:
                continue
            mass = self.segment_mass(p, new)
            norms[p] = mass
            denom = self.normalizers.get(p, 0.0)
            updated[p] = w * min(1.0, mass / denom) if denom > 1e-200 else 0.0
        total = sum(updated.values())
        if total <= 1e-150:
            # This means every plausible holding contradicted an observation.
            # Recover without treating the failed posterior as evidence.
            self.diagnostics['latent_zero_evidence'] += 1
            self.segments = [(0.0, 1.000001, 1.0)]
            self.normalizers = {p: self.interval_mass(p, 0.0, 1.000001) for p in self.latent}
            return
        self.weights = {p: w / total for p, w in updated.items()}
        self.segments = new
        self.normalizers = norms
        self.diagnostics['latent_observations'] += 1

    def update_range(self, o, hole, board):
        self.checkpoint()
        if self.hand != o.get('hand_id'):
            self.hand = o.get('hand_id')
            self.weights = {p: 1.0 for p in COMBOS if not set(hole).intersection(p)}
            self.tables = {}
            self.table_variance = {}
            self.latent_board = None
            self.latent = {}
            self.seen = 0
        history = o.get('history') or []
        hero = o['hero_seat']
        bb, sb = int(o.get('big_blind', 100)), int(o.get('small_blind', 50))
        starting = o.get('starting_stacks', [20000, 20000])
        stacks = [starting[0] - bb, starting[1] - sb]
        bets, raised, last_street = [bb, sb], [False, False], 'preflop'
        increment = bb
        for i, event in enumerate(history):
            self.checkpoint()
            street = event.get('street', 'preflop')
            if street != last_street:
                bets, raised, increment = [0, 0], [False, False], bb
                last_street = street
            seat = event['seat']
            action = event['action']
            call = min(stacks[seat], max(bets) - bets[seat])
            pot = max(1, event.get('pot_before', 1))
            if i >= self.seen and seat != hero:
                n = STREET_N.get(street, len(board))
                public = tuple(board[:n])
                if not n:
                    for index, p in enumerate(list(self.weights)):
                        if index % 32 == 0:
                            self.checkpoint()
                        eq = self.pre_equity(p)
                        if seat == 1 and not any(e.get('action') == 'raise_to' for e in history[:i]):
                            like = 0.997 if opens(p) else 0.003
                        elif call == 0:
                            r = smooth(eq, 0.60, 0.022)
                            like = r if action == 'raise_to' else 1 - r
                        elif premium(p):
                            like = 0.995 if action == 'raise_to' else 0.005
                        else:
                            c = smooth(eq, max(0.44, call / (pot + call) + 0.08), 0.024)
                            like = c if action == 'call' else (1 - c) if action == 'fold' else 0.002
                        self.weights[p] *= max(0.003, like)
                    total = sum(self.weights.values())
                    if total:
                        self.weights = {p: w / total for p, w in self.weights.items()}
                else:
                    self.ensure_latent(public, hole)
                    can_raise = stacks[seat] > call and stacks[1 - seat] > 0
                    support = []
                    if call == 0:
                        if action == 'check':
                            support = [(0.0, 0.62, 0.92)] if can_raise else [(0.0, 1.000001, 1.0)]
                        elif action == 'raise_to':
                            # Match integer size after engine min/max clipping;
                            # several equity branches can emit the same action.
                            minimum = max(bets) + increment
                            maximum = bets[seat] + stacks[seat]
                            minimum = min(minimum, maximum)
                            for low, high, prob, fraction in ((0.0, 0.62, 0.08, 0.5), (0.62, 0.85, 1.0, 0.75), (0.85, 1.000001, 1.0, 1.0)):
                                target = max(minimum, min(maximum, int(max(bets) + fraction * (pot + call) + 0.5)))
                                if target == event.get('raise_to'):
                                    support.append((low, high, prob))
                    else:
                        cutoff = call / (pot + call) + (0.0 if n == 5 else 0.05)
                        cap = 0.80 if can_raise and not raised[seat] else 1.000001
                        if action == 'raise_to' and cap == 0.80:
                            support = [(0.80, 1.000001, 1.0)]
                        elif action == 'call':
                            support = [(cutoff, cap, 1.0)]
                        elif action == 'fold':
                            support = [(0.0, min(cutoff, cap), 1.0)]
                    if not support:
                        self.diagnostics['unmodeled_public_action'] += 1
                    else:
                        self.condition_latent(support)
            if action == 'raise_to':
                target = int(event.get('raise_to') or bets[seat] + event.get('cost', 0))
                increment = max(increment, target - max(bets))
                bets[seat] = target
                raised[seat] = True
            elif action == 'call':
                bets[seat] += int(event.get('cost') or call)
            stacks[seat] -= int(event.get('cost') or 0)
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

    def postflop(self, o, hole, board):
        self.checkpoint()
        self.ensure_latent(tuple(board), hole)
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
        details = {'equity': round(equity, 4), 'passive_ev': round(passive_ev, 1),
                   'latent_support': [(round(a, 5), round(b, 5)) for a, b, f in self.segments]}
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
                    can_reraise = not opp_raised and oppcall < o['stacks'][1 - hero] and cost < o['stacks'][hero]
                    r = self.latent_mass(p, 0.8, 1.000001) if can_reraise else 0.0
                    cp = self.latent_mass(p, odds + (0.0 if n == 5 else 0.05), 0.8 if can_reraise else 1.000001)
                    cont = r + cp
                    response_ev = h * (pot + cost + oppcall) - cost
                    if n < 5:
                        response_ev -= (1 - h) * 0.035 * (pot + cost + oppcall)
                    ev += w * ((1 - cont) * pot + cp * response_ev)
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
            else:
                action, details = self.postflop(obs, hole, board)
            self.checkpoint()
            self.ctx.log('range_decision', hand=obs.get('hand_id'), street=obs.get('street'), action=action, **{k:v for k,v in details.items() if k != 'action'})
            # A turn-game step includes opponent equity work. Do not start one
            # at the deadline boundary (0009's archived failure).
            if self.ctx.budget.remaining()['wall_s'] < self.step_reserve:
                return
            before_step = self.ctx.budget.remaining()['wall_s']
            obs = self.ctx.act(action)
            step_time = before_step - self.ctx.budget.remaining()['wall_s']
            self.step_reserve = max(self.step_reserve, 2.0 * step_time + 3.0)
            if obs is None:
                obs = self.ctx.observe()


def build(ctx):
    return Harness(ctx)
