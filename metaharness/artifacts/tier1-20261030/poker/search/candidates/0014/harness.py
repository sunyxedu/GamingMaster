"""Coherent cached-equity river inference and integer-size response search."""
from __future__ import annotations
import bisect
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter
from metaharness.core.contract import BudgetExceeded

# Standard library only. Cards are rank-major integers; no game imports or state.
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = tuple(range(52))
STREETS = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}


class _StopEpisode(Exception):
    """Local search should yield before the evaluator's deadline grace period."""

def card(c):
    return RANKS.index(c[0]) * 4 + SUITS.index(c[1])

def straight(mask):
    # Wheel requires A,2,3,4,5; never invent a deuce from an ace.
    for hi in range(12, 3, -1):
        if mask & (31 << (hi - 4)) == (31 << (hi - 4)):
            return hi + 2
    if mask & 0x100F == 0x100F:
        return 5
    return 0

def rank(cards):
    """Exact best-five rank of 5--7 cards, higher tuples win."""
    counts = [0] * 15
    suits = [0] * 4
    mask = 0
    for c in cards:
        r = c // 4 + 2
        counts[r] += 1
        suits[c % 4] |= 1 << (r - 2)
        mask |= 1 << (r - 2)
    flushmask = next((m for m in suits if m.bit_count() >= 5), 0)
    if flushmask:
        st = straight(flushmask)
        if st:
            return (8, st)
    quads = [r for r in range(14, 1, -1) if counts[r] == 4]
    trips = [r for r in range(14, 1, -1) if counts[r] >= 3]
    pairs = [r for r in range(14, 1, -1) if counts[r] >= 2]
    if quads:
        return (7, quads[0], max(r for r in range(2, 15) if counts[r] and r != quads[0]))
    if trips and len(pairs) >= 2:
        return (6, trips[0], next(r for r in pairs if r != trips[0]))
    if flushmask:
        return (5,) + tuple(r + 2 for r in range(12, -1, -1) if flushmask & (1 << r))[:5]
    st = straight(mask)
    if st:
        return (4, st)
    if trips:
        return (3, trips[0]) + tuple(r for r in range(14, 1, -1) if counts[r] and r != trips[0])[:2]
    if len(pairs) >= 2:
        return (2, pairs[0], pairs[1], max(r for r in range(2, 15) if counts[r] and r not in pairs[:2]))
    if pairs:
        return (1, pairs[0]) + tuple(r for r in range(14, 1, -1) if counts[r] and r != pairs[0])[:3]
    return (0,) + tuple(r for r in range(14, 1, -1) if counts[r])[:5]

def compare(a, b):
    return 1.0 if a > b else 0.5 if a == b else 0.0

def sigmoid(x):
    return 1 / (1 + math.exp(-max(-30, min(30, x))))

def pre_eq(h):
    a, b = sorted((c // 4 + 2 for c in h), reverse=True)
    if a == b:
        return .50 + .025 * (a - 2)
    # Smooth uniform-hand equity approximation, not a solver chart.
    eq = .27 + .020 * (a - 2) + .013 * (b - 2)
    if h[0] % 4 == h[1] % 4:
        eq += .035
    if a - b <= 1:
        eq += .025
    elif a - b == 2:
        eq += .012
    if a == 14:
        eq += .025
    return min(.72, eq)

def premium(h):
    a, b = sorted((c // 4 + 2 for c in h), reverse=True)
    return (a == b and a >= 12) or (a == 14 and b == 13)

def open_weight(h):
    a, b = sorted((c // 4 + 2 for c in h), reverse=True)
    if a == b or a >= 12 or a == 11 and b >= 5 or a == 10 and b >= 7:
        return 1.
    if h[0] % 4 == h[1] % 4 or a - b <= 2 and a >= 7:
        return .95
    return .12

def fallback(o):
    ids = [c['id'] for c in o.get('candidates', [])]
    for a in ('check', 'fold', 'call'):
        if a in ids:
            return a
    return ids[0] if ids else 'check'

def legalize(o, a):
    if isinstance(a, dict):
        a = 'raise_to ' + str(a.get('amount')) if a.get('action') == 'raise_to' else a.get('action')
    if not isinstance(a, str):
        return None
    a = a.strip()
    if a in [c['id'] for c in o.get('candidates', [])]:
        return a
    m = re.fullmatch(r'raise_to (\d+)', a)
    l = o.get('legal', {})
    if m and l.get('can_raise') and l.get('min_raise_to') <= int(m[1]) <= l.get('max_raise_to'):
        return a
    return None

class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.rng = random.Random(738139)
        self.hand = None
        self.hole = ()
        self.particles = []
        self.eq_cache = {}
        self.history_seen = 0
        self.board_seen = ()
        self.mode = 'balanced'
        self.calls = 0
        self.last_review = -30
        self.last_audit_hand = None
        self.pending_failure = None
        self.stats = Counter()
        self.recent = []
        self.last_result = None
        self.last_predictions = []
        self.last_illegal = 0
        self.table = []
        self.audit_future = None
        self.river_certificate = {}

    def should_stop(self):
        return self.ctx.done or self.ctx.budget.remaining().get('wall_s', 0) <= 15

    def equity(self, h, board):
        key = (tuple(sorted(h)), tuple(board))
        if key in self.eq_cache:
            return self.eq_cache[key]
        if not board:
            return pre_eq(h)
        # Deterministic local sampling is independent of deals and opponent RNG.
        rng = random.Random(int.from_bytes(hashlib.blake2s(bytes(h + tuple(board)), digest_size=8).digest(), 'big'))
        deck = [c for c in DECK if c not in h and c not in board]
        n = 64 if len(board) < 5 else 120
        fixed = rank(h + tuple(board)) if len(board) == 5 else None
        total = 0.
        for i in range(n):
            if i % 16 == 0 and self.should_stop():
                raise _StopEpisode
            draw = rng.sample(deck, 7 - len(board))
            full = tuple(board) + tuple(draw[2:])
            total += compare(fixed if fixed is not None else rank(h + full), rank(tuple(draw[:2]) + full))
        q = total / n
        self.eq_cache[key] = q
        return q

    def new_hand(self, o):
        self.hand = o['hand_id']
        self.hole = tuple(card(c) for c in o['hole_cards'])
        self.eq_cache = {}
        self.history_seen = 0
        self.board_seen = ()
        combos = list(itertools.combinations([c for c in DECK if c not in self.hole], 2))
        # Stratified support: all combos for preflop filtering; later compress.
        self.particles = [[h, 1.] for h in combos]

    def likelihood(self, h, ev, board, current, own_raises):
        act = ev['action']
        q = self.equity(h, board)
        p = max(1, ev.get('pot_before', 1))
        cost = ev.get('cost', 0)
        seat = ev['seat']
        price = max(current.values(), default=0) - current.get(seat, 0)
        street = ev['street']
        if street == 'preflop':
            if act == 'raise_to':
                if price <= 50 and seat == 1:
                    return open_weight(h)
                if price == 0:
                    return .02 + .96 * sigmoid((q - .60) / .035)
                return .985 if premium(h) else .008
            if act == 'call':
                threshold = max(.44, price / (p + price) + .08)
                return (.02 if premium(h) and price > 100 else 1.) * (.02 + .96 * sigmoid((q - threshold) / .035))
            if act == 'check':
                return .02 + .96 * sigmoid((.60 - q) / .035)
            return .5
        if act == 'raise_to':
            if price > 0:
                return .015 + .97 * sigmoid((q - .80) / .055)
            fraction = cost / p
            if fraction >= .9:
                return .012 + .97 * sigmoid((q - .85) / .045)
            if fraction >= .62:
                return .015 + .97 * sigmoid((q - .62) / .055) * sigmoid((.87 - q) / .04)
            # Mixed half-pot branch remains a plausible bluff, never zero support.
            return .025 + .08 * sigmoid((.65 - q) / .05)
        if act == 'check':
            return .02 + .90 * sigmoid((.62 - q) / .05)
        if act == 'call':
            threshold = price / (p + price) + (0 if street == 'river' else .05)
            out = .015 + .97 * sigmoid((q - threshold) / .055)
            # A very strong hand usually raises once on a street when able.
            if own_raises == 0:
                out *= .12 + .88 * sigmoid((.80 - q) / .05)
            return out
        return .5

    def update_range(self, o):
        if self.hand != o.get('hand_id'):
            self.new_hand(o)
        board = tuple(card(c) for c in o.get('board', []))
        if len(board) == 5:
            self.rebuild_river(o, board)
            return
        # Full history replay computes action context without hidden state.
        events = o.get('history', [])
        bets = {0: 0, 1: 0}
        street = None
        raises = {0: 0, 1: 0}
        for i, ev in enumerate(events):
            if self.should_stop():
                raise _StopEpisode
            if ev.get('street') != street:
                street = ev.get('street')
                bets = {0: 100, 1: 50} if street == 'preflop' else {0: 0, 1: 0}
                raises = {0: 0, 1: 0}
            seat = ev.get('seat')
            if i >= self.history_seen and seat != o['hero_seat']:
                b = board[:STREETS.get(street, 0)]
                for j, row in enumerate(self.particles):
                    if j % 32 == 0 and self.should_stop():
                        raise _StopEpisode
                    if not any(c in board for c in row[0]):
                        row[1] *= self.likelihood(row[0], ev, b, bets, raises[seat])
                self.stats['opponent_' + str(ev.get('action'))] += 1
            if ev.get('action') == 'raise_to':
                bets[seat] = int(ev.get('raise_to') or 0)
                raises[seat] += 1
            elif ev.get('action') == 'call':
                bets[seat] += ev.get('cost', 0)
        self.history_seen = len(events)
        self.particles = [row for row in self.particles if not any(c in board for c in row[0])]
        z = sum(row[1] for row in self.particles)
        if z <= 1e-30 or not self.particles:
            self.pending_failure = {'reason': 'range support collapsed', 'board': o.get('board')}
            self.particles = [[h, 1.] for h in itertools.combinations([c for c in DECK if c not in self.hole and c not in board], 2)]
            z = len(self.particles)
        for row in self.particles:
            row[1] /= z
        # Sampling after preflop preserves premium combos without enumerating a
        # nested Monte Carlo over every combo on every board.
        if board and len(self.particles) > 200:
            sampled = self.rng.choices([r[0] for r in self.particles], weights=[r[1] for r in self.particles], k=240)
            counts = Counter(sampled)
            self.particles = [[h, n / 240.] for h, n in counts.items()]
        self.board_seen = board

    def rebuild_river(self, o, board):
        """Restore all 990 legal holes, replay public history, exact river equity.

        A compressed flop particle set can never recover a missing river hand.
        Rebuild only at the terminal street, where ranks cost no runout samples.
        Earlier street likelihoods remain explicitly approximate.
        """
        key = ('complete_river', board)
        if key not in self.eq_cache:
            available = [c for c in DECK if c not in board]
            ranks = {}
            by_card = {c: [] for c in available}
            for i, h in enumerate(itertools.combinations(available, 2)):
                if i % 32 == 0 and self.should_stop():
                    raise _StopEpisode
                r = rank(h + board)
                ranks[h] = r
                by_card[h[0]].append(r)
                by_card[h[1]].append(r)
            ordered = sorted(ranks.values())
            for values in by_card.values():
                values.sort()
            def score(values, r):
                return .5 * (bisect.bisect_left(values, r) + bisect.bisect_right(values, r))
            hr = rank(self.hole + board)
            rows = []
            for h, r in ranks.items():
                if any(c in self.hole for c in h):
                    continue
                # Hypothetical opponent knows its holes and board only.  Hero's
                # cards remain in its possible opponents, not secretly removed.
                q = (score(ordered, r) - score(by_card[h[0]], r)
                     - score(by_card[h[1]], r) + .5) / 990.
                self.eq_cache[(tuple(sorted(h)), board)] = q
                rows.append((h, compare(hr, r)))
            self.eq_cache[key] = rows
        support = self.eq_cache[key]
        weights = [1.] * len(support)
        bets = {0: 0, 1: 0}
        street = None
        raises = {0: 0, 1: 0}
        for ev in o.get('history', []):
            if self.should_stop():
                raise _StopEpisode
            if ev.get('street') != street:
                street = ev.get('street')
                bets = {0: 100, 1: 50} if street == 'preflop' else {0: 0, 1: 0}
                raises = {0: 0, 1: 0}
            seat = ev.get('seat')
            if seat != o['hero_seat']:
                b = board[:STREETS.get(street, 0)]
                for i, (h, win) in enumerate(support):
                    if i % 32 == 0 and self.should_stop():
                        raise _StopEpisode
                    weights[i] *= self.likelihood(h, ev, b, bets, raises[seat])
                z = sum(weights)
                if z > 0:
                    weights = [w/z for w in weights]
            if ev.get('action') == 'raise_to':
                bets[seat] = int(ev.get('raise_to') or 0)
                raises[seat] += 1
            elif ev.get('action') == 'call':
                bets[seat] += ev.get('cost', 0)
        z = sum(weights) or 1.
        self.particles = [[h, w/z] for (h, win), w in zip(support, weights)]
        self.board_seen = board
        self.history_seen = len(o.get('history', []))
        self.river_certificate = {
            'legal_combos': len(support),
            'better_combos': sum(win == 0 for h, win in support),
            'equity': round(sum(win*w/z for (h, win), w in zip(support, weights)), 4),
            'effective_combos': round(1 / max(1e-12, sum((w/z)**2 for w in weights)), 1)
        }

    def river_choose(self, o):
        """Terminal fold/call/bet/raise EV over the reconstructed full range."""
        board = self.board_seen
        hero = o['hero_seat']; villain = 1 - hero
        p = float(o['pot']); c = float(o['to_call'])
        hb, vb = o['street_bets'][hero], o['street_bets'][villain]
        hs, vs = o['stacks'][hero], o['stacks'][villain]
        hr = rank(self.hole + board)
        rows = [(w, self.equity(h, board), compare(hr, rank(h + board))) for h, w in self.particles]
        eq = sum(w*win for w, q, win in rows)
        choices = [{'action':'fold', 'ev':0.}, {'action':'call', 'ev':eq*(p+c)-c}] if c else [{'action':'check', 'ev':eq*p}]
        opp_raises = sum(e.get('seat') == villain and e.get('street') == 'river' and e.get('action') == 'raise_to' for e in o.get('history', []))
        legal = o['legal']
        if legal.get('can_raise'):
            maxbet = max(hb, vb)
            cap = min(legal['max_raise_to'], max(legal['min_raise_to'], maxbet + int(1.25*(p+c))))
            targets = {legal['min_raise_to'], cap}
            for f in (.33, .5, .75, 1., 1.25):
                targets.add(max(legal['min_raise_to'], min(cap, maxbet + int(f*(p+c)))))
            # Preserve the parent's monster eligibility; require the complete
            # caller distribution to certify any amount beyond the ordinary cap.
            strong_private = (hr[0] >= 6 or
                              (hr[0] == 5 and len(hr) > 1 and hr[1] >= 13))
            if strong_private and hr > rank(board) and eq >= .93:
                targets.add(legal['max_raise_to'])
            for target in sorted(targets):
                if self.should_stop():
                    raise _StopEpisode
                effective_target = min(target, vb+vs, hb+hs)
                cost = effective_target-hb
                oc = effective_target-vb
                if cost <= c or oc <= 0:
                    continue
                odds = oc/(p+cost+oc)
                folds = calls_ev = raised_mass = raised_wins = 0.
                calling_mass = calling_wins = 0.
                for w, q, win in rows:
                    # River equity is exact; small smoothing represents policy
                    # uncertainty, not Monte Carlo error in the rank table.
                    pr = .985*sigmoid((q-.80)/.012) if opp_raises == 0 and vs > oc and hs > cost else 0.
                    cp = .015 + .97*sigmoid((q-odds)/.012)
                    pc = (1-pr)*cp
                    pf = 1-pr-pc
                    folds += w*pf
                    calls_ev += w*pc*(win*(p+cost+oc)-cost)
                    raised_mass += w*pr
                    raised_wins += w*pr*win
                    calling_mass += w*cp
                    calling_wins += w*cp*win
                # One continuation decision for the whole observed raise branch;
                # never choose fold/call separately using hidden opponent holes.
                reply = max(effective_target + max(100, effective_target-maxbet),
                            int(effective_target + .75*(p+cost+oc) + .5))
                reply = min(reply, vb+vs, hb+hs)
                final_cost, final_oc = reply-hb, reply-vb
                raise_ev = max(-cost*raised_mass,
                               raised_wins*(p+final_cost+final_oc)-raised_mass*final_cost)
                called_eq = calling_wins/max(1e-12, calling_mass)
                if target > cap and called_eq < .97:
                    continue
                ev = folds*p + calls_ev + raise_ev - .025*cost
                choices.append({'action':'raise_to '+str(target), 'ev':ev,
                                'fold':round(folds,3), 'called_equity':round(called_eq,3),
                                'raise_mass':round(raised_mass,3)})
        self.table = sorted(choices, key=lambda r:r['ev'], reverse=True)
        best = self.table[0]['action']
        # The river repair is not allowed to create a silent strategy failure.
        self.ctx.log('river_endgame', certificate=self.river_certificate,
                     action=best, alternatives=self.table[:4])
        return best

    def preflop(self, o):
        self.table = []
        h = self.hole
        a, b = sorted((c // 4 + 2 for c in h), reverse=True)
        q = pre_eq(h)
        price = o['to_call']
        legal = o['legal']
        bets = o['street_bets']
        maxbet = max(bets)
        hero = o['hero_seat']
        history = o.get('history', [])
        opp_raises = [e for e in history if e['seat'] != hero and e['action'] == 'raise_to']
        def raise_to(n):
            if not legal.get('can_raise'):
                return 'call' if legal.get('can_call') else 'check'
            return 'raise_to ' + str(max(legal['min_raise_to'], min(legal['max_raise_to'], int(n))))
        # Against the documented premium reraises, use actual QQ+/AK equity.
        if maxbet >= 700 or len(opp_raises) >= 2:
            if a == b and a >= 13:
                return raise_to(legal.get('max_raise_to', maxbet))
            if price <= 900 and (a == b or a >= 13 and b >= 11 and h[0] % 4 == h[1] % 4):
                return 'call'
            if a == 14 and b == 13 and price <= 1500:
                return 'call'
            return fallback(o)
        if hero == 1 and not history:
            if a == b and a >= 12:
                return raise_to(400)
            if q >= (.41 if self.mode == 'pressure' else .44) or h[0] % 4 == h[1] % 4:
                return raise_to(250)
            return 'fold'
        if price == 0:
            return raise_to(500) if q >= .60 else 'check'
        if (a == b and a >= 10) or (a >= 13 and b >= 12):
            return raise_to(maxbet * 4)
        # A wide button's small open gives BB a cheap defense, but no automatic
        # 3-bet with mediocre aces and no price-insensitive deep calls.
        threshold = max(.39, price / (o['pot'] + price) + .10)
        return 'call' if q >= threshold else 'fold'

    def postflop(self, o):
        board = self.board_seen
        if len(board) == 5:
            return self.river_choose(o)
        p = float(o['pot']); c = float(o['to_call'])
        hero = o['hero_seat']; villain = 1 - hero
        rows = []
        samples = 22 if len(board) == 3 else 30
        for h, w in self.particles:
            if self.should_stop():
                raise _StopEpisode
            q = self.equity(h, board)
            if len(board) == 5:
                e = compare(rank(self.hole + board), rank(h + board))
            else:
                deck = [x for x in DECK if x not in self.hole and x not in h and x not in board]
                wins = 0.
                for i in range(samples):
                    if i % 16 == 0 and self.should_stop():
                        raise _StopEpisode
                    full = board + tuple(self.rng.sample(deck, 5 - len(board)))
                    wins += compare(rank(self.hole + full), rank(h + full))
                e = wins / samples
            rows.append((w, q, e))
        mass = sum(r[0] for r in rows) or 1
        eq = sum(w * e for w, q, e in rows) / mass
        # Slight realization discount before river; position mitigates it.
        haircut = 0 if len(board) == 5 else (.025 if hero == 1 else .045)
        margin = .015 if self.mode == 'cautious' else 0
        passive = max(0., (eq - haircut - margin) * (p + c) - c) if c else max(0., (eq - haircut) * p)
        best_action = 'call' if c and passive > 0 else 'fold' if c else 'check'
        best_ev = passive
        self.table = [{'action': best_action, 'ev': round(passive, 1), 'equity': round(eq, 3)}]
        legal = o['legal']
        near_nut = False
        cap = None
        if legal.get('can_raise'):
            maxbet = max(o['street_bets'])
            # The prior search often chose the largest legal amount because a
            # noisy posterior assigned too much fold equity to an all-in.  Use
            # a street plan instead: ordinary value/protection bets are capped
            # at 3/4 pot on the flop, pot on the turn, and 1.25 pot on the
            # river.  Stack pressure is reserved for a near-nut hand or an
            # exceptionally stable equity estimate.
            hero_rank = rank(self.hole + board) if len(board) == 5 else (0,)
            board_rank = rank(board) if len(board) == 5 else (-1,)
            made_category = hero_rank[0]
            # A board-only full house/quads is not a private nut hand: the
            # opponent shares it. Require hero's seven-card rank to improve
            # the exposed board before permitting stack-sized value.
            private_nuts = len(board) == 5 and made_category >= 5 and hero_rank > board_rank and eq >= .90
            early_monster = len(board) < 5 and rank(self.hole + board)[0] >= 3 and eq >= .90
            near_nut = private_nuts or early_monster
            cap_fraction = {3: .75, 4: 1.0, 5: 1.25}.get(len(board), .75)
            cap = maxbet + int(cap_fraction * (p + c))
            cap = min(legal['max_raise_to'], max(legal['min_raise_to'], cap))
            targets = {legal['min_raise_to'], cap}
            if near_nut:
                targets.add(legal['max_raise_to'])
            for f in (.33, .5, .75, 1.0, 1.25):
                targets.add(max(legal['min_raise_to'], min(cap, int(maxbet + f * (p + c)))))
            for target in sorted(targets):
                if self.should_stop():
                    raise _StopEpisode
                if target > cap and not near_nut:
                    continue
                cost = target - o['street_bets'][hero]
                opp_cost = min(o['stacks'][villain], target - o['street_bets'][villain])
                real_cost = min(cost, c + opp_cost)  # refund unmatched wager
                if opp_cost <= 0:
                    continue
                price = opp_cost / (p + real_cost + opp_cost)
                threshold = price + (0 if len(board) == 5 else .05)
                value = 0.; calls = 0.; called_wins = 0.
                for w, q, e in rows:
                    cp = .015 + .97 * sigmoid((q - threshold) / .055)
                    calls += w * cp
                    called_wins += w * cp * e
                    # Terminal-value rollout; discount vulnerable equity and
                    # avoid building deep pots with merely marginal callers.
                    realized = max(0., e - haircut)
                    value += w * ((1-cp) * p + cp * (realized * (p + real_cost + opp_cost) - real_cost))
                value /= mass
                ce = called_wins / calls if calls else 0
                foldp = 1 - calls / mass
                # Before river speculative bluffs cannot assume free showdown.
                if len(board) < 5:
                    value -= real_cost * max(0., .65 - ce) * .20
                # Uncertainty charge prevents noisy huge value bets with thin edges.
                value -= real_cost * (.025 if len(board) == 5 else .045)
                self.table.append({'action': 'raise_to ' + str(target), 'ev': round(value, 1), 'fold': round(foldp, 3), 'called_equity': round(ce, 3)})
                if value > best_ev + max(4, .025 * p):
                    best_ev, best_action = value, 'raise_to ' + str(target)
        self.table.sort(key=lambda r: r['ev'], reverse=True)
        # Even if the approximate EV loop preferred an overbet, obey the
        # structural cap unless the hand is near the nuts.  This keeps a single
        # sampled response from committing 200bb with a marginal one-pair hand.
        if best_action.startswith('raise_to ') and not near_nut:
            try:
                amount = int(best_action.split()[1])
                if cap is not None and amount > cap:
                    best_action = 'raise_to ' + str(cap)
            except Exception:
                pass
        return best_action

    def review(self, o, action, reason):
        rem = self.ctx.budget.remaining()
        # Reserve most wall time for completing the match. Three normal calls,
        # with a separate reserve for evidence-triggered recovery.
        if self.ctx.done or self.calls >= 7 or rem.get('system2_calls', 0) < 2 or rem.get('wall_s', 0) < 240:
            return action
        self.calls += 1
        self.last_review = o.get('completed_hands', 0)
        self.last_audit_hand = o.get('hand_id')
        payload = {k: o.get(k) for k in ('hole_cards','board','street','hero_seat','pot','to_call','stacks','street_bets','legal','history','completed_hands')}
        payload.update(reason=reason, proposed=action, alternatives=self.table[:5], mode=self.mode,
                       observations=dict(self.stats), recent_settlements=self.recent[-5:], failure=self.pending_failure,
                       river_certificate=self.river_certificate)
        prompt = (
            'Audit this heads-up poker policy and replan if evidence contradicts it. Opponent mc-pot-odds-v2 '
            'opens wide; premium QQ+/AK reraises. It estimates equity versus UNIFORM cards, not our range. '
            'Documented free bets: .75 pot at equity>=.62, pot at >=.85, 8% half-pot mixed branch otherwise. '
            'Calls require equity>=pot odds+.05 before river, pot odds on river; raises >=.80. '
            'We infer ranges from public actions using softened likelihoods and Monte Carlo, then optimize '
            'legal sizes. This harness uses a street-aware cap (0.75 pot flop, 1 pot turn, '
            '1.25 pot river) and permits an all-in only for near-nut/showdown equity; large '
            'bets from a noisy posterior are otherwise rejected. On the river we restore all 990 legal '
            'opponent holes, replay all public actions, calculate exact uniform-card river equity, '
            'and model one opponent raise with an aggregate fold/call response. '
            'Do not infer overfolding just from results. Avoid board-only hand strength. '
            'Choose an action for THIS observation and a reusable mode: balanced, pressure (wider SB opens), '
            'or cautious (extra marginal-call discount). Respect the sizing cap unless the evidence '
            'shows a near-nut value spot. Explain one concrete model failure if present. '
            'Reply only JSON {"action":"check|call|fold|raise_to N", "mode":"balanced|pressure|cautious", '
            '"lesson":"one short sentence"}.\n' + json.dumps(payload, separators=(',',':'))
        )
        try:
            res = self.ctx.think(prompt, timeout_s=180.)
            text = res.text or ''
            start, end = text.find('{'), text.rfind('}')
            obj = json.loads(text[start:end+1]) if start >= 0 else {}
            if obj.get('mode') in ('balanced','pressure','cautious'):
                self.mode = obj['mode']
            chosen = legalize(o, obj.get('action'))
            self.ctx.memory['last_audit'] = obj
            self.ctx.log('audit', reason=reason, reply=obj, accepted=bool(chosen))
            return chosen or action
        except Exception as exc:
            self.ctx.log('audit_unavailable', error=type(exc).__name__)
            return action
        finally:
            self.pending_failure = None

    def record_result(self, o):
        last = o.get('last_hand')
        if not last or last.get('hand_id') == self.last_result:
            return
        self.last_result = last.get('hand_id')
        self.recent.append({k:last.get(k) for k in ('hole_cards','board','revealed_hole_cards','net_chips','history')})
        self.recent = self.recent[-8:]
        # Loss alone is not evidence of a failed strategy. A shown hand outside
        # posterior support is; preserve it for an actual model audit.
        if self.hand == last.get('hand_id'):
            shown = last.get('revealed_hole_cards', {})
            opp = shown.get(str(1-last.get('hero_seat',0)), shown.get(1-last.get('hero_seat',0)))
            if opp and abs(last.get('net_chips',0)) >= 2500:
                self.pending_failure = {'reason':'large revealed showdown: audit range and value assumptions', 'result':last}

    def run_episode(self, game):
        ctx = self.ctx
        o = ctx.observe()
        while not ctx.done and not o.get('done', False):
            if self.should_stop():
                return
            self.record_result(o)
            try:
                self.update_range(o)
                if self.should_stop():
                    return
                a = self.preflop(o) if o.get('street') == 'preflop' else self.postflop(o)
                a = legalize(o, a) or fallback(o)
                hand_n = o.get('completed_hands',0)
                high_stakes = o.get('pot',0) >= 3500 and o.get('street') != 'preflop'
                review_due = self.calls == 0 or (high_stakes and self.calls < 3 and hand_n-self.last_review >= 15)
                if self.audit_future is not None and self.audit_future.done():
                    # Advice belongs to the old observation; only its reusable
                    # mode/lesson survives. Never post a stale action.
                    try:
                        self.audit_future.result()
                    except Exception as exc:
                        ctx.log('audit_future_error', error=type(exc).__name__)
                    self.audit_future = None
                if (review_due or self.pending_failure) and self.last_audit_hand != o.get('hand_id') and self.audit_future is None:
                    snapshot, proposed = o, a
                    reason = 'failure evidence' if self.pending_failure else 'strategic audit'
                    self.last_audit_hand = o.get('hand_id')
                    self.audit_future = ctx.spawn(lambda observation=snapshot, action=proposed, why=reason: self.review(observation, action, why))
                if self.should_stop():
                    break
                old = o
                histlen = len(old.get('history',[]))
                o = ctx.act(a)
                if o is None:
                    o = ctx.observe()
                progress = ctx.progress()
                illegal = max(o.get('illegal_actions',0), progress.get('illegal_actions',0))
                failed = bool(o.get('action_error')) or illegal > self.last_illegal or progress.get('since_obs_change',0) >= 1
                if failed:
                    self.pending_failure = {'reason':'execution did not advance', 'action':a, 'progress':progress, 'error':o.get('action_error')}
                    self.last_audit_hand = None
                    self.stats['execution_failures'] += 1
                self.last_illegal = illegal
                # Check observable responses against the chosen EV branch.
                if old.get('hand_id') == o.get('hand_id') and a.startswith('raise_to'):
                    response = [e for e in o.get('history',[])[histlen+1:] if e.get('seat') != old['hero_seat']]
                    for e in response:
                        if e.get('street') == old.get('street') and e.get('action') == 'raise_to':
                            self.stats['unexpected_reraises'] += 1
                            if self.stats['unexpected_reraises'] % 3 == 0:
                                self.pending_failure = {'reason':'three raises challenge terminal-call EV approximation', 'history':o.get('history')}
                ctx.log('policy_step', action=a, mode=self.mode, posterior_support=len(self.particles), alternatives=self.table[:3], progress=progress)
            except (_StopEpisode, BudgetExceeded):
                return
            except Exception as exc:
                if ctx.done:
                    break
                self.pending_failure = {'reason':'local computation exception', 'error':type(exc).__name__}
                ctx.log('policy_recovery', error=type(exc).__name__)
                try:
                    o = ctx.observe()
                    if self.audit_future is None and not ctx.done:
                        snapshot = o
                        self.audit_future = ctx.spawn(lambda observation=snapshot: self.review(observation, fallback(observation), 'local computation recovery'))
                    o = ctx.act(fallback(o)) or ctx.observe()
                    ctx.progress()
                except Exception:
                    break


class EquityCDF:
    """Weighted response partitions; q is the opponent's persistent belief."""
    def __init__(self, rows):
        ordered = sorted(rows)
        self.q = [q for q, e, w in ordered]
        self.m = [0.]
        self.w = [0.]
        for q, e, weight in ordered:
            self.m.append(self.m[-1] + weight)
            self.w.append(self.w[-1] + weight * e)
        self.rows = ordered

    def interval(self, lo=0., hi=2.):
        a = bisect.bisect_left(self.q, lo - 1e-12)
        b = bisect.bisect_left(self.q, hi - 1e-12)
        return self.m[b] - self.m[a], self.w[b] - self.w[a]

    def subset(self, lo=0., hi=2., factor=1.):
        return EquityCDF([(q,e,w*factor) for q,e,w in self.rows if lo <= q < hi])


class CoherentRiverHarness(Harness):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.river_model = 'coherent'
        self.river_bias = .012
        self.coherent_rows = None
        self.expected_response = None
        self.last_plan = None
        self.river_failures = 0
        self.prior_river = None

    def new_hand(self, o):
        super().new_hand(o)
        self.prior_river = None
        self.coherent_rows = None

    def river_event_probability(self, q, ev, bets, committed, raises, increment):
        seat = ev['seat']
        other = 1-seat
        price = min(20000-committed[seat], max(bets.values())-bets[seat])
        pot = ev['pot_before']
        action = ev['action']
        ceiling = bets[seat]+20000-committed[seat]
        can_raise = committed[other] < 20000 and committed[seat]+price < 20000
        def matches(target):
            target = min(ceiling, max(max(bets.values())+increment, target))
            return abs(target-int(ev.get('raise_to') or 0)) <= 1
        if price == 0:
            if q >= .85:
                size, prob = 1., 1.
            elif q >= .62:
                size, prob = .75, 1.
            else:
                size, prob = .5, .08
            if action == 'check':
                return 1.-prob if can_raise else 1.
            if action == 'raise_to' and can_raise:
                return prob if matches(int(max(bets.values())+size*pot+.5)) else 0.
            return 0.
        raise_branch = can_raise and raises[seat] == 0 and q >= .80
        if raise_branch:
            if action != 'raise_to':
                return 0.
            target = int(max(bets.values())+.75*(pot+price)+.5)
            return float(matches(target))
        calls = q + 1e-12 >= price / max(1, pot+price)
        return float((action == 'call' and calls) or (action == 'fold' and not calls))

    def rebuild_river(self, o, board):
        # Reconstruct the earlier-street prior once, before applying any river
        # action. Earlier observations are retained, with the parent's noise.
        key = (board, tuple((e['seat'],e['street'],e['action'],e.get('cost',0),e.get('raise_to'))
                            for e in o.get('history',[]) if e['street'] != 'river'))
        if self.prior_river is None or self.prior_river[0] != key:
            view = dict(o)
            view['history'] = [e for e in o.get('history',[]) if e['street'] != 'river']
            super().rebuild_river(view, board)
            self.prior_river = (key, [(h,w) for h,w in self.particles])
        prior = self.prior_river[1]
        self.board_seen = board
        self.history_seen = len(o.get('history', []))
        if self.river_model == 'soft':
            self.coherent_rows = None
            super().rebuild_river(o, board)
            return
        hr = rank(self.hole+board)
        shifts = ((-self.river_bias,.1),(0.,.8),(self.river_bias,.1))
        rows = []
        by_hand = Counter()
        for shift, model_weight in shifts:
            weights = [w*model_weight for h,w in prior]
            qs = [max(0.,min(1.,self.equity(h,board)+shift)) for h,w in prior]
            bets = {0:100,1:50}; committed = {0:100,1:50}
            raises = {0:0,1:0}; street = 'preflop'; increment=100
            for ev in o.get('history',[]):
                if ev['street'] != street:
                    street=ev['street']; bets={0:0,1:0}; raises={0:0,1:0}; increment=100
                seat=ev['seat']
                if street == 'river' and seat != o['hero_seat']:
                    for i,q in enumerate(qs):
                        weights[i] *= self.river_event_probability(q,ev,bets,committed,raises,increment)
                if ev['action'] == 'raise_to':
                    old=max(bets.values()); new=ev['raise_to']
                    increment=max(increment,new-old)
                    bets[seat]=new; raises[seat]+=1
                    committed[seat]+=ev.get('cost',0)
                elif ev['action']=='call':
                    bets[seat]+=ev.get('cost',0); committed[seat]+=ev.get('cost',0)
            for (h,_),q,w in zip(prior,qs,weights):
                if w > 0:
                    rows.append((q,compare(hr,rank(h+board)),w))
                    by_hand[h]+=w
        z=sum(w for q,e,w in rows)
        if z < 1e-20:
            self.river_failures+=1
            self.pending_failure={'reason':'coherent river hypotheses contradicted',
                                  'history':o.get('history'), 'board':o.get('board'),
                                  'count':self.river_failures}
            self.coherent_rows=None
            super().rebuild_river(o,board)
            self.ctx.log('coherent_range_recovery', evidence=self.pending_failure)
            return
        self.coherent_rows=[(q,e,w/z) for q,e,w in rows]
        self.particles=[[h,w/z] for h,w in by_hand.items()]
        self.river_certificate={'legal_combos':len(prior),'supported_combos':len(by_hand),
            'persistent_hypotheses':len(rows),'predictive_mass':round(z,6),
            'equity':round(sum(e*w/z for q,e,w in rows),4)}

    def river_choose(self, o):
        if self.coherent_rows is None:
            self.expected_response=None
            return super().river_choose(o)
        base=EquityCDF(self.coherent_rows)
        hero=o['hero_seat']; villain=1-hero
        p=o['pot']; price=o['to_call']
        hb,vb=o['street_bets'][hero],o['street_bets'][villain]
        hs,vs=o['stacks'][hero],o['stacks'][villain]
        ceiling=min(hb+hs,vb+vs)
        raised=any(e['seat']==villain and e['street']=='river' and e['action']=='raise_to'
                   for e in o.get('history',[]))
        hr=rank(self.hole+self.board_seen)
        monster=(hr[0]>=6 or (hr[0]==5 and hr[1]>=13)) and hr>rank(self.board_seen)
        nodes=0
        def guard():
            nonlocal nodes
            nodes+=1
            if nodes%128==0 and self.should_stop():
                raise _StopEpisode
        def showdown(d,x,y):
            m,w=d.interval()
            return w*(p+x+y)-m*x
        def terminal(d,x,y):
            # Opponent's own raise allowance has already been spent.
            oc=max(0,min(vs-y,hb+x-vb-y))
            odds=oc/max(1,p+x+y+oc)
            cm,cw=d.interval(odds)
            m,w=d.interval()
            return (m-cm)*(p+y)+cw*(p+x+y+oc)-cm*x,cm,cw
        def hero_reply(d,x,y,increment,allow=True):
            m,w=d.interval()
            if m<1e-15:
                return 0.,'empty'
            call=min(hs-x,max(0,vb+y-hb-x))
            best=(-x*m,'fold')
            value=showdown(d,x+call,y)
            if value>best[0]: best=(value,'call')
            if not allow or hs-x<=call or vs<=y:
                return best
            top=max(hb+x,vb+y)
            minimum=min(ceiling,top+max(100,increment))
            cap=min(ceiling,max(minimum,top+int(1.25*(p+x+y+call))))
            # Above .8 every hand calls any heads-up river wager. On this
            # branch value is linear, so endpoints replace integer enumeration.
            high_only=bool(d.q) and d.q[0]>=.80-1e-12
            targets=(minimum,cap) if high_only else range(minimum,cap+1)
            for target in targets:
                guard()
                if target<=top: continue
                nx=target-hb
                ev,cm,cw=terminal(d,nx,y)
                ev-=.025*(nx-x)*m
                if ev>best[0]: best=(ev,'raise_to '+str(target))
            if monster and cap<ceiling:
                nx=ceiling-hb
                ev,cm,cw=terminal(d,nx,y)
                if cw/max(cm,1e-15)>=.97 and w/m>=.93:
                    ev-=.025*(nx-x)*m
                    if ev>best[0]: best=(ev,'raise_to '+str(ceiling))
            return best
        def bet(d,target):
            guard()
            x=target-hb; oc=target-vb
            odds=oc/(p+x+oc)
            can_raise=not raised and vs>oc and hs>x
            rm,rw=d.interval(.80) if can_raise else (0.,0.)
            cm,cw=d.interval(odds,.80) if can_raise else d.interval(odds)
            mass,wins=d.interval()
            fm=max(0.,mass-rm-cm)
            value=fm*p+cw*(p+x+oc)-cm*x
            continuation='none'
            if rm>1e-15:
                last_inc=max(100,target-max(hb,vb))
                reply=min(ceiling,max(target+last_inc,int(target+.75*(p+x+oc)+.5)))
                val,continuation=hero_reply(raised_range,x,reply-vb,reply-target,reply-target>=last_inc)
                value+=val
            ce=(cw+rw)/max(1e-15,cm+rm)
            return value-.025*x, {'fold':fm,'call':cm,'raise_to':rm,'called_equity':ce,'vs_raise':continuation}
        raised_range = base.subset(.80)
        plans=[]
        if price:
            choices=[{'action':'fold','ev':0.},{'action':'call','ev':showdown(base,price,0)}]
        elif hero==o.get('button_seat',1):
            choices=[{'action':'check','ev':showdown(base,0,0)}]
        else:
            # Aggregate hands with the same clipped bet BEFORE hero optimizes.
            checked=base.subset(0.,.62,.92)
            value=showdown(checked,0,0)
            branches={}
            for lo,hi,frac,factor in ((0.,.62,.5,.08),(.62,.85,.75,1.),(.85,2.,1.,1.)):
                size=min(hs,vs,max(100,int(frac*p+.5)))
                branches.setdefault(size,[]).extend(base.subset(lo,hi,factor).rows)
            for size,rr in branches.items():
                d=EquityCDF(rr)
                val,act=hero_reply(d,0,size,size)
                value+=val
                plans.append({'villain_bet':size,'mass':round(d.interval()[0],4),'hero_reply':act})
            choices=[{'action':'check','ev':value}]
        if o['legal'].get('can_raise'):
            minimum=o['legal']['min_raise_to']
            cap=min(ceiling,max(minimum,max(hb,vb)+int(1.25*(p+price))))
            best_bet=None
            for target in range(minimum,cap+1):
                if target<=max(hb,vb): continue
                ev,pred=bet(base,target)
                if best_bet is None or ev>best_bet['ev']:
                    best_bet={'action':'raise_to '+str(target),'ev':ev,'response':pred}
            if monster and cap<ceiling and base.interval()[1]>=.93:
                ev,pred=bet(base,ceiling)
                if pred['called_equity']>=.97 and (best_bet is None or ev>best_bet['ev']):
                    best_bet={'action':'raise_to '+str(ceiling),'ev':ev,'response':pred}
            if best_bet: choices.append(best_bet)
        choices=[v for v in choices if legalize(o,v['action'])]
        self.table=sorted(choices,key=lambda v:v['ev'],reverse=True)
        choice=self.table[0]
        self.expected_response={'hand_id':o['hand_id'],'hero':hero,'history_len':len(o.get('history',[])),
                                'prediction':choice.get('response'),'action':choice['action']}
        self.last_plan={'hand_id':o['hand_id'],'alternatives':self.table,'check_plan':plans,'nodes':nodes}
        self.ctx.log('coherent_river_search',certificate=self.river_certificate,plan=self.last_plan)
        return choice['action']

    def record_result(self,o):
        expected=self.expected_response
        if expected:
            source=o if o.get('hand_id')==expected['hand_id'] else (o.get('last_hand') or {})
            if source.get('hand_id')==expected['hand_id']:
                response=next((e for e in source.get('history',[])[expected['history_len']:]
                               if e['seat']!=expected['hero']),None)
                if response and expected['prediction']:
                    probability=expected['prediction'].get(response['action'],0.)
                    self.ctx.log('coherent_response_check',response=response,predicted=probability)
                    if probability<.02:
                        self.pending_failure={'reason':'unlikely coherent response','expected':expected,'observed':response}
            self.expected_response=None
        last=o.get('last_hand') or {}
        if last.get('hand_id') and last.get('hand_id') != self.last_result:
            self.last_result=last['hand_id']
            self.recent.append({k:last.get(k) for k in ('hole_cards','board','revealed_hole_cards','net_chips','history')})
            self.recent=self.recent[-8:]
            shown=last.get('revealed_hole_cards',{})
            opp=shown.get(str(1-last.get('hero_seat',0)),shown.get(1-last.get('hero_seat',0)))
            if opp and self.hand==last['hand_id'] and self.coherent_rows is not None:
                hand=tuple(sorted(card(c) for c in opp))
                support=sum(w for h,w in self.particles if tuple(sorted(h))==hand)
                if support < 1e-10:
                    self.pending_failure={'reason':'revealed opponent outside coherent range','result':last}

    def review(self,o,action,reason):
        rem=self.ctx.budget.remaining()
        if self.ctx.done or self.calls>=4 or rem.get('system2_calls',0)<2 or rem.get('wall_s',0)<300:
            return action
        self.calls+=1
        self.last_review=o.get('completed_hands',0)
        evidence=self.pending_failure
        payload={k:o.get(k) for k in ('hand_id','hole_cards','board','street','hero_seat','pot','to_call','stacks','street_bets','legal','history','completed_hands')}
        payload.update(reason=reason, proposed=action, failure=evidence, river_plan=self.last_plan,
                       river_certificate=self.river_certificate,mode=self.mode,river_model=self.river_model,
                       river_bias=self.river_bias,recent_public_results=self.recent[-3:])
        prompt=('Review this observation-only poker exploit. We retain a conservative sampled range '
                'policy before river. River reconstructs every legal hand with earlier action likelihoods, '
                'then conditions three persistent equity hypotheses on the WHOLE river sequence. '
                'Opponent equity is exact versus uniform unknown cards on river. Documented river '
                'policy: free .75pot value at .62, pot at .85, otherwise 8% halfpot; facing bet '
                'raise once per street at .80, otherwise call at pot odds. Its own free bet counts '
                'as its raise. A hypothesis cannot independently err on each action. All legal integer '
                'sizes within conservative caps are optimized with check/bet/reraise continuations. '
                'If evidence contradicts thresholds or sizes, choose river_model soft (parent smoothed '
                'model) or widen persistent river_bias within 0-.04. Earlier action likelihoods remain '
                'approximate. Do not treat losses as model failure or prior plans as current states. '
                'Only reusable settings are applied, and the current state is re-solved after your review. '
                'Return JSON under 120 tokens: {"mode":"balanced|pressure|cautious",'
                '"river_model":"coherent|soft","river_bias":0.012,"lesson":"brief evidence"}.\n'
                +json.dumps(payload,separators=(',',':')))
        try:
            result=self.ctx.think(prompt,timeout_s=210.)
            raw=str(getattr(result,'text','') or '')
            a,b=raw.find('{'),raw.rfind('}')
            obj=json.loads(raw[a:b+1]) if 0<=a<b else {}
            if not isinstance(obj,dict): obj={}
            if obj.get('mode') in ('balanced','pressure','cautious'): self.mode=obj['mode']
            if obj.get('river_model') in ('coherent','soft'): self.river_model=obj['river_model']
            bias=obj.get('river_bias')
            if isinstance(bias,(float,int)) and not isinstance(bias,bool) and math.isfinite(bias):
                self.river_bias=max(0.,min(.04,float(bias)))
            self.ctx.memory['coherent_review']=obj
            self.ctx.log('coherent_review',reason=reason,reply=obj)
        except Exception as exc:
            self.ctx.log('coherent_review_failed',error=type(exc).__name__)
        if self.pending_failure is evidence:
            self.pending_failure=None
        return action


    def run_episode(self, game):
        ctx=self.ctx
        o=ctx.observe()
        next_review=35
        since_review=999
        consecutive_errors=0
        while not ctx.done and not o.get('done',False):
            if self.should_stop(): return
            self.record_result(o)
            try:
                self.update_range(o)
                needs_review=self.pending_failure is not None
                routine=o.get('completed_hands',0)>=next_review and self.calls==0
                if (needs_review and since_review>=12) or routine:
                    self.review(o,fallback(o),'failure evidence' if needs_review else 'public-observation calibration')
                    since_review=0
                    next_review=1000
                    if self.should_stop(): return
                    # Apply revised coherent/soft model to the same observation.
                    if o.get('street')=='river': self.rebuild_river(o,self.board_seen)
                action=self.preflop(o) if o.get('street')=='preflop' else self.postflop(o)
                action=legalize(o,action) or fallback(o)
                before=ctx.progress()
                old=o
                o=ctx.act(action) or ctx.observe()
                progress=ctx.progress()
                since_review+=1
                failed=(o.get('action_error') or progress.get('illegal_actions',0)>before.get('illegal_actions',0)
                        or progress.get('since_obs_change',0)>=1)
                ctx.log('verified_coherent_step',action=action,progress=progress,alternatives=self.table[:3])
                if failed:
                    self.pending_failure={'reason':'execution did not advance','action':action,
                                          'progress':progress,'error':o.get('action_error')}
                    self.review(o,fallback(o),'execution failure recovery')
                    since_review=0
                    if self.should_stop(): return
                    safe=fallback(o)
                    before=ctx.progress()
                    o=ctx.act(safe) or ctx.observe()
                    after=ctx.progress()
                    ctx.log('verified_recovery',action=safe,progress=after)
                    if o.get('action_error') or after.get('illegal_actions',0)>before.get('illegal_actions',0):
                        return
                consecutive_errors=0
            except (_StopEpisode,BudgetExceeded):
                return
            except Exception as exc:
                if ctx.done: return
                consecutive_errors+=1
                self.pending_failure={'reason':'local computation exception','error':type(exc).__name__,
                                      'message':str(exc)[:160]}
                ctx.log('coherent_calculation_recovery',evidence=self.pending_failure)
                try:
                    o=ctx.observe()
                    self.review(o,fallback(o),'calculation failure recovery')
                    if self.should_stop() or consecutive_errors>2: return
                    self.river_model='soft'
                    action=fallback(o)
                    before=ctx.progress()
                    o=ctx.act(action) or ctx.observe()
                    progress=ctx.progress()
                    ctx.log('verified_recovery',action=action,progress=progress)
                    if progress.get('illegal_actions',0)>before.get('illegal_actions',0): return
                except Exception:
                    return


def build(ctx):
    return CoherentRiverHarness(ctx)
