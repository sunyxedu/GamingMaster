"""Observation-only range inference and legal-action EV search, audited by Astra."""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter

# Standard library only. Cards are rank-major integers; no game imports or state.
RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = tuple(range(52))
STREETS = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}

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
        for _ in range(n):
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
        # Full history replay computes action context without hidden state.
        events = o.get('history', [])
        bets = {0: 0, 1: 0}
        street = None
        raises = {0: 0, 1: 0}
        for i, ev in enumerate(events):
            if ev.get('street') != street:
                street = ev.get('street')
                bets = {0: 100, 1: 50} if street == 'preflop' else {0: 0, 1: 0}
                raises = {0: 0, 1: 0}
            seat = ev.get('seat')
            if i >= self.history_seen and seat != o['hero_seat']:
                b = board[:STREETS.get(street, 0)]
                for row in self.particles:
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
        p = float(o['pot']); c = float(o['to_call'])
        hero = o['hero_seat']; villain = 1 - hero
        rows = []
        samples = 22 if len(board) == 3 else 30
        for h, w in self.particles:
            if self.ctx.done:
                break
            q = self.equity(h, board)
            if len(board) == 5:
                e = compare(rank(self.hole + board), rank(h + board))
            else:
                deck = [x for x in DECK if x not in self.hole and x not in h and x not in board]
                wins = 0.
                for _ in range(samples):
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
        if legal.get('can_raise'):
            maxbet = max(o['street_bets'])
            targets = {legal['min_raise_to'], legal['max_raise_to']}
            for f in (.33, .5, .75, 1., 1.5, 2.5):
                targets.add(max(legal['min_raise_to'], min(legal['max_raise_to'], int(maxbet + f * (p + c)))))
            for target in sorted(targets):
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
        return best_action

    def review(self, o, action, reason):
        rem = self.ctx.budget.remaining()
        # Reserve most wall time for completing the match. Three normal calls,
        # with a separate reserve for evidence-triggered recovery.
        if self.ctx.done or self.calls >= 7 or rem.get('system2_calls', 0) < 2 or rem.get('wall_s', 0) < 150:
            return action
        self.calls += 1
        self.last_review = o.get('completed_hands', 0)
        self.last_audit_hand = o.get('hand_id')
        payload = {k: o.get(k) for k in ('hole_cards','board','street','hero_seat','pot','to_call','stacks','street_bets','legal','history','completed_hands')}
        payload.update(reason=reason, proposed=action, alternatives=self.table[:5], mode=self.mode,
                       observations=dict(self.stats), recent_settlements=self.recent[-5:], failure=self.pending_failure)
        prompt = (
            'Audit this heads-up poker policy and replan if evidence contradicts it. Opponent mc-pot-odds-v2 '
            'opens wide; premium QQ+/AK reraises. It estimates equity versus UNIFORM cards, not our range. '
            'Documented free bets: .75 pot at equity>=.62, pot at >=.85, 8% half-pot mixed branch otherwise. '
            'Calls require equity>=pot odds+.05 before river, pot odds on river; raises >=.80. '
            'We infer ranges from public actions using softened likelihoods and Monte Carlo, then optimize '
            'legal sizes. EV estimates approximate future streets; large bets require scrutiny. '
            'Do not infer opponent overfolding just from our wins/losses. Avoid board-only hand strength. '
            'Choose an action for THIS observation and a reusable mode: balanced, pressure (wider SB opens), '
            'or cautious (extra marginal-call discount). Explain one concrete model failure if present. '
            'Reply only JSON {"action":"check|call|fold|raise_to N", "mode":"balanced|pressure|cautious", '
            '"lesson":"one short sentence"}.\n' + json.dumps(payload, separators=(',',':'))
        )
        try:
            res = self.ctx.think(prompt, timeout_s=125.)
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
            self.record_result(o)
            try:
                self.update_range(o)
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
                if ctx.done:
                    break
                old = o
                histlen = len(old.get('history',[]))
                o = ctx.act(a)
                if o is None:
                    o = ctx.observe()
                progress = ctx.progress()
                illegal = max(o.get('illegal_actions',0), progress.get('illegal_actions',0))
                failed = bool(o.get('action_error')) or illegal > self.last_illegal or progress.get('since_obs_change',0) >= 2
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

def build(ctx):
    return Harness(ctx)
