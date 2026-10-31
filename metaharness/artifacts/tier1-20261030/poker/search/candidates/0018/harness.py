"""Latent sequence inference with systematic, low-variance range resampling."""
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
        self.noise = 1.0
        self.trust = .85
        self.current_latents = {}
        self.support = None
        self.prediction = None
        self.surprises = []
        self.response_calibration = {
            'n': 0, 'observed': {'fold': 0, 'call': 0, 'raise_to': 0},
            'expected': {'fold': 0., 'call': 0., 'raise_to': 0.},
            'variance': {'fold': 0., 'call': 0., 'raise_to': 0.},
            'brier_sum': 0.,
        }

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





    def contexts(self, o):
        """Reconstruct wager prices and raise availability from public history."""
        bets = [100, 50]
        committed = [100, 50]
        raises = [0, 0]
        street = 'preflop'
        result = []
        for ev in o.get('history', []):
            if ev['street'] != street:
                street = ev['street']
                bets = [0, 0]
                raises = [0, 0]
            s = ev['seat']; v = 1-s
            price = min(20000-committed[s], max(bets)-bets[s])
            may_raise = (20000-committed[s] > price and 20000-committed[v] > 0)
            result.append((ev, tuple(bets), raises[s], price, may_raise))
            cost = int(ev.get('cost') or 0)
            committed[s] += cost
            if ev['action'] == 'raise_to':
                bets[s] = int(ev['raise_to'])
                raises[s] += 1
            elif ev['action'] == 'call':
                bets[s] += cost
        return result

    def latent_nodes(self, h, board):
        q = self.equity(h, board)
        if len(board) == 5:
            return [(q, 1.)]
        # Uncertainty covers our 64-trial estimator plus the opponent's 512
        # trials. It is integrated ONCE per street, not once per action.
        sigma = self.noise * math.sqrt(max(.04, q*(1-q)) * (1/64 + 1/512))
        zs = (-2.4, -1.8, -1.2, -.6, 0., .6, 1.2, 1.8, 2.4)
        ws = [math.exp(-z*z/2) for z in zs]
        den = sum(ws)
        return [(max(0., min(1., q+z*sigma)), w/den) for z,w in zip(zs,ws)]

    def event_probability(self, q, ev, bets, own_raises, price, may_raise):
        action = ev['action']
        p = max(1., ev.get('pot_before', 1))
        if price > 0:
            raises = q >= .80 and own_raises == 0 and may_raise
            calls = q >= price/(p+price) + (0 if ev['street']=='river' else .05)
            predicted = 'raise_to' if raises else 'call' if calls else 'fold'
            prob = 1. if action == predicted else 0.
        elif action == 'check':
            prob = .92 if q < .62 else 0.
        elif action == 'raise_to':
            # Clipped small pots / stacks obscure sizing: use the total
            # public cost only where it identifies a distinct policy branch.
            fraction = float(ev.get('cost', 0))/p
            if not may_raise or fraction < .45:
                prob = 1. if q >= .62 else .08
            elif fraction >= .9:
                prob = 1. if q >= .85 else 0.
            elif fraction >= .62:
                prob = 1. if .62 <= q < .85 else 0.
            else:
                prob = .08 if q < .62 else 0.
        else:
            prob = .1
        # Residual model mismatch has positive support. This is separate from
        # the shared cached-equity error and cannot be eliminated by Astra.
        return .008 + .976*prob

    def update_range(self, o):
        if self.hand != o.get('hand_id'):
            self.new_hand(o)
            self.support = None
        board = tuple(card(c) for c in o.get('board', []))
        contexts = self.contexts(o)
        pre = [x for x in contexts if x[0]['street']=='preflop' and x[0]['seat'] != o['hero_seat']]
        if len(board) == 5:
            # Parent routine installs exact uniform river equities in eq_cache.
            # Its posterior is discarded; only ranks/equity calculation is reused.
            self.rebuild_river(o, board)
            base = [(h, 1.) for h,w in self.particles]
        elif not board or self.support is None:
            base = []
            for h in itertools.combinations([c for c in DECK if c not in self.hole and c not in board], 2):
                weight = 1.
                for ev,bets,n,price,may in pre:
                    weight *= self.likelihood(h,ev,(),dict(enumerate(bets)),n)
                base.append((h,weight))
            if board:
                # Systematic resampling has lower sampling variance than 320
                # independent draws. In particular, a low-probability but
                # strategically important premium hand is less likely to be
                # deleted just because several draws landed in one common
                # region. Keep the same particle budget to isolate resampling.
                entries = [(h, max(0.0, float(w))) for h,w in base]
                total = sum(w for h,w in entries)
                if total <= 0.0:
                    entries = [(h, 1.0) for h,w in base]
                    total = float(len(entries))
                step = total / 320.0
                point = self.rng.random() * step
                draws = []
                cumulative = entries[0][1]
                idx = 0
                for j in range(320):
                    target = point + j * step
                    while idx < len(entries) - 1 and cumulative < target:
                        idx += 1
                        cumulative += entries[idx][1]
                    draws.append(entries[idx][0])
                count = Counter(draws)
                self.support = [(h,n/320.) for h,n in count.items()]
                self.ctx.log('systematic_range_resample', source=len(entries), particles=320,
                             unique=len(self.support), ess=round(1.0/sum((n/320.)**2 for n in count.values()), 2))
                base = self.support
        else:
            base = self.support
        self.current_latents = {}
        rows = []
        for i,(h,prior) in enumerate(base):
            if i % 32 == 0 and self.should_stop():
                raise _StopEpisode
            if any(c in board for c in h):
                continue
            weight = prior
            if len(board) == 5:
                for ev,bets,n,price,may in pre:
                    weight *= self.likelihood(h,ev,(),dict(enumerate(bets)),n)
            for street,ncards in (('flop',3),('turn',4),('river',5)):
                if len(board) < ncards:
                    break
                b = board[:ncards]
                nodes = self.latent_nodes(h,b)
                events = [x for x in contexts if x[0]['street']==street and x[0]['seat']!=o['hero_seat']]
                for ev,bets,n,price,may in events:
                    nodes = [(q,w*self.event_probability(q,ev,bets,n,price,may)) for q,w in nodes]
                evidence = sum(w for q,w in nodes)
                weight *= max(1e-15,evidence)**self.trust
                nodes = [(q,w/max(1e-15,evidence)) for q,w in nodes]
                if ncards == len(board):
                    self.current_latents[h] = nodes
            rows.append([h,weight])
        z = sum(w for h,w in rows)
        if z <= 1e-25 or not rows:
            self.pending_failure = {'reason':'latent range collapsed','history':o.get('history')}
            rows = [[h,1.] for h in itertools.combinations([c for c in DECK if c not in self.hole and c not in board],2)]
            z = len(rows)
        self.particles = [[h,w/z] for h,w in rows]
        self.board_seen = board
        self.history_seen = len(o.get('history',[]))

    def rebuild_river(self, o, board):
        """Full support and exact uniform equity, without independent replay."""
        key = ('complete_river',board)
        if key not in self.eq_cache:
            avail = [c for c in DECK if c not in board]
            ranks = {}; by_card = {c:[] for c in avail}
            for i,h in enumerate(itertools.combinations(avail,2)):
                if i%64 == 0 and self.should_stop(): raise _StopEpisode
                r = rank(h+board); ranks[h]=r
                for c in h: by_card[c].append(r)
            ordered = sorted(ranks.values())
            for values in by_card.values(): values.sort()
            def score(values,r):
                return .5*(bisect.bisect_left(values,r)+bisect.bisect_right(values,r))
            support = []
            for h,r in ranks.items():
                if any(c in self.hole for c in h): continue
                q = (score(ordered,r)-score(by_card[h[0]],r)-score(by_card[h[1]],r)+.5)/990.
                self.eq_cache[(tuple(sorted(h)),board)] = q
                support.append(h)
            self.eq_cache[key] = support
        self.particles = [[h,1.] for h in self.eq_cache[key]]

    def postflop(self, o):
        board = self.board_seen
        hero = o['hero_seat']; villain = 1-hero
        p=float(o['pot']); c=float(o['to_call'])
        hb,vb=o['street_bets'][hero],o['street_bets'][villain]
        hs,vs=o['stacks'][hero],o['stacks'][villain]
        rows=[]
        for h,w in self.particles:
            if self.should_stop(): raise _StopEpisode
            key=('conditional_hero',h,board)
            if key not in self.eq_cache:
                if len(board)==5:
                    e=compare(rank(self.hole+board),rank(h+board))
                else:
                    rng=random.Random(int.from_bytes(hashlib.blake2s(bytes(self.hole+h+board),digest_size=8).digest(),'big'))
                    deck=[x for x in DECK if x not in self.hole and x not in h and x not in board]
                    # All turn rivers are cheap, while flop uses 28 local samples.
                    runs=[(x,) for x in deck] if len(board)==4 else [tuple(rng.sample(deck,2)) for _ in range(28)]
                    e=sum(compare(rank(self.hole+board+r),rank(h+board+r)) for r in runs)/len(runs)
                self.eq_cache[key]=e
            e=self.eq_cache[key]
            nodes=self.current_latents.get(h) or self.latent_nodes(h,board)
            rows.append((w,e,nodes))
        eq=sum(w*e for w,e,nodes in rows)
        haircut=0. if len(board)==5 else (.025 if hero==1 else .045)
        passive=(max(0.,eq-haircut)* (p+c)-c)
        choices=[{'action':'fold','ev':0.},{'action':'call','ev':passive}] if c else [{'action':'check','ev':max(0.,eq-haircut)*p}]
        legal=o['legal']; maxbet=max(hb,vb)
        opp_raises=sum(ev.get('seat')==villain and ev.get('street')==o['street'] and ev.get('action')=='raise_to' for ev in o.get('history',[]))
        if legal.get('can_raise'):
            fraction={3:.75,4:1.,5:1.25}[len(board)]
            cap=min(legal['max_raise_to'],max(legal['min_raise_to'],maxbet+int(fraction*(p+c))))
            targets={legal['min_raise_to'],cap}
            for f in (.33,.5,.75,1.,1.25):
                targets.add(max(legal['min_raise_to'],min(cap,int(maxbet+f*(p+c)))))
            hr=rank(self.hole+board)
            monster=(hr[0]>=3 and eq>=.90) if len(board)<5 else (hr[0]>=6 or hr[0]==5 and hr[1]>=13) and eq>=.93 and hr>rank(board)
            if monster: targets.add(legal['max_raise_to'])
            for target in sorted(targets):
                if self.should_stop(): raise _StopEpisode
                effective=min(target,hb+hs,vb+vs)
                cost=effective-hb; oc=effective-vb
                if cost<=c or oc<=0: continue
                threshold=oc/(p+cost+oc)+(0 if len(board)==5 else .05)
                can_raise=(opp_raises==0 and vs>oc and hs>cost)
                folds=call_mass=call_wins=raise_mass=raise_wins=0.
                for w,e,nodes in rows:
                    pr=sum(nw for q,nw in nodes if q>=.80)*.984 if can_raise else 0.
                    pc=sum(nw for q,nw in nodes if q>=threshold and not(can_raise and q>=.80))
                    pc=.008+(1-.016)*pc
                    pc=min(1-pr,pc)
                    pf=1-pr-pc
                    folds+=w*pf; call_mass+=w*pc; call_wins+=w*pc*e
                    raise_mass+=w*pr; raise_wins+=w*pr*e
                reply=max(effective+max(100,effective-maxbet),int(effective+.75*(p+cost+oc)+.5))
                reply=min(reply,hb+hs,vb+vs)
                final_cost=reply-hb; final_oc=reply-vb
                raise_ev=max(-cost*raise_mass,(raise_wins-haircut*raise_mass)*(p+final_cost+final_oc)-raise_mass*final_cost)
                ce=call_wins/max(1e-12,call_mass)
                if len(board)==5 and target>cap and ce<.97: continue
                ev=folds*p+(call_wins-haircut*call_mass)*(p+cost+oc)-call_mass*cost+raise_ev
                ev-=cost*(.025 if len(board)==5 else .045)
                choices.append({'action':'raise_to '+str(target),'ev':ev,'fold':folds,'call':call_mass,'raise_to':raise_mass,'called_equity':ce})
        self.table=sorted(choices,key=lambda r:r['ev'],reverse=True)
        best=self.table[0]
        self.prediction=best if best['action'].startswith('raise_to') else None
        self.ctx.log('latent_sequence_search',equity=round(eq,4),support=len(rows),choice=best['action'],alternatives=self.table[:4])
        return best['action']


    def review(self, o, action, reason):
        rem=self.ctx.budget.remaining()
        if self.ctx.done or self.calls>=4 or rem.get('system2_calls',0)<2 or rem.get('wall_s',0)<330:
            return action
        self.calls+=1
        self.last_review=o.get('completed_hands',0)
        evidence=self.pending_failure
        payload={k:o.get(k) for k in ('hand_id','hole_cards','board','street','hero_seat','pot','to_call','stacks','street_bets','legal','history','completed_hands')}
        payload.update(reason=reason,proposed=action,current_alternatives=self.table[:4],noise=self.noise,trust=self.trust,mode=self.mode,
                       response_calibration=self.response_calibration,
                       failure=evidence,surprises=self.surprises[-4:],recent_settlements=self.recent[-3:],progress=self.ctx.progress())
        prompt=('Audit a public-observation heads-up poker harness against frozen mc-pot-odds-v2. '
                'Opponent caches one uniform-card equity estimate per street: 512 trials before river, exact river. '
                'We use 64 local trials per hypothetical hand, quadrature uncertainty, and condition the SAME latent '
                'estimate on every action this street, then predict replies from that posterior. '
                'Documented free bets: .75 pot at >=.62, pot at >=.85, else 8% half-pot mix; facing bet '
                'raises once/street at >=.80, else calls at pot odds+.05 before river and pot odds river. '
                'There is residual likelihood floor .008. Before river the range has 320 weighted sampled particles; '
                'river rebuilds all 990 legal hands. River uniform equity is exact; preceding estimates are not. '
                'EV truncates future streets, models one aggregate raise response, and uses staged bet caps. '
                'Each settlement includes hero_seat: seat 1 is SB and seat 0 BB. A +50 settlement with '
                'opponent/SB folding is normal for ANY hero hole cards, including AA. Never diagnose a '
                'settlement error from repeated +50 blind wins. Opponent folds may skip hero turns. '
                'The response_calibration counts ONLY immediate replies to hero postflop raises; expected '
                'counts sum the conditional predicted probabilities for those same opportunities. Variance '
                'is a descriptive sum p(1-p), not an independent-match confidence interval. '
                'Keep noise/trust unchanged unless concrete response or execution evidence challenges them. '
                'Recent losses or weak hole classes alone cannot establish poor decisions or miscalibration. '
                'Diagnose model failure from public response evidence, not a single loss. Choose noise (.7-2.5) '
                'to scale uncertainty in latent equity; trust (.5-1) to temper each street likelihood; '
                'mode balanced/pressure/cautious controls preflop. Supplied alternatives belong to THIS observation. '
                'For a rejected/stale action only, optionally give a legal current action. '
                'Reply JSON under 150 tokens: {"noise":1,"trust":0.85,"mode":"balanced","action":null,"lesson":"short evidence-based diagnosis"}.\n'
                +json.dumps(payload,separators=(',',':')))
        try:
            res=self.ctx.think(prompt,timeout_s=210.)
            raw=str(getattr(res,'text','') or '')
            a,b=raw.find('{'),raw.rfind('}')
            obj=json.loads(raw[a:b+1]) if 0<=a<b else {}
            if not isinstance(obj,dict): obj={}
            for key,lo,hi in (('noise',.7,2.5),('trust',.5,1.)):
                v=obj.get(key)
                if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v):
                    setattr(self,key,max(lo,min(hi,float(v))))
            if obj.get('mode') in ('balanced','pressure','cautious'): self.mode=obj['mode']
            override=legalize(o,obj.get('action'))
            self.ctx.memory['latent_audit']=obj
            self.ctx.log('latent_audit',reason=reason,reply=obj)
            if evidence and evidence.get('reason')=='execution did not advance' and override:
                return override
        except Exception as exc:
            self.ctx.log('latent_audit_failed',error=type(exc).__name__)
        finally:
            self.pending_failure=None
        return action

    def verify(self, old, new, action, before):
        progress=self.ctx.progress()
        rejected=(progress.get('illegal_actions',0)>before.get('illegal_actions',0) or new.get('action_error') or progress.get('since_obs_change',0)>=2)
        if rejected:
            self.pending_failure={'reason':'execution did not advance','action':action,'progress':progress,'error':new.get('action_error')}
        elif self.prediction and action.startswith('raise_to'):
            view=new if old.get('hand_id')==new.get('hand_id') else new.get('last_hand') or {}
            if view.get('hand_id')==old.get('hand_id'):
                response=next((e for e in view.get('history',[])[len(old.get('history',[]))+1:] if e.get('seat')!=old['hero_seat'] and e.get('street')==old['street']),None)
                if response:
                    prob=self.prediction.get(response['action'],1.)
                    if response['action'] in self.response_calibration['observed']:
                        stats=self.response_calibration
                        stats['n']+=1
                        stats['observed'][response['action']]+=1
                        for a in ('fold','call','raise_to'):
                            p=max(0.,min(1.,self.prediction.get(a,0.)))
                            stats['expected'][a]+=p
                            stats['variance'][a]+=p*(1-p)
                            stats['brier_sum']+=(p-float(a==response['action']))**2
                    self.ctx.log('latent_response_check',response=response['action'],probability=prob)
                    if prob<.06:
                        example={'hand':old.get('hand_id'),'hole_cards':old.get('hole_cards'),'board':old.get('board'),'history':view.get('history'),'prediction':self.prediction,'response':response}
                        self.surprises.append(example)
                        if len(self.surprises)%2==0:
                            self.pending_failure={'reason':'two low-probability responses','examples':self.surprises[-2:]}
        self.ctx.log('verified_policy_step',action=action,progress=progress)
        return rejected

    def record_result(self,o):
        last=o.get('last_hand') or {}
        if last.get('hand_id') and last.get('hand_id')!=self.last_result:
            self.last_result=last['hand_id']
            self.recent.append({k:last.get(k) for k in ('hand_id','hero_seat','hole_cards','board','revealed_hole_cards','net_chips','history')})
            self.recent=self.recent[-6:]

    def run_episode(self, game):
        ctx=self.ctx; o=ctx.observe(); next_review=35
        recovery=False; failure_step=-100
        while not ctx.done and not o.get('done',False):
            if self.should_stop(): return
            self.record_result(o)
            try:
                self.prediction=None
                if recovery:
                    self.table=[]
                    action=fallback(o)
                else:
                    self.update_range(o)
                    action=self.preflop(o) if o.get('street')=='preflop' else self.postflop(o)
                action=legalize(o,action) or fallback(o)
                due=o.get('completed_hands',0)>=next_review
                failed=self.pending_failure and ctx.progress().get('acts',0)-failure_step>=8
                if failed or due:
                    if failed: failure_step=ctx.progress().get('acts',0)
                    else: next_review+=35
                    old_noise,old_trust=self.noise,self.trust
                    action=self.review(o,action,'failure evidence' if failed else 'aggregate latent-policy review')
                    if (old_noise,old_trust)!=(self.noise,self.trust) and not recovery:
                        self.update_range(o)
                        action=self.preflop(o) if o.get('street')=='preflop' else self.postflop(o)
                if self.should_stop(): return
                before=ctx.progress(); old=o
                o=ctx.act(legalize(o,action) or fallback(o)) or ctx.observe()
                recovery=self.verify(old,o,action,before)
            except (_StopEpisode,BudgetExceeded):
                return
            except Exception as exc:
                if ctx.done: return
                self.pending_failure={'reason':'local computation exception','error':type(exc).__name__,'message':str(exc)[:150]}
                ctx.log('latent_recovery',failure=self.pending_failure)
                try:
                    o=ctx.observe()
                    self.table=[]; self.prediction=None
                    action=self.review(o,fallback(o),'computation failure recovery')
                    before=ctx.progress(); old=o
                    o=ctx.act(legalize(o,action) or fallback(o)) or ctx.observe()
                    recovery=self.verify(old,o,action,before)
                except Exception:
                    return


def build(ctx):
    return Harness(ctx)
