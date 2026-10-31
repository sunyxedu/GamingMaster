"""Complete-combination public-history inference with shared-runout equity tables."""
from __future__ import annotations

import bisect
import hashlib
import itertools
import json
import math
import random
import re

RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(range(52))


def enc(card):
    return RANKS.index(card[0]) * 4 + SUITS.index(card[1])


def straight(mask):
    for hi in range(12, 3, -1):
        wanted = 31 << (hi - 4)
        if mask & wanted == wanted:
            return hi + 2
    return 5 if mask & 0x100F == 0x100F else 0


def rank7(cards):
    """Best five-card rank from a 5--7 card integer tuple."""
    cnt = [0] * 15
    sm = 0
    suitmask = [0] * 4
    for c in cards:
        r = c // 4 + 2
        cnt[r] += 1
        sm |= 1 << (r - 2)
        suitmask[c % 4] |= 1 << (r - 2)
    flush = next((x for x in suitmask if x.bit_count() >= 5), 0)
    if flush:
        s = straight(flush)
        if s:
            return (8, s)
    quads = [r for r in range(14, 1, -1) if cnt[r] >= 4]
    if quads:
        q = quads[0]
        return (7, q, max(r for r in range(14, 1, -1) if r != q and cnt[r]))
    trips = [r for r in range(14, 1, -1) if cnt[r] >= 3]
    pairs = [r for r in range(14, 1, -1) if cnt[r] >= 2]
    if trips and len(pairs) >= 2:
        t = trips[0]
        p = next(r for r in pairs if r != t)
        return (6, t, p)
    if flush:
        return (5,) + tuple(r + 2 for r in range(12, -1, -1) if flush & (1 << r))[:5]
    s = straight(sm)
    if s:
        return (4, s)
    if trips:
        t = trips[0]
        return (3, t) + tuple(r for r in range(14, 1, -1) if r != t and cnt[r])[:2]
    if len(pairs) >= 2:
        p, q = pairs[:2]
        k = next(r for r in range(14, 1, -1) if r not in (p, q) and cnt[r])
        return (2, p, q, k)
    if pairs:
        p = pairs[0]
        return (1, p) + tuple(r for r in range(14, 1, -1) if r != p and cnt[r])[:3]
    return (0,) + tuple(r for r in range(14, 1, -1) if cnt[r])[:5]


def outcome(a, b):
    return 1.0 if a > b else 0.5 if a == b else 0.0

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def candidate_ids(obs):
    return [x.get("id") for x in obs.get("candidates", []) if isinstance(x, dict) and x.get("id")]


def safe_action(obs, action=None):
    ids = candidate_ids(obs)
    if action in ids:
        return action
    if isinstance(action, dict):
        if action.get("action") == "raise_to":
            try:
                n = int(action.get("amount"))
                exact = "raise_to " + str(n)
                if exact in ids:
                    return exact
                legal = obs.get("legal", {})
                if legal.get("can_raise") and legal.get("min_raise_to") <= n <= legal.get("max_raise_to"):
                    # Every integer is legal even when it was omitted from the compact menu.
                    return exact
            except Exception:
                pass
        elif action.get("action") in ids:
            return action.get("action")
    legal = obs.get("legal", {})
    for x in ("check", "call", "fold"):
        if x in ids and ((x == "check" and legal.get("can_check")) or
                         (x == "call" and legal.get("can_call")) or
                         (x == "fold" and legal.get("can_fold"))):
            return x
    return ids[0] if ids else None


def nearest_raise(obs, amount):
    ids = [x for x in candidate_ids(obs) if x.startswith("raise_to ")]
    legal = obs.get("legal", {})
    if not ids or not legal.get("can_raise"):
        return None
    amount = max(int(legal.get("min_raise_to", amount)), min(int(legal.get("max_raise_to", amount)), int(amount)))
    vals = []
    for x in ids:
        try:
            vals.append((abs(int(x.split()[-1]) - amount), x))
        except Exception:
            continue
    return min(vals)[1] if vals else None


def preflop_quality(cards):
    if len(cards) != 2:
        return 0.0
    a, b = sorted((c // 4 for c in cards), reverse=True)
    suited = cards[0] % 4 == cards[1] % 4
    pair = a == b
    q = 0.18 + 0.028 * a + 0.016 * b
    if pair:
        q += 0.17 + 0.012 * a
    if a == 12:
        q += 0.07
    if suited:
        q += 0.045
    if a - b <= 2:
        q += 0.035
    return max(0.0, min(0.82, q))


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.mode = "balanced"
        self.margin = 0.035
        self.value = 0.60
        self.bluff = 0.08
        self.last_hand = None
        self.last_hist = 0
        self.event_seen = set()
        self.seen_result = None
        self.actions = {"fold": 0, "call": 0, "raise_to": 0, "check": 0}
        self.opp = {"fold": 0, "call": 0, "raise_to": 0, "check": 0}
        self.reviewed = set()
        self.failure = None
        self.reviews = 0
        self.last_review_step = -100
        self.eq_cache = {}
        self.rng = random.Random(917331)

    def equity(self, obs):
        hole = tuple(enc(x) for x in obs.get("hole_cards", []) if isinstance(x, str) and len(x) >= 2)
        board = tuple(enc(x) for x in obs.get("board", []) if isinstance(x, str) and len(x) >= 2)
        key = (hole, board)
        if key in self.eq_cache:
            return self.eq_cache[key]
        if not hole:
            return 0.0
        if not board:
            q = preflop_quality(hole)
            self.eq_cache[key] = q
            return q
        avail = [c for c in DECK if c not in hole and c not in board]
        seed = int.from_bytes(hashlib.blake2s(bytes(hole + board), digest_size=8).digest(), "big")
        rng = random.Random(seed)
        n = 72 if len(board) < 5 else 120
        total = 0.0
        for _ in range(n):
            opp = tuple(rng.sample(avail, 2))
            rem = [c for c in avail if c not in opp]
            run = tuple(board) + tuple(rng.sample(rem, 5 - len(board)))
            total += outcome(rank7(hole + run), rank7(opp + run))
        q = total / n
        self.eq_cache[key] = q
        return q

    def observe_public(self, obs):
        hid = obs.get("hand_id")
        if hid != self.last_hand:
            self.last_hand = hid
            self.eq_cache = {}
        # A fold often advances directly into the next hand. Include the
        # public settlement history, deduplicating already-observed events.
        last = obs.get("last_hand") or {}
        for view in (last, obs):
            hand_id = view.get("hand_id")
            hero = view.get("hero_seat")
            for index, event in enumerate(view.get("history", [])):
                key = (hand_id, index)
                if key in self.event_seen:
                    continue
                self.event_seen.add(key)
                if event.get("seat") != hero:
                    action = event.get("action")
                    if action in self.opp:
                        self.opp[action] += 1

    def choose(self, obs):
        ids = candidate_ids(obs)
        if not ids:
            return None
        hole = tuple(enc(x) for x in obs.get("hole_cards", []) if isinstance(x, str) and len(x) >= 2)
        street = obs.get("street")
        hero = obs.get("hero_seat")
        button = obs.get("button_seat")
        to_call = int(obs.get("to_call", 0) or 0)
        pot = max(1, int(obs.get("pot", 1) or 1))
        q = self.equity(obs)
        if street == "preflop":
            quality = preflop_quality(hole)
            if hero == button and to_call <= 50:
                # Wide positional opening, with a small limp/checkback tail.
                cutoff = .30 if self.mode != "cautious" else .37
                if quality >= cutoff:
                    r = nearest_raise(obs, 250)
                    if r:
                        return r
                return safe_action(obs, "call" if "call" in ids else "fold")
            if to_call == 0:
                if q >= self.value:
                    r = nearest_raise(obs, max(200, int(.75 * pot)))
                    if r:
                        return r
                return safe_action(obs, "check")
            price = to_call / max(1, pot + to_call)
            if q >= .70:
                r = nearest_raise(obs, max(3 * to_call, 850))
                if r and q >= .77:
                    return r
            if q >= price + (.07 if self.mode == "cautious" else .035):
                return safe_action(obs, "call")
            return safe_action(obs, "fold")


class SearchStopped(Exception):
    pass


class CompleteRangeHarness(Harness):
    """Shared-runout equity tables with full public-history conditioning."""
    def __init__(self, ctx):
        super().__init__(ctx)
        self.tables = {}
        self.table_hand = None
        self.width = 1.0
        self.trust = .80
        self.risk = .035
        self.last_choices = []
        self.last_rows = []
        self.last_obs = None
        self.prediction = None
        self.surprises = []
        self.recent_results = []
        self.result_id = None
        self.next_review = 35
        self.local_errors = 0
        self.force_safe = False

    def stop_check(self):
        if self.ctx.done or self.ctx.budget.remaining().get('wall_s', 0) < 8:
            raise SearchStopped()

    def board_table(self, hero, board):
        """h -> (opponent uniform equity, hero conditional equity).

        Every completion uses a CDF with per-card exclusion, so all opposing
        hands are integrated exactly conditional on that completion. Averaging
        only completions compatible with each hole set gives its correct
        information set. Turn/river enumerate all completions, flop samples.
        """
        key = (hero, board)
        if key in self.tables:
            return self.tables[key]
        available = [c for c in DECK if c not in board]
        hands = list(itertools.combinations(available, 2))
        totals = {h: [0., 0, 0., 0] for h in hands}
        if len(board) == 5:
            runs = [()]
        elif len(board) == 4:
            runs = [(c,) for c in available]
        else:
            rng = random.Random(int.from_bytes(hashlib.blake2s(bytes(board), digest_size=8).digest(), 'big'))
            # Keep every legal opponent hole in the table.  Only the shared
            # flop runout quadrature is thinned as the wall budget is spent;
            # this avoids deleting rare premium holdings while preventing a
            # strong policy from ending with an incomplete fixed-denominator
            # score.  Turn and river remain exact over their completions.
            rem = self.ctx.budget.remaining().get('wall_s', 0)
            if rem > 1050:
                n_runs = 64
            elif rem > 650:
                n_runs = 40
            else:
                n_runs = 24
            total_runs = list(itertools.combinations(available, 2))
            runs = rng.sample(total_runs, min(n_runs, len(total_runs)))
        for run in runs:
            self.stop_check()
            final = board + run
            avail = [c for c in available if c not in run]
            ranked = [(rank7(h + final), h) for h in itertools.combinations(avail, 2)]
            ranked.sort()
            all_values = [r for r, h in ranked]
            by_card = {c: [] for c in avail}
            for r, h in ranked:
                by_card[h[0]].append(r)
                by_card[h[1]].append(r)
            hero_ok = not any(c in run for c in hero)
            hero_rank = rank7(hero + final) if hero_ok else None
            # Each completed board leaves 47 cards. For a given hole pair,
            # 45 choose 2 = 990 legal uniform opposing pairs remain.
            scores = {}
            for i, r in enumerate(all_values):
                if r not in scores:
                    scores[r] = .5 * (bisect.bisect_left(all_values, r) + bisect.bisect_right(all_values, r))
            for r, h in ranked:
                a = by_card[h[0]]; b = by_card[h[1]]
                s = scores[r] - .5 * (bisect.bisect_left(a, r) + bisect.bisect_right(a, r)
                                     + bisect.bisect_left(b, r) + bisect.bisect_right(b, r)) + .5
                acc = totals[h]
                acc[0] += s / 990.
                acc[1] += 1
                if hero_ok and hero[0] not in h and hero[1] not in h and not any(c in run for c in h):
                    acc[2] += outcome(hero_rank, r)
                    acc[3] += 1
        result = {h: (v[0] / max(1, v[1]), v[2] / max(1, v[3])) for h, v in totals.items()}
        self.tables[key] = result
        return result

    def action_likelihood(self, h, q, ev, price, own_raises, width, can_raise=True):
        action = ev.get('action')
        street = ev.get('street')
        p = max(1, ev.get('pot_before', 1))
        if street == 'preflop':
            hi, lo = sorted((c // 4 + 2 for c in h), reverse=True)
            premium = (hi == lo and hi >= 12) or (hi == 14 and lo == 13)
            quality = preflop_quality(h)
            if action == 'raise_to':
                if price > 50:
                    return .99 if premium else .01
                if ev.get('seat') == 1 and price > 0:
                    wide = hi == lo or hi >= 12 or (hi == 11 and lo >= 5) or (hi == 10 and lo >= 7) or h[0] % 4 == h[1] % 4 or (hi - lo <= 2 and hi >= 7)
                    return .99 if wide else .08
                return .03 + .94 * sigmoid((quality - .60) / .05)
            if action == 'call':
                threshold = max(.44, price / (p + price) + .08)
                return (.10 if premium and price > 100 and can_raise else 1.) * (.03 + .94 * sigmoid((quality - threshold) / .05))
            if action == 'check':
                return .03 + .94 * sigmoid((.60 - quality) / .05)
            return 1.
        w = width * self.width
        high = sigmoid((q - .85) / w)
        medium = max(0., sigmoid((q - .62) / w) - high)
        low = 1 - sigmoid((q - .62) / w)
        if action == 'check':
            return .008 + .92 * low
        if action == 'raise_to':
            if price > 0:
                return .008 + .984 * sigmoid((q - .80) / w)
            fraction = ev.get('cost', 0) / p
            # Rounding/minimum bet can merge different policy sizing branches.
            candidates = ((1., high), (.75, medium), (.5, .08 * low))
            observed = ev.get('cost', 0)
            matches = [prob for f, prob in candidates if abs(observed - max(100, int(f * p + .5))) <= 2]
            if matches:
                return .008 + sum(matches)
            if fraction >= .90:
                return .01 + .98 * high
            if fraction >= .60:
                return .01 + .98 * medium
            return .01 + .08 * low
        if action == 'call':
            threshold = price / (p + price) + (0 if street == 'river' else .05)
            likelihood = .008 + .984 * sigmoid((q - threshold) / w)
            if own_raises == 0 and can_raise:
                likelihood *= .025 + .975 * sigmoid((.80 - q) / w)
            return likelihood
        return 1.

    def posterior(self, obs):
        hero = tuple(enc(c) for c in obs['hole_cards'])
        board = tuple(enc(c) for c in obs['board'])
        table = self.board_table(hero, board)
        hands = [h for h in table if not any(c in hero for c in h)]
        weights = {h: 1. for h in hands}
        opponent = 1 - obs['hero_seat']
        bets = [100, 50]
        committed = [100, 50]
        street = 'preflop'
        raises = [0, 0]
        for event in obs.get('history', []):
            if event.get('street') != street:
                street = event.get('street')
                bets = [0, 0]
                raises = [0, 0]
            seat = event.get('seat')
            action = event.get('action')
            price = max(bets) - bets[seat]
            if seat == opponent:
                n = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}.get(street, 0)
                earlier = self.board_table(hero, board[:n]) if n else None
                width = .045 if n == 3 else .018 if n == 4 else .009
                can_raise = committed[1-seat] < 20000 and committed[seat] + price < 20000
                for h in hands:
                    q = earlier[h][0] if earlier else 0
                    like = self.action_likelihood(h, q, event, price, raises[seat], width, can_raise)
                    # Temper because repeated actions reuse the same noisy equity.
                    weights[h] *= max(.001, like) ** self.trust
            if action == 'raise_to':
                committed[seat] += event.get('cost', 0)
                bets[seat] = event.get('raise_to', bets[seat])
                raises[seat] += 1
            elif action == 'call':
                committed[seat] += event.get('cost', 0)
                bets[seat] += event.get('cost', 0)
        z = sum(weights.values())
        if z < 1e-30:
            self.failure = {'kind': 'range normalization collapsed', 'mass': z}
            weights = {h: 1. for h in hands}
            z = len(hands)
        return [(table[h][0], table[h][1], weights[h] / z) for h in hands]

    def response_probs(self, q, odds, street, raised, room):
        width = (.045 if street == 'flop' else .018 if street == 'turn' else .009) * self.width
        pr = .985 * sigmoid((q - .80) / width) if not raised and room else 0.
        pc = (1-pr) * (.008 + .984 * sigmoid((q - odds - (0 if street == 'river' else .05)) / width))
        return 1-pr-pc, pc, pr

    def solve(self, obs):
        rows = self.posterior(obs)
        self.last_rows = rows
        self.last_obs = obs
        hero = obs['hero_seat']; villain = 1 - hero
        street = obs['street']
        p = obs['pot']; call = obs['to_call']
        hb, vb = obs['street_bets'][hero], obs['street_bets'][villain]
        hs, vs = obs['stacks'][hero], obs['stacks'][villain]
        eq = sum(e * w for q, e, w in rows)
        discount = 0 if street == 'river' else self.risk + (.015 if hero == 0 else 0)
        eq_real = max(0, eq - discount)
        if call:
            choices = [('fold', 0., None), ('call', eq_real * (p + call) - call, None)]
        else:
            choices = [('check', eq_real * p, None)]
        raised = any(e.get('seat') == villain and e.get('street') == street and e.get('action') == 'raise_to' for e in obs.get('history', []))
        for action in candidate_ids(obs):
            if not action.startswith('raise_to '):
                continue
            target = min(int(action.split()[1]), hb + hs, vb + vs)
            cost = target - hb; opp_cost = max(0, target - vb)
            if cost <= call or opp_cost <= 0:
                continue
            odds = opp_cost / (p + cost + opp_cost)
            folds = calls_ev = rm = rw = cm = cw = 0.
            for q, e, w in rows:
                pf, pc, pr = self.response_probs(q, odds, street, raised, vs > opp_cost and hs > cost)
                e_real = max(0, e - discount)
                folds += w * pf
                calls_ev += w * pc * (e_real * (p + cost + opp_cost) - cost)
                cm += w * pc; cw += w * pc * e
                rm += w * pr; rw += w * pr * e_real
            reply = min(hb + hs, vb + vs, max(target + max(100, target-max(hb,vb)), int(target + .75*(p + cost + opp_cost) + .5)))
            final_cost = reply - hb; final_opp = reply - vb
            raise_ev = max(-cost*rm, rw * (p + final_cost + final_opp) - rm*final_cost)
            ev = folds*p + calls_ev + raise_ev
            called_eq = (cw + rw) / max(1e-12, cm + rm)
            # The inherited terminal solver cannot price later betting exactly.
            # Its very large early bets need an overwhelmingly strong caller edge.
            if cost > 1.5*(p + call) and called_eq < (.97 if street == 'river' else .96):
                continue
            ev -= (.015 if street == 'river' else .025) * cost
            choices.append((action, ev, {'fold': folds, 'call': cm, 'raise_to': rm, 'called_equity': called_eq}))
        legal_ids = candidate_ids(obs)
        choices = [x for x in choices if x[0] in legal_ids]
        chosen = max(choices, key=lambda x: x[1])
        self.last_choices = [{'action': a, 'ev': round(v, 2), 'response': pred} for a,v,pred in choices]
        self.prediction = chosen[2]
        self.ctx.log('complete_range_search', street=street, combinations=len(rows), equity=round(eq,4), choice=chosen[0], alternatives=self.last_choices)
        return chosen[0]

    def choose(self, obs):
        self.prediction = None
        if self.force_safe:
            self.force_safe = False
            return 'check' if 'check' in candidate_ids(obs) else 'fold'
        if obs.get('street') == 'preflop':
            return super().choose(obs)
        return self.solve(obs)

    def review(self, obs, reason, force=False):
        if self.reviews >= 6 or self.ctx.done:
            return
        rem = self.ctx.budget.remaining()
        if rem.get('system2_calls',0) < 2 or rem.get('wall_s',0) < 280:
            return
        steps = sum(self.actions.values())
        if steps-self.last_review_step < 12 and not (self.failure and self.failure.get('kind') == 'rejected or stale action'):
            return
        self.reviews += 1
        self.last_review_step = steps
        payload = {
            'reason': reason, 'failure': self.failure, 'progress': self.ctx.progress(),
            'parameters': {'width':self.width,'trust':self.trust,'risk':self.risk,'mode':self.mode},
            'observation': {k:obs.get(k) for k in ('hole_cards','board','street','hero_seat','pot','to_call','history','candidates','completed_hands')},
            'previous_solved_state': {k:(self.last_obs or {}).get(k) for k in ('hand_id','street','hole_cards','board','pot','to_call','history','candidates')},
            'previous_state_alternatives': self.last_choices[-8:], 'surprising_responses':self.surprises[-5:],
            'recent_public_results':self.recent_results[-3:], 'observed_opponent_action_counts':self.opp,
        }
        prompt = ('Review a full-combination inverse-policy poker solver against mc-pot-odds-v2. '
                  'Opponent uses uniform-card equity: free value bets .75pot at .62, pot at .85, '
                  '8% mixed half-pot branch; calls at pot odds+.05 before river, pot odds on river; '
                  'raises once/street at .80. Documentation may be approximate. Code retains all legal '
                  'holes, computes exact river/turn uniform equity and 64 shared flop runouts, and '
                  'conditions on all public actions with positive softened likelihoods. It solves '
                  'fold/call/bet and one aggregate reply-raise, but truncates future-street betting. '
                  'Detect model mismatch using actions, not wins alone. Counts include visible histories only; '
                  'missed automatic settlements mean they are not complete opportunity-based rates. '
                  'Do not compare raw folds with calls across different opportunities. Replan by selecting bounded '
                  'width (.7-2.5; likelihood smoothing), trust (.45-1; likelihood exponent), '
                  'risk (.01-.08; pre-river equity realization discount), mode (balanced/pressure/cautious). '
                  'Previous-state alternatives refer only to previous_solved_state, never the current observation. '
                  'EV is incremental from the decision with sunk commitments excluded: call EV=equity*(pot+call)-call. '
                  'For execution failure optionally supply a legal action for THIS observation. '
                  'Reply under 180 tokens as JSON only: '
                  '{"width":1,"trust":0.8,"risk":0.035,"mode":"balanced","action":null,"lesson":"short reason"}.\n'
                  + json.dumps(payload,separators=(',',':')))
        override = None
        try:
            result = self.ctx.think(prompt, timeout_s=180.)
            raw = str(getattr(result,'text','') or '')
            start,end = raw.find('{'),raw.rfind('}')
            obj = json.loads(raw[start:end+1]) if 0<=start<end else {}
            if not isinstance(obj,dict):
                obj = {}
            for key,lo,hi in (('width',.7,2.5),('trust',.45,1.),('risk',.01,.08)):
                value=obj.get(key)
                if isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value):
                    setattr(self,key,max(lo,min(hi,value)))
            if obj.get('mode') in ('balanced','pressure','cautious'):
                self.mode=obj['mode']
            if force and obj.get('action') in candidate_ids(obs):
                override=obj['action']
            self.ctx.memory['range_review']=obj
            self.ctx.log('range_review',reason=reason,reply=obj)
        except Exception as exc:
            self.ctx.log('range_review_failed',error=type(exc).__name__)
        self.failure=None
        return override

    def observe_public(self, obs):
        super().observe_public(obs)
        if obs.get('hand_id') != self.table_hand:
            self.table_hand=obs.get('hand_id')
            self.tables={}
        last=obs.get('last_hand') or {}
        if last.get('hand_id') and last.get('hand_id') != self.result_id:
            self.result_id=last['hand_id']
            self.recent_results.append({k:last.get(k) for k in ('hand_id','hole_cards','board','revealed_hole_cards','net_chips')})
            self.recent_results=self.recent_results[-6:]

    def verify_response(self, old, obs, action):
        if not self.prediction or not action.startswith('raise_to '):
            return
        if old.get('hand_id') == obs.get('hand_id'):
            hist=obs.get('history',[])
        else:
            last=obs.get('last_hand') or {}
            if last.get('hand_id') != old.get('hand_id'):
                return
            hist=last.get('history',[])
        response=next((e for e in hist[len(old.get('history',[])):] if e.get('seat') != old.get('hero_seat') and e.get('street') == old.get('street')),None)
        if not response:
            return
        act=response.get('action')
        probability=self.prediction.get(act,1.)
        self.ctx.log('response_check',response=act,predicted_probability=round(probability,4))
        if probability < .08:
            self.surprises.append({'hand':old.get('hand_id'),'board':old.get('board'),'action':action,'response':response,'prediction':self.prediction})
            if len(self.surprises)%3 == 0:
                self.failure={'kind':'three unlikely opponent responses','examples':self.surprises[-3:]}

    def run_episode(self, game):
        ctx=self.ctx
        obs=ctx.observe()
        while not ctx.done and not obs.get('done',False):
            try:
                self.stop_check()
                self.observe_public(obs)
                override=None
                hands=obs.get('completed_hands',0)
                if self.failure:
                    override=self.review(obs,'evidence-triggered replanning',True)
                elif hands>=self.next_review:
                    self.review(obs,'aggregate strategy review')
                    self.next_review+=35
                action=safe_action(obs,override or self.choose(obs))
                if action is None:
                    return
                before=ctx.progress()
                old=obs
                obs=ctx.act(action) or ctx.observe()
                progress=ctx.progress()
                self.actions[action.split()[0]]=self.actions.get(action.split()[0],0)+1
                bad=progress.get('illegal_actions',0)>before.get('illegal_actions',0) or obs.get('action_error') or progress.get('since_obs_change',0)>=2
                if bad:
                    self.failure={'kind':'rejected or stale action','action':action,'progress':progress,'error':obs.get('action_error')}
                    self.force_safe=True
                else:
                    self.verify_response(old,obs,action)
                ctx.log('verified_policy_step',action=action,progress=progress)
            except SearchStopped:
                return
            except Exception as exc:
                if ctx.done:
                    return
                self.local_errors+=1
                self.failure={'kind':'local calculation exception','error':type(exc).__name__,'message':str(exc)[:150]}
                ctx.log('range_recovery',**self.failure)
                self.force_safe=True
                self.tables={}
                try:
                    obs=ctx.observe()
                    override=self.review(obs,'computation failure recovery',True)
                    action=override or ('check' if 'check' in candidate_ids(obs) else 'fold')
                    action=safe_action(obs,action)
                    if not action:
                        return
                    obs=ctx.act(action) or ctx.observe()
                    progress=ctx.progress()
                    ctx.log('verified_recovery',action=action,progress=progress)
                except Exception:
                    return


def build(ctx):
    return CompleteRangeHarness(ctx)
