"""Inverse-policy range inference and response-conditioned value sizing.
Only public observations are consumed. No external evaluator or game imports.
"""
from __future__ import annotations
import bisect
import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = tuple(range(52))


def card(s):
    return RANKS.index(s[0]) * 4 + SUITS.index(s[1])


def rank7(cards):
    """Exact best-five rank for five to seven cards; larger is better."""
    counts = [0] * 13
    suits = [0] * 4
    mask = 0
    for c in cards:
        r = c >> 2
        counts[r] += 1
        suits[c & 3] |= 1 << r
        mask |= 1 << r
    def straight(m):
        z = m & (m >> 1) & (m >> 2) & (m >> 3) & (m >> 4)
        if z:
            return z.bit_length() + 3
        return 3 if m & 0x100f == 0x100f else -1
    def encode(cat, rs):
        n = cat
        for r in list(rs)[:5] + [-1] * (5 - len(rs)):
            n = n * 15 + r + 2
        return n
    flush = next((m for m in suits if m.bit_count() >= 5), 0)
    if flush:
        s = straight(flush)
        if s >= 0:
            return encode(8, [s])
    ordered = sorted((counts[r], r) for r in range(13) if counts[r])
    ordered.reverse()
    if ordered[0][0] == 4:
        q = ordered[0][1]
        return encode(7, [q, max(r for r in range(13) if r != q and counts[r])])
    trips = [r for n, r in ordered if n >= 3]
    if trips:
        pairs = [r for n, r in ordered if n >= 2 and r != trips[0]]
        if pairs:
            return encode(6, [trips[0], max(pairs)])
    if flush:
        return encode(5, [r for r in range(12, -1, -1) if flush >> r & 1][:5])
    s = straight(mask)
    if s >= 0:
        return encode(4, [s])
    if trips:
        return encode(3, [trips[0]] + [r for r in range(12, -1, -1) if counts[r] and r != trips[0]][:2])
    pairs = sorted([r for n, r in ordered if n == 2], reverse=True)
    if len(pairs) >= 2:
        return encode(2, pairs[:2] + [max(r for r in range(13) if counts[r] and r not in pairs[:2])])
    if pairs:
        return encode(1, pairs + [r for r in range(12, -1, -1) if counts[r] and r != pairs[0]][:3])
    return encode(0, [r for r in range(12, -1, -1) if counts[r]][:5])


def share(a, b):
    return 1.0 if a > b else 0.5 if a == b else 0.0


def stable_rng(value):
    return random.Random(int.from_bytes(hashlib.sha256(str(value).encode()).digest()[:8], 'big'))


def soft(x, threshold, width=.06):
    return max(.005, min(.995, .5 + (x - threshold) / (2 * width)))


def handclass(h):
    a, b = sorted((h[0] >> 2, h[1] >> 2), reverse=True)
    return a, b, (h[0] & 3) == (h[1] & 3)


def premium(h):
    a, b, s = handclass(h)
    return (a == b and a >= 10) or (a == 12 and b == 11)


def wide_open(h):
    a, b, s = handclass(h)
    return a == b or a >= 10 or s or a >= 7 and b >= 4 or a - b <= 2 and a >= 5


def fallback(o):
    l = o.get('legal', {})
    for a in ('check', 'fold', 'call'):
        if l.get('can_' + a):
            return a
    cs = o.get('candidates', [])
    return cs[0]['id'] if cs else 'wait'


def legal_action(a, o):
    if isinstance(a, dict):
        if a.get('action') == 'raise_to':
            a = 'raise_to ' + str(a.get('amount'))
        else:
            a = a.get('action')
    if not isinstance(a, str):
        return None
    a = a.strip().lower()
    l = o.get('legal', {})
    if a in ('fold', 'check', 'call') and l.get('can_' + a):
        return a
    m = re.fullmatch(r'raise_to\s+(\d+)', a)
    if m and l.get('can_raise'):
        n = int(m.group(1))
        if l['min_raise_to'] <= n <= l['max_raise_to']:
            return 'raise_to ' + str(n)
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.config = {'call_margin': .05, 'value_threshold': .62, 'raise_threshold': .80,
                       'bluff_rate': .08, 'uncertainty': .06, 'realization': .93,
                       'aggression_penalty': .025}
        self.pre = {}
        self.cache = {}
        self.last_hand = None
        self.calls = 0
        self.audit_at = 24
        self.events = []
        self.pending = None
        self.surprises = 0
        self.failure = None
        self.last_progress = 0
        self.seen_results = set()
        self.stats = {'predictions': 0, 'surprises': 0, 'raises': 0, 'calls': 0, 'folds': 0}

    def peq(self, h):
        key = handclass(h)
        if key not in self.pre:
            a, b, suited = key
            hh = (a * 4, b * 4 + (0 if suited else 1))
            rng = stable_rng(('preflop', key))
            deck = [c for c in DECK if c not in hh]
            value = 0.0
            for _ in range(256):
                cards = rng.sample(deck, 7)
                value += share(rank7(hh + tuple(cards[2:])), rank7(tuple(cards)))
            self.pre[key] = value / 256
        return self.pre[key]

    def table(self, hero, board):
        """For each legal opponent combo: uniform equity q and hero equity e.
        River q is exact (opponent correctly does not know hero cards).
        Earlier q/e share sampled runouts but use the right information sets.
        """
        key = (hero, board)
        if key in self.cache:
            return self.cache[key]
        deck = [c for c in DECK if c not in board]
        combos = list(itertools.combinations(deck, 2))
        out = {}
        if len(board) == 5:
            ranks = {h: rank7(h + board) for h in combos}
            all_ranks = sorted(ranks.values())
            by_card = {c: [] for c in deck}
            for h, r in ranks.items():
                by_card[h[0]].append(r)
                by_card[h[1]].append(r)
            for rs in by_card.values():
                rs.sort()
            hr = rank7(hero + board)
            for h, r in ranks.items():
                if hero[0] in h or hero[1] in h:
                    continue
                lo = bisect.bisect_left(all_ranks, r)
                eq = bisect.bisect_right(all_ranks, r) - lo
                for c in h:
                    rs = by_card[c]
                    z = bisect.bisect_left(rs, r)
                    lo -= z
                    eq -= bisect.bisect_right(rs, r) - z
                eq += 1
                out[h] = ((lo + .5 * eq) / 990, share(hr, r))
        else:
            n = 48 if len(board) == 4 else 40
            need = 5 - len(board)
            for i, h in enumerate(combos):
                if i % 64 == 0 and self.ctx.done:
                    return out
                if hero[0] in h or hero[1] in h:
                    continue
                rng = stable_rng(('equity', h, board))
                available = [c for c in deck if c not in h]
                q = e = count = 0.0
                for _ in range(n):
                    draw = rng.sample(available, need + 2)
                    runout = tuple(draw[:need])
                    b = board + runout
                    vr = rank7(h + b)
                    q += share(vr, rank7(tuple(draw[need:]) + b))
                    if hero[0] not in runout and hero[1] not in runout:
                        e += share(rank7(hero + b), vr)
                        count += 1
                out[h] = (q / n, e / max(1, count))
        self.cache[key] = out
        return out

    def likelihood(self, h, q, event, own_raised, facing, street):
        a = event['action']
        if street == 'preflop':
            if a == 'raise_to':
                if facing <= 100:
                    return .98 if wide_open(h) else .05
                return .98 if premium(h) else .015
            if a == 'call':
                price = event.get('cost', 0) / max(1, event.get('pot_before', 0) + event.get('cost', 0))
                return (.02 if premium(h) else 1) * soft(self.peq(h), max(.44, price + .08), .045)
            if a == 'check':
                return 1 - soft(self.peq(h), .60, .05)
            return 1.0
        cfg = self.config
        width = cfg['uncertainty'] if street != 'river' else .025
        if a == 'check':
            return max(.01, (1 - soft(q, cfg['value_threshold'], width)) * (1 - cfg['bluff_rate']))
        if a == 'raise_to':
            if facing > 0:
                return .015 + .985 * soft(q, cfg['raise_threshold'], width)
            fraction = event.get('cost', 0) / max(1, event.get('pot_before', 0))
            value = soft(q, cfg['value_threshold'], width)
            nut = soft(q, .85, width)
            if fraction >= .9:
                return .01 + .99 * nut
            if fraction >= .62:
                return .01 + .99 * value * (1 - nut)
            return .015 + cfg['bluff_rate'] * (1 - value)
        if a == 'call':
            price = event.get('cost', 0) / max(1, event.get('pot_before', 0) + event.get('cost', 0))
            p = soft(q, price + (0 if street == 'river' else cfg['call_margin']), width)
            # If already raised on this street, the policy cannot raise again.
            if not own_raised:
                p *= 1 - .97 * soft(q, cfg['raise_threshold'], width)
            return max(.01, p)
        return 1.0

    def posterior(self, o):
        hero = tuple(card(c) for c in o['hole_cards'])
        board = tuple(card(c) for c in o['board'])
        current = self.table(hero, board)
        weights = {h: 1.0 for h in current}
        street_bets = [o.get('big_blind', 100), o.get('small_blind', 50)]
        prev_street = 'preflop'
        raised = [False, False]
        bs = {'preflop': 0, 'flop': 3, 'turn': 4, 'river': 5}
        for ev in o.get('history', []):
            street = ev['street']
            seat = ev['seat']
            if street != prev_street:
                street_bets = [0, 0]
                raised = [False, False]
                prev_street = street
            facing = max(street_bets) - street_bets[seat]
            if seat != o['hero_seat']:
                tab = None if street == 'preflop' else self.table(hero, board[:bs[street]])
                for h in weights:
                    q = tab.get(h, (.5, .5))[0] if tab is not None else 0
                    weights[h] *= self.likelihood(h, q, ev, raised[seat], facing, street)
            if ev['action'] == 'raise_to':
                raised[seat] = True
                street_bets[seat] = int(ev['raise_to'])
            elif ev['action'] == 'call':
                street_bets[seat] += int(ev.get('cost', 0))
        total = sum(weights.values()) or 1
        # Tiny contamination prevents treating approximate likelihoods as facts.
        rows = [(h, (w / total) * .99 + .01 / max(1, len(weights)), *current[h]) for h, w in weights.items()]
        return rows

    def preflop(self, o):
        h = tuple(card(c) for c in o['hole_cards'])
        a, b, suited = handclass(h)
        q = self.peq(h)
        call = o['to_call']
        pot = o['pot']
        mx = max(o['street_bets'])
        raises = [x for x in o['history'] if x['action'] == 'raise_to']
        legal = o['legal']
        def raise_to(target):
            if not legal.get('can_raise'):
                return 'call' if legal.get('can_call') else fallback(o)
            return 'raise_to ' + str(max(legal['min_raise_to'], min(legal['max_raise_to'], int(target))))
        if not raises and o['hero_seat'] == 1:
            if wide_open(h) or q >= .42:
                return raise_to(250)
            return 'fold'
        if not raises:
            return raise_to(450) if q >= .59 else fallback(o)
        if len(raises) == 1 and raises[0]['seat'] != o['hero_seat']:
            if a == b and a >= 8 or a >= 11 and b >= 10:
                return raise_to(mx * 4)
            return 'call' if q >= max(.38, call / (pot + call) + .10) else fallback(o)
        # Reraises are premium-heavy; compare to QQ+/AK, not random holdings.
        avail = [c for c in DECK if c not in h]
        opp = [x for x in itertools.combinations(avail, 2) if premium(x)]
        rng = stable_rng(('premium-call', h))
        e = 0
        for _ in range(384):
            v = rng.choice(opp)
            board = tuple(rng.sample([c for c in avail if c not in v], 5))
            e += share(rank7(h + board), rank7(v + board))
        e /= 384
        if a == b == 12:
            return raise_to(mx * 3 if mx < 4000 else legal.get('max_raise_to', mx))
        odds = call / max(1, pot + call)
        if e >= odds + (.035 if call < 3000 else .015):
            return 'call'
        # Small pairs may cheaply buy a set against an overcommitted premium range.
        if a == b and call <= 500 and o['effective_stack'] >= call * 24:
            return 'call'
        return fallback(o)

    def choose(self, o):
        if o['street'] == 'preflop':
            action = self.preflop(o)
            return action, {'kind': 'preflop-range'}, None
        rows = self.posterior(o)
        if not rows:
            return fallback(o), {'kind': 'empty-range'}, None
        p, c = o['pot'], o['to_call']
        hero = o['hero_seat']
        vil = 1 - hero
        hb, vb = o['street_bets']
        hb, vb = o['street_bets'][hero], o['street_bets'][vil]
        river = o['street'] == 'river'
        realization = 1.0 if river else self.config['realization']
        equity = sum(w * e for h, w, q, e in rows)
        # Approximation is explicit: later streets are re-solved on fresh evidence.
        values = {}
        if o['legal'].get('can_fold'):
            values['fold'] = 0.0
        if o['legal'].get('can_check'):
            values['check'] = equity * p * realization
        if o['legal'].get('can_call'):
            values['call'] = equity * (p + c) * realization - c
        probs = {}
        targets = {x['amount'] for x in o.get('candidates', []) if x.get('action') == 'raise_to'}
        if o['legal'].get('can_raise'):
            lo, hi = o['legal']['min_raise_to'], o['legal']['max_raise_to']
            targets.update(max(lo, min(hi, round(max(hb, vb) + f * (p + c)))) for f in (2.0, 3.0, 5.0))
        own_raised = any(x['seat'] == vil and x['street'] == o['street'] and x['action'] == 'raise_to' for x in o['history'])
        for target in sorted(targets):
            cost = target - hb
            vc = min(max(0, target - vb), o['stacks'][vil])
            effective_cost = min(cost, vb + o['stacks'][vil] - hb)
            finalpot = p + effective_cost + vc
            odds = vc / max(1, finalpot)
            width = .025 if river else self.config['uncertainty']
            ev = pf = pc = pr = raise_eq = 0.0
            for h, w, q, e in rows:
                callprob = soft(q, odds + (0 if river else self.config['call_margin']), width)
                raisep = 0 if own_raised or vc >= o['stacks'][vil] or cost >= o['stacks'][hero] else soft(q, self.config['raise_threshold'], width)
                cp = (1 - raisep) * callprob
                fp = (1 - raisep) * (1 - callprob)
                pf += w * fp
                pc += w * cp
                pr += w * raisep
                raise_eq += w * raisep * e
                ev += w * (fp * p + cp * (e * finalpot * realization - effective_cost))
            if pr:
                r_extra = min(o['stacks'][vil] - vc, o['stacks'][hero] - cost, round(.75 * finalpot))
                r_extra = max(0, r_extra)
                re = raise_eq / pr
                # A single common continuation decision, never one per hidden world.
                cont = max(-effective_cost, re * (finalpot + 2 * r_extra) * realization - effective_cost - r_extra)
                ev += pr * cont
            # Penalize large decisions with uncertain range/rollout estimates.
            penalty = 0 if river else self.config['aggression_penalty'] * effective_cost * (1 - equity)
            ev -= penalty
            action = 'raise_to ' + str(target)
            values[action] = ev
            probs[action] = {'fold': pf, 'call': pc, 'raise_to': pr}
        action = max(values, key=values.get)
        top = sorted(values.items(), key=lambda x: -x[1])[:5]
        info = {'kind': 'inverse-policy', 'equity': round(equity, 4), 'pot_odds': round(c / max(1, p+c), 4),
                'top_ev': [(a, round(v, 1)) for a, v in top], 'combos': len(rows),
                'effective_combos': round(1 / sum(w*w for h,w,q,e in rows), 1),
                'assumption': 'one response then showdown; realization discounted before river',
                'response': probs.get(action)}
        return action, info, probs.get(action)

    def ask(self, o, reason, action, info):
        remaining = self.ctx.budget.remaining()
        if self.calls >= 5 or remaining.get('system2_calls', 0) <= 1 or remaining.get('wall_s', 0) < 400:
            return action
        self.calls += 1
        prompt = (
            'You supervise an inverse-policy poker harness, heads-up 200bb fresh stacks, net chips/100 hands. '
            'Opponent mc-pot-odds-v2 uses uniform-random-hand equity, wide 250 button opens, QQ+/AK reraises, '
            'calls preflop at max(.44, odds+.08), free postflop value bets .75pot at q>=.62 and pot at q>=.85, '
            '8% half-pot mixed bets, one .75pot raise per street at q>=.80, calls at odds+.05 (river odds). '
            'These are study priors, not guaranteed current code; public description mentions blocker semibluffs. '
            'Our code infers a combination range from all observed actions and maximizes response-conditioned '
            'one-street EV, using exact river ranks/equities and 40/48 Monte Carlo samples per combo before river. '
            'It allows 2x/3x/5x pot and all-in value bets because opponent ignores our value range. '
            'Check/call EV discounts realization; raising evaluates a common call/fold decision against a reraise. '
            'Future streets are not fully searched. Do not mistake random-range equity for equity against a betting range. '
            'Audit or repair the policy from evidence. Do not chase losses or treat one lost showdown as failure. '
            'Return ONE compact JSON object: {"action":"legal action for current turn", '
            '"config":{"call_margin":0.05,"value_threshold":0.62,"raise_threshold":0.80,'
            '"bluff_rate":0.08,"uncertainty":0.06,"realization":0.93,"aggression_penalty":0.025},'
            '"reason":"short hypothesis"}. Only change parameters supported by evidence; omitted keys retain defaults. '
            'Action must be fold/check/call/raise_to N total CURRENT street. Under 180 words.\n' +
            json.dumps({'trigger': reason, 'observation': o, 'proposed_action': action, 'analysis': info,
                        'config': self.config, 'stats': self.stats, 'recent_evidence': self.events[-5:]}, separators=(',', ':')))
        try:
            res = self.ctx.think(prompt, session_id=None, timeout_s=150)
            s = res.text.strip()
            start, end = s.find('{'), s.rfind('}')
            obj = json.loads(s[start:end+1]) if start >= 0 else {}
            bounds = {'call_margin': (0,.12), 'value_threshold': (.55,.75), 'raise_threshold': (.72,.94),
                      'bluff_rate': (.01,.25), 'uncertainty': (.025,.15), 'realization': (.8,1),
                      'aggression_penalty': (0,.10)}
            for k, v in obj.get('config', {}).items():
                if k in bounds and isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v):
                    self.config[k] = max(bounds[k][0], min(bounds[k][1], v))
            a = legal_action(obj.get('action'), o)
            self.ctx.memory['inverse_policy'] = {'config': self.config.copy(), 'reason': str(obj.get('reason', ''))[:600]}
            self.ctx.log('policy_audit', trigger=reason, reply=obj)
            return a or action
        except Exception as exc:
            self.ctx.log('audit_unavailable', error=type(exc).__name__)
            return action

    def verify(self, before, after, action, prediction):
        progress = self.ctx.progress()
        illegal = int(progress.get('illegal_actions', 0))
        if illegal > self.last_progress or progress.get('since_obs_change', 0) >= 2 or after.get('action_error'):
            self.failure = {'reason': 'rejected or nonadvancing action', 'progress': progress, 'action': action,
                            'error': after.get('action_error')}
        self.last_progress = illegal
        history = after.get('history', []) if after.get('hand_id') == before.get('hand_id') else (after.get('last_hand') or {}).get('history', [])
        if prediction and (after.get('hand_id') == before.get('hand_id') or (after.get('last_hand') or {}).get('hand_id') == before.get('hand_id')):
            tail = history[len(before.get('history', [])) + 1:]
            response = next((e for e in tail if e.get('seat') != before['hero_seat'] and e.get('street') == before['street']), None)
            if response:
                actual = response['action']
                probability = prediction.get(actual, 0)
                self.stats['predictions'] += 1
                if actual in ('fold','call','raise_to'):
                    self.stats[{'fold':'folds','call':'calls','raise_to':'raises'}[actual]] += 1
                if probability < .10:
                    self.stats['surprises'] += 1
                    self.surprises += 1
                    self.events.append({'prediction_miss': actual, 'predicted_probability': round(probability,3),
                                        'cards': before['hole_cards'], 'board': before['board'], 'action': action})
                    if self.surprises >= 3:
                        self.failure = {'reason': 'three unlikely opponent responses', 'stats': self.stats.copy()}
                        self.surprises = 0
        result = after.get('last_hand') or {}
        rid = result.get('hand_id')
        if rid and rid not in self.seen_results:
            self.seen_results.add(rid)
            if abs(result.get('net_chips', 0)) >= 2500 or result.get('revealed_hole_cards'):
                self.events.append({'settlement': result})
        self.events = self.events[-12:]

    def run_episode(self, game):
        ctx = self.ctx
        o = ctx.observe()
        steps = 0
        while not ctx.done and not o.get('done'):
            if o.get('hand_id') != self.last_hand:
                self.last_hand = o.get('hand_id')
                self.cache = {}
            try:
                action, info, prediction = self.choose(o)
            except Exception as exc:
                self.failure = {'reason': 'calculation failure', 'error': type(exc).__name__, 'message': str(exc)[:200]}
                action, info, prediction = fallback(o), {}, None
            if ctx.done:
                break
            reason = self.failure
            if steps == 0:
                reason = 'initial audit: verify priors against the supplied observation and establish policy'
            elif o.get('completed_hands', 0) >= self.audit_at:
                reason = 'scheduled public-evidence policy audit'
                self.audit_at += 30
            if reason:
                old_action = action
                action = self.ask(o, reason, action, info)
                if action != old_action:
                    prediction = None
                self.failure = None
            if ctx.done:
                break
            action = legal_action(action, o) or fallback(o)
            ctx.log('solver_decision', action=action, analysis=info)
            try:
                nxt = ctx.act(action)
                if nxt is None:
                    nxt = ctx.observe()
            except Exception as exc:
                if ctx.done:
                    break
                self.failure = {'reason': 'action exception', 'error': type(exc).__name__}
                nxt = ctx.observe()
            self.verify(o, nxt, action, prediction)
            o = nxt
            steps += 1
            if steps > 2500:
                break


def build(ctx):
    return Harness(ctx)
