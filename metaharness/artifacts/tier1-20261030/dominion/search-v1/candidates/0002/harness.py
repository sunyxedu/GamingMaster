"""Money/attack specialist with selective Astra critiques and exact local choices.
Only public observations are used. No cached occurrence IDs or future deck orders.
"""
from __future__ import annotations

import itertools
import json
import time
from concurrent.futures import TimeoutError as FutureTimeout
from collections import Counter

MONEY = {'Copper': 1, 'Silver': 2, 'Gold': 3}
CANTRIPS = {'Village', 'Festival', 'Market', 'Laboratory', 'Sentry', 'Merchant', 'Cellar'}
TERMINALS = {'Smithy', 'Militia', 'Witch', 'Moat', 'Chapel', 'Workshop', 'Remodel', 'Mine'}
VP = {'Estate': 1, 'Duchy': 3, 'Province': 6, 'Curse': -1}


def points(cards):
    return sum(cards.get(c, 0) * v for c, v in VP.items()) + cards.get('Gardens', 0) * (sum(cards.values()) // 10)


def money(cards):
    return sum(cards.get(c, 0) * v for c, v in MONEY.items())


def parse_object(text):
    if not isinstance(text, str):
        return None
    dec = json.JSONDecoder()
    for i, c in enumerate(text):
        if c == '{':
            try:
                obj, _ = dec.raw_decode(text[i:])
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                pass
    return None


class Policy:
    def __init__(self):
        self.route = None
        self.last_remodel = None

    def setup(self, o):
        if self.route is None:
            # Dev traces refuted an automatic Gardens rush. The safe default is
            # the money specialist; only a valid Astra critique can opt into rush.
            self.route = 'money'

    def state(self, o):
        self.setup(o)
        own = Counter(o['self']['owned'])
        supply = o['supply']
        p = supply['Province']['count']
        turn = o.get('turns_taken', {}).get('agent', 1)
        return own, supply, p, turn

    def junk(self, own):
        return own['Curse'] + own['Estate'] + own['Copper']

    def end_after(self, o, gains):
        supply = o['supply']
        left = {c: s['count'] - gains.get(c, 0) for c, s in supply.items()}
        return left.get('Province', 1) <= 0 or sum(v <= 0 for v in left.values()) >= 3

    def margin_after(self, o, gains):
        owned = Counter(o['self']['owned']) + Counter(gains)
        return points(owned) - o['opponent']['vp']

    def winning_margin(self, o, margin):
        turns = o.get('turns_taken', {})
        return margin > 0 or (margin == 0 and turns.get('agent', 0) < turns.get('opponent', 0))

    def gain_value(self, c, o, free=False):
        own, supply, p, turn = self.state(o)
        n = sum(own.values())
        curses = supply.get('Curse', {}).get('count', 0)
        if self.route == 'gardens':
            if c == 'Province': return 30
            if c == 'Gardens': return 24 if own['Workshop'] >= 2 or turn > 4 else 14
            if c == 'Workshop':
                # Workshop is also the third pile: Gardens, Estate, Workshop.
                if own[c] < 2: return 29
                if supply['Gardens']['count'] == 0: return 25 if supply['Estate']['count'] == 0 else 16
                return 19
            if c == 'Village': return 21 if own[c] < min(2, max(1, own['Workshop'] - 1)) and own['Workshop'] >= 2 and supply['Gardens']['count'] >= 3 else 3
            if c == 'Market': return 20 if p > 3 else 9
            if c == 'Duchy': return 23 if p <= 4 or supply['Gardens']['count'] <= 2 else 13
            if c == 'Estate': return 22 if supply['Gardens']['count'] == 0 else (17 if own['Gardens'] >= 3 or supply['Gardens']['count'] <= 3 else 4)
            if c == 'Silver': return 13 if own[c] < 3 else 5
            if c == 'Gold': return 19
            if c == 'Copper': return 2 if own['Gardens'] >= 2 else -1
            if c == 'Militia': return 15 if own[c] == 0 and own['Workshop'] >= 2 and own['Village'] >= 1 else -2
            if c == 'Smithy': return 11 if own[c] == 0 and own['Village'] else -2
            return -3
        terminal = sum(own[c] for c in ('Smithy', 'Militia', 'Witch', 'Moat'))
        if c == 'Province': return 100
        if c == 'Duchy': return 85 if p <= 4 else -8
        if c == 'Estate': return 75 if p <= 2 else -12
        if c == 'Gardens': return 80 if p <= 3 and n >= 30 else -8
        if c == 'Witch':
            if curses >= 3 and p >= 4 and own[c] == 0: return 96
            if curses >= 7 and p >= 6 and own[c] == 1 and (own['Village'] or n >= 22): return 67
            return -6
        if c == 'Chapel': return 94 if own[c] == 0 and turn <= 3 else -10
        if c == 'Sentry':
            if own[c] == 0 and p >= 4 and self.junk(own) >= 5:
                return 89 if not own['Chapel'] or own['Curse'] >= 2 else 64
            return -3
        if c == 'Gold': return 80
        if c == 'Laboratory': return 76 if p > 3 and own[c] < max(2, n // 5) else 40
        if c == 'Market': return 69 if p > 3 and own[c] < 2 and money(own) >= n else 40
        if c == 'Smithy': return 72 if terminal == 0 and p >= 4 else (63 if own[c] == 0 and own['Village'] and p >= 5 else -3)
        if c == 'Militia': return 74 if terminal == 0 and p >= 4 else -3
        if c == 'Moat':
            return 62 if terminal == 0 and p >= 5 else -3
        if c == 'Village': return 65 if terminal >= 2 and own[c] < terminal - 1 and p >= 4 else -6
        if c == 'Silver': return 60 if p > 2 else 30
        if c == 'Merchant': return 54 if own['Silver'] and own[c] == 0 and p >= 5 else -3
        if c == 'Festival': return 61 if terminal >= 2 and own['Village'] == 0 and own[c] == 0 and p >= 5 else -3
        if c == 'Cellar': return 18 if own[c] == 0 and n >= 16 and own['Estate'] + own['Curse'] + own['Province'] >= 5 else -3
        if c == 'Remodel': return 58 if own[c] == 0 and not own['Chapel'] and not own['Sentry'] and own['Estate'] >= 2 and turn <= 5 else -3
        if c == 'Mine': return -3
        return -15

    def terminal_baskets(self, o):
        """Enumerate current-turn baskets. Finish after all buys, not first empty pile.
        Bounded to six buys / 12000 nodes; returned baskets are fully affordable.
        """
        supply = o['supply']
        coins = max(0, o['resources'].get('coins', 0))
        buys = min(6, max(0, o['resources'].get('buys', 1)))
        cards = [c for c, s in supply.items() if s['count'] > 0 and s['cost'] <= coins and
                 (c in VP and c != 'Curse' or c == 'Gardens' or s['count'] <= buys or c == 'Copper')]
        cards.sort(key=lambda c: (-supply[c]['cost'], c))
        best = None
        count = 0
        gains = Counter()
        def rec(start, cash, slots):
            nonlocal best, count
            count += 1
            if count > 12000: return
            if gains and self.end_after(o, gains):
                margin = self.margin_after(o, gains)
                candidate = (margin, -sum(gains.values()), dict(gains))
                if best is None or candidate[:2] > best[:2]: best = candidate
            if slots == 0: return
            for i in range(start, len(cards)):
                c = cards[i]
                if supply[c]['cost'] <= cash and gains[c] < supply[c]['count']:
                    gains[c] += 1
                    rec(i, cash - supply[c]['cost'], slots - 1)
                    gains[c] -= 1
                    if not gains[c]: del gains[c]
        rec(0, coins, buys)
        if best is not None and self.winning_margin(o, best[0]):
            return best[2], best[0]
        return None

    def buy(self, o):
        p = o['prompt']
        options = p['options']
        ending = self.terminal_baskets(o)
        if ending:
            basket, _ = ending
            return [next(str(x['id']) for x in options if x['label'] in basket)]
        candidates = [(self.gain_value(x['label'], o), x) for x in options]
        # Do not lose immediately if any non-ending purchase or pass is available.
        safe = [(v, x) for v, x in candidates if not self.end_after(o, {x['label']: 1}) or
                self.winning_margin(o, self.margin_after(o, {x['label']: 1}))]
        if safe: candidates = safe
        if not candidates: return []
        v, x = max(candidates, key=lambda t: t[0])
        return [str(x['id'])] if v > 0 else []

    def trash(self, o):
        p = o['prompt']
        opts = p['options']
        source = p.get('source')
        own, supply, provinces, turn = self.state(o)
        if source == 'Mine':
            for c in ('Silver', 'Copper'):
                if any(x['label'] == c for x in opts):
                    return [str(next(x['id'] for x in opts if x['label'] == c))]
            return []
        if source == 'Remodel':
            best = None
            for x in opts:
                c = x['label']; cost = supply.get(c, {}).get('cost', 0)
                eligible = [d for d, s in supply.items() if s['count'] and s['cost'] <= cost + 2]
                if not eligible: continue
                vals = []
                for d in eligible:
                    delta = Counter(own); delta[c] -= 1; delta[d] += 1
                    end = self.end_after(o, {d: 1})
                    margin = points(delta) - o['opponent']['vp']
                    value = self.gain_value(d, o, True) - (0 if c in ('Curse', 'Estate') else self.keep_value(c, o) * 10)
                    if end: value = 1000 + margin if self.winning_margin(o, margin) else -1000 + margin
                    if c == 'Province' and d != 'Province': value -= 100
                    vals.append((value, d))
                value, target = max(vals)
                if c == 'Curse': value += 35
                if c == 'Estate': value += 20
                item = (value, str(x['id']), target)
                if best is None or item[0] > best[0]: best = item
            if best:
                self.last_remodel = best[2]
                return [best[1]]
            return [str(opts[0]['id'])] if opts else []
        # Enumerate trash subsets so Copper removal preserves total and immediate economy.
        best = (0.0, [])
        cap = min(p.get('max', len(opts)), 4)
        for k in range(1, cap + 1):
            for inds in itertools.combinations(range(len(opts)), k):
                removed = Counter(opts[i]['label'] for i in inds)
                after = own - removed
                if removed['Copper'] and money(after) < 6: continue
                value = 0.0
                for c, n in removed.items():
                    if c == 'Curse': v = 12
                    elif c == 'Estate': v = 7 if provinces >= 4 and self.route != 'gardens' else -3
                    elif c == 'Copper': v = 4 if provinces >= 4 and self.route != 'gardens' else -1
                    elif c == 'Chapel': v = 2 if self.junk(own) <= 2 else -8
                    else: v = -15
                    value += v * n
                if source == 'Chapel':
                    hand_money = money(o['self'].get('hand', {}))
                    lost = money(removed)
                    before_tier = self.purchase_power(hand_money, provinces)
                    after_tier = self.purchase_power(hand_money - lost, provinces)
                    value -= 1.4 * (before_tier - after_tier)
                if value > best[0]: best = (value, [str(opts[i]['id']) for i in inds])
        return best[1]

    def purchase_power(self, cash, provinces):
        # Concave thresholds, with stronger imminent scoring incentives.
        if cash >= 8: return 12
        if cash >= 6: return 7
        if cash >= 5: return 6 if provinces <= 4 else 4
        if cash >= 3: return 3
        if cash >= 2: return 2 if provinces <= 2 else 0
        return 0

    def keep_value(self, c, o):
        own, supply, p, _ = self.state(o)
        if c in MONEY: return MONEY[c]
        if c in VP or c == 'Gardens': return -0.1
        if c == 'Witch': return 4.2 if supply['Curse']['count'] else 2.0
        if c == 'Militia': return 3.2
        if c == 'Smithy': return 3.0
        if c == 'Moat': return 2.1
        if c == 'Laboratory': return 4.1
        if c == 'Market': return 3.5
        if c == 'Sentry': return 3.9
        if c == 'Merchant': return 2.3
        if c == 'Workshop': return 4 if self.route == 'gardens' else 1.4
        if c == 'Chapel': return 3 if self.junk(own) >= 5 and p >= 4 else 0
        if c == 'Festival': return 2.7
        if c == 'Village': return 1.3
        if c == 'Cellar': return 0.5
        return 1.0

    def retained_value(self, cards, o):
        """Evaluate a retained Militia hand including terminal collisions and villages."""
        c = Counter(cards)
        value = money(c)
        actions = 1 + c['Village'] + c['Festival']
        terminal_values = []
        for name, n in c.items():
            if name in MONEY: continue
            if name in TERMINALS:
                v = self.keep_value(name, o)
                if name == 'Chapel':
                    v = min(4, 1.8 * (c['Estate'] + c['Curse']) + 0.5 * c['Copper'])
                terminal_values += [v] * n
            elif name in CANTRIPS:
                v = self.keep_value(name, o)
                if name == 'Village': v = 0.9
                value += v * n
        value += sum(sorted(terminal_values, reverse=True)[:actions])
        return value

    def discard(self, o):
        p = o['prompt']; opts = p['options']; source = p.get('source')
        lo = p.get('min', 0); hi = p.get('max', len(opts))
        if lo > 0:
            # Mandatory discard: optimize retained hand as a whole.
            if len(opts) <= 15:
                choices = itertools.combinations(range(len(opts)), min(lo, len(opts)))
                best = max(choices, key=lambda inds: self.retained_value(
                    [x['label'] for i, x in enumerate(opts) if i not in inds], o), default=())
                return [str(opts[i]['id']) for i in best]
            return [str(x['id']) for x in sorted(opts, key=lambda x: self.keep_value(x['label'], o))[:lo]]
        selected = []
        hand = Counter(o['self'].get('hand', {}))
        own = Counter(o['self']['owned'])
        for x in opts:
            c = x['label']
            bad = c in VP or c == 'Gardens'
            if c == 'Chapel' and self.junk(own) <= 3: bad = True
            if c == 'Copper' and source == 'Sentry' and money(own) / max(1, sum(own.values())) > 1.25: bad = True
            if source == 'Cellar' and c in TERMINALS and sum(hand[t] for t in TERMINALS) > o['resources'].get('actions', 1):
                if any(self.keep_value(t, o) > self.keep_value(c, o) and hand[t] for t in TERMINALS): bad = True
            if bad: selected.append(str(x['id']))
        return selected[:hi]

    def action(self, o):
        opts = o['prompt']['options']
        hand = o['self'].get('hand', {})
        own, supply, p, _ = self.state(o)
        priority = {'Village': 100, 'Festival': 98, 'Laboratory': 94, 'Sentry': 96, 'Market': 92, 'Merchant': 90, 'Cellar': 70}
        def val(x):
            c = x['label']
            if c in priority: return priority[c]
            if c == 'Chapel':
                junk = hand.get('Curse', 0) + (hand.get('Estate', 0) if p >= 4 else 0)
                copper = min(hand.get('Copper', 0), max(0, money(own) - 6)) if p >= 4 else 0
                return 84 if junk + copper >= 2 else (52 if junk else -1)
            if c == 'Witch': return 82 if supply['Curse']['count'] else 48
            if c == 'Workshop': return 80 if self.route == 'gardens' else 35
            if c == 'Militia': return 57
            if c == 'Smithy': return 60
            if c == 'Moat': return 46
            if c == 'Remodel': return 63 if hand.get('Curse') or (hand.get('Estate') and p >= 4) or (hand.get('Gold') and p <= 3) else -1
            if c == 'Mine': return 50 if hand.get('Copper') or hand.get('Silver') else -1
            return 1
        if not opts: return []
        best = max(opts, key=val)
        return [str(best['id'])] if val(best) > 0 else []

    def select(self, o):
        self.setup(o)
        p = o['prompt']; kind = p['kind']; opts = p['options']
        if kind == 'buy': return self.buy(o)
        if kind == 'play_action': return self.action(o)
        if kind == 'trash': return self.trash(o)
        if kind == 'discard': return self.discard(o)
        if kind == 'gain':
            if p.get('source') == 'Mine':
                return [str(max(opts, key=lambda x: MONEY.get(x['label'], 0))['id'])] if opts else []
            if p.get('source') == 'Remodel' and self.last_remodel:
                match = [str(x['id']) for x in opts if x['label'] == self.last_remodel]
                self.last_remodel = None
                if match: return match[:1]
            if not opts: return []
            def gv(x):
                c = x['label']; v = self.gain_value(c, o, True)
                if self.end_after(o, {c: 1}):
                    margin = self.margin_after(o, {c: 1})
                    return 1000 + margin if self.winning_margin(o, margin) else -1000 + margin
                return v
            return [str(max(opts, key=gv)['id'])]
        if kind in ('binary', 'reaction') or {str(x.get('label')).lower() for x in opts} == {'true', 'false'}:
            want = True
            if p.get('source') == 'Sentry':
                cards = p.get('relevant_cards', [])
                def name(c): return c.get('label', c.get('name', '')) if isinstance(c, dict) else str(c)
                want = len(cards) >= 2 and self.keep_value(name(cards[1]), o) > self.keep_value(name(cards[0]), o)
            matches = [str(x['id']) for x in opts if str(x.get('label', '')).lower() == str(want).lower()]
            return matches[:1] or [str(opts[0]['id'])]
        return [str(x['id']) for x in opts[:p.get('min', 0)]]


def legal_ids(p, ids):
    allowed = {str(x['id']) for x in p.get('options', [])}
    chosen = []
    if isinstance(ids, list):
        for i in ids:
            if str(i) in allowed and str(i) not in chosen: chosen.append(str(i))
    lo = int(p.get('min', 0)); hi = int(p.get('max', 1))
    chosen = chosen[:hi]
    if len(chosen) < lo:
        for x in p.get('options', []):
            if str(x['id']) not in chosen: chosen.append(str(x['id']))
            if len(chosen) >= lo: break
    return chosen


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.policy = Policy()
        self.calls = 0
        self.failures = 0
        self.reviewed = set()
        self.buy_log = []
        self.abandoned_review = False

    def review_key(self, o):
        p = o['prompt']
        if p['kind'] != 'buy': return None
        turn = o['turns_taken']['agent']; prov = o['supply']['Province']['count']
        cash = o['resources']['coins']
        own = o['self']['owned']
        if turn <= 2: return 'opening-' + str(turn)
        if cash >= 5 and not own.get('Gold') and prov >= 6: return 'first-five'
        if prov <= 2: return 'late'
        if prov <= 4: return 'green'
        if turn >= 12: return 'turn12'
        if turn >= 7: return 'turn7'
        return None

    def critique(self, o, baseline):
        ctx = self.ctx
        key = self.review_key(o)
        if key is None or key in self.reviewed or self.calls >= 7 or self.failures >= 2 or self.abandoned_review: return baseline
        remaining = ctx.budget.remaining()
        if remaining.get('wall_s', 0) < 240 or remaining.get('system2_calls', 0) <= 2: return baseline
        self.reviewed.add(key)
        self.calls += 1
        own = o['self']['owned']; n = sum(own.values())
        recommendations = []
        for x in o['prompt']['options']:
            c = x['label']
            recommendations.append({'id': str(x['id']), 'card': c, 'baseline_value': self.policy.gain_value(c, o),
                                    'immediate_margin': self.policy.margin_after(o, {c: 1}),
                                    'ends_game_if_only_gain': self.policy.end_after(o, {c: 1})})
        snapshot = {k: o[k] for k in ('decision_id', 'agent_seat', 'turns_taken', 'resources', 'opponent_id', 'supply', 'self', 'opponent', 'derived', 'prompt') if k in o}
        prompt = (
            'You are a selective tactical critic for Dominion. Objective: maximize FINAL VP MARGIN against the disclosed single-buy money bot. '
            'Code plays the entire game quickly; you only review seven milestone purchases. This is NOT a request for a target deck or future action IDs. '
            'Choose the CURRENT buy (one option ID, or []). You may switch route between money and gardens, but only if Gardens and Workshop exist. '
            'The default money policy prioritizes one early Chapel, one Witch while curses remain, one Sentry when needed, Gold, a little terminal draw, '
            'Laboratory, and Silver. It avoids excess terminal actions; Duchy at <=4 Provinces, Estate at <=2. '
            'The gardens policy opens two Workshops then gains Gardens, adds one or two Villages, and empties Gardens, Estates and Workshops. '
            'Local code handles all action ordering, multi-card trashing, Militia discards, reactions, and winning current-turn endings. '
            'It does not support elaborate engines well. Prefer coherent specialization over a mix of action cards. '
            'Critique the default only when current public information justifies it. Pass is often best at $0-2 outside Gardens. '
            'Do not end while losing. Cards bought now enter discard. BM-1 buys Province, Duchy at <=4, Estate at <=2, Gold, Silver; '
            'SM-1/WM-1 insert one Smithy/Witch before money while >=5 Provinces. '
            'Reply ONLY JSON: {"option_ids":["id"],"route":"money or gardens","reason":"brief reason"}.\n'
            + json.dumps({'route': self.policy.route, 'baseline_ids': baseline, 'treasure_money': money(own), 'deck_size': n,
                          'recent_buys': self.buy_log[-8:], 'candidates': recommendations, 'state': snapshot}, separators=(',', ':')))
        try:
            # A backend timeout may not include queue/admission time. The main
            # control thread enforces a separate wait bound and can finish locally.
            timeout = min(150.0, remaining['wall_s'] - 120)
            future = ctx.spawn(lambda: ctx.think(prompt, session_id=None, timeout_s=timeout))
            start = time.monotonic()
            while True:
                if ctx.done: return baseline
                if time.monotonic() - start >= 180 or ctx.budget.remaining().get('wall_s', 0) < 120:
                    self.abandoned_review = True
                    ctx.log('critique_wait_abandoned', milestone=key)
                    return baseline
                try:
                    result = future.result(timeout=1.0)
                    break
                except FutureTimeout:
                    if future.done(): raise
            parsed = parse_object(result.text)
            if not parsed: return baseline
            route = parsed.get('route')
            ids = parsed.get('option_ids')
            if not isinstance(ids, list): return baseline
            valid = legal_ids(o['prompt'], ids)
            if len(ids) != len(valid) or [str(x) for x in ids] != valid: return baseline
            if valid:
                c = next(x['label'] for x in o['prompt']['options'] if str(x['id']) == valid[0])
                if self.policy.end_after(o, {c: 1}) and not self.policy.winning_margin(o, self.policy.margin_after(o, {c: 1})):
                    return baseline
            if route == 'money' or (route == 'gardens' and all(c in o['supply'] for c in ('Gardens', 'Workshop'))):
                self.policy.route = route
            ctx.log('purchase_critique', milestone=key, baseline=baseline, selected=valid, route=self.policy.route, reason=str(parsed.get('reason', ''))[:300])
            return valid
        except Exception as exc:
            self.failures += 1
            ctx.log('critique_failed', error=type(exc).__name__, milestone=key)
            return baseline

    def run_episode(self, game):
        ctx = self.ctx
        obs = ctx.observe()
        repeated = 0
        last_id = None
        while not ctx.done:
            if not isinstance(obs, dict):
                if isinstance(obs, str):
                    try: obs = json.loads(obs)
                    except (ValueError, TypeError): return
                else: return
            if obs.get('terminal') or not obs.get('prompt'): return
            p = obs['prompt']
            decision = obs['decision_id']
            repeated = repeated + 1 if decision == last_id else 0
            last_id = decision
            if repeated > 2: return
            ids = self.policy.select(obs)
            # Winning whole-turn endings are exact and do not need a model review.
            if p['kind'] == 'buy' and not self.policy.terminal_baskets(obs):
                ids = self.critique(obs, ids)
            if ctx.done: return
            if repeated: ids = [str(x['id']) for x in p['options'][:p.get('min', 0)]]
            ids = legal_ids(p, ids)
            if p['kind'] == 'buy' and ids:
                self.buy_log.append(next(x['label'] for x in p['options'] if str(x['id']) == ids[0]))
            try:
                new_obs = ctx.act({'decision_id': decision, 'option_ids': ids})
            except Exception as exc:
                if ctx.done: return
                ctx.log('act_recovery', error=type(exc).__name__)
                obs = ctx.observe()
                continue
            obs = new_obs if new_obs is not None else ctx.observe()


def build(ctx):
    return Harness(ctx)
