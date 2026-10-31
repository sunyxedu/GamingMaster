"""Public-belief portfolio with complete-turn, VP-margin endgame decisions."""
from __future__ import annotations
import json
import math
import re
from collections import Counter

TREASURE = {'Copper': 1, 'Silver': 2, 'Gold': 3}
POINTS = {'Estate': 1, 'Duchy': 3, 'Province': 6, 'Curse': -1}
CANTRIPS = {'Village', 'Festival', 'Market', 'Laboratory', 'Merchant', 'Sentry', 'Cellar'}
TERMINALS = {'Smithy', 'Witch', 'Militia', 'Moat', 'Chapel', 'Workshop', 'Remodel', 'Mine'}


def observation(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return {}
    return raw if isinstance(raw, dict) else {}


def vp(cards):
    return sum(cards.get(c, 0) * n for c, n in POINTS.items()) + cards.get('Gardens', 0) * (sum(cards.values()) // 10)


def parse_json(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r'\{', str(text)):
        try:
            obj, _ = decoder.raw_decode(str(text)[match.start():])
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
    return None


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = None
        self.calls = 0
        self.last_plan_turn = -10
        self.last_plan_provinces = 8
        self.failures = 0
        self.history = []
        self.override = None
        self.attempts = Counter()
        self.plan_future = None
        self.plan_started = False


    def finishing_basket(self, o):
        """Enumerate affordable scoring/ending baskets; verify entire-turn VP exactly."""
        supply = o['supply']
        buys = min(o['resources'].get('buys', 1), 6)
        coins = o['resources'].get('coins', 0)
        if buys <= 0:
            return None
        empty = sum(v['count'] == 0 for v in supply.values())
        if supply['Province']['count'] > buys and empty + sum(0 < v['count'] <= buys for v in supply.values()) < 3:
            return None
        names = [c for c,v in supply.items() if v['count']>0 and v['cost']<=coins and
                 (c in ('Province','Duchy','Estate','Gardens','Copper') or v['count']<=buys)]
        names.sort(key=lambda c: (-POINTS.get(c, 0), c))
        base = Counter(o['self']['owned'])
        opponent_vp = o['opponent']['vp']
        best = None
        best_score = -math.inf
        visited = 0
        def visit(index, money, left, gained, used):
            nonlocal best, best_score, visited
            visited += 1
            if visited > 12000:
                return
            new_empty = empty + sum(n == supply[c]['count'] for c,n in used.items())
            end = used.get('Province', 0) == supply['Province']['count'] or new_empty>=3
            if end:
                margin = vp(base + Counter(gained)) - opponent_vp
                turns = o.get('turns_taken', {})
                win = margin>0 or (margin==0 and turns.get('agent', 0)<=turns.get('opponent', 0))
                quality = margin + 0.001*money
                if win and quality > best_score:
                    best_score = quality
                    best = list(gained)
            if left == 0:
                return
            for i in range(index, len(names)):
                c = names[i]
                if used.get(c,0)>=supply[c]['count'] or supply[c]['cost']>money:
                    continue
                used[c] = used.get(c,0)+1
                gained.append(c)
                visit(i, money-supply[c]['cost'], left-1, gained, used)
                gained.pop()
                used[c] -= 1
                if not used[c]: del used[c]
        visit(0, coins, buys, [], {})
        return best

    def losing_end(self, c, o):
        supply = o['supply']
        empty = sum(v['count']==0 for v in supply.values())
        if supply[c]['count'] != 1 or not (c=='Province' or empty>=2):
            return False
        cards = Counter(o['self']['owned']); cards[c] += 1
        margin = vp(cards)-o['opponent']['vp']
        t = o.get('turns_taken', {})
        return margin < 0 or (margin==0 and t.get('agent',0)>t.get('opponent',0))


    def validate(self, ids, p):
        allowed={str(op['id']) for op in p.get('options',[])}
        result=[]
        for i in ids:
            if str(i) in allowed and str(i) not in result:
                result.append(str(i))
        result=result[:p.get('max',1)]
        for op in p.get('options',[]):
            if len(result)>=p.get('min',0): break
            if str(op['id']) not in result: result.append(str(op['id']))
        return result

    def run_episode(self, game):
        ctx=self.ctx
        o=observation(ctx.observe())
        while not ctx.done:
            if not o or o.get('terminal') or not o.get('prompt'):
                return
            if self.plan is None:
                self.plan=self.default_plan(o)
            p=o['prompt']; did=o.get('decision_id')
            self.attempts[did]+=1
            # Poll the single optional planner on every step; ordinary action
            # and attack prompts must never wait for inference.
            if p.get('kind')=='buy' and self.attempts[did]==1:
                self.plan_now(o)
            elif self.plan_future is not None:
                self.plan_now(o)
            if ctx.done: return
            try:
                ids=self.choose(o)
            except Exception as exc:
                ctx.log('local_policy_failure',error=type(exc).__name__,kind=p.get('kind'),source=p.get('source'))
                ids=[]
            if self.attempts[did]>1:
                ids=[op['id'] for op in p.get('options',[])[:p.get('min',0)]]
            if self.attempts[did]>3:
                ctx.log('stalled_prompt',decision_id=did)
                return
            ids=self.validate(ids,p)
            selected=[op['label'] for op in p.get('options',[]) if str(op['id']) in ids]
            if p.get('kind') in ('buy','gain') and selected:
                self.history.append({'turn':o.get('turns_taken',{}).get('agent',0),'kind':p['kind'],'cards':selected})
            action='choose '+str(did)+(' '+' '.join(ids) if ids else '')
            try:
                raw=ctx.act(action)
                o=observation(raw if raw is not None else ctx.observe())
            except Exception as exc:
                if ctx.done: return
                ctx.log('action_failure',error=type(exc).__name__,decision_id=did)
                o=observation(ctx.observe())


# Independent public-observation forward model adapted from candidate 0003.
import random
import time
MONEY = {'Copper': 1, 'Silver': 2, 'Gold': 3}
DRAW = {'Village': 1, 'Smithy': 3, 'Moat': 2, 'Witch': 2,
        'Laboratory': 2, 'Market': 1, 'Merchant': 1, 'Sentry': 1}
PLUS = {'Village': 2, 'Festival': 2, 'Laboratory': 1, 'Market': 1,
        'Merchant': 1, 'Sentry': 1, 'Cellar': 1}
ACTIONS = set(DRAW) | set(PLUS) | {'Chapel', 'Militia', 'Mine', 'Remodel', 'Workshop'}
VICTORY = {'Estate', 'Duchy', 'Province', 'Gardens'}
SUPPORTED = ACTIONS | VICTORY | set(MONEY) | {'Curse'}


def expand(counts):
    return [c for c, n in counts.items() for _ in range(max(0, int(n)))]


def points(owned):
    return (owned.get('Estate', 0) + 3 * owned.get('Duchy', 0)
            + 6 * owned.get('Province', 0) - owned.get('Curse', 0)
            + owned.get('Gardens', 0) * (sum(owned.values()) // 10))


def trash_cards(cards, owned, piles, plan, limit=4):
    """Choose junk to remove, preserving enough total treasure to buy recovery."""
    left = Counter(owned)
    out = []
    late = piles.get('Province', 8) <= 2
    for c in sorted(cards, key=lambda x: {'Curse': 0, 'Estate': 1, 'Copper': 2}.get(x, 9)):
        if len(out) >= limit:
            break
        wealth = sum(MONEY.get(x, 0) * n for x, n in left.items())
        trash = c == 'Curse' or (c == 'Estate' and not late and plan['style'] != 'gardens')
        if c == 'Copper' and not late and plan['style'] != 'gardens':
            # Preserve $3 total at minimum; as good money arrives, Copper can go to zero.
            floor = max(0, plan['copper_floor'] - left.get('Silver', 0) - 2 * left.get('Gold', 0))
            trash = left[c] > floor and wealth - 1 >= 3
        if c == 'Chapel' and left.get('Curse', 0) == 0 and left.get('Estate', 0) == 0 and left.get('Copper', 0) <= 1 and piles.get('Curse', 10) == 0:
            trash = True
        if trash:
            out.append(c)
            left[c] -= 1
    return out


def keep_value(c, hand, owned, piles, plan, actions=1):
    if c in MONEY:
        return MONEY[c] * 10
    if c in VICTORY or c == 'Curse':
        return -5
    if c == 'Chapel':
        return 8 + 12 * len(trash_cards([x for x in hand if x != 'Chapel'], owned, piles, plan))
    if c in PLUS:
        return {'Village': 24 if sum(x in ACTIONS and x not in PLUS for x in hand) else 5,
                'Festival': 30, 'Laboratory': 38, 'Market': 33, 'Merchant': 24,
                'Sentry': 38, 'Cellar': 4}.get(c, 20)
    terminal_count = sum(x in ACTIONS and x not in PLUS for x in hand)
    collision = terminal_count > actions and not any(x in ('Village', 'Festival') for x in hand)
    val = {'Witch': 44 if piles.get('Curse', 0) else 21, 'Militia': 28,
           'Smithy': 34, 'Moat': 22, 'Workshop': 13, 'Mine': 18, 'Remodel': 12}.get(c, 0)
    if collision and c != plan.get('terminal'):
        val -= 12
    return val


def action_value(c, hand, owned, piles, plan, actions=1):
    if c == 'Village': return 100
    if c == 'Festival': return 99
    if c == 'Sentry': return 95
    if c == 'Laboratory': return 92
    if c == 'Market': return 90
    if c == 'Merchant': return 89
    if c == 'Cellar': return 50
    if c == 'Chapel':
        ts = trash_cards([x for x in hand if x != c], owned, piles, plan)
        return 42 + 12 * len(ts) if ts else -1
    if c == 'Witch': return 85 if piles.get('Curse', 0) else 38
    if c == 'Smithy': return 55
    if c == 'Moat': return 38
    if c == 'Militia': return 48
    if c == 'Mine': return 30 if any(x in ('Copper', 'Silver') for x in hand) else -1
    if c == 'Remodel': return 33 if any(x in ('Curse', 'Estate') for x in hand) else 6
    if c == 'Workshop': return 45 if plan['style'] == 'gardens' else 20
    return 0


def reference_values(owned, piles, costs, plan, turn, money, gain=False):
    """Continuation policy, also a low variance prior for sampled action values."""
    n = sum(owned.values())
    p = piles.get('Province', 8)
    style = plan['style']
    term = sum(owned.get(c, 0) for c in ACTIONS if c not in PLUS)
    cap = owned.get('Village', 0) + owned.get('Festival', 0)
    junk = owned.get('Copper', 0) + owned.get('Estate', 0) + owned.get('Curse', 0)
    goodmoney = 2 * owned.get('Silver', 0) + 3 * owned.get('Gold', 0)
    v = {c: -100.0 for c in costs}
    v.update({'Province': 100, 'Gold': 43, 'Silver': 23,
              'Duchy': 65 if p <= plan['duchy_at'] else -10,
              'Estate': 60 if p <= plan['estate_at'] else -20,
              'Copper': -30, 'Curse': -100})
    if p <= 1:
        v['Duchy'], v['Estate'] = 85, 40
    if p >= 4:
        if not owned.get('Chapel') and turn <= 7 and junk >= 5:
            v['Chapel'] = 58 if turn <= 2 else 35
        if not owned.get('Sentry') and junk >= 5:
            v['Sentry'] = 57 if turn <= 7 else 46
        elif owned.get('Sentry', 0) < 2 and junk >= 10:
            v['Sentry'] = 39
        if piles.get('Curse', 0) and owned.get('Witch', 0) < (2 if cap > 0 else 1):
            v['Witch'] = 70 if not owned.get('Witch') else 42
        desired = plan.get('terminal', 'Smithy')
        if term < 1 + cap + (n >= 22):
            for c in ('Smithy', 'Militia', 'Moat'):
                v[c] = (36 if c == desired else 27) - 8 * owned.get(c, 0)
            if term == 0:
                v[desired] = max(v.get(desired, -100), 48)
        v['Laboratory'] = 44 if style == 'engine' else 38
        v['Market'] = 42 if style == 'engine' else 36
        if goodmoney < 6:
            v['Laboratory'] -= 9
            v['Market'] -= 4
        if goodmoney + owned.get('Copper', 0) + 2 * owned.get('Militia', 0) + 2 * owned.get('Festival', 0) + owned.get('Market', 0) < 8:
            # Drawing the entire deck is insufficient if it cannot produce $8.
            v['Silver'] = 40
            v['Gold'] = 52
        if term > cap:
            v['Village'] = 30 if term >= cap + 2 else 14
            v['Festival'] = 47 if term >= cap + 2 else 36
        v['Merchant'] = 25 if owned.get('Silver', 0) >= 2 and owned.get('Merchant', 0) < 2 else 10
        if not owned.get('Cellar') and n >= 18 and sum(owned.get(x, 0) for x in VICTORY) >= 5:
            v['Cellar'] = 15
        if not owned.get('Remodel') and 'Chapel' not in piles and 'Sentry' not in piles and owned.get('Estate', 0) >= 2:
            v['Remodel'] = 27
        if not owned.get('Mine') and n <= 16:
            v['Mine'] = 30
    # Free gains can usefully add a cantrip in place of a second terminal.
    if gain:
        v['Gardens'] = max(v.get('Gardens', -100), 32 if n >= 20 else -5)
    if style == 'gardens' and 'Gardens' in piles:
        v['Gardens'] = 72 if n >= 15 or owned.get('Workshop', 0) >= 2 else 35
        v['Workshop'] = 65 if owned.get('Workshop', 0) < 2 else 20
        v['Village'] = 48 if term > cap + 1 else 16
        v['Estate'] = max(v['Estate'], 5)
        v['Copper'] = 1
    return {c: val for c, val in v.items() if piles.get(c, 0) > 0 and costs.get(c, 100) <= money}


def best_buy(owned, piles, costs, plan, turn, money, gain=False):
    vals = policy_values(owned, piles, costs, plan, turn, money, gain)
    if not vals: return None
    c = max(vals, key=lambda x: (vals[x], costs.get(x, 0), x))
    return c if gain or vals[c] > 0 else None


class Player:
    def __init__(self, owned, hand, deck, discard, played=()):
        self.owned = Counter(owned)
        self.hand, self.deck, self.discard, self.played = list(hand), list(deck), list(discard), list(played)
        self.actions = 1
        self.buys = 1
        self.coins = 0
        self.merchants = 0

    def draw(self, n, rng):
        for _ in range(n):
            if not self.deck:
                self.deck, self.discard = self.discard, []
                rng.shuffle(self.deck)
            if not self.deck: break
            self.hand.append(self.deck.pop())

    def cleanup(self, rng):
        self.discard.extend(self.hand)
        self.discard.extend(self.played)
        self.hand, self.played = [], []
        self.draw(5, rng)


class World:
    """A sampled, deliberately separate forward model made from public observations."""
    def __init__(self, obs, plan, seed):
        self.rng = random.Random(seed)
        self.piles = {c: d['count'] for c, d in obs['supply'].items()}
        self.costs = {c: d['cost'] for c, d in obs['supply'].items()}
        self.plan = plan
        self.opponent = obs['opponent_id']
        self.turns = [obs['turns_taken']['agent'], obs['turns_taken']['opponent']]
        self.players = []
        for i, key in enumerate(('self', 'opponent')):
            d = obs[key]
            played = expand(d.get('in_play', {}))
            unknown = Counter(d['owned']) - Counter(played)
            if i == 0:
                hand, discard = expand(d['hand']), expand(d['discard'])
                deck = expand(unknown - Counter(hand) - Counter(discard))
                self.rng.shuffle(deck)
            else:
                pool = expand(unknown)
                self.rng.shuffle(pool)
                hc, dc = d['hand_count'], d['discard_count']
                hand, discard, deck = pool[:hc], pool[hc:hc+dc], pool[hc+dc:]
            self.players.append(Player(d['owned'], hand, deck, discard, played))
        self.players[0].buys = obs['resources']['buys']
        self.players[0].coins = obs['resources']['coins']

    def gain(self, i, c, hand=False):
        if c is None or self.piles.get(c, 0) <= 0: return
        self.piles[c] -= 1
        p = self.players[i]
        p.owned[c] += 1
        (p.hand if hand else p.discard).append(c)

    def trash(self, i, c, zone):
        if c in zone:
            zone.remove(c)
            self.players[i].owned[c] -= 1

    def ended(self):
        return self.piles.get('Province', 8) == 0 or sum(n == 0 for n in self.piles.values()) >= 3

    def buy(self, i, c):
        p = self.players[i]
        if c and p.buys and self.costs.get(c, 100) <= p.coins and self.piles.get(c, 0):
            p.coins -= self.costs[c]
            p.buys -= 1
            self.gain(i, c)
            return True
        return False

    def enemy_buy(self, money):
        p = self.piles
        own = self.players[1].owned
        if money >= 8 and p.get('Province'): return 'Province'
        if p.get('Province', 8) <= 4 and money >= 5 and p.get('Duchy'): return 'Duchy'
        if p.get('Province', 8) <= 2 and money >= 2 and p.get('Estate'): return 'Estate'
        special = {'SM-1': 'Smithy', 'WM-1': 'Witch'}.get(self.opponent)
        if special and not own[special] and p.get('Province', 8) >= 5 and p.get(special) and self.costs[special] <= money:
            if special != 'Witch' or p.get('Curse', 0): return special
        if money >= 6 and p.get('Gold'): return 'Gold'
        if money >= 3 and p.get('Silver'): return 'Silver'
        return None

    def discard_down(self, i):
        p = self.players[i]
        if i == 1:
            def rank(c):
                return 100 if c in ('Smithy', 'Witch') else MONEY.get(c, 0) * 10 + {'Province': 6, 'Duchy': 3, 'Estate': 1, 'Curse': -1}.get(c, 0)
        else:
            def rank(c): return keep_value(c, p.hand, p.owned, self.piles, self.plan)
        while len(p.hand) > 3:
            c = min(p.hand, key=rank)
            p.hand.remove(c)
            p.discard.append(c)

    def play(self, i, c):
        p, enemy = self.players[i], self.players[1-i]
        p.hand.remove(c)
        p.played.append(c)
        p.actions += PLUS.get(c, 0) - 1
        p.draw(DRAW.get(c, 0), self.rng)
        if c in ('Festival', 'Market'):
            p.buys += 1
            p.coins += 2 if c == 'Festival' else 1
        if c == 'Merchant': p.merchants += 1
        if c == 'Witch' and 'Moat' not in enemy.hand: self.gain(1-i, 'Curse')
        if c == 'Militia':
            p.coins += 2
            if 'Moat' not in enemy.hand: self.discard_down(1-i)
        if c == 'Chapel':
            for x in trash_cards(p.hand, p.owned, self.piles, self.plan): self.trash(i, x, p.hand)
        if c == 'Cellar':
            tossed = [x for x in p.hand if x in VICTORY or x == 'Curse']
            for x in tossed:
                p.hand.remove(x)
                p.discard.append(x)
            p.draw(len(tossed), self.rng)
        if c == 'Sentry':
            inspected = []
            for _ in range(2):
                if not p.deck:
                    p.deck, p.discard = p.discard, []
                    self.rng.shuffle(p.deck)
                if p.deck: inspected.append(p.deck.pop())
            for x in trash_cards(inspected, p.owned, self.piles, self.plan, 2): self.trash(i, x, inspected)
            kept = []
            for x in inspected:
                if x in VICTORY or x == 'Curse': p.discard.append(x)
                else: kept.append(x)
            p.deck.extend(reversed(kept))
        if c == 'Workshop':
            self.gain(i, best_buy(p.owned, self.piles, self.costs, self.plan, self.turns[i], 4, True))
        if c == 'Mine':
            x = 'Silver' if 'Silver' in p.hand and self.piles.get('Gold') else 'Copper' if 'Copper' in p.hand and self.piles.get('Silver') else None
            if x:
                self.trash(i, x, p.hand)
                self.gain(i, 'Gold' if x == 'Silver' else 'Silver', True)
        if c == 'Remodel' and p.hand:
            x = min(p.hand, key=lambda z: keep_value(z, p.hand, p.owned, self.piles, self.plan))
            if 'Gold' in p.hand and self.piles.get('Province', 8) <= 3: x = 'Gold'
            self.trash(i, x, p.hand)
            self.gain(i, best_buy(p.owned, self.piles, self.costs, self.plan, self.turns[i], self.costs[x]+2, True))

    def finish_buy(self, i):
        p = self.players[i]
        for _ in range(min(20, p.buys)):
            c = self.enemy_buy(p.coins) if i else best_buy(p.owned, self.piles, self.costs, self.plan, self.turns[i], p.coins)
            if not self.buy(i, c): break
        p.cleanup(self.rng)

    def turn(self, i):
        self.turns[i] += 1
        p = self.players[i]
        p.actions, p.buys, p.coins, p.merchants = 1, 1, 0, 0
        for _ in range(100):
            opts = [c for c in p.hand if c in ACTIONS]
            if p.actions <= 0 or not opts: break
            c = max(opts, key=lambda c: action_value(c, p.hand, p.owned, self.piles, self.plan, p.actions))
            if action_value(c, p.hand, p.owned, self.piles, self.plan, p.actions) < 0: break
            self.play(i, c)
        for c in list(p.hand):
            if c in MONEY:
                p.hand.remove(c)
                p.played.append(c)
                p.coins += MONEY[c]
                if c == 'Silver':
                    p.coins += p.merchants
                    p.merchants = 0
        self.finish_buy(i)

    def rollout(self, root_card):
        if root_card: self.buy(0, root_card)
        else: self.players[0].buys = 0
        self.finish_buy(0)
        for _ in range(100):
            if self.ended(): break
            self.turn(1)
            if self.ended(): break
            self.turn(0)
        return points(self.players[0].owned) - points(self.players[1].owned)


def portfolio(supply):
    """Distinct executable acquisition policies, not arbitrary card multipliers."""
    names = ['militia_money', 'smithy_money', 'cantrip_money']
    if 'Witch' in supply:
        names.append('witch_money')
    if 'Laboratory' in supply:
        names.extend(['laboratory_engine', 'laboratory_build'])
    if 'Village' in supply and 'Smithy' in supply:
        names.extend(['smithy_engine', 'smithy_build'])
    names.append('reference')
    result = []
    for name in names:
        preferred = 'Smithy' if name.startswith('smithy') else 'Witch' if name == 'witch_money' else 'Militia'
        if preferred not in supply:
            preferred = next((c for c in ('Smithy', 'Witch', 'Moat') if c in supply), 'Smithy')
        result.append(dict(name=name, style='engine' if ('engine' in name or 'build' in name) else 'money',
                           mode='money', terminal=preferred, duchy_at=3, estate_at=1,
                           copper_floor=3, targets=[], province_at=8))
    # Scoring conversion is part of a complete policy, not a hard race gate.
    for plan in list(result):
        if plan['name'] in ('militia_money','smithy_money','laboratory_engine'):
            result.append(dict(plan, name=plan['name']+'_green', duchy_at=4, estate_at=2))
    return result


def policy_values(owned, piles, costs, plan, turn, money, gain=False):
    if plan.get('name') == 'reference':
        return reference_values(owned, piles, costs, plan, turn, money, gain)
    n = sum(owned.values())
    p = piles.get('Province', 8)
    name = plan.get('name', 'militia_money').removesuffix('_green')
    engine = name.endswith(('engine','build'))
    junk = sum(owned.get(c, 0) for c in ('Copper', 'Estate', 'Curse'))
    payload = (sum(owned.get(c, 0)*v for c,v in MONEY.items()) +
               2*owned.get('Militia', 0) + 2*owned.get('Festival', 0) +
               owned.get('Market', 0) + (owned.get('Merchant', 0) if owned.get('Silver', 0) else 0))
    capacity = owned.get('Village', 0) + owned.get('Festival', 0)
    terminals = sum(owned.get(c, 0) for c in ('Witch','Smithy','Moat','Militia','Mine','Remodel','Workshop'))
    if junk >= 3:
        terminals += owned.get('Chapel', 0)
    draws = sum(owned.get(c, 0)*v for c,v in DRAW.items())
    draw_gap = n - 5 - draws
    v = {c: -100.0 for c in costs}
    v.update(Province=110, Duchy=90 if p<=plan['duchy_at'] else -20,
             Estate=65 if p<=plan['estate_at'] else -30, Gold=48, Silver=25,
             Copper=-30, Curse=-100)
    # Scoring uses remaining piles, not a turn-number schedule. Early build cards
    # are dominated by Province in every profile; the portfolio searches plans.
    if p <= 3:
        if gain and 'Gardens' in costs:
            v['Gardens'] = 45 if n >= 20 else 15
        return {c:x for c,x in v.items() if piles.get(c,0)>0 and costs.get(c,100)<=money}
    if not owned.get('Chapel') and turn <= 5 and junk >= 5:
        v['Chapel'] = 85 if turn <= 2 else 58
    if not owned.get('Sentry') and junk >= 6 and (not owned.get('Chapel') or owned.get('Curse',0)>=3):
        v['Sentry'] = 72 if turn <= 8 else 52
    # A curse-giving terminal is useful against all disclosed money opponents.
    # Its policy is evaluated in context rather than keyed to an opponent label.
    if piles.get('Curse',0)>0 and not owned.get('Witch'):
        v['Witch'] = 80
    preferred = plan['terminal']
    desired = 2 if name == 'smithy_money' and n >= 17 else 1
    productive = terminals - (owned.get('Chapel',0) if junk>=3 else 0)
    if owned.get(preferred,0)<desired and productive < (2 if n>=17 else 1):
        v[preferred] = 62 if not owned.get(preferred) else 37
    if productive == 0 and preferred in costs:
        v[preferred] = max(v[preferred], 62)
    if turn<=2 and money>=4 and 'Chapel' in piles and preferred in costs and not owned.get(preferred) and name!='cantrip_money':
        v[preferred] = 88
    if owned.get('Silver',0)==0 and payload<9:
        v['Silver'] = 54
    if name == 'cantrip_money':
        v['Market'] = 51 if owned.get('Gold',0)>=1 else 38
        v['Laboratory'] = 54 if owned.get('Gold',0)>=2 else 41
        v['Merchant'] = 31 if owned.get('Silver',0)>=2 and owned.get('Merchant',0)<2 else 5
    if engine:
        # Reach, action capacity and payload are jointly constrained. Extra
        # draw with no $8 payload and extra terminals without actions get no bid.
        need_support = terminals > capacity+1
        v['Village'] = 64 if need_support else (28 if terminals>capacity else 5)
        v['Festival'] = 68 if need_support else 39
        if payload < 8:
            v['Gold'], v['Silver'] = 78, 57
            v['Market'] = 54
        else:
            v['Gold'] = 50 if payload < 14 else 36
            v['Silver'] = 23 if payload < 14 else 8
            if name in ('laboratory_engine','laboratory_build'):
                v['Laboratory'] = 66 if draw_gap >= -1 else 42
                v['Smithy'] = 57 if capacity>=terminals and draw_gap>=2 else -10
            else:
                v['Smithy'] = 65 if capacity>=terminals and draw_gap>=1 else -10
                v['Laboratory'] = 61 if draw_gap>=0 else 40
            v['Market'] = 62 if draw_gap<=2 and payload<16 else 49
            if owned.get('Market',0)+owned.get('Festival',0)==0 and payload>=10:
                v['Market'] = 69
            if not owned.get('Militia') and terminals <= capacity and payload<12:
                v['Militia'] = 58
        if piles.get('Curse',0)>=4 and not owned.get('Witch'):
            v['Witch'] = 82
        if owned.get('Silver',0)>0:
            v['Merchant'] = 32 if payload<12 else 20
    if name.endswith('build') and p>=5 and turn<20 and money<16:
        # Invest in a draw-to-$16 engine until race or engine readiness says score.
        extra_buys=owned.get('Market',0)+owned.get('Festival',0)
        ready=payload>=14 and draw_gap<=1 and extra_buys>0
        if not ready:
            v['Province']=44
    if not owned.get('Cellar') and sum(owned.get(c,0) for c in VICTORY)>=5 and n>=18:
        v['Cellar'] = 18
    if gain:
        v['Gardens'] = max(v.get('Gardens',-100), 34 if n>=20 else 2)
        # A free Workshop gain still follows the selected construction policy.
        v['Merchant'] = max(v.get('Merchant',-100), 22)
    return {c:x for c,x in v.items() if piles.get(c,0)>0 and costs.get(c,100)<=money}


class PortfolioHarness(Harness):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.policies = []
        self.advice_name = None
        self.last_forecast = []
        self.last_search_turn = -1
        self.search_count = 0

    def default_plan(self, o):
        self.policies = portfolio(o['supply'])
        preferred = 'laboratory_engine' if 'Chapel' in o['supply'] and 'Laboratory' in o['supply'] else 'militia_money'
        return dict(next((p for p in self.policies if p['name']==preferred), self.policies[0]))

    def plan_now(self, o):
        # Nonblocking model advice is a policy suggestion. Sampled evidence must
        # corroborate it; malformed output cannot mutate the controller.
        if self.plan_future is not None:
            if not self.plan_future.done():
                return
            fut, self.plan_future = self.plan_future, None
            try:
                data = parse_json(fut.result().text)
                name = data.get('policy') if isinstance(data,dict) else None
                if isinstance(name,str) and any(p['name']==name for p in self.policies):
                    self.advice_name = name
                    self.ctx.log('portfolio_advice', policy=name, reason=str(data.get('reason',''))[:300])
                else:
                    self.ctx.log('portfolio_advice_unparsed')
            except Exception as e:
                self.ctx.log('portfolio_advice_failure', error=type(e).__name__)
            return
        if self.plan_started:
            return
        rem = self.ctx.budget.remaining()
        if rem.get('wall_s',0)<180 or rem.get('system2_calls',0)<2:
            return
        self.plan_started = True
        state = {k:o.get(k) for k in ('kingdom','opponent_id','supply','self','opponent','resources','derived')}
        prompt = ('Select one complete acquisition policy for Dominion; reply only '
                  '{"policy":"name","reason":"one sentence"}. Code compares complete-game '
                  'VP margins on sampled hidden worlds and executes legal IDs locally. '
                  'Policies: militia_money = one Militia, treasures; smithy_money = one/two Smithies, '
                  'treasures; cantrip_money = treasures then Market/Laboratory; witch_money = Witch '
                  'and treasures; laboratory_engine = thin, reach whole deck with Laboratories, '
                  'maintain $8+ payload and add Markets; build variants invest to $16 while >=5 Provinces. '
                  'Suffix _green means Duchies at <=4 and Estates at <=2. smithy_engine = Village/Smithy capacity '
                  'and draw, payload and Markets; reference = mixed baseline. All policies buy '
                  'early Chapel/Sentry and Witch when available. Except for build and green variants, '
                  'buy Province whenever affordable, Duchy at <=3 Provinces, Estate at <=1. Code handles opponent-specific '
                  'attacks and public final-VP arithmetic. Choose among available policies: '
                  + ', '.join(p['name'] for p in self.policies) + '\n' + json.dumps(state,separators=(',',':')))
        try:
            self.plan_future=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=None,timeout_s=120.0))
        except Exception as e:
            self.ctx.log('portfolio_advice_failure',error=type(e).__name__)

    def policy_root(self, o, plan, allowed):
        piles={c:d['count'] for c,d in o['supply'].items()}
        costs={c:d['cost'] for c,d in o['supply'].items()}
        vals=policy_values(o['self']['owned'],piles,costs,plan,o['turns_taken']['agent'],o['resources']['coins'])
        candidates=[c for c in allowed if vals.get(c,-100)>0]
        return max(candidates,key=lambda c:(vals[c],costs[c],c),default=None)

    def select_policy(self, o, allowed):
        roots={p['name']:self.policy_root(o,p,allowed) for p in self.policies}
        fallback=roots.get(self.plan['name'])
        # If all plans nominate the same acquisition, no samples can alter it.
        if len(set(roots.values()))==1:
            return fallback
        if not set(o['supply']).issubset(SUPPORTED):
            return fallback
        remain=self.ctx.budget.remaining().get('wall_s',0)
        if remain<45:
            return fallback
        started=time.monotonic()
        stop=started+min(9.0,max(0.1,(remain-40)/55))
        samples={p['name']:[] for p in self.policies}
        by_name={p['name']:p for p in self.policies}
        names=list(by_name)
        seed0=197137+int(o['decision_id'])*65537
        # Every policy in each round sees the same sampled initial zones. The
        # second sample batch is independent, preventing a lucky screening mean
        # from being used as the final comparison.
        for k in range(96):
            if self.ctx.done or time.monotonic()>stop:
                break
            seed=seed0+k*104729
            for name in names:
                samples[name].append(World(o,by_name[name],seed).rollout(roots[name]))
        tested=[name for name in names if samples[name]]
        if not tested:
            return fallback
        means={name:sum(samples[name])/len(samples[name]) for name in tested}
        finalists=sorted(tested,key=lambda name:means[name],reverse=True)[:3]
        if self.plan['name'] not in finalists:
            finalists.append(self.plan['name'])
        final={name:[] for name in finalists}
        for k in range(160):
            if self.ctx.done or time.monotonic()>stop:
                break
            seed=seed0+1000000007+k*104729
            for name in finalists:
                final[name].append(World(o,by_name[name],seed).rollout(roots[name]))
        # Prefer the independent validation batch when it has enough samples.
        scores=final if min((len(v) for v in final.values()),default=0)>=16 else {n:samples[n] for n in finalists}
        avg={name:sum(xs)/len(xs) for name,xs in scores.items()}
        winner=max(avg,key=avg.get)
        incumbent=self.plan['name']
        if incumbent in scores and winner!=incumbent:
            diff=[a-b for a,b in zip(scores[winner],scores[incumbent])]
            mean=sum(diff)/len(diff)
            se=(sum((x-mean)**2 for x in diff)/max(1,len(diff)-1)/len(diff))**0.5
            # Avoid changing coherent build policies on noise alone.
            if mean < max(0.5,se):
                winner=incumbent
        if self.advice_name in avg and avg[self.advice_name]>=avg[winner]-0.5:
            winner=self.advice_name
        self.plan=dict(by_name[winner])
        self.ctx.memory['portfolio_policy']=dict(self.plan)
        self.search_count+=1
        self.last_forecast=[{'policy':n,'root':roots[n],'screen':round(means[n],3),
                             'validation':round(avg[n],3) if n in avg else None,
                             'n':len(scores.get(n,[]))} for n in tested]
        self.ctx.log('portfolio_search',decision_id=o['decision_id'],policy=winner,
                     selected=roots[winner],forecasts=self.last_forecast,
                     seconds=round(time.monotonic()-started,3))
        return roots[winner]

    def acquisition(self,o,opts):
        allowed={x['label']:x['id'] for x in opts}
        if o['prompt'].get('source')=='Mine' and o['prompt']['kind']=='gain':
            # Mine gains to hand. Match the model's actual Treasure upgrade,
            # rather than letting a deck-building preference gain Silver back.
            c=next((c for c in ('Gold','Silver','Copper') if c in allowed),None)
            return [allowed[c]] if c else []
        if o['prompt']['kind']=='buy':
            basket=self.finishing_basket(o)
            if basket and basket[0] in allowed:
                self.ctx.log('certified_ending',basket=basket)
                return [allowed[basket[0]]]
            # Preserve the parent's exact one-card losing-end guard. Decisions
            # to decline a loss must not discard the possibility of another buy.
            safe={c:i for c,i in allowed.items() if not self.losing_end(c,o)}
            try:
                c=self.select_policy(o,safe or allowed)
            except Exception as e:
                self.ctx.log('portfolio_failure',error=type(e).__name__)
                c=self.policy_root(o,self.plan,safe or allowed)
        else:
            piles={c:d['count'] for c,d in o['supply'].items()}
            costs={c:d['cost'] for c,d in o['supply'].items()}
            vals=policy_values(o['self']['owned'],piles,costs,self.plan,o['turns_taken']['agent'],max([costs.get(c,0) for c in allowed] or [0]),True)
            c=max(allowed,key=lambda c:(vals.get(c,-100),costs.get(c,0)),default=None)
        return [allowed[c]] if c in allowed else []

    def choose(self,o):
        # Use the same action/thinning/discard rules as the forward model.
        p=o['prompt']; opts=p['options']; kind=p['kind']; src=p.get('source') or ''
        if kind in ('buy','gain'):
            return self.acquisition(o,opts)
        labels=[x['label'] for x in opts]
        hand=expand(o['self'].get('hand',{})); owned=o['self']['owned']
        piles={c:d['count'] for c,d in o['supply'].items()}
        chosen=[]
        if kind=='play_action':
            c=max(labels,key=lambda c:action_value(c,hand,owned,piles,self.plan,o['resources']['actions']),default=None)
            if c and action_value(c,hand,owned,piles,self.plan,o['resources']['actions'])>=0:
                chosen=[c]
        elif kind=='trash':
            if src=='Mine':
                c='Silver' if 'Silver' in labels and piles.get('Gold') else 'Copper' if 'Copper' in labels and piles.get('Silver') else None
                chosen=[c] if c else []
            elif src=='Remodel':
                c=min(labels,key=lambda c:keep_value(c,hand,owned,piles,self.plan),default=None)
                if 'Gold' in labels and piles.get('Province',8)<=3:
                    c='Gold'
                chosen=[c] if c else []
            else:
                chosen=trash_cards(labels,owned,piles,self.plan,p['max'])
        elif kind=='discard':
            if p['min']:
                chosen=sorted(labels,key=lambda c:keep_value(c,hand,owned,piles,self.plan))[:p['min']]
            else:
                chosen=[c for c in labels if c in VICTORY or c=='Curse']
        elif 'True' in labels and 'False' in labels:
            truth=src=='Moat' or 'reveal' in p.get('instruction','').lower()
            if src=='Sentry':
                cards=p.get('relevant_cards',[])
                if len(cards)==2 and all(isinstance(c,str) for c in cards):
                    truth=keep_value(cards[1],hand,owned,piles,self.plan)>keep_value(cards[0],hand,owned,piles,self.plan)
            chosen=['True' if truth else 'False']
        ids=[]; pool=list(opts)
        for c in chosen:
            for i,op in enumerate(pool):
                if op['label']==c:
                    ids.append(op['id']); pool.pop(i); break
        return self.validate(ids,p)


class MarginHarness(PortfolioHarness):
    """Compare terminal and nonterminal complete-turn baskets on one objective."""
    def __init__(self, ctx):
        super().__init__(ctx)
        self.buy_queue = []
        self.queue_turn = None

    def baskets(self, o):
        supply = o['supply']
        coins = o['resources']['coins']
        buys = min(8, o['resources']['buys'])
        owned = Counter(o['self']['owned'])
        empty = sum(v['count'] == 0 for v in supply.values())
        # Include every scoring pile and every pile that these Buys can empty.
        names = [c for c, d in supply.items() if d['count'] > 0 and d['cost'] <= coins
                 and (c in VICTORY or d['count'] <= buys or
                      (c == 'Copper' and (owned.get('Gardens', 0) or
                                         supply.get('Gardens', {}).get('count', 0))))]
        names.sort(key=lambda c: (-POINTS.get(c, 0), c))
        rows = []
        def visit(index, money, left, bought, used):
            if len(rows) >= 12000 or self.ctx.done:
                return
            is_end = (used.get('Province', 0) == supply['Province']['count'] or
                      empty + sum(n == supply[c]['count'] for c, n in used.items()) >= 3)
            margin = vp(owned + Counter(bought)) - o['opponent']['vp']
            rows.append({'cards': tuple(bought), 'end': is_end, 'margin': margin})
            if not left:
                return
            for j in range(index, len(names)):
                c = names[j]
                if supply[c]['cost'] > money or used.get(c, 0) >= supply[c]['count']:
                    continue
                used[c] = used.get(c, 0) + 1
                bought.append(c)
                visit(j, money - supply[c]['cost'], left - 1, bought, used)
                bought.pop()
                used[c] -= 1
                if not used[c]:
                    del used[c]
        visit(0, coins, buys, [], {})
        # Endings have exact values. Only the best needs simulation comparison.
        endings = [r for r in rows if r['end']]
        best_end = max(endings, key=lambda r: (r['margin'], -len(r['cards'])), default=None)
        ongoing = [r for r in rows if not r['end']]
        # Cap large continuation sets by immediate VP, keeping pass explicitly.
        # Exact endings are compared separately and never enter this pruning.
        ongoing.sort(key=lambda r: (r['margin'], -len(r['cards'])), reverse=True)
        if len(ongoing) > 64:
            ongoing = ongoing[:63] + next(([r] for r in rows if not r['cards']), [])
        seen = {r['cards'] for r in ongoing}
        # Retain the incumbent's entire purchase sequence as an economic option,
        # and a root deviation to each affordable positive policy recommendation.
        roots = {None}
        allowed = {z['label'] for z in o['prompt']['options']}
        roots.update(self.policy_root(o, p, allowed) for p in self.policies)
        for root in sorted(roots, key=lambda x: x or ''):
            if root is None:
                continue
            w = World(o, self.plan, 9311)
            seq = []
            c = root
            for _ in range(buys):
                if not w.buy(0, c):
                    break
                seq.append(c)
                p = w.players[0]
                c = best_buy(p.owned, w.piles, w.costs, self.plan, w.turns[0], p.coins)
            cards = tuple(seq)
            if cards in seen:
                continue
            seen.add(cards)
            row = {'cards': cards, 'end': w.ended(),
                   'margin': points(w.players[0].owned) - points(w.players[1].owned)}
            if row['end']:
                if best_end is None or row['margin'] > best_end['margin']:
                    best_end = row
            else:
                ongoing.append(row)
        return best_end, ongoing

    def basket_rollout(self, o, cards, seed):
        w = World(o, self.plan, seed)
        for c in cards:
            if not w.buy(0, c):
                raise ValueError('infeasible basket')
        # Stop buying explicitly; cleanup precedes the first game-end check.
        return w.rollout(None)

    def select_basket(self, o):
        ending, choices = self.baskets(o)
        if not choices:
            return ending
        remain = self.ctx.budget.remaining().get('wall_s', 0)
        if remain < 30:
            return ending
        started = time.monotonic()
        stop = started + min(6.0, max(0.1, (remain - 25) / 40))
        seed0 = 573713 + int(o['decision_id']) * 65537
        samples = [[] for _ in choices]
        # Reserve half the search time for independent validation.
        screen_stop = started + (stop - started) * 0.5
        for k in range(32):
            if self.ctx.done or time.monotonic() > screen_stop:
                break
            for j, row in enumerate(choices):
                samples[j].append(self.basket_rollout(o, row['cards'], seed0 + k * 104729))
        tested = [j for j, xs in enumerate(samples) if xs]
        if not tested:
            return ending
        means = {j: sum(samples[j]) / len(samples[j]) for j in tested}
        finalists = sorted(tested, key=means.get, reverse=True)[:4]
        validation = {j: [] for j in finalists}
        for k in range(96):
            if self.ctx.done or time.monotonic() > stop:
                break
            for j in finalists:
                validation[j].append(self.basket_rollout(o, choices[j]['cards'], seed0 + 1000000007 + k * 104729))
        values = validation if min((len(v) for v in validation.values()), default=0) >= 12 else {j: samples[j] for j in finalists}
        avg = {j: sum(xs) / len(xs) for j, xs in values.items()}
        winner = max(avg, key=avg.get)
        result = choices[winner]
        xs = values[winner]
        stderr = (sum((x - avg[winner]) ** 2 for x in xs) / max(1, len(xs)-1) / len(xs)) ** 0.5
        if ending is not None and avg[winner] <= ending['margin'] + max(0.5, stderr):
            result = ending
        self.ctx.log('margin_endgame', decision_id=o['decision_id'],
                     exact_ending=ending, selected=result,
                     continuations=[{'cards': choices[j]['cards'], 'mean': round(avg[j], 3),
                                     'n': len(values[j])} for j in finalists],
                     stderr=round(stderr, 3), seconds=round(time.monotonic()-started, 3))
        return result

    def acquisition(self, o, opts):
        if o['prompt']['kind'] != 'buy':
            return super().acquisition(o, opts)
        allowed = {z['label']: z['id'] for z in opts}
        turn = o['turns_taken']['agent']
        if self.queue_turn == turn:
            if not self.buy_queue:
                return []
            c = self.buy_queue.pop(0)
            if c in allowed:
                return [allowed[c]]
            self.ctx.log('basket_invalidated', card=c)
            self.queue_turn = None
            self.buy_queue = []
        empty = sum(d['count'] == 0 for d in o['supply'].values())
        buys = min(8, o['resources']['buys'])
        can_end = (o['supply']['Province']['count'] <= buys or
                   empty + sum(0 < d['count'] <= buys for d in o['supply'].values()) >= 3)
        if can_end or o['supply']['Province']['count'] <= 2:
            try:
                result = self.select_basket(o)
                if result is not None:
                    self.buy_queue = list(result['cards'])
                    self.queue_turn = turn
                    if not self.buy_queue:
                        return []
                    c = self.buy_queue.pop(0)
                    if c in allowed:
                        return [allowed[c]]
                    self.queue_turn = None
                    self.buy_queue = []
            except Exception as e:
                self.ctx.log('endgame_failure', error=type(e).__name__)
        # No win-only gate: the unchanged portfolio controls ordinary building.
        try:
            c = self.select_policy(o, allowed)
        except Exception as e:
            self.ctx.log('portfolio_failure', error=type(e).__name__)
            c = self.policy_root(o, self.plan, allowed)
        return [allowed[c]] if c in allowed else []


def build(ctx):
    return MarginHarness(ctx)


# Narrow public race interrupt for the final iteration.  It only fires after
# our deck has taken at least one Province, so a very large early VP deficit
# cannot force a premature green purchase before the deck has conversion power.
def parity_race(own, enemy, piles, money):
    if money < 8 or piles.get('Province', 0) <= 0:
        return False
    own_p = own.get('Province', 0)
    enemy_p = enemy.get('Province', 0)
    if own_p < 1 or enemy_p < 2:
        return False
    return (piles.get('Province', 8) <= 5 and
            enemy_p - own_p >= 1 and
            points(enemy) - points(own) >= 6)


_BaseWorld = World
class World(_BaseWorld):
    """Keep sampled continuation worlds consistent with the live race rule."""
    def finish_buy(self, i):
        p = self.players[i]
        for _ in range(min(20, p.buys)):
            if i:
                c = self.enemy_buy(p.coins)
            elif self.plan.get('parity_race') and parity_race(
                    p.owned, self.players[1].owned, self.piles, p.coins):
                c = 'Province'
            else:
                c = best_buy(p.owned, self.piles, self.costs, self.plan,
                             self.turns[i], p.coins)
            if not self.buy(i, c):
                break
        p.cleanup(self.rng)


class RaceMarginHarness(MarginHarness):
    def default_plan(self, o):
        plan = super().default_plan(o)
        for p in self.policies:
            p['parity_race'] = True
        plan['parity_race'] = True
        return plan

    def policy_root(self, o, plan, allowed):
        if (plan.get('parity_race') and 'Province' in allowed and parity_race(
                o['self']['owned'], o['opponent']['owned'],
                {c: d['count'] for c, d in o['supply'].items()},
                o['resources']['coins'])):
            return 'Province'
        return super().policy_root(o, plan, allowed)

    def acquisition(self, o, opts):
        if o['prompt'].get('kind') == 'buy':
            allowed = {z['label']: z['id'] for z in opts}
            if ('Province' in allowed and self.plan.get('parity_race') and
                    parity_race(o['self']['owned'], o['opponent']['owned'],
                                {c: d['count'] for c, d in o['supply'].items()},
                                o['resources']['coins'])):
                self.buy_queue = []
                self.queue_turn = None
                self.ctx.log('parity_race', decision_id=o['decision_id'],
                             own_provinces=o['self']['owned'].get('Province', 0),
                             opponent_provinces=o['opponent']['owned'].get('Province', 0),
                             province_pile=o['supply']['Province']['count'])
                return [allowed['Province']]
        return super().acquisition(o, opts)


def build(ctx):
    return RaceMarginHarness(ctx)
