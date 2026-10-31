"""Public-belief rollout controller. No game internals or external models are used."""
from __future__ import annotations

import json
import math
import random
import re
import time
from collections import Counter

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


def default_plan(supply):
    return {'style': 'engine' if 'Chapel' in supply and 'Laboratory' in supply else 'money',
            'terminal': 'Witch' if 'Witch' in supply else 'Militia' if 'Militia' in supply else 'Smithy',
            'duchy_at': 3, 'estate_at': 1, 'copper_floor': 3}


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


def buy_values(owned, piles, costs, plan, turn, money, gain=False):
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
    vals = buy_values(owned, piles, costs, plan, turn, money, gain)
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


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.plan = None
        self.advised = set()
        self.calls = 0
        self.searches = 0
        self.advice_future = None

    def accept_advice(self, text, supply):
        obj = None
        for m in re.finditer(r'\{', text):
            try:
                val, _ = json.JSONDecoder().raw_decode(text[m.start():])
                if isinstance(val, dict) and 'style' in val:
                    obj = val
                    break
            except (ValueError, TypeError):
                pass
        if not obj:
            return
        if isinstance(obj.get('style'), str) and obj['style'] in ('money', 'engine', 'gardens'):
            if obj['style'] != 'gardens' or 'Gardens' in supply:
                self.plan['style'] = obj['style']
        if isinstance(obj.get('terminal'), str) and obj['terminal'] in supply and obj['terminal'] in ('Witch', 'Militia', 'Smithy', 'Moat'):
            self.plan['terminal'] = obj['terminal']
        for k, lo, hi in [('duchy_at', 2, 5), ('estate_at', 0, 2), ('copper_floor', 2, 4)]:
            if isinstance(obj.get(k), (int, float)) and math.isfinite(obj[k]):
                self.plan[k] = max(lo, min(hi, int(obj[k])))
        self.ctx.memory['plan'] = dict(self.plan)
        self.ctx.log('strategic_policy', plan=self.plan, reason=str(obj.get('reason', ''))[:300])

    def collect_advice(self, obs):
        if self.advice_future is None or not self.advice_future.done():
            return
        future, self.advice_future = self.advice_future, None
        try:
            self.accept_advice(future.result().text, obs['supply'])
        except Exception as e:
            self.ctx.log('advice_failure', error=type(e).__name__)

    def race_choice(self, obs, labels):
        """Lock scoring only when the public race is genuinely imminent.

        The prior lock fired at five Provinces with only two opponent Provinces,
        converting productive $8 turns into green cards before the opponent could
        actually end.  Use public VP and Province ownership to distinguish a
        trailing race from a race we currently lead.
        """
        supply = obs.get('supply', {})
        resources = obs.get('resources', {})
        coins = resources.get('coins', 0)
        pleft = supply.get('Province', {}).get('count', 8)
        opp = obs.get('opponent', {}).get('owned', {})
        own = obs.get('self', {}).get('owned', {})
        opp_prov = int(opp.get('Province', 0))
        own_prov = int(own.get('Province', 0))
        opp_vp = obs.get('opponent', {}).get('vp', 0)
        own_vp = obs.get('self', {}).get('vp', 0)
        # Immediate threat: opponent has three Provinces, or is ahead while
        # the pile is short.  A five-card pile by itself is not an emergency.
        threat = (opp_prov >= 3 or (opp_prov >= 2 and pleft <= 3)
                  or (opp_vp > own_vp + 3 and pleft <= 4))
        # Once the pile is down to two, scoring is mandatory even if we lead;
        # otherwise preserve payload and let rollout search choose the timing.
        imminent = pleft <= 2
        if 'Province' in labels and coins >= 8 and (threat or imminent):
            return 'Province'
        # Duchy is a terminal fallback only at one Province or when the
        # opponent already owns three and we cannot buy Province.
        if pleft <= 1 and 'Duchy' in labels and coins >= 5:
            return 'Duchy'
        # Against Witch, buy at most one Moat and only while no scoring card is
        # affordable; this keeps defense from displacing the economy.
        if (obs.get('opponent_id') == 'WM-1' and opp.get('Witch', 0) > 0
                and not own.get('Moat', 0) and pleft >= 5
                and 'Moat' in labels and coins >= 2
                and 'Province' not in labels and 'Duchy' not in labels):
            return 'Moat'
        return None

    def forecast_policies(self, obs):
        """Diagnose continuation-policy bias before asking for strategic advice."""
        if not set(obs['supply']).issubset(SUPPORTED):
            return []
        forecasts = []
        piles = {c: d['count'] for c, d in obs['supply'].items()}
        costs = {c: d['cost'] for c, d in obs['supply'].items()}
        for style in ('money', 'engine', 'gardens'):
            if style == 'gardens' and 'Gardens' not in piles:
                continue
            plan = dict(self.plan, style=style)
            root = best_buy(obs['self']['owned'], piles, costs, plan,
                            obs['turns_taken']['agent'], obs['resources']['coins'])
            values = []
            for k in range(96):
                if self.ctx.done:
                    return forecasts
                values.append(World(obs, plan, 91771 + k * 12347).rollout(root))
            forecasts.append({'style': style, 'mean_final_margin': round(sum(values)/len(values), 2),
                              'samples': len(values), 'first_purchase': root})
        if forecasts:
            self.plan['style'] = max(forecasts, key=lambda x: x['mean_final_margin'])['style']
        self.ctx.log('policy_forecast', forecasts=forecasts, default_plan=self.plan)
        return forecasts

    def advise(self, obs):
        self.collect_advice(obs)
        if self.advice_future is not None:
            return
        turn = obs['turns_taken']['agent']
        checkpoint = 0 if turn < 7 else 1
        if checkpoint in self.advised or (checkpoint == 1 and turn < 9): return
        self.advised.add(checkpoint)
        remain = self.ctx.budget.remaining()
        if remain.get('wall_s', 0) < 350 or remain.get('system2_calls', 0) < 2: return
        try:
            forecasts = self.forecast_policies(obs)
        except Exception as e:
            forecasts = []
            self.ctx.log('forecast_failure', error=type(e).__name__)
        prompt = (
            'You select the strategic continuation policy for a Dominion Monte Carlo controller. '
            'The objective is final VP margin against a disclosed money opponent. Code samples unknown '
            'deck orders and compares purchases by completed-game VP margin. You do NOT choose option IDs. '
            'Pick a coherent style: money (few terminals plus treasures), engine (draw and action support), '
            'gardens (Workshop/Gardens pile rush, only if Gardens present). All styles acquire early '
            'Chapel/Sentry thinning and Witch attacks when useful. The terminal field is the preferred '
            'terminal for limited slots; Witch is automatically prioritized while curses remain. '
            'duchy_at and estate_at are Province counts at which continuation starts buying those cards; '
            'copper_floor is 2..4, reduced by one for each Silver and two for each Gold. '
            'Reply only a JSON object with style, terminal (Witch/Militia/Smithy/Moat), duchy_at (2..5), '
            'estate_at (0..2), copper_floor (2..4), and reason (one short sentence). '
            'The supplied forecasts are approximate final margins using each IMPLEMENTED continuation '
            'policy on 96 sampled hidden worlds. They diagnose policy quality, not actual future outcomes. '
            'Prefer the best forecast unless you identify a specific modeling error: a theoretically '
            'strong archetype can be weak under this particular continuation policy. '
            'Prefer reliable, fast decks against money. Do not construct an engine without payload.\n'
            + json.dumps({'plan': self.plan, 'policy_forecasts': forecasts, 'state': obs}, separators=(',', ':')))
        try:
            self.calls += 1
            self.advice_future = self.ctx.spawn(
                lambda: self.ctx.think(prompt, session_id=None,
                                       timeout_s=min(180, remain['wall_s']-120)))
            if checkpoint == 0:
                # CLI queue admission need not be covered by its inference timeout.
                # Only this local wait can delay progress; subsequent advice is polled.
                try:
                    self.advice_future.result(timeout=60)
                except TimeoutError:
                    self.ctx.log('advice_pending', local_wait_s=60)
                except Exception:
                    pass  # collect_advice records completed-call failures below.
            self.collect_advice(obs)
        except Exception as e:
            self.ctx.log('advice_failure', error=type(e).__name__)

    def search_buy(self, obs):
        p = obs['prompt']
        labels = [o['label'] for o in p['options']]
        race = self.race_choice(obs, labels)
        if race is not None:
            self.ctx.log('race_lock', decision_id=obs.get('decision_id'), selected=race,
                         provinces_left=obs.get('supply', {}).get('Province', {}).get('count'),
                         opponent_provinces=obs.get('opponent', {}).get('owned', {}).get('Province', 0),
                         opponent_vp=obs.get('opponent', {}).get('vp', 0),
                         agent_vp=obs.get('self', {}).get('vp', 0))
            return race
        piles = {c: d['count'] for c, d in obs['supply'].items()}
        costs = {c: d['cost'] for c, d in obs['supply'].items()}
        vals = buy_values(obs['self']['owned'], piles, costs, self.plan, obs['turns_taken']['agent'], obs['resources']['coins'])
        fallback = max(labels + [None], key=lambda c: vals.get(c, 0 if c is None else -100))
        # Unknown rules are not silently simulated. A legal heuristic remains available.
        if not set(obs['supply']).issubset(SUPPORTED): return fallback
        candidates = [c for c in labels if c not in ('Curse', 'Copper') and vals.get(c, -100) >= -20]
        if 'Copper' in labels and self.plan['style'] == 'gardens':
            candidates.append('Copper')
        candidates = sorted(candidates, key=lambda c: vals.get(c, -100), reverse=True)[:7]
        candidates.append(None)
        if len(candidates) < 2: return fallback
        # Common random numbers, successive halving, and independent final batches.
        stats = {c: [0.0, 0] for c in candidates}
        started = time.monotonic()
        seed_base = 19073 + obs['decision_id'] * 1009
        try:
            # Give every root option the same initial sample count.  Early
            # elimination was too sensitive to shuffle luck: an option that
            # lost its first 24 worlds never recovered, even when its true
            # mean was better.  Refinement is only applied after this common
            # base.
            for batch, reps in enumerate((48, 48, 64)):
                for k in range(reps):
                    if self.ctx.done: return fallback
                    if time.monotonic()-started > 9 or self.ctx.budget.remaining().get('wall_s', 0) < 60:
                        break
                    seed = seed_base + batch * 1000003 + k * 7919
                    active = candidates if batch == 0 else candidates[:max(2, (len(candidates)+1)//2)]
                    for c in active:
                        v = World(obs, self.plan, seed).rollout(c)
                        stats[c][0] += v
                        stats[c][1] += 1
                ranked = sorted(candidates, key=lambda c: (stats[c][0]/max(1, stats[c][1]), vals.get(c, 0)), reverse=True)
                if batch == 0:
                    candidates = ranked[:max(2, (len(ranked)+1)//2)]
            best = max(candidates, key=lambda c: (stats[c][0]/max(1, stats[c][1]), vals.get(c, 0)))
            self.searches += 1
            self.ctx.log('buy_rollout', decision_id=obs['decision_id'], selected=best,
                         values={str(c): [round(s/max(1,n), 3), n] for c,(s,n) in stats.items()},
                         seconds=round(time.monotonic()-started, 3))
            return best
        except Exception as e:
            self.ctx.log('search_failure', error=type(e).__name__)
            return fallback

    def choose(self, obs):
        p, own = obs['prompt'], obs['self']['owned']
        opts = p['options']
        labels = [o['label'] for o in opts]
        hand = expand(obs['self'].get('hand', {}))
        piles = {c:d['count'] for c,d in obs['supply'].items()}
        costs = {c:d['cost'] for c,d in obs['supply'].items()}
        kind, src = p['kind'], p.get('source') or ''
        chosen = []
        if kind == 'buy':
            c = self.search_buy(obs)
            chosen = [c] if c else []
        elif kind == 'play_action':
            c = max(labels, key=lambda x: action_value(x, hand, own, piles, self.plan, obs['resources']['actions']), default=None)
            if c and action_value(c, hand, own, piles, self.plan, obs['resources']['actions']) >= 0: chosen = [c]
        elif kind == 'trash':
            if src == 'Mine':
                c = 'Silver' if 'Silver' in labels and piles.get('Gold') else 'Copper' if 'Copper' in labels and piles.get('Silver') else None
                chosen = [c] if c else []
            elif src == 'Remodel':
                c = min(labels, key=lambda x: keep_value(x, hand, own, piles, self.plan), default=None)
                if piles.get('Province', 8) <= 3 and 'Gold' in labels: c = 'Gold'
                chosen = [c] if c else []
            else:
                chosen = trash_cards(labels, own, piles, self.plan, p['max'])
        elif kind == 'gain':
            vals = buy_values(own, piles, costs, self.plan, obs['turns_taken']['agent'], max([costs.get(c, 0) for c in labels] or [0]), True)
            c = max(labels, key=lambda x: vals.get(x, -100), default=None)
            if c: chosen = [c]
        elif kind == 'discard':
            if p['min']:
                chosen = sorted(labels, key=lambda x: keep_value(x, hand, own, piles, self.plan))[:p['min']]
            else:
                chosen = [x for x in labels if x in VICTORY or x == 'Curse']
                if src == 'Cellar':
                    terminal = [x for x in labels if x in ACTIONS and x not in PLUS]
                    if len(terminal) > obs['resources']['actions'] and not any(x in ('Village', 'Festival') for x in hand):
                        chosen += sorted(terminal, key=lambda x: action_value(x, hand, own, piles, self.plan))[:len(terminal)-obs['resources']['actions']]
        elif 'True' in labels or 'False' in labels:
            truth = src == 'Moat' or 'reveal' in p.get('instruction', '').lower()
            if src == 'Sentry':
                rel = p.get('relevant_cards', [])
                if len(rel) == 2 and all(isinstance(c, str) for c in rel):
                    truth = keep_value(rel[1], hand, own, piles, self.plan) > keep_value(rel[0], hand, own, piles, self.plan)
            chosen = ['True' if truth else 'False']
        selected, pool = [], list(opts)
        for name in chosen:
            for i, o in enumerate(pool):
                if o['label'] == name:
                    selected.append(str(o['id']))
                    pool.pop(i)
                    break
        selected = selected[:p['max']]
        for o in pool:
            if len(selected) >= p['min']: break
            if str(o['id']) not in selected: selected.append(str(o['id']))
        return selected

    def run_episode(self, game):
        obs = self.ctx.observe()
        previous, repeats = None, 0
        while not self.ctx.done:
            if isinstance(obs, str):
                try: obs = json.loads(obs)
                except ValueError: return
            if not isinstance(obs, dict) or obs.get('terminal') or not obs.get('prompt'): return
            if self.plan is None: self.plan = default_plan(obs['supply'])
            if obs['prompt']['kind'] == 'buy': self.advise(obs)
            if self.ctx.done: return
            ids = self.choose(obs)
            did = obs['decision_id']
            repeats = repeats + 1 if did == previous else 0
            previous = did
            if repeats:
                p = obs['prompt']
                ids = [str(o['id']) for o in p['options'][:p['min']]]
            if repeats > 3: return
            action = 'choose ' + str(did) + ((' ' + ' '.join(ids)) if ids else '')
            try:
                obs = self.ctx.act(action)
                if obs is None: obs = self.ctx.observe()
            except Exception as e:
                if self.ctx.done: return
                self.ctx.log('action_failure', error=type(e).__name__)
                obs = self.ctx.observe()


def build(ctx):
    return Harness(ctx)
