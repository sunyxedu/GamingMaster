"""Candidate 0018: condition candidate 0009's worlds on public opponent zones."""
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


def opponent_rank(card):
    """The disclosed money bots keep their terminal, then money, then VP."""
    return (100 if card in ('Smithy', 'Witch') else
            MONEY.get(card, 0) * 10 +
            {'Province': 6, 'Duchy': 3, 'Estate': 1, 'Curse': -1}.get(card, 0))


class OpponentLedger:
    """Conservative facts, never a reconstruction of an unrevealed order.

    Each own-turn observation closes at most one enemy turn. Intermediate
    reaction observations are buffered so that cleanup can be placed after
    the enemy's last public play/buy. The supported bots only draw with their
    one Smithy/Witch; every such play is public. A shuffle clears known discard
    membership. Unrevealed cleanup cards stay unknown.
    """
    def __init__(self):
        self.last_id = None
        self.pending = []
        self.turn = 0
        self.hand = 5
        self.deck = 5
        self.discard = 0
        self.played = Counter()
        self.known = Counter()
        self.floor = None
        self.resets = 0
        self.shuffles = 0
        self.reason = None

    def draw(self, count):
        if count:
            self.floor = None
        for _ in range(count):
            if self.deck == 0:
                self.deck, self.discard = self.discard, 0
                self.known.clear()
                self.shuffles += 1
            if self.deck == 0:
                break
            self.deck -= 1
            self.hand += 1

    def event(self, event):
        if event.get('player') != 'opponent':
            return
        kind, card = event.get('event'), event.get('card')
        if kind == 'play':
            self.hand -= 1
            self.played[card] += 1
            if card in ACTIONS:
                # Only the disclosed bots' terminal is supported here. Fail
                # closed if a future opponent introduces another action.
                if card not in ('Smithy', 'Witch'):
                    raise ValueError('unexpected_opponent_action')
                self.draw(DRAW[card])
        elif kind in ('buy', 'gain'):
            self.discard += 1
            self.known[card] += 1
        elif kind == 'discard':
            self.hand -= 1
            self.discard += 1
            self.known[card] += 1
            rank = opponent_rank(card)
            self.floor = rank if self.floor is None else max(self.floor, rank)
        elif kind == 'trash':
            raise ValueError('unexpected_opponent_trash')

    def cleanup(self):
        self.discard += self.hand + sum(self.played.values())
        self.known.update(self.played)
        self.hand = 0
        self.played.clear()
        self.floor = None
        self.draw(5)

    def reset_to(self, obs, reason):
        enemy = obs['opponent']
        self.hand = enemy['hand_count']
        self.deck = enemy['deck_count']
        self.discard = enemy['discard_count']
        self.played = Counter(enemy.get('in_play', {}))
        self.known.clear()
        self.floor = None
        self.reason = reason
        self.resets += 1

    def update(self, obs):
        if obs.get('decision_id') == self.last_id:
            return self.facts()
        self.last_id = obs.get('decision_id')
        self.pending.extend(obs.get('public_log_delta', []))
        if obs.get('active_player') != 'agent':
            return None
        events, self.pending = self.pending, []
        turns = obs['turns_taken']['opponent']
        self.reason = None
        try:
            elapsed = turns - self.turn
            if elapsed not in (0, 1):
                raise ValueError('missed_opponent_turn')
            # A gain/discard can be caused by OUR attack before or after the
            # opposing turn. Only play/buy identifies the opponent's own turn.
            active = [i for i, e in enumerate(events)
                      if e.get('player') == 'opponent' and e.get('event') in ('play', 'buy')]
            last = active[-1] if active else len(events) - 1
            for i, event in enumerate(events):
                self.event(event)
                if elapsed and i == last:
                    self.cleanup()
            if elapsed and not events:
                self.cleanup()
            enemy = obs['opponent']
            if (self.hand, self.deck, self.discard) != (
                    enemy['hand_count'], enemy['deck_count'], enemy['discard_count']):
                raise ValueError('zone_count_mismatch')
            if +self.played != +Counter(enemy.get('in_play', {})):
                raise ValueError('played_count_mismatch')
            remaining = Counter(enemy['owned']) - self.played
            if any(n < 0 for n in (self.hand, self.deck, self.discard)):
                raise ValueError('negative_zone')
            if sum(self.known.values()) > self.discard or self.known - remaining:
                raise ValueError('known_discard_mismatch')
            if sum(remaining.values()) != self.hand + self.deck + self.discard:
                raise ValueError('ownership_mismatch')
            if self.floor is not None:
                eligible = sum(n for c, n in (remaining-self.known).items()
                               if opponent_rank(c) >= self.floor)
                if eligible < self.hand:
                    raise ValueError('retention_mismatch')
        except (ValueError, KeyError, TypeError) as exc:
            self.reset_to(obs, str(exc))
        self.turn = turns
        return self.facts()

    def facts(self):
        return {'discard': dict(+self.known), 'hand_rank_floor': self.floor,
                'resets': self.resets, 'shuffles': self.shuffles,
                'reset_reason': self.reason}


def sample_opponent(unknown, hand_count, discard_count, rng, facts=None):
    """Uniform residual allocation subject to certified public constraints."""
    known = Counter((facts or {}).get('discard', {}))
    floor = (facts or {}).get('hand_rank_floor')
    valid = not (known - unknown) and sum(known.values()) <= discard_count
    pool = expand(unknown - known) if valid else expand(unknown)
    if not valid:
        known, floor = Counter(), None
    rng.shuffle(pool)
    eligible = sum(floor is None or opponent_rank(c) >= floor for c in pool)
    if eligible < hand_count:
        # Defensive fallback: do not invent a card to satisfy a stale bound.
        floor = None
    hand, rest = [], []
    for card in pool:
        if len(hand) < hand_count and (floor is None or opponent_rank(card) >= floor):
            hand.append(card)
        else:
            rest.append(card)
    missing = discard_count - sum(known.values())
    discard = expand(known) + rest[:missing]
    return hand, discard, rest[missing:]


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
                hand, discard, deck = sample_opponent(
                    unknown, d['hand_count'], d['discard_count'], self.rng,
                    obs.get('_opponent_belief'))
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



# Complete-policy portfolio implementation follows.

def policy(name, style='money', terminal='Smithy', province_at=8, copper_floor=4, **extra):
    return dict(name=name, style=style, terminal=terminal, province_at=province_at,
                duchy_at=3, estate_at=1, copper_floor=copper_floor, **extra)


def default_plan(supply):
    terminal = 'Witch' if 'Witch' in supply else 'Militia' if 'Militia' in supply else 'Smithy'
    return policy('thin_money', terminal=terminal)


def portfolio(supply):
    plans = [default_plan(supply)]
    for terminal in ('Smithy', 'Militia', 'Witch'):
        if terminal in supply:
            plans.append(policy('money_'+terminal, terminal=terminal, thin=False))
            plans.append(policy('thin_'+terminal, terminal=terminal))
    if any(c in supply for c in ('Market', 'Laboratory')):
        plans.append(policy('cantrip_money', style='cantrip', terminal=default_plan(supply)['terminal']))
    if ('Village' in supply or 'Festival' in supply) and any(c in supply for c in ('Smithy','Witch','Laboratory')):
        plans.append(policy('engine_race', style='engine', terminal=default_plan(supply)['terminal'], province_at=7))
        plans.append(policy('engine_build', style='engine', terminal=default_plan(supply)['terminal'], province_at=5))
        if 'Chapel' in supply:
            plans.append(policy('thin_engine', style='engine', terminal=default_plan(supply)['terminal'], province_at=6, copper_floor=3))
    if 'Chapel' in supply and 'Laboratory' in supply:
        attack = 'Witch' if 'Witch' in supply else 'Militia' if 'Militia' in supply else 'Smithy'
        # A nonterminal draw chain is a structurally different engine from
        # Village/Smithy: buy its payload first, then let Labs find it reliably.
        targets = [['Chapel',1],['Silver',2],[attack,1],['Gold',2],
                   ['Laboratory',4],['Market',1],['Gold',4],['Laboratory',7],['Market',2]]
        targets = [row for row in targets if row[0] in supply]
        plans.append(policy('lab_chain',style='cantrip',terminal=attack,province_at=6,targets=targets))
    unique = []
    for plan in plans:
        signature = {k:v for k,v in plan.items() if k!='name'}
        if not any({k:v for k,v in x.items() if k!='name'} == signature for x in unique):
            unique.append(plan)
    return unique


def buy_values(owned, piles, costs, plan, turn, money, gain=False):
    """Shared acquisition policy: real and simulated decisions use identical logic."""
    n = sum(owned.values())
    p = piles.get('Province', 8)
    wealth = sum(MONEY.get(c, 0)*v for c,v in owned.items())
    payload = wealth + 2*owned.get('Militia',0) + 2*owned.get('Festival',0) + owned.get('Market',0)
    draws = sum(DRAW.get(c,0)*v for c,v in owned.items())
    terminal = plan.get('terminal','Smithy')
    style = plan.get('style','money')
    term = sum(owned.get(c,0) for c in ('Smithy','Witch','Militia','Moat','Workshop','Mine','Remodel'))
    capacity = owned.get('Village',0) + owned.get('Festival',0)
    junk = owned.get('Copper',0)+owned.get('Estate',0)+owned.get('Curse',0)
    v = {c:-100.0 for c in costs}
    v.update(Gold=45, Silver=25, Province=100 if p<=plan.get('province_at',8) else 22,
             Duchy=92 if p<=plan.get('duchy_at',3) else -20,
             Estate=50 if p<=plan.get('estate_at',1) else -30)
    if p == 0:
        v.update(Duchy=98, Estate=80, Gardens=90 if n>=30 else 60, Copper=-20)
    if p<=2:
        v['Duchy']=95
        if p<=1:v['Estate']=85
    # Astra's one candidate has precisely the parent's target-list semantics.
    if 'targets' in plan:
        for index,(card,count) in enumerate(plan['targets']):
            if owned.get(card,0)<count and not (card=='Witch' and not piles.get('Curse',0)):
                v[card]=max(v.get(card,-100), 80-index)
    elif p>=4:
        thin=plan.get('thin',True)
        if thin and not owned.get('Chapel') and turn<=5 and junk>=5:
            v['Chapel']=85 if turn<=2 else 50
        if thin and 'Chapel' not in piles and not owned.get('Sentry') and junk>=6:
            v['Sentry']=67 if turn<=9 else 42
        # All portfolios preserve their own terminal specialty and capacity.
        if terminal=='Witch' and piles.get('Curse',0) and owned.get('Witch',0)<1:
            v['Witch']=82
        elif not owned.get(terminal) and term<1:
            v[terminal]=65
        if style in ('cantrip','engine'):
            if not owned.get('Gold') and wealth>=5:
                v['Gold']=73
            if wealth<5:
                v['Silver']=77
            if style=='cantrip':
                v['Laboratory']=54 if owned.get('Laboratory',0)<4 and wealth>=7 else 20
                v['Market']=49 if owned.get('Market',0)<2 and wealth>=7 else 22
                if term>capacity+1:
                    v['Village']=44
                    v['Festival']=48
            else:
                # Build enough actions to PLAY the acquired draw, enough draw to
                # FIND payload, and enough money/buys to turn it into points.
                need_actions=term + int(owned.get('Chapel',0)>0 and junk>3) - capacity - 1
                if need_actions>0:
                    v['Village']=74
                    v['Festival']=78
                draw_gap=n+2-5-draws
                if draw_gap>0:
                    v['Laboratory']=65
                    v['Smithy']=67 if capacity>=term else 35
                    if terminal=='Witch' and piles.get('Curse',0) and owned.get('Witch',0)<2:
                        v['Witch']=76 if capacity>=term else 39
                else:
                    v['Laboratory']=31
                    v['Smithy']=15
                # Early rebuild and payload expansion win over decorative draw.
                target_payload=10 if n<13 else 16
                if payload<target_payload:
                    v['Gold']=76 if payload<8 else 62
                    v['Silver']=69 if payload<6 else 38
                    v['Festival']=max(v.get('Festival',-100),60 if term>capacity else 30)
                if payload>=8 and owned.get('Market',0)+owned.get('Festival',0)<1:
                    v['Market']=70
                    v['Festival']=max(v.get('Festival',-100),68)
                elif payload>=15 and owned.get('Market',0)+owned.get('Festival',0)<2:
                    v['Market']=64
                if payload>=16 and draws+5>=n:
                    v['Province']=110
        if owned.get('Silver',0)>=2 and owned.get('Merchant',0)<1:
            v['Merchant']=29 if style!='money' else 20
        if gain:
            v['Gardens']=max(v.get('Gardens',-100),35 if n>=25 else -5)
    if p == 0 or sum(count == 0 for count in piles.values()) >= 3:
        current = points(owned)
        for card in costs:
            gained = Counter(owned)
            gained[card] += 1
            v[card] = 100 * (points(gained)-current)
            if v[card] <= 0:
                v[card] = -1
    # A mandatory cheap gain still favors economy over dead weight.
    if gain:
        for c,value in [('Silver',25),('Merchant',18),('Village',8),('Moat',3),('Estate',1),('Copper',0)]:
            v[c]=max(v.get(c,-100),value)
    return {c:value for c,value in v.items() if piles.get(c,0)>0 and costs.get(c,100)<=money}


class Harness:
    def __init__(self, ctx):
        self.ctx=ctx
        self.plan=None
        self.plans=[]
        self.future=None
        self.model_started=False
        self.last_scores=[]
        self.searches=0
        self.search_enabled=True
        self.opponent_ledger=OpponentLedger()

    def accept_advice(self, text, supply):
        obj=None
        for m in re.finditer(r'\{',str(text)):
            try:
                data,_=json.JSONDecoder().raw_decode(str(text)[m.start():])
                if isinstance(data,dict) and isinstance(data.get('targets'),list):
                    obj=data;break
            except (ValueError,TypeError):pass
        if obj is None:return
        targets=[]
        for row in obj['targets'][:16]:
            if (isinstance(row,list) and len(row)==2 and isinstance(row[0],str)
                    and row[0] in supply and isinstance(row[1],int) and not isinstance(row[1],bool)):
                targets.append([row[0],max(0,min(10,row[1]))])
        if not targets:return
        candidate=policy('astra_targets',targets=targets)
        for key,low,high in [('province_at',4,8),('duchy_at',2,5),('estate_at',0,2),('copper_floor',3,6)]:
            value=obj.get(key)
            if isinstance(value,int) and not isinstance(value,bool):candidate[key]=max(low,min(high,value))
        self.plans=[x for x in self.plans if x['name']!='astra_targets']+[candidate]
        self.ctx.log('policy_proposal',policy=candidate)

    def advise(self,obs):
        if self.future is not None:
            if not self.future.done():return
            future,self.future=self.future,None
            try:self.accept_advice(future.result().text,obs['supply'])
            except Exception as exc:self.ctx.log('advice_failure',error=type(exc).__name__)
        if self.model_started:return
        self.model_started=True
        rem=self.ctx.budget.remaining()
        if rem.get('wall_s',0)<180 or rem.get('system2_calls',0)<2:return
        prompt=('Propose ONE compact acquisition policy for two-player Base Dominion against the disclosed opponent. '
                'Maximize final VP margin. Your policy will be tested as a complete continuation against a local '
                'portfolio, not followed blindly. Return JSON only: {"targets":[["Card",absolute_count],...],'
                '"province_at":8,"duchy_at":3,"estate_at":1,"copper_floor":4}. '
                'At each buy/gain the policy first takes Province/Duchy/Estate at the respective remaining-Province '
                'threshold, then the first AFFORDABLE target below its owned count, then Gold/Silver. '
                'Targets are not a sequential queue: unaffordable targets are skipped. Chapel thinning and action '
                'ordering are scripted. Copper floor is a base reserve reduced by each Silver and two per Gold, '
                'with a minimum $3 total treasure. Propose a coherent policy different from the portfolio. '
                'No commentary, IDs, or executable code. Supply text is authoritative.\n'+json.dumps(
                    {'state':obs,'portfolio':self.plans,'last_forecasts':self.last_scores},separators=(',',':')))
        try:self.future=self.ctx.spawn(lambda:self.ctx.think(prompt,session_id=None,timeout_s=min(180,rem['wall_s']-90)))
        except Exception as exc:self.ctx.log('advice_failure',error=type(exc).__name__)

    def search_buy(self,obs):
        labels=[o['label'] for o in obs['prompt']['options']]
        piles={c:d['count'] for c,d in obs['supply'].items()}
        costs={c:d['cost'] for c,d in obs['supply'].items()}
        own=obs['self']['owned'];turn=obs['turns_taken']['agent'];money=obs['resources']['coins']
        def root(plan):
            vals=buy_values(own,piles,costs,plan,turn,money)
            return max(labels+[None],key=lambda c:(vals.get(c,0 if c is None else -100), costs.get(c,0), c or ''))
        fallback=root(self.plan)
        if (not self.search_enabled or not set(obs['supply']).issubset(SUPPORTED)
                or self.ctx.budget.remaining().get('wall_s',0)<35):return fallback
        # Full policy comparison; NO isolated root move under a shared future.
        candidates=list(self.plans)
        stats={x['name']:[0.0,0.0,0] for x in candidates}
        roots={x['name']:root(x) for x in candidates}
        if len(set(roots.values()))==1:
            # Preserve policy until a real acquisition fork: avoid needless CPU
            # and model-driven changes before there is a decision to distinguish.
            return fallback
        started=time.monotonic()
        remaining=self.ctx.budget.remaining().get('wall_s',0)
        allowance=min(14.0,max(0.2,(remaining-45)/35))
        seed_base=7331+obs['decision_id']*1013
        try:
            survivors=candidates
            for batch,reps in enumerate((32,64,96)):
                for k in range(reps):
                    if self.ctx.done:return fallback
                    if time.monotonic()-started>allowance:break
                    seed=seed_base+batch*1000033+k*7907
                    for plan in survivors:
                        val=World(obs,plan,seed).rollout(roots[plan['name']])
                        a=stats[plan['name']];a[0]+=val;a[1]+=val*val;a[2]+=1
                ranked=sorted(survivors,key=lambda p:stats[p['name']][0]/max(1,stats[p['name']][2]),reverse=True)
                survivors=ranked[:3]
                incumbent=next((x for x in candidates if x['name']==self.plan['name']),None)
                if incumbent and incumbent not in survivors:survivors.append(incumbent)
                if time.monotonic()-started>allowance:break
            viable=[p for p in survivors if stats[p['name']][2]>0]
            if not viable:return fallback
            best=max(viable,key=lambda p:stats[p['name']][0]/stats[p['name']][2])
            current=stats.get(self.plan['name'],[0,0,0])
            best_stat=stats[best['name']]
            improvement=best_stat[0]/best_stat[2]-current[0]/max(1,current[2])
            if current[2]==0 or improvement>0.75:
                self.plan=dict(best)
            self.last_scores=[{'name':p['name'],'root':roots[p['name']],
                'mean':round(stats[p['name']][0]/max(1,stats[p['name']][2]),3),
                'n':stats[p['name']][2]} for p in candidates]
            self.searches+=1
            self.ctx.memory['policy']=dict(self.plan)
            self.ctx.log('policy_comparison',decision_id=obs['decision_id'],selected=self.plan['name'],
                         root=roots[self.plan['name']],forecasts=self.last_scores,
                         seconds=round(time.monotonic()-started,3))
            return roots[self.plan['name']]
        except Exception as exc:
            self.ctx.log('policy_search_failure',error=type(exc).__name__)
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
            c = max(labels, key=lambda x: (vals.get(x, -100), costs.get(x, 0), x), default=None)
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
            obs = dict(obs)
            obs['_opponent_belief'] = self.opponent_ledger.update(obs)
            if obs['prompt']['kind'] == 'buy':
                self.ctx.log('opponent_belief', decision_id=obs['decision_id'],
                             facts=obs['_opponent_belief'])
            if self.plan is None:
                self.plan = default_plan(obs['supply'])
                self.plans = portfolio(obs['supply'])
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
