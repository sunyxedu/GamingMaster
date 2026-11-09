"""Public-board tactical controller with sparse Astra arbitration.
No game imports, hidden state, or persistent cross-episode information.
"""
from __future__ import annotations
import itertools
import json
import re
from concurrent.futures import TimeoutError as FutureTimeout
from functools import lru_cache

FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
INTERACTION = {'Book of Moon', 'Rush Recklessly', 'Compulsory Evacuation Device',
               'Trap Hole', 'Mirror Force', 'Mystical Space Typhoon', 'Call of the Haunted'}


def number(x, default=0):
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def unpack(obs):
    if isinstance(obs, str):
        try:
            return json.loads(obs)
        except (ValueError, TypeError):
            return {}
    return obs if isinstance(obs, dict) else {}


def ident(c):
    return (c.get('player'), c.get('zone'), c.get('slot'))


def attack(c):
    return 'attack' in str(c.get('position', '')).lower()


def facedown(c):
    return 'down' in str(c.get('position', '')).lower()


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.round = None
        self.used_calls = 0
        self.duel_calls = 0
        self.asked = set()
        self.last_effect = ''
        self.last_card = {}
        self.target_names = []
        self.last_token = None
        self.failures = {}
        self.attack_card = {}
        self.plan_note = ''
        self.pending = None
        self.reserve = set()
        self.reserve_turn = None
        self.repeated_states = {}
        self.repeat_turn = None

    def setup(self, obs):
        self.obs = obs
        self.p = obs.get('prompt') or {}
        self.seat = obs.get('agent_seat', 0)
        players = obs.get('players', [])
        self.me = next((p for p in players if p.get('player') == self.seat), {})
        self.opp = next((p for p in players if p.get('player') != self.seat), {})
        self.mine = self.me.get('monsters', [])
        self.enemy = self.opp.get('monsters', [])
        self.own_turn = obs.get('turn_player') == self.seat
        self.catalog = obs.get('card_catalog', {})
        self.cards = {}
        for p in players:
            for zone in ('hand', 'monsters', 'spells_traps', 'graveyard', 'banished'):
                for c in p.get(zone, []):
                    self.cards[ident(c)] = c
        if obs.get('round') != self.round:
            self.round = obs.get('round')
            self.duel_calls = 0
            self.asked.clear()
            self.last_effect = ''
            self.target_names = []
            self.plan_note = ''
            self.attack_card = {}
            self.reserve = set()
            self.reserve_turn = None
        if self.reserve_turn != obs.get('turn'):
            self.reserve.clear()
        if self.repeat_turn != (self.round, obs.get('turn')):
            self.repeated_states.clear()
            self.repeat_turn = (self.round, obs.get('turn'))
        if self.p.get('mode') == 'subset' and 'attack target' not in str(self.p.get('hint','')).lower() and 'tribute' not in self.p.get('kind',''):
            # Mandatory flip triggers need not have offered an activation choice.
            for e in reversed(obs.get('recent_events', [])):
                if e.get('event') == 'chain activation':
                    source = e.get('card', {})
                    if source.get('player') == self.seat and source.get('name'):
                        self.last_effect = source['name']
                    break
                if e.get('event') in ('chain ended','new_turn'):
                    break

    def full(self, c):
        if not isinstance(c, dict):
            return {}
        base = dict(self.catalog.get(str(c.get('code')), {}))
        found = self.cards.get(ident(c), {})
        if found and (not c.get('code') or found.get('code') == c.get('code')):
            base.update(found)
        base.update(c)
        return base

    def value(self, c):
        c = self.full(c)
        if c.get('zone') == 'spells_traps':
            return 1700 if c.get('name') == 'Mirror Force' else 1300
        atk, dfn = number(c.get('attack')), number(c.get('defense'))
        val = 800 + max(atk, dfn * .65) * .6
        if c.get('name') in FLIPS and facedown(c):
            val += 700
        return val

    def strongest(self, cards, default=0):
        return max((number(self.full(c).get('attack')) for c in cards), default=default)

    def battle(self):
        # The latest attack announcement remains in the rolling event stream.
        events = self.obs.get('recent_events', [])
        for e in reversed(events):
            if e.get('event') in ('attack', 'attack_announce', 'attack_declared'):
                a = e.get('attacker', e.get('card', {}))
                d = e.get('target', e.get('defender', {}))
                # The native direct-attack target is a zone "0" reference.
                return self.full(a or {}), self.full(d) if d and d.get('zone') == 'monsters' else {}
            if e.get('event') in ('new_turn',):
                break
        return {}, {}

    def activation_score(self, card):
        c = self.full(card)
        name = c.get('name', '')
        if self.own_turn and self.obs.get('phase') in ('Main 1','Main 2') and name in self.reserve:
            return -3000
        ownmax = self.strongest(self.mine)
        enemax = self.strongest(self.enemy)
        ownatk = [x for x in self.mine if attack(x)]
        enemyup = [x for x in self.enemy if not facedown(x)]
        a, d = self.battle()
        in_battle = 'battle' in str(self.obs.get('phase', '')).lower() or 'damage' in str(self.obs.get('phase', '')).lower()
        chain = self.obs.get('chain', [])
        # Engine legal choices decide timing; these scores decide resource use.
        if name == 'Mirror Force':
            return 13000 if any(attack(x) for x in self.enemy) else -500
        if name == 'Trap Hole':
            return 11500 if self.enemy else -500
        if name in FLIPS:
            return 12000
        if name == 'Fissure':
            if not enemyup:
                return -2000
            weakest = min(enemyup, key=lambda x: number(x.get('attack')))
            can_kill = any(number(x.get('attack')) > (number(weakest.get('attack')) if attack(weakest) else number(weakest.get('defense'))) for x in ownatk)
            if not ownatk and any(facedown(x) and x.get('name') in ('Penguin Soldier','Man-Eater Bug') for x in self.mine):
                return -500  # Let the Set flip monster supply the removal first.
            return 6400 if not can_kill else 3200
        if name == 'Dark Hole':
            delta = sum(self.value(x) for x in self.enemy) - sum(self.value(x) for x in self.mine)
            if not self.enemy:
                return -6000
            if not self.mine or delta > 900 or (enemax > ownmax and delta > -300):
                return 8500 + delta
            return -2000
        if name in ('Monster Reborn', 'Call of the Haunted'):
            pool = self.me.get('graveyard', []) + (self.opp.get('graveyard', []) if name == 'Monster Reborn' else [])
            mons = [self.full(x) for x in pool if number(self.full(x).get('type')) & 1]
            best = self.strongest(mons)
            if len(self.mine) >= 5 or best < 1200:
                return -500
            if self.own_turn:
                return 6500 + best if len(self.mine) < 3 or enemax >= ownmax else 1800
            if in_battle and (not self.mine or enemax > ownmax):
                return 8000 + best
            return -1000
        if name == 'Mystical Space Typhoon':
            targets = self.opp.get('spells_traps', [])
            if not targets:
                return -2000
            # Destruction is not negation: don't spend MST on an already active normal spell/trap.
            if chain:
                return 7500 if any(x.get('name') == 'Call of the Haunted' for x in targets) else -1500
            return 7200 if self.own_turn and self.mine else 1900 if self.own_turn else -500
        if name == 'Compulsory Evacuation Device':
            if not self.enemy:
                return -2000
            if self.own_turn:
                return 6900 if enemax >= ownmax or (len(self.enemy) == 1 and ownmax > 0) else 2500
            if in_battle:
                return 8500 if not d or (a and number(a.get('attack')) >= number(d.get('attack'))) else 1200
            return 7500 if enemax >= 2400 else -1000
        if name == 'Book of Moon':
            if in_battle and a:
                if a.get('player') != self.seat:
                    if not d or number(a.get('attack')) >= (number(d.get('attack')) if attack(d) else number(d.get('defense'))):
                        return 7900
                elif d and number(d.get('attack')) >= number(a.get('attack')) and number(d.get('defense')) < number(a.get('attack')):
                    return 6800
            if self.own_turn and any(attack(x) and number(x.get('attack')) >= ownmax > number(x.get('defense')) for x in self.enemy):
                return 6800
            return -1000
        if name == 'Rush Recklessly':
            if a and d and in_battle:
                if a.get('player') == self.seat:
                    enemy_stat = number(d.get('attack')) if attack(d) else number(d.get('defense'))
                    ownstat = number(a.get('attack'))
                elif d.get('player') == self.seat and attack(d):
                    ownstat, enemy_stat = number(d.get('attack')), number(a.get('attack'))
                else:
                    return -1000
                if ownstat <= enemy_stat < ownstat + 700:
                    return 9000
            if a and not d and a.get('player') == self.seat and number(self.opp.get('lp'),8000) <= number(a.get('attack')) + 700:
                return 10000
            return -1000
        return 200

    def target_score(self, card):
        c = self.full(card)
        enemy = c.get('player') != self.seat
        name = self.last_effect
        hint = str(self.p.get('hint', '')).lower()
        val = self.value(c)
        if 'attack target' in hint:
            if ident(c) == getattr(self, 'planned_target', None):
                return 30000
            a = self.full(self.attack_card)
            atk = number(a.get('attack'))
            defense = 1800 if facedown(c) else number(c.get('attack') if attack(c) else c.get('defense'))
            return val + (atk-defense if attack(c) else 0) if atk>defense else -10000+atk-defense
        if name in ('Monster Reborn', 'Call of the Haunted') or 'special summon' in hint:
            return number(c.get('attack')) * 2 + (400 if c.get('name') in FLIPS else 0)
        if name == 'Magician of Faith' or ('hand' in hint and c.get('zone') == 'graveyard'):
            ratings = {'Monster Reborn': 6000, 'Fissure': 5200, 'Dark Hole': 4500 if len(self.enemy)>len(self.mine) else 1700,
                       'Mystical Space Typhoon': 4000 if self.opp.get('spells_traps') else 1900,
                       'Book of Moon': 3000, 'Rush Recklessly': 2300}
            return ratings.get(c.get('name'), 1000)
        if 'tribute' in hint or 'discard' in hint or 'release' in hint:
            dead = number(c.get('level')) >= 7 and c.get('zone') == 'hand'
            return -val + (2500 if dead else 0)
        if name == 'Rush Recklessly':
            a, d = self.battle()
            battle_own = a if a.get('player') == self.seat else d
            return (10000 if ident(c) == ident(battle_own) else val) if not enemy else -10000
        if name == 'Book of Moon':
            a, d = self.battle()
            if enemy:
                return 8000 if ident(c) == ident(a) and not self.own_turn else val + number(c.get('attack')) - number(c.get('defense'))
            return -5000
        if name == 'Mystical Space Typhoon':
            return (val + (1500 if c.get('name') == 'Call of the Haunted' else 0)) if enemy else -10000
        if enemy:
            return val + (1000 if number(c.get('level')) >= 5 else 0)
        if name == 'Penguin Soldier' and c.get('name') in FLIPS:
            return 400
        return -val

    def battle_plan(self, choices):
        """Search public battle assignments, replan after each engine observation.
        Unknown defenders are anonymous 1800 DEF blockers, never assumed identities.
        """
        attackers = []
        for x in choices:
            if 'attack' in x.get('label', '').lower() and x.get('card'):
                c = self.full(x['card'])
                if not any(ident(c) == ident(a[1]) for a in attackers):
                    attackers.append((x['action'], c))
        if not attackers:
            return {}
        defenders = self.enemy
        av = tuple(number(c.get('attack')) for _, c in attackers)
        dv = tuple(1800 if facedown(c) else number(c.get('attack') if attack(c) else c.get('defense')) for c in defenders)
        lp = number(self.opp.get('lp'), 8000)
        n, m = len(av), len(dv)
        @lru_cache(None)
        def solve(amask, dmask, damage):
            best = (1000000 if damage >= lp else damage * .65, ())
            for i in range(n):
                if not amask & (1 << i):
                    continue
                rest = amask ^ (1 << i)
                if not dmask:
                    val, path = solve(rest, dmask, min(lp, damage + av[i]))
                    if val > best[0]:
                        best = (val, ((i, -1),) + path)
                    continue
                for j in range(m):
                    if not dmask & (1 << j):
                        continue
                    is_atk = attack(defenders[j]) and not facedown(defenders[j])
                    if av[i] > dv[j]:
                        gain = self.value(defenders[j])
                        if facedown(defenders[j]):
                            gain -= 450  # unknown flip effects can trade back
                        dealt = av[i] - dv[j] if is_atk else 0
                        val, path = solve(rest, dmask ^ (1 << j), min(lp, damage + dealt))
                        val += gain
                    elif av[i] == dv[j] and is_atk:
                        val, path = solve(rest, dmask ^ (1 << j), damage)
                        val += self.value(defenders[j]) - self.value(attackers[i][1])
                    else:
                        continue
                    if val > best[0]:
                        best = (val, ((i,j),) + path)
            return best
        score, path = solve((1 << n)-1, (1 << m)-1, 0)
        if not path:
            return {}
        i,j = path[0]
        return {'action': attackers[i][0], 'target': ident(defenders[j]) if j >= 0 else None, 'value': score}

    def rank_choices(self):
        choices = self.p.get('choices', [])
        kind = self.p.get('kind', '')
        battle_plan = self.battle_plan(choices) if 'battle' in kind else {}
        if battle_plan:
            self.planned_target = battle_plan['target']
        ranked = []
        ownmax = self.strongest(self.mine)
        enemax = self.strongest(self.enemy)
        for x in choices:
            label = str(x.get('label', '')).lower()
            c = self.full(x.get('card', {}))
            name = c.get('name', '')
            atk, dfn = number(c.get('attack')), number(c.get('defense'))
            score = 0.
            if label == 'yes':
                score = 12000 if kind == 'activate_effect' else 500
                if name == 'Penguin Soldier' and not self.enemy and not any(z.get('name') in FLIPS for z in self.mine):
                    score = -1000
            elif 'shuffle' in label:
                score = -100000
            elif 'decline' in label or label in ('no', 'cancel', 'finish'):
                score = 0
            elif 'activation' in label or label == 'activate' or 'activate effect' in label:
                score = self.activation_score(c)
            elif 'normal summon' in label or 'tribute summon' in label:
                level = number(c.get('level'))
                score = 4500 + atk
                if atk < 1500:
                    score = 600 + atk
                if name in FLIPS:
                    score = -200
                if level >= 5:
                    need = 1 if level < 7 else 2
                    bodies = sorted(self.mine, key=self.value)[:need]
                    lostatk = sum(number(z.get('attack')) for z in bodies if attack(z))
                    score -= sum(self.value(z) for z in bodies) * .8
                    if atk > enemax >= ownmax:
                        score += 2500
                    elif lostatk >= atk:
                        score -= 5500
                    if ownmax >= atk:
                        score -= 6500
                if enemax > atk and not any(self.activation_score(z) > 6000 for z in self.me.get('hand',[]) if not number(z.get('type'))&1):
                    score -= 4500
            elif 'set monster' in label:
                score = 2600 + dfn
                if atk >= 1700 and enemax <= atk:
                    score -= 3000
                if name in ('Man-Eater Bug', 'Penguin Soldier'):
                    score = 5600 + (1300 if enemax >= ownmax and self.enemy else 0)
                elif name == 'Magician of Faith':
                    score = 5400 if any(number(z.get('type'))&2 for z in self.me.get('graveyard', [])) else 1200
                if len(self.mine)>=3:
                    score -= 1800
            elif 'set spell' in label or 'set trap' in label:
                score = 1200 if name in INTERACTION else -800
                if self.obs.get('phase') == 'Main 1' and name in ('Book of Moon','Rush Recklessly','Mystical Space Typhoon'):
                    score = 250
                if len(self.me.get('spells_traps', [])) >= 4:
                    score -= 2000
            elif 'flip summon' in label:
                score = 6000 if name in FLIPS else 2500 + atk if atk >= enemax else -1000
                if name == 'Man-Eater Bug' and not self.enemy:
                    score = -3000
                if name == 'Magician of Faith' and not any(number(z.get('type'))&2 for z in self.me.get('graveyard', [])):
                    score = -1000
            elif 'position' in label and c:
                if attack(c):
                    score = 3000 if atk < enemax and dfn >= atk else -3000
                else:
                    score = 3500 if atk >= enemax and atk > 0 else -2000
            elif 'attack' in label and c and 'battle' in kind:
                score = 10000 + battle_plan.get('value', 0) if battle_plan.get('action') == x['action'] else -1000
            elif 'battle phase' in label:
                score = 500 if any(attack(z) for z in self.mine) else -500
            elif 'main phase 2' in label or 'main 2' in label:
                score = 100
            elif 'end turn' in label or 'end phase' in label:
                score = -100
            elif kind in ('select_position', 'position') or ('face-up' in label or 'face-down' in label):
                summoned_atk = number(c.get('attack', self.last_card.get('attack')))
                score = 100 if ('attack' in label and summoned_atk >= 1500) or ('defense' in label and summoned_atk < 1500) else 0
            elif x.get('cards') is not None:
                score = sum(self.target_score(z) for z in x['cards'])
            elif c:
                score = self.target_score(c)
                if kind not in ('main_command','battle_command','chain_response') and ident(c) == getattr(self, 'planned_target', None):
                    score += 20000
            elif label in ('activate', 'apply effect'):
                score = 1000
            elif 'direct' in label:
                score = 10000
            # Model's target names are only preferences; never stale action tokens.
            if c and name in self.target_names and kind not in ('main_command','battle_command','chain_response'):
                score += 3000
            if x.get('action') in self.failures.get(self.p.get('token'), set()):
                score -= 1000000
            ranked.append((score, x))
        return sorted(ranked, key=lambda z:z[0], reverse=True)

    def candidate_cards(self):
        return [(number(x.get('index'), i), x.get('card', x)) for i,x in enumerate(self.p.get('candidates', []))]

    def bounds(self):
        p = self.p
        con = p.get('constraint', p.get('constraints', {}))
        if not isinstance(con, dict):
            con = {}
        return number(p.get('min', p.get('minimum', con.get('min', con.get('minimum', 1))))), number(p.get('max', p.get('maximum', con.get('max', con.get('maximum', 1)))))

    def subset_valid(self, ids, order=False):
        candidates = self.candidate_cards()
        allowed = {i for i,_ in candidates}
        if len(ids) != len(set(ids)) or not set(ids) <= allowed:
            return False
        if order:
            return set(ids) == allowed
        low, high = self.bounds()
        if 'tribute' in self.p.get('kind','').lower():
            supplied_weights = self.p.get('weights', [])
            weights = {number(c.get('index'),i):number(supplied_weights[i] if i < len(supplied_weights) else c.get('weight',1),1) for i,c in enumerate(self.p.get('candidates',[]))}
            return len(ids) <= high and sum(weights.get(i,1) for i in ids) >= low
        return low <= len(ids) <= high

    def subset_action(self):
        p = self.p
        token = p.get('token', '')
        cards = self.candidate_cards()
        if p.get('mode') == 'order' or 'order' in p.get('kind','') or 'sort' in p.get('kind',''):
            ids = [i for i,c in sorted(cards, key=lambda x:self.target_score(x[1]))]
            return 'order '+token+' '+(','.join(map(str,ids)) or '-')
        low, high = self.bounds()
        tribute = 'tribute' in p.get('kind','')
        ranks = {i: -self.value(c) if tribute else self.target_score(c) for i,c in cards}
        available = [i for i,c in cards]
        best = None
        # Pool selections are small. Bound enumeration on unusually large discards.
        if len(available) <= 18:
            for n in range(0 if tribute else max(0,low), min(high,len(available))+1):
                for ids in itertools.combinations(available,n):
                    if not self.subset_valid(ids):
                        continue
                    value = sum(ranks[i] for i in ids)
                    if tribute:
                        value -= len(ids)*500
                    if best is None or value > best[0]:
                        best = (value,ids)
        else:
            ids = sorted(available,key=lambda i:ranks[i],reverse=True)[:max(low, min(high,sum(v>0 for v in ranks.values())))]
            if self.subset_valid(ids):
                best = (0,ids)
        ids = list(best[1]) if best else available[:max(0,low)]
        return 'select '+token+' '+(','.join(map(str,ids)) or '-')

    def fallback_action(self, ranked):
        # choices on a subset prompt are supplementary cancel/finish actions,
        # NOT its candidate list. Never cancel an attack/summon merely because
        # that is the only enumerated choose action.
        if self.p.get('mode') in ('subset','order') or (not self.p.get('choices') and self.p.get('candidates')):
            return self.subset_action()
        if not ranked:
            return 'noop'
        if self.p.get('kind') not in ('main_command','battle_command'):
            return ranked[0][1]['action']
        # Tokens change even when selecting then cancelling the same attack.
        # Track the public board and semantic choice instead of token equality.
        signature = json.dumps([self.obs.get('phase'),self.p.get('kind'),self.obs.get('players'),self.obs.get('chain')],sort_keys=True,separators=(',',':'))
        for score, choice in ranked:
            semantic = json.dumps([choice.get('label'),choice.get('card')],sort_keys=True)
            key = (signature,semantic)
            if self.repeated_states.get(key,0)<3:
                self.repeated_states[key] = self.repeated_states.get(key,0)+1
                return choice['action']
        return ranked[0][1]['action']

    def parse(self, text):
        target_names = []
        proposed_plan = self.plan_note
        proposed_reserve = None
        try:
            obj = json.loads(text[text.find('{'):text.rfind('}')+1])
            if isinstance(obj,dict):
                text = str(obj.get('action',''))
                target_names = obj.get('targets',[])
                proposed_plan = str(obj.get('plan',''))[:600]
                reserved = obj.get('reserve', [])
                if isinstance(reserved,list):
                    proposed_reserve = {v for v in reserved if isinstance(v,str)}
        except (ValueError,TypeError):
            pass
        for m in re.finditer(r'\b(choose|select|order)\s+(\S+)\s+(-|\d+(?:\s*,\s*\d+)*)',str(text)):
            verb, token, args = m.groups()
            if token != self.p.get('token'):
                continue
            action = f'{verb} {token} '+re.sub(r'\s','',args)
            if verb == 'choose':
                if action not in {x.get('action') for x in self.p.get('choices',[])}:
                    continue
            else:
                order_prompt = self.p.get('mode') == 'order' or 'order' in self.p.get('kind','') or 'sort' in self.p.get('kind','')
                if (verb == 'order') != order_prompt:
                    continue
                if verb == 'select' and self.p.get('mode') != 'subset' and not self.p.get('candidates'):
                    continue
                ids = [] if args == '-' else [int(v.strip()) for v in args.split(',')]
                if not self.subset_valid(ids,order=verb=='order'):
                    continue
            self.target_names = target_names if isinstance(target_names,list) else []
            self.plan_note = proposed_plan
            if proposed_reserve is not None:
                self.reserve = proposed_reserve
                self.reserve_turn = self.obs.get('turn')
            return action
        return None

    def should_ask(self, ranked):
        budget = self.ctx.budget.remaining()
        if self.pending is not None:
            if not self.pending.done():
                return False
            # A late answer refers to an expired prompt and must not be executed.
            try:
                self.pending.result()
            except Exception:
                pass
            self.pending = None
        if number(budget.get('system2_calls')) <= 1 or number(budget.get('wall_s')) < 120:
            return False
        if self.used_calls >= 18 or self.duel_calls >= 6 or self.obs.get('last_action_error'):
            return False
        kind = self.p.get('kind','')
        turn = self.obs.get('turn')
        key = (turn,kind)
        if key in self.asked:
            return False
        if kind == 'main_command':
            meaningful = [(s,c) for s,c in ranked if c.get('card') and c.get('label') not in ('set spell/trap','shuffle own hand')]
            uncertain = len(meaningful)>1 and (self.enemy or any(number(self.full(c.get('card',{})).get('level'))>=5 for s,c in meaningful))
            return uncertain
        if kind == 'chain_response':
            useful = [(s,c) for s,c in ranked if c.get('card') and s>0]
            protective = any(self.full(c.get('card',{})).get('name') in ('Book of Moon','Compulsory Evacuation Device') for s,c in ranked)
            return bool(self.obs.get('chain')) and len(ranked)>1 and ranked[0][0]<10000 and (len(useful)>1 or protective)
        return not ranked and bool(self.p.get('candidates')) and len(self.p.get('candidates',[]))>6 and bool(self.last_effect)

    def ask(self, ranked, fallback):
        self.asked.add((self.obs.get('turn'),self.p.get('kind')))
        self.used_calls += 1
        self.duel_calls += 1
        state = {k:v for k,v in self.obs.items() if k not in ('card_catalog','recent_events','schema_version')}
        state['recent_events'] = self.obs.get('recent_events',[])[-12:]
        relevant = set()
        def visit(x):
            if isinstance(x,dict):
                if x.get('code'): relevant.add(str(x['code']))
                for v in x.values(): visit(v)
            elif isinstance(x,list):
                for v in x: visit(v)
        visit(state)
        state['effect_texts'] = {v.get('name'):v.get('text') for k,v in self.catalog.items() if k in relevant and number(v.get('type')) != 17}
        advice = [{'action':c['action'],'score':round(s)} for s,c in ranked[:8]]
        prompt = (
            'You are a Yu-Gi-Oh tactical adviser in a real native-engine duel. Only current legal actions are allowed. '
            'Win the duel; life-point margin is secondary. Unknown cards have no known identity. '
            'Choose one action; code will execute subsequent routine decisions and recalculate battles. '
            'Return JSON {"action":"exact current action", "targets":["preferred target card names"], "reserve":["optional spells/traps to NOT activate during this turn main phases"], "plan":"one short sentence"}. '
            'For subset prompts use select TOKEN I,J (distinct candidate indices obey constraints); order all indices for order prompts. '
            'Prioritize useful card advantage and pressure. Summon strong attackers, Set useful flip monsters, avoid needless tributes. '
            'Fissure kills the lowest face-up ATK; Dark Hole kills both fields. MST destroys but does NOT negate normal effects. '
            'Mirror Force destroys attack-position monsters; Book of Moon can stop attacks or hide your target. '
            'Do not activate an optional card just because legal. Do not shuffle hand. Avoid setting useful quickplay spells before attacking. '
            'Battle attacks vs defense do not deal damage unless a card explicitly pierces. '
            'Controller suggestion is fallible. Correct it when a better exchange or immediate lethal exists.\n'
            +json.dumps({'previous_plan':self.plan_note,'pending_effect':self.last_effect,'controller_action':fallback,'ranking':advice,'observation':state},separators=(',',':')))
        try:
            timeout = min(65.,max(5.,float(self.ctx.budget.remaining().get('wall_s',100))-15))
            self.pending = self.ctx.spawn(lambda: self.ctx.think(prompt, timeout_s=timeout))
            # Backend timeout does not necessarily bound admission/queue latency.
            # Only the ctx-owned inference call runs in the background; late output
            # is discarded, and the evaluator cancels inference at episode close.
            result = self.pending.result(timeout=70.)
            self.pending = None
            return self.parse(result.text) or fallback
        except FutureTimeout:
            self.ctx.log('adviser_local_timeout', token=self.p.get('token'))
            return fallback
        except Exception as e:
            self.ctx.log('adviser_failure',error=type(e).__name__)
            return fallback

    def remember(self, action):
        choice = next((c for c in self.p.get('choices',[]) if c.get('action')==action),{})
        label = choice.get('label','').lower()
        card = self.full(choice.get('card',{}))
        if 'activat' in label or (label == 'yes' and self.p.get('kind') == 'activate_effect'):
            self.last_effect = card.get('name', self.last_effect)
            self.last_card = card
        elif 'summon' in label or 'set monster' in label:
            self.last_card = card
        elif 'attack' in label and card:
            self.attack_card = card
        # Chain links give authoritative effect identity for mandatory target prompts.

    def run_episode(self, game):
        obs = unpack(self.ctx.observe())
        same = 0
        for _ in range(16000):
            if self.ctx.done or obs.get('done'):
                return
            self.setup(obs)
            token = self.p.get('token')
            if not token:
                return
            if token == self.last_token:
                same += 1
            else:
                same = 0
            self.last_token = token
            ranked = self.rank_choices()
            fallback = self.fallback_action(ranked)
            action = fallback
            if same == 0 and self.should_ask(ranked):
                action = self.ask(ranked, fallback)
            self.remember(action)
            self.ctx.log('tactical_decision',token=token,kind=self.p.get('kind'),action=action,fallback=fallback,effect=self.last_effect)
            try:
                result = self.ctx.act(action)
                obs = unpack(result if result is not None else self.ctx.observe())
            except Exception:
                if self.ctx.done:
                    return
                self.failures.setdefault(token,set()).add(action)
                obs = unpack(self.ctx.observe())
                continue
            if obs.get('last_action_error'):
                self.failures.setdefault(token,set()).add(action)
                # Never spend the response deadline repeatedly asking the model.
                if same > 10:
                    return


def build(ctx):
    return Harness(ctx)
