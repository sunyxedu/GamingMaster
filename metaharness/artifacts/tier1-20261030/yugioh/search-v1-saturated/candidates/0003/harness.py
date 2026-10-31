"""Semantic turn agendas, checked against each fresh native prompt."""
from __future__ import annotations
import itertools
import json
import re
from functools import lru_cache

FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
QUICK = {'Book of Moon', 'Rush Recklessly', 'Mystical Space Typhoon'}
TRAPS = {'Trap Hole', 'Mirror Force', 'Compulsory Evacuation Device', 'Call of the Haunted'}


def number(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def ident(c):
    return (c.get('player'), c.get('zone'), c.get('slot'))


def attack_pos(c):
    return 'attack' in c.get('position', '')


def facedown(c):
    return 'face-down' in c.get('position', '')


def unpack(obs):
    if isinstance(obs, str):
        try:
            return json.loads(obs)
        except (ValueError, TypeError):
            return {}
    return obs if isinstance(obs, dict) else {}


class Board:
    def __init__(self, obs):
        self.o = obs
        self.p = obs.get('prompt') or {}
        self.seat = obs.get('agent_seat', 0)
        self.players = obs.get('players') or []
        self.me = next((p for p in self.players if p.get('player') == self.seat), {})
        self.opp = next((p for p in self.players if p.get('player') != self.seat), {})
        self.mine = self.me.get('monsters', [])
        self.enemy = self.opp.get('monsters', [])
        self.hand = self.me.get('hand', [])
        self.ownturn = obs.get('turn_player') == self.seat
        self.catalog = obs.get('card_catalog') or {}
        self.cards = {}
        for p in self.players:
            for z in ('hand', 'monsters', 'spells_traps', 'graveyard', 'banished'):
                for c in p.get(z, []):
                    self.cards[ident(c)] = c
        self.chain = obs.get('chain') or []

    def full(self, c):
        if not isinstance(c, dict):
            return {}
        x = dict(self.catalog.get(str(c.get('code')), {}))
        x.update(c)
        current = self.cards.get(ident(c))
        if current and (not c.get('code') or not current.get('code') or current.get('code') == c.get('code')):
            x.update(current)
        return x

    def value(self, c):
        c = self.full(c)
        if c.get('zone') == 'spells_traps':
            return 1250 if facedown(c) else 900
        v = 900 + number(c.get('attack')) * .55
        if c.get('name') in FLIPS:
            v = 1400 if facedown(c) else 700
        if c.get('name') == 'unknown card':
            v = 1500
        return v

    def attack_event(self):
        # Events survive many prompts; only use attacks in the current Battle phase.
        if 'battle' not in str(self.o.get('phase', '')).lower() and 'damage' not in str(self.o.get('phase', '')).lower():
            return None
        for e in reversed(self.o.get('recent_events', [])):
            if e.get('event') in ('new_turn', 'phase'):
                break
            if e.get('event') == 'attack':
                return e
        return None

    def battle_gain(self, a, d):
        a = self.full(a)
        av = number(a.get('attack'))
        if d is None:
            return av * .7 + (50000 if av >= number(self.opp.get('lp'), 8000) else 0)
        d = self.full(d)
        if d.get('name') == 'unknown card':
            # Unknown defenses are not assigned a fake known identity.
            return 450 if av >= 2000 else 160 if av >= 1700 else -500
        dv = number(d.get('attack' if attack_pos(d) else 'defense'))
        if av > dv:
            return self.value(d) + (.7 * (av-dv) if attack_pos(d) else 0)
        if av == dv:
            return self.value(d)-self.value(a) if attack_pos(d) else -10
        return -(self.value(a) if attack_pos(d) else 0) - .7*(dv-av)

    def combat_plan(self, attackers):
        """Exact assignment search on revealed static boards; unknown blocker heuristic.
        Recomputed after every attack/effect, never treated as a native simulator.
        """
        aa = [self.full(a) for a in attackers]
        dd = [self.full(d) for d in self.enemy]
        if not aa:
            return (0, None, None)
        @lru_cache(None)
        def solve(am, dm, damage):
            best = (50000 if damage >= number(self.opp.get('lp'), 8000) else 0, -1, -1)
            for i, a in enumerate(aa):
                if not am & (1 << i):
                    continue
                if not dm:
                    gain = number(a.get('attack')) * .7
                    future = solve(am ^ (1 << i), dm, damage + number(a.get('attack')))[0]
                    if gain+future > best[0]:
                        best = (gain+future, i, -1)
                    continue
                for j, d in enumerate(dd):
                    if not dm & (1 << j):
                        continue
                    gain = self.battle_gain(a, d)
                    av = number(a.get('attack'))
                    dv = number(d.get('attack' if attack_pos(d) else 'defense'))
                    known = d.get('name') != 'unknown card'
                    removed = (av > dv or (av == dv and attack_pos(d))) if known else av >= 2000
                    ndm = dm ^ (1 << j) if removed else dm
                    dmg = max(0, av-dv) if known and attack_pos(d) else 0
                    # No continuation past an unknown Flip effect: value is speculative.
                    future = solve(am ^ (1 << i), ndm, damage+dmg)[0]
                    score = gain + future * (1 if known else .65)
                    if score > best[0]:
                        best = (score, i, j)
            return best
        score, ai, di = solve((1 << len(aa))-1, (1 << len(dd))-1, 0)
        return score, (aa[ai] if ai >= 0 else None), (dd[di] if di >= 0 else None)


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.agenda = []
        self.plan_turn = None
        self.plan_round = None
        self.calls = {}
        self.asked = set()
        self.effect = ''
        self.effect_target = None
        self.attacker = None
        self.battle_target = None
        self.rejected = {}
        self.last_action = None
        self.last_token = None
        self.goal = ''
        self.plan_log = []
        self.future = None

    def effect_score(self, b, name):
        enemies = b.enemy
        ours = b.mine
        strongest = max([number(c.get('attack')) for c in ours] or [0])
        ev = b.attack_event()
        attacker = b.full(ev.get('attacker', {})) if ev else {}
        target = b.full(ev.get('target', {})) if ev else {}
        enemy_attack = ev and attacker.get('player') != b.seat
        own_attack = ev and attacker.get('player') == b.seat
        chain_names = [x.get('card', {}).get('name') for x in b.chain]
        if name in chain_names:
            return -100
        if name == 'Mirror Force':
            return 120 + 20*sum(attack_pos(c) for c in enemies) if enemies else -100
        if name == 'Trap Hole':
            return 110 if enemies else -100
        if name == 'Monster Reborn' or name == 'Call of the Haunted':
            pool = b.me.get('graveyard', []) + (b.opp.get('graveyard', []) if name == 'Monster Reborn' else [])
            best = max([number(c.get('attack')) for c in pool if number(c.get('type')) & 1] or [0])
            if best >= 1500 and len(ours) < 5:
                return 85 + best/100 if b.ownturn or enemy_attack else 35
            return 10 if not ours and best else -100
        if name == 'Fissure':
            face = [c for c in enemies if not facedown(c)]
            if not face:
                return -100
            victim = min(face, key=lambda c: number(c.get('attack')))
            return 100 if number(victim.get('attack')) >= strongest or not ours else 55
        if name == 'Dark Hole':
            delta = sum(b.value(c) for c in enemies)-sum(b.value(c) for c in ours)
            return 110 + delta/100 if enemies and delta > 500 else -100
        if name == 'Mystical Space Typhoon':
            back = b.opp.get('spells_traps', [])
            if not back:
                return -100
            if any(x.get('card', {}).get('name') == 'Call of the Haunted' and x.get('card', {}).get('player') != b.seat for x in b.chain):
                return 140
            if b.chain:
                return -100  # destruction is not negation
            return 105 if b.ownturn and ours else 25 if not b.ownturn and 'End' in b.o.get('phase', '') else -20
        if name == 'Compulsory Evacuation Device':
            if not enemies:
                return -100
            if enemy_attack:
                return 105 if not target or b.battle_gain(attacker, target) >= 0 else 45
            if b.ownturn and ours and not b.chain:
                return 75 if len(enemies) <= 2 else 40
            return -50
        if name == 'Book of Moon':
            if 'Mirror Force' in chain_names and any(attack_pos(c) for c in ours):
                return 125
            if enemy_attack:
                return 95 if not target.get('code') or number(attacker.get('attack')) >= number(target.get('attack' if attack_pos(target) else 'defense')) else -40
            if b.ownturn and ours and not b.chain:
                good = [c for c in enemies if attack_pos(c) and number(c.get('attack')) >= strongest > number(c.get('defense'))]
                return 90 if good else -40
            return -60
        if name == 'Rush Recklessly':
            if ev:
                own = attacker if own_attack else target
                foe = target if own_attack else attacker
                av = number(own.get('attack'))
                dv = number(foe.get('attack' if (enemy_attack or attack_pos(foe)) else 'defense'))
                if own.get('player') == b.seat and attack_pos(own) and av <= dv < av+700:
                    return 115
                if own_attack and not enemies and av < number(b.opp.get('lp'), 8000) <= av+700:
                    return 150
            return -80
        if name == 'Man-Eater Bug':
            return 90 if enemies else -50
        if name == 'Penguin Soldier':
            return 90 if enemies else -50
        if name == 'Magician of Faith':
            return 85 if any(number(c.get('type')) & 2 for c in b.me.get('graveyard', [])) else -50
        return -40

    def choice_score(self, b, q):
        label = q.get('label', '').lower()
        c = b.full(q.get('card', {}))
        name = c.get('name', '')
        atk, defense = number(c.get('attack')), number(c.get('defense'))
        enemy_max = max([number(x.get('attack')) for x in b.enemy if attack_pos(x)] or [0])
        if 'shuffle' in label:
            return -10000
        if 'decline' in label or label == 'no' or 'cancel' in label:
            return 0
        if label == 'yes':
            if name:
                return self.effect_score(b, name)
            return 10
        if 'activate' in label or 'activation' in label:
            return self.effect_score(b, name)
        if label == 'normal summon':
            level = number(c.get('level'))
            tribute = 2 if level >= 7 else 1 if level >= 5 else 0
            if tribute:
                sacrificed = sorted(b.mine, key=b.value)[:tribute]
                cost = sum(number(x.get('attack')) for x in sacrificed)
                if len(sacrificed) < tribute:
                    return -100
                if cost >= atk and enemy_max < max([number(x.get('attack')) for x in b.mine] or [0]):
                    return -30
                return 75 + (atk-cost)/100 + (25 if atk > enemy_max >= max([number(x.get('attack')) for x in b.mine] or [0]) else 0)
            if name in FLIPS:
                return 10 if not b.enemy else -20
            return 55 + atk/100 + (10 if atk > enemy_max else -30)
        if label == 'set monster':
            if name == 'Man-Eater Bug':
                return 94 if b.enemy else 60
            if name == 'Penguin Soldier':
                return 96 if len(b.enemy) >= 2 else 80 if b.enemy else 58
            if name == 'Magician of Faith':
                return 75 if any(number(x.get('type')) & 2 for x in b.me.get('graveyard', [])) else 30
            return 45 + defense/100 + (20 if enemy_max >= atk and defense >= enemy_max else 0) - max(0, atk-defense)/50
        if label == 'set spell/trap':
            if len(b.me.get('spells_traps', [])) >= 4:
                return -20
            if name in TRAPS:
                return 45
            if name in QUICK:
                return 12 if b.o.get('phase') == 'Main 2' or b.o.get('turn') == 1 or not any(attack_pos(x) for x in b.mine) else -5
            return 5 if b.me.get('hand_count', 0) > 6 else -30
        if 'flip summon' in label and facedown(c):
            return self.effect_score(b, name) if name in FLIPS else 50 if atk >= enemy_max or not b.enemy else -30
        if 'position' in label:
            if attack_pos(c):
                return 30 if atk < enemy_max and defense > atk else -30
            return 60 if atk >= enemy_max and atk >= 1000 else -25
        if label == 'attack':
            if self.battle_attacker and ident(c) == ident(self.battle_attacker):
                return 100 + min(self.battle_score/100, 1000)
            return -30
        if 'enter battle' in label:
            return 8 if any(attack_pos(x) for x in b.mine) else -5
        if 'enter main phase 2' in label:
            return 2
        if 'end turn' in label or 'finish' in label:
            return 1
        if 'place in' in label:
            return 10
        if 'face-up attack' in label:
            return 20
        if 'face-up defense' in label:
            return 10
        return 0

    def target_score(self, b, candidate):
        c = b.full(candidate)
        hint = str(b.p.get('hint', '')).lower()
        own = c.get('player') == b.seat
        name = c.get('name')
        if 'attack target' in hint:
            if self.battle_target and ident(c) == ident(self.battle_target):
                return 50000
            return b.battle_gain(self.attacker or {}, c)
        if b.p.get('kind') == 'select_tribute' or 'tribute' in hint or 'discard' in hint:
            return -b.value(c)
        if 'special summon' in hint or self.effect in ('Monster Reborn', 'Call of the Haunted') or 'graveyard' in hint or 'add to' in hint:
            if number(c.get('type')) & 1:
                return number(c.get('attack')) + 500
            weights = {'Monster Reborn': 3100, 'Dark Hole': 1800 if len(b.enemy)>len(b.mine) else 500, 'Fissure': 2100 if b.enemy else 900, 'Book of Moon': 1300, 'Mystical Space Typhoon': 1700 if b.opp.get('spells_traps') else 600, 'Rush Recklessly': 1200}
            return weights.get(name, 500)
        if self.effect == 'Rush Recklessly':
            ev = b.attack_event() or {}
            battling = ev.get('attacker') if b.ownturn else ev.get('target')
            return (10000 if battling and ident(c) == ident(battling) else number(c.get('attack'))) if own else -10000
        if self.effect == 'Mystical Space Typhoon':
            for link in b.chain:
                source = link.get('card', {})
                if source.get('name') == 'Call of the Haunted' and source.get('player') != b.seat and ident(source) == ident(c):
                    return 10000
            return b.value(c) if not own else -10000
        if self.effect == 'Book of Moon':
            if any(x.get('card', {}).get('name') == 'Mirror Force' for x in b.chain):
                return b.value(c) if own and attack_pos(c) else -10000
            ev = b.attack_event()
            if ev and not b.ownturn and ident(c) == ident(ev.get('attacker', {})):
                return 10000
            return (number(c.get('attack'))-number(c.get('defense'))+1500) if not own else -5000
        if self.effect_target and name == self.effect_target:
            return 20000 if not own else -2000
        if c.get('zone') == 'graveyard' and own:
            return number(c.get('attack')) + 500
        if not own:
            return b.value(c) + (600 if number(c.get('level')) >= 5 else 0)
        # Mandatory Bug with no opposing target: sacrifice the least valuable body.
        return -b.value(c)

    def subset(self, b):
        p = b.p
        cc = p.get('candidates') or []
        indices = [number(c.get('index'), i) for i,c in enumerate(cc)]
        if p.get('mode') == 'order' or 'order' in p.get('kind', '') or 'sort' in p.get('kind', ''):
            return 'order '+p['token']+' '+','.join(map(str, indices))
        minimum = number(p.get('minimum'))
        maximum = min(len(cc), number(p.get('maximum'), len(cc)))
        weighted = 'weight' in p.get('constraint', '').lower() or 'tribute' in p.get('kind', '')
        weights = p.get('weights') or [1]*len(cc)
        scores = [self.target_score(b,c) for c in cc]
        best = None
        # Pool selections are small. Enumerate all valid subsets, exact weights.
        for n in range(0 if weighted else minimum, maximum+1):
            for combo in itertools.combinations(range(len(cc)), n):
                if weighted and sum(number(weights[i], 1) for i in combo) < minimum:
                    continue
                score = sum(scores[i] for i in combo)
                if best is None or score > best[0]:
                    best = (score, combo)
        if best is None:
            return p.get('choices', [{}])[0].get('action', 'noop') if p.get('choices') else 'noop'
        use = [indices[i] for i in best[1]]
        return 'select '+p['token']+' '+(','.join(map(str,use)) if use else '-')

    def ask_plan(self, b):
        ctx = self.ctx
        if self.future is not None and not self.future.done():
            return
        key = (b.o.get('round'), b.o.get('turn'))
        r = b.o.get('round')
        rem = ctx.budget.remaining()
        if key in self.asked or self.calls.get(r, 0) >= 4 or rem.get('system2_calls', 0) <= 2 or rem.get('wall_s', 0) < 210:
            return
        self.asked.add(key)
        self.calls[r] = self.calls.get(r,0)+1
        # Only current public and owned information is sent. No accumulated session.
        state = {k:b.o.get(k) for k in ('round','your_deck','agent_seat','turn_player','turn','phase','players','chain')}
        state['players'] = [{k:v for k,v in p.items() if k not in ('banished','extra_count')} for p in b.players]
        relevant = {c.get('name') for p in b.players for z in ('hand','monsters','spells_traps','graveyard') for c in p.get(z,[])}
        texts = {c['name']:c.get('text') for c in b.catalog.values() if c.get('name') in relevant and number(c.get('type')) != 17}
        prompt = '''Plan this Yu-Gi-Oh turn against a random-legal opponent. Return ONE compact JSON object, no prose. The executor will use the plan across several decisions, resolving current legal actions by exact label and card name. Only plan actions you intend to spend cards on. Unknown opponent cards stay unknown.
Schema: {"goal":"short reason", "steps":[{"label":"normal summon", "card":"exact name", "target":"optional enemy card name"}, ...]}. Labels: normal summon, set monster, set spell/trap, flip summon, change position, activate, enter Battle Phase, enter Main Phase 2, end turn. Use card "" for phase transitions. At most 10 steps. All future steps are conditional on still being legal; a scripted tactical executor handles chain responses, targets, tributes and attacks. It recalculates battle assignment after each observation. Put enter Battle Phase before setting quick-play spells you want usable this turn. Do not end turn before useful attacks. Do not tribute multiple productive attackers merely for more ATK. Destroying an activated source does not negate. Reborn can use either graveyard. Man-Eater Bug must destroy even a friendly monster if no enemy exists. Prioritize winning, pressure and profitable exchanges over LP hoarding. You may specify a whole turn including actions not in the current prompt, but never hidden identities. Keep answer under 350 tokens.
'''
        prompt += '\nCURRENT LEGAL PROMPT '+json.dumps(b.p,separators=(',',':'))
        prompt += '\nSTATE '+json.dumps(state,separators=(',',':'))
        prompt += '\nTEXT '+json.dumps(texts,separators=(',',':'))
        prompt += '\nRECENT EVENTS '+json.dumps(b.o.get('recent_events',[])[-10:],separators=(',',':'))
        prompt += '\nPREVIOUS GOAL '+self.goal
        try:
            # Backend admission queueing can precede the model timeout.
            # Bound total pending-decision wait, including queueing time.
            self.future = ctx.spawn(lambda: ctx.think(prompt, timeout_s=min(110, rem['wall_s']-75)))
            result = self.future.result(timeout=120)
            text = result.text
            start, end = text.find('{'), text.rfind('}')
            obj = json.loads(text[start:end+1])
            steps = obj.get('steps', [])
            if not isinstance(steps,list):
                return
            self.agenda = [dict(s) for s in steps[:10] if isinstance(s,dict) and isinstance(s.get('label'),str) and isinstance(s.get('card',''),str)]
            self.goal = str(obj.get('goal',''))[:300]
            ctx.log('semantic_plan', round=r, turn=b.o.get('turn'), goal=self.goal, steps=self.agenda)
        except Exception as exc:
            ctx.log('plan_fallback', error=type(exc).__name__)

    def planned_choice(self, b, choices):
        if b.p.get('kind') not in ('main_command','battle_command'):
            return None
        # Battle is delegated until all profitable attacks have been executed.
        if b.p.get('kind') == 'battle_command' and self.battle_attacker is not None:
            return None
        while self.agenda:
            step = self.agenda[0]
            wanted = step['label'].lower()
            name = step.get('card','')
            def matches(q):
                actual = q.get('label', '').lower()
                if actual == 'change position / flip summon':
                    actual = 'flip summon' if facedown(b.full(q.get('card', {}))) else 'change position'
                return actual == wanted and (not name or q.get('card', {}).get('name') == name)
            matched = [q for q in choices if matches(q)]
            if matched:
                q = matched[0]
                # The semantic plan may spend removal more aggressively, but cannot
                # override known catastrophic arithmetic or empty-target safeguards.
                if self.choice_score(b,q) <= -90:
                    self.agenda.pop(0)
                    continue
                self.agenda.pop(0)
                self.effect_target = step.get('target')
                self.ctx.log('agenda_commit', step=step, action=q['action'])
                return q
            # Don't skip a Main2 set while still battling: let combat finish first.
            if b.p.get('kind') == 'battle_command' and wanted not in ('activate','enter main phase 2','end turn'):
                return None
            self.ctx.log('agenda_invalidated', step=step, kind=b.p.get('kind'))
            self.agenda.pop(0)
        return None

    def choose(self, b):
        p = b.p
        choices = p.get('choices') or []
        if b.chain:
            self.effect = b.chain[-1].get('card', {}).get('name', self.effect)
        elif p.get('kind') in ('main_command', 'battle_command'):
            self.effect = ''
            self.effect_target = None
        if p.get('candidates') is not None and p.get('mode') in ('subset','order'):
            return self.subset(b)
        if not choices:
            return 'noop'
        attackers = [q.get('card',{}) for q in choices if q.get('label') == 'attack']
        self.battle_score,self.battle_attacker,self.battle_target = b.combat_plan(attackers)
        q = self.planned_choice(b, choices)
        rejected = self.rejected.get(p.get('token'),set())
        if q is None or q['action'] in rejected:
            available = [x for x in choices if x['action'] not in rejected] or choices
            q = max(available, key=lambda x:self.choice_score(b,x))
        if q.get('label') == 'attack':
            self.attacker = b.full(q.get('card',{}))
        elif q.get('label') in ('activate','chain activation','yes') and q.get('card'):
            self.effect = q['card'].get('name','')
        return q['action']

    def run_episode(self, game):
        ctx = self.ctx
        obs = unpack(ctx.observe())
        while not ctx.done:
            if not obs:
                obs = unpack(ctx.observe())
            b = Board(obs)
            if not b.p:
                if obs.get('done') or ctx.done:
                    return
                obs = unpack(ctx.act('noop'))
                continue
            token = b.p.get('token')
            if obs.get('last_action_error') and token == self.last_token:
                self.rejected.setdefault(token,set()).add(self.last_action)
                self.agenda = []
                ctx.log('rejected_action', action=self.last_action, error=obs['last_action_error'])
            key = (obs.get('round'),obs.get('turn'))
            if key != self.plan_turn:
                self.agenda = []
                self.plan_turn = key
                self.effect = ''
                self.effect_target = None
            if obs.get('round') != self.plan_round:
                self.goal = ''
                self.plan_round = obs.get('round')
            if b.p.get('kind') == 'main_command' and b.ownturn and obs.get('phase') == 'Main 1' and not obs.get('last_action_error'):
                self.ask_plan(b)
                if ctx.done:
                    return
            action = self.choose(b)
            self.last_token,self.last_action = token,action
            try:
                obs = unpack(ctx.act(action))
            except Exception as exc:
                if ctx.done:
                    return
                self.rejected.setdefault(token,set()).add(action)
                ctx.log('action_recovery', error=type(exc).__name__)
                obs = unpack(ctx.observe())


def build(ctx):
    return Harness(ctx)
