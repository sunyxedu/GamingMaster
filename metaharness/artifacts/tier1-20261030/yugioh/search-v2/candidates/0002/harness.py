"""Observation-only tactical controller with sparse model adjudication.
No native state, seed, filesystem, or implementation access during play.
"""
from __future__ import annotations
import itertools
import json
import re

FLIPS = {'Man-Eater Bug', 'Penguin Soldier', 'Magician of Faith'}
RECOVERY = {'Monster Reborn', 'Call of the Haunted'}


def number(x, default=0):
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


def faceup(c):
    return 'down' not in str(c.get('position', '')).lower() and not c.get('hidden', False)


def attackpos(c):
    return 'attack' in str(c.get('position', '')).lower()


def ref(c):
    return (c.get('player'), c.get('zone'), c.get('slot'))


def json_object(text):
    for m in re.finditer(r'\{', text):
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
            if isinstance(obj, dict):
                return obj
        except (ValueError, TypeError):
            pass
    return {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.round = None
        self.calls = 0
        self.total_calls = 0
        self.last_consult = None
        self.source = ''
        self.note = ''
        self.failed = {}
        self.last_action = None
        self.last_token = None
        self.catalog = {}
        self.recent = []

    def hydrate(self, c):
        if not isinstance(c, dict):
            return {}
        out = dict(self.catalog.get(str(c.get('code')), {}))
        for p in self.obs.get('players', []):
            for zone in ('hand', 'monsters', 'spells_traps', 'graveyard', 'banished'):
                for v in p.get(zone, []):
                    if ref(v) == ref(c) and c.get('zone') is not None:
                        out.update(v)
        out.update(c)
        return out

    def value(self, c):
        c = self.hydrate(c)
        if not c.get('name'):
            return 1500
        if number(c.get('type')) & 1:
            a, d = number(c.get('attack')), number(c.get('defense'))
            return 900 + max(a, d * .70) + (500 if c.get('name') in FLIPS else 0)
        return {'Mirror Force': 3000, 'Monster Reborn': 3200, 'Dark Hole': 2800,
                'Fissure': 2300, 'Call of the Haunted': 2600}.get(c.get('name'), 2000)

    def prepare(self, obs):
        self.obs = obs
        self.catalog = obs.get('card_catalog', self.catalog)
        self.seat = obs.get('agent_seat', 0)
        ps = obs.get('players', [])
        self.me = next((p for p in ps if p.get('player') == self.seat), {})
        self.foe = next((p for p in ps if p.get('player') != self.seat), {})
        self.ours = self.me.get('monsters', [])
        self.theirs = self.foe.get('monsters', [])
        self.ownturn = obs.get('turn_player') == self.seat
        self.phase = str(obs.get('phase', ''))
        self.prompt = obs.get('prompt') or {}
        self.kind = self.prompt.get('kind', '')
        if obs.get('round') != self.round:
            self.round = obs.get('round')
            self.calls = 0
            self.last_consult = None
            self.note = ''
            self.source = ''
            self.failed = {}
            self.recent = []
        self.enemymax = max([number(c.get('attack')) for c in self.theirs if faceup(c)] or [0])
        self.ourmax = max([number(c.get('attack')) for c in self.ours if faceup(c)] or [0])

    def battle_value(self, attacker, defender=None):
        a = number(self.hydrate(attacker).get('attack'))
        if defender is None:
            return 100000 if a >= number(self.foe.get('lp'), 8000) else a * .65 + 100
        d = self.hydrate(defender)
        if not faceup(d):
            # Distribution-free risk estimate, not a claim about hidden identity.
            return (700 if a >= 2000 else 200 if a >= 1700 else -1200)
        if attackpos(d):
            b = number(d.get('attack'))
            if a > b:
                return self.value(d) + .65 * (a - b)
            if a == b:
                return self.value(d) - self.value(attacker) + 50
            return -self.value(attacker) - (b-a)
        b = number(d.get('defense'))
        return self.value(d) if a > b else -10 if a == b else -(b-a) - 150

    def current_battle(self):
        # An attack event is meaningful only in a battle phase and after phase changes.
        if 'Battle' not in self.phase and 'Damage' not in self.phase:
            return None, None
        for event in reversed(self.obs.get('recent_events', [])):
            if event.get('event') in ('new_turn', 'phase'):
                break
            if event.get('event') in ('attack', 'attack_declared', 'battle'):
                a = event.get('attacker') or event.get('card') or {}
                d = event.get('target') or event.get('defender') or {}
                if isinstance(a, dict):
                    return self.hydrate(a), self.hydrate(d) if isinstance(d, dict) else {}
        return None, None

    def activation_score(self, c):
        name = c.get('name', '')
        chain = self.obs.get('chain', [])
        battle_a, battle_d = self.current_battle()
        hostile_chain = any(x.get('player', x.get('controller')) != self.seat for x in chain)
        opponents = [x for x in self.theirs if faceup(x)]
        if name == 'Mirror Force':
            return 15000 if any(attackpos(x) for x in self.theirs) else -5000
        if name == 'Trap Hole':
            return 11000 if self.theirs else -5000
        if name in FLIPS:
            if name == 'Magician of Faith':
                return 8500 if any(number(x.get('type')) & 2 for x in self.me.get('graveyard', [])) else -1000
            return 9000 if self.theirs else -5000
        if name == 'Fissure':
            if not opponents:
                return -5000
            victim = min(opponents, key=lambda x: number(x.get('attack')))
            v = 4200 + self.value(victim)
            if number(victim.get('attack')) < self.ourmax and len(self.theirs) == 1:
                v -= 2500
            return v
        if name == 'Dark Hole':
            net = sum(self.value(x) for x in self.theirs) - sum(self.value(x) for x in self.ours)
            return 5500 + net if net >= 1800 else -4000
        if name in RECOVERY:
            grave = self.me.get('graveyard', []) + (self.foe.get('graveyard', []) if name == 'Monster Reborn' else [])
            monsters = [x for x in grave if number(x.get('type')) & 1]
            if not monsters:
                return -5000
            best = max(number(x.get('attack')) for x in monsters)
            if self.ownturn or (battle_a and battle_a.get('player') != self.seat):
                return 5000 + best if best >= 1600 else 500
            return -1000
        if name == 'Mystical Space Typhoon':
            targets = self.foe.get('spells_traps', [])
            if not targets:
                return -5000
            if any(x.get('name') == 'Call of the Haunted' for x in targets):
                return 9000
            if chain:
                return -1000  # Destroying a normal spell/trap does not negate it.
            if self.ownturn and ('Main' in self.phase or 'Battle' in self.phase):
                return 6500 if self.ours or any(number(x.get('attack')) >= 1600 for x in self.me.get('hand', [])) else 500
            return -1000
        if name == 'Book of Moon':
            if battle_a and battle_a.get('player') != self.seat:
                if not battle_d or self.battle_value(battle_a, battle_d) > 0:
                    return 7300
            if self.ownturn and not chain:
                good = [x for x in opponents if attackpos(x) and number(x.get('attack')) >= self.ourmax > number(x.get('defense'))]
                if good and self.ourmax > 0:
                    return 6900
            return -1000
        if name == 'Rush Recklessly':
            if battle_a:
                own = battle_a if battle_a.get('player') == self.seat else battle_d
                enemy = battle_d if battle_a.get('player') == self.seat else battle_a
                if own and enemy and attackpos(own):
                    ours = number(own.get('attack'))
                    theirs = number(enemy.get('attack') if attackpos(enemy) else enemy.get('defense'))
                    if ours <= theirs < ours + 700:
                        return 10000
                if battle_a.get('player') == self.seat and not battle_d and number(self.foe.get('lp'),8000) <= number(battle_a.get('attack')) + 700:
                    return 12000
            return -1200
        if name == 'Compulsory Evacuation Device':
            if battle_a and battle_a.get('player') != self.seat:
                return 8500
            tribute = [x for x in opponents if number(x.get('level')) >= 5]
            if tribute:
                return 8000
            if self.ownturn and self.ours and self.theirs and not chain:
                return 5500 if self.enemymax >= self.ourmax else 500
            return -800
        return 400 if hostile_chain else -500

    def target_score(self, raw):
        c = self.hydrate(raw.get('card', raw))
        name = self.source
        hint = str(self.prompt.get('hint', '')).lower()
        enemy = c.get('player') != self.seat
        zone = c.get('zone', '')
        if 'tribute' in self.kind or 'tribute' in hint or 'discard' in hint:
            return -self.value(c)
        if name in RECOVERY or ('grave' in zone and number(c.get('type')) & 1):
            return number(c.get('attack')) * 2 + number(c.get('defense')) * .1
        if name == 'Magician of Faith' or ('grave' in zone and number(c.get('type')) & 2):
            scores = {'Monster Reborn': 9500, 'Dark Hole': 7000 if len(self.theirs) > len(self.ours) else 2000,
                      'Fissure': 6000 if self.theirs else 3000, 'Mystical Space Typhoon': 4000,
                      'Book of Moon': 3500, 'Rush Recklessly': 3000}
            return scores.get(c.get('name'), 1000)
        if name == 'Rush Recklessly':
            a, d = self.current_battle()
            if a and a.get('player') == self.seat and ref(a) == ref(c):
                return 18000
            if d and d.get('player') == self.seat and ref(d) == ref(c):
                return 18000
            return -10000 if enemy else number(c.get('attack')) + 2000
        if name == 'Book of Moon':
            a, d = self.current_battle()
            if enemy and a and ref(a) == ref(c):
                return 16000
            return (self.value(c) + number(c.get('attack')) - number(c.get('defense'))) if enemy else -5000
        if name == 'Mystical Space Typhoon':
            return (8000 if c.get('name') == 'Call of the Haunted' else 3000) if enemy else -5000
        if name in ('Man-Eater Bug', 'Penguin Soldier', 'Compulsory Evacuation Device', 'Trap Hole', 'Fissure'):
            return self.value(c) + (1000 if number(c.get('level')) >= 5 else 0) if enemy else -self.value(c)-4000
        if 'attack' in hint or self.kind in ('attack_target', 'battle_target'):
            a = getattr(self, 'attacker', None)
            if a:
                return self.battle_value(a, c)
        return self.value(c) if enemy else -self.value(c)

    def score_choice(self, q):
        label = str(q.get('label', '')).lower()
        c = self.hydrate(q.get('card', {}))
        name = c.get('name', '')
        a, d = number(c.get('attack')), number(c.get('defense'))
        if 'shuffle' in label:
            return -100000
        if 'decline' in label or label in ('no', 'cancel', 'finish', 'done'):
            return 0
        if 'activate' in label or 'activation' in label:
            return self.activation_score(c)
        if label == 'normal summon' or label == 'special summon':
            level = number(c.get('level'))
            score = 3500 + a
            if a < 1600:
                score -= 3500
            if name in FLIPS:
                score -= 5000
            if level >= 5:
                n = 2 if level >= 7 else 1
                bodies = sorted(self.ours, key=self.value)[:n]
                score -= sum(self.value(x) for x in bodies) * .9
                if a > self.enemymax >= self.ourmax and self.theirs:
                    score += 3500
                if len(bodies) == 2 and sum(number(x.get('attack')) for x in bodies) > a:
                    score -= 2500
            if self.enemymax > a and not self.ours:
                score -= 2500
            return score
        if label == 'set monster':
            score = 2000 + d
            if name in FLIPS:
                score = 5500 + (1200 if self.theirs else 0)
                if name == 'Magician of Faith' and not any(number(x.get('type')) & 2 for x in self.me.get('graveyard', [])):
                    score = 700
            elif a >= 1700:
                score -= 2500
            if d >= self.enemymax > a:
                score += 1000
            return score
        if label == 'set spell/trap':
            typ = number(c.get('type'))
            if typ & 4 or typ & 65536:
                return 2000 if len(self.me.get('spells_traps', [])) < 4 else 500
            return -1000
        if 'flip summon' in label:
            if name == 'Man-Eater Bug':
                return 9500 if self.theirs else -1000
            if name == 'Penguin Soldier':
                return 9000 if self.theirs else -1000
            if name == 'Magician of Faith':
                return 8000 if any(number(x.get('type')) & 2 for x in self.me.get('graveyard', [])) else -500
            return 3000 if a >= self.enemymax else -800
        if 'position' in label and c:
            if attackpos(c):
                return 1000 if a < self.enemymax and d >= a else -500
            return 3000 if a >= 1600 and a >= self.enemymax else -500
        if label.startswith('attack'):
            target = q.get('target')
            if isinstance(target, dict):
                return self.battle_value(c, target) + 1000
            if 'direct' in label or not self.theirs:
                return self.battle_value(c) + 1000
            return max(self.battle_value(c, x) for x in self.theirs) + 1000
        if 'battle' in label and ('phase' in label or 'enter' in label):
            return 1000 if any(attackpos(x) for x in self.ours) else -20
        if 'main' in label and '2' in label:
            return 0
        if 'end' in label:
            return -10
        if self.kind in ('position', 'select_position'):
            return 20 if ('attack' in label and self.ownturn) or ('defense' in label and not self.ownturn) else 0
        if label in ('yes', 'apply effect', 'activate effect'):
            return 500 if self.theirs or self.source in RECOVERY else -10
        if q.get('card'):
            return self.target_score(q)
        return 0

    def constraints(self):
        p = self.prompt
        cs = p.get('constraints', {})
        if not isinstance(cs, dict):
            cs = {}
        merged = dict(cs)
        merged.update(p)
        lo = number(merged.get('min', merged.get('minimum', merged.get('min_count', 1))), 1)
        hi = number(merged.get('max', merged.get('maximum', merged.get('max_count', lo))), lo)
        return max(0, lo), max(0, hi), merged

    def candidate_index(self, c, i):
        return number(c.get('index', i), i)

    def subset_ok(self, indices):
        p = self.prompt
        candidates = p.get('candidates', [])
        byid = {self.candidate_index(c, i): c for i, c in enumerate(candidates)}
        if len(set(indices)) != len(indices) or any(i not in byid for i in indices):
            return False
        if 'order' in self.kind or p.get('action') == 'order':
            return set(indices) == set(byid)
        lo, hi, cs = self.constraints()
        if 'tribute' in self.kind:
            return len(indices) <= hi and sum(number(byid[i].get('weight', byid[i].get('release_param', 1)), 1) for i in indices) >= lo
        return lo <= len(indices) <= hi

    def valid_action(self, action):
        if not isinstance(action, str):
            return None
        for c in self.prompt.get('choices', []):
            if action.strip() == c.get('action'):
                return c['action']
        parts = action.strip().split()
        if len(parts) != 3 or parts[1] != str(self.prompt.get('token')):
            return None
        required = 'order' if 'order' in self.kind else 'select'
        if parts[0] != required or not self.prompt.get('candidates'):
            return None
        try:
            indices = [] if parts[2] == '-' else [int(x) for x in parts[2].split(',')]
        except ValueError:
            return None
        return action.strip() if self.subset_ok(indices) else None

    def fallback(self):
        p = self.prompt
        banned = self.failed.get(p.get('token'), set())
        choices = [c for c in p.get('choices', []) if c.get('action') not in banned]
        ranked = sorted(choices, key=self.score_choice, reverse=True)
        if ranked:
            return ranked[0]['action'], ranked
        candidates = p.get('candidates', [])
        if not candidates:
            return 'noop', []
        ids = [self.candidate_index(c, i) for i, c in enumerate(candidates)]
        if 'order' in self.kind:
            return 'order %s %s' % (p['token'], ','.join(map(str, ids))), []
        lo, hi, cs = self.constraints()
        weighted = 'tribute' in self.kind
        scores = {self.candidate_index(c, i): self.target_score(c) for i, c in enumerate(candidates)}
        # Bounded exact subset search; pool prompts are small. Large discard prompts
        # use sorted prefixes instead of an exponential enumeration.
        rankedids = sorted(ids, key=lambda i: scores[i], reverse=True)
        best = None
        bestscore = -float('inf')
        if len(ids) <= 18:
            subsets = itertools.chain.from_iterable(itertools.combinations(ids, n) for n in range(0 if weighted else lo, min(hi, len(ids))+1))
        else:
            subsets = (tuple(rankedids[:n]) for n in range(0 if weighted else lo, min(hi, len(ids))+1))
        for inds in subsets:
            if not self.subset_ok(inds):
                continue
            score = sum(scores[i] for i in inds)
            action = 'select %s %s' % (p['token'], ','.join(map(str, inds)) if inds else '-')
            if action in banned:
                continue
            if score > bestscore:
                bestscore, best = score, action
        if best:
            return best, []
        return 'select %s %s' % (p['token'], ','.join(map(str, rankedids[:lo])) if lo else '-'), []

    def should_consult(self, ranked):
        if self.calls >= 7 or self.total_calls >= 24 or self.obs.get('last_action_error'):
            return False
        b = self.ctx.budget.remaining()
        if b.get('system2_calls', 0) <= 2 or b.get('wall_s', 0) < 100:
            return False
        key = (self.round, self.obs.get('turn'), self.kind)
        if key == self.last_consult:
            return False
        if self.kind == 'main_command':
            useful = [q for q in ranked if self.score_choice(q) > 2500]
            return len(useful) > 1 and (bool(self.theirs) or bool(self.ours) or any('activation' in str(x.get('label')) for x in useful))
        if self.kind == 'chain_response':
            return bool(self.obs.get('chain')) and len(ranked) > 1 and (len(self.ours) + len(self.theirs) >= 3)
        return False

    def consult(self, fallback, ranked):
        self.calls += 1
        self.total_calls += 1
        self.last_consult = (self.round, self.obs.get('turn'), self.kind)
        state = {k: self.obs.get(k) for k in ('round','agent_seat','your_deck','turn_player','turn','phase','players','chain','prompt','last_action_error')}
        state['recent_events'] = self.obs.get('recent_events', [])[-18:]
        state['previous_intent'] = self.note
        state['last_effect_selected'] = self.source
        # Card text is included for effect cards only; vanilla flavour prose adds no information.
        state['effect_text'] = {v['name']: v.get('text','') for v in self.catalog.values() if not (number(v.get('type')) & 16)}
        state['controller_suggestion'] = fallback
        state['ranked_candidates'] = [{'action': q.get('action'), 'score': round(self.score_choice(q),1)} for q in ranked[:10]]
        prompt = (
            'You adjudicate a difficult decision for an autonomous Yu-Gi-Oh controller. '
            'Win this duel; LP is only a tiny tie-breaker. Only observed information is available. '
            'Use the current legal actions and exact card text. Destroying a source does not negate its effect. '
            'Preserve removal if ordinary battle achieves the same result; use removal before summoning when valuable. '
            'Do not tribute two productive attackers for a small ATK gain. Flip effects need suitable targets. '
            'The controller handles subsequent targets, battle arithmetic, sets and routine prompts. '
            'Choose the best CURRENT action, not an imagined future action. You have at most seven consultations per duel. '
            'Return compact JSON only: {"action":"exact complete legal action", "intent":"one short sentence about the plan"}. '
            'No additional explanation.\n' + json.dumps(state, separators=(',',':')))
        try:
            response = self.ctx.think(prompt, session_id=None, timeout_s=min(65, max(10, self.ctx.budget.remaining().get('wall_s', 90)-20)))
            text = response.text
            obj = json_object(text)
            action = self.valid_action(obj.get('action'))
            if not action:
                for m in re.finditer(r'\b(?:choose|select|order)\s+\S+\s+(?:\d+(?:,\d+)*|-)', text):
                    action = self.valid_action(m.group())
                    if action:
                        break
            if action:
                self.note = str(obj.get('intent', ''))[:400]
                return action
        except Exception as exc:
            self.ctx.log('adjudication_fallback', error=type(exc).__name__)
        return fallback

    def remember(self, action):
        for c in self.prompt.get('choices', []):
            if c.get('action') == action:
                label = str(c.get('label', '')).lower()
                if 'activ' in label or 'flip summon' in label:
                    self.source = c.get('card', {}).get('name', '')
                if label.startswith('attack'):
                    self.attacker = self.hydrate(c.get('card', {}))
                if label == 'normal summon' and number(self.hydrate(c.get('card', {})).get('level')) >= 5:
                    self.source = 'tribute'
                break
        self.last_action = action
        self.last_token = self.prompt.get('token')

    def run_episode(self, game):
        obs = self.ctx.observe()
        while not self.ctx.done:
            if not isinstance(obs, dict) or obs.get('done'):
                return
            self.prepare(obs)
            if not self.prompt:
                return
            token = self.prompt.get('token')
            if obs.get('last_action_error') and token == self.last_token and self.last_action:
                self.failed.setdefault(token, set()).add(self.last_action)
            action, ranked = self.fallback()
            if self.should_consult(ranked):
                action = self.consult(action, ranked)
            if self.ctx.done:
                return
            self.remember(action)
            try:
                obs = self.ctx.act(action)
            except Exception:
                if self.ctx.done:
                    return
                self.failed.setdefault(token, set()).add(action)
                obs = self.ctx.observe()
            if obs is None and not self.ctx.done:
                obs = self.ctx.observe()


def build(ctx):
    return Harness(ctx)
