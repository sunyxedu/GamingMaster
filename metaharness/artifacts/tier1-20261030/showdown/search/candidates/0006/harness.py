"""Short model-compiled matchup policies with measured outcome contracts."""
from __future__ import annotations
import json
import math
import re
import copy
from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def ident(x):
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def dump(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'), default=str)


def hp(p):
    c = str(p.get('condition', '0'))
    m = re.match(r'(\d+)/(\d+)', c)
    return (int(m[1]), max(1, int(m[2]))) if m else (0, 1)


def fraction(p):
    a, b = hp(p)
    return a / b


def species(p):
    return p.get('details', '').split(',')[0].strip()


def level(p):
    m = re.search(r'(?:^|, )L(\d+)', p.get('details', ''))
    return int(m[1]) if m else 100


def stage(n):
    n = max(-6, min(6, float(n or 0)))
    return (2 + n) / 2 if n >= 0 else 2 / (2 - n)


# Attack type -> super-effective, resisted, immune defender types. Gen 9 chart.
CHART = {
 'Normal': ('', 'Rock Steel', 'Ghost'),
 'Fire': ('Grass Ice Bug Steel', 'Fire Water Rock Dragon', ''),
 'Water': ('Fire Ground Rock', 'Water Grass Dragon', ''),
 'Electric': ('Water Flying', 'Electric Grass Dragon', 'Ground'),
 'Grass': ('Water Ground Rock', 'Fire Grass Poison Flying Bug Dragon Steel', ''),
 'Ice': ('Grass Ground Flying Dragon', 'Fire Water Ice Steel', ''),
 'Fighting': ('Normal Ice Rock Dark Steel', 'Poison Flying Psychic Bug Fairy', 'Ghost'),
 'Poison': ('Grass Fairy', 'Poison Ground Rock Ghost', 'Steel'),
 'Ground': ('Fire Electric Poison Rock Steel', 'Grass Bug', 'Flying'),
 'Flying': ('Grass Fighting Bug', 'Electric Rock Steel', ''),
 'Psychic': ('Fighting Poison', 'Psychic Steel', 'Dark'),
 'Bug': ('Grass Psychic Dark', 'Fire Fighting Poison Flying Ghost Steel Fairy', ''),
 'Rock': ('Fire Ice Flying Bug', 'Fighting Ground Steel', ''),
 'Ghost': ('Psychic Ghost', 'Dark', 'Normal'),
 'Dragon': ('Dragon', 'Steel', 'Fairy'),
 'Dark': ('Psychic Ghost', 'Fighting Dark Fairy', ''),
 'Steel': ('Ice Rock Fairy', 'Fire Water Electric Steel', ''),
 'Fairy': ('Fighting Dragon Dark', 'Fire Poison Steel', ''),
}


def effectiveness(t, types):
    a, b, c = [s.split() for s in CHART.get(t, ('', '', ''))]
    v = 1.0
    for d in types:
        v *= 0 if d in c else 2 if d in a else .5 if d in b else 1
    return v


class Combat:
    """Transparent approximate arithmetic, never a claim to know hidden sets."""
    def __init__(self, obs):
        self.obs = obs
        self.req = obs.get('request') or {}
        self.team = self.req.get('side', {}).get('pokemon', [])
        self.pub = obs.get('public_summary') or {}
        self.records = self.pub.get('pokemon', {})
        self.dex = obs.get('species_dex') or {}
        self.moves = obs.get('move_dex') or {}
        self.own = next((p for p in self.team if p.get('active')), self.team[0] if self.team else {})
        self.foe = self.records.get((obs.get('active') or {}).get('p2'), {})
        if not self.foe:
            self.foe = next((p for p in obs.get('opponent_revealed', []) if p.get('ident') == (obs.get('active') or {}).get('p2')), {})
        self.weather = ident(self.pub.get('weather', ''))
        self.field = dump(self.pub.get('field', {})).lower()

    def public(self, p):
        return self.records.get(p.get('ident'), p)

    def original_types(self, p):
        return self.dex.get(species(p), {}).get('types', [])

    def tera(self, p):
        return p.get('terastallized') or self.public(p).get('teraTypeRevealed') or ''

    def types(self, p, tera=None):
        t = self.tera(p) if tera is None else tera
        if t and t != 'Stellar':
            return [t]
        return self.original_types(p)

    def ability(self, p):
        return ident(p.get('ability') or self.public(p).get('ability', ''))

    def item(self, p):
        # Empty private item means consumed/removed, not unknown.
        return ident(p['item'] if 'item' in p else self.public(p).get('item', ''))

    def attack_types(self, p):
        result = list(self.original_types(p))
        t = self.tera(p)
        if t and t != 'Stellar' and t not in result:
            result.append(t)
        return result

    def boost(self, p, stat):
        return self.public(p).get('boosts', {}).get(stat, 0)

    def stat(self, p, stat, boosts=True):
        if stat == 'hp' and 'stats' in p:
            return hp(p)[1]
        if stat in (p.get('stats') or {}):
            s = p['stats'][stat]
        else:
            base = self.dex.get(species(p), {}).get('base_stats', {}).get(stat, 85)
            # Neutral 31 IV / 85 EV reference, NOT an asserted hidden spread.
            s = math.floor((2 * base + 52) * level(p) / 100) + (level(p) + 10 if stat == 'hp' else 5)
        return s * (stage(self.boost(p, stat)) if boosts and stat != 'hp' else 1)

    def speed(self, p):
        s = self.stat(p, 'spe')
        ab, it = self.ability(p), self.item(p)
        if 'par' in p.get('condition', '') and ab != 'quickfeet': s *= .5
        if it == 'choicescarf': s *= 1.5
        if it in ('ironball', 'machobrace'): s *= .5
        if ab == 'slushrush' and ('snow' in self.weather or 'hail' in self.weather): s *= 2
        if ab == 'sandrush' and 'sand' in self.weather: s *= 2
        if ab == 'swiftswim' and 'rain' in self.weather: s *= 2
        if ab == 'chlorophyll' and ('sun' in self.weather): s *= 2
        side = p.get('ident', '')[:2]
        cond = dump(self.pub.get('side_conditions', {}).get(side, {})).lower()
        if 'tailwind' in ident(cond): s *= 2
        return round(s)

    def effective(self, typ, defender, attacker=None, tera=None):
        v = effectiveness(typ, self.types(defender, tera))
        ab = self.ability(defender)
        ignore = self.ability(attacker or {}) in ('moldbreaker', 'teravolt', 'turboblaze')
        if not ignore:
            if typ == 'Ground' and ab == 'levitate': return 0
            if typ == 'Water' and ab in ('waterabsorb', 'stormdrain', 'dryskin'): return 0
            if typ == 'Electric' and ab in ('voltabsorb', 'lightningrod', 'motordrive'): return 0
            if typ == 'Fire' and ab in ('flashfire', 'wellbakedbody'): return 0
            if typ == 'Grass' and ab == 'sapsipper': return 0
            if ab == 'wonderguard' and v <= 1: return 0
            if typ in ('Fire', 'Ice') and ab == 'thickfat': v *= .5
            if typ == 'Fire' and ab in ('heatproof', 'waterbubble'): v *= .5
            if v > 1 and ab in ('filter', 'solidrock', 'prismarmor'): v *= .75
        if typ == 'Ground' and self.item(defender) == 'airballoon': return 0
        return v

    def damage(self, attacker, defender, move, tera=None, defender_tera=None):
        """Reference 85%-100% roll in defender max-HP %. No hidden item assumed."""
        mid = ident(move.get('name', ''))
        cat, typ = move.get('category'), move.get('type')
        power = move.get('base_power', 0) or 0
        notes = []
        if cat == 'Status': return None
        ta = self.tera(attacker) if tera is None else tera
        if mid == 'terablast' and ta and ta != 'Stellar':
            typ = ta
            cat = 'Physical' if self.stat(attacker, 'atk') > self.stat(attacker, 'spa') else 'Special'
        if mid in ('seismictoss', 'nightshade'):
            d = level(attacker) / self.stat(defender, 'hp') * 100
            return [round(d, 1), round(d, 1)] if self.effective(typ, defender, attacker, defender_tera) else [0, 0]
        if mid in ('eruption', 'waterspout', 'dragonenergy'): power = max(1, 150 * fraction(attacker))
        if mid in ('flail', 'reversal', 'grassknot', 'lowkick', 'heavyslam', 'heatcrash', 'gyroball', 'electroball', 'storedpower', 'powertrip', 'lastrespects', 'ragefist'):
            return None # Unmodeled variable power: do not invent precise damage.
        if power <= 0: return None
        if mid in ('bulletseed', 'rockblast', 'iciclespear', 'pinmissile', 'tailslap', 'scaleshot', 'armthrust', 'bonerush', 'watershuriken'):
            power *= 5 if self.ability(attacker) == 'skilllink' else 4.5 if self.item(attacker) == 'loadeddice' else 3
        if mid in ('doublehit', 'dualwingbeat', 'bonemerang', 'doubleironbash', 'twinbeam'): power *= 2
        if mid == 'populationbomb': power *= 9 if self.item(attacker) == 'widelens' else 6
        if mid == 'tripleaxel': power = 120
        if mid == 'knockoff' and self.item(defender): power *= 1.5
        if mid == 'facade' and any(s in attacker.get('condition', '') for s in (' brn',' par',' psn',' tox')): power *= 2
        if mid == 'hex' and self.public(defender).get('status'): power *= 2
        at, de = ('atk','def') if cat == 'Physical' else ('spa','spd')
        if mid == 'bodypress': at = 'def'
        if mid in ('psyshock', 'psystrike', 'secretsword'): de = 'def'
        attack_src = defender if mid == 'foulplay' else attacker
        a = self.stat(attack_src, at, self.ability(defender) != 'unaware')
        d = self.stat(defender, de, self.ability(attacker) != 'unaware')
        ia, aa, ad = self.item(attacker), self.ability(attacker), self.ability(defender)
        if at == 'atk' and aa in ('hugepower', 'purepower'): a *= 2
        if at == 'atk' and ia == 'choiceband': a *= 1.5
        if at == 'spa' and ia == 'choicespecs': a *= 1.5
        if de == 'spd' and self.item(defender) == 'assaultvest': d *= 1.5
        if self.item(defender) == 'eviolite': d *= 1.5
        if de == 'spd' and 'sand' in self.weather and 'Rock' in self.types(defender, defender_tera): d *= 1.5
        if de == 'def' and ('snow' in self.weather) and 'Ice' in self.types(defender, defender_tera): d *= 1.5
        if cat == 'Physical' and ' brn' in attacker.get('condition', '') and aa != 'guts' and mid != 'facade': a *= .5
        if aa == 'guts' and cat == 'Physical' and self.public(attacker).get('status'): a *= 1.5
        stab = 1.5 if typ in self.original_types(attacker) else 1
        if ta and ta != 'Stellar' and typ == ta: stab = 2 if typ in self.original_types(attacker) else 1.5
        if aa == 'adaptability' and stab > 1: stab = 2.25 if stab == 2 else 2
        eff = self.effective(typ, defender, attacker, defender_tera)
        v = ((2 * level(attacker) / 5 + 2) * power * a / max(1, d) / 50 + 2) * stab * eff
        if ia == 'lifeorb': v *= 1.3
        if aa == 'technician' and power <= 60: v *= 1.5
        if aa == 'tintedlens' and 0 < eff < 1: v *= 2
        if aa == 'waterbubble' and typ == 'Water': v *= 2
        if aa in ('toughclaws', 'sheerforce', 'punkrock', 'strongjaw', 'ironfist', 'sharpness'):
            pass # Move flags unavailable; Astra sees ability and handles this uncertainty.
        if 'rain' in self.weather:
            if typ == 'Water': v *= 1.5
            if typ == 'Fire': v *= .5
        if 'sun' in self.weather:
            if typ == 'Fire': v *= 1.5
            if typ == 'Water': v *= .5
        if ad in ('multiscale', 'shadowshield') and fraction(defender) == 1: v *= .5
        side = defender.get('ident', '')[:2]
        screens = ident(dump(self.pub.get('side_conditions', {}).get(side, {})))
        if aa != 'infiltrator' and ('auroraveil' in screens or ('reflect' in screens and cat == 'Physical') or ('lightscreen' in screens and cat == 'Special')): v *= .5
        v = v / max(1, self.stat(defender, 'hp')) * 100
        return [round(v * .85, 1), round(v, 1)]

    def hazards(self, p):
        cond = self.pub.get('side_conditions', {}).get('p1', {})
        if self.item(p) == 'heavydutyboots' or self.ability(p) == 'magicguard': return 0
        text = ident(dump(cond))
        v = 12.5 * effectiveness('Rock', self.types(p)) if 'stealthrock' in text else 0
        if 'spikes' in text and 'Flying' not in self.types(p) and self.ability(p) != 'levitate' and self.item(p) != 'airballoon':
            # Distinguish Toxic Spikes from Spikes; layers are advisory if unavailable.
            for k, value in cond.items():
                if ident(k).replace('move','') == 'spikes':
                    n = value.get('layers', 1) if isinstance(value, dict) else value if isinstance(value, int) else 1
                    v += {1:12.5, 2:100/6, 3:25}.get(n, 12.5)
        return round(v, 1)



def active_state(obs):
    c = Combat(obs)
    return c, {'own':c.own.get('ident'), 'foe':c.foe.get('ident'), 'selfhp':fraction(c.own), 'foehp':fraction(c.foe)}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.notes = ''
        self.history = []
        self.item_ledger = {}
        self.ability_ledger = {}
        self.policy = []
        self.bad = set()
        self.failure = 'Opening: establish team roles and a short conditional policy.'
        self.last_progress = {}
        self.steps = 0
        self.policy_age = 0
        self.damage = []
        self.last_outcome = {}
        self.policy_calls = 0

    def unpack(self, obs):
        if isinstance(obs, dict): return obs
        try:
            v = json.loads(obs)
            return v if isinstance(v, dict) else {}
        except (ValueError, TypeError): return {}

    def enrich(self, obs):
        """Retain explicitly revealed items, including passive item protocol lines."""
        obs = copy.deepcopy(self.unpack(obs))
        for line in obs.get('recent_events', []):
            parts = line.split('|')
            if len(parts) < 4:
                continue
            owner = re.sub(r'^(p[12])[a-z]:', r'\1:', parts[2])
            for part in parts[4:]:
                if parts[1] not in ('-item', '-enditem') and part.startswith('[of] '):
                    owner = re.sub(r'^(p[12])[a-z]:', r'\1:', part[5:])
            if parts[1] == '-ability':
                self.ability_ledger[owner] = parts[3]
            elif parts[1] == '-endability':
                self.ability_ledger[owner] = ''
            else:
                for part in parts[3:]:
                    if part.startswith('[from] ability: '):
                        self.ability_ledger[owner] = part[len('[from] ability: '):]
            if parts[1] == '-item':
                self.item_ledger[owner] = parts[3]
            elif parts[1] == '-enditem':
                self.item_ledger[owner] = ''
            else:
                for part in parts[4:]:
                    if part.startswith('[from] item: '):
                        self.item_ledger[owner] = part[len('[from] item: '):]
        for p in list((obs.get('public_summary') or {}).get('pokemon', {}).values()) + obs.get('opponent_revealed', []):
            if p.get('ident') in self.item_ledger:
                p['item'] = self.item_ledger[p['ident']]
            if p.get('ident') in self.ability_ledger:
                p['ability'] = self.ability_ledger[p['ident']]
        return obs

    def fallback(self, obs, legal):
        c = Combat(obs)
        scored = []
        active = (c.req.get('active') or [{}])[0]
        for action in legal:
            score = -1000
            bits = action.split()
            n = int(bits[1])-1 if len(bits)>1 and bits[1].isdigit() else -1
            if action.startswith('move ') and 0 <= n < len(active.get('moves', [])):
                m = active['moves'][n]
                data = c.moves.get(m.get('id',ident(m.get('move'))), {})
                damage = c.damage(c.own,c.foe,data,tera=c.own.get('teraType') if 'terastallize' in action else None)
                if damage is not None:
                    acc = data.get('accuracy',100)
                    acc = 1 if acc is True else float(acc or 100)/100
                    score = min(damage[0],100*fraction(c.foe)+10)*acc if damage[1] > 0 else -1000
                    if damage[0] >= 100*fraction(c.foe)+1: score += 45
                    if data.get('priority',0)>0 and fraction(c.own)<.4: score += 15
                else:
                    mid = ident(data.get('name',m.get('move')))
                    score = 4
                    if mid in ('recover','roost','slackoff','softboiled','milkdrink','shoreup','moonlight','morningsun','synthesis') and fraction(c.own)<.5: score=33
                    if mid in ('stealthrock','spikes','toxicspikes','stickyweb'):
                        side=ident(dump(c.pub.get('side_conditions',{}).get('p2',{})))
                        score=18 if mid not in side else -10
                    if mid in ('sleeptalk',) and 'slp' in c.own.get('condition',''): score=45
                if 'terastallize' in action: score-=12
            elif action.startswith('switch ') and 0<=n<len(c.team):
                p=c.team[n]
                best=0
                for mid in p.get('moves',[]):
                    d=c.damage(p,c.foe,c.moves.get(mid,{}))
                    if d: best=max(best,min(100,d[0]))
                threats=[]
                for t in c.attack_types(c.foe):
                    for cat in ('Physical','Special'):
                        d=c.damage(c.foe,p,{'name':'probe','type':t,'category':cat,'base_power':90})
                        if d: threats.append(d[1])
                score=best*.5+fraction(p)*20-max(threats or [30])*.55-c.hazards(p)-25
                if p.get('reviving'): score=30+sum(p.get('stats',{}).values())/100
            scored.append((score,action))
        return max(scored)[1] if scored else None


    def legal(self, obs):
        return [a for a in obs.get('legal_actions', []) if a != 'noop' and a not in self.bad]

    def prompt(self, obs):
        c = Combat(obs)
        state = {
            'turn': obs.get('turn'), 'legal_actions': self.legal(obs),
            'policy_identity_keys': {'foe':'current', 'self':[p.get('ident') for p in c.team]},
            'recent_authoritative_protocol': obs.get('recent_events', []),
            'request': obs.get('request'), 'active': obs.get('active'),
            'public_summary': obs.get('public_summary'),
            'opponent_revealed': obs.get('opponent_revealed'),
            'species_dex': obs.get('species_dex'), 'move_dex': obs.get('move_dex'),
            'verified_outcomes': self.history[-5:], 'observed_damage': self.damage[-8:],
            'strategy_memory': self.notes, 'replan_reason': self.failure,
            'ctx_progress': self.ctx.progress(), 'remaining': self.ctx.budget.remaining(),
        }
        return SYSTEM + '\nCURRENT STATE\n' + dump(state)

    def parse(self, text, legal):
        data = {}
        text = str(text)
        for m in re.finditer(r'\{', text):
            try:
                obj, _ = json.JSONDecoder().raw_decode(text[m.start():])
                if isinstance(obj, dict) and 'action' in obj:
                    data = obj
                    break
            except (ValueError, TypeError):
                pass
        action = data.get('action')
        if not isinstance(action, str) or action not in legal:
            action = None
            # Accept a plain command, not an action embedded in a long invalid policy.
            for line in text.splitlines():
                candidate = line.strip().strip('`').lower()
                if candidate in legal:
                    action = candidate
                    break
        return action, data

    def install(self, obs, data):
        self.policy = []
        self.policy_age = 0
        if isinstance(data.get('memo'), str):
            self.notes = data['memo'][:1100]
        rows = data.get('policy', [])
        if not isinstance(rows, list):
            return
        c = Combat(obs)
        own_names = {ident(species(p)): p for p in c.team}
        foe_names = {ident(species(c.foe))}
        aliases = {}
        for p in c.team:
            canonical = ident(species(p))
            aliases[ident(p.get('ident'))] = canonical
            aliases[ident(str(p.get('ident', '')).partition(':')[2])] = canonical
        for row in rows[:3]:
            if not isinstance(row, dict):
                continue
            r = dict(row)
            for k in ('self', 'foe', 'move', 'switch_to'):
                r[k] = ident(row.get(k, ''))
            r['self'] = aliases.get(r['self'], r['self'])
            r['switch_to'] = aliases.get(r['switch_to'], r['switch_to'])
            if r['foe'] in ('current', ident(c.foe.get('ident'))):
                r['foe'] = ident(species(c.foe))
            if r['self'] not in own_names or r['foe'] not in foe_names:
                continue
            is_switch = bool(r['switch_to'])
            if is_switch:
                if r['switch_to'] not in own_names:
                    continue
            else:
                mon = own_names[r['self']]
                if r['move'] not in mon.get('moves', []):
                    continue
                if c.moves.get(r['move'], {}).get('category') not in ('Physical', 'Special'):
                    continue
            try:
                for k, default in [('min_hp', 65), ('max_foe_hp', 100), ('max_loss', 25), ('min_damage', 3)]:
                    v = float(row.get(k, default))
                    if not math.isfinite(v) or not 0 <= v <= 100:
                        raise ValueError('invalid bound')
                    r[k] = v
                r['uses'] = min(3, max(1, int(row.get('uses', 1))))
            except (TypeError, ValueError, OverflowError):
                continue
            # Require an explicit envelope for any unobserved attacking reply.
            allowed = row.get('foe_moves', [])
            if not isinstance(allowed, list):
                continue
            r['foe_moves'] = {ident(x) for x in allowed if isinstance(x, str)}
            r['used'] = 0
            self.policy.append(r)
        self.ctx.log('policy_compiled', rules=[{**r, 'foe_moves': sorted(r['foe_moves'])} for r in self.policy])

    def cancel(self, reason):
        self.failure = reason
        if self.policy:
            self.ctx.log('policy_invalidated', reason=reason, age=self.policy_age)
        self.policy = []

    def policy_action(self, obs):
        if not self.policy:
            return None
        if self.policy_age >= 4:
            self.cancel('Short policy horizon exhausted; refresh tactical assessment.')
            return None
        c = Combat(obs)
        if self.failure:
            return None
        forced = any(c.req.get('forceSwitch') or [])
        for r in self.policy:
            if r['used'] >= r['uses']:
                continue
            if ident(species(c.own)) != r['self'] or ident(species(c.foe)) != r['foe']:
                continue
            unlisted = set(self.last_outcome.get('foe_moves', [])) - r['foe_moves']
            if unlisted - set(self.last_outcome.get('safe_attack_replies', [])):
                continue
            if 100 * fraction(c.foe) > r['max_foe_hp'] + .01:
                continue
            if r['switch_to']:
                if not forced:
                    continue
                target = next((p for p in c.team if ident(species(p)) == r['switch_to']), None)
                if target is None or fraction(target) * 100 < r['min_hp']:
                    continue
                a = 'switch ' + str(c.team.index(target) + 1)
            else:
                if forced or 100*fraction(c.own) + .01 < r['min_hp']:
                    continue
                # Absolute envelope plus observed-hit guard: never repeat into a lethal known hit.
                incoming = self.last_outcome.get('incoming_pct', 0)
                if incoming > r['max_loss'] + 2:
                    continue
                if incoming and 100*fraction(c.own) <= incoming*1.20 + 3:
                    continue
                active = (c.req.get('active') or [{}])[0]
                match = next((i for i,m in enumerate(active.get('moves', []),1)
                              if ident(m.get('id',m.get('move'))) == r['move']), None)
                if match is None:
                    continue
                a = 'move ' + str(match)
            if a in self.legal(obs):
                r['used'] += 1
                self.policy_age += 1
                return a
        self.cancel('No compiled rule covers current HP/request/matchup; replan with actual outcome.')
        return None

    def verify(self, old, action, new, source):
        cb, b = active_state(old)
        ca, a = active_state(new)
        prog = self.ctx.progress()
        events = new.get('recent_events', [])
        reasons = []
        if new.get('last_error') or prog.get('illegal_actions', 0) > self.last_progress.get('illegal_actions', 0):
            reasons.append('Chosen action rejected: ' + str(new.get('last_error')))
            self.bad.add(action)
        else:
            self.bad.clear()
        if prog.get('since_obs_change', 0) >= 2 or prog.get('since_score_change', 0) >= 4:
            reasons.append('Progress detector: repeated actions without observation/score progress')
        self.last_progress = prog
        # Record actual direct damage, separately from residuals, recoil and healing.
        hps = {p.get('ident'): fraction(p) for p in cb.team + list(cb.records.values())}
        incoming = 0.0
        damage_to_foe = 0.0
        moves_seen = []
        safe_attack_replies = set()
        current = None
        critical = False
        for line in events:
            s = line.split('|')
            if len(s) < 3:
                continue
            who = re.sub(r'^(p[12])[a-z]:', r'\1:', s[2])
            kind = s[1]
            if kind in ('switch', 'drag') and len(s) >= 5:
                hps[who] = fraction({'condition': s[4]})
                current = None
            elif kind in ('move', '-anim') and len(s) >= 5:
                current = (who, ident(s[3]), re.sub(r'^(p[12])[a-z]:', r'\1:', s[4]))
                critical = False
                if who.startswith('p2'):
                    moves_seen.append(ident(s[3]))
            elif kind == '-crit':
                critical = True
            elif kind in ('-damage', '-heal') and len(s) >= 4:
                now = fraction({'condition':s[3]})
                loss = 100 * max(0, hps.get(who, now) - now)
                if kind == '-damage' and '[from]' not in line and current and who == current[2]:
                    if who.startswith('p1'):
                        incoming += loss
                        if loss > 0 and current[0].startswith('p2'):
                            safe_attack_replies.add(current[1])
                    else:
                        damage_to_foe += loss
                    if loss:
                        self.damage.append({'turn':old.get('turn'), 'attacker':current[0],
                            'move':current[1], 'target':who, 'damage_pct':round(loss,1),
                            'critical':critical, 'faint_censored':not now,
                            'context':'single packet; recheck boosts/items/forms and turn effects'})
                hps[who] = now
            # Structural changes are checked relative to the actor, not by blanket event matching.
            if kind in ('replace','detailschange','-formechange','-transform','-swapboost','-copyboost',
                        '-invertboost','-clearboost','-clearallboost'):
                reasons.append('Unmodeled form/boost transformation: '+line)
            if kind == '-terastallize' and who.startswith('p2'):
                reasons.append('Opponent spent Tera')
            if kind in ('-boost', '-unboost') and who.startswith('p2'):
                # A planned Intimidate switch reliably lowers Attack; all other changes need review.
                expected = (kind == '-unboost' and len(s)>3 and s[3] == 'atk'
                            and action.startswith('switch ') and ca.ability(ca.own) == 'intimidate')
                if not expected:
                    reasons.append('Opponent boosts changed: '+line)
            if who.startswith('p1') and (
                    kind == '-unboost' or (kind == '-boost' and source == 'compiled_policy')):
                reasons.append('Our stat stages changed; recheck ability or move effects')
            if kind in ('-status','-curestatus','-start','-end','-singleturn','-singlemove','-prepare','-mustrecharge','-setboost'):
                reasons.append('Status/volatile changed: '+line)
            if kind in ('-immune','-fail','cant','-miss'):
                # The enemy hitting our immunity is evidence of the plan succeeding, not failing.
                if (kind == '-immune' and who.startswith('p1')):
                    if current and current[0].startswith('p2'):
                        safe_attack_replies.add(current[1])
                    continue
                reasons.append('Action blocked/failed/missed: '+line)
            if kind in ('-item','-enditem'):
                reasons.append('Held item changed: '+line)
            if kind == '-ability' and who.startswith('p2') and b['foe'] == a['foe']:
                previous = ident(cb.foe.get('ability'))
                if previous and len(s)>3 and previous != ident(s[3]):
                    reasons.append('Opponent ability changed: '+line)
        outcome = {'turn':old.get('turn'),'action':action,'source':source,
                   'own':b['own'],'foe':b['foe'],'own_after':a['own'],'foe_after':a['foe'],
                   'hp_before':[round(b['selfhp'],3),round(b['foehp'],3)],
                   'hp_after':[round(a['selfhp'],3),round(a['foehp'],3)],
                   'incoming_pct':round(incoming,1),'damage_pct':round(damage_to_foe,1),
                   'foe_moves':list(dict.fromkeys(moves_seen)),
                   'safe_attack_replies':sorted(safe_attack_replies)}
        self.last_outcome = outcome
        if self.policy:
            if b['foe'] != a['foe']:
                reasons.append('Opponent switched: new matchup needs a fresh model decision')
            # A future rule may describe our planned replacement, so do not require old self identity.
            relevant = [r for r in self.policy if r['foe'] == ident(species(ca.foe))
                        and r['self'] == ident(species(ca.own)) and r['used'] < r['uses']]
            if not relevant:
                reasons.append('No future rule for resulting active matchup')
            else:
                allowed = set().union(*(r['foe_moves'] for r in relevant))
                unexpected = set(moves_seen) - allowed - safe_attack_replies
                if unexpected:
                    reasons.append('Enemy reply outside policy: '+','.join(sorted(unexpected)))
                if incoming > max(r['max_loss'] for r in relevant) + 2:
                    reasons.append('Actual incoming damage exceeded policy envelope')
            if b['foe'] == a['foe'] and b['own'] == a['own'] and action.startswith('move '):
                idx = int(action.split()[1]) - 1
                amoves = (cb.req.get('active') or [{}])[0].get('moves', [])
                mid = ident(amoves[idx].get('id', amoves[idx].get('move'))) if idx < len(amoves) else ''
                used = [r for r in self.policy if r['self'] == ident(species(cb.own)) and r['move'] == mid]
                if used and damage_to_foe < min(r['min_damage'] for r in used):
                    reasons.append('Compiled attack did less than promised damage')
                if used and a['foehp'] >= b['foehp']:
                    reasons.append('Policy attack made no net progress after recovery')
            # Only a verified own entry ability may explain a changed environment.
            own_entry = action.startswith('switch ') and b['own'] != a['own']
            terrain_abilities = {'grassysurge':'grassyterrain','electricsurge':'electricterrain',
                                 'mistysurge':'mistyterrain','psychicsurge':'psychicterrain'}
            weather_abilities = {'drought':'sunnyday','drizzle':'raindance',
                                 'sandstream':'sandstorm','snowwarning':'snow'}
            ab = ca.ability(ca.own)
            own_trigger = any('[of] p1' in e and '[from] ability:' in e for e in events)
            def other_fields(pub):
                return {k:v for k,v in (pub.get('field') or {}).items()
                        if ident(k).removeprefix('move') not in terrain_abilities.values()}
            terrain_expected = (own_entry and own_trigger and ab in terrain_abilities
                                and terrain_abilities[ab] in ident(dump(ca.pub.get('field', {})))
                                and other_fields(cb.pub) == other_fields(ca.pub))
            weather_expected = (own_entry and own_trigger and ab in weather_abilities
                                and weather_abilities[ab] == ident(ca.pub.get('weather', '')))
            if cb.pub.get('weather') != ca.pub.get('weather') and not weather_expected:
                reasons.append('Unexpected weather change')
            if cb.pub.get('field') != ca.pub.get('field') and not terrain_expected:
                reasons.append('Unexpected field change')
            if cb.pub.get('side_conditions') != ca.pub.get('side_conditions'):
                reasons.append('Hazards/screens changed')
        if reasons:
            self.cancel('; '.join(dict.fromkeys(reasons))[:1800])
        elif self.policy:
            self.failure = ''
        else:
            self.failure = 'Previous short policy completed; evaluate latest evidence.'
        self.history.append({**outcome,'replan':self.failure})
        self.history = self.history[-8:]
        self.damage = self.damage[-20:]
        self.ctx.memory['strategy'] = self.notes
        self.ctx.log('verified_action', **outcome, replan=self.failure)

    def run_episode(self, game):
        ctx = self.ctx
        obs = self.enrich(ctx.observe())
        for _ in range(230):
            if ctx.done or not obs or obs.get('done'):
                return
            legal = self.legal(obs)
            if not legal:
                return
            action = self.policy_action(obs)
            source = 'compiled_policy'
            if len(legal) == 1:
                action, source = legal[0], 'only_legal'
            budget = ctx.budget.remaining()
            if not action and budget.get('system2_calls', 0) > 1 and budget.get('wall_s', 0) > 270:
                try:
                    result = ctx.think(self.prompt(obs), session_id=None, timeout_s=240.0)
                    action, data = self.parse(result.text, legal)
                    if action:
                        self.install(obs, data)
                        self.failure = ''
                        source = 'astra'
                        self.policy_calls += 1
                        ctx.log('astra_decision', action=action, memo=self.notes,
                                policy_size=len(self.policy))
                    else:
                        self.cancel('Malformed or unavailable action; temporary legal fallback, then replan.')
                except (BudgetExceeded, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    self.cancel('Model failure '+type(exc).__name__+'; temporary fallback then replan.')
            if not action:
                action = self.fallback(obs, legal)
                source = 'fallback'
            if not action or ctx.done:
                return
            old = obs
            try:
                new = ctx.act(action)
                obs = self.enrich(ctx.observe() if new is None else new)
            except (BudgetExceeded, ModelUnavailable):
                return
            except Exception as exc:
                if ctx.done:
                    return
                self.bad.add(action)
                self.cancel('Action rejected '+str(exc)[:300]+'; updated request required.')
                obs = self.enrich(ctx.observe())
                continue
            self.steps += 1
            self.verify(old, action, obs, source)


SYSTEM = '''You play Gen 9 Random Battle singles against a strong MCTS opponent. Win the battle.
You are also compiling a SHORT conditional policy to avoid paying for a model call on every
routine attack or forced replacement. Exact request/protocol are authoritative. Own request stats are unboosted; account for public
boosts, status, items, priority and field effects. Hidden enemy
moves/items remain uncertain; do not assume an immunity covers all likely coverage.
Consider strongest plausible opposing reply, move order/priority, survival before attacking,
entry damage plus the following attack on switches, and preserving a concrete endgame check.
Tera is worth spending now if it saves a key piece or converts a favorable exchange. No preview.

Return compact JSON (aim <=250 visible tokens):
{"action":"exact legal command","memo":"brief strategy, preserve/check assignments and uncertainties",
 "policy":[{"self":"p1: ExactIdentity","foe":"current","move":"moveid","min_hp":55,
 "max_foe_hp":100,"max_loss":30,"min_damage":15,"foe_moves":["moveid"],"uses":2}]}
The action is executed NOW. Policy rows authorize only SUBSEQUENT choices, for up to 4 decisions.
Use the exact supplied identities, not inferred forms (e.g. a suspected Plate does not rename Arceus).
Use self from policy_identity_keys.self and foe:"current"; move is a move ID, never a slot.
Rows are tried in order. Supply 1-3 useful rows if possible.
All rows concern the CURRENT opponent; any opponent switch forces a fresh decision.
Do not assume an opponent lacks dangerous coverage because one move hit our immunity.
HP/damage numbers are percentages of maximum HP (0-100). A row applies only to those active species,
own HP >= min_hp, foe HP <= max_foe_hp, and the incoming damage stays <= max_loss. min_damage is the
minimum expected direct damage when that attack executes. Use realistic conservative bounds.
For a move that acts second, min_hp must exceed the strongest plausible hit plus residual/recoil.
Do not compile a rule on the theory that unrevealed coverage cannot exist.
foe_moves lists expected compatible enemy replies. An unlisted move may also be accepted ONLY
when it actually deals damage within max_loss (or hits our immunity), without disruptive effects.
Thus min_hp/max_loss certify a safe measured exchange, not merely the spelling of an enemy move.
Unlisted nondamaging replies and damage above your envelope trigger replanning.
Only damaging moves can be automated; no Tera/status/setup automation. For a planned forced switch
(pivot/faint replacement only), use switch_to:"p1: ExactIdentity" instead of move, self is the current active
BEFORE replacement, and min_hp bounds the target. Then add a second row for its following attack.
A normal switch or setup may be action NOW, followed by an attacking policy for the resulting self.
Expected Intimidate, own entry terrain/weather abilities, and an enemy attack hitting our immunity
do not invalidate the policy.
Opponent Tera/boosts, changed field/status/items, failed attacks, low HP, unmodeled mechanics, or
no actual progress cancel it. If the battle is genuinely unstable use policy:[], not a reckless rule.
Keep the memo <=45 words. Do not narrate a calculation or output alternatives outside the JSON.
'''


def build(ctx):
    return Harness(ctx)
