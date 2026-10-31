"""Counterfactual combat cards with observed-damage feedback; public information only."""
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

    def cards(self):
        rows = []
        if not self.foe: return rows
        for i, p in enumerate(self.team, 1):
            if fraction(p) <= 0: continue
            attacks = []
            moveids = p.get('moves', [])
            if p.get('active'):
                a = self.req.get('active') or [{}]
                moveids = [m.get('id', ident(m.get('move'))) for m in a[0].get('moves', [])]
            for j, mid in enumerate(moveids, 1):
                m = self.moves.get(mid, {})
                if not m: continue
                r = {'slot':j, 'move':mid}
                dmg = self.damage(p, self.foe, m)
                if dmg is not None: r['reference_damage_pct'] = dmg
                if p.get('active') and p.get('teraType') and not self.tera(p):
                    td = self.damage(p, self.foe, m, tera=p['teraType'])
                    if td != dmg: r['tera_reference_damage_pct'] = td
                attacks.append(r)
            probes = {}
            for t in self.attack_types(self.foe):
                # This is a normalized threat probe, not an inferred hidden move.
                physical = self.damage(self.foe, p, {'name':'probe','type':t,'category':'Physical','base_power':90})
                special = self.damage(self.foe, p, {'name':'probe','type':t,'category':'Special','base_power':90})
                probes[t] = {'90BP_Physical':physical, '90BP_Special':special}
            row = {'slot':i, 'species':species(p), 'hp_pct':round(100*fraction(p),1), 'effective_speed_reference':self.speed(p), 'entry_damage_pct':self.hazards(p), 'attacks':attacks, 'foe_STAB_probes_pct':probes}
            if p.get('active') and p.get('teraType') and not self.tera(p):
                row['after_tera_defense_multipliers'] = {t:self.effective(t,p,self.foe,p['teraType']) for t in self.attack_types(self.foe)}
            rows.append(row)
        return rows


def active_state(obs):
    c = Combat(obs)
    return c, {'own':c.own.get('ident'), 'foe':c.foe.get('ident'), 'selfhp':fraction(c.own), 'foehp':fraction(c.foe)}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.notes = ''
        self.history = []
        self.measurements = []
        self.plan = None
        self.failure = []
        self.bad = set()
        self.last_progress = {}
        self.steps = 0
        self.item_ledger = {}
        self.max_call_elapsed = 0.0

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
        return obs

    def legal(self, obs):
        return [a for a in obs.get('legal_actions', []) if isinstance(a, str) and a != 'noop' and a not in self.bad]

    def fingerprint(self, obs):
        c = Combat(obs)
        return dump({'active':obs.get('active'), 'details':[p.get('details') for p in (c.own,c.foe)], 'effects':[{k:c.public(p).get(k) for k in ('status','boosts','volatiles','ability','teraTypeRevealed')} for p in (c.own,c.foe)], 'item':[p.get('item') for p in (c.own,c.foe)], 'field':c.pub.get('field'), 'weather':c.pub.get('weather'), 'sides':c.pub.get('side_conditions'), 'reveals':[{k:p.get(k) for k in ('ident','moves','ability','item','teraTypeRevealed')} for p in obs.get('opponent_revealed', [])]})

    def feedback(self, before, action, after, source=None):
        cb, b = active_state(before)
        ca, a = active_state(after)
        ev = after.get('recent_events') or []
        prog = self.ctx.progress()
        failures = []
        if after.get('last_error') or prog.get('illegal_actions',0) > self.last_progress.get('illegal_actions',0): failures.append('REJECTED: '+str(after.get('last_error')))
        if prog.get('since_obs_change',0) >= 2: failures.append('Observation unchanged for multiple actions')
        if prog.get('since_score_change',0) >= 4: failures.append('Four actions without score change: check actual progress/setup/recovery loop')
        if a['own'] == b['own'] and a['selfhp'] < b['selfhp']:
            loss = 100 * (b['selfhp'] - a['selfhp'])
            if loss >= 25: failures.append('Lost %.1f%% max HP since last choice; audit survival before attacking/setup' % loss)
        if self.plan and self.fingerprint(after) != self.plan['fingerprint']: failures.append('Plan invalidated: matchup/reveals/effects changed')
        unusual = ('|replace|','|detailschange|','|-formechange|','|-transform|',
                   '|-start|','|-end|','|-activate|','|-singleturn|','|-singlemove|',
                   '|-prepare|','|-mustrecharge|','|-swapboost|','|-copyboost|',
                   '|-invertboost|','|-clearboost|','|-clearallboost|')
        if self.plan and any(e.startswith(unusual) for e in ev):
            failures.append('Plan invalidated: special mechanics require protocol review')
        if any('|-immune|' in e or '|-fail|' in e or '|cant|' in e or '|-miss|' in e for e in ev): failures.append('Action failed/immune/missed/blocked: see protocol')
        if self.plan and a['foe'] == b['foe'] and a['foehp'] >= b['foehp']: failures.append('Attack plan made no net HP progress')
        if source == 'fallback':
            failures.extend(self.failure[:2])
        self.failure = failures
        self.last_progress = prog
        self.history.append({'turn':before.get('turn'), 'action':action, 'source':source, 'own':b['own'], 'foe':b['foe'], 'hp_before':[round(b['selfhp'],3),round(b['foehp'],3)], 'hp_after':[round(a['selfhp'],3),round(a['foehp'],3)], 'failures':failures})
        self.history = self.history[-8:]
        # Keep direct, observed damage packets and move ordering with context. Not inferred speed facts.
        conds = {p.get('ident'):fraction(p) for p in cb.team + list(cb.records.values())}
        orders, current = [], None
        for e in ev:
            parts = e.split('|')
            if len(parts) < 3: continue
            kind = parts[1]
            who = re.sub(r'^(p[12])[a-z]:', r'\1:', parts[2])
            if kind in ('switch','drag') and len(parts)>4:
                conds[who] = fraction({'condition':parts[4]}); current = None
            elif kind == 'move' and len(parts)>3:
                current = {'attacker':who, 'move':parts[3], 'turn':before.get('turn')}
                orders.append(who + ':' + parts[3])
            elif kind == '-damage' and len(parts)>3:
                now = fraction({'condition':parts[3]})
                if current and who in conds and len(parts) == 4:
                    loss = max(0, conds[who]-now)*100
                    if loss:
                        self.measurements.append({**current,'target':who,'damage_pct':round(loss,1),'context_boosts':{p.get('ident'):cb.public(p).get('boosts',{}) for p in (cb.own,cb.foe)},'note':'observed packet; may be critical or one hit of multihit; inspect protocol'})
                conds[who] = now
            elif kind == '-heal' and len(parts)>3: conds[who] = fraction({'condition':parts[3]})
        if len(orders)>=2: self.measurements.append({'turn':before.get('turn'),'move_order':orders,'note':'priority/Trick Room/activation can explain order; not automatically speed'})
        self.measurements = self.measurements[-24:]
        if failures: self.plan = None

    def repeat_action(self, obs):
        p = self.plan
        if not p or p['remaining'] <= 0: return None
        c, s = active_state(obs)
        a = p['action']
        if a not in self.legal(obs) or p['fingerprint'] != self.fingerprint(obs): return None
        if s['selfhp'] < p['minhp'] or s['foehp'] > p['maxfoehp']: return None
        if s['foehp'] >= p['lastfoehp'] or s['selfhp'] <= .3: return None
        if p.get('lastloss',0) * 1.35 >= s['selfhp']: return None
        p['remaining'] -= 1
        p['lastfoehp'] = s['foehp']
        return a

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
                    score = min(damage[0],100*fraction(c.foe)+10)*acc
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

    def prompt(self, obs):
        c=Combat(obs)
        team=[]
        for i,p in enumerate(c.team,1):
            keep={k:p.get(k) for k in ('ident','details','condition','active','stats','moves','ability','item','teraType','terastallized','reviving')}
            keep.update(slot=i, types=c.types(p), effects=c.public(p))
            team.append(keep)
        state={
          'turn':obs.get('turn'), 'legal_actions':self.legal(obs), 'request_active':c.req.get('active'),
          'forceSwitch':c.req.get('forceSwitch'), 'team':team,
          'foe_active':c.foe, 'opponent_revealed':obs.get('opponent_revealed',[]),
          'unrevealed_foes':max(0,6-len(obs.get('opponent_revealed',[]))),
          'public_field':{k:v for k,v in c.pub.items() if k!='pokemon'},
          'move_dex':c.moves, 'species_dex':c.dex,
          'last_error':obs.get('last_error'), 'recent_protocol':obs.get('recent_events',[]),
          'last_choices':self.history, 'observed_combat_evidence':self.measurements,
          'prior_strategy':self.notes, 'plan_failures':self.failure,
          'tactical_cards':c.cards(), 'foe_neutral_speed_reference':c.speed(c.foe),
          'budget':self.ctx.budget.remaining(), 'progress':self.ctx.progress(),
          'max_observed_end_to_end_call_seconds':round(self.max_call_elapsed,1),
        }
        return '''You are an expert Gen 9 Random Battle singles player facing an MCTS opponent. Win the whole six-Pokemon battle. Use only supplied public/private information. This is a NEW stateless consultation; the strategy note below is your memory.
Choose a legal action after a two-turn survival audit: (1) what is the foe's strongest plausible attack, priority, setup or switch? (2) who moves first, can we survive and actually execute our move, including entry hazards/recoil? (3) which teammate must remain alive to stop their most dangerous threat? A slower switch-in usually must survive TWO attacks, the entry hit and next turn's hit, before dealing damage; merely surviving entry does not make it a counter. If attacking loses a valuable mon before it moves, prefer a sound switch or defensive Tera; a deliberate sacrifice needs a concrete revenge-kill plan. Don't take 40% damage for 10% resisted chip. Don't use setup without a credible survival/payoff line. Consider known opponent switch-ins, not only the active target. Tera is usable once; spending it to save a decisive check or obtain a KO is better than dying with it unused. Avoid switching loops and make progress.
Numbers in tactical_cards are REFERENCE estimates, not exact damage bounds: hidden foe uses neutral 31 IV/85 EV reference and no unobserved item; abilities/variable-power/special mechanics are incomplete. A nonzero damage estimate does NOT prove the move connects: check possible ability immunities, especially Ground against possible Levitate. 90BP_STAB probes are hypothetical attacks, NOT claimed moves or a maximum threat: scale to actual likely moves (e.g. 160BP is much stronger), include coverage, priority, Scarf, boosts. Prefer observed damage and protocol over estimates. Own request.stats are unboosted; cards apply public boosts. Opponent HP is rounded. Recent protocol is authoritative for Illusion, Transform, boosts and effects. Look at priority before speed. Never infer hidden coverage is absent. Named Pokémon in memory are stable; team slots reorder. Forced replacements/pivots need a switch now, not an attack.
Return ONE JSON object under 160 words:
{"action":"one exact legal string","audit":"brief survival + opponent response + why this line wins","memory":"updated strategy, preserved check and threat assumptions; distinguish observations from guesses","repeat":0,"min_self_hp":0.6,"max_foe_hp":1.0}
repeat can be 1 or 2 ONLY to authorize that many extra uses of the SAME plain damaging attack if both actives, reveals, status/boosts/field are unchanged and it keeps reducing foe HP. Never authorize repeat for setup, recovery, switching, charge, lock-in, pivot or Tera. Code will also check HP and invalidate on surprises. Use repeat=0 at tactical turning points. min_self_hp/max_foe_hp are fractions [0,1] for extra attacks. No prose outside JSON.
STATE:
''' + dump(state)

    def parse(self, text, legal):
        data = {}
        if not isinstance(text,str): return data
        decoder=json.JSONDecoder()
        for match in re.finditer(r'\{',text):
            try:
                obj,_=decoder.raw_decode(text[match.start():])
                if isinstance(obj,dict) and obj.get('action') in legal: return obj
            except (ValueError,TypeError,RecursionError,OverflowError): pass
        # Accept an exact legal response or a standalone action line, never stale arbitrary indices.
        text=text.strip().strip('`').strip()
        if text in legal: return {'action':text}
        for line in text.splitlines():
            a=line.strip().strip('`').strip()
            if a in legal: return {'action':a}
        return data

    def run_episode(self, game):
        ctx=self.ctx
        obs=self.enrich(ctx.observe())
        while not ctx.done and not obs.get('done') and self.steps<210:
            legal=self.legal(obs)
            if not legal:
                if self.bad:
                    self.bad.clear(); legal=self.legal(obs)
                if not legal: return
            budget=ctx.budget.remaining()
            if budget.get('wall_s',0)<=2: return
            source='fallback'
            action=None
            if len(legal)==1:
                action=legal[0]; source='only_legal'
            if action is None and not self.failure:
                action=self.repeat_action(obs)
                if action: source='verified_repeat'
            if action is None:
                self.plan=None
                # Keep one call unused: reaching the hard budget closes the episode.
                # Leave enough wall time for an ordinary long call and final engine work.
                call_reserve=max(180.0,self.max_call_elapsed*1.2+90)
                if budget.get('system2_calls',0)>1 and budget.get('wall_s',0)>call_reserve:
                    call_started=ctx.clock.now()
                    try:
                        result=ctx.think(self.prompt(obs),session_id=None,timeout_s=min(180.0,budget['wall_s']-30))
                        self.max_call_elapsed=max(self.max_call_elapsed,ctx.clock.now()-call_started)
                        reply=self.parse(result.text,legal)
                        action=reply.get('action')
                        if action:
                            source='astra'
                            self.notes=str(reply.get('memory',reply.get('audit',self.notes)))[:2200]
                            self.failure=[]
                            c,s=active_state(obs)
                            try: repeats=max(0,min(2,int(reply.get('repeat',0))))
                            except (TypeError,ValueError,OverflowError): repeats=0
                            n=int(action.split()[1])-1 if action.startswith('move ') else -1
                            ms=(c.req.get('active') or [{}])[0].get('moves',[])
                            md=c.moves.get(ms[n].get('id'),{}) if 0<=n<len(ms) else {}
                            forbidden=('uturn','voltswitch','flipturn','batonpass','partingshot','teleport','chillyreception','shedtail','outrage','petaldance','thrash','solarbeam','solarblade','meteorbeam','electroshot','skullbash','skyattack','fly','dig','dive','phantomforce','shadowforce','hyperbeam','gigaimpact','blastburn','frenzyplant','hydrocannon','prismaticlaser','eternabeam','gigatonhammer','bloodmoon','recharge')
                            if repeats and re.fullmatch(r'move [1-4]',action) and md.get('category') in ('Physical','Special') and ident(md.get('name')) not in forbidden:
                                try:
                                    minh=max(.3,min(1,float(reply.get('min_self_hp',.6))))
                                    maxh=max(0,min(1,float(reply.get('max_foe_hp',1))))
                                except (TypeError,ValueError,OverflowError): minh,maxh=.6,1
                                self.plan={'action':action,'remaining':repeats,'minhp':minh,'maxfoehp':maxh,'fingerprint':self.fingerprint(obs),'lastfoehp':s['foehp'],'lastloss':0}
                        else:
                            self.failure=['Astra response malformed/unavailable action; choose bounded fallback and replan next decision']
                    except (BudgetExceeded,ModelUnavailable): return
                    except (TimeoutError,RuntimeError) as e:
                        self.max_call_elapsed=max(self.max_call_elapsed,ctx.clock.now()-call_started)
                        self.failure=['Astra call failed: '+type(e).__name__+'; emergency action then replan']
                if ctx.done: return
                if not action: action=self.fallback(obs,legal)
            if not action: return
            ctx.log('combat_choice',action=action,source=source,failures=self.failure,memory=self.notes)
            before=obs
            try:
                out=ctx.act(action)
                obs=self.enrich(out if out is not None else ctx.observe())
            except (BudgetExceeded,ModelUnavailable): return
            except Exception as e:
                if ctx.done: return
                self.bad.add(action); self.plan=None
                self.failure=['Rejected action '+action+': '+str(e)[:300]]
                obs=self.enrich(ctx.observe())
                self.steps+=1
                continue
            self.steps+=1
            if not obs: return
            if obs.get('last_error'):
                self.bad.add(action)
            else:
                self.bad.clear()
            if self.plan:
                cb,b=active_state(before); ca,a=active_state(obs)
                self.plan['lastloss']=max(0,b['selfhp']-a['selfhp'])
            self.feedback(before,action,obs,source)
            ctx.memory['combat_strategy']=self.notes


def build(ctx):
    return Harness(ctx)
