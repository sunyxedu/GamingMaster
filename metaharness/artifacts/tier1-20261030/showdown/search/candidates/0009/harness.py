"""Recurrent, evidence-grounded tactical consultations; public observations only.
Combat arithmetic adapted from 0002 is used ONLY for emergency fallback.
"""
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



# Public move records seen in archive observations; no hidden opponent sets.
PUBLIC_MOVES = {'calmmind': {'name': 'Calm Mind', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk and Sp. Def by 1."}, 'encore': {'name': 'Encore', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'Target repeats its last move for its next 3 turns.'}, 'drainingkiss': {'name': 'Draining Kiss', 'type': 'Fairy', 'category': 'Special', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 75% of the damage dealt.'}, 'terablast': {'name': 'Tera Blast', 'type': 'Normal', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'If Terastallized: Phys. if Atk > SpA, type = Tera.'}, 'shadowball': {'name': 'Shadow Ball', 'type': 'Ghost', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Sp. Def by 1."}, 'willowisp': {'name': 'Will-O-Wisp', 'type': 'Fire', 'category': 'Status', 'base_power': 0, 'accuracy': 85, 'priority': 0, 'description': 'Burns the target.'}, 'nastyplot': {'name': 'Nasty Plot', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk by 2."}, 'stickyweb': {'name': 'Sticky Web', 'type': 'Bug', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Speed of grounded foes by 1 on switch-in.'}, 'tripleaxel': {'name': 'Triple Axel', 'type': 'Ice', 'category': 'Physical', 'base_power': 20, 'accuracy': 90, 'priority': 0, 'description': 'Hits 3 times. Each hit can miss, but power rises.'}, 'bulletseed': {'name': 'Bullet Seed', 'type': 'Grass', 'category': 'Physical', 'base_power': 25, 'accuracy': 100, 'priority': 0, 'description': 'Hits 2-5 times in one turn.'}, 'knockoff': {'name': 'Knock Off', 'type': 'Dark', 'category': 'Physical', 'base_power': 65, 'accuracy': 100, 'priority': 0, 'description': '1.5× damage if foe holds an item. Removes item.'}, 'poisonjab': {'name': 'Poison Jab', 'type': 'Poison', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'}, 'closecombat': {'name': 'Close Combat', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Defense and Sp. Def by 1."}, 'trick': {'name': 'Trick', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."}, 'zenheadbutt': {'name': 'Zen Headbutt', 'type': 'Psychic', 'category': 'Physical', 'base_power': 80, 'accuracy': 90, 'priority': 0, 'description': '20% chance to make the target flinch.'}, 'psyshock': {'name': 'Psyshock', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Damages target based on Defense, not Sp. Def.'}, 'hypervoice': {'name': 'Hyper Voice', 'type': 'Normal', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'}, 'thunderbolt': {'name': 'Thunderbolt', 'type': 'Electric', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to paralyze the target.'}, 'flipturn': {'name': 'Flip Turn', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'}, 'freezedry': {'name': 'Freeze-Dry', 'type': 'Ice', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze. Super effective on Water.'}, 'icebeam': {'name': 'Ice Beam', 'type': 'Ice', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze the target.'}, 'hydropump': {'name': 'Hydro Pump', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 80, 'priority': 0, 'description': 'No additional effect.'}, 'scald': {'name': 'Scald', 'type': 'Water', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'}, 'thunderwave': {'name': 'Thunder Wave', 'type': 'Electric', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': 'Paralyzes the target.'}, 'voltswitch': {'name': 'Volt Switch', 'type': 'Electric', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'}, 'psychoboost': {'name': 'Psycho Boost', 'type': 'Psychic', 'category': 'Special', 'base_power': 140, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."}, 'superpower': {'name': 'Superpower', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Attack and Defense by 1."}, 'sleeptalk': {'name': 'Sleep Talk', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User must be asleep. Uses another known move.'}, 'rest': {'name': 'Rest', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User sleeps 2 turns and restores HP and status.'}, 'wavecrash': {'name': 'Wave Crash', 'type': 'Water', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'curse': {'name': 'Curse', 'type': 'Ghost', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Curses if Ghost, else -1 Spe, +1 Atk, +1 Def.'}, 'revivalblessing': {'name': 'Revival Blessing', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Revives a fainted Pokemon to 50% HP.'}, 'recover': {'name': 'Recover', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'}, 'psychic': {'name': 'Psychic', 'type': 'Psychic', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'bugbuzz': {'name': 'Bug Buzz', 'type': 'Bug', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'facade': {'name': 'Facade', 'type': 'Normal', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if user is burn/poison/paralyzed.'}, 'flareblitz': {'name': 'Flare Blitz', 'type': 'Fire', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil. 10% chance to burn. Thaws user.'}, 'trailblaze': {'name': 'Trailblaze', 'type': 'Grass', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': "100% chance to raise the user's Speed by 1."}, 'bodyslam': {'name': 'Body Slam', 'type': 'Normal', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': '30% chance to paralyze the target.'}, 'earthquake': {'name': 'Earthquake', 'type': 'Ground', 'category': 'Physical', 'base_power': 100, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dig.'}, 'slackoff': {'name': 'Slack Off', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'}, 'bulkup': {'name': 'Bulk Up', 'type': 'Fighting', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Defense by 1."}, 'megahorn': {'name': 'Megahorn', 'type': 'Bug', 'category': 'Physical', 'base_power': 120, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect.'}, 'toxic': {'name': 'Toxic', 'type': 'Poison', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': "Badly poisons the target. Poison types can't miss."}, 'sludgebomb': {'name': 'Sludge Bomb', 'type': 'Poison', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'}, 'protect': {'name': 'Protect', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 4, 'description': 'Prevents moves from affecting the user this turn.'}, 'rapidspin': {'name': 'Rapid Spin', 'type': 'Normal', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'Free user from hazards/bind/Leech Seed; +1 Spe.'}, 'bravebird': {'name': 'Brave Bird', 'type': 'Flying', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'acrobatics': {'name': 'Acrobatics', 'type': 'Flying', 'category': 'Physical', 'base_power': 55, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if the user has no held item.'}, 'swordsdance': {'name': 'Swords Dance', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack by 2."}, 'leafblade': {'name': 'Leaf Blade', 'type': 'Grass', 'category': 'Physical', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'High critical hit ratio.'}, 'substitute': {'name': 'Substitute', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User takes 1/4 its max HP to put in a substitute.'}, 'doubleedge': {'name': 'Double-Edge', 'type': 'Normal', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'originpulse': {'name': 'Origin Pulse', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'}, 'waterspout': {'name': 'Water Spout', 'type': 'Water', 'category': 'Special', 'base_power': 150, 'accuracy': 100, 'priority': 0, 'description': "Less power as user's HP decreases. Hits foe(s)."}, 'thunder': {'name': 'Thunder', 'type': 'Electric', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to paralyze. Can't miss in rain."}, 'drainpunch': {'name': 'Drain Punch', 'type': 'Fighting', 'category': 'Physical', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'}, 'jetpunch': {'name': 'Jet Punch', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'}, 'strengthsap': {'name': 'Strength Sap', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User heals HP=target's Atk stat. Lowers Atk by 1."}, 'earthpower': {'name': 'Earth Power', 'type': 'Ground', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'energyball': {'name': 'Energy Ball', 'type': 'Grass', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'roost': {'name': 'Roost', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Heals 50% HP. Flying-type removed 'til turn ends."}, 'bodypress': {'name': 'Body Press', 'type': 'Fighting', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "Uses user's Def stat as Atk in damage calculation."}, 'stealthrock': {'name': 'Stealth Rock', 'type': 'Rock', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts foes on switch-in. Factors Rock weakness.'}, 'uturn': {'name': 'U-turn', 'type': 'Bug', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'}, 'ironhead': {'name': 'Iron Head', 'type': 'Steel', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to make the target flinch.'}, 'fireblast': {'name': 'Fire Blast', 'type': 'Fire', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': '10% chance to burn the target.'}, 'darkpulse': {'name': 'Dark Pulse', 'type': 'Dark', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '20% chance to make the target flinch.'}, 'yawn': {'name': 'Yawn', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Puts the target to sleep after 1 turn.'}, 'moonblast': {'name': 'Moonblast', 'type': 'Fairy', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': "30% chance to lower the target's Sp. Atk by 1."}, 'synthesis': {'name': 'Synthesis', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'}, 'outrage': {'name': 'Outrage', 'type': 'Dragon', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Lasts 2-3 turns. Confuses the user afterwards.'}, 'dragondance': {'name': 'Dragon Dance', 'type': 'Dragon', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Speed by 1."}, 'heatcrash': {'name': 'Heat Crash', 'type': 'Fire', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the user than the target.'}, 'overheat': {'name': 'Overheat', 'type': 'Fire', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."}, 'highhorsepower': {'name': 'High Horsepower', 'type': 'Ground', 'category': 'Physical', 'base_power': 95, 'accuracy': 95, 'priority': 0, 'description': 'No additional effect.'}, 'supercellslam': {'name': 'Supercell Slam', 'type': 'Electric', 'category': 'Physical', 'base_power': 100, 'accuracy': 95, 'priority': 0, 'description': 'User is hurt by 50% of its max HP if it misses.'}, 'stoneedge': {'name': 'Stone Edge', 'type': 'Rock', 'category': 'Physical', 'base_power': 100, 'accuracy': 80, 'priority': 0, 'description': 'High critical hit ratio.'}, 'suckerpunch': {'name': 'Sucker Punch', 'type': 'Dark', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first. Fails if target is not attacking.'}, 'bugbite': {'name': 'Bug Bite', 'type': 'Bug', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': "User steals and eats the target's Berry."}, 'dualwingbeat': {'name': 'Dual Wingbeat', 'type': 'Flying', 'category': 'Physical', 'base_power': 40, 'accuracy': 90, 'priority': 0, 'description': 'Hits 2 times in one turn.'}, 'psychicnoise': {'name': 'Psychic Noise', 'type': 'Psychic', 'category': 'Special', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'For 2 turns, the target is prevented from healing.'}, 'focusblast': {'name': 'Focus Blast', 'type': 'Fighting', 'category': 'Special', 'base_power': 120, 'accuracy': 70, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."}, 'sludgewave': {'name': 'Sludge Wave', 'type': 'Poison', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': '10% chance to poison adjacent Pokemon.'}, 'roar': {'name': 'Roar', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': -6, 'description': 'Forces the target to switch to a random ally.'}, 'iciclecrash': {'name': 'Icicle Crash', 'type': 'Ice', 'category': 'Physical', 'base_power': 85, 'accuracy': 90, 'priority': 0, 'description': '30% chance to make the target flinch.'}, 'iceshard': {'name': 'Ice Shard', 'type': 'Ice', 'category': 'Physical', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'}, 'leafstorm': {'name': 'Leaf Storm', 'type': 'Grass', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."}, 'spiritshackle': {'name': 'Spirit Shackle', 'type': 'Ghost', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Prevents the target from switching out.'}, 'defog': {'name': 'Defog', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': '-1 evasion; ends user and target hazards/terrain.'}, 'switcheroo': {'name': 'Switcheroo', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."}, 'woodhammer': {'name': 'Wood Hammer', 'type': 'Grass', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'}, 'shellsmash': {'name': 'Shell Smash', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Def, SpD by 1; raises Atk, SpA, Spe by 2.'}, 'liquidation': {'name': 'Liquidation', 'type': 'Water', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Defense by 1."}, 'surf': {'name': 'Surf', 'type': 'Water', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dive.'}, 'hurricane': {'name': 'Hurricane', 'type': 'Flying', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to confuse target. Can't miss in rain."}, 'vacuumwave': {'name': 'Vacuum Wave', 'type': 'Fighting', 'category': 'Special', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'}, 'esperwing': {'name': 'Esper Wing', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% chance to raise user Speed by 1. High crit.'}, 'spikes': {'name': 'Spikes', 'type': 'Ground', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts grounded foes on switch-in. Max 3 layers.'}, 'leechlife': {'name': 'Leech Life', 'type': 'Bug', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'}, 'alluringvoice': {'name': 'Alluring Voice', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% confuse target that had a stat rise this turn.'}, 'mirrorcoat': {'name': 'Mirror Coat', 'type': 'Psychic', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': -5, 'description': 'If hit by special attack, returns double damage.'}, 'seismictoss': {'name': 'Seismic Toss', 'type': 'Fighting', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "Does damage equal to the user's level."}, 'softboiled': {'name': 'Soft-Boiled', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'}, 'healbell': {'name': 'Heal Bell', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Cures the user's party of all status conditions."}, 'flamethrower': {'name': 'Flamethrower', 'type': 'Fire', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to burn the target.'}, 'steameruption': {'name': 'Steam Eruption', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 95, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'}, 'haze': {'name': 'Haze', 'type': 'Ice', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Eliminates all stat changes.'}, 'icespinner': {'name': 'Ice Spinner', 'type': 'Ice', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Ends the effects of terrain.'}, 'bellydrum': {'name': 'Belly Drum', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User loses 50% max HP. Maximizes Attack.'}, 'meteorbeam': {'name': 'Meteor Beam', 'type': 'Rock', 'category': 'Special', 'base_power': 120, 'accuracy': 90, 'priority': 0, 'description': "Raises user's Sp. Atk by 1 on turn 1. Hits turn 2."}, 'nuzzle': {'name': 'Nuzzle', 'type': 'Electric', 'category': 'Physical', 'base_power': 20, 'accuracy': 100, 'priority': 0, 'description': '100% chance to paralyze the target.'}, 'grassknot': {'name': 'Grass Knot', 'type': 'Grass', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the target.'}, 'dazzlinggleam': {'name': 'Dazzling Gleam', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'}, 'moonlight': {'name': 'Moonlight', 'type': 'Fairy', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'}}

SYSTEM = '''Play Gen 9 Random Battle singles to win all six opposing Pokemon against an MCTS opponent.
We use a continuing conversation. The CURRENT snapshot always overrides older slots, HP and beliefs.
Code supplies observed facts; your earlier strategic guesses are NOT observations. Hidden coverage,
items, abilities and unrevealed species remain unknown. Infer conservatively from revealed evidence.
Choose one action per consultation. Think economically: identify the decisive threat, compare the
best attack/switch/Tera, then decide. Do not enumerate every legal move or write a long analysis.
Before committing check: priority/speed; strongest plausible enemy reply, including coverage;
survival both on entry AND next turn for a slower switch-in; what this accomplishes if they switch.
Preserve a practical answer to their strongest threat, but do not sacrifice the team preserving a
hypothetical sweeper. Avoid setup into a faster lethal hit, recovery loops, and pointless switches.
Take reliable KOs. Use Tera now when it creates a concrete win or prevents a decisive loss; avoid
saving it until no surviving Pokemon can benefit. Account for defensive Tera and priority by the foe.
The type tables are arithmetic aids, NOT damage/survival bounds. Neutral is 1, not resistance.
Levels are random-battle balanced. Request.stats are unboosted. Opposing exact stats/HP are unknown.
Protocol controls exceptions, including Illusion, Transform, Roost, grounding, move-type overrides.
Reply with a JSON object: {"action":"exact legal string","risk":"strongest plausible reply, <=18 words"}.
No strategic memo, alternative tree or repeated-action policy. The conversation preserves reasoning.
'''


def canon(who):
    return re.sub(r'^(p[12])[a-z]:', r'\1:', str(who))


def unpack(obs):
    if isinstance(obs, dict): return obs
    try:
        value = json.loads(obs)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError): return {}


def parse_answer(text, legal):
    if not isinstance(text, str): return {}
    dec = json.JSONDecoder()
    for match in re.finditer(r'\{', text):
        try:
            obj, _ = dec.raw_decode(text[match.start():])
            if isinstance(obj, dict) and isinstance(obj.get('action'), str) and obj['action'] in legal:
                return {'action': obj['action'], 'risk': str(obj.get('risk', ''))[:250]}
        except (ValueError, TypeError, RecursionError, OverflowError): pass
    for line in text.splitlines():
        cleaned = line.strip().strip('`').strip()
        if cleaned in legal: return {'action': cleaned, 'risk': ''}
    # Handles terse prose, but never an ambiguous response with competing actions.
    matches = re.findall(r'\b(?:move [1-4](?: terastallize)?|switch [1-6])\b', text)
    valid = set(a for a in matches if a in legal)
    return {'action': valid.pop(), 'risk': ''} if len(valid) == 1 else {}


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.session = None
        self.sent_dex = {}
        self.sent_species = {}
        self.public_facts = {}
        self.history = []
        self.evidence = []
        self.failures = []
        self.bad = set()
        self.last_progress = {}
        self.steps = 0
        self.calls = 0
        self.session_calls = 0
        self.latencies = []
        self.pending_protocol = []
        self.prev_signature = None
        self.rejections = 0

    def reset_session(self, reason):
        self.session = None
        self.session_calls = 0
        self.sent_dex = {}
        self.sent_species = {}
        self.ctx.log('session_reset', reason=reason)

    def ingest(self, raw):
        obs = copy.deepcopy(unpack(raw))
        events = obs.get('recent_events') or []
        # Deduplicate observations, not identical protocol events within different turns.
        sig = dump([obs.get('turn'), obs.get('request'), events, obs.get('last_error')])
        if sig != self.prev_signature:
            self.prev_signature = sig
            self.pending_protocol.extend(events)
            self.pending_protocol = self.pending_protocol[-350:]
            for line in events:
                bits = line.split('|')
                if len(bits) < 4: continue
                kind, owner = bits[1], canon(bits[2])
                if not owner.startswith(('p1:', 'p2:')): continue
                rec = self.public_facts.setdefault(owner, {})
                if kind == '-ability': rec['ability'] = bits[3]
                if kind == '-item': rec['item'] = bits[3]
                if kind == '-enditem': rec['item'] = ''
                if kind == '-terastallize': rec['teraTypeRevealed'] = bits[3]
                # Passive item and ability lines can name another Pokemon with [of].
                attributed = next((canon(b[5:]) for b in bits[4:] if b.startswith('[of] ')), owner)
                if attributed.startswith(('p1:', 'p2:')):
                    target = self.public_facts.setdefault(attributed, {})
                    for b in bits[4:]:
                        if b.startswith('[from] ability: '): target['ability'] = b[16:]
                        if b.startswith('[from] item: '): target['item'] = b[13:]
        records = list((obs.get('public_summary') or {}).get('pokemon', {}).values())
        records += obs.get('opponent_revealed') or []
        for p in records:
            # Protocol-derived facts correct summary omissions, but never own exact request data.
            p.update(self.public_facts.get(p.get('ident'), {}))
        return obs

    def legal(self, obs):
        return [a for a in obs.get('legal_actions', [])
                if isinstance(a, str) and a != 'noop' and a not in self.bad]

    def type_facts(self, c):
        rows = []
        # Exact *type-chart* multipliers. Ability, item and field effects explicitly separate.
        for slot, p in enumerate(c.team, 1):
            if fraction(p) <= 0: continue
            ts = c.types(p)
            row = {'slot':slot, 'types':ts,
                   'type_weaknesses':{t:effectiveness(t,ts) for t in CHART if effectiveness(t,ts)>1},
                   'type_resists_or_immunities':{t:effectiveness(t,ts) for t in CHART if effectiveness(t,ts)<1}}
            if p.get('active') and p.get('teraType') and not c.tera(p):
                t = p['teraType']
                if t != 'Stellar':
                    row['after_tera']={'type':t, 'weaknesses':{a:effectiveness(a,[t]) for a in CHART if effectiveness(a,[t])>1},
                                       'resists_or_immunities':{a:effectiveness(a,[t]) for a in CHART if effectiveness(a,[t])<1}}
            rows.append(row)
        attacks=[]
        active=(c.req.get('active') or [{}])[0]
        for i, m in enumerate(active.get('moves', []),1):
            md=c.moves.get(m.get('id'),{})
            if md.get('category') not in ('Physical','Special'): continue
            typ=md.get('type')
            if m.get('id')=='terablast' and c.tera(c.own): typ=c.tera(c.own)
            # Multipliers for special typing are left unknown to avoid false certainty.
            variable = m.get('id') in ('terastarstorm','terablast','weatherball','terrainpulse','revelationdance','judgment','ragingbull')
            row={'move_slot':i,'move':m.get('id'),'type':typ,'priority':md.get('priority',0)}
            if not variable and c.types(c.foe): row['type_chart_vs_current_foe']=effectiveness(typ,c.types(c.foe))
            else: row['note']='check dynamic type and target in protocol'
            attacks.append(row)
        return {'defensive_type_chart_only':rows,'attacking_type_chart_only':attacks,
                'warning':'NOT damage. Ability/item/field/temporary typing exceptions must be checked; unlisted types are neutral.'}

    def prompt(self, obs, legal):
        c=Combat(obs)
        first=self.session is None
        # Public move facts only: own moves and moves already revealed by the opponent.
        known=dict(c.moves)
        for p in obs.get('opponent_revealed') or []:
            for move in p.get('moves') or []:
                mid=ident(move)
                if mid in PUBLIC_MOVES and mid not in known: known[mid]=PUBLIC_MOVES[mid]
        changed_moves={k:v for k,v in known.items() if first or self.sent_dex.get(k)!=v}
        changed_species={k:v for k,v in c.dex.items() if first or self.sent_species.get(k)!=v}
        own=[]
        for i,p in enumerate(c.team,1):
            row={k:v for k,v in p.items() if k not in ('pokeball','commanding','baseAbility')}
            row['slot']=i
            # Include exact current volatile/boost state but avoid duplicate base identity/HP.
            row['effects']={k:v for k,v in c.public(p).items() if k in ('boosts','volatiles','status')}
            own.append(row)
        state={'turn':obs.get('turn'),'active':obs.get('active'),
               'request_flags':{k:v for k,v in c.req.items() if k!='side'},
               'own_team_CURRENT_SLOTS':own,'foe_current':c.foe,
               'opponent_revealed':obs.get('opponent_revealed') or [],
               'field':{k:v for k,v in c.pub.items() if k!='pokemon'},
               'legal_actions':legal,'last_error':obs.get('last_error'),
               'type_facts':self.type_facts(c),'failure_audit':self.failures,
               'progress':self.ctx.progress(),
               'budget':self.ctx.budget.remaining()}
        parts=['NEW SESSION CHECKPOINT' if first else 'NEXT DECISION: full dynamic snapshot, static rules below are deltas.',
               'AUTHORITATIVE EVENTS SINCE LAST CONSULTATION:\n'+'\n'.join(self.pending_protocol),
               'CURRENT='+dump(state)]
        if changed_moves: parts.append('MOVE_RULES_ADD_OR_REPLACE='+dump(changed_moves))
        if changed_species: parts.append('SPECIES_DATA_ADD_OR_REPLACE='+dump(changed_species))
        if first:
            parts.append('OBSERVED RECENT DECISIONS (facts, not strategic claims)='+dump(self.history[-14:]))
            parts.append('PAST DAMAGE/ORDER EVIDENCE (context-specific, not a speed or damage guarantee)='+dump(self.evidence[-20:]))
        elif self.history:
            parts.append('LAST OUTCOME='+dump(self.history[-1]))
        parts.append('Choose now. Output just action and <=18-word risk. Use the current slots.')
        return '\n'.join(parts),known,c.dex

    def verify(self, before, action, after, source, risk):
        cb,ca=Combat(before),Combat(after)
        progress=self.ctx.progress()
        events=after.get('recent_events') or []
        fail=[]
        if after.get('last_error') or progress.get('illegal_actions',0)>self.last_progress.get('illegal_actions',0):
            fail.append('Rejected action: '+str(after.get('last_error')))
        if progress.get('since_obs_change',0)>=2: fail.append('No observation change: previous action failed to advance.')
        if progress.get('since_score_change',0)>=4: fail.append('Four actions without HP-score change: verify useful progress.')
        conditions={p.get('ident'):fraction(p) for p in list(cb.records.values())+cb.team}
        order=[]; packets=[]; move=None; critical=False
        for event in events:
            b=event.split('|')
            if len(b)<3: continue
            owner=canon(b[2]); kind=b[1]
            if kind=='move' and len(b)>3:
                move={'actor':owner,'move':b[3]}; critical=False
                order.append([owner,b[3]])
            elif kind in ('switch','drag') and len(b)>4:
                conditions[owner]=fraction({'condition':b[4]}); move=None
            elif kind=='-crit': critical=True
            elif kind in ('-damage','-heal') and len(b)>3:
                now=fraction({'condition':b[3]})
                if kind=='-damage' and move and owner!=move['actor'] and owner in conditions and not any('[from]' in x for x in b[4:]):
                    loss=max(0,conditions[owner]-now)
                    if loss:
                        packets.append({**move,'target':owner,'loss_maxhp_pct':round(loss*100,1),
                                        'crit':critical,'faint_censored':'fnt' in b[3],
                                        'boosts_before':{p.get('ident'):cb.public(p).get('boosts',{}) for p in (cb.own,cb.foe)}})
                conditions[owner]=now
            if kind in ('-fail','-immune','cant','-miss'):
                fail.append('A move failed/immune/missed; identify actor and re-evaluate: '+event)
        ownid=cb.own.get('ident'); enemyid=cb.foe.get('ident')
        after_own=next((p for p in ca.team if p.get('ident')==ownid),ca.records.get(ownid,{}))
        after_foe=ca.records.get(enemyid,{})
        if after_own and fraction(cb.own)>0 and fraction(after_own)==0:
            fail.append('Our active fainted: previous survival/coverage belief may be false; plan revenge or defensive response.')
        if any(p['target']==ownid and p['loss_maxhp_pct']>=30 for p in packets):
            fail.append('Large observed incoming damage; do not reuse a previous survival claim.')
        if cb.foe.get('ident')==ca.foe.get('ident') and cb.foe.get('boosts')!=ca.foe.get('boosts'):
            fail.append('Enemy boosts changed: recompute threat; old damage samples do not bound new damage.')
        row={'turn':before.get('turn'),'action':action,'source':source,
             'own':ownid,'foe':enemyid,
             'before_hp':[round(fraction(cb.own),3),round(fraction(cb.foe),3)],
             'same_pokemon_after_hp':[round(fraction(after_own),3),round(fraction(after_foe),3)],
             'actives_after':[ca.own.get('ident'),ca.foe.get('ident')],
             'move_order_not_speed_proof':order,'damage_packets':packets}
        self.history.append(row); self.history=self.history[-24:]
        if packets or len(order)>1:
            self.evidence.append({'turn':before.get('turn'),'order':order,'hits':packets,
                                  'weather':cb.pub.get('weather'),'field':cb.pub.get('field')})
            self.evidence=self.evidence[-24:]
        # Detect recovery/switch cycles even if HP and score change on every action.
        recent=self.history[-4:]
        if len(recent)==4 and len({(r['own'],r['foe']) for r in recent})==1:
            if recent[-1]['same_pokemon_after_hp'][1]>=recent[0]['before_hp'][1]-.03:
                fail.append('Four-decision matchup cycle with no net enemy HP progress; change the plan or explicitly justify the PP/endgame trade.')
        self.failures=fail[-8:]
        self.last_progress=progress
        self.ctx.log('verified_action', **row, risk=risk, failures=self.failures)

    def fallback(self, obs, legal):
        # Emergency only. Reference arithmetic is never presented as factual damage to Astra.
        c=Combat(obs); rows=[]
        active=(c.req.get('active') or [{}])[0]
        for action in legal:
            bits=action.split(); n=int(bits[1])-1
            value=-1000
            if bits[0]=='move' and n<len(active.get('moves',[])):
                m=active['moves'][n]; mid=m.get('id'); md=c.moves.get(mid,{})
                dmg=c.damage(c.own,c.foe,md,tera=c.own.get('teraType') if 'terastallize' in action else None)
                if dmg:
                    acc=md.get('accuracy',100); acc=1 if acc is True else float(acc or 100)/100
                    value=min(dmg[0],100*fraction(c.foe)+10)*acc
                    if dmg[0]>=100*fraction(c.foe)+1: value+=45
                    if md.get('priority',0)>0 and fraction(c.own)<.4: value+=15
                    if dmg[1]==0: value=-100
                else:
                    value=3
                    if mid in ('recover','roost','slackoff','softboiled','milkdrink','shoreup','moonlight','morningsun','synthesis') and fraction(c.own)<.5: value=30
                    if mid=='sleeptalk' and 'slp' in c.own.get('condition',''): value=45
                    if mid in ('stealthrock','spikes','toxicspikes','stickyweb'):
                        side=ident(dump(c.pub.get('side_conditions',{}).get('p2',{})))
                        value=16 if mid not in side else -10
                if 'terastallize' in action: value-=10
            elif bits[0]=='switch' and n<len(c.team):
                p=c.team[n]; attacks=[]
                for mid in p.get('moves',[]):
                    d=c.damage(p,c.foe,c.moves.get(mid,{}))
                    if d: attacks.append(d[0])
                incoming=[]
                for mid in c.foe.get('moves',[]):
                    md=c.moves.get(ident(mid), PUBLIC_MOVES.get(ident(mid),{}))
                    d=c.damage(c.foe,p,md)
                    if d: incoming.append(d[1])
                for t in c.attack_types(c.foe):
                    for category in ('Physical','Special'):
                        d=c.damage(c.foe,p,{'name':'probe','type':t,'category':category,'base_power':90})
                        if d: incoming.append(d[1])
                value=min(100,max(attacks or [0]))*.5+fraction(p)*20-max(incoming or [30])*.6-c.hazards(p)-25
                if p.get('reviving'): value=30+sum(p.get('stats',{}).values())/100
            rows.append((value,action))
        return max(rows)[1] if rows else None

    def run_episode(self, game):
        ctx=self.ctx
        obs=self.ingest(ctx.observe())
        while not ctx.done and not obs.get('done') and self.steps<220:
            legal=self.legal(obs)
            if not legal: return
            budget=ctx.budget.remaining()
            if budget.get('wall_s',0)<=3: return
            action=None; risk=''; source='fallback'
            if len(legal)==1:
                action=legal[0]; source='only_legal'
            # Bounded session checkpoints avoid growing an entire battle's duplicated states.
            if self.session_calls>=10: self.reset_session('ten-consultation checkpoint')
            if action is None:
                # Do not abandon reasoning with five minutes left. Typical latency is 56s;
                # a 120s admission reserve permits a normal call and an engine finish.
                # The call timeout itself always exceeds the worst measured 236s.
                may_call=budget.get('system2_calls',0)>1 and budget.get('wall_s',0)>120
                if may_call:
                    prompt, known, species_data=self.prompt(obs,legal)
                    started=ctx.clock.now()
                    self.calls+=1
                    try:
                        result=ctx.think(prompt,session_id=self.session,system=SYSTEM,timeout_s=300.0)
                        self.latencies.append(ctx.clock.now()-started)
                        reply=parse_answer(result.text,legal)
                        action=reply.get('action'); risk=reply.get('risk','')
                        if action:
                            self.session=getattr(result,'session_id',None)
                            self.session_calls+=1
                            self.sent_dex=copy.deepcopy(known)
                            self.sent_species=copy.deepcopy(species_data)
                            self.pending_protocol=[]
                            source='astra'
                        else:
                            self.failures=['Malformed/illegal model reply. Emergency action then fresh consultation.']
                            self.reset_session('invalid model reply')
                    except (BudgetExceeded,ModelUnavailable): return
                    except (TimeoutError,RuntimeError):
                        self.failures=['Model call failed. Use emergency action, then replan with authoritative evidence.']
                        self.reset_session('failed model call')
                if ctx.done: return
                if action is None: action=self.fallback(obs,legal)
            if not action: return
            ctx.log('decision',action=action,source=source,risk=risk,session_calls=self.session_calls)
            old=obs
            try:
                out=ctx.act(action)
                obs=self.ingest(out if out is not None else ctx.observe())
            except (BudgetExceeded,ModelUnavailable): return
            except Exception as exc:
                if ctx.done: return
                self.bad.add(action); self.rejections+=1; self.steps+=1
                self.failures=['Action rejected '+action+': '+str(exc)[:250]]
                self.reset_session('action exception')
                obs=self.ingest(ctx.observe())
                if self.rejections>=8: return
                continue
            self.steps+=1
            self.verify(old,action,obs,source,risk)
            if obs.get('last_error'):
                self.bad.add(action); self.rejections+=1
                self.reset_session('engine rejected action')
            else:
                self.bad.clear(); self.rejections=0
            ctx.memory['observed_history']=self.history
            if self.rejections>=8: return


def build(ctx):
    return Harness(ctx)
