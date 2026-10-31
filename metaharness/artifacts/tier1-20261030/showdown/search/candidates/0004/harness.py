"""Evidence-first battle reports with checked, short tactical continuations.
Only observations and ctx are used; no engine, team database or seed access.
"""
from __future__ import annotations
import json
import re
from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def ident(s):
    return re.sub(r'[^a-z0-9]', '', str(s).lower())


def dumps(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':'))


def hp(p):
    s = str(p.get('condition', '0'))
    try:
        if 'fnt' in s:
            return 0.0
        a = s.split()[0].split('/')
        return float(a[0]) / float(a[1]) if len(a) == 2 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def team(o):
    return (o.get('request') or {}).get('side', {}).get('pokemon', [])


def me(o):
    return next((p for p in team(o) if p.get('active')), {})


def foe(o):
    key = o.get('active', {}).get('p2', '')
    return o.get('public_summary', {}).get('pokemon', {}).get(key, next(
        (p for p in o.get('opponent_revealed', []) if p.get('ident') == key), {}))


def moves(o):
    a = (o.get('request') or {}).get('active') or [{}]
    return a[0].get('moves', [])


def species(p):
    return p.get('details', '').split(',')[0]


def dex(o, p):
    return o.get('species_dex', {}).get(species(p), {})


# Gen 9 type chart: attack type -> super effective / resisted / immune.
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
    strong, weak, immune = [s.split() for s in CHART.get(t, ('', '', ''))]
    n = 1.0
    for d in types:
        n *= 0 if d in immune else 2 if d in strong else .5 if d in weak else 1
    return n


def types(o, p, tera=None):
    t = tera or p.get('terastallized') or p.get('teraTypeRevealed') or p.get('tera_type')
    if not t:
        m = re.search(r'tera:([A-Za-z]+)', p.get('details', ''))
        t = m.group(1) if m else None
    if t and isinstance(t, str) and t != 'Stellar':
        return [t]
    return dex(o, p).get('types', [])


def stage(n):
    n = max(-6, min(6, n))
    return (2 + n) / 2 if n >= 0 else 2 / (2 - n)


def pub(o, p):
    return o.get('public_summary', {}).get('pokemon', {}).get(p.get('ident'), {})


def speed(o, p):
    b = pub(o, p).get('boosts', p.get('boosts', {}))
    value = p.get('stats', {}).get('spe')
    if value is None:
        return None
    value *= stage(b.get('spe', 0))
    if ' par' in p.get('condition', ''):
        value *= .5
    if ident(p.get('item')) == 'choicescarf':
        value *= 1.5
    return round(value, 1)


def power_hint(o, p, q, move, tera=None):
    """A transparent ranking aid, NOT a damage calculator or KO guarantee."""
    d = o.get('move_dex', {}).get(ident(move), {})
    if d.get('category') == 'Status':
        return 0, 1
    t = d.get('type', 'Normal')
    cat = d.get('category', 'Physical')
    tr = tera or p.get('terastallized')
    if ident(move) == 'terablast' and tr:
        t = tr
        cat = 'Physical' if p.get('stats', {}).get('atk', 0) > p.get('stats', {}).get('spa', 0) else 'Special'
    bp = d.get('base_power', 0) or 70
    if ident(move) in ('seismictoss', 'nightshade'):
        bp = 65
    eff = effectiveness(t, types(o, q))
    ability = ident(q.get('ability'))
    for line in o.get('recent_events', []):
        bits = line.split('|')
        if len(bits) > 3 and re.sub(r'^(p[12])[a-z]:', r'\1:', bits[2]) == q.get('ident'):
            for bit in bits[3:]:
                if bit.startswith('[from] ability: '):
                    ability = ident(bit.removeprefix('[from] ability: '))
    ownability = ident(p.get('ability'))
    if ownability not in ('moldbreaker', 'teravolt', 'turboblaze'):
        immunities = {'Ground': ('levitate', 'eartheater'), 'Water': ('waterabsorb', 'stormdrain', 'dryskin'),
                      'Fire': ('flashfire', 'wellbakedbody'), 'Electric': ('voltabsorb', 'lightningrod', 'motordrive'),
                      'Grass': ('sapsipper',)}
        if ability in immunities.get(t, ()):
            eff = 0
        if ability == 'wonderguard' and eff <= 1:
            eff = 0
    orig = dex(o, p).get('types', [])
    stab = 1.5 if t in orig else 1
    if tr and t == tr:
        stab = 2 if t in orig else 1.5
    stat = 'atk' if cat == 'Physical' else 'spa'
    boosts = pub(o, p).get('boosts', p.get('boosts', {}))
    atk = p.get('stats', {}).get(stat, 180) * stage(boosts.get(stat, 0))
    if ability == 'unaware':
        atk = p.get('stats', {}).get(stat, 180)
    defense = 'def' if cat == 'Physical' or ident(move) in ('psyshock', 'psystrike', 'secretsword') else 'spd'
    bs = dex(o, q).get('base_stats', {}).get(defense, 90)
    # Neutral level-balanced random-set estimate, explicitly uncertain in prompts.
    level = re.search(r'\bL(\d+)', q.get('details', ''))
    level = int(level.group(1)) if level else 80
    df = ((2 * bs + 52) * level / 100 + 5) * stage(q.get('boosts', {}).get(defense, 0))
    if ownability == 'unaware':
        df = ((2 * bs + 52) * level / 100 + 5)
    if cat == 'Physical' and ' brn' in p.get('condition', '') and ownability != 'guts':
        atk *= .5
    if ident(p.get('item')) == ('choiceband' if cat == 'Physical' else 'choicespecs'):
        atk *= 1.5
    acc = d.get('accuracy', 100)
    acc = 1 if acc is True else float(acc or 100) / 100
    return round(bp * stab * eff * atk / max(1, df) * acc, 1), eff


def entry_hint(o, p):
    sc = o.get('public_summary', {}).get('side_conditions', {}).get('p1', {})
    keys = {ident(k.removeprefix('move: ')): v for k, v in sc.items()}
    if ident(p.get('item')) == 'heavydutyboots' or ident(p.get('ability')) == 'magicguard':
        return 0
    loss = 0
    if 'stealthrock' in keys:
        loss += .125 * effectiveness('Rock', types(o, p))
    if 'Flying' not in types(o, p) and ident(p.get('ability')) != 'levitate' and 'spikes' in keys:
        v = keys['spikes']
        layers = v.get('layers', 1) if isinstance(v, dict) else v if isinstance(v, int) else 1
        loss += {1: .125, 2: 1/6, 3: .25}.get(layers, .125)
    return round(loss, 3)


def fallback(o, banned=()):
    legal = [x for x in o.get('legal_actions', []) if x != 'noop' and x not in banned]
    if not legal:
        legal = [x for x in o.get('legal_actions', []) if x != 'noop']
    if not legal:
        return None
    p, q = me(o), foe(o)
    rows = []
    for a in legal:
        score = -1000
        if a.startswith('move '):
            i = int(a.split()[1]) - 1
            mm = moves(o)
            if i < len(mm):
                mid = mm[i].get('id', mm[i].get('move', ''))
                score, eff = power_hint(o, p, q, mid, p.get('teraType') if 'terastallize' in a else None)
                if eff == 0:
                    score = -250
                if 'terastallize' in a:
                    score = score * .85 - 15
                if ident(mid) in ('recover', 'roost', 'slackoff', 'softboiled', 'milkdrink', 'shoreup', 'synthesis', 'moonlight', 'morningsun'):
                    score = 120 if hp(p) < .4 else -10
        elif a.startswith('switch '):
            i = int(a.split()[1]) - 1
            if i < len(team(o)):
                pp = team(o)[i]
                if pp.get('reviving') or any(x.get('reviving') for x in team(o)):
                    score = max(pp.get('stats', {}).values(), default=100)
                else:
                    options = [power_hint(o, pp, q, m)[0] for m in pp.get('moves', [])]
                    weakness = max([effectiveness(t, types(o, pp)) for t in types(o, q)] or [1])
                    score = (max(options or [1]) * .4 + (hp(pp)-entry_hint(o, pp))*40 - weakness*18 - 45)
        rows.append((score, a))
    return max(rows)[1]


class EvidenceHarness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.memo = ''
        self.history = []
        self.damage = []
        self.reveals = {}
        self.plan = None
        self.failure = ''
        self.last_illegal = 0
        self.banned = set()
        self.latencies = []
        self.calls = 0

    def ingest_reveals(self, events):
        # These are public protocol facts, including passive item reveals omitted by summaries.
        for line in events:
            s = line.split('|')
            if len(s) < 4 or not re.match(r'p[12][a-z]?:', s[2]):
                continue
            who = re.sub(r'^(p[12])[a-z]:', r'\1:', s[2])
            facts = self.reveals.setdefault(who, {})
            if s[1] in ('-item', '-enditem'):
                facts['item'] = s[3]
                facts['item_state'] = 'held' if s[1] == '-item' else 'removed_or_consumed'
            elif s[1] == '-ability':
                facts['ability'] = s[3]
            elif s[1] == '-terastallize':
                facts['tera'] = s[3]
            for field in s[3:]:
                if field.startswith('[from] ability: ') and not any(x.startswith('[of]') for x in s[3:]):
                    facts['ability'] = field.removeprefix('[from] ability: ')
                if field.startswith('[from] item: ') and not any(x.startswith('[of]') for x in s[4:]):
                    facts['item'] = field.removeprefix('[from] item: ')
                    facts['item_state'] = 'observed_active'
            if not facts:
                self.reveals.pop(who, None)

    def report(self, o):
        p, q = me(o), foe(o)
        ps = o.get('public_summary', {})
        lines = ['CURRENT REQUEST T' + str(o.get('turn')),
                 'MOST RECENT AUTHORITATIVE EVENTS:\n' + '\n'.join(o.get('recent_events', [])),
                 'PERSISTENT PUBLIC REVEALS=' + dumps(self.reveals),
                 'legal_actions=' + dumps(o.get('legal_actions', [])),
                 'request_flags=' + dumps({k: v for k, v in (o.get('request') or {}).items() if k != 'side'}),
                 'ACTIVE=' + dumps(o.get('active', {})),
                 'FIELD=' + dumps({k: v for k, v in ps.items() if k != 'pokemon'}),
                 'OPPONENT REVEALED (others unknown)=' + dumps(o.get('opponent_revealed', []))]
        for i, mon in enumerate(team(o), 1):
            row = {k: v for k, v in mon.items() if k not in ('pokeball', 'commanding', 'baseAbility')}
            row.update(slot=i, species_data=dex(o, mon), current_types=types(o, mon),
                       public_effects=pub(o, mon), speed_before_weather_terrain_ability=speed(o, mon),
                       entry_hp_fraction_approx=entry_hint(o, mon))
            row['attack_hints_vs_current_foe'] = {m: power_hint(o, mon, q, m) for m in mon.get('moves', [])
                if o.get('move_dex', {}).get(m, {}).get('category') != 'Status'}
            lines.append('OWN ' + dumps(row))
        lines.append('REVEALED SPECIES DATA=' + dumps({species(x): dex(o, x) for x in o.get('opponent_revealed', [])}))
        lines.append('Attack hint arrays are [rough power ranking, type/known-immunity multiplier]. Rankings are NOT HP damage. Entry hints omit unusual grounding effects. The raw field and protocol override all helpers.')
        lines.append('MOVE RULES=' + dumps(o.get('move_dex', {})))
        lines.append('OBSERVED DAMAGE EXAMPLES (context-specific, NOT guarantees)=' + dumps(self.damage[-14:]))
        lines.append('RECENT DECISIONS/OUTCOMES=' + dumps(self.history[-6:]))
        lines.append('STRATEGIC MEMORY (recheck against current facts)=' + self.memo)
        lines.append('REPLAN SIGNAL=' + self.failure + '; last_error=' + str(o.get('last_error')))
        lines.append('PROGRESS=' + dumps(self.ctx.progress()))
        lines.append('REMAINING=' + dumps(self.ctx.budget.remaining()))
        return '\n'.join(lines)

    def record(self, old, action, new, source):
        """Record consequences, including direct hit amounts before residual recovery."""
        self.ingest_reveals(new.get('recent_events', []))
        hps = {}
        for x in list(team(old)) + old.get('opponent_revealed', []):
            hps[x.get('ident', '')] = hp(x)
        lastmove = None
        critical = False
        order = []
        for line in new.get('recent_events', []):
            s = line.split('|')
            if len(s) < 3:
                continue
            kind = s[1]
            who = re.sub(r'^(p[12])[a-z]:', r'\1:', s[2])
            if kind in ('move', '-anim') and len(s) >= 5:
                lastmove = [who, s[3], re.sub(r'^(p[12])[a-z]:', r'\1:', s[4])]
                critical = False
                if kind == 'move':
                    order.append(s[2] + ':' + s[3])
            elif kind == '-crit':
                critical = True
            elif kind in ('switch', 'drag') and len(s) >= 5:
                hps[who] = hp({'condition': s[4]})
                lastmove = None
            elif kind in ('-damage', '-heal') and len(s) >= 4:
                now = hp({'condition': s[3]})
                if kind == '-damage' and lastmove and who == lastmove[2] and '[from]' not in line and who in hps:
                    delta = hps[who] - now
                    if delta > 0:
                        self.damage.append({'turn': old.get('turn'), 'attacker': lastmove[0],
                                            'move': lastmove[1], 'target': who, 'damage_pct': round(delta*100, 1),
                                            'critical': critical, 'faint_censored': now == 0,
                                            'pre_decision_boosts': {k: v.get('boosts', {}) for k, v in old.get('public_summary', {}).get('pokemon', {}).items() if k in (who, lastmove[0])},
                                            'note': 'Recheck turn effects, forms, Tera and items; percent is of max HP'})
                hps[who] = now
        row = {'turn': old.get('turn'), 'action': action, 'source': source,
               'own': species(me(old)), 'foe': species(foe(old)),
               'own_after': species(me(new)), 'foe_after': species(foe(new)),
               'own_hp_before': round(hp(me(old)), 3), 'own_hp_after': round(hp(me(new)), 3),
               'foe_hp_before': round(hp(foe(old)), 3), 'foe_hp_after': round(hp(foe(new)), 3),
               'move_order_not_speed_proof': order}
        self.history.append(row)
        self.damage = self.damage[-30:]
        self.history = self.history[-12:]
        pr = self.ctx.progress()
        reasons = []
        if new.get('last_error') or pr.get('illegal_actions', 0) > self.last_illegal:
            reasons.append('Action rejected; abandon plan and choose from updated request')
            self.banned.add(action)
        else:
            self.banned.clear()
        self.last_illegal = pr.get('illegal_actions', 0)
        if pr.get('since_obs_change', 0) >= 2 or pr.get('since_score_change', 0) >= 4:
            reasons.append('No recent measurable progress; reassess setup/switch/recovery cycle')
        if me(old).get('ident') == me(new).get('ident') and hp(me(old))-hp(me(new)) > .35:
            reasons.append('Lost >35% HP in one decision: survival assumption needs review')
        if any('|-fail|' in x or '|-immune|' in x or '|-miss|' in x for x in new.get('recent_events', [])):
            reasons.append('Move failed/missed/met immunity; do not blindly continue')
        self.failure = '; '.join(reasons)
        self.ctx.memory['battle_plan'] = self.memo
        self.ctx.log('verified_action', **row, replan=self.failure)

    def make_plan(self, o, action, data):
        self.plan = None
        if not isinstance(data, dict):
            return
        target = me(o)
        if action.startswith('switch '):
            i = int(action.split()[1])-1
            if i < len(team(o)):
                target = team(o)[i]
        follow = data.get('followup')
        if isinstance(follow, dict) and follow.get('move') and ident(follow.get('foe')) == ident(species(foe(o))):
            try:
                minimum = max(.5, min(1., float(follow.get('min_hp', .65))))
            except (TypeError, ValueError):
                minimum = .65
            self.plan = {'mid': ident(follow['move']), 'remaining': 1, 'minhp': minimum, 'follow': True}
        else:
            try:
                repeat = max(0, min(2, int(data.get('repeat', 0))))
            except (TypeError, ValueError):
                repeat = 0
            if repeat and action.startswith('move '):
                i = int(action.split()[1])-1
                if i < len(moves(o)):
                    mid = moves(o)[i].get('id', '')
                    if o.get('move_dex', {}).get(mid, {}).get('category') in ('Physical', 'Special'):
                        self.plan = {'mid': mid, 'remaining': repeat, 'minhp': .35, 'follow': False}
        if self.plan:
            self.plan.update(own=target.get('ident'), foe=foe(o).get('ident'), before=o)

    def continuation(self, o):
        pl = self.plan
        if not pl or pl['remaining'] <= 0:
            return None
        old = pl['before']
        reason = ''
        if self.failure:
            reason = self.failure
        elif me(o).get('ident') != pl['own'] or foe(o).get('ident') != pl['foe']:
            reason = 'Active matchup changed'
        elif hp(me(o)) < pl['minhp'] or (o.get('request') or {}).get('forceSwitch'):
            reason = 'Continuation HP/request guard failed'
        elif dumps(foe(o).get('moves', [])) != dumps(foe(old).get('moves', [])):
            reason = 'New opponent move revealed'
        else:
            # Ordinary damage and existing passive recovery are expected. Any other mechanic/reveal
            # requires a new decision, including a new setup boost or switch. Followup allows our
            # intended switch/setup; the opponent's changed state never gets this exception.
            for line in o.get('recent_events', []):
                s = line.split('|')
                k = s[1] if len(s) > 1 else ''
                benign = ('', 'move', '-damage', '-heal', '-resisted', '-supereffective', 'upkeep', 'turn')
                if k in benign:
                    continue
                if pl['follow'] and len(s) > 2 and s[2].startswith('p1') and k in ('switch', '-boost'):
                    continue
                reason = 'Changed mechanics/reveal: ' + line
                break
        if not reason and not pl['follow']:
            lost = max(0, hp(me(old))-hp(me(o)))
            # Recovery/drain can hide a lethal next hit in the net HP delta.
            incoming = sum(x['damage_pct']/100 for x in self.damage
                           if x['turn'] == old.get('turn') and x['target'] == me(o).get('ident'))
            lost = max(lost, incoming)
            if hp(foe(old)) - hp(foe(o)) < .03:
                reason = 'Authorized attack did not make sufficient net progress'
            elif hp(me(o)) <= lost*1.5 + .12:
                reason = 'Another observed hit may KO us'
        found = None
        for i, m in enumerate(moves(o), 1):
            if ident(m.get('id', m.get('move'))) == pl['mid'] and f'move {i}' in o.get('legal_actions', []):
                found = f'move {i}'
        if reason or not found:
            self.failure = reason or 'Planned move is unavailable'
            self.plan = None
            self.ctx.log('plan_invalidated', reason=self.failure)
            return None
        pl['remaining'] -= 1
        pl['before'] = o
        return found

    def parse(self, text, legal):
        data = {}
        dec = json.JSONDecoder()
        for match in re.finditer(r'\{', str(text)):
            try:
                obj, _ = dec.raw_decode(str(text)[match.start():])
                if isinstance(obj, dict) and 'action' in obj:
                    data = obj
                    break
            except (ValueError, TypeError):
                pass
        action = str(data.get('action', '')).strip().lower()
        if action not in legal:
            found = re.findall(r'\b(?:move\s+[1-4](?:\s+terastallize)?|switch\s+[1-6])\b', str(text).lower())
            action = next((' '.join(x.split()) for x in found if ' '.join(x.split()) in legal), None)
        return action, data

# The parent ledger and transparent type hints remain above; the policy below
# replaces its explicit-repeat control loop with observation-checked exchanges.
SYSTEM = """You play Gen 9 Random Battle singles against a strong search opponent.
Choose a legal action to win the whole battle. Exact own stats, public events and current slots
are authoritative. Hidden coverage/items/abilities are unknown, not absent. Analyze the strongest
credible opposing coverage move, not only its last attack or STAB. A voluntary switch takes a hit;
a forced replacement after fainting normally does not. Assess the next turn too. Preserve the
specific remaining answer to their sweeper. Use defensive Tera BEFORE that answer is lost.
Compare a reliable attack, a safe switch, and Tera/setup/recovery when useful. For a switch, check
all REVEALED attacks against the incoming Pokemon. For setup, check plausible lethal coverage.
For a finishing attack, check speed, priority, accuracy, immunity, possible Tera and known pivots.
A low-HP faster foe often needs priority, not a stronger attack that never executes.
Power hints are only relative rankings, NOT damage or KO guarantees. Public damage examples are
context-specific; boosts, weather, types, items and abilities can invalidate them. Use actual
move descriptions and observed damage. Don't invent a KO when observed damage is insufficient.
Code may continue the chosen damaging move for at most TWO decisions if actual damage demonstrates
a safe short finish and all mechanical guards pass. This is an empirical policy, not an assumption
that the opponent has no hidden coverage. Opposing switches, boosts and failed effects force re-planning.
Output ONLY compact JSON, <=110 words:
{"action":"move 1","why":"brief concrete enemy threat and why this action survives or trades well",
"memory":"<=45 words: win condition, indispensable checks, Tera plan, uncertain threat",
"followup":null}
On a FORCED replacement, optionally give followup={"move":"moveid","foe":"Species","min_hp":0.7}
for ONE attack you already intend after entry. It is checked against entry effects and current slots.
Never give a followup after voluntary switching. Action must be in legal_actions. No long analysis.
"""


# Static public MOVE rules observed in the archive, never opponent sets.
# Current observation data overrides these entries. Unknown moves stay unknown.
PUBLIC_MOVE_RULES = {
    'acrobatics': {'name': 'Acrobatics', 'type': 'Flying', 'category': 'Physical', 'base_power': 55, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if the user has no held item.'},
    'alluringvoice': {'name': 'Alluring Voice', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% confuse target that had a stat rise this turn.'},
    'bellydrum': {'name': 'Belly Drum', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User loses 50% max HP. Maximizes Attack.'},
    'bodypress': {'name': 'Body Press', 'type': 'Fighting', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "Uses user's Def stat as Atk in damage calculation."},
    'bodyslam': {'name': 'Body Slam', 'type': 'Normal', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': '30% chance to paralyze the target.'},
    'bravebird': {'name': 'Brave Bird', 'type': 'Flying', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'bugbite': {'name': 'Bug Bite', 'type': 'Bug', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': "User steals and eats the target's Berry."},
    'bugbuzz': {'name': 'Bug Buzz', 'type': 'Bug', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'bulkup': {'name': 'Bulk Up', 'type': 'Fighting', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Defense by 1."},
    'bulletseed': {'name': 'Bullet Seed', 'type': 'Grass', 'category': 'Physical', 'base_power': 25, 'accuracy': 100, 'priority': 0, 'description': 'Hits 2-5 times in one turn.'},
    'calmmind': {'name': 'Calm Mind', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk and Sp. Def by 1."},
    'closecombat': {'name': 'Close Combat', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Defense and Sp. Def by 1."},
    'curse': {'name': 'Curse', 'type': 'Ghost', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Curses if Ghost, else -1 Spe, +1 Atk, +1 Def.'},
    'darkpulse': {'name': 'Dark Pulse', 'type': 'Dark', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '20% chance to make the target flinch.'},
    'dazzlinggleam': {'name': 'Dazzling Gleam', 'type': 'Fairy', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'},
    'defog': {'name': 'Defog', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': '-1 evasion; ends user and target hazards/terrain.'},
    'doubleedge': {'name': 'Double-Edge', 'type': 'Normal', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'dragondance': {'name': 'Dragon Dance', 'type': 'Dragon', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack and Speed by 1."},
    'drainingkiss': {'name': 'Draining Kiss', 'type': 'Fairy', 'category': 'Special', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 75% of the damage dealt.'},
    'drainpunch': {'name': 'Drain Punch', 'type': 'Fighting', 'category': 'Physical', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'},
    'dualwingbeat': {'name': 'Dual Wingbeat', 'type': 'Flying', 'category': 'Physical', 'base_power': 40, 'accuracy': 90, 'priority': 0, 'description': 'Hits 2 times in one turn.'},
    'earthpower': {'name': 'Earth Power', 'type': 'Ground', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'earthquake': {'name': 'Earthquake', 'type': 'Ground', 'category': 'Physical', 'base_power': 100, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dig.'},
    'encore': {'name': 'Encore', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'Target repeats its last move for its next 3 turns.'},
    'energyball': {'name': 'Energy Ball', 'type': 'Grass', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'esperwing': {'name': 'Esper Wing', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '100% chance to raise user Speed by 1. High crit.'},
    'facade': {'name': 'Facade', 'type': 'Normal', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'Power doubles if user is burn/poison/paralyzed.'},
    'fireblast': {'name': 'Fire Blast', 'type': 'Fire', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': '10% chance to burn the target.'},
    'flamethrower': {'name': 'Flamethrower', 'type': 'Fire', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to burn the target.'},
    'flareblitz': {'name': 'Flare Blitz', 'type': 'Fire', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil. 10% chance to burn. Thaws user.'},
    'flipturn': {'name': 'Flip Turn', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'},
    'focusblast': {'name': 'Focus Blast', 'type': 'Fighting', 'category': 'Special', 'base_power': 120, 'accuracy': 70, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'freezedry': {'name': 'Freeze-Dry', 'type': 'Ice', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze. Super effective on Water.'},
    'grassknot': {'name': 'Grass Knot', 'type': 'Grass', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the target.'},
    'haze': {'name': 'Haze', 'type': 'Ice', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Eliminates all stat changes.'},
    'healbell': {'name': 'Heal Bell', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Cures the user's party of all status conditions."},
    'heatcrash': {'name': 'Heat Crash', 'type': 'Fire', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': 'More power the heavier the user than the target.'},
    'highhorsepower': {'name': 'High Horsepower', 'type': 'Ground', 'category': 'Physical', 'base_power': 95, 'accuracy': 95, 'priority': 0, 'description': 'No additional effect.'},
    'hurricane': {'name': 'Hurricane', 'type': 'Flying', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to confuse target. Can't miss in rain."},
    'hydropump': {'name': 'Hydro Pump', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 80, 'priority': 0, 'description': 'No additional effect.'},
    'hypervoice': {'name': 'Hyper Voice', 'type': 'Normal', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'},
    'icebeam': {'name': 'Ice Beam', 'type': 'Ice', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to freeze the target.'},
    'iceshard': {'name': 'Ice Shard', 'type': 'Ice', 'category': 'Physical', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'},
    'icespinner': {'name': 'Ice Spinner', 'type': 'Ice', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Ends the effects of terrain.'},
    'iciclecrash': {'name': 'Icicle Crash', 'type': 'Ice', 'category': 'Physical', 'base_power': 85, 'accuracy': 90, 'priority': 0, 'description': '30% chance to make the target flinch.'},
    'ironhead': {'name': 'Iron Head', 'type': 'Steel', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to make the target flinch.'},
    'jetpunch': {'name': 'Jet Punch', 'type': 'Water', 'category': 'Physical', 'base_power': 60, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'},
    'knockoff': {'name': 'Knock Off', 'type': 'Dark', 'category': 'Physical', 'base_power': 65, 'accuracy': 100, 'priority': 0, 'description': '1.5× damage if foe holds an item. Removes item.'},
    'leafblade': {'name': 'Leaf Blade', 'type': 'Grass', 'category': 'Physical', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'High critical hit ratio.'},
    'leafstorm': {'name': 'Leaf Storm', 'type': 'Grass', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."},
    'leechlife': {'name': 'Leech Life', 'type': 'Bug', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'User recovers 50% of the damage dealt.'},
    'liquidation': {'name': 'Liquidation', 'type': 'Water', 'category': 'Physical', 'base_power': 85, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Defense by 1."},
    'megahorn': {'name': 'Megahorn', 'type': 'Bug', 'category': 'Physical', 'base_power': 120, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect.'},
    'meteorbeam': {'name': 'Meteor Beam', 'type': 'Rock', 'category': 'Special', 'base_power': 120, 'accuracy': 90, 'priority': 0, 'description': "Raises user's Sp. Atk by 1 on turn 1. Hits turn 2."},
    'mirrorcoat': {'name': 'Mirror Coat', 'type': 'Psychic', 'category': 'Special', 'base_power': 0, 'accuracy': 100, 'priority': -5, 'description': 'If hit by special attack, returns double damage.'},
    'moonblast': {'name': 'Moonblast', 'type': 'Fairy', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': "30% chance to lower the target's Sp. Atk by 1."},
    'moonlight': {'name': 'Moonlight', 'type': 'Fairy', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'},
    'nastyplot': {'name': 'Nasty Plot', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Sp. Atk by 2."},
    'nuzzle': {'name': 'Nuzzle', 'type': 'Electric', 'category': 'Physical', 'base_power': 20, 'accuracy': 100, 'priority': 0, 'description': '100% chance to paralyze the target.'},
    'originpulse': {'name': 'Origin Pulse', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 85, 'priority': 0, 'description': 'No additional effect. Hits adjacent foes.'},
    'outrage': {'name': 'Outrage', 'type': 'Dragon', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Lasts 2-3 turns. Confuses the user afterwards.'},
    'overheat': {'name': 'Overheat', 'type': 'Fire', 'category': 'Special', 'base_power': 130, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."},
    'poisonjab': {'name': 'Poison Jab', 'type': 'Poison', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'},
    'protect': {'name': 'Protect', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 4, 'description': 'Prevents moves from affecting the user this turn.'},
    'psychic': {'name': 'Psychic', 'type': 'Psychic', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': "10% chance to lower the target's Sp. Def by 1."},
    'psychicnoise': {'name': 'Psychic Noise', 'type': 'Psychic', 'category': 'Special', 'base_power': 75, 'accuracy': 100, 'priority': 0, 'description': 'For 2 turns, the target is prevented from healing.'},
    'psychoboost': {'name': 'Psycho Boost', 'type': 'Psychic', 'category': 'Special', 'base_power': 140, 'accuracy': 90, 'priority': 0, 'description': "Lowers the user's Sp. Atk by 2."},
    'psyshock': {'name': 'Psyshock', 'type': 'Psychic', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Damages target based on Defense, not Sp. Def.'},
    'rapidspin': {'name': 'Rapid Spin', 'type': 'Normal', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': 'Free user from hazards/bind/Leech Seed; +1 Spe.'},
    'recover': {'name': 'Recover', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'},
    'rest': {'name': 'Rest', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User sleeps 2 turns and restores HP and status.'},
    'revivalblessing': {'name': 'Revival Blessing', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Revives a fainted Pokemon to 50% HP.'},
    'roar': {'name': 'Roar', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': -6, 'description': 'Forces the target to switch to a random ally.'},
    'roost': {'name': 'Roost', 'type': 'Flying', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Heals 50% HP. Flying-type removed 'til turn ends."},
    'scald': {'name': 'Scald', 'type': 'Water', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'},
    'seismictoss': {'name': 'Seismic Toss', 'type': 'Fighting', 'category': 'Physical', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "Does damage equal to the user's level."},
    'shadowball': {'name': 'Shadow Ball', 'type': 'Ghost', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': "20% chance to lower the target's Sp. Def by 1."},
    'shellsmash': {'name': 'Shell Smash', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Def, SpD by 1; raises Atk, SpA, Spe by 2.'},
    'slackoff': {'name': 'Slack Off', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'},
    'sleeptalk': {'name': 'Sleep Talk', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User must be asleep. Uses another known move.'},
    'sludgebomb': {'name': 'Sludge Bomb', 'type': 'Poison', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '30% chance to poison the target.'},
    'sludgewave': {'name': 'Sludge Wave', 'type': 'Poison', 'category': 'Special', 'base_power': 95, 'accuracy': 100, 'priority': 0, 'description': '10% chance to poison adjacent Pokemon.'},
    'softboiled': {'name': 'Soft-Boiled', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by 50% of its max HP.'},
    'spikes': {'name': 'Spikes', 'type': 'Ground', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts grounded foes on switch-in. Max 3 layers.'},
    'spiritshackle': {'name': 'Spirit Shackle', 'type': 'Ghost', 'category': 'Physical', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'Prevents the target from switching out.'},
    'stealthrock': {'name': 'Stealth Rock', 'type': 'Rock', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Hurts foes on switch-in. Factors Rock weakness.'},
    'steameruption': {'name': 'Steam Eruption', 'type': 'Water', 'category': 'Special', 'base_power': 110, 'accuracy': 95, 'priority': 0, 'description': '30% chance to burn the target. Thaws target.'},
    'stickyweb': {'name': 'Sticky Web', 'type': 'Bug', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Lowers Speed of grounded foes by 1 on switch-in.'},
    'stoneedge': {'name': 'Stone Edge', 'type': 'Rock', 'category': 'Physical', 'base_power': 100, 'accuracy': 80, 'priority': 0, 'description': 'High critical hit ratio.'},
    'strengthsap': {'name': 'Strength Sap', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User heals HP=target's Atk stat. Lowers Atk by 1."},
    'substitute': {'name': 'Substitute', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'User takes 1/4 its max HP to put in a substitute.'},
    'suckerpunch': {'name': 'Sucker Punch', 'type': 'Dark', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first. Fails if target is not attacking.'},
    'supercellslam': {'name': 'Supercell Slam', 'type': 'Electric', 'category': 'Physical', 'base_power': 100, 'accuracy': 95, 'priority': 0, 'description': 'User is hurt by 50% of its max HP if it misses.'},
    'superpower': {'name': 'Superpower', 'type': 'Fighting', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': "Lowers the user's Attack and Defense by 1."},
    'surf': {'name': 'Surf', 'type': 'Water', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': 'Hits adjacent Pokemon. Double damage on Dive.'},
    'switcheroo': {'name': 'Switcheroo', 'type': 'Dark', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."},
    'swordsdance': {'name': 'Swords Dance', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': "Raises the user's Attack by 2."},
    'synthesis': {'name': 'Synthesis', 'type': 'Grass', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Heals the user by a weather-dependent amount.'},
    'terablast': {'name': 'Tera Blast', 'type': 'Normal', 'category': 'Special', 'base_power': 80, 'accuracy': 100, 'priority': 0, 'description': 'If Terastallized: Phys. if Atk > SpA, type = Tera.'},
    'thunder': {'name': 'Thunder', 'type': 'Electric', 'category': 'Special', 'base_power': 110, 'accuracy': 70, 'priority': 0, 'description': "30% chance to paralyze. Can't miss in rain."},
    'thunderbolt': {'name': 'Thunderbolt', 'type': 'Electric', 'category': 'Special', 'base_power': 90, 'accuracy': 100, 'priority': 0, 'description': '10% chance to paralyze the target.'},
    'thunderwave': {'name': 'Thunder Wave', 'type': 'Electric', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': 'Paralyzes the target.'},
    'toxic': {'name': 'Toxic', 'type': 'Poison', 'category': 'Status', 'base_power': 0, 'accuracy': 90, 'priority': 0, 'description': "Badly poisons the target. Poison types can't miss."},
    'trailblaze': {'name': 'Trailblaze', 'type': 'Grass', 'category': 'Physical', 'base_power': 50, 'accuracy': 100, 'priority': 0, 'description': "100% chance to raise the user's Speed by 1."},
    'trick': {'name': 'Trick', 'type': 'Psychic', 'category': 'Status', 'base_power': 0, 'accuracy': 100, 'priority': 0, 'description': "User switches its held item with the target's."},
    'tripleaxel': {'name': 'Triple Axel', 'type': 'Ice', 'category': 'Physical', 'base_power': 20, 'accuracy': 90, 'priority': 0, 'description': 'Hits 3 times. Each hit can miss, but power rises.'},
    'uturn': {'name': 'U-turn', 'type': 'Bug', 'category': 'Physical', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'},
    'vacuumwave': {'name': 'Vacuum Wave', 'type': 'Fighting', 'category': 'Special', 'base_power': 40, 'accuracy': 100, 'priority': 1, 'description': 'Usually goes first.'},
    'voltswitch': {'name': 'Volt Switch', 'type': 'Electric', 'category': 'Special', 'base_power': 70, 'accuracy': 100, 'priority': 0, 'description': 'User switches out after damaging the target.'},
    'waterspout': {'name': 'Water Spout', 'type': 'Water', 'category': 'Special', 'base_power': 150, 'accuracy': 100, 'priority': 0, 'description': "Less power as user's HP decreases. Hits foe(s)."},
    'wavecrash': {'name': 'Wave Crash', 'type': 'Water', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'willowisp': {'name': 'Will-O-Wisp', 'type': 'Fire', 'category': 'Status', 'base_power': 0, 'accuracy': 85, 'priority': 0, 'description': 'Burns the target.'},
    'woodhammer': {'name': 'Wood Hammer', 'type': 'Grass', 'category': 'Physical', 'base_power': 120, 'accuracy': 100, 'priority': 0, 'description': 'Has 33% recoil.'},
    'yawn': {'name': 'Yawn', 'type': 'Normal', 'category': 'Status', 'base_power': 0, 'accuracy': True, 'priority': 0, 'description': 'Puts the target to sleep after 1 turn.'},
    'zenheadbutt': {'name': 'Zen Headbutt', 'type': 'Psychic', 'category': 'Physical', 'base_power': 80, 'accuracy': 90, 'priority': 0, 'description': '20% chance to make the target flinch.'},
}



def canon(who):
    return re.sub(r'^(p[12])[a-z]:', r'\1:', str(who))


def mechanical_state(o):
    """Only observed facts; move revelation itself can supply an exchange sample."""
    ps = o.get('public_summary', {})
    result = {'field': {k: v for k, v in ps.items() if k != 'pokemon'}}
    for side, p in [('us', me(o)), ('them', foe(o))]:
        public = pub(o, p)
        result[side] = {
            'ident': p.get('ident'), 'species': species(p), 'types': types(o, p),
            'status': public.get('status', ''), 'boosts': public.get('boosts', {}),
            'volatiles': public.get('volatiles', {}), 'item': p.get('item', public.get('item')),
            'ability': p.get('ability', public.get('ability')),
        }
    return result


NO_REPEAT = set(('uturn voltswitch flipturn partingshot chillyreception teleport batonpass '
    'outrage petaldance thrash rollout iceball futuresight doomdesire solarbeam solarblade '
    'meteorbeam electroshot skullbash fly dig dive phantomforce shadowforce bounce '
    'gigatonhammer bloodmoon hyperbeam gigaimpact frenzyplant blastburn hydrocannon '
    'suckerpunch upperhand fakeout firstimpression counter mirrorcoat metalburst '
    'endeavor finalgambit explosion selfdestruct lastrespects ragefist focuspunch '
    'recharge struggle rapidspin mortalspin fellstinger poweruppunch flamecharge '
    'uturn dragonTail circlethrow waterspout eruption dragonenergy flail reversal '
    'stompingtantrum retaliate').lower().split())


class Harness(EvidenceHarness):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.exchange = None
        self.last_contract = {}
        self.audit = ''
        self.max_call_elapsed = 0.0
        self.fast_steps = 0
        self.model_steps = 0
        self.previous_source = ''

    def known_move(self, o, name):
        mid = ident(name)
        return o.get('move_dex', {}).get(mid, PUBLIC_MOVE_RULES.get(mid, {}))

    def defensive_multiplier(self, o, attacker, defender, move, tera=None):
        md = self.known_move(o, move)
        if not md or md.get('category') == 'Status':
            return None
        mt = md.get('type')
        mid = ident(move)
        if mid in ('judgment', 'multiattack', 'technoblast', 'terrainpulse', 'aurawheel'):
            return None  # Hidden item/grounding/form mechanics are not resolved here.
        if mid == 'terablast':
            tr = attacker.get('terastallized') or attacker.get('teraTypeRevealed')
            if not tr:
                match = re.search(r'tera:([A-Za-z]+)', attacker.get('details', ''))
                tr = match.group(1) if match else None
            if tr == 'Stellar':
                return None
            if tr:
                mt = tr
        if mid == 'weatherball':
            weather = ident(o.get('public_summary', {}).get('weather'))
            mt = {'sunnyday': 'Fire', 'desolateland': 'Fire', 'raindance': 'Water',
                  'primordialsea': 'Water', 'sandstorm': 'Rock', 'hail': 'Ice', 'snow': 'Ice'}.get(weather, mt)
        dt = types(o, defender, tera)
        eff = effectiveness(mt, dt)
        if ident(move) == 'freezedry' and 'Water' in dt:
            eff *= 4
        if ident(move) == 'thousandarrows' and 'Flying' in dt:
            eff = effectiveness(mt, [t for t in dt if t != 'Flying'])
        ability = ident(defender.get('ability'))
        if ident(attacker.get('ability')) not in ('moldbreaker', 'teravolt', 'turboblaze'):
            immunities = {'Ground': ('levitate', 'eartheater'), 'Water': ('waterabsorb', 'stormdrain', 'dryskin'),
                          'Fire': ('flashfire', 'wellbakedbody'), 'Electric': ('voltabsorb', 'lightningrod', 'motordrive'),
                          'Grass': ('sapsipper',)}
            if ability in immunities.get(mt, ()) and not (mid == 'thousandarrows' and ability == 'levitate'):
                eff = 0
            if ability == 'wonderguard' and eff <= 1:
                eff = 0
            if ability == 'bulletproof' and ident(move) in ('shadowball', 'sludgebomb', 'focusblast', 'energyball', 'aurasphere', 'weatherball', 'seedbomb', 'pollenpuff', 'electroball', 'gyroball'):
                eff = 0
            if ability == 'thickfat' and mt in ('Fire', 'Ice'):
                eff *= .5
        return eff

    def coverage_report(self, o):
        q = foe(o)
        names = q.get('moves', [])
        rows = []
        for i, p in enumerate(team(o), 1):
            if hp(p) <= 0:
                continue
            row = {'slot': i, 'name': species(p), 'hp': p.get('condition'),
                   'entry_loss': entry_hint(o, p),
                   'revealed_attack_multipliers': {name: self.defensive_multiplier(o, q, p, name) for name in names}}
            if p.get('active') and any('terastallize' in a for a in o.get('legal_actions', [])):
                row['if_tera_' + str(p.get('teraType'))] = {
                    name: self.defensive_multiplier(o, q, p, name, p.get('teraType')) for name in names}
            rows.append(row)
        rules = {name: self.known_move(o, name) or 'UNKNOWN move data' for name in names}
        return ('REVEALED OPPONENT MOVE RULES (not an assumed full set)=' + dumps(rules)
                + '\nDEFENSIVE COVERAGE (type and listed ability multipliers, NOT HP damage; null=unknown/status)='
                + dumps(rows))

    def report(self, o):
        # Keep the parent's authoritative state. Condense descriptions to avoid
        # duplicating full explanatory prose for every bench move each turn.
        original = o.get('move_dex', {})
        compact = {}
        for key, value in original.items():
            compact[key] = {k: v for k, v in value.items()
                            if k not in ('description', 'desc', 'shortDesc', 'short_description')}
            desc = value.get('short_description') or value.get('shortDesc') or value.get('description') or value.get('desc')
            if desc:
                compact[key]['rules'] = desc
        view = dict(o, move_dex=compact)
        result = self.coverage_report(o) + '\n' + super().report(view)
        flags = (o.get('request') or {}).get('forceSwitch')
        result += '\nDECISION MODE=' + ('FORCED REPLACEMENT: evaluate entry and following attack.' if flags else
                                       'NORMAL TURN: switching gives the foe an action.')
        result += '\nEXCHANGE AUDIT=' + self.audit
        result += '\nCALL ALLOCATION=' + dumps({'model_actions': self.model_steps,
                         'verified_exchange_actions': self.fast_steps,
                         'max_elapsed_call_s': round(self.max_call_elapsed, 1)})
        return result

    def extract_exchange(self, old, action, new):
        """Measure direct hits before recovery; never interpret net HP as damage."""
        if not action.startswith('move '):
            return None, 'No ordinary attack sample'
        if not isinstance(new, dict) or new.get('done'):
            return None, 'Battle ended'
        if me(old).get('ident') != me(new).get('ident') or foe(old).get('ident') != foe(new).get('ident'):
            return None, 'Active matchup changed'
        if hp(me(new)) <= 0 or hp(foe(new)) <= 0:
            return None, 'A Pokemon fainted'
        if any(' tox' in p.get('condition', '') or p.get('status') == 'tox' for p in (me(new), foe(new))):
            return None, 'Toxic damage escalates outside the stationary exchange model'
        if any(' ' + status in me(new).get('condition', '') for status in ('par', 'slp', 'frz')):
            return None, 'Status can prevent our next action; require a fresh decision'
        idx = int(action.split()[1])-1
        if idx >= len(moves(old)):
            return None, 'Unavailable old move slot'
        mid = ident(moves(old)[idx].get('id', moves(old)[idx].get('move')))
        md = old.get('move_dex', {}).get(mid, {})
        if md.get('category') not in ('Physical', 'Special') or mid in NO_REPEAT:
            return None, 'Move requires fresh tactical judgment'
        accuracy = md.get('accuracy', 100)
        if accuracy is not True and (not isinstance(accuracy, (int, float)) or accuracy < 90):
            return None, 'Inaccurate attack needs a fresh decision'
        before_context, after_context = mechanical_state(old), mechanical_state(new)
        if 'terastallize' in action:
            # Our explicitly chosen Tera resolves before both attacks. Therefore
            # these observed hit sizes already describe the new defensive typing.
            before_context['us']['types'] = after_context['us']['types']
        if before_context != after_context:
            return None, 'Types/status/boosts/item/ability/field changed'
        us, them = me(old).get('ident'), foe(old).get('ident')
        hps = {us: hp(me(old)), them: hp(foe(old))}
        direct = {us: 0.0, them: 0.0}
        residual = {us: 0.0, them: 0.0}
        heals = {us: 0.0, them: 0.0}
        orders = []
        last = None
        # Passive existing healing/damage is measured; all other effects abort.
        benign = {'', 'move', '-damage', '-heal', '-resisted', '-supereffective', 'upkeep', 'turn'}
        for line in new.get('recent_events', []):
            b = line.split('|')
            kind = b[1] if len(b) > 1 else ''
            if kind == '-weather' and '[upkeep]' in line:
                continue
            if kind == '-terastallize' and len(b) > 2 and b[2].startswith('p1') and 'terastallize' in action:
                continue
            if kind not in benign:
                return None, 'Exchange interrupted: ' + line[:150]
            who = canon(b[2]) if len(b) > 2 else ''
            if kind == 'move' and len(b) > 4:
                if who not in hps or canon(b[4]) not in hps or '[still]' in line or '[miss]' in line:
                    return None, 'Nonstandard or self-targeting opposing response'
                orders.append(who)
                last = (who, ident(b[3]), canon(b[4]))
                if who == us and last[1] != mid:
                    return None, 'Executed move did not match the selected move ID'
                if who == them and (last[1] in ('ragefist', 'flail', 'reversal', 'brine', 'furycutter', 'echoedvoice', 'rollout', 'iceball')
                                    or ident(foe(new).get('item')) == 'metronome'):
                    return None, 'Opposing move can grow stronger during this exchange'
            if kind in ('-damage', '-heal') and len(b) > 3 and who in hps:
                now = hp({'condition': b[3]})
                delta = hps[who] - now
                if kind == '-heal':
                    heals[who] += max(0, -delta)
                elif '[from]' not in line and last and last[2] == who:
                    direct[who] += max(0, delta)
                else:
                    residual[who] += max(0, delta)
                hps[who] = now
        if orders.count(us) != 1 or orders.count(them) != 1 or len(orders) != 2:
            return None, 'Need one observed attack by each side'
        if direct[them] < .08 or direct[us] <= 0:
            return None, 'Need uncensored damaging attacks on both sides'
        # Worst roll relative to an unknown observed roll; extra two percentage
        # points cover rounded opponent HP and minor arithmetic uncertainty.
        outgoing_low = max(0, direct[them] * .85 - .02)
        incoming_high = direct[us] / .85 + .02
        their_heal = heals[them]
        if outgoing_low <= their_heal:
            return None, 'Opponent recovery defeats net damage progress'
        # Ignore our healing for safety; this deliberately rejects fragile drain loops.
        effective_out = outgoing_low - their_heal
        current_foe = hp(foe(new))
        hits = 1 if current_foe <= outgoing_low else 2 if current_foe <= outgoing_low + effective_out else 3
        if hits > 2:
            return None, 'No conservative finish within two attacks'
        # Order is empirical, not proof of raw Speed. Only use the advantage
        # for the same attack in unchanged context, with a substantial HP
        # cushion for hidden priority. Unknown coverage remains a limitation.
        enemy_moves = [b.split('|')[3] for b in new.get('recent_events', [])
                       if b.startswith('|move|p2') and len(b.split('|')) > 3]
        enemy_md = self.known_move(new, enemy_moves[-1]) if enemy_moves else {}
        # A foe's negative-priority move does not establish ordinary speed order.
        first = (orders[0] == us and bool(enemy_md)
                 and (md.get('priority', 0) > 0 or enemy_md.get('priority', 0) >= 0))
        incoming_hits = hits - 1 if first else hits
        need = incoming_hits * incoming_high + hits * residual[us] + max(.15, incoming_high * .3)
        if hp(me(new)) <= need:
            return None, ('Unsafe damage race: need %d hits at >=%.1f%% each; observed enemy hit widened to %.1f%%; '
                              'current HP %.1f%% versus required cushion %.1f%%; observed first=%s' %
                              (hits, outgoing_low*100, incoming_high*100, hp(me(new))*100, need*100, first))
        if hp(me(new)) < .30:
            return None, 'Preserve planning for low HP positions'
        # A revealed bench immunity/resistance can turn the same attack into lost
        # momentum. Ask Astra rather than blindly repeating into an obvious pivot.
        _, active_eff = power_hint(new, me(new), foe(new), mid)
        for enemy in new.get('opponent_revealed', []):
            if enemy.get('ident') == them or hp(enemy) <= 0:
                continue
            _, eff = power_hint(new, me(new), enemy, mid)
            if eff == 0:
                return None, 'Revealed opposing bench is immune to this continuation'
        return {'mid': mid, 'own': us, 'foe': them, 'context': mechanical_state(new),
                'incoming_high': incoming_high, 'outgoing_low': outgoing_low,
                'foe_heal': their_heal, 'own_residual': residual[us], 'hits': hits,
                'acted_first_at_valid_priority': first,
                'remaining': 2, 'before': new}, ''

    def recovery_cycle(self):
        rows = self.history[-4:]
        if len(rows) < 4:
            return ''
        keys = {(r.get('own'), r.get('foe'), r.get('move_id')) for r in rows}
        if len(keys) != 1 or not rows[-1].get('move_id'):
            return ''
        if any(r.get('own') != r.get('own_after') or r.get('foe') != r.get('foe_after') for r in rows):
            return ''
        own_loss = rows[0]['own_hp_before'] - rows[-1]['own_hp_after']
        foe_loss = rows[0]['foe_hp_before'] - rows[-1]['foe_hp_after']
        if own_loss >= .15 and foe_loss <= .05:
            return ('Recovery cycle: four repetitions lost us %.0f%% max HP with only %.0f%% net foe damage. '
                    'Reassess the win condition, coverage, switching and whether this sacrifice serves a concrete plan.'
                    % (own_loss*100, foe_loss*100))
        return ''

    def record(self, old, action, new, source):
        super().record(old, action, new, source)
        if action.startswith('move '):
            idx = int(action.split()[1])-1
            if idx < len(moves(old)):
                self.history[-1]['move_id'] = ident(moves(old)[idx].get('id', moves(old)[idx].get('move')))
        cycle = self.recovery_cycle()
        if cycle:
            self.failure = (self.failure + '; ' + cycle).strip('; ')
            self.ctx.log('recovery_cycle', reason=cycle)
        if not isinstance(new, dict) or new.get('done'):
            self.exchange = None
            return
        old_remaining = self.exchange.get('remaining', 0) if self.exchange else 0
        if source in ('astra', 'empirical_exchange'):
            cert, reason = self.extract_exchange(old, action, new)
            if cert and not self.failure:
                cert['remaining'] = 2 if source == 'astra' else old_remaining
                self.exchange = cert if cert['remaining'] > 0 else None
                self.audit = 'Observed exchange permits a bounded continuation: ' + dumps({
                    k: cert[k] for k in ('mid', 'outgoing_low', 'incoming_high', 'hits', 'remaining')})
            else:
                self.exchange = None
                self.audit = reason or self.failure
                if source == 'empirical_exchange':
                    self.failure = 'Empirical continuation ended: ' + self.audit + '; ' + self.failure
            self.ctx.log('exchange_audit', permitted=bool(self.exchange), reason=self.audit)
        else:
            self.exchange = None
        self.previous_source = source

    def continuation(self, o):
        # Forced-entry plans remain the parent's identity-bound, effect-checked plans.
        follow = super().continuation(o)
        if follow:
            self.exchange = None
            return follow, 'checked_entry_plan'
        cert = self.exchange
        if not cert or cert.get('remaining', 0) <= 0:
            return None, ''
        reason = self.failure
        pr = self.ctx.progress()
        if pr.get('since_obs_change', 0) >= 2 or pr.get('since_score_change', 0) >= 3:
            reason = 'Progress guard: no measurable exchange progress'
        if mechanical_state(o) != cert['context']:
            reason = 'Mechanical context no longer matches observed exchange'
        if me(o).get('ident') != cert['own'] or foe(o).get('ident') != cert['foe']:
            reason = 'Identity changed'
        if (o.get('request') or {}).get('forceSwitch'):
            reason = 'Forced replacement'
        found = next((f'move {i}' for i, m in enumerate(moves(o), 1)
                      if ident(m.get('id', m.get('move'))) == cert['mid']
                      and f'move {i}' in o.get('legal_actions', [])), None)
        if not found or found in self.banned:
            reason = 'Bound move is unavailable'
        if reason:
            self.exchange = None
            self.failure = 'Exchange guard failed: ' + reason
            self.ctx.log('plan_invalidated', reason=self.failure)
            return None, ''
        cert['remaining'] -= 1
        self.fast_steps += 1
        return found, 'empirical_exchange'

    def run_episode(self, game):
        ctx = self.ctx
        o = ctx.observe()
        if isinstance(o, dict):
            self.ingest_reveals(o.get('recent_events', []))
        for _ in range(230):
            if ctx.done or not isinstance(o, dict) or o.get('done'):
                return
            legal = [x for x in o.get('legal_actions', []) if x != 'noop']
            if not legal:
                return
            action, source = self.continuation(o)
            if len(legal) == 1:
                action, source = legal[0], 'only_legal_choice'
                self.exchange = None
            budget = ctx.budget.remaining()
            # 190-240s timeouts exceed 178s observed latency. Leave at least
            # 35s after the call; elapsed-call measurements can raise the reserve.
            reserve = max(225., min(330., self.max_call_elapsed + 40.))
            can_think = budget.get('system2_calls', 0) > 1 and budget.get('wall_s', 0) > reserve
            if not action and can_think:
                started = ctx.clock.now()
                try:
                    result = ctx.think(self.report(o), system=SYSTEM, session_id=None, timeout_s=min(240., budget['wall_s'] - 35.))
                    self.calls += 1
                    self.latencies.append(result.latency_s)
                    action, data = self.parse(result.text, legal)
                    if action in self.banned:
                        action = None
                    if action:
                        self.memo = str(data.get('memory', self.memo))[:800]
                        self.last_contract = data
                        self.plan = None
                        self.exchange = None
                        # A free replacement and intended attack can be planned in
                        # one call. No speculative scripted voluntary switch sequence.
                        if (o.get('request') or {}).get('forceSwitch') and action.startswith('switch '):
                            self.make_plan(o, action, dict(data, repeat=0))
                            if self.plan and self.known_move(o, self.plan['mid']).get('category') not in ('Physical', 'Special'):
                                self.plan = None
                        source = 'astra'
                        self.model_steps += 1
                        ctx.log('tactical_decision', action=action, why=str(data.get('why', ''))[:500],
                                memory=self.memo)
                    else:
                        self.failure = 'Malformed/unavailable answer; legal fallback then replan'
                        self.plan = self.exchange = None
                except (BudgetExceeded, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    self.failure = 'Model failed: ' + type(exc).__name__ + '; reconsider fallback next decision'
                    self.plan = self.exchange = None
                    ctx.log('model_recovery', reason=self.failure)
                finally:
                    self.max_call_elapsed = max(self.max_call_elapsed, ctx.clock.now() - started)
            if not action:
                action = fallback(o, self.banned)
                source = 'budget_or_parse_fallback'
            if not action or ctx.done:
                return
            old = o
            try:
                o = ctx.act(action)
                if o is None:
                    o = ctx.observe()
            except (BudgetExceeded, ModelUnavailable):
                return
            except Exception as exc:
                if ctx.done:
                    return
                self.banned.add(action)
                self.plan = self.exchange = None
                self.failure = 'Act rejected: ' + type(exc).__name__ + ': ' + str(exc)[:200]
                o = ctx.observe()
                ctx.log('action_error_recovery', action=action, reason=self.failure)
                if len(self.banned) >= len(legal):
                    return
                continue
            if not isinstance(o, dict):
                return
            self.record(old, action, o, source)


def build(ctx):
    return Harness(ctx)
