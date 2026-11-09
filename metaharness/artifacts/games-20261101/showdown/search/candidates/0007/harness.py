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


SYSTEM = """You are an expert Gen 9 Random Battle singles player against a strong MCTS opponent.
Win the six-Pokemon battle. Use the complete current state, not remembered move/switch indices.
Protocol is authoritative. Own request stats are unboosted; boosts, status and item modify them.
Opposing hidden sets are UNKNOWN: distinguish revealed evidence from likely coverage. No team preview.
Before choosing, compare: best attack; safest switch; setup/recovery only if the opponent cannot punish;
Tera if it changes survival or wins a decisive exchange. Consider both their strongest attack and a
plausible switch. Check who acts first, priority, KO/survival, coverage, hazards and the next turn.
Do not lose a unique defensive check to gain chip. Preserve a concrete win condition. Do not blindly
set up against a super-effective attack or Psyshock. A slower super-effective attack is worthless if
we faint first. Proactive defensive Tera is worthwhile to save a critical piece; don't wait until last mon.
Avoid repeatedly conceding free attacks by switching, but do not let hazards justify a hopeless matchup.
Numeric attack hints are only rough power rankings: NOT exact damage, NOT KO ranges; special move
power, hidden abilities/items, weather, screens and other exceptions may invalidate them. Use actual
observed damage, authoritative move descriptions, and your mechanics knowledge over hints.
Return exactly one compact JSON object and no markdown or analysis:
{"action":"move 1", "why":"<=20 words", "memory":"<=50 words", "queue":[]}
queue is optional and is the only way to request call-free repetition. It must be a list of 0-2
move IDs (for example ["earthquake"]). Each queued move is attempted only against the same
active identities, with the same request shape, no new opponent move/status/boost/field/item event,
our active HP still safe, and measurable damage on the preceding use. Any failed, immune, missed,
low-progress or rejected action cancels it and triggers a fresh assessment. Queue attacks only; never
queue setup, recovery, switching, Tera or a move that depends on an unrevealed coverage assumption.
The immediate action must be an exact member of legal_actions. Keep the answer under 100 words.
"""


class Harness:
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
        """Install only an explicit, conservative attack queue from Astra."""
        self.plan = None
        if not isinstance(data, dict) or not action or not action.startswith('move '):
            return
        # Queue entries are move IDs; reject malformed entries and all status/setup moves.
        q = data.get('queue', [])
        if not isinstance(q, list):
            return
        q = [ident(x) for x in q if isinstance(x, str)][:2]
        if not q:
            return
        p, f = me(o), foe(o)
        allowed = []
        for mid in q:
            if not mid or o.get('move_dex', {}).get(mid, {}).get('category') not in ('Physical', 'Special'):
                return
            # Pivoting attacks create a second request and change the active identity;
            # they are useful Astra choices but poor call-free queue candidates.
            if mid in {'uturn', 'voltswitch', 'flipturn', 'partingshot', 'teleport', 'batonpass'}:
                return
            if not any(ident(m.get('id', m.get('move', ''))) == mid for m in moves(o)):
                return
            allowed.append(mid)
        # A queue is a same-move continuation. Different queued moves are rejected because
        # their safety cannot be inferred from the first observed exchange.
        if len(set(allowed)) != 1:
            return
        i = int(action.split()[1]) - 1
        if i < 0 or i >= len(moves(o)):
            return
        first = ident(moves(o)[i].get('id', moves(o)[i].get('move', '')))
        if first != allowed[0] or 'terastallize' in action:
            return
        self.plan = {'mid': first, 'remaining': len(allowed), 'minhp': .35,
                     'follow': False, 'own': p.get('ident'), 'foe': f.get('ident'), 'before': o}
        self.ctx.log('queue_installed', move=first, remaining=len(allowed), own=p.get('ident'), foe=f.get('ident'))

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
        elif hp(me(o)) <= 0.30:
            reason = 'Queue stopped at low active HP'
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
            action = self.continuation(o)
            source = 'checked_plan'
            if len(legal) == 1:
                action, source = legal[0], 'only_legal_choice'
            budget = ctx.budget.remaining()
            # Never start a call too close to the deadline; observed worst is 104s.
            # Keep one call unspent so a long battle can finish under the cheap controller.
            can_think = budget.get('system2_calls', 0) > 1 and budget.get('wall_s', 0) > 290
            if not action and can_think:
                try:
                    result = ctx.think(self.report(o), system=SYSTEM, session_id=None,
                                       timeout_s=min(260., budget['wall_s']-20))
                    self.calls += 1
                    self.latencies.append(result.latency_s)
                    action, data = self.parse(result.text, legal)
                    if action in self.banned:
                        action = None
                    if action:
                        self.memo = str(data.get('memory', self.memo))[:1000]
                        self.make_plan(o, action, data)
                        # A fresh model decision supersedes stale failure evidence.  The
                        # queue verifier will create a new failure only from this action's
                        # observed result.
                        self.failure = ''
                        source = 'astra'
                        ctx.log('tactical_decision', action=action, why=str(data.get('why', ''))[:600], memory=self.memo)
                    else:
                        self.failure = 'Malformed/unavailable model choice; used legal fallback, replan next decision'
                        self.plan = None
                except (BudgetExceeded, ModelUnavailable):
                    return
                except (TimeoutError, RuntimeError) as exc:
                    self.failure = 'Model call failed: ' + type(exc).__name__ + '; fallback must be reconsidered'
                    self.plan = None
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
                # Hidden trapping can reject an available switch. Re-read and replan.
                if ctx.done:
                    return
                self.banned.add(action)
                self.plan = None
                self.failure = 'Act rejected: ' + type(exc).__name__ + ': ' + str(exc)[:250]
                o = ctx.observe()
                ctx.log('action_error_recovery', action=action, reason=self.failure)
                if len(self.banned) >= len(legal):
                    return
                continue
            self.record(old, action, o, source)


def build(ctx):
    return Harness(ctx)
