"""Public-evidence controller with bounded model-authored tactical policies.

No engine imports, hidden-set access, network, or persistent cross-episode state.
"""
from __future__ import annotations

import json
import math
import re
from collections import deque

from metaharness.core.contract import BudgetExceeded, ModelUnavailable


def ident(s):
    return re.sub(r'^(p[12])[a-z]:', r'\1:', str(s))


def key(s):
    return re.sub('[^a-z0-9]', '', str(s).lower())


def unpack(o):
    if isinstance(o, str):
        try:
            o = json.loads(o)
        except (ValueError, TypeError):
            return {}
    return o if isinstance(o, dict) else {}


def lines(x):
    return x.splitlines() if isinstance(x, str) else (x or [])


def hp(p):
    s = str(p.get('condition', '0'))
    if 'fnt' in s:
        return 0.0
    m = re.match(r'(\d+)/(\d+)', s)
    return int(m[1]) / max(1, int(m[2])) if m else 0.0


def team(o):
    return (o.get('request') or {}).get('side', {}).get('pokemon', [])


def active(o, side='p1'):
    if side == 'p1':
        return next((p for p in team(o) if p.get('active')), {})
    name = o.get('active', {}).get(side)
    return o.get('public_summary', {}).get('pokemon', {}).get(name, {})


def species(p):
    return p.get('details', '').split(',')[0]


def types(o, p):
    tera = p.get('terastallized') or p.get('teraTypeRevealed')
    if isinstance(tera, str) and tera and tera != 'Stellar':
        return [tera]
    return o.get('species_dex', {}).get(species(p), {}).get('types', [])


# Ordinary Gen 9 type chart; exceptions are left explicit in the prompt.
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


def effectiveness(t, ts):
    strong, weak, immune = [s.split() for s in CHART.get(t, ('', '', ''))]
    v = 1.0
    for d in ts:
        v *= 0 if d in immune else 2 if d in strong else .5 if d in weak else 1
    return v


def move_info(o, action):
    m = re.fullmatch(r'move ([1-4])(?: terastallize)?', action)
    if not m:
        return {}, {}
    a = o.get('request', {}).get('active') or []
    ms = a[0].get('moves', []) if a else []
    i = int(m[1]) - 1
    if i >= len(ms):
        return {}, {}
    req = ms[i]
    return req, o.get('move_dex', {}).get(req.get('id', key(req.get('move'))), {})


def attack_proxy(o, action, target=None):
    """Only a transparent fallback ranking, NOT a damage or KO calculation."""
    req, d = move_info(o, action)
    if not req:
        return -1000
    if d.get('category') == 'Status':
        return -1
    p = active(o)
    foe = active(o, 'p2') if target is None else target
    t = d.get('type', '')
    mid = req.get('id', '')
    tera = p.get('terastallized') or (p.get('teraType') if 'terastallize' in action else '')
    if mid == 'terablast' and tera:
        t = tera
    eff = effectiveness(t, types(o, foe))
    ability = key(foe.get('ability', ''))
    immune = {'Ground': {'levitate', 'eartheater'}, 'Water': {'waterabsorb', 'stormdrain', 'dryskin'},
              'Fire': {'flashfire', 'wellbakedbody'}, 'Electric': {'voltabsorb', 'lightningrod', 'motordrive'},
              'Grass': {'sapsipper'}}
    if ability in immune.get(t, set()) and key(p.get('ability')) not in {'moldbreaker', 'teravolt', 'turboblaze'}:
        eff = 0
    power = d.get('base_power', 0) or 60  # variable power explicitly approximate
    category = d.get('category', 'Special')
    stat = p.get('stats', {}).get('atk' if category == 'Physical' else 'spa', 180)
    original = o.get('species_dex', {}).get(species(p), {}).get('types', [])
    stab = (2 if t == tera else 1.5) if t in original else (1.5 if t == tera else 1)
    accuracy = d.get('accuracy', 100)
    accuracy = 100 if accuracy is True else accuracy
    val = power * stab * eff * stat / 180 * float(accuracy or 100) / 100
    if 'terastallize' in action:
        val *= .93
    if mid in {'uturn', 'voltswitch', 'flipturn'}:
        val *= 1.08
    if 'slp' in p.get('condition', '') and mid not in {'sleeptalk', 'snore'}:
        val *= .05
    if 'brn' in p.get('condition', '') and category == 'Physical' and key(p.get('ability')) != 'guts':
        val *= .5
    return val


def fallback(o, legal):
    moves = [a for a in legal if a.startswith('move ')]
    if moves:
        # If every available attack is immune, escape rather than spamming a zero.
        damaging = [a for a in moves if move_info(o,a)[1].get('category') in ('Physical','Special')]
        switches = [a for a in legal if a.startswith('switch ')]
        if damaging and max(attack_proxy(o,a) for a in damaging) <= 0 and switches:
            return fallback(o,switches)
        p = active(o)
        for a in moves:
            req, d = move_info(o, a)
            if req.get('id') in {'recover', 'roost', 'slackoff', 'softboiled', 'milkdrink', 'shoreup'} and hp(p) < .38:
                return a
        return max(moves, key=lambda a: attack_proxy(o, a))
    foes = types(o, active(o, 'p2'))
    scores = {}
    for a in legal:
        m = re.fullmatch(r'switch ([1-6])', a)
        if m and int(m[1]) <= len(team(o)):
            p = team(o)[int(m[1])-1]
            vulnerability = max([effectiveness(t, types(o, p)) for t in foes] or [1])
            scores[a] = hp(p) / max(.3, vulnerability) + .0001 * p.get('stats', {}).get('spe', 0)
    return max(scores, key=scores.get) if scores else legal[0]


class Ledger:
    def __init__(self):
        self.samples = deque(maxlen=36)
        self.orders = deque(maxlen=12)
        self.failures = deque(maxlen=6)
        self.history = deque(maxlen=7)
        self.last_events = []
        self.last_direct = {}
        self.last_cost = {}

    def observe_effect(self, before, after, action, progress):
        events = lines(after.get('recent_events'))
        self.last_events = events
        self.last_direct = {}
        self.last_cost = {}
        mons = dict(before.get('public_summary', {}).get('pokemon', {}))
        for p in team(before):
            name=ident(p.get('ident'))
            mons[name]={**mons.get(name,{}),**p}  # exact private HP/items override caches
        health = {n: hp(p) for n, p in mons.items()}
        current = None
        order = []
        crit = False
        hits = {}
        for line in events:
            s = line.split('|')
            if len(s) < 2:
                continue
            event = s[1]
            if event in {'switch', 'drag', 'replace'} and len(s) > 4:
                health[ident(s[2])] = hp({'condition': s[4]})
                current = None
            elif event == 'move' and len(s) > 4:
                current = (ident(s[2]), s[3], ident(s[4]))
                order.append([current[0], current[1]])
                crit = False
            elif event == '-crit':
                crit = True
            elif event in {'-damage', '-heal'} and len(s) > 3:
                target = ident(s[2])
                now = hp({'condition': s[3]})
                old = health.get(target)
                if event == '-damage' and old is not None and old > now:
                    self.last_cost[target[:2]] = self.last_cost.get(target[:2],0) + old - now
                if (event == '-damage' and current and target == current[2]
                    and old is not None and '[from]' not in line and old > now):
                    k = (current[0], current[1], target)
                    hits[k] = hits.get(k, 0) + old - now
                    self.last_direct[target[:2]] = self.last_direct.get(target[:2], 0) + old - now
                    self.samples.append({'turn': before.get('turn'), 'attacker': current[0],
                                         'move': current[1], 'target': target,
                                         'damage_pct': round(100*(old-now), 1), 'crit': crit,
                                         'target_fainted': now == 0,
                                         'pre_turn_context': {n: {f: p.get(f) for f in ('boosts', 'status', 'ability', 'item', 'teraTypeRevealed') if p.get(f)} for n,p in mons.items() if n in (current[0], target)}})
                health[target] = now
            elif event in {'upkeep', 'turn', 'faint'}:
                current = None
        if len(order) >= 2:
            self.orders.append({'turn': before.get('turn'), 'order': order})
        failed = []
        if after.get('last_error'):
            failed.append(str(after['last_error']))
        if progress.get('since_obs_change', 0) >= 2:
            failed.append('observation stalled')
        if progress.get('since_score_change', 0) >= 4:
            failed.append('four actions without score change: check stalled line')
        if any(any(t in e for t in ('|-immune|', '|-fail|', '|-miss|', '|cant|')) for e in events):
            failed.append('action blocked, missed, immune, or unable to move: inspect protocol')
        own_loss = hp(active(before)) - hp(active(after)) if active(before).get('ident') == active(after).get('ident') else 0
        if own_loss >= .4:
            failed.append('active lost >=40% HP: reassess survival before setup/attack')
        for f in failed:
            self.failures.append({'turn': before.get('turn'), 'action': action, 'evidence': f})
        self.history.append({'turn': before.get('turn'), 'action': action,
                             'self_active_hp': round(hp(active(after)), 3),
                             'foe_active_hp': round(hp(active(after, 'p2')), 3), 'failure': failed})
        return failed


# Policies contain data, never generated Python or arbitrary expressions.
RECOVERY = {'recover','roost','slackoff','softboiled','milkdrink','shoreup','synthesis','moonlight','morningsun','healorder'}
ENTRY_TERRAINS = {'grassysurge':'Grassy Terrain','electricsurge':'Electric Terrain',
                  'mistysurge':'Misty Terrain','psychicsurge':'Psychic Terrain'}
CONDITIONAL = {'uturn','voltswitch','flipturn','futuresight','doomdesire','fakeout','firstimpression','suckerpunch','upperhand','gigatonhammer','bloodmoon','outrage','petaldance','thrash','recharge','solarbeam','solarblade','skyattack','meteorbeam','electroshot','fly','dig','dive','bounce','phantomforce','shadowforce','focuspunch','shelltrap','beakblast','counter','mirrorcoat','metalburst','bide','rollout','iceball'}


def parse_answer(text, legal):
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', str(text)):
        try:
            obj, _ = decoder.raw_decode(str(text)[m.start():])
        except (ValueError, TypeError):
            continue
        if isinstance(obj, dict) and 'action' in obj:
            a = obj.get('action')
            return (a if isinstance(a, str) and a in legal else None), obj
    for m in re.finditer(r'\b(?:move [1-4](?: terastallize)?|switch [1-6])\b', str(text).lower()):
        if m[0] in legal:
            return m[0], {}
    return None, {}


def numeric(v, default, lo, hi):
    try:
        n = float(v)
        return max(lo, min(hi, n)) if math.isfinite(n) else default
    except (TypeError, ValueError, OverflowError):
        return default


def lookup_move(o, mid):
    a = (o.get('request') or {}).get('active') or [{}]
    for i, m in enumerate(a[0].get('moves', []), 1):
        if key(m.get('id', m.get('move'))) == mid and f'move {i}' in o.get('legal_actions', []):
            return f'move {i}'
    return None


def public(o, p):
    return o.get('public_summary', {}).get('pokemon', {}).get(p.get('ident'), {})


def tactical_fingerprint(o, p, switching=False, tera=None):
    q = public(o, p)
    return {'ident':p.get('ident'), 'species':species(p), 'stats':p.get('stats'),
            'ability':p.get('ability', q.get('ability')), 'item':p.get('item', q.get('item')),
            'tera':tera if tera is not None else (p.get('terastallized') or q.get('teraTypeRevealed') or ''),
            'status':p.get('condition','').split()[1:],
            'boosts':{} if switching else {s:n for s,n in q.get('boosts',{}).items() if n},
            'volatiles':{} if switching else q.get('volatiles',{})}


def field_fingerprint(o):
    q = o.get('public_summary', {})
    # Turn counters in summaries may matter; conservative invalidation is intentional.
    return {k:q.get(k) for k in ('weather','field','side_conditions')}


SYSTEM = '''You play p1 in Gen 9 Random Battle singles against a strong MCTS opponent. Win the whole battle. All information is public or our private request; no team preview. Use current slots. Protocol is authoritative. A forced switch is a replacement, not a free attack. Own stats are unboosted. Do not assume hidden moves/items or level 100.
Choose a robust action AND compile its conditional next step. After an ordinary attack or switch, provide a ONE-step policy whenever any safe conditional continuation exists. A policy is only valid IF this foe stays, IF it uses a listed move, and IF the resulting HP/state pass your bounds. You do not need certainty about their response: code checks it. A possible switch, possible unlisted coverage, or needing to check actual damage is NOT by itself a reason to omit the policy. Narrow the guard instead. Use null only if no safe bounded continuation exists or the next decision requires fresh strategic judgment; explain that concrete obstacle in no_policy_reason. Do not propose an unsafe continuation merely to save a call.
Code remaps named moves to current slots and checks every transition. Unknown or changed opponent, unexpected enemy move, changed boosts/status/types/items, missed/failed own attack, excessive damage, illegal move, or no progress causes replanning. An EXPECTED newly revealed enemy move is allowed. The chosen switch's own Surge terrain is recognized only when confirmed by protocol; other field changes cause replanning. Incoming damage is checked before passive recovery; empirical damage overrides your estimate. Policies last at most THREE additional decisions, never across a different opponent. Setup must be reconsidered after its result.
Tactics: compare taking damage now, switching, and defensive Tera. Count TWO enemy hits for a slower voluntary switch-in before it can attack; a resistant switch is not safe merely because it survives entry. Consider plausible coverage especially when a proposed setup/switch could lose a unique check. Don't infer absence of coverage from unrevealed moves. Respect observed damage, priority, speed, hazards, status, choice lock and recovery. Distinguish critical or faint-censored samples. Keep pressure on setup opponents and don't chase switches endlessly. Spend Tera when it secures a critical exchange, not only on your final Pokemon. Preserve an explicit win condition and its check assignments. Check move descriptions for variable power and unusual mechanics. Type multipliers are only ordinary chart facts, not damage predictions.
Return JSON, no prose outside it. Put the executable policy before explanations. Keep the whole answer about 180-260 tokens:
{"action":"move 1","policy":{"attack":"moveid","min_hp":0.45,"max_hit":0.3,"foe_moves":["enemy_moveid","other_expected_moveid"],"uses":1},"why":"<=12 words: strongest reply and tactical purpose","memory":"<=25 words: checks, win condition, Tera"}
Policy is for the SAME FOE against the active Pokemon AFTER your chosen action, including the teammate you switch into. attack must be an ordinary damage move of that Pokemon. Optional heal is an immediate recovery move, used only below an added heal_below threshold. All HP bounds are fractions of maximum HP (0..1). min_hp must leave room to survive the strongest plausible reply; max_hit is a conservative bound on total own HP damage per decision, including attack, recoil, residual and entry damage before healing. Optional max_foe_hp limits the foe's health; default is 1. foe_moves names explicitly anticipated attacks/status moves (max four), never wildcards. Policy cannot repeat setup, switches, Tera, pivots, conditional priority or charge/recharge moves. It may follow a switch, attack, recovery, or Tera if conditions match. A forced replacement normally warrants a conditional first attack; replacement itself takes no enemy hit. Use uses=1 for a narrow continuation, up to 3 for a stable exchange. Output action must be in legal_actions. Assigned teraType is NOT evidence Tera remains available: trust the factual Tera resource and current legal actions over memory.'''


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ledger = Ledger()
        self.memo = ''
        self.rejected = set()
        self.plan = None
        self.replan = 'Opening position: identify a robust line and threats.'
        self.reveals = {}
        self.policy_actions = 0
        self.model_actions = 0

    def ingest(self, o):
        for line in lines(o.get('recent_events')):
            s = line.split('|')
            if len(s)<4:
                continue
            if not re.match(r'p[12][a-z]?:', s[2]):
                continue
            who = ident(s[2])
            d = self.reveals.setdefault(who, {})
            if s[1] in ('-item','-enditem'):
                d['item'] = s[3]
                d['item_state'] = 'held' if s[1]=='-item' else 'removed/consumed'
            elif s[1]=='-ability':
                d['ability'] = s[3]
            for v in s[3:]:
                if v.startswith('[from] ability: ') and not any(t.startswith('[of]') for t in s[3:]):
                    d['ability'] = v.removeprefix('[from] ability: ')
                if v.startswith('[from] item: ') and not any(t.startswith('[of]') for t in s[3:]):
                    d['item'] = v.removeprefix('[from] item: ')
                    d['item_state'] = 'active'
        # Public ability facts also improve the fallback; never overwrite private fields.
        for who, d in self.reveals.items():
            q = o.get('public_summary', {}).get('pokemon', {}).get(who)
            if isinstance(q,dict) and d.get('ability'):
                q['ability'] = d['ability']
            if isinstance(q,dict) and 'item_state' in d:
                q['item'] = d['item'] if d['item_state'] in ('held','active') else ''

    def prompt(self, o):
        ps=[]
        for i,p in enumerate(team(o),1):
            q={k:p[k] for k in ('ident','details','condition','active','stats','moves','ability','item','teraType','terastallized','reviving') if k in p}
            q['slot']=i
            q['types_now']=types(o,p)
            q['ordinary_move_type_multipliers']={
                mid:effectiveness(o.get('move_dex',{}).get(mid,{}).get('type'),types(o,active(o,'p2')))
                for mid in p.get('moves',[]) if o.get('move_dex',{}).get(mid,{}).get('category') in ('Physical','Special')}
            ps.append(q)
        data={'turn':o.get('turn'),'legal_actions':o.get('legal_actions',[]),
              'tera_resource':{'own_spent_by':[p.get('ident') for p in team(o) if p.get('terastallized')],
                               'eligible_now':any('terastallize' in a for a in o.get('legal_actions',[]))},
              'request_flags':{k:v for k,v in (o.get('request') or {}).items() if k!='side'},
              'own_team_current_slots':ps,'active':o.get('active'),
              'public_summary':o.get('public_summary'),'opponent_revealed':o.get('opponent_revealed'),
              'public_item_ability_evidence':self.reveals,'species_dex':o.get('species_dex'),
              'move_rules':o.get('move_dex'),
              'strategic_memory_recheck':self.memo,
              'direct_damage_contextual_not_predictions':list(self.ledger.samples)[-12:],
              'move_orders_not_always_speed':list(self.ledger.orders)[-4:],
              'recent_choices':list(self.ledger.history),
              'why_replanning':self.replan,'failures':list(self.ledger.failures),
              'last_error':o.get('last_error'),
              'progress':{k:v for k,v in self.ctx.progress().items() if k!='score'},
              'budget':self.ctx.budget.remaining()}
        # Exact recent protocol precedes bulk data; nothing is cut mid-object.
        return 'LATEST PROTOCOL\n'+'\n'.join(lines(o.get('recent_events')))+'\nSTATE\n'+json.dumps(data,separators=(',',':'),ensure_ascii=False)

    def compile(self, o, action, obj):
        self.plan=None
        rule=obj.get('policy')
        if not isinstance(rule,dict):
            return
        target=active(o)
        switching=action.startswith('switch ')
        if switching:
            i=int(action.split()[1])-1
            if i>=len(team(o)):
                return
            target=team(o)[i]
            if hp(target)<=0 or target.get('reviving'):
                return
        else:
            req, d=move_info(o,action)
            # Effects of setup and unusual moves require direct reconsideration.
            if d.get('category')=='Status' and req.get('id') not in RECOVERY:
                return
            if req.get('id') in CONDITIONAL:
                return
        attack=key(rule.get('attack',''))
        heal=key(rule.get('heal',''))
        d=o.get('move_dex',{}).get(attack,{})
        if attack not in target.get('moves',[]) or attack in CONDITIONAL or d.get('category') not in ('Physical','Special'):
            return
        if any(t in d.get('description','').lower() for t in ('must recharge','charges on','two-turn','2 turns','next turn')):
            return
        if heal not in RECOVERY or heal not in target.get('moves',[]):
            heal=''
        expected=rule.get('foe_moves')
        if not isinstance(expected,list) or not 1<=len(expected)<=4 or any(not isinstance(v,str) or not key(v) for v in expected):
            return
        maxhit=numeric(rule.get('max_hit'),.4,.01,1.)
        minhp=numeric(rule.get('min_hp'),.6,.05,1.)
        uses=int(numeric(rule.get('uses'),1,0,3))
        if uses<=0:
            return
        tera=target.get('teraType') if 'terastallize' in action else None
        self.plan={'attack':attack,'heal':heal,'heal_below':numeric(rule.get('heal_below'),0,0,.8),
                   'min_hp':minhp,'max_foe_hp':numeric(rule.get('max_foe_hp'),1.,0,1),
                   'max_hit':maxhit,'foe_moves':{key(v) for v in expected},'remaining':uses,
                   'own':tactical_fingerprint(o,target,switching=switching,tera=tera),
                   'foe':tactical_fingerprint(o,active(o,'p2')),'field':field_fingerprint(o),
                   'last_before':o,'last_action':action,'stalls':0,'incoming':0,
                   'illegal_count':self.ctx.progress().get('illegal_actions',0),
                   'entry_terrain':ENTRY_TERRAINS.get(key(target.get('ability'))) if switching else None}
        self.ctx.log('policy_compiled', attack=attack, heal=heal, own=target.get('ident'),
                     foe=active(o,'p2').get('ident'), min_hp=minhp, max_hit=maxhit,
                     expected_moves=sorted(self.plan['foe_moves']), uses=uses)

    def invalidate(self, why):
        if self.plan:
            self.ctx.log('policy_invalidated', reason=why)
        self.plan=None
        self.replan=why
        return None

    def policy_action(self, o):
        p=self.plan
        if not p:
            return None
        if p['remaining']<=0:
            return self.invalidate('Local policy expired; reassess the strategic position.')
        if o.get('last_error'):
            return self.invalidate('Engine rejected previous choice: '+str(o['last_error']))
        progress=self.ctx.progress()
        if progress.get('illegal_actions',0)>p['illegal_count']:
            return self.invalidate('Progress monitor reports a rejected action; abandon the policy.')
        if progress.get('since_obs_change',0)>=2 or progress.get('since_score_change',0)>=4:
            return self.invalidate('Progress monitor says the policy has stalled: '+str(progress))
        if (o.get('request') or {}).get('forceSwitch'):
            return self.invalidate('New forced replacement is outside the local policy.')
        terrain=p.pop('entry_terrain',None)
        if terrain:
            actual=field_fingerprint(o)
            expected=dict(p['field'])
            expected['field']={k:v for k,v in (expected.get('field') or {}).items()
                               if key(k.removeprefix('move: ')) not in {key(t) for t in ENTRY_TERRAINS.values()}}
            expected['field']['move: '+terrain]=True
            confirmed=any(e.startswith('|-fieldstart|move: '+terrain+'|')
                          and any(v.startswith('[from] ability: ') and key(v.removeprefix('[from] ability: '))==key(p['own']['ability']) for v in e.split('|'))
                          and any(v.startswith('[of] ') and ident(v[5:])==p['own']['ident'] for v in e.split('|'))
                          for e in lines(o.get('recent_events')))
            if actual==expected and (actual==p['field'] or confirmed):
                p['field']=actual
            else:
                return self.invalidate('Expected entry terrain was suppressed, unconfirmed, or accompanied by another field change.')
        if tactical_fingerprint(o,active(o))!=p['own']:
            return self.invalidate('Own identity, status, item, type, stats, boosts or volatile changed.')
        if tactical_fingerprint(o,active(o,'p2'))!=p['foe']:
            return self.invalidate('Opponent identity, status, item, type, boosts or volatile changed.')
        if field_fingerprint(o)!=p['field']:
            return self.invalidate('Weather, terrain, screens or hazards changed.')
        if hp(active(o))<p['min_hp'] or hp(active(o,'p2'))>p['max_foe_hp']+.005:
            return self.invalidate('Model-authored HP guard failed.')
        incoming=max(self.ledger.last_direct.get('p1',0),self.ledger.last_cost.get('p1',0),p['incoming'])
        if incoming>p['max_hit']+.015:
            return self.invalidate('Observed incoming damage exceeded model forecast: '+str(round(incoming,3)))
        p['incoming']=incoming
        # Both the model's conservative bound and the observed gross hit must leave margin.
        if hp(active(o))<=max(p['max_hit'],1.2*incoming)+.04:
            return self.invalidate('Another plausible/observed hit leaves insufficient survival margin.')
        old=p['last_before']
        event_failure=None
        for e in lines(o.get('recent_events')):
            s=e.split('|')
            if len(s)<2:
                continue
            k=s[1]
            actor=s[2] if len(s)>2 else ''
            if k=='move' and actor.startswith('p2') and len(s)>3 and key(s[3]) not in p['foe_moves']:
                event_failure='Unexpected opponent move: '+s[3]
            elif k in ('replace','detailschange','-formechange','-transform','-swapboost','-copyboost','-invertboost','-clearallboost','-clearboost','-clearnegativeboost'):
                event_failure='Unmodeled mechanics: '+e
            elif k in ('-crit','cant'):
                event_failure='Critical hit or inability to move: '+e
            elif k in ('-miss','-fail') and actor.startswith('p1'):
                event_failure='Our intended effect failed: '+e
            elif k=='-immune' and actor.startswith('p2'):
                event_failure='Opponent immune to our action: '+e
            elif k in ('-activate','-start','-end'):
                # Effects can be omitted by summary; require replanning instead of guessing.
                event_failure='A triggered/volatile effect needs review: '+e
        if event_failure:
            return self.invalidate(event_failure)
        prior_action=p['last_action']
        prior_req,_=move_info(old,prior_action)
        was_attack=prior_action.startswith('move ') and prior_req.get('id') not in RECOVERY
        if was_attack and hp(active(old,'p2'))-hp(active(o,'p2'))<.015:
            p['stalls']+=1
        elif was_attack:
            p['stalls']=0
        if p['stalls']>=2:
            return self.invalidate('Two attacks made no net progress; reconsider Protect/healing/coverage.')
        chosen=p['heal'] if p['heal'] and hp(active(o))<p['heal_below'] else p['attack']
        action=lookup_move(o,chosen)
        if not action or action in self.rejected:
            return self.invalidate('Named policy move is unavailable in the current request.')
        p['remaining']-=1
        p['last_before']=o
        p['last_action']=action
        self.policy_actions+=1
        self.ctx.log('verified_policy', action=action, remaining=p['remaining'], incoming=round(incoming,3),
                     own_hp=round(hp(active(o)),3), foe_hp=round(hp(active(o,'p2')),3))
        return action

    def run_episode(self, game):
        ctx=self.ctx
        o=unpack(ctx.observe())
        self.ingest(o)
        errors=0
        for decision in range(240):
            if ctx.done or o.get('done'):
                return
            legal=[a for a in o.get('legal_actions',[]) if a!='noop' and a not in self.rejected]
            if not legal:
                return
            action=self.policy_action(o)
            origin='verified_policy'
            if len(legal)==1:
                action=legal[0]
                origin='sole_legal'
                self.plan=None
            if not action:
                remaining=ctx.budget.remaining()
                # 240 seconds is above the archive's 178-second worst call.
                # Retain a call and engine time so a finite fallback can finish.
                enough=remaining.get('system2_calls',0)>1 and (remaining.get('wall_s',0)>260 or (decision==0 and remaining.get('wall_s',0)>10))
                if enough:
                    try:
                        res=ctx.think(self.prompt(o),system=SYSTEM,session_id=None,timeout_s=240.)
                        action,obj=parse_answer(res.text,legal)
                        if action:
                            self.memo=str(obj.get('memory',self.memo))[:850]
                            self.compile(o,action,obj)
                            self.replan=''
                            self.model_actions+=1
                            ctx.log('tactical_policy_decision',action=action,why=str(obj.get('why',''))[:500],memory=self.memo,
                                    policy_offered=isinstance(obj.get('policy'),dict),policy_compiled=self.plan is not None,
                                    no_policy_reason=str(obj.get('no_policy_reason',''))[:300])
                            origin='astra'
                        else:
                            self.invalidate('Model answer malformed or unavailable; review fallback result.')
                    except (BudgetExceeded,ModelUnavailable):
                        return
                    except (TimeoutError,RuntimeError,ValueError,TypeError) as exc:
                        self.invalidate('Model call failed: '+type(exc).__name__+'; reconsider fallback next decision.')
                if not action:
                    action=fallback(o,legal)
                    origin='fallback'
            before=o
            ctx.log('decision',action=action,origin=origin,turn=o.get('turn'))
            try:
                nxt=ctx.act(action)
                o=unpack(nxt if nxt is not None else ctx.observe())
            except (BudgetExceeded,ModelUnavailable):
                return
            except Exception as exc:
                errors+=1
                self.rejected.add(action)
                self.invalidate('Action rejected: '+type(exc).__name__+': '+str(exc)[:150])
                if errors>=5 or ctx.done:
                    return
                o=unpack(ctx.observe())
                continue
            self.ingest(o)
            failures=self.ledger.observe_effect(before,o,action,ctx.progress())
            if failures:
                self.replan='Recent observed effects: '+json.dumps(failures)
            if o.get('last_error'):
                errors+=1
                self.rejected.add(action)
                self.invalidate('Action rejection: '+str(o.get('last_error')))
            else:
                errors=0
                self.rejected.clear()
            if errors>=5:
                return
            ctx.memory['battle_plan']=self.memo
            ctx.memory['policy_stats']={'model_actions':self.model_actions,'policy_actions':self.policy_actions}
            ctx.memory['recent_choices']=list(self.ledger.history)


def build(ctx):
    return Harness(ctx)
