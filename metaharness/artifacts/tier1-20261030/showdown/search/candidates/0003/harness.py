"""Public-evidence, adversarial response-matrix controller for Gen 9 singles.

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


def context_signature(o):
    """Conservative invariants for a one-step repeat; HP and PP intentionally excluded."""
    pub = o.get('public_summary', {})
    mons = []
    for side in ('p1', 'p2'):
        p = active(o, side)
        q = pub.get('pokemon', {}).get(p.get('ident'), {})
        mons.append([p.get('ident'), species(p), p.get('stats'), p.get('ability'), p.get('item'),
                     p.get('terastallized'), q.get('teraTypeRevealed'), q.get('moves'), q.get('boosts'),
                     q.get('volatiles'), q.get('status'), p.get('condition', '').split(' ')[1:]])
    return json.dumps([mons, pub.get('side_conditions'), pub.get('weather'), pub.get('field')], sort_keys=True)


class Ledger:
    def __init__(self):
        self.samples = deque(maxlen=36)
        self.orders = deque(maxlen=12)
        self.failures = deque(maxlen=6)
        self.history = deque(maxlen=7)
        self.last_events = []
        self.last_direct = {}

    def observe_effect(self, before, after, action, progress):
        events = lines(after.get('recent_events'))
        self.last_events = events
        self.last_direct = {}
        mons = {ident(p.get('ident')): p for p in team(before)}
        mons.update(before.get('public_summary', {}).get('pokemon', {}))
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


def matrix_choice(text, legal):
    """Strict finite payoff vectors; garbage text is recoverable, never executable code."""
    obj = None
    decoder = json.JSONDecoder()
    for m in re.finditer(r'\{', str(text)):
        try:
            candidate, _ = decoder.raw_decode(str(text)[m.start():])
            if isinstance(candidate, dict) and ('options' in candidate or 'action' in candidate):
                obj = candidate
                break
        except (ValueError, TypeError):
            continue
    if obj:
        responses = obj.get('responses', [])
        if isinstance(responses, list) and 1 <= len(responses) <= 4:
            try:
                weights = [float(r['p']) for r in responses]
                if any(not math.isfinite(w) or w < 0 for w in weights) or sum(weights) <= 0:
                    raise ValueError('weights')
                weights = [w/sum(weights) for w in weights]
                ranked = []
                for row in obj.get('options', [])[:8]:
                    a = row.get('action')
                    vs = [float(v) for v in row.get('v', [])]
                    if a not in legal or len(vs) != len(weights) or any(not math.isfinite(v) or abs(v)>100 for v in vs):
                        continue
                    likely = [v for v,w in zip(vs,weights) if w >= .08]
                    worst = min(likely or vs)
                    ev = sum(v*w for v,w in zip(vs,weights))
                    score = .65*ev + .35*worst
                    ranked.append((score, ev, a, row))
                if ranked:
                    best = max(ranked, key=lambda r:(r[0], r[1]))
                    return best[2], obj, best[3], [{'action':r[2], 'robust':round(r[0],2), 'expected':round(r[1],2)} for r in ranked]
            except (TypeError, ValueError, KeyError, AttributeError, OverflowError):
                pass
        if obj.get('action') in legal:
            return obj['action'], obj, {}, []
        # A malformed matrix is not a ranked list. Do not execute its first row.
        return None, {'parse_failure':'matrix has no complete finite legal payoff row'}, {}, []
    # Handles arbitrary first-option stand-ins and plain legal answers.
    for m in re.finditer(r'\b(?:move [1-4](?: terastallize)?|switch [1-6])\b', str(text).lower()):
        if m[0] in legal:
            return m[0], {}, {}, []
    return None, {}, {}, []


SYSTEM = '''You are an expert Gen 9 Random Battle singles tactician. You control p1 against a strong MCTS opponent, with no team preview. Win first; preserve strategically useful HP second. Only supplied public observations and private p1 request are available. Unknown sets remain uncertain. Do not assume level 100 or OU spreads.
Your task is an adversarial two-turn decision matrix, not a single intuitive move. Identify exactly TWO plausible opponent responses (specific strongest attack, likely switch, setup/recovery as relevant), with probabilities summing to 1. Estimate the position after our action, their response, and our best next reply. Evaluate THREE genuinely distinct legal actions (two if only two are available), including best attack, best defensive alternative (switch or Tera), and setup/recovery only when viable. On forced replacement, scenarios mean the opponent's next-turn plan; replacement itself does not grant a free attack. Revival Blessing can select a fainted ally.
Scores are comparable values from -100 to 100: -100 immediate/forced loss, -70 lose an essential counter without compensation, -30 bad trade or lose tempo, 0 balanced, +30 favorable concrete progress, +70 decisive winning advantage, +100 forced win. Do not score mere type advantage as a win. Include speed/priority, entry hazards, residual/recoil, all boosts, abilities, accuracy, PP, switching and opponent defensive Tera. Check approximate survival and KO qualitatively before endorsing setup; do not reconstruct exact hidden stats or enumerate long battle trees; if slower, your attack may never happen. Preserve an essential revealed-threat counter. Tera is a resource to spend when it changes the battle; do not save it until the final helpless Pokemon. Opponent switches are common; do not blindly chase them.
Code chooses 0.65*expected score + 0.35*worst score among responses with p>=0.08. Give honest counterfactual values; do not manipulate the matrix to force a favorite. Only options explicitly in legal_actions are executable. Match move and switch slots to CURRENT request, not previous turns.
Return compact JSON only, <=220 visible output tokens. Every v MUST contain exactly TWO finite numbers, one per response in order. Each why is at most 8 words. Use rough ordinal judgments; do not write damage calculations:
{"responses":[{"name":"specific foe response","p":0.6},{"name":"alternative","p":0.4}],"options":[{"action":"move 1","v":[20,-15],"why":"at most eight words","repeat":false}],"memory":"<=25 words: win condition, essential counter, next test"}.
The repeat flag may be true ONLY for a straightforward damaging attack worth repeating once if both active Pokemon and all tactical conditions stay unchanged, it deals substantial damage, and we safely survive another observed hit. Never authorize repeat for setup, recovery, pivot, charge/recharge, or conditional attacks. Public summaries can miss unusual effects; protocol overrides them. Damage samples carry PRE-TURN context (same-turn setup may change it); each sample is one damage event, so sum multi-hit events. They are observations, not universal predictions (rolls, criticals, boosts, switches and items change damage). Move-order evidence is not necessarily speed order because priorities differ. A fainted target censors damage. Unknown opponent bench members are still threats.'''


class Harness:
    def __init__(self, ctx):
        self.ctx = ctx
        self.ledger = Ledger()
        self.memo = ''
        self.short_mode = False
        self.rejected = set()

    def prompt(self, o, progress):
        ps = []
        for i,p in enumerate(team(o), 1):
            q = {k:p[k] for k in ('ident','details','condition','active','stats','moves','ability','item','teraType','terastallized','reviving') if k in p}
            q['slot'] = i
            q['types_now'] = types(o,p)
            q['ordinary_attack_matchups_vs_current_foe'] = {
                mid: {'type':o.get('move_dex',{}).get(mid,{}).get('type'),
                      'multiplier':effectiveness(o.get('move_dex',{}).get(mid,{}).get('type'),types(o,active(o,'p2')))}
                for mid in p.get('moves',[]) if o.get('move_dex',{}).get(mid,{}).get('category') in ('Physical','Special')}
            ps.append(q)
        moves = []
        for a in o.get('legal_actions', []):
            if a.startswith('move ') and 'tera' not in a:
                req, d = move_info(o,a)
                moves.append({'action':a, **req, **d,
                              'ordinary_type_multiplier_vs_current_foe':effectiveness(d.get('type'),types(o,active(o,'p2')))})
        data = {'turn':o.get('turn'), 'request_flags':{k:v for k,v in o.get('request',{}).items() if k!='side'},
                'team_CURRENT_slots':ps, 'legal_actions':o.get('legal_actions',[]),
                'active':o.get('active'), 'public_summary':o.get('public_summary'),
                'opponent_revealed':o.get('opponent_revealed'), 'species_dex':o.get('species_dex'),
                'active_moves':moves, 'bench_move_dex':o.get('move_dex'),
                'recent_protocol':lines(o.get('recent_events'))[-100:],
                'prior_plan_not_binding':self.memo, 'last_choices':list(self.ledger.history),
                'empirical_damage_samples':list(self.ledger.samples)[-18:],
                'move_order_observations':list(self.ledger.orders)[-5:],
                'failure_evidence_REPLAN':list(self.ledger.failures),
                'progress':{k:v for k,v in progress.items() if k!='score'},
                'last_error':o.get('last_error'), 'budget':self.ctx.budget.remaining()}
        short = '\nA prior call timed out. Use only TWO candidate actions and TWO responses; omit why fields, <=150 visible tokens. Quick qualitative triage only.\n' if self.short_mode else ''
        return SYSTEM + short + '\nSTATE\n' + json.dumps(data, separators=(',',':'), ensure_ascii=False)

    def repeat_ok(self, before, after, action, row, failures):
        if row.get('repeat') is not True or failures or 'terastallize' in action:
            return False
        if action not in after.get('legal_actions', []) or context_signature(before) != context_signature(after):
            return False
        req,d = move_info(after,action)
        old_req,_ = move_info(before,action)
        if req.get('id') != old_req.get('id'):
            return False
        if d.get('category') not in ('Physical','Special') or not d.get('base_power'):
            return False
        mid=req.get('id','')
        if mid in {'uturn','voltswitch','flipturn','futuresight','doomdesire','fakeout','firstimpression','suckerpunch','upperhand','gigatonhammer','bloodmoon','outrage','petaldance','thrash','recharge'}:
            return False
        if any(t in d.get('description','').lower() for t in ('turn 2','recharge','last turn','first turn','must recharge')):
            return False
        events=lines(after.get('recent_events'))
        if any(any(t in e for t in ('|switch|','|drag|','|replace|','|faint|','|-crit|','|-status|','|-activate|','|-start|','|-end|','|-boost|','|-unboost|','|-item|','|-enditem|','|-terastallize|')) for e in events):
            return False
        incoming = self.ledger.last_direct.get('p1',0)
        outgoing = self.ledger.last_direct.get('p2',0)
        if len(active(after).get('condition','').split())>1 or outgoing < .20 or hp(active(after,'p2')) <= 0 or incoming <= 0 or hp(active(after)) <= incoming*1.3+.12:
            return False
        # Repeating into an obvious revealed resist gives the search opponent a free pivot.
        for p in after.get('opponent_revealed',[]):
            if hp(p)>0 and p.get('ident')!=active(after,'p2').get('ident') and effectiveness(d.get('type'),types(after,p)) < 1:
                return False
        return True

    def run_episode(self, game):
        ctx=self.ctx
        o=unpack(ctx.observe())
        repeat_action=None
        errors=0
        for decision in range(240):
            if ctx.done or o.get('done'):
                return
            legal=[a for a in o.get('legal_actions',[]) if a!='noop' and a not in self.rejected]
            if not legal:
                legal=[a for a in o.get('legal_actions',[]) if a!='noop']
            if not legal:
                return
            progress=ctx.progress()
            row={}
            expected_responses=[]
            origin='fallback'
            if len(legal)==1:
                action=legal[0]
                origin='sole_legal'
            elif repeat_action in legal and not o.get('last_error'):
                action=repeat_action
                origin='verified_repeat'
            else:
                action=None
                remaining=ctx.budget.remaining()
                # Calls have reached 104s in archive. Fixed timeout is 180s;
                # do not start one if too little wall remains to recover.
                enough=(remaining.get('wall_s',0)>185 or (decision==0 and remaining.get('wall_s',0)>10)) and remaining.get('system2_calls',0)>1
                if enough:
                    try:
                        res=ctx.think(self.prompt(o,progress), session_id=None, timeout_s=180.0)
                        action,obj,row,ranked=matrix_choice(res.text,legal)
                        expected_responses=[str(r.get('name','')) for r in (obj.get('responses') or []) if isinstance(r,dict)]
                        if obj.get('parse_failure'):
                            self.ledger.failures.append({'model_schema_failure':obj['parse_failure']})
                            self.short_mode=True
                        self.memo=str(obj.get('memory',self.memo))[:700]
                        ctx.memory['battle_plan']=self.memo
                        ctx.log('adversarial_matrix', turn=o.get('turn'), ranking=ranked,
                                responses=obj.get('responses'), selected=action, rationale=row.get('why'))
                        origin='matrix' if ranked else 'parsed_answer'
                    except (BudgetExceeded,ModelUnavailable):
                        return
                    except (TimeoutError,RuntimeError,ValueError,TypeError) as e:
                        self.short_mode=True
                        self.ledger.failures.append({'model_failure':str(e)[:180]})
                if action is None:
                    action=fallback(o,legal)
                    origin='fallback'
            repeat_action=None
            before=o
            ctx.log('decision', action=action, origin=origin, turn=o.get('turn'))
            try:
                nxt=ctx.act(action)
                o=unpack(nxt if nxt is not None else ctx.observe())
            except (BudgetExceeded,ModelUnavailable):
                return
            except Exception as e:
                errors+=1
                self.rejected.add(action)
                self.ledger.failures.append({'action_rejected':action,'error':str(e)[:180]})
                if errors>=5 or ctx.done:
                    return
                o=unpack(ctx.observe())
                continue
            failures=self.ledger.observe_effect(before,o,action,ctx.progress())
            if expected_responses:
                observed=[]
                for event in lines(o.get('recent_events')):
                    fields=event.split('|')
                    if len(fields)>3 and fields[2].startswith('p2'):
                        if fields[1]=='move':
                            observed.append(fields[3])
                        elif fields[1] in ('switch','drag'):
                            observed.append('switch')
                for actual in observed:
                    if not any(key(actual) in key(label) for label in expected_responses):
                        warning={'opponent_response_not_named':actual, 'expected_labels':expected_responses}
                        self.ledger.failures.append(warning)
                        failures.append(warning)
            if o.get('last_error'):
                self.rejected.add(action)
                errors+=1
            else:
                self.rejected.clear()
                errors=0
            if errors>=5:
                return
            if origin=='matrix' and self.repeat_ok(before,o,action,row,failures):
                repeat_action=action
            ctx.memory['recent_choices']=list(self.ledger.history)


def build(ctx):
    return Harness(ctx)
